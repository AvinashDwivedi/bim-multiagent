from __future__ import annotations

import asyncio
import os

from agents import OutputGuardrailTripwireTriggered, RunConfig, Runner, trace

from bim_context import BimContext, Settings

from .models import BimRunContext, ProjectScope, SupervisorReport
from .observability import BimRunHooks
from .graph_contract import load_graph_contract, validate_live_schema
from .guardrails import supervisor_report_from_evidence
from .registry import build_agent_registry


async def answer_bim_question(
    question: str,
    *,
    settings: Settings | None = None,
    model: str | None = None,
    worker_model: str | None = None,
    timeout_seconds: float | None = None,
    hooks: BimRunHooks | None = None,
) -> SupervisorReport:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    settings = settings or Settings.from_env()
    bim = BimContext(settings)
    try:
        bim.connect()
        graph_contract = load_graph_contract(
            os.getenv("BIM_GRAPH_SCHEMA_PATH") or None,
            client_id=settings.client_id,
            project_id=settings.project_id,
        )
        allowed_sources = bim.resolve_allowed_sources(graph_contract)
        if not allowed_sources:
            raise PermissionError("The configured client/project has no authorized BIM sources.")
        schema_report = validate_live_schema(bim, graph_contract)
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(
                client_id=settings.client_id,
                project_id=settings.project_id,
                allowed_sources=allowed_sources,
            ),
            graph_contract=graph_contract,
            max_llm_calls=int(os.getenv("BIM_MAX_LLM_CALLS", "12")),
            max_tool_calls=int(os.getenv("BIM_MAX_TOOL_CALLS", "20")),
            max_agent_starts=int(os.getenv("BIM_MAX_AGENT_STARTS", "8")),
        )
        hooks = hooks or BimRunHooks()
        supervisor_model = model or os.getenv("BIM_AGENT_MODEL", "gpt-5.6-sol")
        selected_worker_model = worker_model or os.getenv("BIM_AGENT_WORKER_MODEL", "gpt-5.6-terra")
        registry = build_agent_registry(
            supervisor_model,
            worker_model=selected_worker_model,
            hooks=hooks,
        )
        hooks.logger.info(
            "run.start   | project=%s | sources=%d | supervisor_model=%s | worker_model=%s",
            settings.project_id,
            len(allowed_sources),
            supervisor_model,
            selected_worker_model,
        )
        hooks.logger.info(
            "schema.ok   | contract_version=%d | labels=%d | relationships=%d | properties=%d",
            schema_report.contract_version,
            len(schema_report.labels_checked),
            len(schema_report.relationships_checked),
            len(schema_report.properties_checked),
        )
        scoped_input = (
            f"Question: {question}\n"
            f"Project scope: client_id={settings.client_id}, project_id={settings.project_id}.\n"
            "Do not answer outside this scope."
        )
        with trace("BIM supervised answer", metadata={"project_id": settings.project_id}):
            timeout_seconds = timeout_seconds or float(os.getenv("BIM_RUN_TIMEOUT_SECONDS", "180"))
            fallback_report = None
            try:
                async with asyncio.timeout(timeout_seconds):
                    result = await Runner.run(
                        registry.supervisor,
                        scoped_input,
                        context=context,
                        max_turns=int(os.getenv("BIM_SUPERVISOR_MAX_TURNS", "6")),
                        hooks=hooks,
                        run_config=RunConfig(trace_include_sensitive_data=False),
                    )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"BIM run stopped after {timeout_seconds:.0f}s. "
                    f"Progress: agents={context.agent_starts}, LLM calls={context.llm_calls}, "
                    f"tool calls={context.tool_calls}."
                ) from exc
            except OutputGuardrailTripwireTriggered:
                hooks.logger.warning(
                    "guardrail   | supervisor output contradicted deterministic evidence; "
                    "using evidence-derived report"
                )
                fallback_report = supervisor_report_from_evidence(context)
        report = fallback_report or result.final_output_as(SupervisorReport)
        hooks.logger.info(
            "run.end     | agents=%d | llm_calls=%d | tool_calls=%d",
            context.agent_starts,
            context.llm_calls,
            context.tool_calls,
        )
        return report
    finally:
        bim.close()
