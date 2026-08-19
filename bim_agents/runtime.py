from __future__ import annotations

import asyncio
import os

from agents import MaxTurnsExceeded, Runner, trace

from bim_context import BimContext, Settings

from .graph_contract import load_graph_contract, validate_live_schema
from .guardrails import pipeline_report_from_evidence
from .models import BimRunContext, PipelineReport, ProjectScope
from .observability import AgentRunHooks, PipelineEvents
from .registry import build_agent_registry
from .tools import ensure_bim_verification


async def answer_bim_question(
    question: str,
    *,
    settings: Settings | None = None,
    model: str | None = None,
    worker_model: str | None = None,
    timeout_seconds: float | None = None,
    hooks: PipelineEvents | None = None,
) -> PipelineReport:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    settings = settings or Settings.from_env()
    bim = BimContext(settings)
    events = hooks or PipelineEvents()
    bim.connect()
    try:
        contract = load_graph_contract(
            os.getenv("BIM_GRAPH_SCHEMA_PATH") or None,
            client_id=settings.client_id,
            project_id=settings.project_id,
        )
        sources = bim.resolve_allowed_sources(contract)
        if not sources:
            raise PermissionError("The configured client/project has no authorized BIM sources.")
        validate_live_schema(bim, contract, authorization_only=True)
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(
                client_id=settings.client_id,
                project_id=settings.project_id,
                allowed_sources=sources,
            ),
            graph_contract=contract,
            question=question,
            max_llm_calls=int(os.getenv("BIM_MAX_LLM_CALLS", "20")),
            max_tool_calls=int(os.getenv("BIM_MAX_TOOL_CALLS", "30")),
            max_agent_starts=int(os.getenv("BIM_MAX_AGENT_STARTS", "30")),
            max_starts_per_agent=int(os.getenv("BIM_MAX_STARTS_PER_AGENT", "6")),
        )
        run_hooks = AgentRunHooks(events)
        registry = build_agent_registry(
            model or os.getenv("BIM_AGENT_MODEL", "gpt-5.6-sol"),
            worker_model=worker_model or os.getenv("BIM_AGENT_WORKER_MODEL", "gpt-5.6-terra"),
            hooks=run_hooks,
        )
        scoped_input = (
            f"Question: {question}\n"
            f"Project scope: client_id={settings.client_id}, project_id={settings.project_id}.\n"
            "Use only authorized project sources."
        )
        events.stage("pipeline_start", question=question)
        timeout = timeout_seconds or float(os.getenv("BIM_RUN_TIMEOUT_SECONDS", "240"))
        with trace("BIM graph-to-Cypher answer", metadata={"project_id": settings.project_id}):
            async with asyncio.timeout(timeout):
                try:
                    await Runner.run(
                        registry.supervisor,
                        scoped_input,
                        context=context,
                        max_turns=int(os.getenv("BIM_SUPERVISOR_MAX_TURNS", "24")),
                        hooks=run_hooks,
                    )
                except MaxTurnsExceeded:
                    context.runtime_limitations.append(
                        "The investigation reached its configured turn limit; any completed query "
                        "evidence was still independently verified."
                    )
                    events.stage("turn_limit_reached")
        ensure_bim_verification(context)
        report = pipeline_report_from_evidence(context)
        events.stage("pipeline_end", verification_status=report.verification_status)
        return report
    finally:
        bim.close()
