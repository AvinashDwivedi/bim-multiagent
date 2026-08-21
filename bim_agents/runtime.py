from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from agents import MaxTurnsExceeded, Runner, trace

from bim_context import BimContext, Settings

from .graph_contract import load_graph_contract, validate_live_schema
from .guardrails import pipeline_report_from_evidence
from .models import BimRunContext, PipelineReport, ProjectScope
from .observability import AgentRunHooks, PipelineEvents
from .registry import build_agent_registry
from .tools import ensure_bim_verification, ensure_compliance_evidence


def _redact_text(value: str, sensitive_values: list[str]) -> str:
    redacted = value
    for secret in sensitive_values:
        if secret and len(secret) >= 8:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(
        r"(?i)\b(client_id|project_id|neo4j_uri|neo4j_username|neo4j_password|openai_api_key)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        redacted,
    )
    return redacted


def _redact_value(value: Any, sensitive_values: list[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, sensitive_values)
    if isinstance(value, list):
        return [_redact_value(item, sensitive_values) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item, sensitive_values) for key, item in value.items()}
    return value


def redact_pipeline_report(report: PipelineReport, settings: Settings) -> PipelineReport:
    """Remove configured identifiers and credentials from all user-visible report fields."""
    sensitive_values = [
        settings.client_id,
        settings.project_id,
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        os.getenv("OPENAI_API_KEY", ""),
    ]
    return PipelineReport.model_validate(
        _redact_value(report.model_dump(mode="json"), sensitive_values)
    )


def _capability_gap(question: str, contract) -> str | None:
    """Fail fast when a requested derived metric has no trusted query representation."""
    text = question.casefold()
    fields = {
        field_name.casefold()
        for entity in contract.query_entities.values()
        for field_name in entity.fields
    }
    if ("opening percentage" in text or "facade percentage" in text or "façade percentage" in text):
        if not ({"opening_area_m2", "facade_area_m2"} <= fields or "opening_percentage" in fields):
            return (
                "The façade opening percentage is not assessable: the trusted BIM query contract has "
                "neither a modelled opening percentage nor compatible opening-area and façade-area metrics."
            )
    requested_height_fields: set[str] = set()
    label = ""
    # A building "section height" commonly means the absolute height/elevation of
    # a massing section (for example a roof or setback), not a clear or
    # floor-to-floor height.  Let the investigator resolve that meaning against
    # section/roof elements and the trusted level elevations instead of rejecting
    # the question before any graph inspection occurs.
    if "space height" in text or "clear height" in text:
        requested_height_fields = {"space_height_m", "clear_height_m"}
        label = "space height"
    elif "how high" in text or "model height" in text or "building height" in text:
        requested_height_fields = {"model_height_m", "building_height_m"}
        label = "model height"
    if requested_height_fields and not (requested_height_fields & fields):
        return (
            f"The requested {label} cannot be assessed because no corresponding trusted height metric "
            "is modelled. Storey reference elevations, where present, are not treated as heights."
        )
    return None


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
        capability_gap = _capability_gap(question, contract)
        if capability_gap:
            report = PipelineReport(
                answer=capability_gap,
                limitations=[capability_gap],
                stages_used=["Capability Guard"],
                investigation_trace=[f"Capability Guard: {capability_gap}"],
                verification_status="insufficient_evidence",
            )
            events.stage("capability_gap")
            return redact_pipeline_report(report, settings)
        run_hooks = AgentRunHooks(events)
        registry = build_agent_registry(
            model or os.getenv("BIM_AGENT_MODEL", "gpt-5.6-sol"),
            worker_model=worker_model or os.getenv("BIM_AGENT_WORKER_MODEL", "gpt-5.6-terra"),
            hooks=run_hooks,
        )
        scoped_input = (
            f"Question: {question}\n"
            "Use only the configured authorized BIM scope. Do not expose scope identifiers or credentials."
        )
        events.stage("pipeline_start", question=question)
        timeout = timeout_seconds or float(os.getenv("BIM_RUN_TIMEOUT_SECONDS", "240"))
        with trace("BIM graph-to-Cypher answer", metadata={"authorized_scope": True}):
            async with asyncio.timeout(timeout):
                try:
                    run_result = await Runner.run(
                        registry.supervisor,
                        scoped_input,
                        context=context,
                        max_turns=int(os.getenv("BIM_SUPERVISOR_MAX_TURNS", "24")),
                        hooks=run_hooks,
                    )
                    completion = run_result.final_output
                    context.completion_status = getattr(completion, "status", None)
                    if getattr(completion, "status", None) == "insufficient_evidence":
                        context.runtime_limitations.extend(
                            str(item) for item in getattr(completion, "limitations", []) if str(item).strip()
                        )
                except MaxTurnsExceeded:
                    context.runtime_limitations.append(
                        "The investigation reached its configured turn limit; any completed query "
                        "evidence was still independently verified."
                    )
                    events.stage("turn_limit_reached")
        if re.search(r"\b(compliance|comply|compliant|requirement|requirements)\b", question, re.I):
            evidence_id = ensure_compliance_evidence(context)
            if evidence_id:
                events.stage("compliance_evidence", evidence_id=evidence_id)
        ensure_bim_verification(context)
        report = pipeline_report_from_evidence(context)
        events.stage("pipeline_end", verification_status=report.verification_status)
        return redact_pipeline_report(report, settings)
    finally:
        bim.close()
