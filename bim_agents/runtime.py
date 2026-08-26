from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from agents import MaxTurnsExceeded, Runner, trace
from openai import APIConnectionError, APITimeoutError, RateLimitError

from bim_context import BimContext, Settings

from .graph_contract import load_graph_contract, validate_live_schema
from .guardrails import pipeline_report_from_evidence
from .knowledge import (
    LearnedKnowledgeStore, activate_promoted_knowledge, compute_live_schema_fingerprint,
    curate_verified_mappings, default_knowledge_path,
)
from .models import BimRunContext, PipelineReport, ProjectScope, WorkstreamDiagnostic
from .observability import AgentRunHooks, PipelineEvents
from .orchestration import run_evidence_workstreams
from .registry import build_agent_registry
from .tools import (
    PipelineContext, define_bim_task, ensure_bim_verification, ensure_compliance_evidence,
    inspect_completion_gates, build_compact_model_profile,
)


def _is_budget_exhaustion(exc: Exception) -> bool:
    """Recognize a guardrail error even when the Agents SDK wraps it as UserError."""
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).casefold()
        if "bim run guardrail stopped execution" in message or "agent start budget exceeded" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _budget_exhaustion_diagnostic(context: BimRunContext) -> tuple[str, str]:
    if context.tool_calls > context.max_tool_calls:
        return (
            "tool_call_budget_exhausted",
            f"Tool-call budget exhausted ({context.tool_calls} calls; limit {context.max_tool_calls}).",
        )
    if context.llm_calls > context.max_llm_calls:
        return (
            "llm_call_budget_exhausted",
            f"Model-call budget exhausted ({context.llm_calls} calls; limit {context.max_llm_calls}).",
        )
    over_agent = next(
        (
            (name, count) for name, count in context.agent_starts_by_name.items()
            if count > context.max_starts_per_agent
        ),
        None,
    )
    if over_agent:
        name, count = over_agent
        return (
            "agent_start_budget_exhausted",
            f"Agent-start budget exhausted for {name} ({count} starts; limit {context.max_starts_per_agent}).",
        )
    if context.agent_starts > context.max_agent_starts:
        return (
            "agent_start_budget_exhausted",
            f"Total agent-start budget exhausted ({context.agent_starts} starts; limit {context.max_agent_starts}).",
        )
    return (
        "investigation_budget_exhausted",
        "The investigation work budget was exhausted before a more specific counter was available.",
    )


def _agent_work_timeout(total_timeout: float, reserve_seconds: float | None = None) -> float:
    """Reserve part of the public timeout for replay, guardrails, curation, and serialization."""
    if total_timeout <= 0:
        raise ValueError("BIM run timeout must be positive.")
    configured = (
        float(os.getenv("BIM_FINALIZATION_RESERVE_SECONDS", "30"))
        if reserve_seconds is None else reserve_seconds
    )
    reserve = min(max(configured, 0.0), total_timeout * 0.2)
    return max(total_timeout - reserve, min(total_timeout, 0.1))


async def _run_parallel_workstreams(*workstreams):
    """Compatibility helper; active scheduling lives in orchestration.py."""
    return await asyncio.gather(*workstreams)


def _has_relevant_promoted_mapping(context: BimRunContext) -> bool:
    """Compatibility predicate for deployments migrating from challenger mode."""
    if context.task_contract is None:
        return False
    ontology = context.bim.ontology
    terms = [context.question, context.task_contract.entity_concept]
    for value in list(terms):
        for method_name in ("keyword_synonyms_for", "retrieval_terms_for"):
            method = getattr(ontology, method_name, None)
            if callable(method):
                terms.extend(str(item) for item in (method(value) or []))
    tokens = set(re.findall(r"[^\W_]{3,}", " ".join(terms).casefold(), flags=re.UNICODE))
    for record in context.learned_knowledge.values():
        if getattr(record, "status", None) != "promoted":
            continue
        payload = getattr(record, "payload", {}) or {}
        proposal = payload.get("proposal") or {}
        record_text = " ".join([
            str(getattr(record, "concept", "")),
            *(str(item) for item in (getattr(record, "aliases", []) or [])),
            str(proposal.get("entity_name") or ""),
        ]).casefold()
        if tokens & set(re.findall(r"[^\W_]{3,}", record_text, flags=re.UNICODE)):
            return True
    return False


def _should_run_parallel_challenger(context: BimRunContext) -> bool:
    """Legacy configuration parser; the DAG scheduler does not launch challengers."""
    if os.getenv("BIM_PARALLEL_CHALLENGER_ENABLED", "1") == "0":
        return False
    mode = os.getenv("BIM_PARALLEL_CHALLENGER_MODE", "always").strip().casefold()
    if mode == "always":
        return True
    if mode == "auto":
        return _has_relevant_promoted_mapping(context)
    if mode in {"off", "never"}:
        return False
    raise ValueError("BIM_PARALLEL_CHALLENGER_MODE must be always, auto, or off.")


def _answer_query_ids(context: BimRunContext) -> list[str]:
    result = []
    for evidence_id, evidence in context.evidence.items():
        if evidence.kind != "query":
            continue
        payload = json.loads(evidence.payload)
        plan = payload.get("plan") or {}
        if (
            plan.get("role", "answer_producing") in {"answer_producing", "supporting"}
            and (
                plan.get("role", "answer_producing") == "supporting"
                or plan.get("include_in_answer", True) is not False
            )
        ):
            result.append(evidence_id)
    return result


def _merge_isolated_workstream(
    root: BimRunContext, branch: BimRunContext, *, workstream_id: str,
) -> list[str]:
    """Account branch work, then atomically commit its valid evidence delta.

    Usage is authoritative even when a branch has no answer or its data commit is
    rejected. This prevents failed/repaired workers from silently returning budget
    to later workstreams.
    """
    with root._lock:
        root.llm_calls += branch.llm_calls
        root.tool_calls += branch.tool_calls
        root.agent_starts += branch.agent_starts
        for name, count in branch.agent_starts_by_name.items():
            root.agent_starts_by_name[name] = root.agent_starts_by_name.get(name, 0) + count
    answer_ids = _answer_query_ids(branch)
    if not answer_ids:
        return []
    mapping_conflict = None
    for mapping_id in sorted(branch.schema_mappings):
        mapping = branch.schema_mappings[mapping_id]
        existing = root.schema_mappings.get(mapping_id)
        if existing is not None:
            existing_dump = getattr(existing, "model_dump", lambda **_: existing)(mode="json")
            mapping_dump = getattr(mapping, "model_dump", lambda **_: mapping)(mode="json")
            if existing_dump != mapping_dump:
                mapping_conflict = mapping_id
                break
    if mapping_conflict is not None:
        root.failure_categories.append("parallel_mapping_conflict")
        root.runtime_limitations.append(
            f"Isolated workstream {workstream_id} conflicted on mapping {mapping_conflict}."
        )
        return []
    evidence_conflict = None
    for evidence_id in sorted(branch.evidence):
        evidence = branch.evidence[evidence_id]
        existing = root.evidence.get(evidence_id)
        if existing is not None and existing.model_dump(mode="json") != evidence.model_dump(mode="json"):
            evidence_conflict = evidence_id
            break
    if evidence_conflict is not None:
        root.failure_categories.append("parallel_evidence_conflict")
        root.runtime_limitations.append(
            f"Isolated workstream {workstream_id} conflicted on evidence {evidence_conflict}."
        )
        return []

    # Stage artifact IDs before mutating authoritative state. Collisions are
    # namespaced deterministically so one branch cannot overwrite another.
    staged_artifacts: list[tuple[str, Any]] = []
    for artifact_id in sorted(branch.artifacts):
        if artifact_id == "task-contract":
            continue
        artifact = branch.artifacts[artifact_id]
        committed_id = artifact_id
        if committed_id in root.artifacts:
            committed_id = f"{workstream_id}-{artifact_id}"
        if committed_id in root.artifacts or any(key == committed_id for key, _ in staged_artifacts):
            root.failure_categories.append("parallel_artifact_conflict")
            root.runtime_limitations.append(
                f"Isolated workstream {workstream_id} produced a duplicate artifact ID."
            )
            return []
        staged_artifacts.append((committed_id, artifact))

    with root._lock:
        for mapping_id in sorted(branch.schema_mappings):
            root.schema_mappings.setdefault(mapping_id, branch.schema_mappings[mapping_id])
        for evidence_id in sorted(branch.evidence):
            root.evidence.setdefault(evidence_id, branch.evidence[evidence_id])
        for committed_id, artifact in staged_artifacts:
            root.artifacts[committed_id] = artifact.model_copy(update={"artifact_id": committed_id})
    return answer_ids


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


def _capability_gap(question: str, contract, project_knowledge: dict | None = None) -> str | None:
    """Fail fast when a requested derived metric has no trusted query representation."""
    text = question.casefold()
    fields = {
        field_name.casefold()
        for entity in contract.query_entities.values()
        for field_name in entity.fields
    }
    if ("opening percentage" in text or "facade percentage" in text or "façade percentage" in text):
        has_scoped_calculation = bool((project_knowledge or {}).get("tower_facade"))
        if not has_scoped_calculation and not (
            {"opening_area_m2", "facade_area_m2"} <= fields or "opening_percentage" in fields
        ):
            return (
                "The façade opening percentage is not assessable: the trusted BIM query contract has "
                "neither a modelled opening percentage nor compatible opening-area and façade-area metrics."
            )
    # A building "section height" commonly means the absolute height/elevation of
    # a massing section (for example a roof or setback), not a clear or
    # floor-to-floor height.  Let the investigator resolve that meaning against
    # section/roof elements and the trusted level elevations instead of rejecting
    # the question before any graph inspection occurs.
    # Height questions must reach graph inspection. Revit placements and storey/roof elevations
    # can support qualified geometric derivations even when no direct height quantity exists.
    if "space height" in text or "clear height" in text:
        if not ({"space_height_m", "clear_height_m"} & fields):
            return (
                "The requested clear/space height cannot be assessed because no trusted Revit "
                "clear-height or space-height quantity is modelled. Placement and storey elevations "
                "describe vertical positions, not the vertical extent of a space."
            )
    return None


async def answer_bim_question(
    question: str,
    *,
    settings: Settings | None = None,
    client_id: str | None = None,
    project_id: str | None = None,
    model: str | None = None,
    worker_model: str | None = None,
    timeout_seconds: float | None = None,
    hooks: PipelineEvents | None = None,
) -> PipelineReport:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if settings is not None and (client_id is not None or project_id is not None):
        raise ValueError("Pass either settings or request scope IDs, not both.")
    settings = settings or Settings.from_env(client_id=client_id, project_id=project_id)
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
        project_knowledge = bim.ontology.bim_query_knowledge
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(
                client_id=settings.client_id,
                project_id=settings.project_id,
                allowed_sources=sources,
            ),
            graph_contract=contract,
            question=question,
            max_llm_calls=int(os.getenv("BIM_MAX_LLM_CALLS", "48")),
            max_tool_calls=int(os.getenv("BIM_MAX_TOOL_CALLS", "60")),
            max_agent_starts=int(os.getenv("BIM_MAX_AGENT_STARTS", "30")),
            max_starts_per_agent=int(os.getenv("BIM_MAX_STARTS_PER_AGENT", "6")),
        )
        try:
            context.schema_fingerprint = compute_live_schema_fingerprint(context)
        except Exception:
            context.failure_categories.append("schema_fingerprint_unavailable")
            events.stage("schema_fingerprint_unavailable")
        if os.getenv("BIM_LEARNING_ENABLED", "1") != "0" and context.schema_fingerprint:
            try:
                context.knowledge_store = LearnedKnowledgeStore(default_knowledge_path())
                activated = activate_promoted_knowledge(context)
                events.stage("knowledge_loaded", compatible_records=len(activated))
            except Exception:
                context.knowledge_store = None
                context.failure_categories.append("learned_knowledge_unavailable")
                events.stage("knowledge_unavailable")
        try:
            build_compact_model_profile(context)
            events.stage(
                "model_profile_ready",
                node_types=len(context.model_profile.node_types),
                relationship_types=len(context.model_profile.relationship_types),
            )
        except Exception:
            context.failure_categories.append("model_profile_unavailable")
            events.stage("model_profile_unavailable")
        capability_gap = _capability_gap(question, contract, project_knowledge)
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
            worker_model=worker_model or os.getenv("BIM_AGENT_WORKER_MODEL", "gpt-5.6-sol"),
            hooks=run_hooks,
            project_knowledge=project_knowledge,
        )
        scoped_input = (
            f"Question: {question}\n"
            "Use only the configured authorized BIM scope. Do not expose scope identifiers or credentials."
        )
        events.stage("pipeline_start", question=question)
        timeout = (
            timeout_seconds if timeout_seconds is not None
            else float(os.getenv("BIM_RUN_TIMEOUT_SECONDS", "600"))
        )
        work_timeout = _agent_work_timeout(timeout)

        try:
            with trace("BIM graph-to-Cypher answer", metadata={"authorized_scope": "true"}):
                async with asyncio.timeout(work_timeout):
                    architect_result = await Runner.run(
                        registry.task_architect,
                        scoped_input,
                        context=context,
                        max_turns=1,
                        hooks=run_hooks,
                    )
                    define_bim_task(PipelineContext(context), architect_result.final_output)
                    events.stage(
                        "workstream_schedule_start",
                        run_id=context.run_id,
                        workstreams=len(context.task_contract.work_packages) or 1,
                    )
                    outcomes = await run_evidence_workstreams(
                        settings=settings,
                        root=context,
                        registry=registry,
                        events=events,
                        merge=lambda root, branch: _merge_isolated_workstream(
                            root, branch, workstream_id=branch.workstream_id
                        ),
                    )
                    context.workstream_diagnostics = [
                        WorkstreamDiagnostic(
                            package_id=outcome.package.package_id,
                            specialist=outcome.specialist,
                            status=outcome.status,
                            attempts=outcome.specialist_attempts,
                            typed_output_failures=outcome.typed_output_failures,
                            recovery_strategy=outcome.recovery_strategy,
                            required_outputs=list(outcome.package.required_outputs),
                            satisfied_outputs=list(outcome.satisfied_outputs),
                        )
                        for outcome in outcomes
                    ]
                    merged_ids = [
                        evidence_id for outcome in outcomes for evidence_id in outcome.evidence_ids
                    ]
                    for outcome in outcomes:
                        context.runtime_limitations.extend(
                            item for item in outcome.limitations if item.strip()
                        )
                        if outcome.status in {
                            "turn_limit", "error", "merge_rejected", "partial_completed",
                        }:
                            category = "workstream_" + outcome.status
                            if category not in context.failure_categories:
                                context.failure_categories.append(category)
                    context.completion_status = (
                        "ready_for_verification" if merged_ids else "insufficient_evidence"
                    )
                    events.stage(
                        "workstream_schedule_end",
                        run_id=context.run_id,
                        completed=sum(item.status == "query_completed" for item in outcomes),
                        partial=sum(item.status == "partial_completed" for item in outcomes),
                        evidence=len(merged_ids),
                        specialist_attempts=sum(item.specialist_attempts for item in outcomes),
                        typed_output_failures=sum(item.typed_output_failures for item in outcomes),
                        recovered=sum(bool(item.recovery_strategy) for item in outcomes),
                    )
        except asyncio.CancelledError:
            events.stage("run_cancelled")
            raise
        except MaxTurnsExceeded:
            context.failure_categories.append("agent_turn_limit")
            context.runtime_limitations.append(
                "The investigation reached its configured turn limit; completed evidence was retained."
            )
            events.stage("turn_limit_reached")
        except (TimeoutError, APITimeoutError):
            context.failure_categories.append("investigation_timeout")
            context.runtime_limitations.append(
                "The investigation timed out; completed evidence was retained for verification."
            )
            events.stage("investigation_timeout")
        except RateLimitError:
            context.failure_categories.append("model_rate_limit")
            context.runtime_limitations.append(
                "The model service rate-limited the investigation; completed evidence was retained."
            )
            events.stage("model_rate_limit")
        except APIConnectionError:
            context.failure_categories.append("model_connection_error")
            context.runtime_limitations.append(
                "The model service connection failed; completed evidence was retained."
            )
            events.stage("model_connection_error")
        except Exception as exc:
            if _is_budget_exhaustion(exc):
                context.failure_categories.append("investigation_budget_exhausted")
                budget_category, budget_detail = _budget_exhaustion_diagnostic(context)
                if budget_category not in context.failure_categories:
                    context.failure_categories.append(budget_category)
                context.runtime_limitations.append(
                    "The investigation reached its configured work budget; completed evidence was retained. "
                    + budget_detail
                )
                events.stage("investigation_budget_exhausted", category=budget_category)
            else:
                # The public BIM endpoint must fail closed with a typed report rather
                # than turn one worker/runtime defect into an HTTP 500 that discards
                # every durable evidence checkpoint. Detailed exceptions remain in
                # server logs/events and are never exposed to the client.
                context.failure_categories.append("pipeline_runtime_error")
                context.runtime_limitations.append(
                    "The investigation encountered an internal execution error; completed "
                    "evidence was retained and independently verified where possible."
                )
                events.stage(
                    "pipeline_runtime_error", category="internal_execution_error",
                    error_type=type(exc).__name__,
                )
        if re.search(r"\b(compliance|comply|compliant|requirement|requirements)\b", question, re.I):
            try:
                evidence_id = ensure_compliance_evidence(context)
                if evidence_id:
                    events.stage("compliance_evidence", evidence_id=evidence_id)
            except Exception as exc:
                context.failure_categories.append("compliance_evidence_error")
                context.runtime_limitations.append(
                    "Applicable requirement evidence could not be finalized; completed model "
                    "evidence remains available."
                )
                events.stage("compliance_evidence_error", error_type=type(exc).__name__)
        try:
            ensure_bim_verification(context)
        except Exception as exc:
            context.failure_categories.append("verification_runtime_error")
            context.runtime_limitations.append(
                "Deterministic replay could not be completed for all retained evidence."
            )
            events.stage("verification_runtime_error", error_type=type(exc).__name__)
        try:
            gates = json.loads(inspect_completion_gates(PipelineContext(context)))
        except Exception as exc:
            gates = {"ready_to_respond": False, "missing": ["completion_gate_runtime_error"]}
            context.failure_categories.append("completion_gate_runtime_error")
            events.stage("completion_gate_runtime_error", error_type=type(exc).__name__)
        if not gates.get("ready_to_respond"):
            context.completion_status = "insufficient_evidence"
            missing = gates.get("missing") or []
            if missing:
                context.runtime_limitations.append(
                    "Completion gates remain open: " + ", ".join(str(item) for item in missing) + "."
                )
        try:
            curation = curate_verified_mappings(context)
            events.stage(
                "knowledge_curated", status=curation.status,
                learned_records=len(curation.knowledge_ids),
            )
        except Exception:
            context.failure_categories.append("knowledge_curation_failed")
            events.stage("knowledge_curation_failed")
        try:
            report = pipeline_report_from_evidence(context)
        except Exception as exc:
            events.stage("report_composition_error", error_type=type(exc).__name__)
            report = PipelineReport(
                answer=(
                    "The investigation completed, but its verified evidence could not be "
                    "assembled into a safe response."
                ),
                limitations=[
                    "The response composer rejected the retained evidence; no unverified "
                    "answer value is being returned."
                ],
                failure_categories=list(dict.fromkeys([
                    *context.failure_categories, "report_composition_error",
                ])),
                investigation_trace=[
                    f"{artifact.producer}: {artifact.summary}"
                    for artifact in context.artifacts.values()
                ],
                workstream_diagnostics=context.workstream_diagnostics,
                verification_status="insufficient_evidence",
            )
        events.stage("pipeline_end", verification_status=report.verification_status)
        return redact_pipeline_report(report, settings)
    finally:
        bim.close()
