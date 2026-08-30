from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .local_python import LocalPythonSandbox
from .model_tools import ModelAssistedTools, UNVERIFIED_STANDARDS_DISCLAIMER, _review_disclosure_text
from .models import AnswerReport
from .project_tools import RawProjectTools
from .pricing import estimate_run_cost, response_usage_record
from .tracing import TraceLog
from .question_planning import QuestionPlan


_ROUTING_DIRECTORY = Path(__file__).with_name("skills") / "bim-routing"
_ROUTING_PATHS = tuple(
    _ROUTING_DIRECTORY / name
    for name in (
        "SKILL.md",
        "SCOPE_AND_METRIC_CONTRACT.md",
        "IDENTITY_RECONCILIATION_POLICY.md",
        "QUERY_AND_STOP_PLAYBOOK.md",
    )
)
_ROUTING_PATH = _ROUTING_PATHS[0]
ROUTING_INSTRUCTIONS = "\n\n".join(
    path.read_text(encoding="utf-8") for path in _ROUTING_PATHS
)

COST_BUDGET_NOTICE = (
    "⚠️ Partial answer: the configured per-answer cost budget was reached. The findings above are limited to "
    "evidence already collected; unresolved checks were not completed."
)
EVIDENCE_PARTIAL_NOTICE = (
    "⚠️ Partial answer: only claims supported by cited tool observations are included; unsupported or "
    "unresolved claims from the draft were omitted."
)
NO_SUPPORTED_PARTIAL_NOTICE = (
    "⚠️ Partial answer: no factual statement from the final draft could be retained as supported by its cited "
    "observations. The collected evidence remains available for a revised answer or rerun."
)


AGENT_INSTRUCTIONS = """You are the BIM analysis execution agent. A schema-aware preflight contract may be
provided with the request. Treat its validated population, metric, unit, relationship, ambiguity, and route as
execution constraints, but never as project evidence. Decide which permitted tools to call, in which order, and
when the investigation is finished.

Ground every project claim in the supplied read-only tools. Begin by inspecting or searching the project when
you need project facts. You may revisit tools, change interpretation, inspect counterexamples, retrieve raw
evidence, or omit a claim that remains unsupported. Never replace supported findings with a blanket refusal:
when some checks remain unresolved, return the strongest cited partial answer and state its limitation.
records, search IFC relationships, and calculate as often as useful. For project filters, joins, counts, and
aggregates, use the read-only SQL workspace; use calculate only for arithmetic over values already retrieved.
Distinguish hierarchy/definition records from
physical instances using the evidence you inspect; do not assume a category boundary without looking at paths
and properties. Treat the three supplied files as the available project scope and disclose material uncertainty.
Scope explorers and evidence reviewers are model-assisted advice, not project observations: cite a direct project
tool as well for every category, exclusion, count, or other project fact, in every language.
Do not assume the files are current or mutually synchronized. When revision/export metadata is unavailable, scope
the answer to the loaded snapshot. If JSON properties and IFC data disagree for the same resolved identity and
concept, report both source-labelled values and treat the conflict as unresolved unless units or provenance explain it.

For bounded, tool-heavy analysis, call describe_bim_workspace once and then author compact queries with
query_bim_workspace. Return grouped counts and totals in the same compact SQL observation whenever the answer will
state both; every cited total must occur literally in its cited observation. Do not retrieve individual leaves when
an aggregate plus a few representative IDs will answer the question. Use named IFC tables and identity candidates before raw IFC text. Never assume a fixed tree
depth identifies instances, and never treat a repeated GUID or property value as proof that physical records are
duplicates. When a concept or category boundary is ambiguous, use explore_object_scope and then
choose among its alternatives yourself. Use analyze_ifc_geometry whenever placement, actual dimensions, area,
volume, or spatial containment matters; compare geometry-derived and property-derived data. Use rank_ifc_geometry
for largest/smallest/longest comparisons across a population. Preserve every material dimension axis: do not merge
objects that share width but differ in height, depth, material, type, or unit unless you explicitly disclose that
aggregation. Keep property length, bounding-box X/Y/Z, maximum extent, bounding-box volume, and solid volume distinct.
When available, use run_local_python for multi-stage Python, graph traversal, all-model geometry ranking,
statistical checks, or reconciliations that are cumbersome in one SQL query. It runs in a local Docker sandbox,
not in an OpenAI container. Read /workspace/bim_workspace_guide.json first. Raw project files are read-only under
/project and bim_workspace.sqlite is read-only under /workspace. Use sqlite3 URI mode=ro, write temporary data only
under /tmp, and print compact evidence, row counts, excluded populations, and reconciliation totals. Networking is disabled.
For exhaustive, ranking, cross-source, or compliance conclusions, use reconcile_populations (or an equivalent
explicit Python reconciliation) to reconcile the selected population and compare the relevant raw-record,
external-ID, IFC-ID, and geometry identities. A routine count over a clear SQL-defined population does not require
reconciliation by itself. A mismatch is a reason to investigate, not permission to pick the smallest count.
For connectivity questions, inspect named relationship roles and use analyze_ifc_graph when reachability matters.
Distinguish missing metadata, containment, logical system membership, physical port connectivity, and actual graph
reachability; raw IFC references alone are not connectivity evidence.
Use model-authored programmatic tool calling, when available, only to filter, join, deduplicate, aggregate, or
validate several predictable tool results into a smaller evidence object. Keep adaptive semantic decisions and
final validation in direct model turns.

Before finishing a geometric, connectivity, compliance, materially ambiguous, or multi-source answer, call
review_scope_and_evidence with your proposed scope, exclusions, evidence, and draft, then decide whether follow-up
investigation is needed. A routine single-source count or list over a schema-validated SQL population does not
need a separate model review. For a
normative compliance question, use research_standards when external requirements are necessary, and keep those
requirements distinct from project facts. Do not ask the user to choose a floor or interpretation when the project
data lets you report all material alternatives concisely.
Answer in the user's language. Give the direct result first, then concise evidence and limitations. Do not expose
private chain-of-thought. Do not author snapshot, scope, uniqueness, source-mismatch, unit, or type-semantics
disclosure sentences yourself; the runtime appends required disclosures in canonical cited form. Return a normal
assistant message only when you judge the substantive answer ready. Preserve any material, evidence-backed
exclusion that changes the meaning of a reported total, such as separately modeled accessories or fittings.

The versioned routing and evidence rules below are mandatory:
""" + ROUTING_INSTRUCTIONS


class ModelConfigurationError(RuntimeError):
    pass


class ToolObservationError(RuntimeError):
    pass


class ModelDirectedBimAgent:
    """The model owns the loop and tool selection; Python only executes requested tools."""

    def __init__(
        self,
        *,
        tools: RawProjectTools,
        model: str,
        reasoning_effort: str,
        max_iterations: int,
        max_tool_output_chars: int,
        max_answer_cost_usd: float = 0.0,
        enable_local_python: bool,
        local_python_image: str,
        python_memory_limit: str,
        local_python_cpus: float,
        local_python_timeout_seconds: float,
        local_python_output_chars: int,
        python_cache_root: Path,
        model_tool_reasoning_effort: str,
        finalization_reasoning_effort: str,
        openai_max_retries: int = 4,
        openai_timeout_seconds: float = 180.0,
        pricing_overrides: dict[str, dict[str, float]] | None = None,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from openai import OpenAI

                try:
                    client = OpenAI(max_retries=openai_max_retries, timeout=openai_timeout_seconds)
                except TypeError:
                    # Compatibility with lightweight test/adaptor clients that expose
                    # the same API surface but do not accept SDK transport options.
                    client = OpenAI()
            except Exception as exc:
                raise ModelConfigurationError(
                    "OPENAI_API_KEY is required; deterministic and offline fallbacks were removed."
                ) from exc
        self.client = client
        self.tools = tools
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.model_tool_reasoning_effort = model_tool_reasoning_effort
        self.finalization_reasoning_effort = finalization_reasoning_effort
        self.max_iterations = max(1, max_iterations)
        self.max_tool_output_chars = max(2000, max_tool_output_chars)
        self.max_answer_cost_usd = max(0.0, float(max_answer_cost_usd))
        self.openai_timeout_seconds = max(10.0, float(openai_timeout_seconds))
        self.pricing_overrides = copy.deepcopy(pricing_overrides or {})
        self.model_tools = ModelAssistedTools(
            client=client,
            model=model,
            reasoning_effort=self.model_tool_reasoning_effort,
            project_tools=tools,
        )
        self.python_sandbox = LocalPythonSandbox(
            project_tools=tools,
            cache_root=python_cache_root,
            enabled=enable_local_python,
            image=local_python_image,
            memory_limit=python_memory_limit,
            cpus=local_python_cpus,
            execution_timeout_seconds=local_python_timeout_seconds,
            max_output_characters=local_python_output_chars,
        )
        self.programmatic_tool_calling = model.casefold().startswith("gpt-5.6")
        self.prompt_cache_key = _prompt_cache_key("agent", model, tools)

    def run(
        self,
        question: str,
        trace: TraceLog,
        *,
        question_plan: QuestionPlan | None = None,
    ) -> AnswerReport:
        input_items: list[Any] = [{"role": "user", "content": question}]
        if question_plan is not None:
            planning_payload = question_plan.as_dict()
            input_items.append({
                "role": "developer",
                "content": (
                    "Schema-aware preflight contract (constraints, not project evidence): "
                    + json.dumps(planning_payload, ensure_ascii=False, separators=(",", ":"))
                    + " Execute within this contract. If execution_decision is clarify, perform only bounded "
                      "read-only inspection that can resolve the stated ambiguity; otherwise ask the supplied "
                      "clarification question. If execution_decision is report_alternatives, preserve every "
                      "material alternative in the answer."
                ),
            })
            if question_plan.schema_resolution_observation is not None:
                input_items.append({
                    "role": "developer",
                    "content": (
                        "Mandatory runtime schema-resolution observation (validated planning constraint, not "
                        "answer evidence): "
                        + json.dumps(
                            question_plan.schema_resolution_observation,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                })
        iterations: list[dict[str, Any]] = []
        response_ids: list[str] = []
        final_answer = ""
        status = "limited"
        termination_reason = "iteration_limit"
        route = (
            copy.deepcopy(question_plan.route)
            if question_plan is not None
            else _classify_route(question)
        )
        if question_plan is not None:
            route["scope_preflight_resolved"] = _plan_has_resolved_scope(question_plan)
            route["model_review_required"] = _plan_requires_model_review(question_plan)
        reconciliation_required = (
            "reconciliation" in set(route.get("required_capabilities", []))
            or _requires_population_reconciliation(question)
        )
        api_tools = self._api_tools(question, planned_route=route if question_plan is not None else None)
        tool_names = [_tool_label(item) for item in api_tools]
        exposed_function_names = {
            str(item.get("name")) for item in api_tools if item.get("type") == "function"
        }
        cache: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        evidence_aliases: dict[str, str] = {}
        tool_call_ordinal = 0
        # A valid preflight inspected the runtime schema. This satisfies the planned
        # schema capability, but it is deliberately not counted as project evidence.
        tool_categories: set[str] = (
            {"schema"} if question_plan is not None and question_plan.contract_valid else set()
        )
        observed_sources: set[str] = set()
        outstanding_cursors: set[str] = set()
        completeness_forced = False
        completeness_attempts = 0
        completion_limited = False
        unresolved_completion_issues: list[str] = []
        best_grounded_candidate = ""
        best_grounded_candidate_score: tuple[int, int, int] = (-10_000, -1, -1)
        best_grounded_candidate_iteration: int | None = None
        best_grounded_candidate_issues: list[str] = []
        best_grounded_candidate_is_salvaged = False
        grounding_content_forced = False
        grounding_format_forced = False
        cost_budget_exceeded = False
        fetch_more_calls = 0
        unverified_disclaimer_required = False
        review_seen = False
        review_can_finalize = True
        review_stale = False
        review_required_follow_up: list[str] = []
        review_required_disclosures: list[str] = []
        review_unsupported_claims: list[str] = []
        population_mismatch_observed = False
        disclosure_state: dict[str, dict[str, Any]] = {}
        citation_repairs: list[dict[str, Any]] = []
        usage_records: list[dict[str, Any]] = (
            [copy.deepcopy(item) for item in question_plan.all_usage_records]
            if question_plan is not None
            else []
        )
        model_tool_usage_index = 0
        self.model_tools.reset_usage()
        python_status_at_start = self.python_sandbox.status()
        trace.transcript("user", content=question)
        trace.event(
            "model_agent_start",
            model=self.model,
            max_iterations=self.max_iterations,
            tools=tool_names,
            route=route,
            programmatic_tool_calling=self.programmatic_tool_calling,
            local_python=python_status_at_start,
            max_answer_cost_usd=self.max_answer_cost_usd or None,
            reconciliation_required=reconciliation_required,
            planning=question_plan.as_dict() if question_plan is not None else None,
        )

        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=AGENT_INSTRUCTIONS,
                    input=input_items,
                    tools=api_tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    reasoning={"effort": self.reasoning_effort},
                    store=False,
                    max_output_tokens=5000,
                    prompt_cache_key=self.prompt_cache_key,
                    timeout=self.openai_timeout_seconds,
                )
            except Exception as exc:
                trace.event(
                    "model_api_error",
                    iteration=iteration,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    retryable=_is_retryable_api_error(exc),
                )
                raise
            response_id = str(getattr(response, "id", ""))
            if response_id:
                response_ids.append(response_id)
            output = list(getattr(response, "output", []) or [])
            calls = [item for item in output if getattr(item, "type", None) == "function_call"]
            candidate_answer = str(getattr(response, "output_text", "") or "").strip()
            python_calls: list[dict[str, Any]] = []
            usage_records.append(response_usage_record(
                response,
                configured_model=self.model,
                purpose="agent_turn",
            ))
            trace.transcript(
                "assistant",
                response_id=response_id,
                text=candidate_answer,
                function_calls=[
                    {
                        "name": str(getattr(item, "name", "")),
                        "call_id": str(getattr(item, "call_id", "")),
                        "arguments": str(getattr(item, "arguments", "{}") or "{}"),
                    }
                    for item in calls
                ],
            )
            trace.event(
                "model_decision",
                iteration=iteration,
                response_id=response_id,
                tool_calls=[getattr(item, "name", "") for item in calls],
                program_items=sum(1 for item in output if getattr(item, "type", None) == "program"),
                program_outputs=sum(1 for item in output if getattr(item, "type", None) == "program_output"),
                python_calls=len(python_calls),
            )
            response_cost_limit_reached = _cost_limit_reached(
                usage_records,
                limit_usd=self.max_answer_cost_usd,
                pricing_overrides=self.pricing_overrides,
            )
            # A no-tool draft is grounded and checkpointed below before a budget
            # stop can replace it. Tool calls still stop before execution.
            if response_cost_limit_reached and (calls or not candidate_answer):
                cost_budget_exceeded = True
                termination_reason = "cost_budget_exceeded"
                final_answer, budget_metadata, disclosure_state, current_repairs = self._budget_answer(
                    question=question,
                    input_items=(
                        [*input_items, *output]
                        if not calls and not candidate_answer and output
                        else input_items
                    ),
                    candidate_answer=best_grounded_candidate,
                    evidence=evidence,
                    evidence_aliases=evidence_aliases,
                    disclosure_codes=review_required_disclosures,
                    require_disclaimer=unverified_disclaimer_required,
                    trace=trace,
                    usage_records=usage_records,
                    response_ids=response_ids,
                    iteration=iteration,
                    trigger="before_tool_execution" if calls else "after_answer_generation",
                )
                citation_repairs.extend(current_repairs)
                iterations.append({
                    "iteration": iteration,
                    "response_id": response_id,
                    "action": "cost_budget_answer",
                    "budget_finalization": budget_metadata,
                    "python_calls": python_calls,
                })
                break

            if not calls:
                if candidate_answer:
                    if _is_single_clarifying_question(candidate_answer):
                        final_answer = candidate_answer
                        status = "completed"
                        termination_reason = "clarification_requested"
                        iterations.append({
                            "iteration": iteration,
                            "response_id": response_id,
                            "action": "request_clarification",
                            "python_calls": python_calls,
                        })
                        break
                    candidate_answer, disclosure_state = _ensure_review_disclosures(
                        question,
                        candidate_answer,
                        evidence,
                        evidence_aliases=evidence_aliases,
                        disclosure_codes=review_required_disclosures,
                    )
                    completion_issues = _completion_issues(
                        question,
                        tool_categories=tool_categories,
                        outstanding_cursors=outstanding_cursors,
                        route=route,
                        observed_sources=observed_sources,
                        reconciliation_required=reconciliation_required,
                    )
                    completion_issues.extend(_review_completion_issues(
                        candidate_answer,
                        review_seen=review_seen,
                        review_can_finalize=review_can_finalize,
                        review_stale=review_stale,
                        required_follow_up=review_required_follow_up,
                        required_disclosures=review_required_disclosures,
                        disclosure_state=disclosure_state,
                        unsupported_claims=review_unsupported_claims,
                    ))
                    completion_issues = list(dict.fromkeys(completion_issues))
                    trusted_disclosures = {
                        str(item.get("text", ""))
                        for item in disclosure_state.values()
                        if item.get("citation_valid")
                    }
                    candidate_answer, current_repairs = _repair_citation_placement(
                        question,
                        candidate_answer,
                        evidence,
                        evidence_aliases=evidence_aliases,
                        trusted_claims=trusted_disclosures,
                    )
                    if current_repairs:
                        citation_repairs.extend(current_repairs)
                        trace.transcript(
                            "gate",
                            gate="citation_placement_repair",
                            accepted=True,
                            repairs=current_repairs,
                        )
                    grounding_issues = _grounding_issues(
                        question,
                        candidate_answer,
                        evidence,
                        evidence_aliases=evidence_aliases,
                        require_disclaimer=unverified_disclaimer_required,
                        trusted_claims=trusted_disclosures,
                    )
                    salvaged_candidate = ""
                    if grounding_issues:
                        salvaged_candidate = _evidence_supported_partial_answer(
                            question,
                            candidate_answer,
                            evidence,
                            evidence_aliases=evidence_aliases,
                            trusted_claims=trusted_disclosures,
                        )
                        if salvaged_candidate:
                            salvaged_score = (
                                -len(completion_issues) - 1,
                                len(set(_REF_RE.findall(salvaged_candidate))),
                                len(salvaged_candidate),
                            )
                            if salvaged_score > best_grounded_candidate_score:
                                best_grounded_candidate = salvaged_candidate
                                best_grounded_candidate_score = salvaged_score
                                best_grounded_candidate_iteration = iteration
                                best_grounded_candidate_issues = list(completion_issues)
                                best_grounded_candidate_is_salvaged = True
                                trace.transcript(
                                    "gate",
                                    gate="evidence_partial_checkpoint",
                                    accepted=True,
                                    iteration=iteration,
                                    omitted_claim_count=len(grounding_issues),
                                    citation_count=salvaged_score[1],
                                )
                    if grounding_issues:
                        trace.transcript("gate", gate="grounding", accepted=False, issues=grounding_issues)
                        issue_category = _grounding_issue_category(grounding_issues)
                        if (
                            completion_issues
                            and iteration < self.max_iterations
                            and not response_cost_limit_reached
                        ):
                            completeness_forced = True
                            completeness_attempts += 1
                            feedback = (
                                "Do not finalize yet. Resolve these outstanding verification checks: "
                                + "; ".join(completion_issues)
                                + ". Also correct the citation-grounding issues: "
                                + " ".join(grounding_issues)
                                + " Use project tools only where evidence is missing, then cite the resulting "
                                  "direct observations inline."
                            )
                            input_items.extend([
                                {"role": "assistant", "content": candidate_answer},
                                {"role": "user", "content": feedback},
                            ])
                            trace.transcript(
                                "gate", gate="completeness", accepted=False, issues=completion_issues
                            )
                            iterations.append({
                                "iteration": iteration,
                                "response_id": response_id,
                                "action": "completeness_continue",
                                "issues": completion_issues,
                                "grounding_issues": grounding_issues,
                                "checkpointed": False,
                                "python_calls": python_calls,
                            })
                            continue
                        if response_cost_limit_reached:
                            cost_budget_exceeded = True
                            termination_reason = "cost_budget_exceeded"
                            if best_grounded_candidate:
                                final_answer, budget_metadata, disclosure_state, current_repairs = self._budget_answer(
                                    question=question,
                                    input_items=input_items,
                                    candidate_answer=best_grounded_candidate,
                                    evidence=evidence,
                                    evidence_aliases=evidence_aliases,
                                    disclosure_codes=review_required_disclosures,
                                    require_disclaimer=unverified_disclaimer_required,
                                    trace=trace,
                                    usage_records=usage_records,
                                    response_ids=response_ids,
                                    iteration=iteration,
                                    trigger="after_grounding_rejection",
                                )
                                citation_repairs.extend(current_repairs)
                            else:
                                final_answer = _append_budget_notice(NO_SUPPORTED_PARTIAL_NOTICE)
                                budget_metadata = {
                                    "trigger": "after_grounding_rejection",
                                    "finalization_attempted": False,
                                    "generated": False,
                                    "fallback_used": True,
                                    "response_id": "",
                                    "generation_error": None,
                                    "grounding_issues": grounding_issues,
                                }
                            iterations.append({
                                "iteration": iteration,
                                "response_id": response_id,
                                "action": "cost_budget_answer",
                                "budget_finalization": budget_metadata,
                                "issues": grounding_issues,
                                "python_calls": python_calls,
                            })
                            break
                        retry_used = (
                            grounding_format_forced
                            if issue_category == "formatting"
                            else grounding_content_forced
                        )
                        if not retry_used and iteration < self.max_iterations:
                            if issue_category == "formatting":
                                grounding_format_forced = True
                                grounding_feedback = (
                                    "The factual content has supporting evidence, but citation placement is "
                                    "incomplete: " + " ".join(grounding_issues)
                                    + " Attach an existing supporting reference to every affected sentence. "
                                      "Do not repeat an identical tool call solely to correct formatting."
                                )
                            else:
                                grounding_content_forced = True
                                grounding_feedback = (
                                    "The proposed answer failed the citation-grounding check: "
                                    + " ".join(grounding_issues)
                                    + " Either call a tool to verify each claim or remove it. Return inline "
                                      "references in the form [ref: call_id]. For multiple observations, "
                                      "repeat the tag, for example [ref: call_3] [ref: call_7]. Cite only IDs "
                                      "from direct function-call outputs, their supplied _citation_reference "
                                      "aliases, or automatic reconciliation observations."
                                )
                            input_items.extend([
                                {"role": "assistant", "content": candidate_answer},
                                {"role": "user", "content": grounding_feedback},
                            ])
                            iterations.append({
                                "iteration": iteration,
                                "response_id": response_id,
                                "action": "grounding_continue",
                                "grounding_issue_category": issue_category,
                                "issues": grounding_issues,
                                "python_calls": python_calls,
                            })
                            continue
                        if best_grounded_candidate:
                            completion_limited = True
                            unresolved_completion_issues = list(best_grounded_candidate_issues)
                            final_answer = best_grounded_candidate
                            if unresolved_completion_issues:
                                final_answer = _append_incomplete_checks(
                                    final_answer, unresolved_completion_issues
                                )
                            if best_grounded_candidate_is_salvaged:
                                final_answer = _append_partial_notice(final_answer)
                            termination_reason = "evidence_supported_partial_after_rejected_retry"
                            iterations.append({
                                "iteration": iteration,
                                "response_id": response_id,
                                "action": "grounding_rejected_use_partial",
                                "issues": grounding_issues,
                                "checkpoint_iteration": best_grounded_candidate_iteration,
                                "python_calls": python_calls,
                            })
                        else:
                            final_answer = NO_SUPPORTED_PARTIAL_NOTICE
                            termination_reason = "no_supported_partial_after_rejected_retry"
                            iterations.append({
                                "iteration": iteration,
                                "response_id": response_id,
                                "action": "grounding_rejected_no_supported_partial",
                                "issues": grounding_issues,
                                "python_calls": python_calls,
                            })
                        break

                    candidate_score = (
                        -len(completion_issues),
                        len(set(_REF_RE.findall(candidate_answer))),
                        len(candidate_answer),
                    )
                    if candidate_score > best_grounded_candidate_score:
                        best_grounded_candidate = candidate_answer
                        best_grounded_candidate_score = candidate_score
                        best_grounded_candidate_iteration = iteration
                        best_grounded_candidate_issues = list(completion_issues)
                        best_grounded_candidate_is_salvaged = False
                        trace.transcript(
                            "gate",
                            gate="grounded_candidate_checkpoint",
                            accepted=True,
                            iteration=iteration,
                            unresolved_checks=completion_issues,
                            citation_count=candidate_score[1],
                        )

                    if response_cost_limit_reached:
                        cost_budget_exceeded = True
                        termination_reason = "cost_budget_exceeded"
                        final_answer, budget_metadata, disclosure_state, current_repairs = self._budget_answer(
                            question=question,
                            input_items=input_items,
                            candidate_answer=best_grounded_candidate,
                            evidence=evidence,
                            evidence_aliases=evidence_aliases,
                            disclosure_codes=review_required_disclosures,
                            require_disclaimer=unverified_disclaimer_required,
                            trace=trace,
                            usage_records=usage_records,
                            response_ids=response_ids,
                            iteration=iteration,
                            trigger="after_answer_generation",
                        )
                        citation_repairs.extend(current_repairs)
                        iterations.append({
                            "iteration": iteration,
                            "response_id": response_id,
                            "action": "cost_budget_answer",
                            "budget_finalization": budget_metadata,
                            "checkpoint_iteration": best_grounded_candidate_iteration,
                            "python_calls": python_calls,
                        })
                        break

                    if completion_issues and iteration < self.max_iterations:
                        completeness_forced = True
                        completeness_attempts += 1
                        feedback = (
                            "Do not finalize yet. Continue from all accumulated observations and resolve only "
                            "these outstanding verification checks: "
                            + "; ".join(completion_issues)
                            + ". Reuse existing evidence and completed reconciliations. Call additional tools "
                              "only where a listed check still lacks evidence, then return the strongest fully "
                              "supported answer. The prior grounded draft is checkpointed and will not be lost."
                        )
                        input_items.extend([
                            {"role": "assistant", "content": candidate_answer},
                            {"role": "user", "content": feedback},
                        ])
                        trace.transcript("gate", gate="completeness", accepted=False, issues=completion_issues)
                        iterations.append({
                            "iteration": iteration,
                            "response_id": response_id,
                            "action": "completeness_continue",
                            "issues": completion_issues,
                            "checkpointed": True,
                            "python_calls": python_calls,
                        })
                        continue
                    if completion_issues:
                        completion_limited = True
                        unresolved_completion_issues = list(best_grounded_candidate_issues)
                        candidate_answer = best_grounded_candidate
                        trace.transcript(
                            "gate", gate="completeness", accepted=False,
                            issues=unresolved_completion_issues,
                        )
                        iterations.append({
                            "iteration": iteration,
                            "response_id": response_id,
                            "action": "completeness_limited",
                            "issues": unresolved_completion_issues,
                            "checkpoint_iteration": best_grounded_candidate_iteration,
                            "python_calls": python_calls,
                        })

                    final_answer = (
                        _append_incomplete_checks(candidate_answer, unresolved_completion_issues)
                        if completion_limited
                        else candidate_answer
                    )
                    status = "limited" if completion_limited else "completed"
                    termination_reason = (
                        "completed_with_incomplete_checks" if completion_limited else "completed"
                    )
                    trace.transcript("gate", gate="grounding", accepted=True)
                    iterations.append({
                        "iteration": iteration,
                        "response_id": response_id,
                        "action": "finish",
                        "python_calls": python_calls,
                    })
                    break
                input_items.extend(output)
                iterations.append({
                    "iteration": iteration,
                    "response_id": response_id,
                    "action": "program_continue" if any(
                        getattr(item, "type", None) == "program_output" for item in output
                    ) else ("python_continue" if python_calls else "empty_response"),
                    "python_calls": python_calls,
                })
                if not output:
                    input_items.append({
                        "role": "user",
                        "content": "Continue the investigation or provide the final answer.",
                    })
                continue

            input_items.extend(output)
            call_records = []
            cost_reached_during_calls = False
            for call in calls:
                tool_call_ordinal += 1
                citation_alias = f"call_{tool_call_ordinal}"
                name = str(getattr(call, "name", ""))
                call_id = str(getattr(call, "call_id", ""))
                raw_arguments = str(getattr(call, "arguments", "{}") or "{}")
                result: dict[str, Any] | None = None
                cached = False
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    if name not in exposed_function_names:
                        raise ToolObservationError(
                            f"Tool {name} is not exposed for this question's enforced route."
                        )
                    if name == "fetch_more":
                        if fetch_more_calls >= 5:
                            raise ToolObservationError(
                                "Pagination stopped after 5 continuation calls. Use a compact SQL aggregate or "
                                "a narrower query instead of paging a large population."
                            )
                        fetch_more_calls += 1
                    cache_key = _tool_cache_key(name, arguments)
                    if cache_key in cache:
                        result = copy.deepcopy(cache[cache_key])
                        cached = True
                    else:
                        result = self._execute_tool(name, arguments)
                        cache[cache_key] = copy.deepcopy(result)
                    _validate_tool_result(name, result)
                    model_result = _model_tool_result(name, result)
                    model_result["_citation_reference"] = citation_alias
                    output_text, truncated = _tool_output(model_result, self.max_tool_output_chars)
                    outcome = "ok"
                except Exception as exc:
                    arguments = {"raw": raw_arguments}
                    output_text = json.dumps(
                        {"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False
                    )
                    truncated = False
                    outcome = "error"
                current_model_tool_usage = self.model_tools.usage_records()
                if len(current_model_tool_usage) > model_tool_usage_index:
                    usage_records.extend(current_model_tool_usage[model_tool_usage_index:])
                    model_tool_usage_index = len(current_model_tool_usage)
                    cost_reached_during_calls = _cost_limit_reached(
                        usage_records,
                        limit_usd=self.max_answer_cost_usd,
                        pricing_overrides=self.pricing_overrides,
                    )
                call_output: dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
                caller = _caller_payload(getattr(call, "caller", None))
                if caller is not None:
                    call_output["caller"] = caller
                input_items.append(call_output)
                category = _tool_category(name)
                if outcome == "ok":
                    tool_categories.add(category)
                    observed_sources.update(_observed_sources(name, result))
                    if name == "fetch_more":
                        outstanding_cursors.discard(str(arguments.get("cursor", "")))
                    if isinstance(result, dict) and result.get("cursor"):
                        outstanding_cursors.add(str(result["cursor"]))
                    evidence_class = _evidence_class(name, result)
                    evidence[call_id] = {
                        "tool": name,
                        "output": output_text,
                        "result": result,
                        "evidence_class": evidence_class,
                    }
                    evidence_aliases[citation_alias] = call_id
                    if name == "reconcile_populations" and isinstance(result, dict):
                        population_mismatch_observed = (
                            population_mismatch_observed or _reconciliation_mismatch(result)
                        )
                    if name == "review_scope_and_evidence" and isinstance(result, dict):
                        review_seen = True
                        review_can_finalize = bool(result.get("can_finalize", True))
                        review_stale = False
                        review_required_follow_up = [
                            str(item) for item in result.get("required_follow_up", [])
                        ]
                        requested_disclosures = [
                            str(item) for item in result.get("required_disclosures", [])
                        ]
                        # A model review is advice, not project evidence. It may
                        # require a mismatch disclosure only after a direct
                        # reconciliation observation has actually found one.
                        review_required_disclosures = _validated_review_disclosures(
                            requested_disclosures,
                            population_mismatch_observed=population_mismatch_observed,
                        )
                        if "cross_source_mismatch" in requested_disclosures and not population_mismatch_observed:
                            trace.event(
                                "review_disclosure_rejected",
                                iteration=iteration,
                                code="cross_source_mismatch",
                                reason="no_direct_population_mismatch_observed",
                            )
                        disclosure_state = {}
                        review_unsupported_claims = [
                            str(item) for item in result.get("unsupported_claims", [])
                        ]
                    elif review_seen and _is_project_observation(name):
                        review_stale = True
                    if (
                        name == "research_standards"
                        and isinstance(result, dict)
                        and not result.get("verified")
                    ):
                        unverified_disclaimer_required = True
                record = {
                    "tool": name,
                    "call_id": call_id,
                    "citation_alias": citation_alias,
                    "arguments": arguments,
                    "outcome": outcome,
                    "output_characters": len(output_text),
                    "truncated": truncated,
                    "cached": cached,
                }
                call_records.append(record)
                trace.event("tool_execution", iteration=iteration, **record)
                trace.transcript(
                    "tool",
                    tool=name,
                    call_id=call_id,
                    citation_alias=citation_alias,
                    arguments=arguments,
                    outcome=outcome,
                    cached=cached,
                    output=result if outcome == "ok" else output_text,
                )

                if cost_reached_during_calls:
                    break

                automatic_population_ids: list[str] = []
                if (
                    outcome == "ok"
                    and isinstance(result, dict)
                    and reconciliation_required
                    and "reconciliation" not in tool_categories
                    and _needs_reconciliation(name, result)
                ):
                    automatic_population_ids = _population_ids(name, arguments, result, self.tools)
                if automatic_population_ids:
                    auto_id = f"auto_reconcile_{call_id}"
                    reconciliation_args = {
                        "populations": [{
                            "label": f"{name}:{call_id}",
                            "object_ids": automatic_population_ids,
                        }]
                    }
                    reconciliation_key = _tool_cache_key("reconcile_populations", reconciliation_args)
                    reconciliation_cached = reconciliation_key in cache
                    if reconciliation_cached:
                        reconciliation = copy.deepcopy(cache[reconciliation_key])
                    else:
                        reconciliation = self.tools.execute("reconcile_populations", reconciliation_args)
                        cache[reconciliation_key] = copy.deepcopy(reconciliation)
                    reconciliation["automatic"] = True
                    reconciliation["trigger_call_id"] = call_id
                    reconciliation["population_ids_available"] = True
                    reconciliation["mismatch"] = _reconciliation_mismatch(reconciliation)
                    population_mismatch_observed = (
                        population_mismatch_observed or bool(reconciliation["mismatch"])
                    )
                    reconciliation_text, reconciliation_truncated = _tool_output(
                        reconciliation, self.max_tool_output_chars
                    )
                    input_items.append({
                        "role": "user",
                        "content": (
                            f"Automatic population reconciliation observation [ref: {auto_id}]: "
                            + reconciliation_text
                        ),
                    })
                    evidence[auto_id] = {
                        "tool": "reconcile_populations",
                        "output": reconciliation_text,
                        "result": reconciliation,
                        "evidence_class": "project",
                    }
                    tool_categories.add("reconciliation")
                    auto_record = {
                        "tool": "reconcile_populations",
                        "call_id": auto_id,
                        "arguments": reconciliation_args,
                        "outcome": "ok",
                        "output_characters": len(reconciliation_text),
                        "truncated": reconciliation_truncated,
                        "cached": reconciliation_cached,
                        "automatic": True,
                        "mismatch": reconciliation["mismatch"],
                    }
                    call_records.append(auto_record)
                    trace.event("tool_execution", iteration=iteration, **auto_record)
                    trace.transcript(
                        "tool",
                        tool="reconcile_populations",
                        call_id=auto_id,
                        arguments=reconciliation_args,
                        outcome="ok",
                        cached=reconciliation_cached,
                        automatic=True,
                        output=reconciliation,
                    )
            iterations.append({
                "iteration": iteration,
                "response_id": response_id,
                "action": "tool_calls",
                "calls": call_records,
                "python_calls": python_calls,
            })
            if _cost_limit_reached(
                usage_records,
                limit_usd=self.max_answer_cost_usd,
                pricing_overrides=self.pricing_overrides,
            ):
                cost_budget_exceeded = True
                termination_reason = "cost_budget_exceeded"
                final_answer, budget_metadata, disclosure_state, current_repairs = self._budget_answer(
                    question=question,
                    input_items=input_items,
                    candidate_answer=best_grounded_candidate,
                    evidence=evidence,
                    evidence_aliases=evidence_aliases,
                    disclosure_codes=review_required_disclosures,
                    require_disclaimer=unverified_disclaimer_required,
                    trace=trace,
                    usage_records=usage_records,
                    response_ids=response_ids,
                    iteration=iteration,
                    trigger="after_tool_execution",
                )
                citation_repairs.extend(current_repairs)
                iterations[-1]["action"] = "cost_budget_answer"
                iterations[-1]["budget_finalization"] = budget_metadata
                break

        if not final_answer and best_grounded_candidate:
            completion_limited = True
            unresolved_completion_issues = list(best_grounded_candidate_issues)
            final_answer = _append_incomplete_checks(
                best_grounded_candidate, unresolved_completion_issues
            )
            if best_grounded_candidate_is_salvaged:
                final_answer = _append_partial_notice(final_answer)
            termination_reason = "best_grounded_candidate_at_iteration_limit"

        limitations = [
            (
                "Semantic scope was constrained by a schema-aware preflight; deterministic completeness, "
                "pagination, reconciliation, and citation-grounding guards check execution and termination."
                if question_plan is not None and question_plan.contract_valid
                else "Semantic scope remains model-directed; deterministic completeness, pagination, "
                     "reconciliation, and citation-grounding guards check termination."
            )
        ]
        python_status = self.python_sandbox.status()
        if python_status["enabled"] and not any(
            item.get("name") == "run_local_python" for item in api_tools
        ):
            if route["project_data_operation"]:
                limitations.append(
                    "Local Python was intentionally not exposed because this project filter/aggregate is routed to read-only SQL."
                )
            else:
                limitations.append(
                    "The optional local Docker Python sandbox was unavailable for this run: "
                    + str(python_status.get("last_error") or "Docker is not ready")
                )
        if status != "completed":
            final_answer = final_answer or (
                "The model-directed agent did not produce a final answer before its iteration safety limit."
            )
            if termination_reason in {
                "evidence_supported_partial_after_rejected_retry",
                "no_supported_partial_after_rejected_retry",
            }:
                limitations.append(
                    "Unsupported final-draft claims were omitted; the returned result is limited to the "
                    "evidence-supported portion."
                )
            elif termination_reason == "completed_with_incomplete_checks":
                limitations.append(
                    "The answer preserves grounded findings, but one or more required scope, population, "
                    "source, reconciliation, or pagination checks remained incomplete at the iteration limit."
                )
            elif termination_reason == "cost_budget_exceeded":
                limitations.append(
                    f"The run reached the configured ${self.max_answer_cost_usd:g} per-answer cost guard."
                )
            else:
                limitations.append(f"The run stopped after {self.max_iterations} model iterations.")
        cost = estimate_run_cost(
            usage_records,
            pricing_overrides=self.pricing_overrides,
        )
        estimated_cost = cost.get("estimated_cost_usd")
        if (
            self.max_answer_cost_usd > 0
            and isinstance(estimated_cost, (int, float))
            and estimated_cost >= self.max_answer_cost_usd
            and not cost_budget_exceeded
        ):
            cost_budget_exceeded = True
            termination_reason = "cost_budget_exceeded"
            status = "limited"
            final_answer = _append_budget_notice(final_answer)
            limitations.append(
                f"The run reached the configured ${self.max_answer_cost_usd:g} per-answer cost guard."
            )
        cost["budget_usd"] = self.max_answer_cost_usd or None
        cost["budget_exceeded"] = cost_budget_exceeded
        trace.transcript("cost", **cost)
        trace.transcript("final", status=status, answer=final_answer)
        trace.event(
            "model_agent_finish",
            status=status,
            iterations=len(iterations),
            response_ids=response_ids,
        )
        return AnswerReport(
            answer=final_answer,
            status=status,
            sources=self.tools.manifest(),
            trace_path=str(trace.path),
            cost=cost,
            limitations=limitations,
            agent_loop={
                "pattern": "model_directed_tool_loop",
                "model": self.model,
                "finished": status == "completed",
                "iterations_used": len(iterations),
                "max_iterations": self.max_iterations,
                "response_ids": response_ids,
                "trace_session_id": trace.session_id,
                "tools_available": tool_names,
                "route": route,
                "planning": question_plan.as_dict() if question_plan is not None else None,
                "completeness_forced": completeness_forced,
                "completeness_attempts": completeness_attempts,
                "completion_limited": completion_limited,
                "unresolved_completion_issues": unresolved_completion_issues,
                "best_grounded_candidate": {
                    "available": bool(best_grounded_candidate),
                    "iteration": best_grounded_candidate_iteration,
                    "unresolved_checks": best_grounded_candidate_issues,
                    "citation_count": (
                        len(set(_REF_RE.findall(best_grounded_candidate)))
                        if best_grounded_candidate else 0
                    ),
                    "salvaged_partial": best_grounded_candidate_is_salvaged,
                },
                "grounding_forced": grounding_content_forced or grounding_format_forced,
                "grounding_content_forced": grounding_content_forced,
                "grounding_format_forced": grounding_format_forced,
                "citation_repairs": citation_repairs,
                "review_seen": review_seen,
                "review_can_finalize": review_can_finalize,
                "review_stale": review_stale,
                "review_required_follow_up": review_required_follow_up,
                "review_required_disclosures": review_required_disclosures,
                "review_satisfied_disclosures": sorted(
                    code for code, item in disclosure_state.items() if item.get("citation_valid")
                ),
                "disclosure_state": disclosure_state,
                "review_unsupported_claims": review_unsupported_claims,
                "evidence_aliases": evidence_aliases,
                "cost_budget_usd": self.max_answer_cost_usd or None,
                "cost_budget_exceeded": cost_budget_exceeded,
                "termination_reason": termination_reason,
                "programmatic_tool_calling": self.programmatic_tool_calling,
                "iterations": iterations,
            },
        )

    def _budget_answer(
        self,
        *,
        question: str,
        input_items: list[Any],
        candidate_answer: str,
        evidence: dict[str, dict[str, Any]],
        evidence_aliases: dict[str, str],
        disclosure_codes: list[str],
        require_disclaimer: bool,
        trace: TraceLog,
        usage_records: list[dict[str, Any]],
        response_ids: list[str],
        iteration: int,
        trigger: str,
    ) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Produce one bounded, no-tools answer from accumulated evidence when investigation spending stops."""
        draft = candidate_answer.strip()
        generated_response_id = ""
        generation_error = ""
        finalization_attempted = not bool(draft)
        generated = False
        if not draft:
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=(
                        AGENT_INSTRUCTIONS
                        + "\nThe investigation cost budget is exhausted. Do not call any tool. Give a concise "
                          "best-effort answer using only observations already present in the input. Cite every "
                          "factual sentence with an existing [ref: call_id]. State that a requested value is "
                          "unavailable when the observations do not establish it. Do not write disclosure or "
                          "cost-budget notices; the runtime appends them."
                    ),
                    input=[
                        *input_items,
                        {
                            "role": "user",
                            "content": (
                                "Stop investigating and answer now from the evidence already collected. "
                                "Do not request or call another tool."
                            ),
                        },
                    ],
                    reasoning={"effort": self.finalization_reasoning_effort},
                    store=False,
                    max_output_tokens=1200,
                    prompt_cache_key=f"{self.prompt_cache_key}-budget"[:64],
                    timeout=self.openai_timeout_seconds,
                )
                usage_records.append(response_usage_record(
                    response,
                    configured_model=self.model,
                    purpose="budget_finalization",
                ))
                generated_response_id = str(getattr(response, "id", "") or "")
                if generated_response_id:
                    response_ids.append(generated_response_id)
                draft = str(getattr(response, "output_text", "") or "").strip()
                generated = bool(draft)
                trace.transcript(
                    "assistant",
                    response_id=generated_response_id,
                    text=draft,
                    function_calls=[],
                    budget_finalization=True,
                )
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                trace.event(
                    "budget_finalization_error",
                    iteration=iteration,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
        draft = draft.replace(COST_BUDGET_NOTICE, "").strip()
        if not draft:
            draft = NO_SUPPORTED_PARTIAL_NOTICE
        if require_disclaimer and UNVERIFIED_STANDARDS_DISCLAIMER not in draft:
            draft = f"{draft.rstrip()}\n\n{UNVERIFIED_STANDARDS_DISCLAIMER}"
        draft, disclosure_state = _ensure_review_disclosures(
            question,
            draft,
            evidence,
            evidence_aliases=evidence_aliases,
            disclosure_codes=disclosure_codes,
        )
        trusted_disclosures = {
            str(item.get("text", ""))
            for item in disclosure_state.values()
            if item.get("citation_valid")
        }
        draft, citation_repairs = _repair_citation_placement(
            question,
            draft,
            evidence,
            evidence_aliases=evidence_aliases,
            trusted_claims=trusted_disclosures,
        )
        grounding_issues = _grounding_issues(
            question,
            draft,
            evidence,
            evidence_aliases=evidence_aliases,
            require_disclaimer=require_disclaimer,
            trusted_claims=trusted_disclosures,
        )
        partial_salvage_used = False
        if grounding_issues:
            supported_partial = _evidence_supported_partial_answer(
                question,
                draft,
                evidence,
                evidence_aliases=evidence_aliases,
                trusted_claims=trusted_disclosures,
            )
            if supported_partial:
                draft = supported_partial
                partial_salvage_used = True
            else:
                draft = NO_SUPPORTED_PARTIAL_NOTICE
        final_answer = _append_budget_notice(draft)
        metadata = {
            "trigger": trigger,
            "finalization_attempted": finalization_attempted,
            "generated": generated,
            "fallback_used": finalization_attempted and not generated,
            "response_id": generated_response_id,
            "generation_error": generation_error or None,
            "grounding_issues": grounding_issues,
            "partial_salvage_used": partial_salvage_used,
        }
        trace.event(
            "cost_budget_exceeded",
            iteration=iteration,
            max_answer_cost_usd=self.max_answer_cost_usd,
            best_effort_answer=True,
            **metadata,
        )
        return final_answer, metadata, disclosure_state, citation_repairs

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "run_local_python":
            return self.python_sandbox.execute(arguments)
        if name in {item["name"] for item in self.model_tools.definitions()}:
            return self.model_tools.execute(name, arguments)
        return self.tools.execute(name, arguments)

    def _api_tools(
        self,
        question: str | None = None,
        *,
        planned_route: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        project_definitions = copy.deepcopy(self.tools.definitions())
        model_definitions = copy.deepcopy(self.model_tools.definitions())
        python_definition: dict[str, Any] | None = None
        if planned_route is not None:
            allowed = {
                str(item) for item in planned_route.get("exposed_tool_names", []) if str(item)
            }
            # The deprecated aggregate tool is never restored by a plan; SQL is
            # the single certifying path for JSON-backed aggregates.
            allowed.discard("aggregate_records")
            if planned_route.get("scope_preflight_resolved"):
                allowed.discard("explore_object_scope")
            if not planned_route.get("model_review_required", True):
                allowed.discard("review_scope_and_evidence")
            project_definitions = [
                item for item in project_definitions
                if item.get("name") in allowed and item.get("name") != "aggregate_records"
            ]
            model_definitions = [
                item for item in model_definitions if item.get("name") in allowed
            ]
            if "run_local_python" in allowed:
                python_definition = self.python_sandbox.tool_definition()
        elif question is not None:
            route = _classify_route(question)
            # SQL is the sole general aggregation/filter/join path over project data.
            project_definitions = [
                item for item in project_definitions if item.get("name") != "aggregate_records"
            ]
            if route["project_data_operation"]:
                project_definitions = [
                    item for item in project_definitions if item.get("name") != "calculate"
                ]
            else:
                python_definition = self.python_sandbox.tool_definition()
        else:
            python_definition = self.python_sandbox.tool_definition()
        local_tools = [python_definition] if python_definition is not None else []
        if not self.programmatic_tool_calling:
            return [*project_definitions, *model_definitions, *local_tools]
        for definition in project_definitions:
            definition["allowed_callers"] = ["direct", "programmatic"]
        for definition in model_definitions:
            definition["allowed_callers"] = ["direct"]
        for definition in local_tools:
            definition["allowed_callers"] = ["direct"]
        return [
            *project_definitions,
            *model_definitions,
            *local_tools,
            {"type": "programmatic_tool_calling"},
        ]

    def inspect(self) -> dict[str, Any]:
        api_tools = self._api_tools()
        return {
            "pattern": "model_directed_tool_loop",
            "model": self.model,
            "max_iterations": self.max_iterations,
            "max_answer_cost_usd": self.max_answer_cost_usd or None,
            "tools": [
                {"name": _tool_label(item), "description": item.get("description", "")}
                for item in api_tools
            ],
            "programmatic_tool_calling": self.programmatic_tool_calling,
            "local_python": self.python_sandbox.status(),
            "deterministic_semantic_components": [
                "tool_route_filter", "call_deduplication", "automatic_reconciliation",
                "completeness_gate", "pagination_gate", "citation_grounding_gate", "cost_budget_gate",
            ],
            "remaining_code": "Semantic scope and analysis choices remain model-controlled; read-only execution, routing, pagination, reconciliation, citation validation, tracing, and safety limits are enforced in code.",
        }


_REF_RE = re.compile(r"\[ref:\s*([A-Za-z0-9_.:-]+)\s*\]", re.IGNORECASE)
_SPACED_REF_RE = re.compile(r"\s*\[ref:\s*[A-Za-z0-9_.:-]+\s*\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?")
_NUMBER_WORD_VALUES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "אפס": "0", "אחד": "1", "אחת": "1", "שניים": "2", "שתיים": "2", "שני": "2",
    "שתי": "2", "שלושה": "3", "שלוש": "3", "ארבעה": "4", "ארבע": "4",
    "חמישה": "5", "חמש": "5", "שישה": "6", "שש": "6", "שבעה": "7", "שבע": "7",
    "שמונה": "8", "תשעה": "9", "תשע": "9", "עשרה": "10", "עשר": "10",
}
_PROJECT_TERMS = re.compile(
    r"\b(projects?|models?|bim|ifc|objects?|records?|elements?|floors?|levels?|rooms?|zones?|"
    r"buildings?|walls?|doors?|windows?|pipes?|ducts?|switch(?:es)?|equipment|fixtures?|"
    r"geometry|volumes?|areas?|lengths?|properties)\b",
    re.IGNORECASE,
)
_HEBREW_PROJECT_TERMS = re.compile(
    "(?:\u05e4\u05e8\u05d5\u05d9\u05e7\u05d8|\u05de\u05d5\u05d3\u05dc|\u05e7\u05d5\u05de(?:\u05d4|\u05d5\u05ea)|"
    "\u05d7\u05d3\u05e8(?:\u05d9\u05dd)?|\u05e9\u05d8\u05d7(?:\u05d9\u05dd)?|\u05ea\u05d0\u05d5\u05e8\u05d4|"
    "\u05d0\u05dc\u05de\u05e0\u05d8(?:\u05d9\u05dd)?|\u05d2\u05d5\u05e4\u05d9 \u05ea\u05d0\u05d5\u05e8\u05d4)"
)
_HEBREW_DATA_OPERATION_TERMS = re.compile(
    "(?:\u05dc\u05db\u05dc|\u05d1\u05db\u05dc|\u05db\u05de\u05d4|\u05e1\u05db\u05d5\u05dd|\u05e1\u05da|"
    "\u05de\u05de\u05d5\u05e6\u05e2|\u05de\u05d9\u05e0\u05d9\u05de\u05d5\u05dd|\u05de\u05e7\u05e1\u05d9\u05de\u05d5\u05dd|"
    "\u05dc\u05e4\u05d9|\u05e6\u05e4\u05d9\u05e4\u05d5\u05ea)"
)


def _is_project_question(value: str) -> bool:
    return bool(_PROJECT_TERMS.search(value) or _HEBREW_PROJECT_TERMS.search(value))


def _classify_route(question: str) -> dict[str, Any]:
    normalized = question.casefold()
    data_operation = bool(re.search(
        r"\b(how many|count|total|sum|average|mean|minimum|maximum|min|max|group|per|each|"
        r"filter|where|whose|with|without|join|list all|all objects|all records)\b",
        normalized,
    )) or bool(_HEBREW_DATA_OPERATION_TERMS.search(normalized))
    data_operation = data_operation and _is_project_question(question)
    return {
        "project_data_operation": data_operation,
        "general_compute_path": "sql_only" if data_operation else "adaptive",
        "calculate_exposed": not data_operation,
        "local_python_exposed": not data_operation,
        "routing_rules": str(_ROUTING_PATH),
    }


def _tool_cache_key(name: str, arguments: dict[str, Any]) -> str:
    normalized_arguments = copy.deepcopy(arguments)
    if name == "query_bim_workspace" and isinstance(normalized_arguments.get("sql"), str):
        normalized_arguments["sql"] = normalized_arguments["sql"].strip().rstrip(";").rstrip()
    canonical = json.dumps(
        normalized_arguments,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{name}\0{canonical}".encode("utf-8")).hexdigest()


def _tool_category(name: str) -> str:
    mapping = {
        "inspect_project": "hierarchy",
        "list_tree_children": "hierarchy",
        "search_records": "records",
        "get_records": "records",
        "aggregate_records": "sql",
        "describe_bim_workspace": "schema",
        "query_bim_workspace": "sql",
        "search_ifc": "ifc",
        "fetch_more": "pagination",
        "calculate": "arithmetic",
        "analyze_ifc_geometry": "geometry",
        "rank_ifc_geometry": "geometry",
        "analyze_ifc_graph": "graph",
        "reconcile_populations": "reconciliation",
        "explore_object_scope": "scope",
        "review_scope_and_evidence": "review",
        "research_standards": "standards",
        "run_local_python": "custom_compute",
    }
    return mapping.get(name, "other")


def _plan_has_resolved_scope(plan: QuestionPlan) -> bool:
    """Return true only when preflight resolved every material scope choice."""
    if not plan.contract_valid:
        return False
    ambiguities = plan.interpretation_plan.get("ambiguities", [])
    if any(isinstance(item, dict) and item.get("material") for item in ambiguities):
        return False
    resolution = plan.schema_resolution_observation
    if isinstance(resolution, dict):
        return bool(
            resolution.get("runtime_validated")
            and resolution.get("complete")
            and resolution.get("primary_candidate_id")
        )
    population = plan.interpretation_plan.get("population")
    filters = population.get("filters", []) if isinstance(population, dict) else []
    mappings = plan.interpretation_plan.get("schema_grounded_mappings", [])
    return bool(filters or mappings)


def _plan_requires_model_review(plan: QuestionPlan) -> bool:
    """Reserve the critic for evidence contracts where it adds a distinct check."""
    if not plan.contract_valid or not _plan_has_resolved_scope(plan):
        return True
    route = plan.route
    answer_shape = str(route.get("answer_shape") or "")
    if answer_shape in {"ranking", "measurement", "comparison", "connectivity", "compliance"}:
        return True
    capabilities = {str(item) for item in route.get("required_capabilities", [])}
    if capabilities.intersection({"geometry", "graph", "standards_research"}):
        return True
    sources = {str(item) for item in route.get("required_sources", [])}
    return len(sources) > 1


def _completion_issues(
    question: str,
    *,
    tool_categories: set[str],
    outstanding_cursors: set[str],
    route: dict[str, Any] | None = None,
    observed_sources: set[str] | None = None,
    reconciliation_required: bool | None = None,
    reconciliation_status: str | None = None,
) -> list[str]:
    normalized = question.casefold()
    issues: list[str] = []
    substantive = tool_categories.difference({"reconciliation", "review", "pagination", "schema"})
    project_question = _is_project_question(question)
    quantitative = bool(re.search(
        r"\b(how many|count|total|sum|average|mean|minimum|maximum|min|max|largest|smallest|longest|shortest)\b",
        normalized,
    )) or bool(_HEBREW_DATA_OPERATION_TERMS.search(normalized))
    if project_question and not substantive:
        issues.append("any project observation")
    if project_question and quantitative and not ({"sql", "geometry"} & substantive):
        issues.append("the requested project aggregate or complete comparison through SQL/geometry evidence")
    complete_population = quantitative or bool(re.search(r"\b(all|every|complete|list|ranking)\b", normalized))
    requires_reconciliation = (
        _requires_population_reconciliation(question)
        if reconciliation_required is None
        else reconciliation_required
    )
    if (
        project_question
        and requires_reconciliation
        and "reconciliation" not in tool_categories
    ):
        issues.append(
            "population reconciliation before an exhaustive, ranking, cross-source, or compliance claim"
        )
    if (
        re.search(r"\b(floor|level|storey|story|room|zone|location|located)\b", normalized)
        or re.search("(?:\u05e7\u05d5\u05de(?:\u05d4|\u05d5\u05ea)|\u05d7\u05d3\u05e8(?:\u05d9\u05dd)?)", normalized)
    ) and not (
        {"sql", "records", "geometry"} & substantive
    ):
        issues.append("the location condition")
    if re.search(r"\b(largest|smallest|longest|shortest|compare|versus|\bvs\b)\b", normalized) and not (
        {"sql", "geometry"} & substantive
    ):
        issues.append("the complete comparison population")
    if " and " in normalized and len(substantive) < 2:
        issues.append("all compound conditions with distinct evidence categories")
    if outstanding_cursors and complete_population:
        issues.append("all remaining result pages via fetch_more")
    ambiguous_scope = bool(re.search(
        r"\b(types?|kinds?|categories|category|famil(?:y|ies)|scope|which room|which location|"
        r"property name|ambiguous)\b",
        normalized,
    ))
    scope_preflight_resolved = bool(route and route.get("scope_preflight_resolved"))
    model_review_required = bool(route and route.get("model_review_required", True))
    if (
        project_question and ambiguous_scope and not scope_preflight_resolved
        and "scope" not in tool_categories
    ):
        issues.append("the ambiguous category, location, or property boundary with explore_object_scope")
    if (
        project_question and ambiguous_scope and model_review_required
        and (scope_preflight_resolved or "scope" in tool_categories)
        and "review" not in tool_categories
    ):
        issues.append("the chosen ambiguous scope with review_scope_and_evidence")
    if route and route.get("requirements_enforced"):
        capability_categories = {
            "schema_inspection": {"schema"},
            "hierarchy": {"hierarchy"},
            "record_query": {"sql", "records"},
            "ifc_semantics": {"ifc", "graph"},
            "geometry": {"geometry"},
            "graph": {"graph"},
            "reconciliation": {"reconciliation"},
            "arithmetic": {"arithmetic"},
            "local_python": {"custom_compute"},
            "standards_research": {"standards"},
        }
        for capability in route.get("required_capabilities", []):
            accepted = capability_categories.get(str(capability), set())
            if accepted and not accepted.intersection(tool_categories):
                issues.append(f"the planned {capability} capability")
        source_categories = {
            "tree": {"tree"},
            "properties": {"properties"},
            "ifc_semantics": {"ifc_semantics"},
            "ifc_geometry": {"ifc_geometry"},
            "external_standard": {"external_standard"},
        }
        seen_sources = observed_sources or set()
        for source in route.get("required_sources", []):
            accepted = source_categories.get(str(source), set())
            if accepted and not accepted.intersection(seen_sources):
                issues.append(f"the planned {source} evidence source")
        if (
            "reconciliation" in route.get("required_capabilities", [])
            and reconciliation_status in {"mismatched", "incomplete"}
        ):
            issues.append("a matched, provenance-valid population reconciliation")
    return list(dict.fromkeys(issues))


def _observed_sources(name: str, result: dict[str, Any] | None) -> set[str]:
    """Map successful observations to source roles without inferring facts."""
    if name in {"inspect_project", "list_tree_children"}:
        return {"tree"}
    if name in {"search_records", "get_records"}:
        return {"properties"}
    if name == "query_bim_workspace":
        serialized = json.dumps(result or {}, ensure_ascii=True).casefold()
        sources: set[str] = set()
        if any(term in serialized for term in ("record", "propert", "tree_node")):
            sources.add("properties")
        if "tree_node" in serialized:
            sources.add("tree")
        if any(term in serialized for term in ("ifc_", "global_id", "ifc_step_id")):
            sources.add("ifc_semantics")
        # A SQL observation may intentionally project aliases without table
        # names. Treat it as properties evidence unless it proves an IFC role.
        return sources or {"properties"}
    if name in {"search_ifc", "analyze_ifc_graph"}:
        return {"ifc_semantics"}
    if name in {"analyze_ifc_geometry", "rank_ifc_geometry"}:
        return {"ifc_geometry", "ifc_semantics"}
    if name == "research_standards":
        return {"external_standard"}
    if name == "run_local_python":
        return {"properties", "ifc_semantics", "ifc_geometry"}
    if name == "reconcile_populations":
        return {"tree", "properties", "ifc_semantics"}
    return set()


def _review_completion_issues(
    answer: str,
    *,
    review_seen: bool,
    review_can_finalize: bool,
    review_stale: bool,
    required_follow_up: list[str],
    required_disclosures: list[str],
    disclosure_state: dict[str, dict[str, Any]],
    unsupported_claims: list[str],
) -> list[str]:
    if not review_seen:
        return []
    issues: list[str] = []
    if review_stale:
        issues.append("the new project evidence with a fresh review_scope_and_evidence call")
    if not review_can_finalize:
        detail = "; ".join(required_follow_up[:4]) or "the material gaps identified by evidence review"
        issues.append(f"the evidence review's required follow-up: {detail}")
    for disclosure_code in required_disclosures:
        state = disclosure_state.get(disclosure_code, {})
        if not state.get("citation_valid"):
            disclosure = _review_disclosure_text(
                disclosure_code, bool(re.search(r"[\u0590-\u05FF]", answer))
            )
            issues.append(f"the evidence review's required disclosure: {disclosure}")
    normalized_answer = " ".join(answer.casefold().split())
    for claim in unsupported_claims:
        normalized = " ".join(claim.casefold().split())
        if normalized and normalized in normalized_answer:
            issues.append(f"removal or verification of the unsupported claim: {claim}")
    return issues


def _validated_review_disclosures(
    requested: list[str], *, population_mismatch_observed: bool,
) -> list[str]:
    """Require direct project evidence before asserting a cross-source mismatch."""
    return list(dict.fromkeys(
        code for code in requested
        if code != "cross_source_mismatch" or population_mismatch_observed
    ))


def _split_line_claims(line: str) -> list[str]:
    """Split prose into sentences while binding each inline ref to the sentence immediately before it."""
    claims: list[str] = []
    cursor = 0
    while cursor < len(line):
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
        if cursor >= len(line):
            break
        punctuation = re.search(r"[.!?](?=\s|$)", line[cursor:])
        if punctuation is None:
            claims.append(line[cursor:].strip())
            break
        end = cursor + punctuation.end()
        reference_end = end
        while True:
            reference = _SPACED_REF_RE.match(line, reference_end)
            if reference is None:
                break
            reference_end = reference.end()
        claims.append(line[cursor:reference_end].strip())
        cursor = reference_end
    return [item for item in claims if item]


def _grounding_issues(
    question: str,
    answer: str,
    evidence: dict[str, dict[str, Any]],
    *,
    evidence_aliases: dict[str, str] | None = None,
    require_disclaimer: bool,
    trusted_claims: set[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    if require_disclaimer and UNVERIFIED_STANDARDS_DISCLAIMER not in answer:
        issues.append(f"The mandatory disclaimer is missing: {UNVERIFIED_STANDARDS_DISCLAIMER}")
    requires_grounding = bool(evidence) or _is_project_question(question) or bool(_NUMBER_RE.search(answer))
    if not requires_grounding:
        return issues

    claims = [
        item.strip(" -•\t")
        for line in answer.splitlines()
        for item in _split_line_claims(line)
        if item.strip(" -•\t")
    ]
    for claim in claims:
        if claim == UNVERIFIED_STANDARDS_DISCLAIMER or claim.endswith(":"):
            continue
        if _markdown_heading_scaffolding(claim) or _markdown_table_scaffolding(claim):
            continue
        refs = _REF_RE.findall(claim)
        claim_text = _REF_RE.sub("", claim).strip()
        if not any(character.isalnum() for character in claim_text):
            continue
        normalized_claim = " ".join(claim_text.casefold().split())
        if any(
            normalized_claim == " ".join(item.casefold().split())
            for item in (trusted_claims or set())
        ):
            continue
        if not refs:
            issues.append(f"Claim has no inline tool reference: {claim_text[:180]!r}.")
            continue
        aliases = evidence_aliases or {}
        resolved_refs = [ref if ref in evidence else aliases.get(ref, ref) for ref in refs]
        missing = [
            ref for ref, resolved in zip(refs, resolved_refs)
            if resolved not in evidence
        ]
        if missing:
            issues.append(f"Claim cites unknown tool output(s): {', '.join(missing)}.")
            continue
        cited = [evidence[ref] for ref in resolved_refs]
        combined_output = "\n".join(str(item.get("output", "")) for item in cited)
        project_context = any(item.get("evidence_class") == "project" for item in evidence.values())
        if _looks_project_claim(question, claim_text, project_context=project_context) and not any(
            item.get("evidence_class") == "project" for item in cited
        ):
            issues.append(f"Project claim is backed only by non-project evidence: {claim_text[:180]!r}.")
            continue
        numbers = _claim_numbers(claim_text)
        evidence_numbers = [
            value.replace(",", "") for value in _NUMBER_RE.findall(combined_output)
        ]
        unmatched_numbers = [
            value for value in numbers
            if not any(_numeric_strings_equal(value, observed) for observed in evidence_numbers)
        ]
        if unmatched_numbers:
            issues.append(
                f"Claimed value(s) {', '.join(unmatched_numbers)} do not occur in the cited output(s) for: "
                f"{claim_text[:180]!r}."
            )
            continue
        if not numbers and not _textual_evidence_overlap(claim_text, combined_output):
            issues.append(f"Claim text does not match the cited output: {claim_text[:180]!r}.")
    return issues


def _grounding_issue_category(issues: list[str]) -> str:
    if issues and all(item.startswith("Claim has no inline tool reference:") for item in issues):
        return "formatting"
    return "content"


def _evidence_supported_partial_answer(
    question: str,
    answer: str,
    evidence: dict[str, dict[str, Any]],
    *,
    evidence_aliases: dict[str, str],
    trusted_claims: set[str] | None = None,
) -> str:
    """Retain independently grounded draft lines while omitting rejected claims.

    Presentation-only headings and table headers are carried forward only when
    at least one substantive cited claim survives. The warning is appended by
    the caller after validation so it is never mistaken for a project claim.
    """
    kept_lines: list[str] = []
    pending_scaffolding: list[str] = []
    substantive_claims = 0
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                pending_scaffolding.append("")
            continue
        claims = [item for item in _split_line_claims(line) if item.strip()]
        if claims and all(
            claim.rstrip().endswith(":")
            or _markdown_heading_scaffolding(claim)
            or _markdown_table_scaffolding(claim)
            for claim in claims
        ):
            pending_scaffolding.append(line)
            continue
        supported_claims = [
            claim for claim in claims
            if not _grounding_issues(
                question,
                claim,
                evidence,
                evidence_aliases=evidence_aliases,
                require_disclaimer=False,
                trusted_claims=trusted_claims,
            )
        ]
        if not supported_claims:
            pending_scaffolding = []
            continue
        if pending_scaffolding:
            kept_lines.extend(pending_scaffolding)
            pending_scaffolding = []
        kept_lines.append(" ".join(supported_claims))
        substantive_claims += len(supported_claims)
    if not substantive_claims:
        return ""
    while kept_lines and not kept_lines[-1].strip():
        kept_lines.pop()
    return "\n".join(kept_lines).strip()


def _repair_citation_placement(
    question: str,
    answer: str,
    evidence: dict[str, dict[str, Any]],
    *,
    evidence_aliases: dict[str, str],
    trusted_claims: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Copy trailing citations to earlier sentences only when those citations independently ground them."""
    repaired_lines: list[str] = []
    repairs: list[dict[str, Any]] = []
    for line in answer.splitlines():
        sentences = _split_line_claims(line)
        if len(sentences) < 2:
            repaired_lines.append(line)
            continue
        trailing_refs = _REF_RE.findall(sentences[-1])
        if not trailing_refs:
            repaired_lines.append(line)
            continue
        citation_text = " ".join(f"[ref: {reference}]" for reference in trailing_refs)
        for index, sentence in enumerate(sentences[:-1]):
            claim_text = _REF_RE.sub("", sentence).strip()
            if not claim_text or _REF_RE.search(sentence) or claim_text.endswith(":"):
                continue
            trial = f"{sentence.rstrip()} {citation_text}"
            if not _grounding_issues(
                question,
                trial,
                evidence,
                evidence_aliases=evidence_aliases,
                require_disclaimer=False,
                trusted_claims=trusted_claims,
            ):
                sentences[index] = trial
                repairs.append({"claim": claim_text, "references": trailing_refs})
        repaired_lines.append(" ".join(sentences))
    return "\n".join(repaired_lines), repairs


def _looks_project_claim(question: str, claim: str, *, project_context: bool = False) -> bool:
    standards_claim = bool(re.search(
        r"\b(standard|code|regulation|requirement|guidance)\b", claim, re.IGNORECASE
    ))
    return _is_project_question(claim) or (
        _is_project_question(question)
        and not standards_claim
    ) or (
        project_context and not standards_claim
    )


def _numeric_strings_equal(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except ValueError:
        return left == right


def _claim_numbers(value: str) -> list[str]:
    numbers = [item.replace(",", "") for item in _NUMBER_RE.findall(value)]
    words = re.findall(r"[^\W\d_]+", value.casefold(), flags=re.UNICODE)
    numbers.extend(_NUMBER_WORD_VALUES[word] for word in words if word in _NUMBER_WORD_VALUES)
    return numbers


def _textual_evidence_overlap(claim: str, output: str) -> bool:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
        "and", "or", "that", "this", "there", "it", "from", "with", "as", "by", "answer",
        "record", "records", "found",
    }
    def tokens(value: str) -> set[str]:
        raw = re.findall(r"[^\W\d_][\w-]{2,}", value.casefold(), flags=re.UNICODE)
        return {item[:-1] if item.endswith("s") and len(item) > 4 else item for item in raw if item not in stop}
    claim_tokens = tokens(claim)
    if not claim_tokens:
        return True
    overlap = claim_tokens.intersection(tokens(output))
    return len(overlap) >= min(2, len(claim_tokens))


def _evidence_class(name: str, result: dict[str, Any] | None) -> str:
    if name == "research_standards":
        if isinstance(result, dict):
            return str(result.get("evidence_class") or "external_standard")
        return "external_standard"
    if name == "calculate":
        return "derived_calculation"
    if name == "run_local_python":
        return "project"
    if name in {"explore_object_scope", "review_scope_and_evidence"}:
        return "model_review"
    return "project"


def _needs_reconciliation(name: str, result: dict[str, Any]) -> bool:
    if name in {
        "reconcile_populations", "calculate", "run_local_python", "describe_bim_workspace",
        "explore_object_scope", "review_scope_and_evidence", "research_standards",
    }:
        return False
    if result.get("truncated") or result.get("cursor"):
        return False
    if name == "query_bim_workspace":
        columns = {str(item).casefold() for item in result.get("columns", [])}
        identity_columns = {
            "object_id", "record_object_id", "record_objectid", "ifc_step_id", "global_id",
        }
        return bool(columns) and columns.issubset(identity_columns)
    return name in {
        "list_tree_children", "search_records", "get_records", "fetch_more",
        "analyze_ifc_geometry", "rank_ifc_geometry",
    }


def _requires_population_reconciliation(question: str) -> bool:
    normalized = question.casefold()
    if re.search(
        r"\b(all|every|complete|exhaustive|entire|whole|total|ranking|ranked|largest|smallest|"
        r"longest|shortest|top\s+\d+|bottom\s+\d+|compliance|compliant|comply|audit|"
        r"reconcile|cross[- ]source|across (?:the )?sources)\b",
        normalized,
    ):
        return True
    if re.search(
        "(?:\u05dc\u05db\u05dc|\u05d1\u05db\u05dc|\u05db\u05dc \u05e7\u05d5\u05de(?:\u05d4|\u05d5\u05ea))",
        normalized,
    ):
        return True
    return any(term in normalized for term in (
        "כל הפרויקט", "בכל הפרויקט", "כל ה", "סה״כ", "סה\"כ", "דירוג",
        "הגדול ביותר", "הקטן ביותר", "הארוך ביותר", "הקצר ביותר", "תאימות", "תקן",
    ))


def _population_ids(
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    tools: RawProjectTools,
) -> list[str]:
    argument_ids: list[str] = []
    result_ids: list[str] = []

    def visit(value: Any, found: list[str], key: str = "") -> None:
        normalized_key = key.casefold()
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, found, str(child_key))
        elif isinstance(value, list):
            if "object_ids" in normalized_key or normalized_key == "record_object_ids":
                found.extend(str(item) for item in value if str(item))
            else:
                for child in value:
                    visit(child, found, key)
        elif value is not None and normalized_key in {
            "object_id", "record_object_id", "record_objectid",
        }:
            found.append(str(value))

    visit(arguments, argument_ids)
    if argument_ids:
        # Preserve duplicates explicitly supplied by the caller so reconciliation can report them.
        return argument_ids
    visit(result, result_ids)
    if name == "inspect_project" and not result_ids:
        result_ids.extend(str(item) for item in tools.by_id)
    # Project tools expose some compatibility aliases (for example results/records).
    # Deduplicate IDs discovered in an observation so aliases do not create false mismatches.
    return list(dict.fromkeys(result_ids))


def _reconciliation_mismatch(result: dict[str, Any]) -> bool:
    if result.get("union_unique_object_ids") != result.get("union_found_records"):
        return True
    for population in result.get("populations", []):
        if population.get("duplicate_occurrences") or population.get("missing_object_ids"):
            return True
    return False


def _validate_tool_result(name: str, result: Any) -> None:
    if result is None:
        raise ToolObservationError(f"Tool {name} returned no observation.")
    if not isinstance(result, dict):
        raise ToolObservationError(
            f"Tool {name} returned {type(result).__name__}; a JSON object observation is required."
        )
    if not result:
        raise ToolObservationError(f"Tool {name} returned an empty observation.")
    if name in {"analyze_ifc_geometry", "rank_ifc_geometry"} and result.get("available") is True:
        rows = result.get("rows")
        if not isinstance(rows, list):
            raise ToolObservationError(f"Tool {name} reported availability but did not return a rows array.")
        returned = result.get("returned_entities")
        if not isinstance(returned, int) or returned != len(rows):
            raise ToolObservationError(
                f"Tool {name} returned an inconsistent entity count ({returned!r} versus {len(rows)} rows)."
            )
        population_key = "selected_entities" if name == "analyze_ifc_geometry" else "matching_entities"
        population = result.get(population_key)
        if not isinstance(population, int) or population < returned:
            raise ToolObservationError(f"Tool {name} returned an invalid {population_key} value.")
    if name == "query_bim_workspace":
        if not isinstance(result.get("columns"), list) or not isinstance(result.get("rows"), list):
            raise ToolObservationError("The BIM workspace query did not return columns and rows arrays.")


def _tool_output(result: dict[str, Any], max_chars: int) -> tuple[str, bool]:
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return serialized, False
    return json.dumps({
        "truncated": True,
        "original_characters": len(serialized),
        "results_omitted": True,
        "instruction": (
            "The complete observation exceeded the transport safety bound. Narrow the query or use a smaller "
            "page size; no partial blob is supplied as evidence."
        ),
    }, ensure_ascii=False), True


def _model_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Create a compact evidence-preserving observation for model context.

    The audit trace retains the complete tool result. Repetitive inventories and
    raw query rows are bounded here so one exploratory call does not inflate every
    later Responses request.
    """
    compact = copy.deepcopy(result)
    if "results" in compact:
        for alias in ("records", "children", "matches"):
            if compact.get(alias) == compact.get("results"):
                compact.pop(alias, None)
        if compact.get("total_matches") == compact.get("total_count"):
            compact.pop("total_matches", None)
        if compact.get("returned") == compact.get("returned_count"):
            compact.pop("returned", None)
    if name == "inspect_project":
        keys = list(compact.get("property_keys", []))
        compact["property_key_count"] = len(keys)
        compact["property_keys"] = keys[:40]
        compact["property_keys_omitted"] = max(0, len(keys) - 40)
        for source in compact.get("source_files", []):
            if isinstance(source, dict) and source.get("path"):
                source["file_name"] = Path(str(source["path"])).name
                source.pop("path", None)
    if name == "query_bim_workspace":
        rows = list(compact.get("rows", []))
        model_row_limit = 30
        if len(rows) > model_row_limit:
            compact["rows"] = rows[:model_row_limit]
            compact["model_rows_omitted"] = len(rows) - model_row_limit
            compact["model_instruction"] = (
                "The full result is retained in the audit trace, but omitted rows are not model evidence. "
                "Issue a compact aggregate or narrower representative query before citing omitted facts."
            )
    if _is_project_observation(name):
        compact["_snapshot_scope"] = {
            "basis": "loaded local project files",
            "freshness": "unverified",
            "cross_file_version_alignment": "unverified",
            "required_disclosure_en": (
                "Scope: results apply to the loaded project snapshot; source freshness and cross-file "
                "version alignment are unverified."
            ),
            "required_disclosure_he": (
                "היקף: הממצאים מתייחסים לצילום נתוני הפרויקט שנטען; עדכניות המקורות והתאמת "
                "הגרסאות בין הקבצים לא אומתו."
            ),
        }
    return compact


def _is_project_observation(name: str) -> bool:
    return name in {
        "inspect_project", "list_tree_children", "search_records", "get_records", "fetch_more",
        "aggregate_records", "search_ifc", "query_bim_workspace", "analyze_ifc_geometry",
        "rank_ifc_geometry", "analyze_ifc_graph", "reconcile_populations", "run_local_python",
    }


def _ensure_snapshot_disclosure(
    question: str,
    answer: str,
    evidence: dict[str, dict[str, Any]],
    *,
    evidence_aliases: dict[str, str],
) -> str:
    alias_by_id = {call_id: alias for alias, call_id in evidence_aliases.items()}
    reference = ""
    for call_id, observation in evidence.items():
        if observation.get("evidence_class") == "project":
            reference = alias_by_id.get(call_id, call_id)
            break
    if not reference:
        return answer
    if "cross-file version alignment" in answer or "התאמת הגרסאות בין הקבצים" in answer:
        return answer
    if re.search(r"[\u0590-\u05FF]", question):
        disclosure = (
            "היקף: הממצאים מתייחסים לצילום נתוני הפרויקט שנטען; עדכניות המקורות והתאמת "
            "הגרסאות בין הקבצים לא אומתו."
        )
    else:
        disclosure = (
            "Scope: results apply to the loaded project snapshot; source freshness and cross-file "
            "version alignment are unverified."
        )
    return f"{answer.rstrip()}\n\n{disclosure} [ref: {reference}]"


def _ensure_review_disclosures(
    question: str,
    answer: str,
    evidence: dict[str, dict[str, Any]],
    *,
    evidence_aliases: dict[str, str],
    disclosure_codes: list[str],
) -> tuple[str, dict[str, dict[str, Any]]]:
    alias_by_id = {call_id: alias for alias, call_id in evidence_aliases.items()}
    project_reference = ""
    review_reference = ""
    project_reference_id = ""
    review_reference_id = ""
    for call_id, observation in evidence.items():
        if observation.get("evidence_class") == "project" and not project_reference:
            project_reference = alias_by_id.get(call_id, call_id)
            project_reference_id = call_id
        if observation.get("tool") == "review_scope_and_evidence":
            review_reference = alias_by_id.get(call_id, call_id)
            review_reference_id = call_id
    desired_codes = list(dict.fromkeys([
        *(("loaded_snapshot",) if project_reference else ()),
        *disclosure_codes,
    ]))
    if not desired_codes:
        return answer, {}
    hebrew = bool(re.search(r"[\u0590-\u05FF]", question))
    output = _strip_model_disclosures(answer, desired_codes).rstrip()
    state: dict[str, dict[str, Any]] = {}
    for code in desired_codes:
        disclosure = _review_disclosure_text(code, hebrew)
        references = [project_reference] if project_reference else []
        resolved_references = [project_reference_id] if project_reference_id else []
        review_required = code in disclosure_codes
        if review_required and review_reference:
            references.append(review_reference)
            resolved_references.append(review_reference_id)
        citation_valid = bool(project_reference) and (not review_required or bool(review_reference))
        state[code] = {
            "code": code,
            "text": disclosure,
            "references": references,
            "resolved_references": resolved_references,
            "citation_valid": citation_valid,
        }
        if citation_valid:
            citations = " ".join(f"[ref: {reference}]" for reference in references)
            output += f"\n\n{disclosure} {citations}"
    return output, state


def _strip_model_disclosures(answer: str, disclosure_codes: list[str]) -> str:
    output_lines: list[str] = []
    for line in answer.splitlines():
        kept: list[str] = []
        claims = [item.strip() for item in _split_line_claims(line) if item.strip()]
        for claim in claims:
            claim_text = _REF_RE.sub("", claim).strip()
            if any(_disclosure_present(code, claim_text) for code in disclosure_codes):
                continue
            kept.append(claim)
        output_lines.append(" ".join(kept))
    return "\n".join(output_lines)


def _disclosure_present(code: str, answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    standardized = {
        " ".join(_review_disclosure_text(code, False).casefold().split()),
        " ".join(_review_disclosure_text(code, True).casefold().split()),
    }
    if any(text and text in normalized for text in standardized):
        return True
    if code == "loaded_snapshot":
        return (
            "cross-file version alignment" in normalized
            or ("loaded" in normalized and "snapshot" in normalized and "freshness" in normalized)
            or ("צילום" in normalized and "גרס" in normalized)
        )
    if code == "record_count_not_physical_uniqueness":
        return (
            "physical" in normalized and any(
                phrase in normalized for phrase in ("not proof", "does not prove", "not necessarily prove")
            )
        ) or (
            "פיזי" in normalized and "מוכיח" in normalized
        )
    if code == "selected_scope_only":
        return (
            "scope" in normalized and any(
                phrase in normalized for phrase in ("only", "outside", "exclude")
            )
        ) or (
            "בלבד" in normalized and any(phrase in normalized for phrase in ("אינה כוללת", "לא כולל"))
        )
    if code == "cross_source_mismatch":
        return (
            "source" in normalized and "mismatch" in normalized and "unresolved" in normalized
        ) or ("מקור" in normalized and "פער" in normalized and "פתור" in normalized)
    if code == "unit_uncertainty":
        return (
            "unit" in normalized and "uncertain" in normalized
        ) or ("יחיד" in normalized and "ודא" in normalized)
    if code == "ambiguous_type_semantics":
        return (
            "hierarchy label" in normalized and ("family" in normalized or "type" in normalized)
        ) or ("תוויות היררכיה" in normalized and ("משפחה" in normalized or "טיפוס" in normalized))
    return _review_disclosure_text(code, bool(re.search(r"[\u0590-\u05FF]", answer))).casefold() in normalized


def _markdown_table_scaffolding(claim: str) -> bool:
    stripped = claim.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell):
        return True
    return not _NUMBER_RE.search(stripped)


def _markdown_heading_scaffolding(claim: str) -> bool:
    stripped = _REF_RE.sub("", claim).strip()
    if re.fullmatch(r"\*\*[^*\n]+:?\*\*", stripped):
        return True
    if not re.match(r"^#{1,6}\s+", stripped):
        return False
    heading = re.sub(r"^#{1,6}\s+", "", stripped).strip()
    # Plain section labels are presentation, while headings containing a value,
    # rating, or explicit proposition still pass through factual grounding.
    return bool(heading) and not any(marker in heading for marker in (":", "**", "`")) and not _NUMBER_RE.search(heading)


def _is_single_clarifying_question(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped.endswith("?") or _REF_RE.search(stripped) or _NUMBER_RE.search(stripped):
        return False
    sentences = [part for part in re.split(r"[.!?]+", stripped) if part.strip()]
    return len(sentences) == 1 and len(stripped) <= 500


def _append_budget_notice(answer: str) -> str:
    substantive = answer.replace(COST_BUDGET_NOTICE, "").strip()
    return f"{substantive}\n\n{COST_BUDGET_NOTICE}" if substantive else COST_BUDGET_NOTICE


def _append_partial_notice(answer: str) -> str:
    substantive = answer.replace(EVIDENCE_PARTIAL_NOTICE, "").strip()
    return (
        f"{substantive}\n\n{EVIDENCE_PARTIAL_NOTICE}"
        if substantive else NO_SUPPORTED_PARTIAL_NOTICE
    )


def _append_incomplete_checks(answer: str, issues: list[str]) -> str:
    """Preserve grounded findings while making unfinished verification explicit."""
    substantive = answer.strip()
    unique_issues = [item for item in dict.fromkeys(issues) if str(item).strip()]
    if not unique_issues:
        return substantive
    checklist = "\n".join(f"- {item}" for item in unique_issues)
    notice = (
        "Verification still pending because the investigation reached its iteration limit:\n"
        + checklist
        + "\nThe result above is retained only to the extent supported by its cited evidence."
    )
    return f"{substantive}\n\n{notice}" if substantive else notice


def _cost_limit_reached(
    records: list[dict[str, Any]],
    *,
    limit_usd: float,
    pricing_overrides: dict[str, dict[str, float]],
) -> bool:
    if limit_usd <= 0:
        return False
    estimate = estimate_run_cost(records, pricing_overrides=pricing_overrides)
    value = estimate.get("estimated_cost_usd")
    return isinstance(value, (int, float)) and value >= limit_usd


def _prompt_cache_key(purpose: str, model: str, tools: RawProjectTools) -> str:
    digest = hashlib.sha256()
    digest.update(ROUTING_INSTRUCTIONS.encode("utf-8"))
    for item in tools.manifest():
        digest.update(str(item.get("sha256", "")).encode("ascii", errors="ignore"))
    return f"bim-{purpose}-{model}-{digest.hexdigest()[:20]}"[:64]


def _tool_label(definition: dict[str, Any]) -> str:
    return str(definition.get("name") or definition.get("type") or "tool")


def _is_retryable_api_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def _caller_payload(caller: Any) -> Any | None:
    if caller is None:
        return None
    if isinstance(caller, dict):
        return caller
    model_dump = getattr(caller, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    caller_id = getattr(caller, "caller_id", None)
    caller_type = getattr(caller, "type", None)
    if caller_id or caller_type:
        return {"type": str(caller_type or "program"), "caller_id": str(caller_id or "")}
    return caller

# End of module.
