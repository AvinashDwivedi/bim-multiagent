from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .hosted_python import HostedPythonWorkspace
from .model_tools import ModelAssistedTools, UNVERIFIED_STANDARDS_DISCLAIMER
from .models import AnswerReport
from .project_tools import RawProjectTools
from .pricing import estimate_run_cost, response_usage_record
from .tracing import TraceLog


_ROUTING_PATH = Path(__file__).with_name("skills") / "bim-routing" / "SKILL.md"
ROUTING_INSTRUCTIONS = _ROUTING_PATH.read_text(encoding="utf-8")


AGENT_INSTRUCTIONS = """You are the only BIM analysis agent. There is no fixed planner, executor,
verifier, answer template, supervisor, or fallback agent. Decide for yourself which tools to call, in which
order, and when the investigation is finished.

Ground every project claim in the supplied read-only tools. Begin by inspecting or searching the project when
you need project facts. You may revisit tools, change interpretation, inspect counterexamples, retrieve raw
records, search IFC relationships, and calculate as often as useful. For project filters, joins, counts, and
aggregates, use the read-only SQL workspace; use calculate only for arithmetic over values already retrieved.
Distinguish hierarchy/definition records from
physical instances using the evidence you inspect; do not assume a category boundary without looking at paths
and properties. Treat the three supplied files as the available project scope and disclose material uncertainty.
Do not assume the files are current or mutually synchronized. When revision/export metadata is unavailable, scope
the answer to the loaded snapshot. If JSON properties and IFC data disagree for the same resolved identity and
concept, report both source-labelled values and treat the conflict as unresolved unless units or provenance explain it.

For bounded, tool-heavy analysis, call describe_bim_workspace once and then author compact queries with
query_bim_workspace. Use named IFC tables and identity candidates before raw IFC text. Never assume a fixed tree
depth identifies instances, and never treat a repeated GUID or property value as proof that physical records are
duplicates. When a concept or category boundary is ambiguous, use explore_object_scope and then
choose among its alternatives yourself. Use analyze_ifc_geometry whenever placement, actual dimensions, area,
volume, or spatial containment matters; compare geometry-derived and property-derived data. Use rank_ifc_geometry
for largest/smallest/longest comparisons across a population. Preserve every material dimension axis: do not merge
objects that share width but differ in height, depth, material, type, or unit unless you explicitly disclose that
aggregation. Keep property length, bounding-box X/Y/Z, maximum extent, bounding-box volume, and solid volume distinct.
When available, use the sandboxed code_interpreter for multi-stage dataframe work, graph traversal, all-model geometry ranking,
statistical checks, or reconciliations that are cumbersome in one SQL query. Read bim_workspace_guide.json first;
also read upload_manifest.json and reconstruct any gzip or multipart-gzip artifacts before opening them.
the container contains the three raw source files and bim_workspace.sqlite, including the full ifc_geometry table.
Print compact evidence, row counts, excluded populations, and reconciliation totals. Its network is disabled.
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

Before finishing a materially quantitative, geometric, connectivity, ambiguous, or multi-source answer, call
review_scope_and_evidence with your proposed scope, exclusions, evidence, and draft, then decide whether follow-up
investigation is needed. For a
normative compliance question, use research_standards when external requirements are necessary, and keep those
requirements distinct from project facts. Do not ask the user to choose a floor or interpretation when the project
data lets you report all material alternatives concisely.
Answer in the user's language. Give the direct result first, then concise evidence and limitations. Do not expose
private chain-of-thought. Return a normal assistant message only when you judge the answer ready.

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
        enable_hosted_python: bool,
        python_memory_limit: str,
        python_expiry_minutes: int,
        python_cache_root: Path,
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
        self.max_iterations = max(1, max_iterations)
        self.max_tool_output_chars = max(2000, max_tool_output_chars)
        self.max_answer_cost_usd = max(0.0, float(max_answer_cost_usd))
        self.openai_timeout_seconds = max(10.0, float(openai_timeout_seconds))
        self.pricing_overrides = copy.deepcopy(pricing_overrides or {})
        self.model_tools = ModelAssistedTools(
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            project_tools=tools,
        )
        self.python_workspace = HostedPythonWorkspace(
            client=client,
            project_tools=tools,
            cache_root=python_cache_root,
            enabled=enable_hosted_python,
            memory_limit=python_memory_limit,
            expiry_minutes=python_expiry_minutes,
            execution_timeout_seconds=self.openai_timeout_seconds,
        )
        self.programmatic_tool_calling = model.casefold().startswith("gpt-5.6")
        self.prompt_cache_key = _prompt_cache_key("agent", model, tools)

    def run(self, question: str, trace: TraceLog) -> AnswerReport:
        input_items: list[Any] = [{"role": "user", "content": question}]
        iterations: list[dict[str, Any]] = []
        response_ids: list[str] = []
        final_answer = ""
        status = "limited"
        termination_reason = "iteration_limit"
        route = _classify_route(question)
        reconciliation_required = _requires_population_reconciliation(question)
        api_tools = self._api_tools(question)
        tool_names = [_tool_label(item) for item in api_tools]
        exposed_function_names = {
            str(item.get("name")) for item in api_tools if item.get("type") == "function"
        }
        cache: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        evidence_aliases: dict[str, str] = {}
        tool_call_ordinal = 0
        tool_categories: set[str] = set()
        outstanding_cursors: set[str] = set()
        completeness_forced = False
        grounding_forced = False
        cost_budget_exceeded = False
        fetch_more_calls = 0
        unverified_disclaimer_required = False
        usage_records: list[dict[str, Any]] = []
        model_tool_usage_index = 0
        self.model_tools.reset_usage()
        python_status_at_start = self.python_workspace.status()
        trace.transcript("user", content=question)
        trace.event(
            "model_agent_start",
            model=self.model,
            max_iterations=self.max_iterations,
            tools=tool_names,
            route=route,
            programmatic_tool_calling=self.programmatic_tool_calling,
            hosted_python=python_status_at_start,
            max_answer_cost_usd=self.max_answer_cost_usd or None,
            reconciliation_required=reconciliation_required,
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
                    include=["code_interpreter_call.outputs"],
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
            python_calls = [
                _code_interpreter_record(item)
                for item in output if getattr(item, "type", None) == "code_interpreter_call"
            ]
            usage_records.append(response_usage_record(
                response,
                configured_model=self.model,
                purpose="agent_turn",
                excluded_tool_fees=["code_interpreter"] if python_calls else None,
            ))
            if calls and _cost_limit_reached(
                usage_records,
                limit_usd=self.max_answer_cost_usd,
                pricing_overrides=self.pricing_overrides,
            ):
                cost_budget_exceeded = True
                termination_reason = "cost_budget_exceeded"
                final_answer = (
                    "The answer was stopped before another tool call because the configured per-answer cost "
                    "budget was reached. The available evidence was not sufficient for a grounded answer."
                )
                iterations.append({
                    "iteration": iteration,
                    "response_id": response_id,
                    "action": "cost_budget_exceeded",
                    "python_calls": python_calls,
                })
                trace.event(
                    "cost_budget_exceeded",
                    iteration=iteration,
                    max_answer_cost_usd=self.max_answer_cost_usd,
                )
                break
            for python_call in python_calls:
                tool_categories.add("custom_compute")
                python_call_id = str(python_call.get("call_id", ""))
                if python_call_id:
                    evidence[python_call_id] = {
                        "tool": "code_interpreter",
                        "output": json.dumps(python_call, ensure_ascii=False, default=str),
                        "result": python_call,
                        "evidence_class": "derived_calculation",
                    }
                trace.event("python_execution", iteration=iteration, **python_call)
                trace.transcript("tool", tool="code_interpreter", **python_call)
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
                    completion_issues = _completion_issues(
                        question,
                        tool_categories=tool_categories,
                        outstanding_cursors=outstanding_cursors,
                    )
                    if completion_issues and not completeness_forced and iteration < self.max_iterations:
                        completeness_forced = True
                        feedback = "You haven't verified " + "; ".join(completion_issues) + ". Continue."
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
                            "python_calls": python_calls,
                        })
                        continue
                    if completion_issues:
                        final_answer = (
                            "The answer was withheld because the required project scope, population, or "
                            "pagination checks were still incomplete after one correction attempt."
                        )
                        termination_reason = "completeness_rejected"
                        trace.transcript(
                            "gate", gate="completeness", accepted=False, issues=completion_issues
                        )
                        iterations.append({
                            "iteration": iteration,
                            "response_id": response_id,
                            "action": "completeness_rejected",
                            "issues": completion_issues,
                            "python_calls": python_calls,
                        })
                        break

                    grounding_issues = _grounding_issues(
                        question,
                        candidate_answer,
                        evidence,
                        evidence_aliases=evidence_aliases,
                        require_disclaimer=unverified_disclaimer_required,
                    )
                    if grounding_issues:
                        trace.transcript("gate", gate="grounding", accepted=False, issues=grounding_issues)
                        if not grounding_forced and iteration < self.max_iterations:
                            grounding_forced = True
                            input_items.extend([
                                {"role": "assistant", "content": candidate_answer},
                                {
                                    "role": "user",
                                    "content": (
                                        "The proposed answer failed the citation-grounding check: "
                                        + " ".join(grounding_issues)
                                        + " Either call a tool to verify each claim or remove it. Return inline "
                                          "references in the form [ref: call_id]."
                                          " For multiple observations, repeat the tag, for example "
                                          "[ref: call_3] [ref: call_7]. Cite only IDs from direct "
                                          "function-call outputs, their supplied _citation_reference aliases, "
                                          "or automatic reconciliation observations."
                                    ),
                                },
                            ])
                            iterations.append({
                                "iteration": iteration,
                                "response_id": response_id,
                                "action": "grounding_continue",
                                "issues": grounding_issues,
                                "python_calls": python_calls,
                            })
                            continue
                        final_answer = (
                            "The answer was withheld because its factual claims could not be validated against "
                            "the cited tool observations."
                        )
                        termination_reason = "grounding_rejected"
                        iterations.append({
                            "iteration": iteration,
                            "response_id": response_id,
                            "action": "grounding_rejected",
                            "issues": grounding_issues,
                            "python_calls": python_calls,
                        })
                        break

                    final_answer = candidate_answer
                    status = "completed"
                    termination_reason = "completed"
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
                    model_result = _model_tool_result(result)
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

                if (
                    outcome == "ok"
                    and isinstance(result, dict)
                    and reconciliation_required
                    and "reconciliation" not in tool_categories
                    and _needs_reconciliation(name, result)
                ):
                    auto_id = f"auto_reconcile_{call_id}"
                    population_ids = _population_ids(name, arguments, result, self.tools)
                    reconciliation_args = {
                        "populations": [{"label": f"{name}:{call_id}", "object_ids": population_ids}]
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
                    reconciliation["population_ids_available"] = bool(population_ids)
                    reconciliation["mismatch"] = _reconciliation_mismatch(reconciliation)
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
                    if population_ids:
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
                final_answer = (
                    "The answer was stopped because the configured per-answer cost budget was reached before "
                    "the evidence could be finalized."
                )
                trace.event(
                    "cost_budget_exceeded",
                    iteration=iteration,
                    max_answer_cost_usd=self.max_answer_cost_usd,
                )
                break

        limitations = [
            "Semantic scope remains model-directed; deterministic completeness, pagination, reconciliation, and citation-grounding guards check termination."
        ]
        python_status = self.python_workspace.status()
        if python_status["enabled"] and not any(
            item.get("type") == "code_interpreter" for item in api_tools
        ):
            if route["project_data_operation"]:
                limitations.append(
                    "Hosted Python was intentionally not exposed because this project filter/aggregate is routed to read-only SQL."
                )
            else:
                limitations.append(
                    "The optional hosted Python workspace was unavailable for this run: "
                    + str(python_status.get("last_error") or "the configured client does not expose containers")
                )
        if status != "completed":
            final_answer = final_answer or (
                "The model-directed agent did not produce a final answer before its iteration safety limit."
            )
            if termination_reason == "grounding_rejected":
                limitations.append("The final draft failed the citation-grounding check after one forced retry.")
            elif termination_reason == "completeness_rejected":
                limitations.append(
                    "The required scope, population reconciliation, or pagination check remained incomplete "
                    "after one forced correction."
                )
            elif termination_reason == "cost_budget_exceeded":
                limitations.append(
                    f"The run reached the configured ${self.max_answer_cost_usd:.2f} per-answer cost guard."
                )
            else:
                limitations.append(f"The run stopped after {self.max_iterations} model iterations.")
        cost = estimate_run_cost(
            usage_records,
            pricing_overrides=self.pricing_overrides,
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
                "completeness_forced": completeness_forced,
                "grounding_forced": grounding_forced,
                "evidence_aliases": evidence_aliases,
                "cost_budget_usd": self.max_answer_cost_usd or None,
                "cost_budget_exceeded": cost_budget_exceeded,
                "termination_reason": termination_reason,
                "programmatic_tool_calling": self.programmatic_tool_calling,
                "iterations": iterations,
            },
        )

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in {item["name"] for item in self.model_tools.definitions()}:
            return self.model_tools.execute(name, arguments)
        return self.tools.execute(name, arguments)

    def _api_tools(self, question: str | None = None) -> list[dict[str, Any]]:
        project_definitions = copy.deepcopy(self.tools.definitions())
        model_definitions = copy.deepcopy(self.model_tools.definitions())
        python_definition: dict[str, Any] | None = None
        if question is not None:
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
                python_definition = self.python_workspace.tool_definition()
        else:
            python_definition = self.python_workspace.tool_definition()
        native_tools = [python_definition] if python_definition is not None else []
        if not self.programmatic_tool_calling:
            return [*project_definitions, *model_definitions, *native_tools]
        for definition in project_definitions:
            definition["allowed_callers"] = ["direct", "programmatic"]
        for definition in model_definitions:
            definition["allowed_callers"] = ["direct"]
        return [
            *project_definitions,
            *model_definitions,
            *native_tools,
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
            "hosted_python": self.python_workspace.status(),
            "deterministic_semantic_components": [
                "tool_route_filter", "call_deduplication", "automatic_reconciliation",
                "completeness_gate", "pagination_gate", "citation_grounding_gate", "cost_budget_gate",
            ],
            "remaining_code": "Semantic scope and analysis choices remain model-controlled; read-only execution, routing, pagination, reconciliation, citation validation, tracing, and safety limits are enforced in code.",
        }


_REF_RE = re.compile(r"\[ref:\s*([A-Za-z0-9_.:-]+)\s*\]", re.IGNORECASE)
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


def _classify_route(question: str) -> dict[str, Any]:
    normalized = question.casefold()
    data_operation = bool(re.search(
        r"\b(how many|count|total|sum|average|mean|minimum|maximum|min|max|group|per|each|"
        r"filter|where|whose|with|without|join|list all|all objects|all records)\b",
        normalized,
    )) and bool(_PROJECT_TERMS.search(question))
    return {
        "project_data_operation": data_operation,
        "general_compute_path": "sql_only" if data_operation else "adaptive",
        "calculate_exposed": not data_operation,
        "hosted_python_exposed": not data_operation,
        "routing_rules": str(_ROUTING_PATH),
    }


def _tool_cache_key(name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
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
    }
    return mapping.get(name, "other")


def _completion_issues(
    question: str,
    *,
    tool_categories: set[str],
    outstanding_cursors: set[str],
) -> list[str]:
    normalized = question.casefold()
    issues: list[str] = []
    substantive = tool_categories.difference({"reconciliation", "review", "pagination", "schema"})
    project_question = bool(_PROJECT_TERMS.search(question))
    quantitative = bool(re.search(
        r"\b(how many|count|total|sum|average|mean|minimum|maximum|min|max|largest|smallest|longest|shortest)\b",
        normalized,
    ))
    if project_question and not substantive:
        issues.append("any project observation")
    if project_question and quantitative and not ({"sql", "geometry"} & substantive):
        issues.append("the requested project aggregate or complete comparison through SQL/geometry evidence")
    complete_population = quantitative or bool(re.search(r"\b(all|every|complete|list|ranking)\b", normalized))
    if (
        project_question
        and _requires_population_reconciliation(question)
        and "reconciliation" not in tool_categories
    ):
        issues.append(
            "population reconciliation before an exhaustive, ranking, cross-source, or compliance claim"
        )
    if re.search(r"\b(floor|level|storey|story|room|zone|location|located)\b", normalized) and not (
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
    if project_question and ambiguous_scope and "scope" not in tool_categories:
        issues.append("the ambiguous category, location, or property boundary with explore_object_scope")
    if project_question and ambiguous_scope and "scope" in tool_categories and "review" not in tool_categories:
        issues.append("the chosen ambiguous scope with review_scope_and_evidence")
    return list(dict.fromkeys(issues))


def _grounding_issues(
    question: str,
    answer: str,
    evidence: dict[str, dict[str, Any]],
    *,
    evidence_aliases: dict[str, str] | None = None,
    require_disclaimer: bool,
) -> list[str]:
    issues: list[str] = []
    if require_disclaimer and UNVERIFIED_STANDARDS_DISCLAIMER not in answer:
        issues.append(f"The mandatory disclaimer is missing: {UNVERIFIED_STANDARDS_DISCLAIMER}")
    requires_grounding = bool(evidence) or bool(_PROJECT_TERMS.search(question)) or bool(_NUMBER_RE.search(answer))
    if not requires_grounding:
        return issues

    claims = [
        item.strip(" -•\t")
        for item in re.split(r"(?<=[.!?])\s+(?!\[ref:)|\n+", answer, flags=re.IGNORECASE)
        if item.strip(" -•\t")
    ]
    for claim in claims:
        if claim == UNVERIFIED_STANDARDS_DISCLAIMER or claim.endswith(":"):
            continue
        if _markdown_table_scaffolding(claim):
            continue
        refs = _REF_RE.findall(claim)
        claim_text = _REF_RE.sub("", claim).strip()
        if not any(character.isalnum() for character in claim_text):
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
        if _looks_project_claim(question, claim_text) and not any(
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
                f"Claimed value(s) {', '.join(unmatched_numbers)} do not occur in the cited output(s)."
            )
            continue
        if not numbers and not _textual_evidence_overlap(claim_text, combined_output):
            issues.append(f"Claim text does not match the cited output: {claim_text[:180]!r}.")
    return issues


def _looks_project_claim(question: str, claim: str) -> bool:
    return bool(_PROJECT_TERMS.search(claim)) or (
        bool(_PROJECT_TERMS.search(question))
        and not re.search(r"\b(standard|code|regulation|requirement|guidance)\b", claim, re.IGNORECASE)
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
    if name in {"explore_object_scope", "review_scope_and_evidence"}:
        return "model_review"
    return "project"


def _needs_reconciliation(name: str, result: dict[str, Any]) -> bool:
    if name in {
        "reconcile_populations", "calculate", "describe_bim_workspace",
        "explore_object_scope", "review_scope_and_evidence", "research_standards",
    }:
        return False
    return name in {
        "inspect_project", "list_tree_children", "search_records", "get_records", "fetch_more",
        "aggregate_records", "search_ifc", "query_bim_workspace", "analyze_ifc_geometry",
        "rank_ifc_geometry", "analyze_ifc_graph",
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


def _model_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove exact compatibility aliases before placing observations in model context."""
    compact = copy.deepcopy(result)
    if "results" in compact:
        for alias in ("records", "children", "matches"):
            if compact.get(alias) == compact.get("results"):
                compact.pop(alias, None)
        if compact.get("total_matches") == compact.get("total_count"):
            compact.pop("total_matches", None)
        if compact.get("returned") == compact.get("returned_count"):
            compact.pop("returned", None)
    return compact


def _markdown_table_scaffolding(claim: str) -> bool:
    stripped = claim.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell):
        return True
    return not _NUMBER_RE.search(stripped) and "`" not in stripped


def _is_single_clarifying_question(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped.endswith("?") or _REF_RE.search(stripped) or _NUMBER_RE.search(stripped):
        return False
    sentences = [part for part in re.split(r"[.!?]+", stripped) if part.strip()]
    return len(sentences) == 1 and len(stripped) <= 500


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


def _code_interpreter_record(item: Any) -> dict[str, Any]:
    outputs = []
    for output in list(getattr(item, "outputs", []) or []):
        output_type = str(getattr(output, "type", "") or "")
        if output_type == "logs":
            logs = str(getattr(output, "logs", "") or "")
            outputs.append({
                "type": "logs", "characters": len(logs), "preview": logs[:2000], "logs": logs,
            })
        elif output_type == "image":
            outputs.append({"type": "image"})
        else:
            outputs.append({"type": output_type or "unknown"})
    code = str(getattr(item, "code", "") or "")
    return {
        "call_id": str(getattr(item, "id", "") or ""),
        "container_id": str(getattr(item, "container_id", "") or ""),
        "status": str(getattr(item, "status", "") or ""),
        "code": code,
        "code_characters": len(code),
        "outputs": outputs,
    }
