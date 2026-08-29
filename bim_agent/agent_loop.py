from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .hosted_python import HostedPythonWorkspace
from .model_tools import ModelAssistedTools
from .models import AnswerReport
from .project_tools import RawProjectTools
from .tracing import TraceLog


AGENT_INSTRUCTIONS = """You are the only BIM analysis agent. There is no fixed planner, executor,
verifier, answer template, supervisor, or fallback agent. Decide for yourself which tools to call, in which
order, and when the investigation is finished.

Ground every project claim in the supplied read-only tools. Begin by inspecting or searching the project when
you need project facts. You may revisit tools, change interpretation, inspect counterexamples, retrieve raw
records, search IFC relationships, and calculate as often as useful. For counts and calculations, use
aggregate_records or calculate rather than estimating mentally. Distinguish hierarchy/definition records from
physical instances using the evidence you inspect; do not assume a category boundary without looking at paths
and properties. Treat the three supplied files as the available project scope and disclose material uncertainty.

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
the container contains the three raw source files and bim_workspace.sqlite, including the full ifc_geometry table.
Print compact evidence, row counts, excluded populations, and reconciliation totals. Its network is disabled.
For every material quantitative conclusion, use reconcile_populations (or an equivalent explicit Python
reconciliation) to reconcile the grand total to subgroup totals and compare the relevant raw-record, external-ID,
IFC-ID, and geometry populations. A mismatch is a reason to investigate, not permission to pick the smallest count.
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
private chain-of-thought. Return a normal assistant message only when you judge the answer ready."""


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
        enable_hosted_python: bool,
        python_memory_limit: str,
        python_expiry_minutes: int,
        python_cache_root: Path,
        openai_max_retries: int = 4,
        openai_timeout_seconds: float = 180.0,
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
        )
        self.programmatic_tool_calling = model.casefold().startswith("gpt-5.6")

    def run(self, question: str, trace: TraceLog) -> AnswerReport:
        input_items: list[Any] = [{"role": "user", "content": question}]
        iterations: list[dict[str, Any]] = []
        response_ids: list[str] = []
        final_answer = ""
        status = "limited"
        api_tools = self._api_tools()
        tool_names = [_tool_label(item) for item in api_tools]
        python_status_at_start = self.python_workspace.status()
        trace.event(
            "model_agent_start",
            model=self.model,
            max_iterations=self.max_iterations,
            tools=tool_names,
            programmatic_tool_calling=self.programmatic_tool_calling,
            hosted_python=python_status_at_start,
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
            python_calls = [
                _code_interpreter_record(item)
                for item in output if getattr(item, "type", None) == "code_interpreter_call"
            ]
            for python_call in python_calls:
                trace.event("python_execution", iteration=iteration, **python_call)
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
                final_answer = str(getattr(response, "output_text", "") or "").strip()
                if final_answer:
                    status = "completed"
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
                name = str(getattr(call, "name", ""))
                call_id = str(getattr(call, "call_id", ""))
                raw_arguments = str(getattr(call, "arguments", "{}") or "{}")
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    result = self._execute_tool(name, arguments)
                    _validate_tool_result(name, result)
                    output_text, truncated = _tool_output(result, self.max_tool_output_chars)
                    outcome = "ok"
                except Exception as exc:
                    arguments = {"raw": raw_arguments}
                    output_text = json.dumps(
                        {"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False
                    )
                    truncated = False
                    outcome = "error"
                call_output: dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
                caller = _caller_payload(getattr(call, "caller", None))
                if caller is not None:
                    call_output["caller"] = caller
                input_items.append(call_output)
                record = {
                    "tool": name,
                    "call_id": call_id,
                    "arguments": arguments,
                    "outcome": outcome,
                    "output_characters": len(output_text),
                    "truncated": truncated,
                }
                call_records.append(record)
                trace.event("tool_execution", iteration=iteration, **record)
            iterations.append({
                "iteration": iteration,
                "response_id": response_id,
                "action": "tool_calls",
                "calls": call_records,
                "python_calls": python_calls,
            })

        limitations = [
            "No independent deterministic planner, executor, verifier, or answer checker is present; the model decides tool use and answer completion."
        ]
        python_status = self.python_workspace.status()
        if python_status["enabled"] and not any(
            item.get("type") == "code_interpreter" for item in api_tools
        ):
            limitations.append(
                "The optional hosted Python workspace was unavailable for this run: "
                + str(python_status.get("last_error") or "the configured client does not expose containers")
            )
        if status != "completed":
            final_answer = final_answer or (
                "The model-directed agent did not produce a final answer before its iteration safety limit."
            )
            limitations.append(
                f"The run stopped after {self.max_iterations} model iterations."
            )
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
            limitations=limitations,
            agent_loop={
                "pattern": "model_directed_tool_loop",
                "model": self.model,
                "finished": status == "completed",
                "iterations_used": len(iterations),
                "max_iterations": self.max_iterations,
                "response_ids": response_ids,
                "tools_available": tool_names,
                "programmatic_tool_calling": self.programmatic_tool_calling,
                "iterations": iterations,
            },
        )

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in {item["name"] for item in self.model_tools.definitions()}:
            return self.model_tools.execute(name, arguments)
        return self.tools.execute(name, arguments)

    def _api_tools(self) -> list[dict[str, Any]]:
        project_definitions = copy.deepcopy(self.tools.definitions())
        model_definitions = copy.deepcopy(self.model_tools.definitions())
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
            "tools": [
                {"name": _tool_label(item), "description": item.get("description", "")}
                for item in api_tools
            ],
            "programmatic_tool_calling": self.programmatic_tool_calling,
            "hosted_python": self.python_workspace.status(),
            "deterministic_semantic_components": [],
            "remaining_code": "Read-only file/IFC parsing, geometry extraction, requested SQL/arithmetic, API transport, tracing, and safety limits. Semantic scope, analysis programs, review, and completion remain model-controlled.",
        }


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
    preview = serialized[: max_chars - 200]
    return json.dumps({
        "truncated": True,
        "original_characters": len(serialized),
        "preview": preview,
        "instruction": "Narrow the query or paginate to retrieve the omitted data.",
    }, ensure_ascii=False), True


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
            outputs.append({"type": "logs", "characters": len(logs), "preview": logs[:2000]})
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
