from __future__ import annotations

import time
from typing import Any, Callable

from .errors import AgentRunCancelled
from .models import AnswerReport
from .pricing import estimate_run_cost, response_usage_record
from .project_data import ProjectFiles
from .readonly_shell import ReadonlyProjectShell
from .terminal_ui import TerminalStepLogger
from .tracing import TraceLog


# The Claude reference exposes Read/Grep/Glob/Bash and WebSearch/WebFetch.
# OpenAI provides the same capability surface through two Responses built-ins.
PROJECT_EVIDENCE_TOOLS = ["shell"]
EXTERNAL_EVIDENCE_TOOLS = ["web_search"]
BUILTIN_TOOL_NAMES = [*PROJECT_EVIDENCE_TOOLS, *EXTERNAL_EVIDENCE_TOOLS]
REFERENCE_TOOL_EQUIVALENTS = {
    "shell": ["Read", "Grep", "Glob", "Bash"],
    "web_search": ["WebSearch", "WebFetch"],
}


def builtin_tool_definitions() -> list[dict[str, Any]]:
    return [
        {"type": "shell", "environment": {"type": "local"}},
        {"type": "web_search"},
    ]


def tool_policy() -> dict[str, Any]:
    return {
        "permission_mode": "dontAsk",
        "intended_use": "read_only_analysis",
        "shell_execution": "native_bash",
        "write_restriction": "agent_instruction_and_no_dedicated_write_tools",
        "dedicated_write_tools_enabled": False,
        "built_in": list(BUILTIN_TOOL_NAMES),
        "custom": [],
        "mcp_servers": [],
        "project_evidence": list(PROJECT_EVIDENCE_TOOLS),
        "external_evidence": list(EXTERNAL_EVIDENCE_TOOLS),
        "reference_tool_equivalents": dict(REFERENCE_TOOL_EQUIVALENTS),
    }


def build_system_prompt(project: ProjectFiles) -> str:
    return f"""You are a BIM (Building Information Modeling) data analyst.

The project folder is the shell's current working directory. Use relative paths only. The project evidence is:
`{project.ifc.name}`

### File handling

Do not assume the file format from the `.ifc` extension. Inspect the header first:

* `SQLite format 3` → Autodesk property database. Use Python `sqlite3` in read-only mode. Discover the schema and query the relevant entity/attribute/value data. Resolve type inheritance and parent/child relationships when relevant. Preserve raw values, data types, units, and display precision.
* `ISO-10303-21` → STEP IFC model. Use IfcOpenShell for entities, properties, relationships, placements, quantities, and geometry. If unavailable, inspect the STEP data conservatively and state the limitation.

Use the shell to inspect, search, parse, cross-reference, and aggregate project data. Use read-only commands only. Never modify files, install/download anything, or access paths outside the project directory.

### Analysis rules

* Base project-specific answers only on available project evidence.
* Cross-reference Element IDs and GlobalIds when needed.
* Exclude internal/non-physical entities only when the question's scope requires physical or view-visible objects.
* Distinguish missing/unknown data from zero.
* State quantities, units, scope, and relevant limitations precisely.
* If the available project evidence cannot establish the answer, say so. Do not guess.

### External evidence

Use `web_search` only when external information is required, such as standards, regulations, product references, or current public information. Cite URLs and clearly separate external evidence from project evidence.

Never use external sources as a substitute for project facts or for ordinary project quantity questions. Claim compliance only when both the applicable requirement and sufficient project evidence are available.

### Response

Answer the user's question directly and concisely. Do not include BIM object details unless requested or necessary to support the answer. Do not expose tools, commands, or agent steps.
"""


class BuiltinBimAgent:
    """Responses API loop with no custom functions, MCP, skills, or subagents."""

    def __init__(
        self,
        *,
        project: ProjectFiles,
        shell: ReadonlyProjectShell,
        model: str,
        reasoning_effort: str,
        max_iterations: int,
        openai_timeout_seconds: float,
        pricing_overrides: dict[str, dict[str, float]] | None,
        client: Any,
    ) -> None:
        self.project = project
        self.shell = shell
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_iterations = max(1, max_iterations)
        self.openai_timeout_seconds = openai_timeout_seconds
        self.pricing_overrides = pricing_overrides or {}
        self.client = client

    def run(
        self,
        question: str,
        trace: TraceLog,
        *,
        should_cancel: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AnswerReport:
        started_at = time.monotonic()
        terminal = TerminalStepLogger(trace.session_id)
        terminal.start(
            question=question,
            model=self.model,
            tools=list(BUILTIN_TOOL_NAMES),
            trace_path=str(trace.path),
        )
        input_items: list[Any] = [{"role": "user", "content": question}]
        response_ids: list[str] = []
        usage_records: list[dict[str, Any]] = []
        iterations: list[dict[str, Any]] = []
        limitations: list[str] = []
        final_answer = ""
        termination_reason = "iteration_limit"
        shell_successes = 0
        shell_errors = 0

        for iteration in range(1, self.max_iterations + 1):
            if should_cancel is not None and should_cancel():
                raise AgentRunCancelled("The BIM request was cancelled by its caller.")

            terminal.iteration(iteration, self.max_iterations)
            model_started = time.monotonic()
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=build_system_prompt(self.project),
                    input=input_items,
                    tools=builtin_tool_definitions(),
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    reasoning={"effort": self.reasoning_effort},
                    store=False,
                    include=["reasoning.encrypted_content", "web_search_call.action.sources"],
                    max_output_tokens=5000,
                    max_tool_calls=8,
                    timeout=self.openai_timeout_seconds,
                )
            except Exception as exc:
                terminal.error(stage="model request", error=exc)
                trace.event("builtin_agent_error", iteration=iteration, error=str(exc))
                raise
            model_elapsed = time.monotonic() - model_started
            response_id = str(_value(response, "id") or "")
            if response_id:
                response_ids.append(response_id)
            output = list(_value(response, "output") or [])
            shell_calls = [item for item in output if _value(item, "type") == "shell_call"]
            web_calls = [item for item in output if _value(item, "type") == "web_search_call"]
            candidate = str(_value(response, "output_text") or "").strip()
            terminal.model_response(
                response_id=response_id,
                elapsed=model_elapsed,
                tools=["shell"] * len(shell_calls) + ["web_search"] * len(web_calls),
                has_answer=bool(candidate),
            )
            usage_records.append(response_usage_record(
                response,
                configured_model=self.model,
                purpose="answering",
                excluded_tool_fees=["web_search"] if web_calls else [],
            ))

            call_records: list[dict[str, Any]] = []
            shell_outputs: list[dict[str, Any]] = []
            for call in shell_calls:
                action = _value(call, "action")
                commands = [str(command) for command in (_value(action, "commands") or [])]
                shell_status = self.shell.status()
                backend = str(shell_status.get("mode") or "test")
                for command in commands:
                    terminal.tool_start(name="shell", command=command, backend=backend)
                tool_started = time.monotonic()
                results, max_output_length = self.shell.run_action(
                    action, should_cancel=should_cancel
                )
                tool_elapsed = time.monotonic() - tool_started
                shell_outputs.append({
                    "type": "shell_call_output",
                    "call_id": str(_value(call, "call_id") or ""),
                    "max_output_length": max_output_length,
                    "output": results,
                })
                failed = any(
                    _value(_value(item, "outcome"), "type") != "exit"
                    or int(_value(_value(item, "outcome"), "exit_code") or 0) != 0
                    for item in results
                )
                output_characters = sum(
                    len(str(_value(item, "stdout") or ""))
                    + len(str(_value(item, "stderr") or ""))
                    for item in results
                )
                for result in results:
                    outcome = _value(result, "outcome")
                    result_failed = (
                        _value(outcome, "type") != "exit"
                        or int(_value(outcome, "exit_code") or 0) != 0
                    )
                    characters = (
                        len(str(_value(result, "stdout") or ""))
                        + len(str(_value(result, "stderr") or ""))
                    )
                    preview = str(
                        _value(result, "stderr") or _value(result, "stdout") or ""
                    )
                    terminal.tool_result(
                        name="shell",
                        outcome="error" if result_failed else "success",
                        elapsed=tool_elapsed / max(1, len(results)),
                        characters=characters,
                        preview=preview,
                    )
                    if result_failed:
                        shell_errors += 1
                    else:
                        shell_successes += 1
                call_records.append({
                    "tool": "shell",
                    "call_id": str(_value(call, "call_id") or ""),
                    "outcome": "error" if failed else "success",
                    "output_characters": output_characters,
                    "truncated": output_characters >= max_output_length,
                    "commands": commands,
                    "backend": getattr(self.shell, "last_backend", backend),
                })
                trace.transcript(
                    "tool", tool="shell", call_id=str(_value(call, "call_id") or ""),
                    action=action, output=results,
                )
            for call in web_calls:
                web_status = str(_value(call, "status") or "completed")
                web_call_id = str(_value(call, "id") or "")
                terminal.web_search(status=web_status, call_id=web_call_id)
                call_records.append({
                    "tool": "web_search",
                    "call_id": web_call_id,
                    "outcome": web_status,
                    "output_characters": 0,
                    "truncated": False,
                })
                trace.transcript("tool", tool="web_search", output=_dump(call))

            iterations.append({
                "iteration": iteration,
                "response_id": response_id,
                "action": "tool_calls" if call_records else "finish",
                "calls": call_records,
                "candidate_answer": candidate,
            })
            trace.transcript("assistant", response_id=response_id, output=_dump(output))
            trace.event(
                "builtin_agent_iteration", iteration=iteration, response_id=response_id,
                tools=[record["tool"] for record in call_records], has_answer=bool(candidate),
            )

            cost = estimate_run_cost(usage_records, pricing_overrides=self.pricing_overrides)
            if progress_callback is not None:
                progress_callback({"iteration": iteration, "phase": "response_complete", "cost": cost})

            if shell_calls:
                input_items.extend(output)
                input_items.extend(shell_outputs)
                continue
            if candidate:
                final_answer = candidate
                termination_reason = "completed"
                break

            input_items.extend(output)
            input_items.append({
                "role": "developer",
                "content": "Return the final answer now, or state that the available evidence is insufficient.",
            })

        if not final_answer:
            final_answer = "The agent did not return an answer within the configured turn limit."
            limitations.append(final_answer)

        if final_answer and shell_errors and not shell_successes:
            termination_reason = "tool_unavailable"
            limitations.append(
                "All project-evidence shell commands failed, so the answer could not be verified from the project files."
            )

        completed = termination_reason == "completed"
        cost = estimate_run_cost(usage_records, pricing_overrides=self.pricing_overrides)
        status = "completed" if completed else "limited"
        trace.transcript("cost", **cost)
        trace.transcript("final", status=status, answer=final_answer)
        terminal.finish(
            status=status,
            iterations=len(iterations),
            elapsed=time.monotonic() - started_at,
            cost=cost.get("estimated_cost_usd"),
            answer=final_answer,
        )
        return AnswerReport(
            answer=final_answer,
            status=status,
            sources=self.project.manifest(),
            trace_path=str(trace.path),
            cost=cost,
            limitations=limitations,
            status_dimensions={
                "evidence": "agent_completed" if completed else "insufficient_evidence",
                "ambiguity": "not_separately_evaluated",
                "reconciliation": "not_separately_evaluated",
            },
            agent_loop={
                "pattern": "responses_builtin_tools",
                "model": self.model,
                "finished": completed,
                "iterations_used": len(iterations),
                "max_iterations": self.max_iterations,
                "response_ids": response_ids,
                "trace_session_id": trace.session_id,
                "tools_available": list(BUILTIN_TOOL_NAMES),
                "custom_tools": [],
                "mcp_servers": [],
                "termination_reason": termination_reason,
                "iterations": iterations,
            },
        )

    def inspect(self) -> dict[str, Any]:
        return {"tool_policy": tool_policy(), "shell": self.shell.status()}


def _value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _dump(value: Any) -> Any:
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    return model_dump(mode="json") if callable(model_dump) else value
