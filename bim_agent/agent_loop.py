from __future__ import annotations

import json
from typing import Any

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
Answer in the user's language. Give the direct result first, then concise evidence and limitations. Do not expose
private chain-of-thought. Return a normal assistant message only when you judge the answer ready."""


class ModelConfigurationError(RuntimeError):
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
        client: Any | None = None,
    ):
        if client is None:
            try:
                from openai import OpenAI

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

    def run(self, question: str, trace: TraceLog) -> AnswerReport:
        input_items: list[Any] = [{"role": "user", "content": question}]
        iterations: list[dict[str, Any]] = []
        response_ids: list[str] = []
        final_answer = ""
        status = "limited"
        trace.event(
            "model_agent_start",
            model=self.model,
            max_iterations=self.max_iterations,
            tools=[item["name"] for item in self.tools.definitions()],
        )

        for iteration in range(1, self.max_iterations + 1):
            response = self.client.responses.create(
                model=self.model,
                instructions=AGENT_INSTRUCTIONS,
                input=input_items,
                tools=self.tools.definitions(),
                tool_choice="auto",
                parallel_tool_calls=False,
                reasoning={"effort": self.reasoning_effort},
                store=False,
                max_output_tokens=5000,
            )
            response_id = str(getattr(response, "id", ""))
            if response_id:
                response_ids.append(response_id)
            output = list(getattr(response, "output", []) or [])
            calls = [item for item in output if getattr(item, "type", None) == "function_call"]
            trace.event(
                "model_decision",
                iteration=iteration,
                response_id=response_id,
                tool_calls=[getattr(item, "name", "") for item in calls],
            )

            if not calls:
                final_answer = str(getattr(response, "output_text", "") or "").strip()
                if final_answer:
                    status = "completed"
                    iterations.append({
                        "iteration": iteration,
                        "response_id": response_id,
                        "action": "finish",
                    })
                    break
                iterations.append({
                    "iteration": iteration,
                    "response_id": response_id,
                    "action": "empty_response",
                })
                input_items.extend(output)
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
                    result = self.tools.execute(name, arguments)
                    output_text, truncated = _tool_output(result, self.max_tool_output_chars)
                    outcome = "ok"
                except Exception as exc:
                    arguments = {"raw": raw_arguments}
                    output_text = json.dumps(
                        {"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False
                    )
                    truncated = False
                    outcome = "error"
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                })
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
            })

        limitations = [
            "No independent deterministic planner, executor, verifier, or answer checker is present; the model decides tool use and answer completion."
        ]
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
                "tools_available": [item["name"] for item in self.tools.definitions()],
                "iterations": iterations,
            },
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "pattern": "model_directed_tool_loop",
            "model": self.model,
            "max_iterations": self.max_iterations,
            "tools": [
                {"name": item["name"], "description": item["description"]}
                for item in self.tools.definitions()
            ],
            "deterministic_semantic_components": [],
            "remaining_code": "Read-only file parsing, search, aggregation, arithmetic, API transport, tracing, and safety limits.",
        }


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
