from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from .claude_runtime import RunHooks

from .models import BimRunContext


LOGGER_NAME = "bim_pipeline"


def configure_logging(*, verbose: bool = True, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.addHandler(console)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class PipelineEvents:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.event_sink = event_sink
        self.started_at = time.monotonic()
        self._sequence = 0
        self._lock = Lock()

    def stage(self, name: str, **data: Any) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        event = {
            "type": "pipeline_stage",
            "stage": name,
            "sequence": sequence,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 1),
            **data,
        }
        details = " | ".join(
            f"{key}={event[key]}" for key in (
                "workstream_id", "agent", "tool", "status", "category", "error_type",
            )
            if event.get(key) is not None
        )
        self.logger.info(
            "pipeline | %s%s | elapsed=%.1fs",
            name, f" | {details}" if details else "", event["elapsed_seconds"],
        )
        if self.event_sink is not None:
            self.event_sink(event)


class AgentRunHooks(RunHooks[BimRunContext]):
    """Track every agent and tool invocation while enforcing shared run budgets."""

    # Notebook/control calls do not query Neo4j, calculate geometry, or expand external work.
    CONTROL_LOOP_TOOLS = {
        "create_task_contract", "propose_hypothesis", "select_next_action",
        "record_observation", "check_completion_gates",
    }

    def __init__(self, events: PipelineEvents) -> None:
        self.events = events
        self._active: dict[str, list[str]] = {}

    async def on_agent_start(self, context, agent) -> None:
        sequence, invocation = context.context.consume_agent_start(agent.name)
        event_id = f"{context.context.workstream_id}-agent-{sequence}"
        self._active.setdefault(agent.name, []).append(event_id)
        self.events.stage(
            "agent_start", agent=agent.name, event_id=event_id,
            agent_sequence=sequence, invocation=invocation,
            run_id=context.context.run_id,
            workstream_id=context.context.workstream_id,
        )

    async def on_agent_end(self, context, agent, output) -> None:
        active = self._active.get(agent.name) or []
        event_id = active.pop() if active else None
        self.events.stage(
            "agent_end", agent=agent.name, event_id=event_id,
            run_id=context.context.run_id, workstream_id=context.context.workstream_id,
        )

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        count = context.context.consume_budget("llm_calls", "max_llm_calls")
        active = self._active.get(agent.name) or []
        self.events.stage(
            "llm_start", agent=agent.name,
            event_id=active[-1] if active else None, llm_call=count,
            run_id=context.context.run_id, workstream_id=context.context.workstream_id,
        )

    async def on_tool_start(self, context, agent, tool) -> None:
        count = context.context.tool_calls
        if tool.name not in self.CONTROL_LOOP_TOOLS:
            count = context.context.consume_budget("tool_calls", "max_tool_calls")
        active = self._active.get(agent.name) or []
        self.events.stage(
            "tool_start", agent=agent.name, tool=tool.name,
            event_id=active[-1] if active else None, tool_call=count,
            control_loop=tool.name in self.CONTROL_LOOP_TOOLS,
            run_id=context.context.run_id, workstream_id=context.context.workstream_id,
        )

    async def on_tool_end(self, context, agent, tool, result) -> None:
        active = self._active.get(agent.name) or []
        self.events.stage(
            "tool_end", agent=agent.name, tool=tool.name,
            event_id=active[-1] if active else None,
            run_id=context.context.run_id, workstream_id=context.context.workstream_id,
        )
