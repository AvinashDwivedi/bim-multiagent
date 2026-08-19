from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from agents import RunHooks

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

    def stage(self, name: str, **data: Any) -> None:
        event = {
            "type": "pipeline_stage",
            "stage": name,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 1),
            **data,
        }
        self.logger.info("pipeline | %s | elapsed=%.1fs", name, event["elapsed_seconds"])
        if self.event_sink is not None:
            self.event_sink(event)


class AgentRunHooks(RunHooks[BimRunContext]):
    """Track every agent and tool invocation while enforcing shared run budgets."""

    def __init__(self, events: PipelineEvents) -> None:
        self.events = events
        self._active: dict[str, list[str]] = {}

    async def on_agent_start(self, context, agent) -> None:
        sequence, invocation = context.context.consume_agent_start(agent.name)
        event_id = f"agent-{sequence}"
        self._active.setdefault(agent.name, []).append(event_id)
        self.events.stage(
            "agent_start", agent=agent.name, event_id=event_id,
            sequence=sequence, invocation=invocation
        )

    async def on_agent_end(self, context, agent, output) -> None:
        active = self._active.get(agent.name) or []
        event_id = active.pop() if active else None
        self.events.stage("agent_end", agent=agent.name, event_id=event_id)

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        count = context.context.consume_budget("llm_calls", "max_llm_calls")
        active = self._active.get(agent.name) or []
        self.events.stage(
            "llm_start", agent=agent.name,
            event_id=active[-1] if active else None, llm_call=count,
        )

    async def on_tool_start(self, context, agent, tool) -> None:
        count = context.context.consume_budget("tool_calls", "max_tool_calls")
        active = self._active.get(agent.name) or []
        self.events.stage(
            "tool_start", agent=agent.name, tool=tool.name,
            event_id=active[-1] if active else None, tool_call=count,
        )

    async def on_tool_end(self, context, agent, tool, result) -> None:
        active = self._active.get(agent.name) or []
        self.events.stage(
            "tool_end", agent=agent.name, tool=tool.name,
            event_id=active[-1] if active else None,
        )
