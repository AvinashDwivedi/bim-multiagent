from __future__ import annotations

import logging
import inspect
import time
from pathlib import Path
from typing import Any, Callable

from agents import RunHooks

from .models import BimRunContext


LOGGER_NAME = "bim_agents"


def configure_logging(*, verbose: bool = True, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.addHandler(console)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class BimRunHooks(RunHooks[BimRunContext]):
    """Visible lifecycle logging plus global run budgets shared by all nested agents."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.started_at = time.monotonic()
        self.event_sink = event_sink
        self._active_agent_events: dict[str, list[str]] = {}

    def _elapsed(self) -> float:
        return time.monotonic() - self.started_at

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        result = self.event_sink(event)
        if inspect.isawaitable(result):
            await result

    async def on_agent_start(self, context, agent) -> None:
        count = context.context.consume_budget("agent_starts", "max_agent_starts")
        event_id = f"agent-{count}"
        self._active_agent_events.setdefault(agent.name, []).append(event_id)
        self.logger.info("agent.start | %s | agent_start=%d | elapsed=%.1fs", agent.name, count, self._elapsed())
        await self._emit({
            "type": "agent_start",
            "id": event_id,
            "sequence": count,
            "agent": agent.name,
            "elapsed_seconds": round(self._elapsed(), 1),
        })

    async def on_agent_end(self, context, agent, output: Any) -> None:
        self.logger.info("agent.end   | %s | elapsed=%.1fs", agent.name, self._elapsed())
        active = self._active_agent_events.get(agent.name) or []
        event_id = active.pop() if active else None
        await self._emit({
            "type": "agent_end",
            "id": event_id,
            "agent": agent.name,
            "elapsed_seconds": round(self._elapsed(), 1),
        })

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        count = context.context.consume_budget("llm_calls", "max_llm_calls")
        self.logger.info("llm.start   | %s | llm_call=%d | elapsed=%.1fs", agent.name, count, self._elapsed())

    async def on_llm_end(self, context, agent, response) -> None:
        usage = getattr(response, "usage", None)
        self.logger.info("llm.end     | %s | usage=%s | elapsed=%.1fs", agent.name, usage or "n/a", self._elapsed())

    async def on_tool_start(self, context, agent, tool) -> None:
        count = context.context.consume_budget("tool_calls", "max_tool_calls")
        self.logger.info("tool.start  | %s | %s | tool_call=%d | elapsed=%.1fs", agent.name, tool.name, count, self._elapsed())
        await self._emit({
            "type": "tool_start",
            "agent": agent.name,
            "tool": tool.name,
            "tool_call": count,
            "elapsed_seconds": round(self._elapsed(), 1),
        })

    async def on_tool_end(self, context, agent, tool, result: object) -> None:
        size = len(str(result))
        self.logger.info("tool.end    | %s | %s | result_chars=%d | elapsed=%.1fs", agent.name, tool.name, size, self._elapsed())
        await self._emit({
            "type": "tool_end",
            "agent": agent.name,
            "tool": tool.name,
            "elapsed_seconds": round(self._elapsed(), 1),
        })

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self.logger.info("handoff     | %s -> %s | elapsed=%.1fs", from_agent.name, to_agent.name, self._elapsed())
