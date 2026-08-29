from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_loop import ModelDirectedBimAgent
from .config import Settings
from .models import AnswerReport
from .project_tools import RawProjectTools
from .tracing import TraceLog


class BimAgent:
    """Single model-directed agent with generic read-only project tools."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
    ):
        self.settings = settings or Settings.from_env(data_dir=data_dir)
        self.tools = RawProjectTools(self.settings.data_dir)
        self.agent = ModelDirectedBimAgent(
            tools=self.tools,
            model=self.settings.model,
            reasoning_effort=self.settings.reasoning_effort,
            max_iterations=self.settings.max_agent_iterations,
            max_tool_output_chars=self.settings.max_tool_output_chars,
            max_answer_cost_usd=self.settings.max_answer_cost_usd,
            enable_hosted_python=self.settings.enable_hosted_python,
            python_memory_limit=self.settings.python_memory_limit,
            python_expiry_minutes=self.settings.python_expiry_minutes,
            python_cache_root=self.settings.trace_dir.parent / "python-workspaces",
            openai_max_retries=self.settings.openai_max_retries,
            openai_timeout_seconds=self.settings.openai_timeout_seconds,
            pricing_overrides=self.settings.pricing_overrides,
            client=client,
        )

    def ask(self, question: str) -> AnswerReport:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        return self.agent.run(question, TraceLog(self.settings.trace_dir, question))

    def inspect(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.settings.data_dir),
            "raw_record_count": len(self.tools.records),
            "sources": self.tools.manifest(),
            "agentic_flow": self.agent.inspect(),
        }


def report_json(report: AnswerReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
