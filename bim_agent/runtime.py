from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .builtin_agent import BuiltinBimAgent, tool_policy
from .config import Settings
from .models import AnswerReport
from .project_data import ProjectFiles, resolve_project_files
from .readonly_shell import ReadonlyProjectShell
from .tracing import TraceLog


class BimAgent:
    """BIM analyst exposing only native shell and web-search built-ins."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
        shell: ReadonlyProjectShell | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env(data_dir=data_dir)
        self.project: ProjectFiles = resolve_project_files(self.settings.data_dir)
        self.shell = shell or ReadonlyProjectShell(
            self.project.project_dir,
            bash_path=os.getenv("BIM_BASH_PATH") or None,
            timeout_seconds=max(1.0, float(os.getenv("BIM_SHELL_TIMEOUT_SECONDS", "120"))),
            max_output_chars=self.settings.max_tool_output_chars,
        )
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                max_retries=self.settings.openai_max_retries,
                timeout=self.settings.openai_timeout_seconds,
            )
        self.agent = BuiltinBimAgent(
            project=self.project,
            shell=self.shell,
            model=self.settings.model,
            reasoning_effort=self.settings.reasoning_effort,
            max_iterations=self.settings.max_agent_iterations,
            openai_timeout_seconds=self.settings.openai_timeout_seconds,
            pricing_overrides=self.settings.pricing_overrides,
            client=client,
        )

    def ask(
        self,
        question: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AnswerReport:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        trace = TraceLog(self.settings.trace_dir, question)
        return self.agent.run(
            question.strip(), trace,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.project.project_dir),
            "source_count": len(self.project.paths()),
            "sources": self.project.manifest(),
            "agentic_flow": self.agent.inspect(),
            "tool_policy": tool_policy(),
        }


def report_json(report: AnswerReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
