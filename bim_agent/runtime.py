from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_loop import ModelDirectedBimAgent
from .config import Settings
from .models import AnswerReport
from .project_tools import RawProjectTools
from .question_planning import QuestionPlan, QuestionPlanner
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
            model_tool_reasoning_effort=self.settings.model_tool_reasoning_effort,
            finalization_reasoning_effort=self.settings.finalization_reasoning_effort,
            max_iterations=self.settings.max_agent_iterations,
            max_tool_output_chars=self.settings.max_tool_output_chars,
            max_answer_cost_usd=self.settings.max_answer_cost_usd,
            enable_local_python=self.settings.enable_local_python,
            local_python_image=self.settings.local_python_image,
            python_memory_limit=self.settings.python_memory_limit,
            local_python_cpus=self.settings.local_python_cpus,
            local_python_timeout_seconds=self.settings.local_python_timeout_seconds,
            local_python_output_chars=self.settings.local_python_output_chars,
            python_cache_root=self.settings.trace_dir.parent / "local-python-workspaces",
            openai_max_retries=self.settings.openai_max_retries,
            openai_timeout_seconds=self.settings.openai_timeout_seconds,
            pricing_overrides=self.settings.pricing_overrides,
            client=client,
        )
        self.planner = (
            QuestionPlanner(
                client=self.agent.client,
                model=self.settings.planning_model or self.settings.model,
                reasoning_effort=self.settings.planning_reasoning_effort,
                project_tools=self.tools,
            )
            if self.settings.enable_question_planning
            else None
        )

    def ask(self, question: str) -> AnswerReport:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        trace = TraceLog(self.settings.trace_dir, question)
        plan: QuestionPlan | None = None
        if self.planner is not None:
            plan = self.planner.plan(question)
            plan = self.planner.resolve_schema_population(question, plan)
            trace.event(
                "question_plan",
                contract_valid=plan.contract_valid,
                contract_error=plan.contract_error,
                schema_fingerprint=plan.schema_fingerprint,
                route=plan.route,
                interpretation_plan=plan.interpretation_plan,
                normalization_warnings=list(plan.normalization_warnings),
                schema_resolution=plan.schema_resolution_observation,
            )
        return self.agent.run(question, trace, question_plan=plan)

    def inspect(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.settings.data_dir),
            "raw_record_count": len(self.tools.records),
            "sources": self.tools.manifest(),
            "agentic_flow": self.agent.inspect(),
            "question_planning": {
                "enabled": self.planner is not None,
                "model": self.settings.planning_model or self.settings.model,
                "reasoning_effort": self.settings.planning_reasoning_effort,
            },
        }


def report_json(report: AnswerReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
