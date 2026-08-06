from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, ModelSettings, RunHooks

from . import prompts
from .models import BimQueryReport, SupervisorReport, VerificationReport
from .guardrails import (
    query_failure_from_evidence,
    query_matches_evidence,
    supervisor_matches_evidence,
    verification_failure_from_evidence,
    verification_matches_evidence,
)
from .tools import (
    get_bim_query_catalog,
    inspect_project_graph_structure,
    query_bim,
    verify_bim_evidence,
)


@dataclass(frozen=True)
class BimAgentRegistry:
    supervisor: Agent
    query_agent: Agent
    verifier: Agent


def _settings(effort: str) -> ModelSettings:
    return ModelSettings(reasoning={"effort": effort})


def build_agent_registry(
    model: str = "gpt-5.6-sol",
    *,
    worker_model: str | None = None,
    hooks: RunHooks | None = None,
) -> BimAgentRegistry:
    worker_model = worker_model or model

    query_agent = Agent(
        name="BIM Analyst",
        instructions=prompts.QUERY,
        model=worker_model,
        model_settings=_settings("low"),
        tools=[get_bim_query_catalog, inspect_project_graph_structure, query_bim],
        output_type=BimQueryReport,
        output_guardrails=[query_matches_evidence],
    )
    verifier = Agent(
        name="Verification Agent",
        instructions=prompts.VERIFIER,
        model=worker_model,
        model_settings=_settings("low"),
        tools=[verify_bim_evidence],
        output_type=VerificationReport,
        output_guardrails=[verification_matches_evidence],
    )
    supervisor = Agent(
        name="BIM Supervisor",
        instructions=prompts.SUPERVISOR,
        model=model,
        model_settings=_settings("low"),
        tools=[
            query_agent.as_tool(
                "query_bim",
                "Answer a BIM question using general schema-grounded project queries.",
                max_turns=6,
                hooks=hooks,
                failure_error_function=query_failure_from_evidence,
            ),
            verifier.as_tool(
                "verify_bim_result",
                "Independently verify candidate BIM claims.",
                max_turns=3,
                hooks=hooks,
                failure_error_function=verification_failure_from_evidence,
            ),
        ],
        output_type=SupervisorReport,
        output_guardrails=[supervisor_matches_evidence],
    )
    return BimAgentRegistry(supervisor=supervisor, query_agent=query_agent, verifier=verifier)
