from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, ModelSettings, RunHooks

from . import prompts
from .models import BimQueryReport, SupervisorReport, VerificationReport
from .guardrails import (
    supervisor_matches_evidence,
    verification_failure_from_evidence,
    verification_matches_evidence,
)
from .tools import (
    count_elements_by_type_and_level,
    count_project_nodes,
    read_evidence,
    resolve_bim_term,
    verify_element_count,
    verify_project_node_count,
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
        name="BIM Query Agent",
        instructions=prompts.QUERY,
        model=worker_model,
        model_settings=_settings("low"),
        tools=[resolve_bim_term, count_project_nodes, count_elements_by_type_and_level],
        output_type=BimQueryReport,
    )
    verifier = Agent(
        name="Verification Agent",
        instructions=prompts.VERIFIER,
        model=worker_model,
        model_settings=_settings("low"),
        tools=[read_evidence, verify_project_node_count, verify_element_count],
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
                "Resolve and answer a supported BIM question using typed project-scoped tools.",
                max_turns=4,
                hooks=hooks,
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
