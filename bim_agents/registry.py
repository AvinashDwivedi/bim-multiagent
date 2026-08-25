from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, ModelSettings, RunHooks

from . import prompts
from .agent_tools import (
    activate_project_mapping,
    inspect_evidence_capabilities,
    inspect_graph_structure,
    inspect_node_inventory,
    inspect_query_capabilities,
    profile_classification_hierarchy,
    profile_node_properties,
    register_schema_mapping,
    search_nodes,
    search_properties,
    search_property_values,
    submit_geometry_query,
    submit_query_plan,
)
from .models import BimTaskContract, EvidenceHandoff, EvidenceWorkstreamResult


@dataclass(frozen=True)
class BimAgentRegistry:
    """Agents are workers; deterministic runtime code is the coordinator."""

    task_architect: Agent
    schema_scout: Agent
    query_executor: Agent


def _settings(effort: str) -> ModelSettings:
    # Every worker owns one isolated mutable context. Runtime scheduling provides
    # parallelism; parallel calls inside one worker would race its artifact ledger.
    return ModelSettings(reasoning={"effort": effort}, parallel_tool_calls=False)


def build_agent_registry(
    model: str = "gpt-5.6-sol",
    *,
    worker_model: str | None = None,
    hooks: RunHooks | None = None,
    project_knowledge: dict | None = None,
) -> BimAgentRegistry:
    """Build role-minimal agents without serializing project knowledge into prompts.

    ``hooks`` and ``project_knowledge`` remain accepted for API compatibility. Live,
    scoped knowledge is fetched on demand by the scout instead of duplicated in all
    system prompts.
    """
    del hooks, project_knowledge
    worker = worker_model or model
    task_architect = Agent(
        name="Task Architect",
        instructions=prompts.TASK_ARCHITECT,
        model=model,
        model_settings=_settings("high"),
        tools=[],
        output_type=BimTaskContract,
    )
    schema_scout = Agent(
        name="BIM Schema Scout",
        instructions=prompts.SCHEMA_SCOUT,
        model=worker,
        model_settings=_settings("high"),
        tools=[
            inspect_evidence_capabilities,
            activate_project_mapping,
            inspect_graph_structure,
            inspect_node_inventory,
            profile_classification_hierarchy,
            profile_node_properties,
            search_nodes,
            search_properties,
            search_property_values,
            register_schema_mapping,
        ],
        output_type=EvidenceHandoff,
    )
    query_executor = Agent(
        name="BIM Query Worker",
        instructions=prompts.QUERY_EXECUTOR,
        model=worker,
        model_settings=_settings("high"),
        tools=[
            inspect_query_capabilities,
            submit_query_plan,
            submit_geometry_query,
        ],
        output_type=EvidenceWorkstreamResult,
    )
    return BimAgentRegistry(task_architect, schema_scout, query_executor)
