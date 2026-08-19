from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, ModelSettings, RunHooks

from . import prompts
from .agent_tools import (
    create_task_contract,
    inspect_graph_structure,
    inspect_node_inventory,
    profile_node_properties,
    register_schema_mapping,
    replay_and_verify,
    search_nodes,
    search_properties,
    search_property_values,
    submit_query_plan,
)
from .models import BimQueryReport, PipelineReport, VerificationReport
from .schema_mapping import SchemaMappingReport


@dataclass(frozen=True)
class BimAgentRegistry:
    supervisor: Agent
    graph_explorer: Agent
    query_planner: Agent
    verifier: Agent


def _settings(effort: str) -> ModelSettings:
    return ModelSettings(reasoning={"effort": effort})


def build_agent_registry(
    model: str = "gpt-5.6-sol",
    *,
    worker_model: str | None = None,
    hooks: RunHooks | None = None,
) -> BimAgentRegistry:
    worker = worker_model or model
    explorer = Agent(
        name="Graph Explorer",
        instructions=prompts.GRAPH_EXPLORER,
        model=worker,
        model_settings=_settings("low"),
        tools=[
            inspect_graph_structure,
            inspect_node_inventory,
            profile_node_properties,
            search_nodes,
            search_properties,
            search_property_values,
            register_schema_mapping,
        ],
        output_type=SchemaMappingReport,
    )
    planner = Agent(
        name="Query Planner",
        instructions=prompts.QUERY_PLANNER,
        model=worker,
        model_settings=_settings("low"),
        tools=[
            explorer.as_tool(
                "explore_graph",
                "Inspect live Neo4j and register the schema mapping needed for this task.",
                max_turns=10,
                hooks=hooks,
            ),
            submit_query_plan,
        ],
        output_type=BimQueryReport,
    )
    verifier = Agent(
        name="Verifier",
        instructions=prompts.VERIFIER,
        model=worker,
        model_settings=_settings("low"),
        tools=[replay_and_verify],
        output_type=VerificationReport,
    )
    supervisor = Agent(
        name="Supervisor",
        instructions=prompts.SUPERVISOR,
        model=model,
        model_settings=_settings("high"),
        tools=[
            create_task_contract,
            inspect_graph_structure,
            inspect_node_inventory,
            profile_node_properties,
            search_nodes,
            search_properties,
            search_property_values,
            register_schema_mapping,
            submit_query_plan,
            replay_and_verify,
        ],
        output_type=PipelineReport,
    )
    return BimAgentRegistry(supervisor, explorer, planner, verifier)
