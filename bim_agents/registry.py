from __future__ import annotations

from dataclasses import dataclass
import json

from agents import Agent, ModelSettings, RunHooks

from . import prompts
from .agent_tools import (
    create_task_contract,
    inspect_graph_structure,
    inspect_query_capabilities,
    inspect_project_knowledge,
    inspect_node_inventory,
    profile_node_properties,
    register_schema_mapping,
    replay_and_verify,
    search_nodes,
    search_properties,
    search_property_values,
    submit_query_plan,
    submit_project_knowledge_query,
    submit_geometry_query,
)
from .models import BimQueryReport, InvestigationCompletion, VerificationReport
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
    project_knowledge: dict | None = None,
) -> BimAgentRegistry:
    worker = worker_model or model
    scoped_knowledge = (
        "\nClient/project-scoped BIM knowledge (authoritative only for the active scope):\n"
        + json.dumps(project_knowledge, ensure_ascii=False, sort_keys=True)
        if project_knowledge else ""
    )
    explorer = Agent(
        name="Graph Explorer",
        instructions=prompts.GRAPH_EXPLORER + scoped_knowledge,
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
        instructions=prompts.QUERY_PLANNER + scoped_knowledge,
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
        instructions=prompts.VERIFIER + scoped_knowledge,
        model=worker,
        model_settings=_settings("low"),
        tools=[replay_and_verify],
        output_type=VerificationReport,
    )
    supervisor = Agent(
        name="Supervisor",
        instructions=prompts.SUPERVISOR + scoped_knowledge,
        model=model,
        model_settings=_settings("high"),
        tools=[
            create_task_contract,
            inspect_query_capabilities,
            inspect_project_knowledge,
            inspect_graph_structure,
            inspect_node_inventory,
            profile_node_properties,
            search_nodes,
            search_properties,
            search_property_values,
            register_schema_mapping,
            submit_query_plan,
            submit_project_knowledge_query,
            submit_geometry_query,
            replay_and_verify,
        ],
        output_type=InvestigationCompletion,
    )
    return BimAgentRegistry(supervisor, explorer, planner, verifier)
