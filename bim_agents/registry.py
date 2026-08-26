from __future__ import annotations

from dataclasses import dataclass

from .claude_runtime import Agent, ModelSettings, RunHooks

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
    schema_mapping: Agent
    quantity: Agent
    relationship: Agent
    geometry: Agent
    requirements: Agent

    @property
    def schema_scout(self) -> Agent:
        """Compatibility alias for callers using the former two-worker registry."""
        return self.schema_mapping

    @property
    def query_executor(self) -> Agent:
        """Compatibility alias; ordinary queries are quantity-specialist work."""
        return self.quantity

    def execution_specialist(self, role: str) -> Agent:
        specialists = {
            "quantity": self.quantity,
            "relationship": self.relationship,
            "geometry": self.geometry,
            "requirements": self.requirements,
        }
        try:
            return specialists[role]
        except KeyError as exc:
            raise ValueError(f"No execution specialist is registered for {role!r}.") from exc


def _settings(effort: str) -> ModelSettings:
    # Every worker owns one isolated mutable context. Runtime scheduling provides
    # parallelism; parallel calls inside one worker would race its artifact ledger.
    return ModelSettings(effort=effort, parallel_tool_calls=False)


def build_agent_registry(
    model: str = "claude-sonnet-4-6",
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
    schema_mapping = Agent(
        name="BIM Schema Mapping Specialist",
        instructions=prompts.SCHEMA_SPECIALIST,
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
    quantity = Agent(
        name="BIM Quantity Specialist",
        instructions=prompts.QUANTITY_SPECIALIST,
        model=worker,
        model_settings=_settings("medium"),
        tools=[
            inspect_query_capabilities,
            submit_query_plan,
        ],
        output_type=EvidenceWorkstreamResult,
    )
    relationship = Agent(
        name="BIM Relationship Specialist",
        instructions=prompts.RELATIONSHIP_SPECIALIST,
        model=worker,
        model_settings=_settings("high"),
        tools=[inspect_query_capabilities, submit_query_plan],
        output_type=EvidenceWorkstreamResult,
    )
    geometry = Agent(
        name="BIM Geometry Specialist",
        instructions=prompts.GEOMETRY_SPECIALIST,
        model=worker,
        model_settings=_settings("medium"),
        tools=[submit_geometry_query],
        output_type=EvidenceWorkstreamResult,
    )
    requirements = Agent(
        name="BIM Requirements Specialist",
        instructions=prompts.REQUIREMENTS_SPECIALIST,
        model=worker,
        model_settings=_settings("medium"),
        tools=[inspect_query_capabilities, submit_query_plan],
        output_type=EvidenceWorkstreamResult,
    )
    return BimAgentRegistry(
        task_architect, schema_mapping, quantity, relationship, geometry, requirements,
    )
