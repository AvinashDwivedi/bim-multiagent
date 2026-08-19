from __future__ import annotations

from agents import RunContextWrapper, function_tool

from .models import BimRunContext, BimTaskContract
from .schema_mapping import SchemaMappingProposal
from .tools import (
    BimQueryPlan,
    PipelineContext,
    define_bim_task as _define_bim_task,
    inspect_project_graph_structure as _inspect_graph,
    inspect_queryable_node_types as _inspect_nodes,
    profile_project_properties as _profile_properties,
    query_bim as _query_bim,
    register_schema_mapping as _register_mapping,
    semantic_search_project_nodes as _search_nodes,
    semantic_search_project_properties as _search_properties,
    semantic_search_property_values as _search_values,
    verify_bim_evidence as _verify,
)


def _ctx(wrapper: RunContextWrapper[BimRunContext]) -> PipelineContext:
    return PipelineContext(wrapper.context)


@function_tool
def create_task_contract(
    ctx: RunContextWrapper[BimRunContext], contract: BimTaskContract
) -> str:
    """Create the question's operation, constraints, open questions, and success criteria."""
    return _define_bim_task(_ctx(ctx), contract)


@function_tool
def inspect_graph_structure(
    ctx: RunContextWrapper[BimRunContext],
    sample_limit: int = 5,
    include_properties: bool = True,
    focus: str = "",
) -> str:
    """Inspect authorized live Neo4j node labels and relationship patterns before mapping."""
    return _inspect_graph(_ctx(ctx), sample_limit, include_properties, focus)


@function_tool
def inspect_node_inventory(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Inspect authorized node labels, properties, counts, and representative names."""
    return _inspect_nodes(_ctx(ctx))


@function_tool
def profile_node_properties(
    ctx: RunContextWrapper[BimRunContext],
    label: str,
    properties: list[str],
    sample_limit: int = 5,
) -> str:
    """Profile population, distinctness, and bounded live values for observed properties."""
    return _profile_properties(_ctx(ctx), label, properties, sample_limit)


@function_tool
def search_nodes(
    ctx: RunContextWrapper[BimRunContext],
    query: str,
    label: str,
    name_properties: list[str],
    top_k: int = 5,
) -> str:
    """Semantically rank authorized live nodes for a user concept."""
    return _search_nodes(_ctx(ctx), query, label, name_properties, top_k)


@function_tool
def search_properties(
    ctx: RunContextWrapper[BimRunContext],
    label: str,
    concepts: list[str],
    top_k: int = 5,
) -> str:
    """Semantically rank observed properties for identity, classification, level, or metrics."""
    return _search_properties(_ctx(ctx), label, concepts, top_k)


@function_tool
def search_property_values(
    ctx: RunContextWrapper[BimRunContext],
    label: str,
    property: str,
    requested_concepts: list[str],
    top_k: int = 5,
) -> str:
    """Map user constraints to exact distinct values stored in an observed property."""
    return _search_values(_ctx(ctx), label, property, requested_concepts, top_k)


@function_tool
def register_schema_mapping(
    ctx: RunContextWrapper[BimRunContext], proposal: SchemaMappingProposal
) -> str:
    """Validate and register a mapping supported by live graph, property, and value evidence."""
    return _register_mapping(_ctx(ctx), proposal)


@function_tool
def submit_query_plan(
    ctx: RunContextWrapper[BimRunContext], plan: BimQueryPlan
) -> str:
    """Compile and execute one validated plan through the read-only Cypher Query Handler."""
    return _query_bim(_ctx(ctx), plan)


@function_tool
def replay_and_verify(
    ctx: RunContextWrapper[BimRunContext], evidence_ids: list[str]
) -> str:
    """Recompile and replay all query evidence, then run deterministic semantic checks."""
    return _verify(_ctx(ctx), evidence_ids)
