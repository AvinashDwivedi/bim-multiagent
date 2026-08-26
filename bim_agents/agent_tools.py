from __future__ import annotations

import json

from .claude_runtime import RunContextWrapper, function_tool

from .models import (
    BimRunContext, BimTaskContract, InvestigationAction,
    InvestigationHypothesis, InvestigationObservation,
)
from .knowledge import (
    KnowledgeProposal, save_knowledge_candidate, verify_and_promote_candidate,
)
from .schema_mapping import SchemaMappingProposal
from .tools import (
    BimQueryPlan,
    build_compact_model_profile,
    GeometryQueryPlan,
    PipelineContext,
    define_bim_task as _define_bim_task,
    inspect_project_graph_structure as _inspect_graph,
    get_bim_query_catalog as _get_catalog,
    get_project_knowledge_catalog as _get_project_knowledge,
    inspect_completion_gates as _inspect_completion_gates,
    inspect_queryable_node_types as _inspect_nodes,
    profile_project_classification_hierarchy as _profile_classification,
    profile_project_properties as _profile_properties,
    query_bim as _query_bim,
    query_project_geometry as _query_project_geometry,
    record_investigation_observation as _record_observation,
    register_investigation_hypothesis as _register_hypothesis,
    register_schema_mapping as _register_mapping,
    semantic_search_project_nodes as _search_nodes,
    semantic_search_project_properties as _search_properties,
    semantic_search_property_values as _search_values,
    select_investigation_action as _select_action,
    verify_bim_evidence as _verify,
    activate_project_knowledge_mapping as _activate_project_mapping,
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
def propose_hypothesis(
    ctx: RunContextWrapper[BimRunContext], hypothesis: InvestigationHypothesis
) -> str:
    """Register one falsifiable BIM interpretation before testing it."""
    return _register_hypothesis(_ctx(ctx), hypothesis)


@function_tool
def select_next_action(
    ctx: RunContextWrapper[BimRunContext], action: InvestigationAction
) -> str:
    """Select one bounded action and its expected information gain."""
    return _select_action(_ctx(ctx), action)


@function_tool
def record_observation(
    ctx: RunContextWrapper[BimRunContext], observation: InvestigationObservation
) -> str:
    """Record the observed result before choosing another action."""
    return _record_observation(_ctx(ctx), observation)


@function_tool
def check_completion_gates(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Check scope, identity, classification, units, completeness, conflicts, and replay."""
    return _inspect_completion_gates(_ctx(ctx))


@function_tool
def inspect_query_capabilities(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Inspect trusted contract-backed entities, fields, operations, and measurement semantics."""
    return _get_catalog(_ctx(ctx))


@function_tool
def inspect_project_knowledge(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Inspect semantic mappings and calculation rules for the active client/project only."""
    return _get_project_knowledge(_ctx(ctx))


def _learned_knowledge_payload(ctx: RunContextWrapper[BimRunContext]) -> dict:
    records = []
    for item in ctx.context.learned_knowledge.values():
        mapping = (item.payload or {}).get("proposal") or {}
        records.append({
            "knowledge_id": item.knowledge_id,
            "kind": item.kind,
            "concept": item.concept,
            "aliases": item.aliases,
            "confidence": item.confidence,
            "status": item.status,
            "mapping_id": (
                "learned-" + item.knowledge_id.removeprefix("knowledge-")
                if item.kind == "schema_mapping" and item.status == "promoted" else None
            ),
            "mapping_summary": {
                "entity_name": mapping.get("entity_name"),
                "label": mapping.get("label"),
                "identity_property": mapping.get("identity_property"),
                "fields": {
                    field.get("semantic_name"): field.get("property")
                    for field in mapping.get("fields") or []
                    if field.get("semantic_name") and field.get("property")
                },
                "exact_values": {
                    binding.get("semantic_name"): [
                        match.get("value") for match in binding.get("matches") or []
                    ]
                    for binding in mapping.get("value_bindings") or []
                    if binding.get("semantic_name")
                },
            } if mapping else None,
        })
    return {"knowledge": records, "scope": "active project and compatible schema only"}


@function_tool
def inspect_learned_knowledge(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Inspect compatible project-scoped learned mappings without exposing scope identifiers."""
    return json.dumps(_learned_knowledge_payload(ctx), ensure_ascii=False, sort_keys=True)


@function_tool
def inspect_evidence_capabilities(
    ctx: RunContextWrapper[BimRunContext], concept: str,
) -> str:
    """Inspect trusted query, geometry, project, and learned routes in one bounded catalog call."""
    return json.dumps({
        "requested_concept": concept,
        "query_capabilities": json.loads(_get_catalog(_ctx(ctx))),
        "project_knowledge": json.loads(_get_project_knowledge(_ctx(ctx))),
        "learned_knowledge": _learned_knowledge_payload(ctx),
        "model_profile": build_compact_model_profile(ctx.context).model_dump(mode="json"),
    }, ensure_ascii=False, sort_keys=True)


@function_tool
def propose_learned_knowledge(
    ctx: RunContextWrapper[BimRunContext], proposal: KnowledgeProposal
) -> str:
    """Save project-only reusable semantics; direct answers and unknown evidence are rejected."""
    return save_knowledge_candidate(ctx.context, proposal).model_dump_json()


@function_tool
def promote_learned_knowledge(
    ctx: RunContextWrapper[BimRunContext], knowledge_id: str
) -> str:
    """Promote a candidate only after deterministic answer-evidence replay succeeds."""
    return verify_and_promote_candidate(ctx.context, knowledge_id).model_dump_json()


@function_tool
def submit_geometry_query(
    ctx: RunContextWrapper[BimRunContext], plan: GeometryQueryPlan
) -> str:
    """Calculate a metric from scoped Revit geometry and project interpretation rules."""
    return _query_project_geometry(_ctx(ctx), plan)


@function_tool
def inspect_graph_structure(
    ctx: RunContextWrapper[BimRunContext],
    sample_limit: int = 5,
    include_properties: bool = False,
    focus: str = "",
) -> str:
    """Inspect authorized live Neo4j node labels and relationship patterns before mapping."""
    return _inspect_graph(_ctx(ctx), sample_limit, include_properties, focus)


@function_tool
def inspect_node_inventory(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Inspect authorized labels, counts, representative names, and compact property hints."""
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
def profile_classification_hierarchy(
    ctx: RunContextWrapper[BimRunContext],
    label: str,
    properties: list[str],
    concept: str = "",
    limit: int = 30,
) -> str:
    """Profile exact category/family/type branches and counts for one observed live label."""
    return _profile_classification(_ctx(ctx), label, properties, concept, limit)


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
def activate_project_mapping(
    ctx: RunContextWrapper[BimRunContext], knowledge_key: str,
) -> str:
    """Activate one governed project mapping after deterministic live-schema validation."""
    return _activate_project_mapping(_ctx(ctx), knowledge_key)


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
