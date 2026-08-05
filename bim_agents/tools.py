from __future__ import annotations

import json
import re
import unicodedata
from typing import Any
from uuid import uuid4

from agents import RunContextWrapper, function_tool

from .models import BimRunContext, Evidence


_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize(value: str | None) -> str:
    text = "".join(ch for ch in (value or "") if unicodedata.category(ch) != "Cf")
    return " ".join(text.casefold().strip().split())


def _level_number(value: str) -> int | None:
    normalized = _normalize(value)
    for word, number in _ORDINALS.items():
        if re.search(rf"\b{word}\b", normalized):
            return number
    match = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)?\b", normalized)
    return int(match.group(1)) if match else None


def _level_matches(
    requested: str,
    stored: str | None,
    level_aliases: dict[str, list[str]] | None = None,
) -> bool:
    requested_norm = _normalize(requested)
    stored_norm = _normalize(stored)
    if not stored_norm:
        return False
    if requested_norm == stored_norm:
        return True
    for aliases in (level_aliases or {}).values():
        normalized_aliases = {_normalize(alias) for alias in aliases}
        if requested_norm in normalized_aliases and stored_norm in normalized_aliases:
            return True
    requested_number = _level_number(requested_norm)
    stored_number = _level_number(stored_norm)
    return requested_number is not None and stored_number == requested_number


def _resolve_canonical_type(ctx: BimRunContext, term: str) -> tuple[str, dict[str, Any]]:
    ontology = ctx.bim.ontology
    space_function = ontology.resolve_space_function(term)
    ifc_classes = ontology.resolve_element_classes(term) or []
    canonical_type = space_function or _normalize(term).removesuffix("s")
    resolution = {
        "original_term": term,
        "canonical_type": canonical_type,
        "space_function": space_function,
        "ifc_classes": ifc_classes,
        "retrieval_terms": ontology.retrieval_terms_for(term),
        "absence_notes": ontology.absence_notes_for_query(term),
    }
    return canonical_type, resolution


def _count_by_type_and_level(ctx: BimRunContext, element_term: str, level: str) -> dict[str, Any]:
    capability_name = "count_elements_by_type_and_level"
    capability = ctx.graph_contract.capability(capability_name)
    if not all((capability.node_type, capability.identity_property, capability.type_property, capability.level_property)):
        raise RuntimeError("The graph contract has an incomplete element-count capability.")
    node = ctx.graph_contract.node(capability.node_type)
    canonical_type, resolution = _resolve_canonical_type(ctx, element_term)
    label = node.label
    source_property = capability.source_property
    type_property = capability.type_property
    level_property = capability.level_property
    identity_property = capability.identity_property
    rows = ctx.bim.query(
        f"MATCH (b:`{label}`) "
        f"WHERE b.`{source_property}` IN $allowed_sources "
        f"AND toLower(coalesce(b.`{type_property}`, '')) = $canonical_type "
        f"RETURN b.`{level_property}` AS level, "
        f"count(DISTINCT b.`{identity_property}`) AS count, "
        "collect(DISTINCT b.function_source_term)[0..20] AS function_source_terms, "
        "collect(DISTINCT b.function_confidence)[0..10] AS function_confidences, "
        "collect(DISTINCT b.name)[0..10] AS sample_names ORDER BY level",
        {
            "allowed_sources": ctx.scope.allowed_sources,
            "canonical_type": canonical_type.casefold(),
        },
    )
    project_level_rows = ctx.bim.query(
        f"MATCH (b:`{label}`) WHERE b.`{source_property}` IN $allowed_sources "
        f"AND b.`{level_property}` IS NOT NULL "
        f"RETURN DISTINCT b.`{level_property}` AS level ORDER BY level",
        {"allowed_sources": ctx.scope.allowed_sources},
    )
    level_aliases = ctx.graph_contract.level_aliases
    matches = [
        row for row in rows
        if _level_matches(level, row.get("level"), level_aliases)
    ]
    project_level_matches = [
        row["level"] for row in project_level_rows
        if _level_matches(level, row.get("level"), level_aliases)
    ]
    count = sum(int(row.get("count") or 0) for row in matches)
    available_levels = [str(row["level"]) for row in rows if row.get("level") is not None]
    classification_issues = []
    for row in rows:
        for source_term in row.get("function_source_terms") or []:
            resolved_source = ctx.bim.ontology.resolve_space_function(str(source_term))
            if resolved_source != canonical_type:
                classification_issues.append(
                    f"Sample {row.get('sample_names') or ['unnamed element']} is classified as "
                    f"{canonical_type!r} from source term {source_term!r}, but the current "
                    "ontology does not resolve that source term to the same concept."
                )
    return {
        "element_term": element_term,
        "requested_level": level,
        "ontology_resolution": resolution,
        "matched_levels": [row["level"] for row in matches],
        "count": count,
        "identity": f"DISTINCT {label}.{identity_property}",
        "graph_contract_version": ctx.graph_contract.version,
        "capability": capability_name,
        "graph_mapping": {
            "node_label": label,
            "identity_property": identity_property,
            "source_property": source_property,
            "type_property": type_property,
            "level_property": level_property,
            "relationships": capability.relationships,
        },
        "available_levels_for_type": available_levels,
        "project_matching_levels": project_level_matches,
        "level_found": bool(project_level_matches),
        "classification_issues": classification_issues,
        "claim_text": f"The BIM contains {count} {element_term} on {level}.",
        "unit": element_term,
    }


def _count_project_nodes(ctx: BimRunContext) -> dict[str, Any]:
    capability_name = "count_project_nodes"
    capability = ctx.graph_contract.capability(capability_name)
    path = ctx.graph_contract.authorization_path
    client = ctx.graph_contract.node(path.start_node)
    project = ctx.graph_contract.node("project")
    hub = ctx.graph_contract.node("bim_hub")
    if not client.identity_property or not project.identity_property:
        raise RuntimeError("The graph contract is missing hierarchy identity properties.")
    client_project = ctx.graph_contract.relationship_types[path.relationships[0]]
    project_hub = ctx.graph_contract.relationship_types[path.relationships[1]]
    rows = ctx.bim.query(
        f"MATCH (c:`{client.label}` {{`{client.identity_property}`: $client_id}}) "
        f"-[:`{client_project.name}`]->"
        f"(p:`{project.label}` {{`{project.identity_property}`: $project_id}}) "
        f"OPTIONAL MATCH (p)-[:`{project_hub.name}`]->(h:`{hub.label}`) "
        "WITH collect(DISTINCT c) + collect(DISTINCT p) + collect(DISTINCT h) AS hierarchy_nodes "
        f"MATCH (n) WHERE n.`{capability.source_property}` IN $allowed_sources "
        "WITH hierarchy_nodes, collect(DISTINCT n) AS source_nodes "
        "UNWIND hierarchy_nodes + source_nodes AS scoped_node "
        "RETURN count(DISTINCT scoped_node) AS total_nodes, "
        "size(source_nodes) AS source_scoped_nodes, "
        "size(hierarchy_nodes) AS hierarchy_nodes",
        {
            "client_id": ctx.scope.client_id,
            "project_id": ctx.scope.project_id,
            "allowed_sources": ctx.scope.allowed_sources,
        },
    )
    counts = rows[0] if rows else {}
    total = int(counts.get("total_nodes") or 0)
    return {
        "count": total,
        "source_scoped_nodes": int(counts.get("source_scoped_nodes") or 0),
        "hierarchy_nodes": int(counts.get("hierarchy_nodes") or 0),
        "element_term": "project-scoped graph nodes",
        "requested_level": "the authorized project",
        "level_found": True,
        "identity": "DISTINCT Neo4j nodes in the authorized project scope",
        "graph_contract_version": ctx.graph_contract.version,
        "capability": capability_name,
        "claim_text": f"The authorized project graph contains {total} nodes.",
        "unit": "nodes",
        "classification_issues": [],
    }


@function_tool
def resolve_bim_term(ctx: RunContextWrapper[BimRunContext], term: str) -> str:
    """Resolve one term through the composed global/client/project BIM ontology."""
    canonical_type, resolution = _resolve_canonical_type(ctx.context, term)
    resolution["canonical_type"] = canonical_type
    evidence_id = f"ontology-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="ontology",
        summary=f"Ontology resolution for {term!r}",
        payload=_json(resolution),
    ))
    return _json({"evidence_id": evidence_id, **resolution})


@function_tool
def count_elements_by_type_and_level(
    ctx: RunContextWrapper[BimRunContext], element_term: str, level: str
) -> str:
    """Count distinct project-scoped BIM elements of an ontology-resolved type on a requested level."""
    result = _count_by_type_and_level(ctx.context, element_term, level)
    evidence_id = f"query-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="query",
        summary=f"Distinct {element_term!r} count on level {level!r}",
        payload=_json(result),
    ))
    return _json({"evidence_id": evidence_id, **result})


@function_tool
def count_project_nodes(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Count all distinct Neo4j nodes in the authorized project, including hierarchy nodes."""
    result = _count_project_nodes(ctx.context)
    evidence_id = f"query-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="query",
        summary="Total distinct Neo4j node count for the authorized project",
        payload=_json(result),
    ))
    return _json({"evidence_id": evidence_id, **result})


@function_tool
def verify_element_count(
    ctx: RunContextWrapper[BimRunContext],
    element_term: str,
    level: str,
    expected_count: int,
) -> str:
    """Independently rerun a scoped distinct count and compare it with a candidate count."""
    result = _count_by_type_and_level(ctx.context, element_term, level)
    result["expected_count"] = expected_count
    result["verified"] = result["level_found"] and result["count"] == expected_count
    evidence_id = f"verification-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="verification",
        summary=f"Independent verification of {element_term!r} count on {level!r}",
        payload=_json(result),
    ))
    return _json({"evidence_id": evidence_id, **result})


@function_tool
def verify_project_node_count(
    ctx: RunContextWrapper[BimRunContext], expected_count: int
) -> str:
    """Independently rerun the authorized project-node count and compare it with a candidate."""
    result = _count_project_nodes(ctx.context)
    result["expected_count"] = expected_count
    result["verified"] = result["count"] == expected_count
    evidence_id = f"verification-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="verification",
        summary="Independent verification of the authorized project-node count",
        payload=_json(result),
    ))
    return _json({"evidence_id": evidence_id, **result})


@function_tool
def read_evidence(ctx: RunContextWrapper[BimRunContext], evidence_ids: list[str]) -> str:
    """Read selected evidence records from this supervised run."""
    found = [ctx.context.evidence[item].model_dump() for item in evidence_ids if item in ctx.context.evidence]
    missing = [item for item in evidence_ids if item not in ctx.context.evidence]
    return _json({"evidence": found, "missing": missing})
