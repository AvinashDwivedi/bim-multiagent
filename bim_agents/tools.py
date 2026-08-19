from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, Field

from .graph_contract import QueryEntity, QueryField
from .cypher_handler import CypherQueryHandler
from .models import BimRunContext, BimTaskContract, Evidence, RunArtifact
from .schema_mapping import RegisteredSchemaMapping, SchemaMappingProposal


Operation = Literal[
    "count", "list", "group_count", "group_summary", "distinct",
    "sum", "average", "minimum", "maximum"
]


@dataclass(frozen=True)
class PipelineContext:
    """Small compatibility wrapper used by deterministic pipeline operations."""

    context: BimRunContext
FilterOperator = Literal[
    "equals", "not_equals", "contains", "starts_with", "in",
    "greater_than", "greater_or_equal", "less_than", "less_or_equal",
    "exists", "is_missing",
]


class BimFilter(BaseModel):
    field: str
    operator: FilterOperator
    value: str = ""
    values: list[str] = Field(default_factory=list)


class BimQueryPlan(BaseModel):
    entity: str
    mapping_id: str = ""
    operation: Operation
    distinct_by: Literal["identity"] = Field(
        default="identity",
        description="Deduplicate counts by the registered stable entity identity.",
    )
    filters: list[BimFilter] = Field(default_factory=list)
    select: list[str] = Field(default_factory=list)
    metric: str = ""
    group_by: str = ""
    sort_by: str = ""
    sort_order: Literal["ascending", "descending"] = "ascending"
    limit: int = Field(default=10, ge=1, le=20)
    include_details: bool = Field(
        default=False,
        description="Include bounded technical record details when supported.",
    )
    include_in_answer: bool = Field(
        default=True,
        description="False only for a supporting exploration query whose claim should not be rendered.",
    )


def define_bim_task(
    ctx: PipelineContext, contract: BimTaskContract
) -> str:
    """Register the supervisor's goal, constraints, open questions, and definition of done."""
    if ctx.context.task_contract is not None:
        raise ValueError("The BIM task contract has already been defined for this run.")
    if not contract.goal.strip() or not contract.entity_concept.strip():
        raise ValueError("The task goal and entity concept cannot be blank.")
    ctx.context.task_contract = contract
    artifact = RunArtifact(
        artifact_id="task-contract",
        kind="task_contract",
        producer="Pipeline",
        summary=contract.goal,
        payload=contract.model_dump(mode="json"),
    )
    ctx.context.add_artifact(artifact)
    return artifact.model_dump_json()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize(value: str | None) -> str:
    text = "".join(ch for ch in (value or "") if unicodedata.category(ch) != "Cf")
    return " ".join(text.casefold().strip().split())


def _level_number(value: str) -> int | None:
    normalized = _normalize(value)
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

    def contains_phrase(text: str, phrase: str) -> bool:
        if not phrase:
            return False
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    for aliases in (level_aliases or {}).values():
        normalized_aliases = {_normalize(alias) for alias in aliases}
        requested_in_group = any(
            alias == requested_norm or contains_phrase(requested_norm, alias)
            for alias in normalized_aliases
        )
        stored_in_group = any(
            alias == stored_norm or contains_phrase(stored_norm, alias)
            for alias in normalized_aliases
        )
        if requested_in_group and stored_in_group:
            return True
    requested_number = _level_number(requested_norm)
    stored_number = _level_number(stored_norm)
    return requested_number is not None and stored_number == requested_number


def _resolve_canonical_type(ctx: BimRunContext, term: str) -> tuple[str, dict[str, Any]]:
    ontology = ctx.bim.ontology
    space_function = ontology.resolve_space_function(term)
    ifc_classes = ontology.resolve_element_classes(term) or []
    canonical_type = space_function or _normalize(term).removesuffix("s")
    return canonical_type, {
        "original_term": term,
        "canonical_type": canonical_type,
        "space_function": space_function,
        "ifc_classes": ifc_classes,
        "retrieval_terms": ontology.retrieval_terms_for(term),
        "absence_notes": ontology.absence_notes_for_query(term),
    }


def _catalog(ctx: BimRunContext) -> dict[str, Any]:
    common_operations = [
        "count", "list", "group_count", "group_summary", "distinct",
        "sum", "average", "minimum", "maximum"
    ]
    entities: dict[str, Any] = {}
    for name, entity in ctx.graph_contract.query_entities.items():
        entities[name] = {
            "description": entity.description,
            "operations": ["count"] if entity.kind == "project_graph" else common_operations,
            "default_select": entity.default_select,
            "fields": {
                field_name: {
                    "type": field.data_type,
                    "description": field.description,
                    "unit": field.unit,
                }
                for field_name, field in entity.fields.items()
            },
        }
    return {
        "entities": entities,
        "filter_operators": [
            "equals", "not_equals", "contains", "starts_with", "in",
            "greater_than", "greater_or_equal", "less_than", "less_or_equal",
            "exists", "is_missing",
        ],
        "rules": {
            "group_count_and_distinct": "set group_by",
            "group_summary": "set group_by and a numeric metric; returns count and summed metric per group",
            "numeric_aggregates": "set metric to a numeric field",
            "list": "select fields or use the entity defaults",
            "planning": "select the entity, fields, operation, and whether record details help answer the question",
            "missing_values": "is_missing already means absent, null, or blank; do not query extra sentinel values",
            "list_limit": "use the default 10 rows; the hard maximum is 20",
            "scope": "every operation is restricted to the authorized project",
        },
    }


def _project_graph_structure(
    ctx: BimRunContext,
    sample_limit: int = 3,
    include_properties: bool = False,
    focus: str = "",
) -> dict[str, Any]:
    """Return a bounded, project-scoped profile of live node and relationship types."""
    sample_limit = max(1, min(sample_limit, 10))
    path = ctx.graph_contract.authorization_path
    client = ctx.graph_contract.node(path.start_node)
    project = ctx.graph_contract.node("project")
    hub = ctx.graph_contract.node("bim_hub")
    if not client.identity_property or not project.identity_property:
        raise RuntimeError("The graph contract is missing hierarchy identity properties.")
    client_project = ctx.graph_contract.relationship_types[path.relationships[0]]
    project_hub = ctx.graph_contract.relationship_types[path.relationships[1]]
    scope_match = (
        f"MATCH (c:`{client.label}` {{`{client.identity_property}`: $client_id}}) "
        f"-[:`{client_project.name}`]->"
        f"(p:`{project.label}` {{`{project.identity_property}`: $project_id}}) "
        f"OPTIONAL MATCH (p)-[:`{project_hub.name}`]->(h:`{hub.label}`) "
    )
    parameters = {
        "client_id": ctx.scope.client_id,
        "project_id": ctx.scope.project_id,
        "allowed_sources": ctx.scope.allowed_sources,
        "sample_limit": sample_limit,
    }
    node_property_projection = (
        ", collect(DISTINCT keys(n)) AS property_key_groups"
        if include_properties else ", [] AS property_key_groups"
    )
    node_rows = ctx.bim.query(
        scope_match
        + "WITH c, p, h "
        + f"MATCH (n) WHERE n.`{path.source_property}` IN $allowed_sources "
        + "WITH c, p, h, collect(DISTINCT n) AS source_nodes "
        + "WITH [c, p, h] + source_nodes AS scoped_nodes "
        + "UNWIND scoped_nodes AS n WITH DISTINCT n WHERE n IS NOT NULL "
        + "RETURN labels(n) AS labels, count(DISTINCT n) AS node_count, "
        + "collect(DISTINCT coalesce(n.name, n.long_name, n.id, n.GlobalID, n.object_id))"
        + "[0..$sample_limit] AS sample_names"
        + node_property_projection
        + " ORDER BY node_count DESC",
        parameters,
    )
    relationship_property_projection = (
        "collect(DISTINCT keys(r)) AS property_key_groups, "
        if include_properties else "[] AS property_key_groups, "
    )
    relationship_rows = ctx.bim.query(
        scope_match
        + "WITH c, p, h "
        + "MATCH (a)-[r]->(b) "
        + f"WHERE (a.`{path.source_property}` IN $allowed_sources OR a = c OR a = p OR a = h) "
        + f"AND (b.`{path.source_property}` IN $allowed_sources OR b = c OR b = p OR b = h) "
        + "RETURN labels(a) AS from_labels, type(r) AS relationship_type, "
        + "labels(b) AS to_labels, count(r) AS relationship_count, "
        + relationship_property_projection
        + "collect(DISTINCT coalesce(r.name, r.id))[0..$sample_limit] AS sample_names "
        + "ORDER BY relationship_count DESC",
        parameters,
    )

    contract_nodes_by_label: dict[str, list[str]] = {}
    for contract_name, node in ctx.graph_contract.node_types.items():
        contract_nodes_by_label.setdefault(node.label, []).append(contract_name)
    node_types = []
    for row in node_rows:
        labels = sorted(row.get("labels") or [])
        property_keys = sorted({
            key
            for group in row.get("property_key_groups") or []
            for key in (group or [])
        })
        node_types.append({
            "labels": labels,
            "node_count": int(row.get("node_count") or 0),
            "sample_names": row.get("sample_names") or [],
            "properties": property_keys,
            "contract_node_types": sorted({
                contract_name
                for label in labels
                for contract_name in contract_nodes_by_label.get(label, [])
            }),
        })

    relationship_types = []
    for row in relationship_rows:
        property_keys = sorted({
            key
            for group in row.get("property_key_groups") or []
            for key in (group or [])
        })
        relationship_type = str(row.get("relationship_type") or "")
        from_labels = sorted(row.get("from_labels") or [])
        to_labels = sorted(row.get("to_labels") or [])
        contract_relationship_types = sorted(
            contract_name
            for contract_name, relationship in ctx.graph_contract.relationship_types.items()
            if relationship.name == relationship_type
            and ctx.graph_contract.node(relationship.from_node).label in from_labels
            and ctx.graph_contract.node(relationship.to_node).label in to_labels
        )
        relationship_types.append({
            "type": relationship_type,
            "from_labels": from_labels,
            "to_labels": to_labels,
            "relationship_count": int(row.get("relationship_count") or 0),
            "property_keys": property_keys,
            "sample_names": row.get("sample_names") or [],
            "contract_relationship_types": contract_relationship_types,
        })
    normalized_focus = _normalize(focus)
    if normalized_focus:
        node_types = [
            node_type for node_type in node_types
            if normalized_focus in _normalize(" ".join([
                *node_type["labels"],
                *[str(name) for name in node_type["sample_names"]],
                *node_type["properties"],
            ]))
        ]
        relationship_types = [
            relationship for relationship in relationship_types
            if normalized_focus in _normalize(" ".join([
                relationship["type"],
                *relationship["from_labels"],
                *relationship["to_labels"],
                *[str(name) for name in relationship["sample_names"]],
                *relationship["property_keys"],
            ]))
        ]
    unique_node_labels = sorted({
        label for node_type in node_types for label in node_type["labels"]
    })
    unique_relationship_names = sorted({
        relationship["type"] for relationship in relationship_types
    })
    return {
        "project_id": ctx.scope.project_id,
        "authorized_source_count": len(ctx.scope.allowed_sources),
        "focus": focus or None,
        "properties_included": include_properties,
        "node_types": node_types,
        "relationship_types": relationship_types,
        "unique_node_labels": unique_node_labels,
        "unique_relationship_names": unique_relationship_names,
        "node_type_count": len(node_types),
        "relationship_pattern_count": len(relationship_types),
        "unique_node_label_count": len(unique_node_labels),
        "unique_relationship_name_count": len(unique_relationship_names),
        "trusted_scope_property": path.source_property,
        "scope_note": "Only nodes and relationships belonging to the authorized project are included.",
        "usage_note": (
            "Use these live names to select a contract-backed declarative query plan. "
            "Do not generate or execute raw Cypher."
        ),
    }


def _record_mapping(ctx: BimRunContext, entity: QueryEntity) -> tuple[str, str, str]:
    if not entity.node_type or not entity.identity_property or not entity.source_property:
        raise ValueError("The selected entity is not a record collection.")
    return (
        ctx.graph_contract.node(entity.node_type).label,
        entity.identity_property,
        entity.source_property,
    )


def _entity_from_registered_mapping(
    ctx: BimRunContext, plan: BimQueryPlan
) -> tuple[QueryEntity, str | None]:
    if not plan.mapping_id:
        if plan.entity == "permit_knowledge":
            # A compliance investigation must be able to prove that no scoped
            # requirement records exist, even after another entity was mapped.
            return ctx.graph_contract.query_entity(plan.entity), None
        if len(ctx.schema_mappings) == 1:
            plan.mapping_id = next(iter(ctx.schema_mappings))
        else:
            entity = ctx.graph_contract.query_entity(plan.entity)
            if entity.kind == "records":
                raise ValueError("Record queries require one unambiguous registered live schema mapping.")
            return entity, None
    registered = ctx.schema_mappings.get(plan.mapping_id)
    if not isinstance(registered, RegisteredSchemaMapping):
        raise ValueError(f"Unknown or expired schema mapping {plan.mapping_id!r}.")
    proposal = registered.proposal
    fields = {
        field.semantic_name: QueryField(
            property=field.property,
            data_type=field.data_type,
            unit=field.unit,
            ontology_kind=field.ontology_kind,
            description=f"Live-mapped field {field.semantic_name}.",
        )
        for field in proposal.fields
    }
    return QueryEntity(
        kind="records",
        description=f"Live-mapped {proposal.entity_name} records.",
        node_type=None,
        identity_property=proposal.identity_property,
        source_property=proposal.source_property,
        classification_source_property=proposal.classification_source_property,
        classification_name_property=proposal.classification_name_property,
        default_select=[name for name in ("object_id", "name", "ifc_class", "type", "level") if name in fields],
        fields=fields,
    ), proposal.label


def _registered_match_clause(mapping: RegisteredSchemaMapping | None, label: str) -> str:
    """Build a safe MATCH clause from a previously validated live relationship path."""
    if not mapping or not mapping.proposal.relationship_path:
        return f"MATCH (n:`{label}`)"
    parts = [f"(p0:`{mapping.proposal.relationship_path[0].from_label}`)"]
    for index, step in enumerate(mapping.proposal.relationship_path):
        target = "n" if index == len(mapping.proposal.relationship_path) - 1 else f"p{index + 1}"
        target_node = f"({target}:`{step.to_label}`)"
        relationship = f"[:`{step.relationship_type}`]"
        parts.append(
            f"-{relationship}->{target_node}"
            if step.direction == "outgoing"
            else f"<-{relationship}-{target_node}"
        )
    return "MATCH " + "".join(parts)


def _field(entity: QueryEntity, name: str) -> QueryField:
    try:
        return entity.fields[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown field {name!r}; allowed fields are {sorted(entity.fields)}."
        ) from exc


def _raw_filter_values(item: BimFilter) -> list[str]:
    values = [*item.values]
    if item.value != "":
        values.insert(0, item.value)
    return [str(value) for value in values]


def _semantic_values(
    ctx: BimRunContext,
    label: str,
    source_property: str,
    field: QueryField,
    values: list[str],
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if field.ontology_kind == "canonical_type":
        resolved = [_resolve_canonical_type(ctx, value)[0] for value in values]
        return list(dict.fromkeys(resolved)), notes
    if field.ontology_kind == "ifc_class":
        resolved: list[str] = []
        for value in values:
            resolved.extend(ctx.bim.ontology.resolve_element_classes(value) or [value])
        return list(dict.fromkeys(resolved)), notes
    if field.ontology_kind == "level":
        rows = ctx.bim.query(
            f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
            f"AND n.`{field.property}` IS NOT NULL "
            f"RETURN DISTINCT n.`{field.property}` AS value",
            {"allowed_sources": ctx.scope.allowed_sources},
        )
        available = [str(row["value"]) for row in rows if row.get("value") is not None]
        matched = [
            stored for stored in available
            if any(_level_matches(requested, stored, ctx.graph_contract.level_aliases) for requested in values)
        ]
        if values and not matched:
            notes.append(f"No modelled level matched {', '.join(values)}.")
        return list(dict.fromkeys(matched or values)), notes
    if field.property == "load_bearing":
        aliases = {"true": "Yes", "yes": "Yes", "load bearing": "Yes", "false": "No", "no": "No"}
        return [aliases.get(_normalize(value), value) for value in values], notes
    return values, notes


def _compile_filters(
    ctx: BimRunContext,
    entity: QueryEntity,
    label: str,
    source_property: str,
    filters: list[BimFilter],
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]], list[str]]:
    clauses: list[str] = []
    parameters: dict[str, Any] = {"allowed_sources": ctx.scope.allowed_sources}
    normalized_filters: list[dict[str, Any]] = []
    limitations: list[str] = []
    for index, item in enumerate(filters):
        field = _field(entity, item.field)
        prop = f"n.`{field.property}`"
        key = f"filter_{index}"
        requested_values = _raw_filter_values(item)
        values, notes = _semantic_values(
            ctx, label, source_property, field, requested_values
        )
        limitations.extend(notes)
        normalized_filters.append({
            "field": item.field,
            "operator": item.operator,
            "values": values,
            "requested_values": requested_values,
        })
        if item.operator in {"exists", "is_missing"}:
            exists = f"{prop} IS NOT NULL AND trim(toString({prop})) <> ''"
            clauses.append(f"({exists})" if item.operator == "exists" else f"(NOT ({exists}))")
            continue
        if not values:
            raise ValueError(f"Filter {item.field!r} requires a value.")
        if item.operator in {"greater_than", "greater_or_equal", "less_than", "less_or_equal"}:
            if field.data_type != "number":
                raise ValueError(f"Numeric comparison is not allowed for {item.field!r}.")
            try:
                parameters[key] = float(values[0])
            except ValueError as exc:
                raise ValueError(f"Filter {item.field!r} requires a numeric value.") from exc
            symbol = {
                "greater_than": ">", "greater_or_equal": ">=",
                "less_than": "<", "less_or_equal": "<=",
            }[item.operator]
            clauses.append(f"toFloat({prop}) {symbol} ${key}")
            continue
        if field.data_type == "number" and item.operator in {"equals", "not_equals", "in"}:
            try:
                parameters[key] = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(f"Filter {item.field!r} requires numeric values.") from exc
            numeric_clause = f"toFloat({prop}) IN ${key}"
            clauses.append(
                f"NOT ({numeric_clause})" if item.operator == "not_equals" else numeric_clause
            )
            continue
        if field.data_type == "number" and item.operator in {"contains", "starts_with"}:
            raise ValueError(f"Text matching is not allowed for numeric field {item.field!r}.")
        lowered = [_normalize(value) for value in values]
        if item.operator == "equals":
            parameters[key] = lowered
            clauses.append(f"toLower(trim(toString({prop}))) IN ${key}")
        elif item.operator == "not_equals":
            parameters[key] = lowered
            clauses.append(f"NOT toLower(trim(toString({prop}))) IN ${key}")
        elif item.operator == "in":
            parameters[key] = lowered
            clauses.append(f"toLower(trim(toString({prop}))) IN ${key}")
        elif item.operator in {"contains", "starts_with"}:
            parameters[key] = lowered[0]
            predicate = "CONTAINS" if item.operator == "contains" else "STARTS WITH"
            clauses.append(f"toLower(toString({prop})) {predicate} ${key}")
        else:
            raise ValueError(f"Unsupported filter operator {item.operator!r}.")
    return clauses, parameters, normalized_filters, limitations


def _clean_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 500:
        return value[:497] + "..."
    return value


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero length.")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _rank_embedding_candidates(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(candidate_embeddings) != len(candidates):
        raise ValueError("Every semantic candidate requires one embedding.")
    ranked = []
    for candidate, embedding in zip(candidates, candidate_embeddings):
        ranked.append({
            **candidate,
            "similarity": round(_cosine_similarity(query_embedding, embedding), 6),
        })
    return sorted(ranked, key=lambda item: (-item["similarity"], item["node_ref"]))


def _format_filters(filters: list[dict[str, Any]]) -> str:
    if not filters:
        return "without additional filters"
    parts = []
    for item in filters:
        values = ", ".join(
            str(value) for value in (item.get("requested_values") or item["values"])
        )
        parts.append(
            f"{item['field'].replace('_', ' ')} "
            f"{item['operator'].replace('_', ' ')} {values}".strip()
        )
    return "where " + " and ".join(parts)


def _format_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    formatted: list[str] = []
    for row in rows:
        values = [
            f"{field.replace('_', ' ')}: {row.get(field)}"
            for field in fields
            if row.get(field) not in (None, "")
        ]
        formatted.append(", ".join(values) or "no populated selected fields")
    return formatted


def _display_number(value: Any) -> str:
    if value is None:
        return "not modelled"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _pluralize(value: str, count: int) -> str:
    noun = value.replace("_", " ").strip()
    if count == 1:
        return noun
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith("s"):
        return noun
    return noun + "s"


def _space_list_headline(
    matched_count: int,
    displayed_count: int,
    normalized_filters: list[dict[str, Any]],
) -> str:
    type_filter = next(
        (item for item in normalized_filters if item["field"] == "type"), None
    )
    level_filter = next(
        (item for item in normalized_filters if item["field"] == "level"), None
    )
    type_values = (type_filter or {}).get("values") or []
    noun = _pluralize(str(type_values[0]), matched_count) if len(type_values) == 1 else _pluralize("space", matched_count)
    verb = "is" if matched_count == 1 else "are"
    level_text = ""
    requested_levels = (level_filter or {}).get("requested_values") or []
    if len(requested_levels) == 1:
        level_text = f" on the {requested_levels[0]}"
    headline = f"There {verb} {matched_count} {noun}{level_text}."
    if displayed_count < matched_count:
        headline += f" Showing {displayed_count} of {matched_count} detailed records."
    return headline


def _format_space_rows(rows: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for row in rows:
        ifc_class = str(row.get("ifc_class") or "IfcSpace")
        object_id = str(row.get("object_id") or row.get("global_id") or "unknown ID")
        lines = [f"{ifc_class} {object_id}"]
        values = [
            ("Name", row.get("name")),
            ("Level", row.get("level")),
            (
                "Area",
                f"{_display_number(row.get('area_m2'))} m²"
                if row.get("area_m2") is not None else None,
            ),
        ]
        segment = row.get("segment")
        if segment not in (None, ""):
            values.append(("Segment", segment))
        values.extend([
            ("Owner", row.get("owner")),
            ("Room Count", row.get("room_count")),
            ("Dwelling Type", row.get("dwelling_type")),
        ])
        lines.extend(
            f"  - {label}: {value}"
            for label, value in values
            if value not in (None, "")
        )
        formatted.append("\n".join(lines))
    return formatted


def _classification_limitations(
    ctx: BimRunContext,
    entity: QueryEntity,
    label: str,
    source_property: str,
    normalized_filters: list[dict[str, Any]],
) -> list[str]:
    if not entity.classification_source_property:
        return []
    type_filters = [
        item for item in normalized_filters
        if item["field"] == "type" and item["operator"] in {"equals", "in"}
    ]
    if not type_filters:
        return []
    type_field = _field(entity, "type")
    name_property = entity.classification_name_property or entity.identity_property
    issues: list[str] = []
    for item in type_filters:
        rows = ctx.bim.query(
            f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
            f"AND toLower(toString(n.`{type_field.property}`)) IN $types "
            f"AND n.`{entity.classification_source_property}` IS NOT NULL "
            f"RETURN DISTINCT n.`{name_property}` AS name, "
            f"n.`{entity.classification_source_property}` AS source_term LIMIT 20",
            {
                "allowed_sources": ctx.scope.allowed_sources,
                "types": [_normalize(value) for value in item["values"]],
            },
        )
        for row in rows:
            source_term = str(row.get("source_term") or "")
            resolved = ctx.bim.ontology.resolve_space_function(source_term)
            if resolved not in item["values"]:
                issues.append(
                    f"{row.get('name') or 'An element'} is stored under type "
                    f"{', '.join(item['values'])}, but its source term {source_term!r} is not "
                    "resolved to that type by the current ontology."
                )
    return issues


def _count_project_nodes(
    ctx: BimRunContext, cypher: CypherQueryHandler | None = None
) -> dict[str, Any]:
    path = ctx.graph_contract.authorization_path
    client = ctx.graph_contract.node(path.start_node)
    project = ctx.graph_contract.node("project")
    hub = ctx.graph_contract.node("bim_hub")
    if not client.identity_property or not project.identity_property:
        raise RuntimeError("The graph contract is missing hierarchy identity properties.")
    client_project = ctx.graph_contract.relationship_types[path.relationships[0]]
    project_hub = ctx.graph_contract.relationship_types[path.relationships[1]]
    execute = cypher.execute if cypher is not None else ctx.bim.query
    rows = execute(
        f"MATCH (c:`{client.label}` {{`{client.identity_property}`: $client_id}}) "
        f"-[:`{client_project.name}`]->"
        f"(p:`{project.label}` {{`{project.identity_property}`: $project_id}}) "
        f"OPTIONAL MATCH (p)-[:`{project_hub.name}`]->(h:`{hub.label}`) "
        "WITH collect(DISTINCT c) + collect(DISTINCT p) + collect(DISTINCT h) AS hierarchy_nodes "
        f"MATCH (n) WHERE n.`{path.source_property}` IN $allowed_sources "
        "WITH hierarchy_nodes, collect(DISTINCT n) AS source_nodes "
        "UNWIND hierarchy_nodes + source_nodes AS scoped_node "
        "RETURN count(DISTINCT scoped_node) AS total_nodes, "
        "size(source_nodes) AS source_scoped_nodes, size(hierarchy_nodes) AS hierarchy_nodes",
        {
            "client_id": ctx.scope.client_id,
            "project_id": ctx.scope.project_id,
            "allowed_sources": ctx.scope.allowed_sources,
        },
    )
    counts = rows[0] if rows else {}
    total = int(counts.get("total_nodes") or 0)
    return {
        "rows": [{
            "total_nodes": total,
            "source_scoped_nodes": int(counts.get("source_scoped_nodes") or 0),
            "hierarchy_nodes": int(counts.get("hierarchy_nodes") or 0),
        }],
        "matched_count": total,
        "claim": {
            "statement": f"The authorized project graph contains {total} nodes.",
            "value": total,
            "unit": "nodes",
            "basis": "Distinct source-scoped nodes plus the authorized Client, Project, and BIMHub nodes.",
        },
        "limitations": ["This is not a count of the entire Neo4j database."],
    }


def _validate_plan(entity: QueryEntity, plan: BimQueryPlan) -> None:
    if entity.kind == "project_graph":
        if plan.operation != "count" or plan.filters:
            raise ValueError("project_graph supports only an unfiltered count operation.")
        return
    for item in plan.filters:
        _field(entity, item.field)
    if plan.operation in {"group_count", "group_summary", "distinct"}:
        if not plan.group_by:
            raise ValueError(f"{plan.operation} requires group_by.")
        _field(entity, plan.group_by)
    if plan.operation == "group_summary":
        if not plan.metric:
            raise ValueError("group_summary requires a numeric metric.")
        if _field(entity, plan.metric).data_type != "number":
            raise ValueError(f"Metric {plan.metric!r} is not numeric.")
    if plan.operation in {"sum", "average", "minimum", "maximum"}:
        if not plan.metric:
            raise ValueError(f"{plan.operation} requires a numeric metric.")
        if _field(entity, plan.metric).data_type != "number":
            raise ValueError(f"Metric {plan.metric!r} is not numeric.")
    if plan.operation == "list":
        for field_name in plan.select or entity.default_select:
            _field(entity, field_name)
        if plan.sort_by:
            _field(entity, plan.sort_by)


def _execute_plan(ctx: BimRunContext, plan: BimQueryPlan) -> dict[str, Any]:
    cypher = CypherQueryHandler(ctx.bim.query, ctx.scope.allowed_sources)
    entity, live_label = _entity_from_registered_mapping(ctx, plan)
    entity_name = plan.entity.replace("_", " ")
    _validate_plan(entity, plan)
    registered_mapping = ctx.schema_mappings.get(plan.mapping_id)
    if isinstance(registered_mapping, RegisteredSchemaMapping):
        for binding in registered_mapping.proposal.value_bindings:
            selected_values = {_normalize(match.value) for match in binding.matches}
            plan_values = {
                _normalize(value)
                for item in plan.filters
                if item.field == binding.semantic_name
                for value in _raw_filter_values(item)
            }
            if not selected_values.issubset(plan_values):
                raise ValueError(
                    f"The query plan must use the registered exact value binding for "
                    f"{binding.user_concept!r} on semantic field {binding.semantic_name!r}."
                )
    if entity.kind == "project_graph":
        result = _count_project_nodes(ctx, cypher)
    else:
        if live_label is None:
            label, identity_property, source_property = _record_mapping(ctx, entity)
        else:
            if not entity.identity_property or not entity.source_property:
                raise ValueError("The registered schema mapping is incomplete.")
            label, identity_property, source_property = (
                live_label, entity.identity_property, entity.source_property
            )
        match_clause = _registered_match_clause(
            registered_mapping if isinstance(registered_mapping, RegisteredSchemaMapping) else None,
            label,
        )
        clauses, parameters, normalized_filters, limitations = _compile_filters(
            ctx, entity, label, source_property, plan.filters
        )
        limitations.extend(_classification_limitations(
            ctx, entity, label, source_property, normalized_filters
        ))
        where = f"n.`{source_property}` IN $allowed_sources"
        if clauses:
            where += " AND " + " AND ".join(f"({clause})" for clause in clauses)
        identity = f"n.`{identity_property}`"
        filter_text = _format_filters(normalized_filters)
        rows: list[dict[str, Any]]
        matched_count: int
        if plan.operation == "count":
            rows = cypher.execute(
                f"{match_clause} WHERE {where} "
                f"RETURN count(DISTINCT {identity}) AS count",
                parameters,
            )
            matched_count = int(rows[0].get("count") or 0) if rows else 0
            missing_level = next(
                (note for note in limitations if note.startswith("No modelled level matched")), None
            )
            if plan.entity == "spaces" and plan.include_details and matched_count and not missing_level:
                selected = entity.default_select
                projections = ", ".join(
                    f"n.`{_field(entity, name).property}` AS `{name}`" for name in selected
                )
                parameters["limit"] = plan.limit
                rows = cypher.execute(
                    f"{match_clause} WHERE {where} RETURN {projections} "
                    f"ORDER BY {identity} ASC LIMIT $limit",
                    parameters,
                )
                rows = [{key: _clean_value(value) for key, value in row.items()} for row in rows]
                details = _format_space_rows(rows)
                claim = {
                    "statement": _space_list_headline(
                        matched_count, len(details), normalized_filters
                    ),
                    "value": matched_count,
                    "unit": entity_name,
                    "basis": (
                        f"Distinct {label}.{identity_property} values plus a bounded technical "
                        "record breakdown in the authorized source scope."
                    ),
                    "details": details,
                    "total_count": matched_count,
                    "displayed_count": len(details),
                }
            else:
                claim = {
                    "statement": (
                        f"{missing_level} The requested count cannot be determined."
                        if missing_level
                        else f"The authorized project contains {matched_count} {entity_name} {filter_text}."
                    ),
                    "value": None if missing_level else matched_count,
                    "unit": entity_name,
                    "basis": f"Distinct {label}.{identity_property} values in the authorized source scope.",
                }
        elif plan.operation == "list":
            selected = plan.select or entity.default_select
            projections = ", ".join(
                f"n.`{_field(entity, name).property}` AS `{name}`" for name in selected
            )
            count_rows = cypher.execute(
                f"{match_clause} WHERE {where} RETURN count(DISTINCT {identity}) AS count",
                parameters,
            )
            matched_count = int(count_rows[0].get("count") or 0) if count_rows else 0
            order_fields = []
            if plan.sort_by:
                direction = "DESC" if plan.sort_order == "descending" else "ASC"
                order_fields.append(f"n.`{_field(entity, plan.sort_by).property}` {direction}")
            order_fields.append(f"{identity} ASC")
            order = " ORDER BY " + ", ".join(order_fields)
            parameters["limit"] = plan.limit
            rows = cypher.execute(
                f"{match_clause} WHERE {where} RETURN {projections}"
                f"{order} LIMIT $limit",
                parameters,
            )
            rows = [{key: _clean_value(value) for key, value in row.items()} for row in rows]
            details = _format_space_rows(rows) if plan.entity == "spaces" else _format_rows(rows, selected)
            claim = {
                "statement": (
                    (
                        "No project-scoped permit knowledge records were found. "
                        "Compliance cannot be assessed from the available BIM evidence."
                    )
                    if matched_count == 0 and plan.entity == "permit_knowledge"
                    else f"No project-scoped {entity_name} were found {filter_text}."
                    if matched_count == 0
                    else _space_list_headline(matched_count, len(details), normalized_filters)
                    if plan.entity == "spaces"
                    else f"Found {matched_count} project-scoped {entity_name} {filter_text}."
                ),
                "value": matched_count,
                "unit": entity_name,
                "basis": f"Project-scoped {label} records; output is limited to {plan.limit} rows.",
                "details": details,
                "total_count": matched_count,
                "displayed_count": len(details),
            }
        elif plan.operation == "group_summary":
            group = _field(entity, plan.group_by)
            metric = _field(entity, plan.metric)
            parameters["limit"] = plan.limit
            totals = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN count(DISTINCT n.`{group.property}`) AS total_groups, "
                f"count(DISTINCT {identity}) AS total_records",
                parameters,
            )
            total_groups = int(totals[0].get("total_groups") or 0) if totals else 0
            matched_count = int(totals[0].get("total_records") or 0) if totals else 0
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN n.`{group.property}` AS value, count(DISTINCT {identity}) AS count, "
                f"sum(toFloat(n.`{metric.property}`)) AS metric_value, "
                f"count(n.`{metric.property}`) AS metric_records "
                "ORDER BY count DESC, value LIMIT $limit",
                parameters,
            )

            is_function_schedule = plan.entity == "spaces" and plan.group_by == "type"
            detail_rows = []
            for row in rows:
                group_name = str(row.get("value") or "unknown").replace("_", " ").title()
                record_count = int(row.get("count") or 0)
                metric_value = _display_number(row.get("metric_value"))
                unit_text = f" {metric.unit}" if metric.unit else ""
                record_unit = (
                    "space" if is_function_schedule and record_count == 1
                    else "spaces" if is_function_schedule
                    else "record" if record_count == 1
                    else "records"
                )
                detail_rows.append(
                    f"{group_name}: {record_count} {record_unit}, covering {metric_value}{unit_text}."
                )
            level_filter = next(
                (item for item in normalized_filters if item["field"] == "level"), None
            )
            if is_function_schedule:
                location = ""
                if level_filter:
                    requested = (level_filter.get("requested_values") or ["requested level"])[0]
                    stored = ", ".join(str(value) for value in level_filter.get("values") or [])
                    location = f" on the {requested} ({stored})"
                headline = f"There are {total_groups} types of functions{location}:"
            else:
                headline = (
                    f"There are {total_groups} {plan.group_by.replace('_', ' ')} groups "
                    f"across {matched_count} project-scoped {entity_name}:"
                )
            statement = headline
            if detail_rows:
                statement += "\n- " + "\n- ".join(detail_rows)
            claim = {
                "statement": statement,
                "value": total_groups,
                "unit": f"{plan.group_by.replace('_', ' ')} groups",
                "basis": (
                    f"Distinct {label}.{identity_property} counts and summed "
                    f"{label}.{metric.property} values grouped by {group.property}."
                ),
                "total_count": total_groups,
                "displayed_count": len(rows),
            }
        elif plan.operation in {"group_count", "distinct"}:
            group = _field(entity, plan.group_by)
            totals = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN count(DISTINCT n.`{group.property}`) AS total_groups, "
                f"count(DISTINCT {identity}) AS total_records",
                parameters,
            )
            total_groups = int(totals[0].get("total_groups") or 0) if totals else 0
            total_records = int(totals[0].get("total_records") or 0) if totals else 0
            parameters["limit"] = plan.limit
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN n.`{group.property}` AS value, count(DISTINCT {identity}) AS count "
                "ORDER BY count DESC, value LIMIT $limit",
                parameters,
            )
            matched_count = total_records
            details = [f"{row.get('value')}: {row.get('count')}" for row in rows]
            noun = "Distinct values" if plan.operation == "distinct" else "Counts"
            claim = {
                "statement": (
                    f"{noun} for project-scoped {entity_name} by {plan.group_by.replace('_', ' ')}: "
                    f"{total_groups} groups covering {matched_count} records."
                ),
                "value": total_groups if plan.operation == "distinct" else matched_count,
                "unit": "distinct values" if plan.operation == "distinct" else entity_name,
                "basis": f"Grouped distinct {label}.{identity_property} values in the authorized source scope.",
                "details": details,
                "total_count": total_groups,
                "displayed_count": len(details),
            }
        else:
            metric = _field(entity, plan.metric)
            function_name = {
                "sum": "sum", "average": "avg", "minimum": "min", "maximum": "max"
            }[plan.operation]
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{metric.property}` IS NOT NULL "
                f"RETURN {function_name}(toFloat(n.`{metric.property}`)) AS value, "
                f"count(n.`{metric.property}`) AS matched_records",
                parameters,
            )
            value = rows[0].get("value") if rows else None
            matched_count = int(rows[0].get("matched_records") or 0) if rows else 0
            display = "no modelled value" if value is None else str(round(value, 4) if isinstance(value, float) else value)
            unit_text = f" {metric.unit}" if metric.unit else ""
            claim = {
                "statement": (
                    f"The {plan.operation} {plan.metric.replace('_', ' ')} for project-scoped {entity_name} {filter_text} "
                    f"is {display}{unit_text}, based on {matched_count} populated records."
                ),
                "value": value,
                "unit": metric.unit or plan.metric,
                "basis": f"{plan.operation} of populated {label}.{metric.property} values in scope.",
            }
        result = {
            "rows": rows,
            "matched_count": matched_count,
            "claim": claim,
            "limitations": limitations,
            "normalized_filters": normalized_filters,
        }
    stable = {
        "plan": plan.model_dump(mode="json"),
        "rows": result["rows"],
        "matched_count": result["matched_count"],
        "claim": result["claim"],
    }
    result["plan"] = stable["plan"]
    result["result_digest"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    result["graph_contract_version"] = ctx.graph_contract.version
    result["capability"] = "general_bim_query"
    result["cypher_executions"] = cypher.audit_log()
    return result


def get_bim_query_catalog(ctx: PipelineContext) -> str:
    """Return the authorized semantic entities, fields, operations, and filters available for BIM questions."""
    return _json(_catalog(ctx.context))


def inspect_project_graph_structure(
    ctx: PipelineContext,
    sample_limit: int = 3,
    include_properties: bool = False,
    focus: str = "",
) -> str:
    """Fetch distinct project node signatures and relationship patterns with bounded names and properties."""
    result = _project_graph_structure(
        ctx.context,
        sample_limit=sample_limit,
        include_properties=include_properties,
        focus=focus,
    )
    ctx.context.schema_discovery["project_graph_structure"] = result
    if "graph-discovery" not in ctx.context.artifacts:
        ctx.context.add_artifact(RunArtifact(
            artifact_id="graph-discovery",
            kind="graph_discovery",
            producer="Graph Inspector",
            summary=(
                f"Inspected {result['unique_node_label_count']} node labels and "
                f"{result['relationship_pattern_count']} relationship patterns."
            ),
            payload=result,
        ))
    return _json(result)


def _observed_node_type(ctx: BimRunContext, label: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
        raise ValueError("Live labels must be safe Neo4j identifiers.")
    inventory = _queryable_node_inventory(ctx)
    matches = [item for item in inventory["node_types"] if label in item["labels"]]
    if not matches:
        raise ValueError(f"Label {label!r} was not observed in the authorized project.")
    return max(matches, key=lambda item: item["node_count"])


def _queryable_node_inventory(ctx: BimRunContext) -> dict[str, Any]:
    cache_key = "queryable_node_inventory_v1"
    cached = ctx.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return cached
    source_property = ctx.graph_contract.authorization_path.source_property
    rows = ctx.bim.query(
        f"MATCH (n) WHERE n.`{source_property}` IN $allowed_sources "
        "RETURN labels(n) AS labels, count(n) AS node_count, "
        "collect(DISTINCT keys(n)) AS property_key_groups, "
        "collect(DISTINCT coalesce(n.name, n.long_name, n.id))[0..3] AS sample_names "
        "ORDER BY node_count DESC LIMIT 40",
        {"allowed_sources": ctx.scope.allowed_sources},
    )
    node_types = []
    for row in rows:
        node_types.append({
            "labels": sorted(str(value) for value in row.get("labels") or []),
            "node_count": int(row.get("node_count") or 0),
            "properties": sorted({
                str(key)
                for group in row.get("property_key_groups") or []
                for key in (group or [])
            }),
            "sample_names": [str(value) for value in row.get("sample_names") or []],
        })
    result = {
        "project_id": ctx.scope.project_id,
        "trusted_scope_property": source_property,
        "node_types": node_types,
        "scope": "authorized project only",
    }
    ctx.schema_discovery[cache_key] = result
    return result


def inspect_queryable_node_types(ctx: PipelineContext) -> str:
    """Return a compact cached inventory of authorized node labels, properties, counts, and names."""
    return _json(_queryable_node_inventory(ctx.context))


def profile_project_properties(
    ctx: PipelineContext,
    label: str,
    properties: list[str],
    sample_limit: int = 5,
) -> str:
    """Profile bounded values and cardinality for observed properties on one authorized live label."""
    node_type = _observed_node_type(ctx.context, label)
    observed = set(node_type["properties"])
    requested = list(dict.fromkeys(properties))
    if not requested or len(requested) > 8:
        raise ValueError("Profile between one and eight properties per call.")
    unknown = set(requested) - observed
    if unknown:
        raise ValueError(f"Properties were not observed on {label!r}: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in requested):
        raise ValueError("Live property names contain unsafe characters.")
    sample_limit = max(1, min(sample_limit, 10))
    cache_key = "property_profile:" + hashlib.sha256(
        _json([label, requested, sample_limit]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    source_property = ctx.context.graph_contract.authorization_path.source_property
    profiles = []
    for property_name in requested:
        rows = ctx.context.bim.query(
            f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
            f"RETURN count(n) AS node_count, count(n.`{property_name}`) AS populated_count, "
            f"count(DISTINCT n.`{property_name}`) AS distinct_count, "
            f"collect(DISTINCT n.`{property_name}`)[0..$sample_limit] AS sample_values",
            {"allowed_sources": ctx.context.scope.allowed_sources, "sample_limit": sample_limit},
        )
        row = rows[0] if rows else {}
        profiles.append({
            "property": property_name,
            "node_count": int(row.get("node_count") or 0),
            "populated_count": int(row.get("populated_count") or 0),
            "distinct_count": int(row.get("distinct_count") or 0),
            "sample_values": [_clean_value(value) for value in row.get("sample_values") or []],
        })
    result = {"label": label, "profiles": profiles, "scope": "authorized project only"}
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def semantic_search_project_nodes(
    ctx: PipelineContext,
    query: str,
    label: str,
    name_properties: list[str],
    top_k: int = 5,
    candidate_limit: int = 200,
) -> str:
    """Rank authorized live nodes by embedding similarity over their labels and observed name fields."""
    if not query.strip():
        raise ValueError("Semantic node search requires a non-empty concept.")
    node_type = _observed_node_type(ctx.context, label)
    observed = set(node_type["properties"])
    properties = list(dict.fromkeys(name_properties))
    if not properties or len(properties) > 6:
        raise ValueError("Choose between one and six observed name properties.")
    unknown = set(properties) - observed
    if unknown:
        raise ValueError(f"Name properties were not observed on {label!r}: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in properties):
        raise ValueError("Live property names contain unsafe characters.")
    top_k = max(1, min(top_k, 20))
    candidate_limit = max(top_k, min(candidate_limit, 500))
    source_property = ctx.context.graph_contract.authorization_path.source_property
    model = os.getenv("BIM_EMBEDDING_MODEL", "text-embedding-3-small")
    cache_key = "semantic_nodes:" + hashlib.sha256(
        _json([query, label, properties, top_k, candidate_limit, model]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    rows = ctx.context.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        "RETURN elementId(n) AS node_ref, labels(n) AS labels, "
        "[property IN $name_properties | n[property]] AS name_values "
        "ORDER BY node_ref LIMIT $candidate_limit",
        {
            "allowed_sources": ctx.context.scope.allowed_sources,
            "name_properties": properties,
            "candidate_limit": candidate_limit,
        },
    )
    candidates = []
    texts = []
    for row in rows:
        labels = [str(value) for value in row.get("labels") or []]
        values = [str(value)[:200] for value in row.get("name_values") or [] if value not in (None, "")]
        text_value = "type: " + ", ".join(labels) + "; name: " + " | ".join(values)
        candidate = {
            "node_ref": str(row.get("node_ref") or ""),
            "labels": labels,
            "name_values": values,
            "embedding_text": text_value,
        }
        candidates.append(candidate)
        texts.append(text_value)
    if not candidates:
        return _json({"query": query, "matches": [], "scope": "authorized project only"})
    response = OpenAI().embeddings.create(model=model, input=[query, *texts], encoding_format="float")
    vectors = [list(item.embedding) for item in response.data]
    if len(vectors) != len(candidates) + 1:
        raise RuntimeError("Embedding provider returned an incomplete result set.")
    ranked = _rank_embedding_candidates(vectors[0], vectors[1:], candidates)[:top_k]
    result = {
        "query": query,
        "label": label,
        "name_properties": properties,
        "embedding_model": model,
        "candidate_count": len(candidates),
        "matches": ranked,
        "scope": "authorized project only",
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def _project_property_candidates(ctx: BimRunContext, label: str) -> list[dict[str, Any]]:
    cache_key = f"property_candidates:{label}"
    cached = ctx.schema_discovery.get(cache_key)
    if isinstance(cached, list):
        return cached
    node_type = _observed_node_type(ctx, label)
    properties = list(node_type["properties"])[:60]
    if not properties:
        return []
    source_property = ctx.graph_contract.authorization_path.source_property
    rows = ctx.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        "UNWIND $properties AS property "
        "RETURN property, count(n) AS node_count, count(n[property]) AS populated_count, "
        "count(DISTINCT n[property]) AS distinct_count, "
        "collect(DISTINCT n[property])[0..$sample_limit] AS sample_values "
        "ORDER BY property",
        {
            "allowed_sources": ctx.scope.allowed_sources,
            "properties": properties,
            "sample_limit": 6,
        },
    )
    candidates = []
    for row in rows:
        property_name = str(row.get("property") or "")
        sample_values = [
            str(value)[:160] for value in row.get("sample_values") or [] if value not in (None, "")
        ]
        node_count = int(row.get("node_count") or 0)
        populated_count = int(row.get("populated_count") or 0)
        distinct_count = int(row.get("distinct_count") or 0)
        embedding_text = (
            f"property: {property_name}; samples: {' | '.join(sample_values)}; "
            f"populated: {populated_count}/{node_count}; distinct: {distinct_count}"
        )
        candidates.append({
            "node_ref": property_name,
            "property": property_name,
            "node_count": node_count,
            "populated_count": populated_count,
            "distinct_count": distinct_count,
            "sample_values": sample_values,
            "embedding_text": embedding_text,
        })
    ctx.schema_discovery[cache_key] = candidates
    return candidates


def semantic_search_project_properties(
    ctx: PipelineContext,
    label: str,
    concepts: list[str],
    top_k: int = 8,
) -> str:
    """Rank observed live properties for semantic roles using names, values, and cardinality context."""
    requested_concepts = [concept.strip() for concept in concepts if concept.strip()]
    if not requested_concepts or len(requested_concepts) > 6:
        raise ValueError("Search between one and six semantic property concepts.")
    top_k = max(1, min(top_k, 15))
    candidates = _project_property_candidates(ctx.context, label)
    if not candidates:
        return _json({"label": label, "matches_by_concept": {}, "scope": "authorized project only"})
    model = os.getenv("BIM_EMBEDDING_MODEL", "text-embedding-3-small")
    similarity_threshold = float(os.getenv("BIM_SCHEMA_SIMILARITY_THRESHOLD", "0.35"))
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("BIM_SCHEMA_SIMILARITY_THRESHOLD must be between -1 and 1.")
    cache_key = "semantic_properties:" + hashlib.sha256(
        _json([label, requested_concepts, top_k, model, similarity_threshold]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    texts = [candidate["embedding_text"] for candidate in candidates]
    response = OpenAI().embeddings.create(
        model=model,
        input=[*requested_concepts, *texts],
        encoding_format="float",
    )
    vectors = [list(item.embedding) for item in response.data]
    expected = len(requested_concepts) + len(candidates)
    if len(vectors) != expected:
        raise RuntimeError("Embedding provider returned an incomplete result set.")
    candidate_vectors = vectors[len(requested_concepts):]
    matches = {}
    for concept, vector in zip(requested_concepts, vectors[:len(requested_concepts)]):
        ranked = _rank_embedding_candidates(vector, candidate_vectors, candidates)[:top_k]
        matches[concept] = [
            {**item, "meets_threshold": item["similarity"] >= similarity_threshold}
            for item in ranked
        ]
    result = {
        "label": label,
        "embedding_model": model,
        "property_count": len(candidates),
        "similarity_threshold": similarity_threshold,
        "matches_by_concept": matches,
        "scope": "authorized project only",
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def semantic_search_property_values(
    ctx: PipelineContext,
    label: str,
    property: str,
    concepts: list[str],
    top_k: int = 12,
    value_limit: int = 300,
) -> str:
    """Rank exact stored values of an observed property against user concepts in project scope."""
    node_type = _observed_node_type(ctx.context, label)
    if property not in set(node_type["properties"]):
        raise ValueError(f"Property {property!r} was not observed on {label!r}.")
    if "`" in property or any(ord(ch) < 32 for ch in property):
        raise ValueError("Live property names contain unsafe characters.")
    requested_concepts = [concept.strip() for concept in concepts if concept.strip()]
    if not requested_concepts or len(requested_concepts) > 6:
        raise ValueError("Search between one and six value concepts.")
    top_k = max(1, min(top_k, 20))
    value_limit = max(top_k, min(value_limit, 500))
    model = os.getenv("BIM_EMBEDDING_MODEL", "text-embedding-3-small")
    similarity_threshold = float(os.getenv("BIM_SCHEMA_SIMILARITY_THRESHOLD", "0.35"))
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("BIM_SCHEMA_SIMILARITY_THRESHOLD must be between -1 and 1.")
    cache_key = "semantic_values:" + hashlib.sha256(
        _json([label, property, requested_concepts, top_k, value_limit, model, similarity_threshold]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    source_property = ctx.context.graph_contract.authorization_path.source_property
    rows = ctx.context.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        f"AND n.`{property}` IS NOT NULL "
        f"RETURN DISTINCT n.`{property}` AS value ORDER BY toString(value) LIMIT $value_limit",
        {"allowed_sources": ctx.context.scope.allowed_sources, "value_limit": value_limit},
    )
    candidates = []
    for row in rows:
        value = str(row.get("value"))[:300]
        candidates.append({
            "node_ref": value,
            "value": value,
            "embedding_text": f"stored value: {value}",
        })
    if not candidates:
        result = {
            "label": label,
            "property": property,
            "similarity_threshold": similarity_threshold,
            "matches_by_concept": {},
            "scope": "authorized project only",
        }
        ctx.context.schema_discovery[cache_key] = result
        return _json(result)
    response = OpenAI().embeddings.create(
        model=model,
        input=[*requested_concepts, *[item["embedding_text"] for item in candidates]],
        encoding_format="float",
    )
    vectors = [list(item.embedding) for item in response.data]
    if len(vectors) != len(requested_concepts) + len(candidates):
        raise RuntimeError("Embedding provider returned an incomplete result set.")
    candidate_vectors = vectors[len(requested_concepts):]
    matches = {}
    for concept, vector in zip(requested_concepts, vectors[:len(requested_concepts)]):
        ranked = _rank_embedding_candidates(vector, candidate_vectors, candidates)[:top_k]
        matches[concept] = [
            {**item, "meets_threshold": item["similarity"] >= similarity_threshold}
            for item in ranked
        ]
    result = {
        "label": label,
        "property": property,
        "embedding_model": model,
        "similarity_threshold": similarity_threshold,
        "candidate_count": len(candidates),
        "matches_by_concept": matches,
        "scope": "authorized project only",
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def register_schema_mapping(
    ctx: PipelineContext, proposal: SchemaMappingProposal
) -> str:
    """Validate and register a semantic mapping observed in the authorized project."""
    structure = ctx.context.schema_discovery.get("project_graph_structure")
    if not isinstance(structure, dict):
        raise ValueError(
            "Inspect project graph structure, including nodes and relationships, before registering a mapping."
        )
    node_type = _observed_node_type(ctx.context, proposal.label)
    if proposal.relationship_path:
        patterns = structure.get("relationship_types") or []
        previous_label = proposal.relationship_path[0].from_label
        for step in proposal.relationship_path:
            if any(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier) is None
                for identifier in (step.from_label, step.relationship_type, step.to_label)
            ):
                raise ValueError("Relationship paths contain an unsafe label or relationship type.")
            if step.from_label != previous_label:
                raise ValueError("The proposed relationship path must be contiguous.")
            if step.direction == "outgoing":
                observed = any(
                    step.relationship_type == item.get("type")
                    and step.from_label in (item.get("from_labels") or [])
                    and step.to_label in (item.get("to_labels") or [])
                    for item in patterns
                )
            else:
                observed = any(
                    step.relationship_type == item.get("type")
                    and step.from_label in (item.get("to_labels") or [])
                    and step.to_label in (item.get("from_labels") or [])
                    for item in patterns
                )
            if not observed:
                raise ValueError(
                    f"Relationship {step.from_label!r} {step.relationship_type!r} "
                    f"{step.to_label!r} with direction {step.direction!r} was not observed."
                )
            previous_label = step.to_label
        if previous_label != proposal.label:
            raise ValueError("The relationship path must end at the mapped entity label.")
    observed = set(node_type["properties"])
    required = {
        proposal.identity_property,
        proposal.source_property,
        *(field.property for field in proposal.fields),
    }
    for optional in (proposal.classification_source_property, proposal.classification_name_property):
        if optional:
            required.add(optional)
    unknown = required - observed
    if unknown:
        raise ValueError(f"The proposed properties were not observed: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in required):
        raise ValueError("The proposed properties contain unsafe characters.")
    trusted_source = ctx.context.graph_contract.authorization_path.source_property
    if proposal.source_property != trusted_source:
        raise ValueError("The mapping must use the trusted authorization source property.")
    semantic_names = [field.semantic_name for field in proposal.fields]
    if len(semantic_names) != len(set(semantic_names)):
        raise ValueError("Mapped semantic field names must be unique.")
    fields_by_name = {field.semantic_name: field for field in proposal.fields}
    for binding in proposal.value_bindings:
        field = fields_by_name.get(binding.semantic_name)
        if field is None or field.property != binding.property:
            raise ValueError(
                f"Value binding {binding.semantic_name!r} must reference its mapped semantic field."
            )
    required_evidence = {
        "identity": proposal.identity_property,
        **{
            binding.semantic_name: binding.property
            for binding in proposal.value_bindings
        },
    }
    supplied_evidence = {item.semantic_name: item for item in proposal.match_evidence}
    missing_evidence = set(required_evidence) - set(supplied_evidence)
    if missing_evidence:
        raise ValueError(f"Semantic similarity evidence is required for: {sorted(missing_evidence)}")
    semantic_results = [
        value for key, value in ctx.context.schema_discovery.items()
        if key.startswith("semantic_properties:") and isinstance(value, dict)
    ]
    for semantic_name, property_name in required_evidence.items():
        evidence = supplied_evidence[semantic_name]
        if evidence.property != property_name:
            raise ValueError(f"Similarity evidence for {semantic_name!r} names a different property.")
        verified_match = None
        verified_threshold = None
        for result in semantic_results:
            concept_matches = (result.get("matches_by_concept") or {}).get(evidence.concept) or []
            match = next((item for item in concept_matches if item.get("property") == property_name), None)
            if match is not None:
                verified_match = match
                verified_threshold = float(result.get("similarity_threshold", 0.35))
                break
        if verified_match is None:
            raise ValueError(f"No live semantic-search evidence supports {semantic_name!r}.")
        verified_similarity = float(verified_match.get("similarity") or 0.0)
        if abs(verified_similarity - evidence.similarity) > 1e-6:
            raise ValueError(f"Reported similarity for {semantic_name!r} does not match live evidence.")
        if verified_similarity < verified_threshold:
            raise ValueError(
                f"Semantic match for {semantic_name!r} scored {verified_similarity:.3f}, below "
                f"the required threshold {verified_threshold:.3f}."
            )
    semantic_value_results = [
        value for key, value in ctx.context.schema_discovery.items()
        if key.startswith("semantic_values:") and isinstance(value, dict)
    ]
    for binding in proposal.value_bindings:
        verified_result = next(
            (
                result for result in semantic_value_results
                if result.get("label") == proposal.label
                and result.get("property") == binding.property
                and binding.user_concept in (result.get("matches_by_concept") or {})
            ),
            None,
        )
        if verified_result is None:
            raise ValueError(f"No live value-search evidence supports {binding.user_concept!r}.")
        available_matches = {
            str(item.get("value")): item
            for item in verified_result["matches_by_concept"][binding.user_concept]
        }
        threshold = float(verified_result.get("similarity_threshold", 0.35))
        for selected in binding.matches:
            match = available_matches.get(selected.value)
            if match is None:
                raise ValueError(f"Stored value {selected.value!r} was not returned by semantic search.")
            similarity = float(match.get("similarity") or 0.0)
            if abs(similarity - selected.similarity) > 1e-6:
                raise ValueError(f"Reported value similarity for {selected.value!r} was altered.")
            if similarity < threshold:
                raise ValueError(
                    f"Stored value {selected.value!r} scored {similarity:.3f}, below the "
                    f"required threshold {threshold:.3f}."
                )
    digest_input = {
        "project_id": ctx.context.scope.project_id,
        "proposal": proposal.model_dump(mode="json"),
    }
    mapping_id = "mapping-" + hashlib.sha256(
        _json(digest_input).encode("utf-8")
    ).hexdigest()[:12]
    existing = ctx.context.schema_mappings.get(mapping_id)
    if isinstance(existing, RegisteredSchemaMapping):
        return existing.model_dump_json()
    pending_mapping = RegisteredSchemaMapping(
        mapping_id="pending",
        proposal=proposal,
        node_count=1,
        populated_identity_count=1,
        distinct_identity_count=1,
    )
    match_clause = _registered_match_clause(pending_mapping, proposal.label)
    rows = ctx.context.bim.query(
        f"{match_clause} WHERE n.`{proposal.source_property}` IN $allowed_sources "
        f"RETURN count(DISTINCT n) AS node_count, "
        f"count(DISTINCT CASE WHEN n.`{proposal.identity_property}` IS NOT NULL THEN n END) AS populated_count, "
        f"count(DISTINCT n.`{proposal.identity_property}`) AS distinct_count",
        {"allowed_sources": ctx.context.scope.allowed_sources},
    )
    row = rows[0] if rows else {}
    node_count = int(row.get("node_count") or 0)
    populated = int(row.get("populated_count") or 0)
    distinct = int(row.get("distinct_count") or 0)
    if not node_count:
        raise ValueError("The proposed entity has no authorized project records.")
    if populated != node_count or distinct != node_count:
        raise ValueError("The proposed identity property must be populated and unique for every record.")
    for field in proposal.fields:
        if field.data_type != "number":
            continue
        numeric_rows = ctx.context.bim.query(
            f"{match_clause} WHERE n.`{proposal.source_property}` IN $allowed_sources "
            f"RETURN count(DISTINCT CASE WHEN n.`{field.property}` IS NOT NULL THEN n END) AS populated_count, "
            f"count(DISTINCT CASE WHEN toFloat(n.`{field.property}`) IS NOT NULL THEN n END) AS numeric_count",
            {"allowed_sources": ctx.context.scope.allowed_sources},
        )
        numeric_row = numeric_rows[0] if numeric_rows else {}
        if int(numeric_row.get("populated_count") or 0) != int(numeric_row.get("numeric_count") or 0):
            raise ValueError(f"Mapped numeric field {field.semantic_name!r} contains nonnumeric values.")
    registered = RegisteredSchemaMapping(
        mapping_id=mapping_id,
        proposal=proposal,
        node_count=node_count,
        populated_identity_count=populated,
        distinct_identity_count=distinct,
    )
    ctx.context.schema_mappings[mapping_id] = registered
    ctx.context.add_artifact(RunArtifact(
        artifact_id=mapping_id,
        kind="mapping",
        producer="Schema Mapper",
        summary=f"Mapped {proposal.entity_name} to live label {proposal.label}.",
        payload=registered.model_dump(mode="json"),
    ))
    return registered.model_dump_json()


def query_bim(ctx: PipelineContext, plan: BimQueryPlan) -> str:
    """Execute one validated, project-scoped declarative BIM query plan; raw Cypher is never accepted."""
    if ctx.context.task_contract is None:
        raise ValueError("The pipeline must define the BIM task contract before querying.")
    result = _execute_plan(ctx.context, plan)
    evidence_id = f"query-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="query",
        summary=result["claim"]["statement"],
        payload=_json(result),
    ))
    ctx.context.add_artifact(RunArtifact(
        artifact_id=f"plan-{evidence_id}",
        kind="query_plan",
        producer="Query Planner",
        summary=result["claim"]["statement"],
        payload={"evidence_id": evidence_id, **result},
    ))
    ctx.context.add_artifact(RunArtifact(
        artifact_id=f"cypher-{evidence_id}",
        kind="cypher_query",
        producer="Cypher Query Handler",
        summary=f"Executed {len(result.get('cypher_executions') or [])} read-only Cypher calculation(s).",
        payload={
            "evidence_id": evidence_id,
            "executions": result.get("cypher_executions") or [],
        },
    ))
    return _json({"evidence_id": evidence_id, **result})


def _verify_query_evidence(
    context: BimRunContext, requested_evidence_ids: list[str]
) -> dict[str, Any]:
    """Replay every query produced in this run, regardless of model-supplied IDs."""
    checks: list[dict[str, Any]] = []
    available_evidence_ids = [
        evidence_id
        for evidence_id, evidence in context.evidence.items()
        if evidence.kind == "query"
    ]
    for evidence_id in available_evidence_ids:
        evidence = context.evidence[evidence_id]
        original = json.loads(evidence.payload)
        rerun = _execute_plan(context, BimQueryPlan.model_validate(original["plan"]))
        plan = BimQueryPlan.model_validate(rerun["plan"])
        registered = context.schema_mappings.get(plan.mapping_id)
        replay_ok = rerun["result_digest"] == original.get("result_digest")
        scope_ok = bool(context.scope.allowed_sources)
        identity_ok = True
        binding_ok = True
        identity_explanation = "The contract-backed entity defines a distinct identity property."
        binding_explanation = "No registered live value bindings apply to this plan."
        if isinstance(registered, RegisteredSchemaMapping):
            identity_ok = (
                registered.node_count == registered.populated_identity_count
                == registered.distinct_identity_count
            )
            identity_explanation = (
                f"{registered.distinct_identity_count} of {registered.node_count} records have "
                "populated, unique identities."
            )
            missing_bindings: list[str] = []
            for binding in registered.proposal.value_bindings:
                selected = {_normalize(match.value) for match in binding.matches}
                supplied = {
                    _normalize(value)
                    for item in plan.filters
                    if item.field == binding.semantic_name
                    for value in _raw_filter_values(item)
                }
                if not selected.issubset(supplied):
                    missing_bindings.append(binding.user_concept)
            binding_ok = not missing_bindings
            binding_explanation = (
                "Every live-mapped user constraint is present with its exact stored value."
                if binding_ok else f"Missing exact bindings for: {', '.join(missing_bindings)}."
            )
        counting_unit_ok = True
        counting_unit_explanation = "The contract-backed identity defines the record counting unit."
        boundary_ok = all(
            item.operator in {"equals", "in"}
            for item in plan.filters
            if _normalize(item.field) in {"level", "floor", "storey", "story"}
        )
        boundary_explanation = (
            "Level constraints use exact values, so ground cannot match underground."
            if boundary_ok else "A level constraint uses a non-exact operator."
        )
        source_dedup_ok = identity_ok and plan.distinct_by == "identity"
        source_dedup_explanation = (
            "The query counts the globally unique identity across all authorized sources."
            if source_dedup_ok else "Cross-source identity deduplication is not established."
        )
        classification_ok = True
        classification_explanation = "No mapped classification field applies to this plan."
        if isinstance(registered, RegisteredSchemaMapping):
            counting_unit_ok = bool(
                registered.proposal.counting_unit.strip()
                and registered.proposal.counting_unit_evidence.strip()
            )
            counting_unit_explanation = (
                f"One distinct identity represents {registered.proposal.counting_unit}: "
                f"{registered.proposal.counting_unit_evidence}"
                if counting_unit_ok else
                "The mapping does not prove whether one identity is the requested entity or a child record."
            )
            classification_fields = {
                field.semantic_name for field in registered.proposal.fields
                if field.ontology_kind in {"canonical_type", "ifc_class"}
            }
            if classification_fields:
                requested_classification_fields = {
                    binding.semantic_name
                    for binding in registered.proposal.value_bindings
                    if binding.semantic_name in classification_fields
                }
                classification_ok = not requested_classification_fields or all(
                    any(
                        item.field == semantic_name
                        and item.operator in {"equals", "in", "is_missing"}
                        for item in plan.filters
                    )
                    for semantic_name in requested_classification_fields
                )
                classification_explanation = (
                    "No specific classification was requested; the query retains the mapped entity scope."
                    if not requested_classification_fields else
                    "Every requested classification uses an exact mapped boundary."
                    if classification_ok else
                    "A requested classification is not constrained by its exact mapped boundary."
                )
        semantic_checks = [
            {"name": "replay_stability", "passed": replay_ok,
             "explanation": "The rerun result digest matches the original." if replay_ok
             else "The rerun result digest differs from the original."},
            {"name": "authorized_scope", "passed": scope_ok,
             "explanation": f"Execution is restricted to {len(context.scope.allowed_sources)} authorized source(s)."},
            {"name": "identity_integrity", "passed": identity_ok,
             "explanation": identity_explanation},
            {"name": "constraint_binding", "passed": binding_ok,
             "explanation": binding_explanation},
            {"name": "counting_unit", "passed": counting_unit_ok,
             "explanation": counting_unit_explanation},
            {"name": "boundary_exactness", "passed": boundary_ok,
             "explanation": boundary_explanation},
            {"name": "source_deduplication", "passed": source_dedup_ok,
             "explanation": source_dedup_explanation},
            {"name": "classification_purity", "passed": classification_ok,
             "explanation": classification_explanation},
            {"name": "constraint_coverage", "passed": binding_ok,
             "explanation": binding_explanation},
        ]
        checks.append({
            "evidence_id": evidence_id,
            "verified": all(check["passed"] for check in semantic_checks),
            "expected_digest": original.get("result_digest"),
            "actual_digest": rerun["result_digest"],
            "claim": rerun["claim"],
            "limitations": rerun.get("limitations", []),
            "plan": rerun["plan"],
            "semantic_checks": semantic_checks,
        })
    return {
        "verified": bool(checks) and all(check.get("verified") is True for check in checks),
        "checks": checks,
        "requested_evidence_ids": requested_evidence_ids,
        "verified_evidence_ids": available_evidence_ids,
    }


def ensure_bim_verification(context: BimRunContext) -> str | None:
    """Record deterministic verification when query evidence exists but the reviewer did not finish."""
    query_ids = [
        evidence_id for evidence_id, evidence in context.evidence.items()
        if evidence.kind == "query"
    ]
    existing = next(
        (
            evidence_id for evidence_id, evidence in reversed(list(context.evidence.items()))
            if evidence.kind == "verification"
            and set(json.loads(evidence.payload).get("verified_evidence_ids") or []) == set(query_ids)
        ),
        None,
    )
    if existing is not None:
        return existing
    if not query_ids:
        return None
    result = _verify_query_evidence(context, query_ids)
    verification_id = f"verification-{uuid4().hex[:10]}"
    context.add_evidence(Evidence(
        evidence_id=verification_id,
        kind="verification",
        summary=f"Runtime verification of {len(result['checks'])} general BIM query result(s)",
        payload=_json(result),
    ))
    context.add_artifact(RunArtifact(
        artifact_id=f"review-{verification_id}",
        kind="semantic_review",
        producer="Verifier",
        summary=f"Reviewed {len(result['checks'])} query artifact(s).",
        payload={"verification_evidence_id": verification_id, **result},
    ))
    return verification_id


def ensure_compliance_evidence(context: BimRunContext) -> str | None:
    """Ensure a compliance question has an explicit scoped requirements-availability query."""
    for evidence_id, evidence in context.evidence.items():
        if evidence.kind != "query":
            continue
        payload = json.loads(evidence.payload)
        if (payload.get("plan") or {}).get("entity") == "permit_knowledge":
            return evidence_id
    if context.task_contract is None:
        return None
    return json.loads(query_bim(
        PipelineContext(context),
        BimQueryPlan(
            entity="permit_knowledge",
            operation="list",
            select=["name", "summary", "knowledge"],
            limit=20,
        ),
    ))["evidence_id"]


def verify_bim_evidence(
    ctx: PipelineContext, evidence_ids: list[str]
) -> str:
    """Independently rerun saved general BIM query plans and compare their complete result digests."""
    verification_id = ensure_bim_verification(ctx.context)
    if verification_id is None:
        return _json({
            "evidence_id": None,
            "verified": False,
            "checks": [],
            "requested_evidence_ids": evidence_ids,
            "verified_evidence_ids": [],
        })
    result = json.loads(ctx.context.evidence[verification_id].payload)
    return _json({"evidence_id": verification_id, **result})
