from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Literal
from uuid import uuid4

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from .graph_contract import QueryEntity, QueryField
from .models import BimRunContext, Evidence


Operation = Literal[
    "count", "list", "group_count", "group_summary", "distinct",
    "sum", "average", "minimum", "maximum"
]
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
    operation: Operation
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


def _count_project_nodes(ctx: BimRunContext) -> dict[str, Any]:
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
    entity = ctx.graph_contract.query_entity(plan.entity)
    entity_name = plan.entity.replace("_", " ")
    _validate_plan(entity, plan)
    if entity.kind == "project_graph":
        result = _count_project_nodes(ctx)
    else:
        label, identity_property, source_property = _record_mapping(ctx, entity)
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
            rows = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} "
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
                rows = ctx.bim.query(
                    f"MATCH (n:`{label}`) WHERE {where} RETURN {projections} "
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
            count_rows = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} RETURN count(DISTINCT {identity}) AS count",
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
            rows = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} RETURN {projections}"
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
            totals = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN count(DISTINCT n.`{group.property}`) AS total_groups, "
                f"count(DISTINCT {identity}) AS total_records",
                parameters,
            )
            total_groups = int(totals[0].get("total_groups") or 0) if totals else 0
            matched_count = int(totals[0].get("total_records") or 0) if totals else 0
            rows = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} AND n.`{group.property}` IS NOT NULL "
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
            totals = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN count(DISTINCT n.`{group.property}`) AS total_groups, "
                f"count(DISTINCT {identity}) AS total_records",
                parameters,
            )
            total_groups = int(totals[0].get("total_groups") or 0) if totals else 0
            total_records = int(totals[0].get("total_records") or 0) if totals else 0
            parameters["limit"] = plan.limit
            rows = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} AND n.`{group.property}` IS NOT NULL "
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
            rows = ctx.bim.query(
                f"MATCH (n:`{label}`) WHERE {where} AND n.`{metric.property}` IS NOT NULL "
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
    return result


@function_tool
def get_bim_query_catalog(ctx: RunContextWrapper[BimRunContext]) -> str:
    """Return the authorized semantic entities, fields, operations, and filters available for BIM questions."""
    return _json(_catalog(ctx.context))


@function_tool
def inspect_project_graph_structure(
    ctx: RunContextWrapper[BimRunContext],
    sample_limit: int = 3,
    include_properties: bool = False,
    focus: str = "",
) -> str:
    """Fetch distinct project node signatures and relationship patterns with bounded names and properties."""
    return _json(_project_graph_structure(
        ctx.context,
        sample_limit=sample_limit,
        include_properties=include_properties,
        focus=focus,
    ))


@function_tool
def query_bim(ctx: RunContextWrapper[BimRunContext], plan: BimQueryPlan) -> str:
    """Execute one validated, project-scoped declarative BIM query plan; raw Cypher is never accepted."""
    result = _execute_plan(ctx.context, plan)
    evidence_id = f"query-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="query",
        summary=result["claim"]["statement"],
        payload=_json(result),
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
        checks.append({
            "evidence_id": evidence_id,
            "verified": rerun["result_digest"] == original.get("result_digest"),
            "expected_digest": original.get("result_digest"),
            "actual_digest": rerun["result_digest"],
            "claim": rerun["claim"],
            "limitations": rerun.get("limitations", []),
            "plan": rerun["plan"],
        })
    return {
        "verified": bool(checks) and all(check.get("verified") is True for check in checks),
        "checks": checks,
        "requested_evidence_ids": requested_evidence_ids,
        "verified_evidence_ids": available_evidence_ids,
    }


@function_tool
def verify_bim_evidence(
    ctx: RunContextWrapper[BimRunContext], evidence_ids: list[str]
) -> str:
    """Independently rerun saved general BIM query plans and compare their complete result digests."""
    result = _verify_query_evidence(ctx.context, evidence_ids)
    verification_id = f"verification-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=verification_id,
        kind="verification",
        summary=f"Independent verification of {len(result['checks'])} general BIM query result(s)",
        payload=_json(result),
    ))
    return _json({"evidence_id": verification_id, **result})
