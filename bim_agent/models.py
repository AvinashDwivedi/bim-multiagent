from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Intent = Literal["count", "list", "group_count", "sum", "describe"]
Calculation = Literal["count", "sum", "average", "min", "max", "distinct_count", "percentage"]
Scope = Literal["physical_instances", "all_records"]


@dataclass(frozen=True)
class ElementRecord:
    object_id: int
    name: str
    external_id: str | None
    global_id: str | None
    path: tuple[str, ...]
    category: str
    family: str
    type_name: str
    level: str
    properties: dict[str, Any]
    flat_properties: dict[str, str]
    is_leaf: bool
    is_physical: bool

    @property
    def identity(self) -> str:
        # Autodesk/Revit exports can populate IfcGUID with a reused type GUID.
        # externalId carries the stable per-element identity in this three-file format.
        return self.external_id or self.global_id or f"object:{self.object_id}"

    @property
    def searchable_text(self) -> str:
        values = [self.name, self.category, self.family, self.type_name, self.level]
        values.extend(self.flat_properties.values())
        return " | ".join(value for value in values if value)


@dataclass(frozen=True)
class PropertyFilter:
    field: str
    operator: Literal[
        "equals", "not_equals", "contains", "not_contains", "in", "not_in",
        "missing", "not_missing", "gt", "gte", "lt", "lte",
    ]
    value: str | float | list[str]


@dataclass
class QueryBranch:
    label: str = ""
    categories: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    match_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    filters: list[PropertyFilter] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryBranch":
        return cls(
            label=str(value.get("label") or ""),
            categories=_strings(value.get("categories")),
            families=_strings(value.get("families")),
            types=_strings(value.get("types")),
            match_terms=_strings(value.get("match_terms")),
            exclude_terms=_strings(value.get("exclude_terms")),
            filters=_property_filters(value.get("filters")),
        )


@dataclass
class QueryPlan:
    intent: Intent = "count"
    scope: Scope = "physical_instances"
    target_label: str = "matching elements"
    search_terms: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    filters: list[PropertyFilter] = field(default_factory=list)
    # OR across groups, AND inside a group. `filters` remain global AND constraints.
    filter_groups: list[list[PropertyFilter]] = field(default_factory=list)
    population_branches: list[QueryBranch] = field(default_factory=list)
    match_terms: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=lambda: ["family", "type_name"])
    group_by_properties: list[str] = field(default_factory=list)
    measure_property: str | None = None
    calculation: Calculation = "count"
    distinct_property: str | None = None
    metric_filters: list[PropertyFilter] = field(default_factory=list)
    sort_by: str | None = None
    sort_direction: Literal["asc", "desc"] = "desc"
    limit: int | None = None
    minimum_group_count: int | None = None
    select_properties: list[str] = field(default_factory=list)
    output_unit: str | None = None
    analysis: Literal["standard", "connectivity"] = "standard"
    required_facets: list[str] = field(default_factory=list)
    include_related: bool = True
    interpretation: str = ""
    assumptions: list[str] = field(default_factory=list)
    unsupported_requirements: list[str] = field(default_factory=list)
    planner: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any], planner: str = "llm") -> "QueryPlan":
        allowed_intents = {"count", "list", "group_count", "sum", "describe"}
        intent = value.get("intent", "count")
        if intent not in allowed_intents:
            intent = "count"
        scope = value.get("scope", "physical_instances")
        if scope not in {"physical_instances", "all_records"}:
            scope = "physical_instances"
        filters = _property_filters(value.get("filters"))
        return cls(
            intent=intent,
            scope=scope,
            target_label=str(value.get("target_label") or "matching elements"),
            search_terms=_strings(value.get("search_terms")),
            categories=_strings(value.get("categories")),
            families=_strings(value.get("families")),
            types=_strings(value.get("types")),
            exclude_terms=_strings(value.get("exclude_terms")),
            filters=filters,
            filter_groups=[
                _property_filters(group) for group in value.get("filter_groups", [])
                if isinstance(group, list)
            ],
            population_branches=[
                QueryBranch.from_dict(item)
                for item in value.get("population_branches", [])
                if isinstance(item, dict)
            ],
            match_terms=_strings(value.get("match_terms")),
            group_by=[item for item in _strings(value.get("group_by")) if item in {
                "category", "family", "type_name", "level", "name"
            }],
            group_by_properties=_strings(value.get("group_by_properties")),
            measure_property=(str(value["measure_property"]) if value.get("measure_property") else None),
            calculation=(
                value.get("calculation")
                if value.get("calculation") in {
                    "count", "sum", "average", "min", "max", "distinct_count", "percentage"
                }
                else ("sum" if intent == "sum" else "count")
            ),
            distinct_property=(str(value["distinct_property"]) if value.get("distinct_property") else None),
            metric_filters=_property_filters(value.get("metric_filters")),
            sort_by=(str(value["sort_by"]) if value.get("sort_by") else None),
            sort_direction=value.get("sort_direction") if value.get("sort_direction") in {"asc", "desc"} else "desc",
            limit=(max(1, int(value["limit"])) if value.get("limit") else None),
            minimum_group_count=(max(1, int(value["minimum_group_count"])) if value.get("minimum_group_count") else None),
            select_properties=_strings(value.get("select_properties")),
            output_unit=(str(value["output_unit"]) if value.get("output_unit") else None),
            analysis=value.get("analysis") if value.get("analysis") in {"standard", "connectivity"} else "standard",
            required_facets=_strings(value.get("required_facets")),
            include_related=bool(value.get("include_related", True)),
            interpretation=str(value.get("interpretation") or ""),
            assumptions=_strings(value.get("assumptions")),
            unsupported_requirements=_strings(value.get("unsupported_requirements")),
            planner=planner,
        )


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _property_filters(value: Any) -> list[PropertyFilter]:
    output = []
    if not isinstance(value, list):
        return output
    for item in value:
        if not isinstance(item, dict):
            continue
        operator = item.get("operator", "equals")
        if operator not in {
            "equals", "not_equals", "contains", "not_contains", "in", "not_in",
            "missing", "not_missing", "gt", "gte", "lt", "lte",
        }:
            continue
        output.append(PropertyFilter(str(item.get("field") or ""), operator, item.get("value", "")))
    return output


@dataclass
class QueryEvidence:
    selected_count: int
    distinct_identity_count: int
    groups: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    related_groups: list[dict[str, Any]]
    operation_value: float | int | None = None
    unit: str | None = None
    complete: bool = True
    grouped_measurements: list[dict[str, Any]] = field(default_factory=list)
    property_summaries: list[dict[str, Any]] = field(default_factory=list)
    connectivity: dict[str, Any] | None = None
    category_counts: list[dict[str, Any]] = field(default_factory=list)
    metric_numerator: int | float | None = None
    metric_denominator: int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verification:
    status: Literal["verified", "limited", "failed"]
    checks: list[dict[str, Any]]
    limitations: list[str]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerReport:
    answer: str
    status: str
    plan: QueryPlan
    evidence: QueryEvidence
    verification: Verification
    sources: list[dict[str, Any]]
    trace_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "plan": self.plan.to_dict(),
            "evidence": self.evidence.to_dict(),
            "verification": self.verification.to_dict(),
            "sources": self.sources,
            "trace_path": self.trace_path,
        }
