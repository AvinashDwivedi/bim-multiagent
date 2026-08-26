from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from .cypher_handler import CypherQueryHandler


Scalar = str | int | float | bool
QueryFunction = Callable[[str, dict[str, Any]], list[dict[str, Any]]]
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _safe_identifier(value: str, *, kind: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe {kind} in governed computation: {value!r}.")
    return value


def _safe_property(value: str) -> str:
    if not value or "`" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"Unsafe property in governed computation: {value!r}.")
    return value


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class ExactFilter(BaseModel):
    """One governed, byte-for-byte property boundary.

    The computation layer intentionally performs no linguistic normalization. Values
    in a recipe are trusted project configuration and must match stored values exactly.
    """

    property: str
    operator: Literal["in", "not_in"] = "in"
    values: list[Scalar] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_property(self) -> "ExactFilter":
        _safe_property(self.property)
        return self


class GovernedEntity(BaseModel):
    label: str
    identity_property: str
    source_property: str = "source"
    filters: list[ExactFilter] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "GovernedEntity":
        _safe_identifier(self.label, kind="label")
        _safe_property(self.identity_property)
        _safe_property(self.source_property)
        return self


class GroupRelationship(BaseModel):
    relationship_type: str
    target_label: str
    target_property: str
    direction: Literal["outgoing", "incoming", "either"] = "either"
    prefer_relationship: bool = False

    @model_validator(mode="after")
    def validate_identifiers(self) -> "GroupRelationship":
        _safe_identifier(self.relationship_type, kind="relationship type")
        _safe_identifier(self.target_label, kind="target label")
        _safe_property(self.target_property)
        return self


class GroupedCountComponent(GovernedEntity):
    key: str
    group_property: str
    group_fallback_properties: list[str] = Field(default_factory=list, max_length=5)
    group_relationship: GroupRelationship | None = None
    cross_source_identity_join: bool = False

    @model_validator(mode="after")
    def validate_component(self) -> "GroupedCountComponent":
        if not _KEY.fullmatch(self.key):
            raise ValueError(f"Unsafe component key: {self.key!r}.")
        _safe_property(self.group_property)
        for property_name in self.group_fallback_properties:
            _safe_property(property_name)
        return self


class CompositeGroupedCountRecipe(BaseModel):
    """Combine governed entity populations and count distinct identities per group."""

    recipe: Literal["composite_grouped_count"]
    components: list[GroupedCountComponent] = Field(min_length=1, max_length=12)
    identity_namespace: Literal["global", "component"] = "global"
    counting_unit: str = Field(min_length=1, max_length=120)
    group_unit: str = Field(min_length=1, max_length=120)
    missing_group_label: str = Field(default="unassigned", min_length=1, max_length=120)
    semantics: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_components(self) -> "CompositeGroupedCountRecipe":
        keys = [component.key for component in self.components]
        if len(keys) != len(set(keys)):
            raise ValueError("Composite grouped-count component keys must be unique.")
        return self


class NamedScope(BaseModel):
    key: str
    filters: list[ExactFilter] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_key(self) -> "NamedScope":
        if not _KEY.fullmatch(self.key):
            raise ValueError(f"Unsafe scope key: {self.key!r}.")
        return self


class ScopeComparisonRecipe(BaseModel):
    """Count one governed entity under several additive, named scopes."""

    recipe: Literal["scope_comparison"]
    entity: GovernedEntity
    scopes: list[NamedScope] = Field(min_length=2, max_length=12)
    baseline_scope: str
    counting_unit: str = Field(min_length=1, max_length=120)
    semantics: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_scopes(self) -> "ScopeComparisonRecipe":
        keys = [scope.key for scope in self.scopes]
        if len(keys) != len(set(keys)):
            raise ValueError("Scope-comparison keys must be unique.")
        if self.baseline_scope not in keys:
            raise ValueError("baseline_scope must reference one configured scope key.")
        return self


class PropertyCoverageRecipe(BaseModel):
    """Measure explicit assignment/completeness for one governed property.

    This recipe is intentionally domain-neutral: the property may represent a feeding
    panel, fire rating, asset tag, system classification, specification reference, or
    any other project-governed field. It measures stored values only and never upgrades
    a property value into evidence of a graph relationship or physical connection.
    """

    recipe: Literal["property_coverage"]
    entity: GovernedEntity
    property: str
    missing_values: list[Scalar] = Field(default_factory=list, max_length=20)
    counting_unit: str = Field(min_length=1, max_length=120)
    assignment_unit: str = Field(min_length=1, max_length=120)
    semantics: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_property(self) -> "PropertyCoverageRecipe":
        _safe_property(self.property)
        return self


class GovernedComputationResult(BaseModel):
    calculation_key: str
    recipe: str
    recipe_version: int = Field(ge=1)
    config_digest: str
    result_digest: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int | None = Field(default=None, ge=0)
    unit: str
    basis: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    cypher_executions: list[dict[str, Any]] = Field(default_factory=list)


class RecipeExecutor(Protocol):
    def __call__(
        self,
        calculation_key: str,
        config: BaseModel,
        handler: CypherQueryHandler,
        recipe_version: int,
    ) -> GovernedComputationResult: ...


@dataclass(frozen=True)
class _RecipeRegistration:
    config_model: type[BaseModel]
    executor: RecipeExecutor
    version: int


class ComputationRegistry:
    """Code-owned allowlist of governed computation recipe compilers."""

    def __init__(self) -> None:
        self._recipes: dict[str, _RecipeRegistration] = {}
        self._frozen = False

    def register(
        self,
        recipe: str,
        config_model: type[BaseModel],
        executor: RecipeExecutor,
        *,
        version: int = 1,
    ) -> None:
        if self._frozen:
            raise RuntimeError("The governed computation registry is frozen.")
        if not _KEY.fullmatch(recipe):
            raise ValueError(f"Unsafe recipe key: {recipe!r}.")
        if recipe in self._recipes:
            raise ValueError(f"Governed computation recipe {recipe!r} is already registered.")
        if version < 1:
            raise ValueError("Recipe versions must be positive integers.")
        self._recipes[recipe] = _RecipeRegistration(config_model, executor, version)

    def freeze(self) -> None:
        self._frozen = True

    def available_recipes(self) -> dict[str, int]:
        return {
            name: registration.version
            for name, registration in sorted(self._recipes.items())
        }

    def execute(
        self,
        *,
        calculation_key: str,
        governed_config: dict[str, Any],
        query: QueryFunction,
        allowed_sources: list[str],
    ) -> GovernedComputationResult:
        if not _KEY.fullmatch(calculation_key):
            raise ValueError(f"Unsafe calculation key: {calculation_key!r}.")
        if not allowed_sources or any(not str(source).strip() for source in allowed_sources):
            raise PermissionError("Governed computations require a non-empty authorized source scope.")
        recipe = str(governed_config.get("recipe") or "")
        registration = self._recipes.get(recipe)
        if registration is None:
            raise ValueError(
                f"Unknown governed computation recipe {recipe!r}; allowed recipes are "
                f"{sorted(self._recipes)}."
            )
        config = registration.config_model.model_validate(governed_config)
        handler = CypherQueryHandler(query, allowed_sources)
        result = registration.executor(
            calculation_key, config, handler, registration.version
        )
        if result.recipe != recipe or result.recipe_version != registration.version:
            raise RuntimeError("The governed computation executor returned inconsistent metadata.")
        return result.model_copy(update={"cypher_executions": handler.audit_log()})


def _filter_clauses(
    alias: str,
    filters: list[ExactFilter],
    parameter_prefix: str,
    parameters: dict[str, Any],
) -> list[str]:
    clauses: list[str] = []
    for index, item in enumerate(filters):
        parameter = f"{parameter_prefix}_filter_{index}"
        parameters[parameter] = list(item.values)
        expression = f"{alias}.`{item.property}` IN ${parameter}"
        clauses.append(expression if item.operator == "in" else f"NOT ({expression})")
    return clauses


def _first_populated_property(alias: str, property_names: list[str]) -> str:
    """Return the first non-null, non-blank governed grouping property.

    Revit exports commonly materialize an optional text property as ``""``. Cypher's
    ``coalesce`` treats that as a value, which can hide a later canonical fallback and
    create an empty group. Keep the compact expression for a single field, while
    treating blank text as absent when a governed fallback chain is configured.
    """
    expressions = [f"{alias}.`{property_name}`" for property_name in property_names]
    if len(expressions) == 1:
        return expressions[0]
    populated = [
        f"CASE WHEN {expression} IS NOT NULL "
        f"AND trim(toString({expression})) <> '' THEN {expression} END"
        for expression in expressions
    ]
    return "coalesce(" + ", ".join(populated) + ")"


def _component_branch(
    component: GroupedCountComponent,
    index: int,
    identity_namespace: str,
    parameters: dict[str, Any],
) -> str:
    prefix = f"component_{index}"
    parameters[f"{prefix}_key"] = component.key
    parameters["missing_group_label"] = "unassigned"
    where = [
        f"n.`{component.source_property}` IN $allowed_sources",
        f"n.`{component.identity_property}` IS NOT NULL",
        *_filter_clauses("n", component.filters, prefix, parameters),
    ]
    identity = f"toString(n.`{component.identity_property}`)"
    if identity_namespace == "component":
        identity = f"${prefix}_key + ':' + {identity}"
    start = f"MATCH (n:`{component.label}`) WHERE " + " AND ".join(where) + " "
    group_properties = [component.group_property, *component.group_fallback_properties]
    direct_expression = _first_populated_property("n", group_properties)
    if component.group_relationship is None:
        if component.cross_source_identity_join:
            peer_expression = _first_populated_property("peer", group_properties)
            return (
                start
                + f"OPTIONAL MATCH (peer:`{component.label}`) WHERE peer.`{component.source_property}` IN $allowed_sources "
                + f"AND peer <> n AND peer.`{component.identity_property}` = n.`{component.identity_property}` "
                + f"WITH n, coalesce({direct_expression}, "
                + peer_expression
                + ", "
                + "$missing_group_label) AS governed_group "
                + f"RETURN toString(governed_group) AS group_value, {identity} AS identity_key"
            )
        return (
            start
            + f"RETURN toString(coalesce({direct_expression}, "
            "$missing_group_label)) AS group_value, "
            f"{identity} AS identity_key"
        )
    relationship = component.group_relationship
    edge = f"[:`{relationship.relationship_type}`]"
    if relationship.direction == "outgoing":
        pattern = f"(n)-{edge}->(group_node:`{relationship.target_label}`)"
    elif relationship.direction == "incoming":
        pattern = f"(n)<-{edge}-(group_node:`{relationship.target_label}`)"
    else:
        pattern = f"(n)-{edge}-(group_node:`{relationship.target_label}`)"
    direct = direct_expression
    related = f"group_node.`{relationship.target_property}`"
    group_expression = (
        f"coalesce({related}, {direct})"
        if relationship.prefer_relationship else
        f"coalesce({direct}, {related})"
    )
    relationship_match = (
        start
        + f"OPTIONAL MATCH {pattern} "
        + f"WHERE group_node.`{component.source_property}` IN $allowed_sources "
    )
    if not component.cross_source_identity_join:
        return (
            relationship_match
            + f"WITH n, coalesce({group_expression}, $missing_group_label) AS governed_group "
            + f"RETURN toString(governed_group) AS group_value, {identity} AS identity_key"
        )
    if relationship.direction == "outgoing":
        peer_pattern = f"(peer)-{edge}->(peer_group_node:`{relationship.target_label}`)"
    elif relationship.direction == "incoming":
        peer_pattern = f"(peer)<-{edge}-(peer_group_node:`{relationship.target_label}`)"
    else:
        peer_pattern = f"(peer)-{edge}-(peer_group_node:`{relationship.target_label}`)"
    peer_direct = _first_populated_property("peer", group_properties)
    peer_related = f"peer_group_node.`{relationship.target_property}`"
    peer_expression = (
        f"coalesce({peer_related}, {peer_direct})"
        if relationship.prefer_relationship else
        f"coalesce({peer_direct}, {peer_related})"
    )
    return (
        relationship_match
        + f"OPTIONAL MATCH (peer:`{component.label}`) WHERE peer.`{component.source_property}` IN $allowed_sources "
        + f"AND peer <> n AND peer.`{component.identity_property}` = n.`{component.identity_property}` "
        + f"OPTIONAL MATCH {peer_pattern} "
        + f"WHERE peer_group_node.`{component.source_property}` IN $allowed_sources "
        + f"WITH n, coalesce({group_expression}, {peer_expression}, "
        + "$missing_group_label) AS governed_group "
        + f"RETURN toString(governed_group) AS group_value, {identity} AS identity_key"
    )


def _composite_union(
    config: CompositeGroupedCountRecipe,
) -> tuple[str, dict[str, Any]]:
    parameters: dict[str, Any] = {}
    branches = [
        _component_branch(component, index, config.identity_namespace, parameters)
        for index, component in enumerate(config.components)
    ]
    parameters["missing_group_label"] = config.missing_group_label
    return " UNION ALL ".join(branches), parameters


def _final_result(
    *,
    calculation_key: str,
    recipe: str,
    recipe_version: int,
    config: BaseModel,
    rows: list[dict[str, Any]],
    total_count: int | None,
    unit: str,
    basis: str,
    provenance: dict[str, Any],
    limitations: list[str] | None = None,
) -> GovernedComputationResult:
    config_payload = config.model_dump(mode="json")
    config_digest = _stable_hash(config_payload)
    stable_result = {
        "calculation_key": calculation_key,
        "recipe": recipe,
        "recipe_version": recipe_version,
        "config_digest": config_digest,
        "rows": rows,
        "total_count": total_count,
        "unit": unit,
    }
    return GovernedComputationResult(
        **stable_result,
        result_digest=_stable_hash(stable_result),
        basis=basis,
        provenance=provenance,
        limitations=limitations or [],
    )


def _execute_composite_grouped_count(
    calculation_key: str,
    raw_config: BaseModel,
    handler: CypherQueryHandler,
    recipe_version: int,
) -> GovernedComputationResult:
    config = CompositeGroupedCountRecipe.model_validate(raw_config)
    union, parameters = _composite_union(config)
    grouped_rows = handler.execute(
        "CALL () { " + union + " } "
        "RETURN group_value, count(DISTINCT identity_key) AS count "
        "ORDER BY group_value",
        parameters,
    )
    total_rows = handler.execute(
        "CALL () { " + union + " } "
        "RETURN count(DISTINCT identity_key) AS total_count",
        parameters,
    )
    if not total_rows or total_rows[0].get("total_count") is None:
        raise RuntimeError("The composite grouped-count query omitted its total population.")
    rows = [
        {
            "group": str(row.get("group_value") or ""),
            "count": int(row.get("count") or 0),
        }
        for row in grouped_rows
    ]
    total_count = int(total_rows[0]["total_count"])
    unassigned_count = sum(
        int(row["count"]) for row in rows if row["group"] == config.missing_group_label
    )
    limitations = []
    if unassigned_count:
        limitations.append(
            f"{unassigned_count} {config.counting_unit} remain in the "
            f"{config.missing_group_label!r} group after governed direct, relationship, "
            "and cross-source identity resolution."
        )
    return _final_result(
        calculation_key=calculation_key,
        recipe=config.recipe,
        recipe_version=recipe_version,
        config=config,
        rows=rows,
        total_count=total_count,
        unit=config.counting_unit,
        basis=config.semantics,
        provenance={
            "component_keys": [component.key for component in config.components],
            "group_unit": config.group_unit,
            "identity_namespace": config.identity_namespace,
            "cross_source_identity_join_components": [
                component.key for component in config.components
                if component.cross_source_identity_join
            ],
            "unassigned_count": unassigned_count,
            "scope": "authorized sources only",
        },
        limitations=limitations,
    )


def _scope_branch(
    entity: GovernedEntity,
    scope: NamedScope,
    index: int,
    parameters: dict[str, Any],
) -> str:
    prefix = f"scope_{index}"
    parameters[f"{prefix}_key"] = scope.key
    where = [
        f"n.`{entity.source_property}` IN $allowed_sources",
        f"n.`{entity.identity_property}` IS NOT NULL",
        *_filter_clauses("n", entity.filters, f"{prefix}_base", parameters),
        *_filter_clauses("n", scope.filters, prefix, parameters),
    ]
    return (
        f"MATCH (n:`{entity.label}`) WHERE " + " AND ".join(where) + " "
        f"RETURN ${prefix}_key AS scope_key, "
        f"count(DISTINCT n.`{entity.identity_property}`) AS count"
    )


def _execute_scope_comparison(
    calculation_key: str,
    raw_config: BaseModel,
    handler: CypherQueryHandler,
    recipe_version: int,
) -> GovernedComputationResult:
    config = ScopeComparisonRecipe.model_validate(raw_config)
    parameters: dict[str, Any] = {}
    branches = [
        _scope_branch(config.entity, scope, index, parameters)
        for index, scope in enumerate(config.scopes)
    ]
    raw_rows = handler.execute(
        "CALL () { " + " UNION ALL ".join(branches) + " } "
        "RETURN scope_key, count ORDER BY scope_key",
        parameters,
    )
    counts = {str(row.get("scope_key")): int(row.get("count") or 0) for row in raw_rows}
    missing_scopes = [scope.key for scope in config.scopes if scope.key not in counts]
    if missing_scopes:
        raise RuntimeError(
            "The scope-comparison query omitted configured scopes: "
            + ", ".join(missing_scopes)
            + "."
        )
    baseline = counts[config.baseline_scope]
    rows = []
    for scope in config.scopes:
        count = counts.get(scope.key, 0)
        rows.append({
            "scope": scope.key,
            "count": count,
            "difference_from_baseline": count - baseline,
            "percent_of_baseline": (
                round(100.0 * count / baseline, 6) if baseline else None
            ),
        })
    limitations = (
        ["Percent-of-baseline is undefined because the baseline population is zero."]
        if baseline == 0 else []
    )
    return _final_result(
        calculation_key=calculation_key,
        recipe=config.recipe,
        recipe_version=recipe_version,
        config=config,
        rows=rows,
        total_count=baseline,
        unit=config.counting_unit,
        basis=config.semantics,
        provenance={
            "baseline_scope": config.baseline_scope,
            "scope_keys": [scope.key for scope in config.scopes],
            "scope": "authorized sources only",
        },
        limitations=limitations,
    )


def _execute_property_coverage(
    calculation_key: str,
    raw_config: BaseModel,
    handler: CypherQueryHandler,
    recipe_version: int,
) -> GovernedComputationResult:
    config = PropertyCoverageRecipe.model_validate(raw_config)
    entity = config.entity
    parameters: dict[str, Any] = {
        "missing_values": list(config.missing_values),
    }
    where = [
        f"n.`{entity.source_property}` IN $allowed_sources",
        f"n.`{entity.identity_property}` IS NOT NULL",
        *_filter_clauses("n", entity.filters, "coverage", parameters),
    ]
    populated = (
        f"n.`{config.property}` IS NOT NULL "
        f"AND trim(toString(n.`{config.property}`)) <> '' "
        f"AND NOT n.`{config.property}` IN $missing_values"
    )
    raw_rows = handler.execute(
        f"MATCH (n:`{entity.label}`) WHERE " + " AND ".join(where) + " "
        f"RETURN count(DISTINCT n.`{entity.identity_property}`) AS candidate_count, "
        f"count(DISTINCT CASE WHEN {populated} "
        f"THEN n.`{entity.identity_property}` END) AS populated_count",
        parameters,
    )
    if not raw_rows or raw_rows[0].get("candidate_count") is None:
        raise RuntimeError("The property-coverage query omitted its candidate population.")
    candidate_count = int(raw_rows[0]["candidate_count"])
    populated_count = int(raw_rows[0].get("populated_count") or 0)
    if populated_count > candidate_count:
        raise RuntimeError("Property coverage cannot exceed its governed candidate population.")
    missing_count = candidate_count - populated_count
    fill_rate = (
        round(100.0 * populated_count / candidate_count, 6)
        if candidate_count else None
    )
    rows = [{
        "status": "populated",
        "count": populated_count,
    }, {
        "status": "missing",
        "count": missing_count,
    }]
    limitations = [
        "Property assignment coverage does not establish relationship-graph or physical "
        "connectivity; those require separately governed relationship evidence."
    ]
    if candidate_count == 0:
        limitations.append(
            "Fill rate is undefined because the governed candidate population is zero."
        )
    return _final_result(
        calculation_key=calculation_key,
        recipe=config.recipe,
        recipe_version=recipe_version,
        config=config,
        rows=rows,
        total_count=candidate_count,
        unit=config.counting_unit,
        basis=config.semantics,
        provenance={
            "property": config.property,
            "assignment_unit": config.assignment_unit,
            "candidate_count": candidate_count,
            "populated_count": populated_count,
            "missing_count": missing_count,
            "fill_rate_percent": fill_rate,
            "scope": "authorized sources only",
        },
        limitations=limitations,
    )


DEFAULT_COMPUTATION_REGISTRY = ComputationRegistry()
DEFAULT_COMPUTATION_REGISTRY.register(
    "composite_grouped_count",
    CompositeGroupedCountRecipe,
    _execute_composite_grouped_count,
)
DEFAULT_COMPUTATION_REGISTRY.register(
    "scope_comparison",
    ScopeComparisonRecipe,
    _execute_scope_comparison,
)
DEFAULT_COMPUTATION_REGISTRY.register(
    "property_coverage",
    PropertyCoverageRecipe,
    _execute_property_coverage,
)
DEFAULT_COMPUTATION_REGISTRY.freeze()


def execute_governed_computation(
    *,
    calculation_key: str,
    governed_config: dict[str, Any],
    query: QueryFunction,
    allowed_sources: list[str],
    registry: ComputationRegistry = DEFAULT_COMPUTATION_REGISTRY,
) -> GovernedComputationResult:
    """Execute one server-configured computation through an allowlisted compiler."""

    return registry.execute(
        calculation_key=calculation_key,
        governed_config=governed_config,
        query=query,
        allowed_sources=allowed_sources,
    )
