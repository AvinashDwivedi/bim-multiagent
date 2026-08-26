from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from bim_context import BimContext


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "graph_schema.yaml"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALIDATED: set[tuple[str, str, str, int, bool]] = set()
_VALIDATION_LOCK = Lock()


class NodeType(BaseModel):
    label: str
    identity_property: str | None = None
    properties: list[str] = Field(default_factory=list)


class QueryField(BaseModel):
    property: str
    data_type: Literal["string", "number"] = "string"
    description: str
    unit: str | None = None
    source_unit: str | None = None
    conversion_factor: float = Field(default=1.0, gt=0)
    conversion_basis: str = "identity conversion"
    ontology_kind: Literal[
        "canonical_type", "ifc_class", "level", "aggregate_identity"
    ] | None = None

    @model_validator(mode="after")
    def validate_unit_conversion(self) -> "QueryField":
        if self.source_unit and self.unit and self.source_unit != self.unit and self.conversion_factor == 1:
            raise ValueError("Different source and canonical units require an explicit conversion factor.")
        return self


class QueryEntity(BaseModel):
    kind: Literal["records", "project_graph"] = "records"
    description: str
    node_type: str | None = None
    identity_property: str | None = None
    source_property: str | None = None
    classification_source_property: str | None = None
    classification_name_property: str | None = None
    default_select: list[str] = Field(default_factory=list)
    fields: dict[str, QueryField] = Field(default_factory=dict)


class RelationshipType(BaseModel):
    name: str
    from_node: str
    to_node: str


class AuthorizationPath(BaseModel):
    start_node: str
    relationships: list[str]
    source_node: str
    source_property: str


class GraphSchemaContract(BaseModel):
    version: int = Field(ge=1)
    level_aliases: dict[str, list[str]] = Field(default_factory=dict)
    node_types: dict[str, NodeType]
    relationship_types: dict[str, RelationshipType]
    authorization_path: AuthorizationPath
    query_entities: dict[str, QueryEntity]

    @model_validator(mode="after")
    def validate_references_and_identifiers(self) -> "GraphSchemaContract":
        for canonical_level, aliases in self.level_aliases.items():
            _assert_identifier(canonical_level)
            if not aliases or any(not str(alias).strip() for alias in aliases):
                raise ValueError(f"Level alias group {canonical_level!r} must contain non-empty values.")
        for node_key, node in self.node_types.items():
            _assert_identifier(node_key)
            _assert_identifier(node.label)
            for prop in [*node.properties, node.identity_property]:
                if prop:
                    _assert_identifier(prop)
        for rel_key, rel in self.relationship_types.items():
            _assert_identifier(rel_key)
            _assert_identifier(rel.name)
            if rel.from_node not in self.node_types or rel.to_node not in self.node_types:
                raise ValueError(f"Relationship {rel_key!r} references an unknown node type.")
        path = self.authorization_path
        if path.start_node not in self.node_types or path.source_node not in self.node_types:
            raise ValueError("Authorization path references an unknown node type.")
        for rel_key in path.relationships:
            if rel_key not in self.relationship_types:
                raise ValueError(f"Authorization path references unknown relationship {rel_key!r}.")
        current_node = path.start_node
        for rel_key in path.relationships:
            relationship = self.relationship_types[rel_key]
            if relationship.from_node != current_node:
                raise ValueError("Authorization relationships must form one contiguous directed path.")
            current_node = relationship.to_node
        if current_node != path.source_node:
            raise ValueError("Authorization path must end at its configured source node.")
        _assert_identifier(path.source_property)
        for entity_name, entity in self.query_entities.items():
            _assert_identifier(entity_name)
            if entity.kind == "records":
                if not all((entity.node_type, entity.identity_property, entity.source_property)):
                    raise ValueError(f"Record entity {entity_name!r} has an incomplete graph mapping.")
                if entity.node_type not in self.node_types:
                    raise ValueError(f"Entity {entity_name!r} references an unknown node type.")
                _assert_property(entity.identity_property)
                _assert_property(entity.source_property)
            for quality_property in (
                entity.classification_source_property,
                entity.classification_name_property,
            ):
                if quality_property:
                    _assert_property(quality_property)
            for field_name, field in entity.fields.items():
                _assert_identifier(field_name)
                _assert_property(field.property)
            unknown_defaults = set(entity.default_select) - set(entity.fields)
            if unknown_defaults:
                raise ValueError(
                    f"Entity {entity_name!r} has unknown default fields: {sorted(unknown_defaults)}"
                )
        return self

    def query_entity(self, name: str) -> QueryEntity:
        try:
            return self.query_entities[name]
        except KeyError as exc:
            raise KeyError(f"No registered BIM query entity named {name!r}.") from exc

    def node(self, name: str) -> NodeType:
        return self.node_types[name]


@dataclass(frozen=True)
class SchemaValidationReport:
    contract_version: int
    labels_checked: tuple[str, ...]
    relationships_checked: tuple[str, ...]
    properties_checked: tuple[str, ...]


def _assert_identifier(value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe Neo4j identifier in graph schema contract: {value!r}")


def _assert_property(value: str) -> None:
    if not value or "`" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"Unsafe Neo4j property in graph schema contract: {value!r}")


def _merge_contract(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply a client/project mapping overlay; lists replace base lists."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_contract(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_graph_contract(
    path: str | Path | None = None,
    *,
    client_id: str | None = None,
    project_id: str | None = None,
) -> GraphSchemaContract:
    contract_path = Path(path) if path else DEFAULT_CONTRACT_PATH
    with contract_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    overlay_root = contract_path.parent / "graph_schema_overlays"
    for kind, scope_id in (("clients", client_id), ("projects", project_id)):
        overlay_path = overlay_root / kind / f"{scope_id}.yaml" if scope_id else None
        if overlay_path and overlay_path.is_file():
            with overlay_path.open("r", encoding="utf-8") as stream:
                data = _merge_contract(data, yaml.safe_load(stream) or {})
    return GraphSchemaContract.model_validate(data)


def validate_live_schema(
    bim: BimContext,
    contract: GraphSchemaContract,
    *,
    use_cache: bool = True,
    authorization_only: bool = False,
) -> SchemaValidationReport:
    cache_key = (
        bim.settings.neo4j_uri,
        bim.settings.neo4j_database,
        bim.settings.project_id,
        contract.version,
        authorization_only,
    )
    with _VALIDATION_LOCK:
        cached = use_cache and cache_key in _VALIDATED
    if authorization_only:
        path = contract.authorization_path
        relationship_keys = path.relationships
        node_keys = {path.start_node, path.source_node}
        for key in relationship_keys:
            node_keys.add(contract.relationship_types[key].from_node)
            node_keys.add(contract.relationship_types[key].to_node)
        nodes = [contract.node_types[key] for key in node_keys]
        relationships = [contract.relationship_types[key] for key in relationship_keys]
        entities: list[QueryEntity] = []
    else:
        nodes = list(contract.node_types.values())
        relationships = list(contract.relationship_types.values())
        entities = list(contract.query_entities.values())
    required_labels = {node.label for node in nodes}
    required_relationships = {rel.name for rel in relationships}
    required_properties = {
        prop
        for node in nodes
        for prop in [*node.properties, node.identity_property]
        if prop
    } | {contract.authorization_path.source_property} | {
        prop
        for entity in entities
        for prop in [
            entity.identity_property,
            entity.source_property,
            entity.classification_source_property,
            entity.classification_name_property,
            *(field.property for field in entity.fields.values()),
        ]
        if prop
    }

    if not cached:
        live_labels = {row["label"] for row in bim.query("CALL db.labels() YIELD label RETURN label")}
        live_relationships = {
            row["relationshipType"]
            for row in bim.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        }
        live_properties = {
            row["propertyKey"]
            for row in bim.query("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")
        }
        missing_labels = required_labels - live_labels
        missing_relationships = required_relationships - live_relationships
        missing_properties = required_properties - live_properties
        errors = []
        if missing_labels:
            errors.append(f"missing labels: {sorted(missing_labels)}")
        if missing_relationships:
            errors.append(f"missing relationships: {sorted(missing_relationships)}")
        if missing_properties:
            errors.append(f"missing properties: {sorted(missing_properties)}")
        if errors:
            raise RuntimeError("Graph schema contract does not match Neo4j: " + "; ".join(errors))
        with _VALIDATION_LOCK:
            _VALIDATED.add(cache_key)

    return SchemaValidationReport(
        contract_version=contract.version,
        labels_checked=tuple(sorted(required_labels)),
        relationships_checked=tuple(sorted(required_relationships)),
        properties_checked=tuple(sorted(required_properties)),
    )
