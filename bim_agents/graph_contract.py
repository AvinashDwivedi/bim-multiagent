from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from bim_context import BimContext


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "graph_schema.yaml"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALIDATED: set[tuple[str, str, str, int]] = set()
_VALIDATION_LOCK = Lock()


class NodeType(BaseModel):
    label: str
    identity_property: str | None = None
    properties: list[str] = Field(default_factory=list)


class RelationshipType(BaseModel):
    name: str
    from_node: str
    to_node: str


class AuthorizationPath(BaseModel):
    start_node: str
    relationships: list[str]
    source_node: str
    source_property: str


class Capability(BaseModel):
    description: str
    executor: Literal["count_elements_by_type_and_level", "count_project_nodes"]
    node_type: str | None = None
    identity_property: str | None = None
    source_property: str
    type_property: str | None = None
    level_property: str | None = None
    relationships: list[str] = Field(default_factory=list)
    source_scope_required: bool = True


class GraphSchemaContract(BaseModel):
    version: int = Field(ge=1)
    level_aliases: dict[str, list[str]] = Field(default_factory=dict)
    node_types: dict[str, NodeType]
    relationship_types: dict[str, RelationshipType]
    authorization_path: AuthorizationPath
    capabilities: dict[str, Capability]

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
        _assert_identifier(path.source_property)
        for name, capability in self.capabilities.items():
            _assert_identifier(name)
            if capability.node_type is not None and capability.node_type not in self.node_types:
                raise ValueError(f"Capability {name!r} references an unknown node type.")
            for prop in (
                capability.identity_property,
                capability.source_property,
                capability.type_property,
                capability.level_property,
            ):
                if prop:
                    _assert_identifier(prop)
            if capability.executor == "count_elements_by_type_and_level" and not all((
                capability.node_type,
                capability.identity_property,
                capability.type_property,
                capability.level_property,
            )):
                raise ValueError(f"Capability {name!r} is missing its element-count graph mapping.")
            for rel_key in capability.relationships:
                if rel_key not in self.relationship_types:
                    raise ValueError(f"Capability {name!r} references unknown relationship {rel_key!r}.")
        return self

    def capability(self, name: str) -> Capability:
        try:
            return self.capabilities[name]
        except KeyError as exc:
            raise KeyError(f"No registered BIM graph capability named {name!r}.") from exc

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


def load_graph_contract(path: str | Path | None = None) -> GraphSchemaContract:
    contract_path = Path(path) if path else DEFAULT_CONTRACT_PATH
    with contract_path.open("r", encoding="utf-8") as stream:
        return GraphSchemaContract.model_validate(yaml.safe_load(stream) or {})


def validate_live_schema(
    bim: BimContext,
    contract: GraphSchemaContract,
    *,
    use_cache: bool = True,
) -> SchemaValidationReport:
    cache_key = (
        bim.settings.neo4j_uri,
        bim.settings.neo4j_database,
        bim.settings.project_id,
        contract.version,
    )
    with _VALIDATION_LOCK:
        cached = use_cache and cache_key in _VALIDATED
    required_labels = {node.label for node in contract.node_types.values()}
    required_relationships = {rel.name for rel in contract.relationship_types.values()}
    required_properties = {
        prop
        for node in contract.node_types.values()
        for prop in [*node.properties, node.identity_property]
        if prop
    } | {contract.authorization_path.source_property}

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
