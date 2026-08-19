from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SchemaFieldMapping(BaseModel):
    semantic_name: str
    property: str
    data_type: Literal["string", "number"] = "string"
    unit: str | None = None
    ontology_kind: Literal["canonical_type", "ifc_class", "level"] | None = None


class SchemaRelationshipStep(BaseModel):
    from_label: str
    relationship_type: str
    to_label: str
    direction: Literal["outgoing", "incoming"] = "outgoing"
    purpose: str = Field(description="Why this traversal is relevant to the user's question.")


class SchemaMatchEvidence(BaseModel):
    semantic_name: str
    property: str
    concept: str
    similarity: float = Field(ge=-1, le=1)


class SchemaValueMatch(BaseModel):
    value: str
    similarity: float = Field(ge=-1, le=1)


class SchemaValueBinding(BaseModel):
    semantic_name: str
    property: str
    user_concept: str
    matches: list[SchemaValueMatch] = Field(min_length=1)


class SchemaMappingProposal(BaseModel):
    entity_name: str
    label: str
    identity_property: str
    source_property: str
    relationship_path: list[SchemaRelationshipStep] = Field(
        default_factory=list,
        max_length=4,
        description="Observed path used to reach the mapped entity; the final step must end at label.",
    )
    fields: list[SchemaFieldMapping] = Field(default_factory=list)
    match_evidence: list[SchemaMatchEvidence] = Field(default_factory=list)
    value_bindings: list[SchemaValueBinding] = Field(default_factory=list)
    classification_source_property: str | None = None
    classification_name_property: str | None = None
    counting_unit: str = Field(default="", description="What one distinct identity represents.")
    counting_unit_evidence: str = Field(
        default="", description="Live evidence that the identity is the requested entity, not a child record."
    )
    reasoning_summary: str


class RegisteredSchemaMapping(BaseModel):
    mapping_id: str
    proposal: SchemaMappingProposal
    node_count: int = Field(ge=1)
    populated_identity_count: int = Field(ge=1)
    distinct_identity_count: int = Field(ge=1)


class SchemaMappingReport(BaseModel):
    status: Literal["registered", "unsupported"]
    mapping_id: str | None = None
    entity_name: str | None = None
    mapped_fields: list[str] = Field(default_factory=list)
    value_bindings: list[str] = Field(default_factory=list)
    relationship_path: list[str] = Field(default_factory=list)
    explanation: str
    limitations: list[str] = Field(default_factory=list)
