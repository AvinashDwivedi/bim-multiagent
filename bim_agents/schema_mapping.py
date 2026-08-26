from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SchemaFieldMapping(BaseModel):
    semantic_name: str
    property: str
    aliases: list[str] = Field(default_factory=list, max_length=20)
    data_type: Literal["string", "number"] = "string"
    unit: str | None = None
    source_unit: str | None = None
    conversion_factor: float = Field(default=1.0, gt=0)
    conversion_basis: str = "identity conversion"
    ontology_kind: Literal[
        "canonical_type", "ifc_class", "level", "aggregate_identity"
    ] | None = None

    @model_validator(mode="after")
    def validate_unit_conversion(self) -> "SchemaFieldMapping":
        if any(not alias.strip() for alias in self.aliases):
            raise ValueError("Schema field aliases must be non-blank.")
        if self.source_unit and self.unit and self.source_unit != self.unit and self.conversion_factor == 1:
            raise ValueError("Different source and canonical units require an explicit conversion factor.")
        return self


class SchemaRelationshipStep(BaseModel):
    from_label: str
    relationship_type: str
    to_label: str
    direction: Literal["outgoing", "incoming"] = "outgoing"
    purpose: str = Field(description="Why this traversal is relevant to the user's question.")


class SchemaRelationshipBinding(BaseModel):
    """A named, fixed, live-validated relationship path rooted at the mapped entity."""

    semantic_name: str
    steps: list[SchemaRelationshipStep] = Field(min_length=1, max_length=4)
    purpose: str
    evidence_kind: Literal[
        "unspecified", "physical_topology", "logical_assignment", "containment",
        "hosting", "system_membership",
    ] = "unspecified"
    target_identity_property: str = Field(
        default="",
        description=(
            "Stable identity on the final path node. Required for physical-topology "
            "coverage so a bare edge cannot be mistaken for a resolved connection."
        ),
    )
    target_cardinality: Literal["any", "at_most_one", "exactly_one"] = "any"

    @model_validator(mode="after")
    def validate_topology_contract(self) -> "SchemaRelationshipBinding":
        if self.evidence_kind == "physical_topology" and not self.target_identity_property:
            raise ValueError(
                "Physical-topology bindings require a stable target_identity_property."
            )
        if self.target_identity_property and (
            "`" in self.target_identity_property
            or any(ord(char) < 32 for char in self.target_identity_property)
        ):
            raise ValueError("Relationship target identity properties must be safe identifiers.")
        return self


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
    relationship_bindings: list[SchemaRelationshipBinding] = Field(default_factory=list)
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
