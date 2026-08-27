from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


VerificationStatus = Literal["verified", "partially_verified", "insufficient_evidence", "error"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    client_id: UUID
    project_id: UUID
    session_id: str | None = Field(default=None, max_length=200)
    request_id: str | None = Field(default=None, max_length=200)
    evaluation_run_id: str | None = Field(default=None, max_length=200)
    evaluation_case_index: int | None = None


class SemanticContract(BaseModel):
    name: str
    population: str
    entity_role: Literal[
        "physical_instance", "type_definition", "space_record", "curated_record", "mixed", "unknown"
    ] = "unknown"
    spatial_scope: str
    classification_rule: str
    measure: str
    measurement_basis: str
    aggregation: str
    unit: str
    identity_key: str
    inclusion_rules: list[str] = Field(default_factory=list)
    exclusion_rules: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AnswerRequirement(BaseModel):
    requirement_id: str
    description: str
    mandatory: bool = True


class InvestigationPlan(BaseModel):
    interpretation: str
    language: str = "English"
    answerable_from_graph: bool = True
    answer_requirements: list[AnswerRequirement] = Field(default_factory=list)
    selected_contract: SemanticContract
    alternative_contracts: list[SemanticContract] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class IndependentAnalysis(BaseModel):
    agent: str
    language: str
    direct_question: str
    candidates: list[SemanticContract] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class EvidenceArtifact(BaseModel):
    artifact_id: str
    tool: str
    purpose: str
    phase: str = "investigation"
    query: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0
    population: str = ""
    entity_role: str = ""
    spatial_scope: str = ""
    measurement_basis: str = ""
    aggregation: str = ""
    unit: str = ""
    identity_key: str = ""
    inclusion_rules: list[str] = Field(default_factory=list)
    exclusion_rules: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str
    statement: str
    artifact_ids: list[str] = Field(default_factory=list)
    population: str
    measurement_basis: str
    unit: str
    claim_kind: Literal["direct", "supporting", "inference", "unavailable"]
    requirement_ids: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    draft_answer: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    tested_interpretations: list[str] = Field(default_factory=list)
    rejected_interpretations: list[str] = Field(default_factory=list)


class ClaimAudit(BaseModel):
    claim_id: str
    supported: bool
    reasons: list[str] = Field(default_factory=list)
    supporting_artifact_ids: list[str] = Field(default_factory=list)
    counterexample_artifact_ids: list[str] = Field(default_factory=list)


class AuditResult(BaseModel):
    semantic_contract_supported: bool
    contract_issues: list[str] = Field(default_factory=list)
    claim_audits: list[ClaimAudit] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)
    needs_more_investigation: bool = False


class VerificationResult(BaseModel):
    status: VerificationStatus
    issues: list[str] = Field(default_factory=list)
    supported_artifact_ids: list[str] = Field(default_factory=list)
    needs_more_investigation: bool = False
    semantic_checks: list[dict[str, Any]] = Field(default_factory=list)
    claim_audits: list[ClaimAudit] = Field(default_factory=list)
    missing_requirement_ids: list[str] = Field(default_factory=list)


class AnswerReport(BaseModel):
    answer: str
    verification_status: VerificationStatus
    limitations: list[str] = Field(default_factory=list)
    stages_used: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    investigation_trace: list[str] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    semantic_checks: list[dict[str, Any]] = Field(default_factory=list)


class HealthReport(BaseModel):
    status: Literal["ok"] = "ok"
    neo4j_status: Literal["connected"] = "connected"
    model: str
    provider: str = "unknown"
    client_id: str
    project_id: str
    source_count: int
    element_count: int
    contract_version: str = "2.1"
