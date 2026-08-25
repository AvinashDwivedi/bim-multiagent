from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bim_context import BimContext

from .graph_contract import GraphSchemaContract


ScalarValue: TypeAlias = str | int | float | bool | None
InvestigationPhase: TypeAlias = Literal[
    "understand", "explore", "analyze", "act", "observe", "verify", "respond", "blocked"
]


class ProjectScope(BaseModel):
    client_id: str
    project_id: str
    allowed_sources: list[str] = Field(default_factory=list)


class ModelNodeSignature(BaseModel):
    model_config = ConfigDict(frozen=True)

    labels: list[str]
    node_count: int = Field(ge=0)
    properties: list[str] = Field(default_factory=list)


class ModelRelationshipPattern(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    from_labels: list[str]
    to_labels: list[str]
    relationship_count: int = Field(ge=0)


class ProjectModelProfile(BaseModel):
    """Immutable routing cache; never authoritative evidence for answer values."""

    model_config = ConfigDict(frozen=True)

    profile_version: int = 1
    schema_fingerprint: str = ""
    authorized_source_count: int = Field(ge=0)
    node_types: list[ModelNodeSignature] = Field(default_factory=list)
    relationship_types: list[ModelRelationshipPattern] = Field(default_factory=list)
    scope_note: str
    freshness_note: str


class TaskConstraint(BaseModel):
    concept: str
    requested_value: str


class EvidenceWorkPackage(BaseModel):
    """One independently schedulable evidence objective with a bounded context."""

    package_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")
    objective: str = Field(min_length=1)
    required_outputs: list[str] = Field(min_length=1)
    constraints: list[TaskConstraint] | None = Field(
        default=None,
        description=(
            "Constraints owned by this package. Null inherits contract constraints for "
            "legacy single-package tasks; an empty list explicitly means project scope only."
        ),
    )
    depends_on: list[str] = Field(default_factory=list)
    route_hint: Literal["auto", "contract", "geometry", "live_schema", "requirements"] = "auto"


class BimTaskContract(BaseModel):
    """The pipeline's explicit definition of done for one BIM question."""

    goal: str
    operation: Literal[
        "count", "count_distinct", "list", "group_count", "group_summary", "distinct",
        "sum", "average", "minimum", "maximum", "maximum_group_sum", "coverage",
        "multi_group_count", "multi_group_summary", "project_graph_count",
    ]
    entity_concept: str
    constraints: list[TaskConstraint] = Field(default_factory=list)
    questions_to_resolve: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(
        min_length=1,
        description="Atomic facts/dimensions that a complete answer must verify.",
    )
    work_packages: list[EvidenceWorkPackage] = Field(
        default_factory=list,
        max_length=6,
        description="A small DAG of independently schedulable evidence objectives.",
    )
    success_criteria: list[str] = Field(min_length=1)
    complexity: Literal["simple", "moderate", "complex"] = "moderate"

    @model_validator(mode="after")
    def validate_work_packages(self) -> "BimTaskContract":
        if any(not item.strip() for item in self.required_outputs):
            raise ValueError("required_outputs cannot contain blank values.")
        if len(set(self.required_outputs)) != len(self.required_outputs):
            raise ValueError("required_outputs cannot contain duplicates.")
        if not self.work_packages:
            return self
        ids = [item.package_id for item in self.work_packages]
        if len(set(ids)) != len(ids):
            raise ValueError("work package IDs must be unique.")
        known = set(ids)
        for item in self.work_packages:
            unknown_dependencies = set(item.depends_on) - known
            if unknown_dependencies or item.package_id in item.depends_on:
                raise ValueError("work package dependencies must reference other package IDs.")
        assigned = [output for item in self.work_packages for output in item.required_outputs]
        if len(set(assigned)) != len(assigned) or set(assigned) != set(self.required_outputs):
            raise ValueError("work packages must partition required_outputs exactly once.")
        contract_constraints = {
            (item.concept.casefold(), item.requested_value.casefold())
            for item in self.constraints
        }
        for package in self.work_packages:
            for constraint in package.constraints or []:
                key = (constraint.concept.casefold(), constraint.requested_value.casefold())
                if key not in contract_constraints:
                    raise ValueError(
                        "work package constraints must be declared by the task contract."
                    )
        return self


class InvestigationObservation(BaseModel):
    action_id: str
    phase: InvestigationPhase = "observe"
    result_summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    unexpected_findings: list[str] = Field(default_factory=list)
    supports_hypotheses: list[str] = Field(default_factory=list)
    contradicts_hypotheses: list[str] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)


class InvestigationHypothesis(BaseModel):
    hypothesis_id: str
    statement: str
    expected_observation: str
    status: Literal["proposed", "testing", "supported", "rejected"] = "proposed"
    evidence_ids: list[str] = Field(default_factory=list)


class InvestigationAction(BaseModel):
    action_id: str
    phase: InvestigationPhase
    objective: str
    unresolved_question: str
    proposed_action: str
    expected_information_gain: str
    evidence_considered: list[str] = Field(default_factory=list)


class InvestigationNotebook(BaseModel):
    phase: InvestigationPhase = "understand"
    goal: str = ""
    required_outputs: list[str] = Field(default_factory=list)
    observations: list[InvestigationObservation] = Field(default_factory=list)
    hypotheses: list[InvestigationHypothesis] = Field(default_factory=list)
    rejected_interpretations: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    next_action: InvestigationAction | None = None
    verified_outputs: list[str] = Field(default_factory=list)


class CompletionGateReport(BaseModel):
    scope_verified: bool
    entity_verified: bool
    identity_verified: bool
    classification_verified: bool
    units_verified: bool
    required_outputs_verified: bool
    contradictions_resolved: bool
    replay_verified: bool
    ready_to_respond: bool
    missing: list[str] = Field(default_factory=list)


class RunArtifact(BaseModel):
    artifact_id: str
    kind: Literal[
        "task_contract", "graph_discovery", "mapping", "query_plan",
        "cypher_query", "semantic_review", "investigation_observation", "investigation_action",
        "knowledge_candidate",
    ]
    producer: Literal[
        "Pipeline", "Graph Inspector", "Schema Mapper", "Query Planner",
        "Cypher Query Handler", "Verifier", "Knowledge Curator",
    ]
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    workstream_id: str = "root"


class SemanticCheck(BaseModel):
    name: Literal[
        "replay_stability", "authorized_scope", "identity_integrity",
        "constraint_binding", "counting_unit", "boundary_exactness",
        "source_deduplication", "classification_purity", "constraint_coverage",
        "population_coverage", "measurement_plausibility",
    ]
    passed: bool
    explanation: str


class PopulationCoverage(BaseModel):
    """Machine-readable denominator for a BIM result or absence finding."""

    candidate_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    missing_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    excluded_count: int = Field(default=0, ge=0)
    exhaustive: bool = False

    @model_validator(mode="after")
    def validate_population(self) -> "PopulationCoverage":
        if self.evaluated_count + self.unknown_count > self.candidate_count:
            raise ValueError("Evaluated and unknown populations exceed the candidate population.")
        if self.matched_count > self.evaluated_count:
            raise ValueError("Matched population cannot exceed the evaluated population.")
        if self.matched_count + self.missing_count > self.evaluated_count:
            raise ValueError("Matched and missing populations exceed the evaluated population.")
        if self.exhaustive and self.unknown_count:
            raise ValueError("An exhaustive population cannot contain unknown records.")
        return self


class MeasurementMetadata(BaseModel):
    """Source-to-canonical measurement provenance; conversion is applied exactly once."""

    source_value: ScalarValue = None
    source_unit: str | None = None
    canonical_unit: str | None = None
    conversion_factor: float = Field(default=1.0, gt=0)
    conversion_basis: str = "identity conversion"
    source_property: str = ""


class PlausibilityFlag(BaseModel):
    code: str
    severity: Literal["warning", "error"]
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class OutputStatus(BaseModel):
    output: str
    status: Literal["verified", "unsupported", "conflict"]
    evidence_ids: list[str] = Field(default_factory=list)
    limitation: str = ""


class Claim(BaseModel):
    statement: str
    value: ScalarValue = None
    unit: str | None = None
    basis: str
    evidence_ids: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    total_count: int | None = Field(default=None, ge=0)
    displayed_count: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    coverage: PopulationCoverage | None = None
    measurement: MeasurementMetadata | None = None
    source_tags: list[str] = Field(default_factory=list)
    method: str = ""
    caveats: list[str] = Field(default_factory=list)
    plausibility_flags: list[PlausibilityFlag] = Field(default_factory=list)
    answer_key: str = ""
    satisfies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_display_population(self) -> "Claim":
        if (
            self.total_count is not None
            and self.displayed_count is not None
            and self.displayed_count > self.total_count
        ):
            raise ValueError("displayed_count cannot exceed total_count.")
        return self


class Evidence(BaseModel):
    evidence_id: str
    kind: Literal["query", "ontology", "verification"]
    summary: str
    payload: str = "{}"
    workstream_id: str = "root"


class BimQueryReport(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceWorkstreamResult(BaseModel):
    """Strict handoff from an isolated evidence worker to the orchestrator."""

    status: Literal["query_completed", "unsupported"]
    package_id: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "EvidenceWorkstreamResult":
        if self.status == "query_completed" and not self.evidence_ids:
            raise ValueError("A completed evidence workstream requires query evidence IDs.")
        if self.status == "unsupported" and not self.limitations:
            raise ValueError("An unsupported evidence workstream requires a precise limitation.")
        return self


class ResolvedConstraint(BaseModel):
    semantic_field: str
    requested_value: str
    exact_values: list[str] = Field(default_factory=list)


class EvidenceHandoff(BaseModel):
    """Compact scout artifact consumed by a fresh query-execution context."""

    package_id: str
    status: Literal["ready_for_query", "unsupported"]
    route: Literal["contract", "geometry", "live_mapping", "learned_mapping", "unsupported"]
    entity: str = ""
    mapping_id: str = ""
    calculation: str = ""
    operation: str = ""
    metric: str = ""
    grouping_fields: list[str] = Field(default_factory=list, max_length=3)
    constraints: list[ResolvedConstraint] = Field(default_factory=list)
    evidence_summary: str = ""
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "EvidenceHandoff":
        if self.status == "unsupported" and not self.limitations:
            raise ValueError("An unsupported scout handoff requires a precise limitation.")
        if self.status == "ready_for_query" and self.route == "unsupported":
            raise ValueError("A ready scout handoff requires an executable route.")
        if self.route in {"live_mapping", "learned_mapping"} and not self.mapping_id:
            raise ValueError("A mapped route requires a mapping_id.")
        if self.route == "geometry" and not self.calculation:
            raise ValueError("A geometry route requires a calculation key.")
        return self


class VerificationReport(BaseModel):
    status: Literal["verified", "needs_correction", "insufficient_evidence", "conflict"]
    verified_claims: list[Claim] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    plausibility_flags: list[PlausibilityFlag] = Field(default_factory=list)


class PipelineReport(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stages_used: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    investigation_trace: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    plausibility_flags: list[PlausibilityFlag] = Field(default_factory=list)
    output_statuses: list[OutputStatus] = Field(default_factory=list)
    verification_status: Literal["verified", "insufficient_evidence", "conflict"]


class InvestigationCompletion(BaseModel):
    """Small agent-loop terminator; the runtime builds the authoritative report."""

    status: Literal["ready_for_verification", "insufficient_evidence"]
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "InvestigationCompletion":
        if self.status == "ready_for_verification" and not self.evidence_ids:
            raise ValueError("Ready completion requires query evidence IDs.")
        if self.status == "insufficient_evidence" and not self.limitations:
            raise ValueError("Insufficient completion requires a precise limitation.")
        return self


@dataclass
class BimRunContext:
    """Trusted state shared by one BIM pipeline run."""

    bim: BimContext
    scope: ProjectScope
    graph_contract: GraphSchemaContract
    question: str = ""
    run_id: str = field(default_factory=lambda: uuid4().hex)
    workstream_id: str = "root"
    parent_workstream_id: str | None = None
    task_contract: BimTaskContract | None = None
    artifacts: dict[str, RunArtifact] = field(default_factory=dict)
    schema_mappings: dict[str, object] = field(default_factory=dict)
    schema_discovery: dict[str, object] = field(default_factory=dict)
    model_profile: ProjectModelProfile | None = None
    evidence: dict[str, Evidence] = field(default_factory=dict)
    knowledge_store: Any | None = None
    schema_fingerprint: str = ""
    learned_knowledge: dict[str, object] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    agent_starts: int = 0
    agent_starts_by_name: dict[str, int] = field(default_factory=dict)
    max_llm_calls: int = 48
    max_tool_calls: int = 60
    max_agent_starts: int = 30
    max_starts_per_agent: int = 6
    runtime_limitations: list[str] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    completion_status: Literal["ready_for_verification", "insufficient_evidence"] | None = None
    notebook: InvestigationNotebook = field(default_factory=InvestigationNotebook)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def add_evidence(self, item: Evidence) -> None:
        with self._lock:
            if item.evidence_id in self.evidence:
                raise ValueError(f"Evidence {item.evidence_id!r} already exists.")
            if item.workstream_id == "root" and self.workstream_id != "root":
                item = item.model_copy(update={"workstream_id": self.workstream_id})
            self.evidence[item.evidence_id] = item

    def add_artifact(self, item: RunArtifact) -> None:
        with self._lock:
            if item.artifact_id in self.artifacts:
                raise ValueError(f"Artifact {item.artifact_id!r} already exists.")
            if item.workstream_id == "root" and self.workstream_id != "root":
                item = item.model_copy(update={"workstream_id": self.workstream_id})
            self.artifacts[item.artifact_id] = item

    def consume_budget(self, counter: str, limit_name: str) -> int:
        with self._lock:
            value = getattr(self, counter) + 1
            setattr(self, counter, value)
            limit = getattr(self, limit_name)
            if value > limit:
                raise RuntimeError(
                    f"BIM run guardrail stopped execution: {counter} exceeded {limit}."
                )
            return value

    def consume_agent_start(self, name: str) -> tuple[int, int]:
        with self._lock:
            self.agent_starts += 1
            self.agent_starts_by_name[name] = self.agent_starts_by_name.get(name, 0) + 1
            if self.agent_starts > self.max_agent_starts:
                raise RuntimeError("Agent start budget exceeded.")
            if self.agent_starts_by_name[name] > self.max_starts_per_agent:
                raise RuntimeError(f"Agent start budget exceeded for {name}.")
            return self.agent_starts, self.agent_starts_by_name[name]
