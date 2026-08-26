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
SpecialistRole: TypeAlias = Literal[
    "auto", "quantity", "relationship", "geometry", "requirements"
]
DependencyPolicy: TypeAlias = Literal[
    "auto", "all_success", "allow_partial", "independent"
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


class SemanticBoundary(BaseModel):
    """Canonical, query-addressable part of the requested population boundary."""

    semantic_field: str = Field(min_length=1)
    exact_values: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_values(self) -> "SemanticBoundary":
        if any(not value.strip() for value in self.exact_values):
            raise ValueError("Semantic boundary values must be non-blank.")
        if len({value.casefold().strip() for value in self.exact_values}) != len(self.exact_values):
            raise ValueError("Semantic boundary values cannot contain duplicates.")
        return self


class SemanticIntent(BaseModel):
    """Typed meaning of an output, independent of how a query happens to execute.

    Empty/``unspecified`` defaults preserve old stored contracts.  Once a field is
    supplied, deterministic verification treats it as part of the definition of
    done rather than trusting a replay-stable query to have the right meaning.
    """

    entity_grain: str = Field(
        default="", description="What one distinct result identity represents."
    )
    measurement_basis: str = Field(
        default="", description="Canonical metric/quantity being measured, not merely its unit."
    )
    population_boundary: list[SemanticBoundary] = Field(default_factory=list)
    value_origin: Literal["unspecified", "actual", "planned", "comparison"] = "unspecified"
    absence_semantics: Literal[
        "unspecified", "population_missing", "property_missing", "verified_zero", "unsupported"
    ] = "unspecified"
    requested_projection: list[Literal[
        "value", "unit", "details", "groups", "coverage", "comparison", "compliance"
    ]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_intent(self) -> "SemanticIntent":
        if self.entity_grain and not self.entity_grain.strip():
            raise ValueError("entity_grain must be non-blank when supplied.")
        if self.measurement_basis and not self.measurement_basis.strip():
            raise ValueError("measurement_basis must be non-blank when supplied.")
        if len(set(self.requested_projection)) != len(self.requested_projection):
            raise ValueError("requested_projection cannot contain duplicates.")
        return self

    @property
    def is_specified(self) -> bool:
        return bool(
            self.entity_grain.strip()
            or self.measurement_basis.strip()
            or self.population_boundary
            or self.value_origin != "unspecified"
            or self.absence_semantics != "unspecified"
            or self.requested_projection
        )


class OutputSpec(BaseModel):
    """Typed definition of one atomic answer output.

    ``required_outputs`` remains available as a list of strings for older agents and
    stored task contracts.  This model carries the semantics needed by deterministic
    completion gates without making prose labels do double duty as a schema.
    """

    key: str = Field(min_length=1)
    kind: Literal[
        "fact", "count", "measurement", "coverage", "compliance", "list",
        "grouped_summary", "relationship_coverage",
    ] = "fact"
    metric: str = ""
    grouping_dimensions: list[str] = Field(default_factory=list)
    scope: str = ""
    required_unit: str | None = None
    semantic_intent: SemanticIntent = Field(default_factory=SemanticIntent)


class ConstraintBinding(BaseModel):
    """Auditable binding from a package-owned request constraint to a query field."""

    concept: str
    requested_value: str
    semantic_field: str
    exact_values: list[str] = Field(default_factory=list)
    operator: Literal[
        "equals", "in", "greater_than", "greater_or_equal",
        "less_than", "less_or_equal", "exists", "is_missing",
    ] = "equals"
    mapping_id: str = ""
    applied_as: Literal[
        "filter", "entity_boundary", "metric", "grouping", "relationship", "scope"
    ] = "filter"
    package_id: str = ""


class EvidenceWorkPackage(BaseModel):
    """One independently schedulable evidence objective with a bounded context."""

    package_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")
    objective: str = Field(min_length=1)
    required_outputs: list[str] = Field(default_factory=list)
    output_specs: list[OutputSpec] = Field(default_factory=list)
    constraints: list[TaskConstraint] | None = Field(
        default=None,
        description=(
            "Constraints owned by this package. Null inherits contract constraints for "
            "legacy single-package tasks; an empty list explicitly means project scope only."
        ),
    )
    depends_on: list[str] = Field(default_factory=list)
    dependency_policy: DependencyPolicy = Field(
        default="auto",
        description=(
            "How dependency outcomes affect this package. Auto requires successful "
            "dependencies except for requirements-route availability work, which is "
            "independent. Compliance outputs always require successful dependencies."
        ),
    )
    route_hint: Literal["auto", "contract", "geometry", "live_schema", "requirements"] = "auto"
    specialist: SpecialistRole = Field(
        default="auto",
        description=(
            "Bounded execution job. Auto is resolved from typed outputs and route_hint; "
            "schema mapping remains a conditional preflight rather than a terminal job."
        ),
    )
    semantic_intent: SemanticIntent = Field(default_factory=SemanticIntent)

    @model_validator(mode="before")
    @classmethod
    def populate_output_labels(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("required_outputs") and data.get("output_specs"):
            data = dict(data)
            data["required_outputs"] = [
                item.key if isinstance(item, OutputSpec) else item.get("key")
                for item in data["output_specs"]
            ]
        return data

    @model_validator(mode="after")
    def validate_output_specs(self) -> "EvidenceWorkPackage":
        if not self.required_outputs or any(not item.strip() for item in self.required_outputs):
            raise ValueError("required_outputs must contain non-blank values.")
        if not self.output_specs:
            self.output_specs = [OutputSpec(key=item) for item in self.required_outputs]
        spec_keys = [item.key for item in self.output_specs]
        if len(set(spec_keys)) != len(spec_keys):
            raise ValueError("output_specs keys must be unique within a work package.")
        if len(spec_keys) != len(self.required_outputs):
            raise ValueError(
                "output_specs must correspond one-for-one with required_outputs in the same order."
            )
        return self


class BimTaskContract(BaseModel):
    """The pipeline's explicit definition of done for one BIM question."""

    goal: str
    operation: Literal[
        "count", "count_distinct", "list", "group_count", "group_summary", "distinct",
        "sum", "average", "minimum", "maximum", "maximum_group_sum", "coverage",
        "multi_group_count", "multi_group_summary", "relationship_coverage",
        "project_graph_count",
    ]
    entity_concept: str
    constraints: list[TaskConstraint] = Field(default_factory=list)
    questions_to_resolve: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(
        default_factory=list,
        description="Atomic facts/dimensions that a complete answer must verify.",
    )
    output_specs: list[OutputSpec] = Field(
        default_factory=list,
        description="Typed semantics for required_outputs; populated for legacy contracts.",
    )
    work_packages: list[EvidenceWorkPackage] = Field(
        default_factory=list,
        max_length=4,
        description="A small DAG of independently schedulable evidence objectives.",
    )
    success_criteria: list[str] = Field(min_length=1)
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    semantic_intent: SemanticIntent = Field(
        default_factory=SemanticIntent,
        description="Default semantic intent inherited by outputs that do not override it.",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_output_labels(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("required_outputs") and data.get("output_specs"):
            data = dict(data)
            data["required_outputs"] = [
                item.key if isinstance(item, OutputSpec) else item.get("key")
                for item in data["output_specs"]
            ]
        return data

    @model_validator(mode="after")
    def validate_work_packages(self) -> "BimTaskContract":
        if not self.required_outputs or any(not item.strip() for item in self.required_outputs):
            raise ValueError("required_outputs must contain non-blank values.")
        if len(set(self.required_outputs)) != len(self.required_outputs):
            raise ValueError("required_outputs cannot contain duplicates.")
        if not self.output_specs:
            self.output_specs = [OutputSpec(key=item) for item in self.required_outputs]
        spec_keys = [item.key for item in self.output_specs]
        if len(set(spec_keys)) != len(spec_keys):
            raise ValueError("output_specs keys must be unique.")
        if len(spec_keys) != len(self.required_outputs):
            raise ValueError(
                "output_specs must correspond one-for-one with required_outputs in the same order."
            )
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
        dependencies = {item.package_id: set(item.depends_on) for item in self.work_packages}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(package_id: str) -> None:
            if package_id in visiting:
                raise ValueError("work package dependencies must form an acyclic graph.")
            if package_id in visited:
                return
            visiting.add(package_id)
            for dependency in dependencies[package_id]:
                visit(dependency)
            visiting.remove(package_id)
            visited.add(package_id)

        for package_id in dependencies:
            visit(package_id)
        assigned = [output for item in self.work_packages for output in item.required_outputs]
        if len(set(assigned)) != len(assigned) or set(assigned) != set(self.required_outputs):
            raise ValueError("work packages must partition required_outputs exactly once.")
        typed_by_output = dict(zip(self.required_outputs, self.output_specs, strict=True))
        for package in self.work_packages:
            expected_specs = [typed_by_output[output] for output in package.required_outputs]
            supplied_keys = [item.key for item in package.output_specs]
            expected_keys = [item.key for item in expected_specs]
            if (
                supplied_keys
                and supplied_keys != expected_keys
                and supplied_keys != package.required_outputs
            ):
                raise ValueError(
                    "work package output_specs must use the root contract's typed keys "
                    "for their corresponding required_outputs."
                )
            package.output_specs = [item.model_copy(deep=True) for item in expected_specs]
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
    # Keep an auditable state trace in addition to the current phase.  The
    # first entry makes the initial state explicit for both new and legacy
    # notebooks.
    phase_history: list[InvestigationPhase] = Field(default_factory=lambda: ["understand"])
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
    semantic_adequacy_verified: bool = True
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
        "population_coverage", "relationship_binding", "measurement_plausibility",
        "entity_grain_matches", "measurement_basis_matches", "population_complete",
        "planned_actual_distinguished", "absence_semantics_correct",
        "projection_answers_question", "requested_outputs_present",
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
    spec: OutputSpec | None = None
    package_id: str = ""


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
    work_package_id: str = ""
    constraint_bindings: list[ConstraintBinding] = Field(default_factory=list)
    semantic_intent: SemanticIntent = Field(default_factory=SemanticIntent)

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
    work_package_id: str = ""
    constraint_bindings: list[ConstraintBinding] = Field(default_factory=list)


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
    constraint_bindings: list[ConstraintBinding] = Field(default_factory=list)

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
    operator: Literal[
        "equals", "in", "greater_than", "greater_or_equal",
        "less_than", "less_or_equal", "exists", "is_missing",
    ] = "equals"
    mapping_id: str = ""
    applied_as: Literal[
        "filter", "entity_boundary", "metric", "grouping", "relationship", "scope"
    ] = "filter"


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
    constraint_bindings: list[ConstraintBinding] = Field(default_factory=list)
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
    supporting_claims: list[Claim] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    plausibility_flags: list[PlausibilityFlag] = Field(default_factory=list)


class WorkstreamDiagnostic(BaseModel):
    """Compact, client-safe account of one specialist workstream."""

    package_id: str
    specialist: str
    status: str
    attempts: int = Field(default=0, ge=0)
    typed_output_failures: int = Field(default=0, ge=0)
    recovery_strategy: str = ""
    required_outputs: list[str] = Field(default_factory=list)
    satisfied_outputs: list[str] = Field(default_factory=list)


class PipelineReport(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    supporting_claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stages_used: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    investigation_trace: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    plausibility_flags: list[PlausibilityFlag] = Field(default_factory=list)
    output_statuses: list[OutputStatus] = Field(default_factory=list)
    workstream_diagnostics: list[WorkstreamDiagnostic] = Field(default_factory=list)
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
    active_specialist: Literal[
        "runtime", "quantity", "relationship", "geometry", "requirements"
    ] = "runtime"
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
    workstream_diagnostics: list[WorkstreamDiagnostic] = field(default_factory=list)
    completion_status: Literal["ready_for_verification", "insufficient_evidence"] | None = None
    notebook: InvestigationNotebook = field(default_factory=InvestigationNotebook)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def add_evidence(self, item: Evidence) -> None:
        with self._lock:
            if item.evidence_id in self.evidence:
                raise ValueError(f"Evidence {item.evidence_id!r} already exists.")
            if item.workstream_id == "root" and self.workstream_id != "root":
                item = item.model_copy(update={"workstream_id": self.workstream_id})
            if not item.work_package_id and item.workstream_id != "root":
                item = item.model_copy(update={"work_package_id": item.workstream_id})
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
