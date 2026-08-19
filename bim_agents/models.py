from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from bim_context import BimContext

from .graph_contract import GraphSchemaContract


ScalarValue: TypeAlias = str | int | float | bool | None


class ProjectScope(BaseModel):
    client_id: str
    project_id: str
    allowed_sources: list[str] = Field(default_factory=list)


class TaskConstraint(BaseModel):
    concept: str
    requested_value: str


class BimTaskContract(BaseModel):
    """The pipeline's explicit definition of done for one BIM question."""

    goal: str
    operation: Literal[
        "count", "count_distinct", "list", "group_count", "group_summary", "distinct",
        "sum", "average", "minimum", "maximum", "project_graph_count",
    ]
    entity_concept: str
    constraints: list[TaskConstraint] = Field(default_factory=list)
    questions_to_resolve: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)


class RunArtifact(BaseModel):
    artifact_id: str
    kind: Literal[
        "task_contract", "graph_discovery", "mapping", "query_plan",
        "cypher_query", "semantic_review",
    ]
    producer: Literal[
        "Pipeline", "Graph Inspector", "Schema Mapper", "Query Planner",
        "Cypher Query Handler", "Verifier",
    ]
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SemanticCheck(BaseModel):
    name: Literal[
        "replay_stability", "authorized_scope", "identity_integrity",
        "constraint_binding", "counting_unit", "boundary_exactness",
        "source_deduplication", "classification_purity", "constraint_coverage",
    ]
    passed: bool
    explanation: str


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


class Evidence(BaseModel):
    evidence_id: str
    kind: Literal["query", "ontology", "verification"]
    summary: str
    payload: str = "{}"


class BimQueryReport(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    status: Literal["verified", "needs_correction", "insufficient_evidence", "conflict"]
    verified_claims: list[Claim] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)


class PipelineReport(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stages_used: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    investigation_trace: list[str] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    verification_status: Literal["verified", "insufficient_evidence", "conflict"]


class InvestigationCompletion(BaseModel):
    """Small agent-loop terminator; the runtime builds the authoritative report."""

    status: Literal["ready_for_verification", "insufficient_evidence"]
    evidence_ids: list[str] = Field(default_factory=list)


@dataclass
class BimRunContext:
    """Trusted state shared by one BIM pipeline run."""

    bim: BimContext
    scope: ProjectScope
    graph_contract: GraphSchemaContract
    question: str = ""
    task_contract: BimTaskContract | None = None
    artifacts: dict[str, RunArtifact] = field(default_factory=dict)
    schema_mappings: dict[str, object] = field(default_factory=dict)
    schema_discovery: dict[str, object] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    agent_starts: int = 0
    agent_starts_by_name: dict[str, int] = field(default_factory=dict)
    max_llm_calls: int = 20
    max_tool_calls: int = 30
    max_agent_starts: int = 30
    max_starts_per_agent: int = 6
    runtime_limitations: list[str] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def add_evidence(self, item: Evidence) -> None:
        with self._lock:
            self.evidence[item.evidence_id] = item

    def add_artifact(self, item: RunArtifact) -> None:
        with self._lock:
            if item.artifact_id in self.artifacts:
                raise ValueError(f"Artifact {item.artifact_id!r} already exists.")
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
