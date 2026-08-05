from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from bim_context import BimContext

from .graph_contract import GraphSchemaContract


ScalarValue: TypeAlias = str | int | float | bool | None


class ProjectScope(BaseModel):
    client_id: str
    project_id: str
    allowed_sources: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    statement: str
    value: ScalarValue = None
    unit: str | None = None
    basis: str
    evidence_ids: list[str] = Field(default_factory=list)
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


class SupervisorReport(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)
    verification_status: Literal["verified", "insufficient_evidence", "conflict"]


@dataclass
class BimRunContext:
    """Trusted state shared by one supervised BIM run."""

    bim: BimContext
    scope: ProjectScope
    graph_contract: GraphSchemaContract
    evidence: dict[str, Evidence] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    agent_starts: int = 0
    max_llm_calls: int = 12
    max_tool_calls: int = 20
    max_agent_starts: int = 8
    _lock: RLock = field(default_factory=RLock, repr=False)

    def add_evidence(self, item: Evidence) -> None:
        with self._lock:
            self.evidence[item.evidence_id] = item

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
