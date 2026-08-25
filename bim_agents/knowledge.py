from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import BimRunContext, RunArtifact
from .schema_mapping import RegisteredSchemaMapping, SchemaMappingProposal


KnowledgeKind = Literal[
    "schema_mapping", "alias", "identity_rule", "relationship_path",
    "measurement_rule", "calculation_recipe",
]
KnowledgeStatus = Literal["candidate", "verified", "promoted", "rejected", "deprecated"]


class KnowledgeProposal(BaseModel):
    """A reusable interpretation proposal. Query answers never belong here."""

    kind: KnowledgeKind = "schema_mapping"
    concept: str = Field(min_length=1, max_length=160)
    mapping_id: str = ""
    aliases: list[str] = Field(default_factory=list, max_length=40)
    definition: str = Field(default="", max_length=1000)
    property_rules: list["KnowledgePropertyRule"] = Field(default_factory=list, max_length=40)
    relationship_path: list[str] = Field(default_factory=list, max_length=8)
    unit: str = Field(default="", max_length=80)
    basis: str = Field(default="", max_length=300)
    recipe_steps: list[str] = Field(default_factory=list, max_length=20)
    rationale: str = Field(default="", max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, min_length=1, max_length=40)
    confidence: float = Field(default=0.8, ge=0, le=1)


class KnowledgePropertyRule(BaseModel):
    semantic_name: str = Field(min_length=1, max_length=120)
    property: str = Field(min_length=1, max_length=200)


class LearnedKnowledgeRecord(BaseModel):
    knowledge_id: str
    kind: KnowledgeKind
    concept: str
    schema_fingerprint: str
    payload: dict[str, Any]
    evidence_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    status: KnowledgeStatus
    created_at: str
    updated_at: str
    validation_note: str = ""


class KnowledgeCurationReport(BaseModel):
    status: Literal["saved", "promoted", "no_change", "rejected"]
    knowledge_ids: list[str] = Field(default_factory=list)
    explanation: str


_FORBIDDEN_KEYS = {
    "answer", "answers", "answer_value", "actual_answer", "expected_answer",
    "expected_answers", "facts", "golden_values", "query_rows", "claim", "claims",
    "result", "result_value", "count", "total_count",
}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_knowledge_path() -> Path:
    configured = os.getenv("BIM_LEARNED_KNOWLEDGE_DB")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "storage" / "learned_knowledge.sqlite3"


def _reject_direct_answers(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        lowered = {str(key).casefold() for key in value}
        forbidden = lowered.intersection(_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(
                "Learned knowledge cannot store answer/result fields: "
                + ", ".join(sorted(forbidden))
            )
        if "statement" in lowered and "value" in lowered:
            raise ValueError("Learned knowledge cannot store a claim statement with its value.")
        for key, child in value.items():
            _reject_direct_answers(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_direct_answers(child, (*path, str(index)))


def _mapping_payload(mapping: RegisteredSchemaMapping) -> dict[str, Any]:
    """Persist semantics, never live cardinalities or answer values."""
    proposal = mapping.proposal.model_copy(deep=True)
    proposal.reasoning_summary = "Verified project-scoped schema interpretation."
    proposal.counting_unit_evidence = "Revalidate identity population and uniqueness before reuse."
    payload = {"proposal": proposal.model_dump(mode="json")}
    _reject_direct_answers(payload)
    return payload


def compute_live_schema_fingerprint(context: BimRunContext) -> str:
    """Fingerprint the scoped live label/property surface plus contract version."""
    source_property = context.graph_contract.authorization_path.source_property
    rows = context.bim.query(
        f"MATCH (n) WHERE n.`{source_property}` IN $allowed_sources "
        "UNWIND labels(n) AS label UNWIND keys(n) AS property "
        "RETURN label, collect(DISTINCT property) AS properties ORDER BY label",
        {"allowed_sources": context.scope.allowed_sources},
    )
    stable = {
        "contract_version": context.graph_contract.version,
        "nodes": [
            {
                "label": str(row.get("label") or ""),
                "properties": sorted(str(item) for item in row.get("properties") or []),
            }
            for row in rows
        ],
    }
    return hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class LearnedKnowledgeStore:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS learned_knowledge (
                    knowledge_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    schema_fingerprint TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    validation_note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS learned_knowledge_scope_status
                ON learned_knowledge(client_id, project_id, status, schema_fingerprint);
                """
            )
            connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row) -> LearnedKnowledgeRecord:
        return LearnedKnowledgeRecord(
            knowledge_id=row["knowledge_id"], kind=row["kind"], concept=row["concept"],
            schema_fingerprint=row["schema_fingerprint"],
            payload=json.loads(row["payload_json"]),
            evidence_ids=json.loads(row["evidence_ids_json"]),
            aliases=json.loads(row["aliases_json"]), confidence=float(row["confidence"]),
            status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
            validation_note=row["validation_note"],
        )

    def save_candidate(
        self, *, client_id: str, project_id: str, schema_fingerprint: str,
        proposal: KnowledgeProposal, payload: dict[str, Any],
    ) -> LearnedKnowledgeRecord:
        _reject_direct_answers(payload)
        stable = {
            "client_id": client_id, "project_id": project_id,
            "schema_fingerprint": schema_fingerprint, "kind": proposal.kind,
            "concept": proposal.concept.casefold(), "payload": payload,
        }
        knowledge_id = "knowledge-" + hashlib.sha256(
            json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        timestamp = _now()
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO learned_knowledge (
                    knowledge_id, client_id, project_id, schema_fingerprint, kind, concept,
                    payload_json, evidence_ids_json, aliases_json, confidence, status,
                    validation_note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', '', ?, ?)
                ON CONFLICT(knowledge_id) DO UPDATE SET
                    evidence_ids_json=excluded.evidence_ids_json,
                    aliases_json=excluded.aliases_json,
                    confidence=max(learned_knowledge.confidence, excluded.confidence),
                    updated_at=excluded.updated_at
                """,
                (
                    knowledge_id, client_id, project_id, schema_fingerprint,
                    proposal.kind, proposal.concept, json.dumps(payload, ensure_ascii=False),
                    json.dumps(list(dict.fromkeys(proposal.evidence_ids)), ensure_ascii=False),
                    json.dumps(list(dict.fromkeys(proposal.aliases)), ensure_ascii=False),
                    proposal.confidence, timestamp, timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM learned_knowledge WHERE knowledge_id = ?", (knowledge_id,)
            ).fetchone()
            connection.commit()
        return self._record(row)

    def transition(
        self, knowledge_id: str, *, client_id: str, project_id: str,
        from_statuses: tuple[KnowledgeStatus, ...], to_status: KnowledgeStatus,
        note: str,
    ) -> LearnedKnowledgeRecord:
        placeholders = ",".join("?" for _ in from_statuses)
        parameters: list[Any] = [to_status, note, _now(), knowledge_id, client_id, project_id]
        parameters.extend(from_statuses)
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                f"UPDATE learned_knowledge SET status=?, validation_note=?, updated_at=? "
                f"WHERE knowledge_id=? AND client_id=? AND project_id=? "
                f"AND status IN ({placeholders})",
                parameters,
            )
            if cursor.rowcount != 1:
                raise ValueError("Knowledge transition was rejected by its scope or lifecycle guard.")
            row = connection.execute(
                "SELECT * FROM learned_knowledge WHERE knowledge_id = ?", (knowledge_id,)
            ).fetchone()
            connection.commit()
        return self._record(row)

    def compatible(
        self, *, client_id: str, project_id: str, schema_fingerprint: str,
        statuses: tuple[KnowledgeStatus, ...] = ("promoted",),
    ) -> list[LearnedKnowledgeRecord]:
        placeholders = ",".join("?" for _ in statuses)
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM learned_knowledge WHERE client_id=? AND project_id=? "
                f"AND schema_fingerprint=? AND status IN ({placeholders}) "
                "ORDER BY confidence DESC, updated_at DESC",
                [client_id, project_id, schema_fingerprint, *statuses],
            ).fetchall()
        return [self._record(row) for row in rows]

    def deprecate_incompatible(
        self, *, client_id: str, project_id: str, schema_fingerprint: str,
    ) -> int:
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE learned_knowledge SET status='deprecated', "
                "validation_note='Live schema fingerprint changed.', updated_at=? "
                "WHERE client_id=? AND project_id=? AND schema_fingerprint<>? "
                "AND status IN ('verified','promoted')",
                (_now(), client_id, project_id, schema_fingerprint),
            )
            connection.commit()
            return int(cursor.rowcount)


def _latest_verified_checks(context: BimRunContext) -> dict[str, dict[str, Any]]:
    for evidence in reversed(list(context.evidence.values())):
        if evidence.kind != "verification":
            continue
        payload = json.loads(evidence.payload)
        return {
            str(check.get("evidence_id")): check
            for check in payload.get("checks") or []
            if check.get("evidence_id")
        }
    return {}


def save_knowledge_candidate(
    context: BimRunContext, proposal: KnowledgeProposal,
) -> LearnedKnowledgeRecord:
    if context.knowledge_store is None or not context.schema_fingerprint:
        raise ValueError("The learned-knowledge store is not configured for this run.")
    unknown = sorted(set(proposal.evidence_ids) - set(context.evidence))
    if unknown:
        raise ValueError("Knowledge references unknown evidence: " + ", ".join(unknown))
    if proposal.kind == "schema_mapping":
        mapping = context.schema_mappings.get(proposal.mapping_id)
        if not isinstance(mapping, RegisteredSchemaMapping):
            raise ValueError("Schema knowledge can only reference a registered live mapping.")
        payload = _mapping_payload(mapping)
    else:
        if proposal.mapping_id:
            raise ValueError("Only schema-mapping knowledge may carry a mapping_id.")
        payload = {
            "definition": proposal.definition,
            "property_rules": [item.model_dump(mode="json") for item in proposal.property_rules],
            "relationship_path": proposal.relationship_path,
            "unit": proposal.unit or None,
            "basis": proposal.basis or None,
            "recipe_steps": proposal.recipe_steps,
        }
        if not any(value for value in payload.values()):
            raise ValueError("Non-mapping knowledge requires a reusable definition or rule.")
        _reject_direct_answers(payload)
    record = context.knowledge_store.save_candidate(
        client_id=context.scope.client_id, project_id=context.scope.project_id,
        schema_fingerprint=context.schema_fingerprint, proposal=proposal,
        payload=payload,
    )
    context.learned_knowledge[record.knowledge_id] = record
    return record


def verify_and_promote_candidate(
    context: BimRunContext, knowledge_id: str, *, auto_promote: bool = True,
) -> LearnedKnowledgeRecord:
    record = context.learned_knowledge.get(knowledge_id)
    if not isinstance(record, LearnedKnowledgeRecord):
        raise ValueError("Knowledge candidate is not part of the active scoped run.")
    if record.status in {"verified", "promoted", "rejected", "deprecated"}:
        return record
    if record.schema_fingerprint != context.schema_fingerprint:
        raise ValueError("Knowledge candidate does not match the live schema fingerprint.")
    checks = _latest_verified_checks(context)
    relevant = [checks.get(evidence_id) for evidence_id in record.evidence_ids]
    if not relevant or any(check is None for check in relevant):
        return record
    if any(check.get("verified") is not True for check in relevant):
        rejected = context.knowledge_store.transition(
            knowledge_id, client_id=context.scope.client_id, project_id=context.scope.project_id,
            from_statuses=("candidate",), to_status="rejected",
            note="Referenced query evidence did not pass deterministic replay verification.",
        )
        context.learned_knowledge[knowledge_id] = rejected
        return rejected
    if any(
        (
            check.get("matched_count") is not None
            and int(check.get("matched_count") or 0) == 0
        )
        or bool(check.get("diagnostics"))
        for check in relevant
    ):
        rejected = context.knowledge_store.transition(
            knowledge_id, client_id=context.scope.client_id, project_id=context.scope.project_id,
            from_statuses=("candidate",), to_status="rejected",
            note="Zero-match or diagnostic-bearing evidence cannot be promoted as reusable semantics.",
        )
        context.learned_knowledge[knowledge_id] = rejected
        return rejected
    answer_evidence = any(
        (check.get("plan") or {}).get("role", "answer_producing") == "answer_producing"
        and (check.get("plan") or {}).get("include_in_answer", True) is not False
        for check in relevant
    )
    if not answer_evidence:
        return record
    minimum = float(os.getenv("BIM_KNOWLEDGE_PROMOTION_CONFIDENCE", "0.65"))
    if record.confidence < minimum:
        rejected = context.knowledge_store.transition(
            knowledge_id, client_id=context.scope.client_id, project_id=context.scope.project_id,
            from_statuses=("candidate",), to_status="rejected",
            note=f"Confidence {record.confidence:.3f} is below promotion threshold {minimum:.3f}.",
        )
        context.learned_knowledge[knowledge_id] = rejected
        return rejected
    verified = context.knowledge_store.transition(
        knowledge_id, client_id=context.scope.client_id, project_id=context.scope.project_id,
        from_statuses=("candidate",), to_status="verified",
        note="All referenced answer evidence passed deterministic replay verification.",
    )
    if record.kind != "schema_mapping":
        auto_promote = False
    if not auto_promote:
        context.learned_knowledge[knowledge_id] = verified
        return verified
    promoted = context.knowledge_store.transition(
        knowledge_id, client_id=context.scope.client_id, project_id=context.scope.project_id,
        from_statuses=("verified",), to_status="promoted",
        note="Automatically promoted only within the originating project and schema fingerprint.",
    )
    context.learned_knowledge[knowledge_id] = promoted
    return promoted


def curate_verified_mappings(context: BimRunContext) -> KnowledgeCurationReport:
    """Persist mappings; only mappings used by verified answer evidence are promoted."""
    if context.knowledge_store is None or not context.schema_fingerprint:
        return KnowledgeCurationReport(status="no_change", explanation="Knowledge learning is disabled.")
    if context.completion_status != "ready_for_verification" or context.failure_categories:
        return KnowledgeCurationReport(
            status="no_change",
            explanation="Only a completed, failure-free investigation may curate knowledge.",
        )
    checks = _latest_verified_checks(context)
    learned_ids: list[str] = []
    promoted = False
    for mapping_id, mapping in context.schema_mappings.items():
        if not isinstance(mapping, RegisteredSchemaMapping) or mapping_id.startswith("learned-"):
            continue
        evidence_ids = [
            evidence_id for evidence_id, check in checks.items()
            if (check.get("plan") or {}).get("mapping_id") == mapping_id
        ]
        if not evidence_ids:
            continue
        similarities = [item.similarity for item in mapping.proposal.match_evidence]
        confidence = min(similarities) if similarities else 0.8
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            kind="schema_mapping", concept=mapping.proposal.entity_name,
            mapping_id=mapping_id, evidence_ids=evidence_ids, confidence=confidence,
            rationale="Reusable live schema mapping backed by replayed project evidence.",
        ))
        learned_ids.append(candidate.knowledge_id)
        result = verify_and_promote_candidate(context, candidate.knowledge_id)
        promoted = promoted or result.status == "promoted"
        artifact_id = f"knowledge-{candidate.knowledge_id}"
        if artifact_id not in context.artifacts:
            context.add_artifact(RunArtifact(
                artifact_id=artifact_id, kind="knowledge_candidate",
                producer="Knowledge Curator",
                summary=f"{result.status.title()} reusable mapping for {candidate.concept}.",
                payload={
                    "knowledge_id": candidate.knowledge_id, "kind": candidate.kind,
                    "concept": candidate.concept, "status": result.status,
                    "confidence": candidate.confidence,
                },
            ))
    if not learned_ids:
        return KnowledgeCurationReport(
            status="no_change", explanation="No new registered mapping had replayed query evidence."
        )
    return KnowledgeCurationReport(
        status="promoted" if promoted else "saved", knowledge_ids=learned_ids,
        explanation=(
            "Promoted verified mappings within the originating project."
            if promoted else "Saved scoped candidates; answer-producing verification is still required."
        ),
    )


def activate_promoted_knowledge(context: BimRunContext) -> list[LearnedKnowledgeRecord]:
    """Revalidate promoted mappings against the live scoped graph before reuse."""
    if context.knowledge_store is None or not context.schema_fingerprint:
        return []
    store = context.knowledge_store
    store.deprecate_incompatible(
        client_id=context.scope.client_id, project_id=context.scope.project_id,
        schema_fingerprint=context.schema_fingerprint,
    )
    records = store.compatible(
        client_id=context.scope.client_id, project_id=context.scope.project_id,
        schema_fingerprint=context.schema_fingerprint,
        statuses=("promoted", "verified", "candidate"),
    )
    active: list[LearnedKnowledgeRecord] = []
    for record in records:
        if record.kind != "schema_mapping" or record.status != "promoted":
            context.learned_knowledge[record.knowledge_id] = record
            active.append(record)
            continue
        proposal = SchemaMappingProposal.model_validate(record.payload.get("proposal") or {})
        identifiers = {
            proposal.label, proposal.identity_property, proposal.source_property,
            *(field.property for field in proposal.fields),
        }
        if any(not identifier or "`" in identifier for identifier in identifiers):
            continue
        if not _SAFE_IDENTIFIER.fullmatch(proposal.label):
            continue
        rowset = context.bim.query(
            f"MATCH (n:`{proposal.label}`) WHERE n.`{proposal.source_property}` IN $allowed_sources "
            f"RETURN count(DISTINCT n) AS node_count, "
            f"count(DISTINCT CASE WHEN n.`{proposal.identity_property}` IS NOT NULL THEN n END) AS populated_count, "
            f"count(DISTINCT n.`{proposal.identity_property}`) AS distinct_count, "
            "collect(DISTINCT keys(n)) AS property_key_groups",
            {"allowed_sources": context.scope.allowed_sources},
        )
        row = rowset[0] if rowset else {}
        node_count = int(row.get("node_count") or 0)
        populated = int(row.get("populated_count") or 0)
        distinct = int(row.get("distinct_count") or 0)
        observed = {
            str(key) for group in row.get("property_key_groups") or [] for key in (group or [])
        }
        properties = identifiers - {proposal.label}
        binding_values_exist = True
        for binding in proposal.value_bindings:
            selected = [match.value.casefold().strip() for match in binding.matches]
            value_rows = context.bim.query(
                f"MATCH (n:`{proposal.label}`) "
                f"WHERE n.`{proposal.source_property}` IN $allowed_sources "
                f"AND toLower(trim(toString(n.`{binding.property}`))) IN $selected_values "
                "RETURN count(DISTINCT n) AS matching_count",
                {
                    "allowed_sources": context.scope.allowed_sources,
                    "selected_values": selected,
                },
            )
            if not value_rows or int(value_rows[0].get("matching_count") or 0) == 0:
                binding_values_exist = False
                break
        if (
            not node_count or populated != node_count or distinct != node_count
            or not properties <= observed or not binding_values_exist
        ):
            if record.status == "promoted":
                deprecated = store.transition(
                    record.knowledge_id,
                    client_id=context.scope.client_id,
                    project_id=context.scope.project_id,
                    from_statuses=("promoted",),
                    to_status="deprecated",
                    note="Live identity, property, or exact value-binding revalidation failed.",
                )
                context.learned_knowledge[record.knowledge_id] = deprecated
            continue
        mapping_id = "learned-" + record.knowledge_id.removeprefix("knowledge-")
        context.schema_mappings[mapping_id] = RegisteredSchemaMapping(
            mapping_id=mapping_id, proposal=proposal, node_count=node_count,
            populated_identity_count=populated, distinct_identity_count=distinct,
        )
        context.learned_knowledge[record.knowledge_id] = record
        context.add_artifact(RunArtifact(
            artifact_id=mapping_id, kind="mapping", producer="Knowledge Curator",
            summary=f"Revalidated promoted mapping for {record.concept}.",
            payload={
                "knowledge_id": record.knowledge_id, "mapping_id": mapping_id,
                "concept": record.concept, "status": "promoted",
            },
        ))
        active.append(record)
    return active
