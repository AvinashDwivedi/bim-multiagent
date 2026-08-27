from pathlib import Path

from bim_agents.anthropic_agent import BIMAgent, ReasoningPolicy
from bim_agents.config import Settings
from bim_agents.contracts import (
    AnswerRequirement,
    AuditResult,
    Claim,
    ClaimAudit,
    InvestigationResult,
    InvestigationPlan,
    SemanticContract,
    VerificationResult,
)
from bim_agents.evidence import EvidenceStore
from bim_agents.llm.provider import _openai_tool


def _set_graph_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j://example")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")


def test_openai_is_default_and_only_selected_key_is_required(monkeypatch):
    _set_graph_env(monkeypatch)
    monkeypatch.delenv("BIM_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings.from_env(Path("nonexistent-test.env"))
    assert settings.llm_provider == "openai"
    assert settings.agent_model == "gpt-5.6-sol"


def test_anthropic_can_be_selected_without_openai_key(monkeypatch):
    _set_graph_env(monkeypatch)
    monkeypatch.setenv("BIM_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env(Path("nonexistent-test.env"))
    assert settings.llm_provider == "anthropic"
    assert settings.agent_model.startswith("claude-")


def test_openai_tool_conversion_uses_responses_function_shape():
    source = {
        "name": "example",
        "description": "Example tool",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }
    converted = _openai_tool(source)
    assert converted == {
        "type": "function",
        "name": "example",
        "description": "Example tool",
        "parameters": source["input_schema"],
        "strict": True,
    }


def _claim(artifact_id: str) -> Claim:
    return Claim(
        claim_id="claim-1",
        statement="There are five physical instances.",
        artifact_ids=[artifact_id],
        population="Scoped physical instances",
        measurement_basis="Distinct GlobalID",
        unit="instances",
        claim_kind="direct",
    )


def test_gate_requires_both_independent_reviews_and_valid_citations():
    store = EvidenceStore(total_limit=2, phase_limits={"investigation": 1, "audit": 1})
    artifact = store.add(
        tool="cypher-query", purpose="count", columns=["count"], rows=[{"count": 5}],
        row_count=1, truncated=False, elapsed_ms=1,
    )
    investigation = InvestigationResult(draft_answer="Five", claims=[_claim(artifact.artifact_id)])
    supported = ClaimAudit(
        claim_id="claim-1", supported=True, supporting_artifact_ids=[artifact.artifact_id]
    )
    verification = VerificationResult(
        status="verified", claim_audits=[supported], supported_artifact_ids=[artifact.artifact_id]
    )
    rejected_audit = AuditResult(
        semantic_contract_supported=True,
        claim_audits=[ClaimAudit(claim_id="claim-1", supported=False, reasons=["Counterexample found"])],
    )
    gated = BIMAgent._enforce_gate(verification, investigation, rejected_audit, store)
    assert gated.status == "insufficient_evidence"
    assert not gated.claim_audits[0].supported

    accepted_audit = AuditResult(semantic_contract_supported=True, claim_audits=[supported])
    store.set_phase("audit")
    audit_artifact = store.add(
        tool="cypher-query", purpose="independent check", columns=["count"],
        rows=[{"count": 5}], row_count=1, truncated=False, elapsed_ms=1,
    )
    accepted_audit = AuditResult(
        semantic_contract_supported=True,
        claim_audits=[ClaimAudit(
            claim_id="claim-1", supported=True,
            supporting_artifact_ids=[audit_artifact.artifact_id],
        )],
    )
    gated = BIMAgent._enforce_gate(verification, investigation, accepted_audit, store)
    assert gated.status == "verified"
    assert gated.claim_audits[0].supported


def test_gate_downgrades_a_semantically_rejected_contract():
    store = EvidenceStore(total_limit=2, phase_limits={"investigation": 1, "audit": 1})
    artifact = store.add(tool="cypher-query", purpose="count", columns=["count"],
                         rows=[{"count": 5}], row_count=1, truncated=False, elapsed_ms=1)
    investigation = InvestigationResult(draft_answer="Five", claims=[_claim(artifact.artifact_id)])
    review = ClaimAudit(claim_id="claim-1", supported=True,
                        supporting_artifact_ids=[artifact.artifact_id])
    store.set_phase("audit")
    audit_artifact = store.add(tool="cypher-query", purpose="independent count",
                               columns=["count"], rows=[{"count": 5}], row_count=1,
                               truncated=False, elapsed_ms=1)
    verification = VerificationResult(status="verified", claim_audits=[review])
    audit_review = ClaimAudit(claim_id="claim-1", supported=True,
                              supporting_artifact_ids=[audit_artifact.artifact_id])
    audit = AuditResult(semantic_contract_supported=False, claim_audits=[audit_review],
                        contract_issues=["The population boundary is ambiguous."])
    gated = BIMAgent._enforce_gate(verification, investigation, audit, store)
    assert gated.status == "partially_verified"
    assert "population boundary" in " ".join(gated.issues)


def _plan(**updates):
    contract = SemanticContract(
        name="instances", population="Scoped elements", entity_role="physical_instance",
        spatial_scope="Authorized project", classification_rule="Recorded classification",
        measure="count", measurement_basis="Distinct GlobalID", aggregation="count distinct",
        unit="instances", identity_key="GlobalID",
    )
    values = {"interpretation": "Count scoped instances", "selected_contract": contract}
    values.update(updates)
    return InvestigationPlan(**values)


def test_reasoning_policy_escalates_semantic_ambiguity_and_repairs():
    policy = ReasoningPolicy(default="medium", high="high", low="low")
    assert policy.investigation(_plan(), is_repair=False) == "medium"
    assert policy.investigation(
        _plan(unresolved_terms=["Which floor convention?", "Which containment relation?"]),
        is_repair=False,
    ) == "high"
    assert policy.investigation(_plan(), is_repair=True) == "high"


def test_gate_downgrades_when_a_mandatory_answer_part_is_uncovered():
    store = EvidenceStore(total_limit=2, phase_limits={"investigation": 1, "audit": 1})
    evidence = store.add(
        tool="cypher-query", purpose="count", columns=["count"], rows=[{"count": 5}],
        row_count=1, truncated=False, elapsed_ms=1,
    )
    claim = _claim(evidence.artifact_id).model_copy(update={"requirement_ids": ["count"]})
    investigation = InvestigationResult(draft_answer="Five", claims=[claim])
    verifier = ClaimAudit(claim_id="claim-1", supported=True)
    store.set_phase("audit")
    audit_evidence = store.add(
        tool="cypher-query", purpose="audit", columns=["count"], rows=[{"count": 5}],
        row_count=1, truncated=False, elapsed_ms=1,
    )
    auditor = ClaimAudit(
        claim_id="claim-1", supported=True,
        supporting_artifact_ids=[audit_evidence.artifact_id],
    )
    result = VerificationResult(status="verified", claim_audits=[verifier])
    audit = AuditResult(semantic_contract_supported=True, claim_audits=[auditor])
    plan = _plan(answer_requirements=[
        AnswerRequirement(requirement_id="count", description="Give the count"),
        AnswerRequirement(requirement_id="area", description="Give total area"),
    ])
    gated = BIMAgent._enforce_gate(result, investigation, audit, store, plan)
    assert gated.status == "partially_verified"
    assert gated.missing_requirement_ids == ["area"]
