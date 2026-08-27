import pytest

from bim_agents.evidence import EvidenceBudgetExhausted, EvidenceStore


def _add(store: EvidenceStore, purpose: str):
    return store.add(
        tool="cypher-query", purpose=purpose, columns=["count"], rows=[{"count": 1}],
        row_count=1, truncated=False, elapsed_ms=1,
    )


def test_phase_limits_reserve_capacity_for_auditor():
    store = EvidenceStore(
        total_limit=3, phase_limits={"investigation": 2, "audit": 1}
    )
    _add(store, "first")
    _add(store, "second")
    with pytest.raises(EvidenceBudgetExhausted):
        _add(store, "third")
    assert store.remaining("audit") == 1
    store.set_phase("audit")
    audit = _add(store, "independent")
    assert audit.phase == "audit"
    assert len(store.artifacts) == 3


def test_identical_tool_requests_can_reuse_an_artifact():
    store = EvidenceStore(total_limit=1, phase_limits={"investigation": 1})
    request = {"query": "RETURN 1", "parameters_json": "{}", "purpose": "one"}
    key = store.request_key("run_readonly_cypher", request)
    artifact = _add(store, "one")
    store.remember(key, artifact)
    equivalent = {**request, "purpose": "same evidence, different wording"}
    assert store.cached(store.request_key("run_readonly_cypher", equivalent)) == artifact
