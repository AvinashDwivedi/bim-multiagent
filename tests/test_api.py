from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bim_agent.api import create_app


def test_api_health_and_ask(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = TestClient(create_app(sample_data, use_llm=False))
    assert client.get("/api/health").status_code == 200
    response = client.post("/api/ask", json={"question": "how many pipes?"})
    assert response.status_code == 200
    assert response.json()["evidence"]["distinct_identity_count"] == 2
