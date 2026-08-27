from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from bim_agent import BimAgent
from bim_agents.adapter import evaluator_payload


def test_evaluator_payload_contract(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    report = BimAgent(sample_data, use_llm=False).ask("how many pipes?")
    payload = evaluator_payload(report, client_id="client", project_id="project")
    assert payload["verification_status"] == "verified"
    assert payload["answer"]
    assert payload["stages_used"]
    assert payload["artifact_ids"]
    assert payload["semantic_checks"]


def test_compatibility_http_chat(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_DATA_DIR", str(sample_data))
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    import bim_agents.webapp as webapp

    webapp._agent.cache_clear()
    client = TestClient(webapp.app)
    response = client.post(
        "/api/chat",
        json={"question": "how many pipes?", "client_id": "client", "project_id": "project"},
    )
    assert response.status_code == 200
    assert response.json()["verification_status"] == "verified"


def test_http_project_id_routes_through_projects_root(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_PROJECTS_ROOT", str(sample_data.parent))
    monkeypatch.setenv("BIM_DATA_DIR", str(sample_data))
    monkeypatch.setenv("BIM_USE_LLM", "false")
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    import bim_agents.webapp as webapp

    webapp._agent.cache_clear()
    client = TestClient(webapp.app)
    response = client.post(
        "/api/chat",
        json={"question": "how many pipes?", "project_id": sample_data.name},
    )
    assert response.status_code == 200
    assert "2" in response.json()["answer"]

    missing = client.post(
        "/api/chat", json={"question": "how many pipes?", "project_id": "missing-project"}
    )
    assert missing.status_code == 404


def test_compatibility_cli_does_not_force_llm(monkeypatch, capsys) -> None:
    report = SimpleNamespace()
    payload = {
        "verification_status": "verified", "answer": "ok", "stages_used": [],
        "artifact_ids": [], "semantic_checks": [], "limitations": [],
        "investigation_trace": [], "failure_categories": [],
    }
    with patch("bim_agents.cli.BimAgent") as agent, patch(
        "bim_agents.cli.evaluator_payload", return_value=payload
    ):
        agent.return_value.ask.return_value = report
        from bim_agents.cli import main

        assert main(["question", "--quiet"]) == 0
        agent.assert_called_once_with(None, use_llm=None)
    assert '"answer": "ok"' in capsys.readouterr().out
