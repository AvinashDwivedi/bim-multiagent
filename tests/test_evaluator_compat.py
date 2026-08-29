from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from bim_agent import BimAgent
from bim_agents.adapter import evaluator_payload

from conftest import final_response, tool_response


def test_evaluator_payload_reports_model_tools_without_fake_verification(
    sample_data: Path, fake_client_factory
) -> None:
    client = fake_client_factory(
        tool_response(1, "search_records", {"terms": ["Pipe"], "limit": 20}),
        tool_response(2, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        final_response(3, "There are 2 pipes. [ref: call-2]"),
    )
    report = BimAgent(sample_data, client=client).ask("How many pipes?")
    payload = evaluator_payload(report, client_id="client", project_id="project")
    assert payload["verification_status"] == "completed"
    assert payload["cost"] == report.cost
    assert "Tool: search_records" in payload["stages_used"]
    assert payload["semantic_checks"] == []
    assert payload["artifact_ids"]


def test_compatibility_http_chat(sample_data: Path, monkeypatch, fake_client_factory) -> None:
    monkeypatch.setenv("BIM_DATA_DIR", str(sample_data))
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(2, "reconcile_populations", {
            "populations": [{"label": "pipes", "object_ids": ["43", "44"]}],
        }),
        final_response(3, "There are 2 pipes. [ref: call-1] [ref: call-2]"),
    )
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    import bim_agents.webapp as webapp

    webapp._agent.cache_clear()
    response = TestClient(webapp.app).post(
        "/api/chat", json={"question": "How many pipes?", "project_id": "project"}
    )
    assert response.status_code == 200
    assert response.json()["verification_status"] == "completed"


def test_http_project_id_routes_through_projects_root(
    sample_data: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_PROJECTS_ROOT", str(sample_data.parent))
    monkeypatch.setenv("BIM_DATA_DIR", str(sample_data))
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(2, "reconcile_populations", {
            "populations": [{"label": "pipes", "object_ids": ["43", "44"]}],
        }),
        final_response(3, "There are 2 pipes. [ref: call-1] [ref: call-2]"),
    )
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    import bim_agents.webapp as webapp

    webapp._agent.cache_clear()
    http = TestClient(webapp.app)
    response = http.post(
        "/api/chat", json={"question": "How many pipes?", "project_id": sample_data.name}
    )
    assert response.status_code == 200
    assert "2" in response.json()["answer"]
    assert http.post("/api/chat", json={"question": "x", "project_id": "missing"}).status_code == 404


def test_compatibility_cli_uses_model_agent(monkeypatch, capsys) -> None:
    report = SimpleNamespace()
    payload = {
        "verification_status": "completed", "answer": "ok", "stages_used": [],
        "artifact_ids": [], "semantic_checks": [], "limitations": [],
        "investigation_trace": [], "failure_categories": [],
    }
    with patch("bim_agents.cli.BimAgent") as agent, patch(
        "bim_agents.cli.evaluator_payload", return_value=payload
    ):
        agent.return_value.ask.return_value = report
        from bim_agents.cli import main

        assert main(["question", "--quiet"]) == 0
        agent.assert_called_once_with(None)
    assert '"answer": "ok"' in capsys.readouterr().out
