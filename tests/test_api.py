from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bim_agent.api import create_app

from conftest import final_response, tool_response


def test_api_health_and_model_answer(sample_data: Path, monkeypatch, fake_client_factory) -> None:
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
    http = TestClient(create_app(sample_data))
    health = http.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_directed"] is True
    response = http.post("/api/ask", json={"question": "How many pipes?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "There are 2 pipes. [ref: call-1] [ref: call-2]"
    assert "cost" in response.json()
    assert response.json()["agent_loop"]["finished"] is True
