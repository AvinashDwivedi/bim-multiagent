from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bim_agent.api import create_app

from conftest import final_response


def test_api_health_and_model_answer(sample_data: Path, monkeypatch, fake_client_factory) -> None:
    client = fake_client_factory(final_response(1, "There are two pipes."))
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    http = TestClient(create_app(sample_data))
    health = http.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_directed"] is True
    response = http.post("/api/ask", json={"question": "How many pipes?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "There are two pipes."
    assert response.json()["agent_loop"]["finished"] is True
