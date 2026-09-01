from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bim_agent.api import create_app

from conftest import final_response


def test_api_health_tools_and_answer(sample_data: Path, monkeypatch, fake_client_factory) -> None:
    client = fake_client_factory(final_response(1, "There are 2 pipes in the loaded snapshot."))
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: client)
    http = TestClient(create_app(sample_data))

    health = http.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_directed"] is True
    assert health.json()["source_count"] == 3

    tools = http.get("/api/tools")
    assert tools.status_code == 200
    assert tools.json()["built_in"] == ["shell", "web_search"]
    assert tools.json()["custom"] == []

    response = http.post("/api/ask", json={"question": "How many pipes?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "There are 2 pipes in the loaded snapshot."
    assert response.json()["agent_loop"]["finished"] is True
