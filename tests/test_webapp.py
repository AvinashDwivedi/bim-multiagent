import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from bim_agents.contracts import AnswerReport
from bim_agents.webapp import create_app


class FakeGraph:
    def verify_connectivity(self):
        return None

    def scope_summary(self, client_id, project_id):
        return {"source_count": 2, "element_count": 42}


class FakeAgent:
    def __init__(self):
        self.graph = FakeGraph()

    async def answer(self, *, question, client_id, project_id, sink=None, session_id=None):
        if sink:
            await sink(
                {
                    "type": "pipeline_stage",
                    "sequence": 1,
                    "event_id": "event-1",
                    "stage": "agent_start",
                    "agent": "Planner",
                }
            )
        return AnswerReport(answer=f"Answer to: {question}", verification_status="verified")


def client():
    app = create_app(
        agent=FakeAgent(),
        settings=SimpleNamespace(agent_model="test-model"),
    )
    return TestClient(app)


def test_health_contract():
    with client() as api:
        response = api.get(
            "/api/health", params={"client_id": str(uuid4()), "project_id": str(uuid4())}
        )
    assert response.status_code == 200
    assert response.json()["element_count"] == 42
    assert response.json()["contract_version"] == "2.1"


def test_chat_and_stream_contracts():
    payload = {
        "question": "Count walls",
        "client_id": str(uuid4()),
        "project_id": str(uuid4()),
    }
    with client() as api:
        response = api.post("/api/chat", json=payload)
        streamed = api.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert response.json()["verification_status"] == "verified"
    events = [json.loads(line) for line in streamed.text.splitlines()]
    assert events[0]["type"] == "pipeline_stage"
    assert events[-1]["type"] == "result"
