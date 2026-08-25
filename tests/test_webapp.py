import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from bim_agents.models import PipelineReport
from bim_agents.webapp import app


class WebAppTests(unittest.TestCase):
    def test_chat_request_does_not_require_an_extra_request_query_parameter(self):
        schema = app.openapi()
        parameters = schema["paths"]["/api/chat"]["post"].get("parameters", [])
        self.assertNotIn("request", {item.get("name") for item in parameters})

    def test_root_exposes_api_metadata_not_a_ui(self):
        with TestClient(app) as client:
            index = client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertEqual(index.json()["docs"], "/docs")
        self.assertEqual(client.get("/assets/app.js").status_code, 404)

    def test_health_uses_authorization_only_schema_validation(self):
        fake_bim = SimpleNamespace(
            settings=SimpleNamespace(client_id="c", project_id="p"),
            connect=lambda: None,
            close=lambda: None,
            get_project_summary=lambda contract: {"allowed_sources": ["one.ifc"], "element_count": 12},
        )
        contract = SimpleNamespace(version=3)
        with (
            patch("bim_agents.webapp.BimContext", return_value=fake_bim),
            patch("bim_agents.webapp.Settings.from_env") as from_env,
            patch("bim_agents.webapp.load_graph_contract", return_value=contract),
            patch("bim_agents.webapp.validate_live_schema") as validate,
            TestClient(app) as client,
        ):
            response = client.get("/api/health", params={"client_id": "c", "project_id": "p"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_count"], 1)
        self.assertNotIn("project_id", response.json())
        validate.assert_called_once_with(fake_bim, contract, authorization_only=True)
        from_env.assert_called_once_with(client_id="c", project_id="p")

    def test_stream_emits_lifecycle_event_then_result(self):
        async def answer(question, hooks, client_id=None, project_id=None):
            hooks.stage("agent_start", agent="Supervisor", event_id="agent-1", sequence=1)
            self.assertEqual(client_id, "653fbe80-e4c5-11ed-95e8-fdb8a484b2c4")
            self.assertEqual(project_id, "858ef0f0-454a-11f1-8957-1fe1b101e373")
            return PipelineReport(answer="Five.", verification_status="verified")

        with patch("bim_agents.webapp.answer_bim_question", side_effect=answer):
            with TestClient(app) as client:
                response = client.post(
                    "/api/chat/stream", json={
                        "question": "How many apartments?",
                        "client_id": "653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
                        "project_id": "858ef0f0-454a-11f1-8957-1fe1b101e373",
                    }
                )
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]["stage"], "agent_start")
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["report"]["answer"], "Five.")

    def test_invalid_stream_request_returns_validation_error(self):
        with TestClient(app) as client:
            response = client.post("/api/chat/stream", json={"question": ""})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
