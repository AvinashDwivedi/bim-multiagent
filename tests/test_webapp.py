import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from starlette.testclient import TestClient

from bim_agents.models import PipelineReport
from bim_agents.webapp import app


class WebAppTests(unittest.TestCase):
    def test_index_and_assets_are_served(self):
        with TestClient(app) as client:
            index = client.get("/")
            asset = client.get("/assets/app.js")
            self.assertEqual(index.status_code, 200)
            self.assertEqual(asset.status_code, 200)
            self.assertEqual(index.headers["cache-control"], "no-cache")
            self.assertEqual(asset.headers["cache-control"], "no-cache")
            self.assertIn('id="pipelineWorkflow"', index.text)
            self.assertIn('/assets/app.js?v=20260819-2', index.text)
            self.assertIn('fetch("/api/chat/stream"', asset.text)
            self.assertIn("report.investigation_trace", asset.text)
            self.assertEqual(client.get("/assets/unknown.js").status_code, 404)

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
            patch("bim_agents.webapp.Settings.from_env"),
            patch("bim_agents.webapp.load_graph_contract", return_value=contract),
            patch("bim_agents.webapp.validate_live_schema") as validate,
            TestClient(app) as client,
        ):
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_count"], 1)
        validate.assert_called_once_with(fake_bim, contract, authorization_only=True)

    def test_stream_emits_lifecycle_event_then_result(self):
        async def answer(question, hooks):
            hooks.stage("agent_start", agent="Supervisor", event_id="agent-1", sequence=1)
            return PipelineReport(answer="Five.", verification_status="verified")

        with patch("bim_agents.webapp.answer_bim_question", side_effect=answer):
            with TestClient(app) as client:
                response = client.post(
                    "/api/chat/stream", json={"question": "How many apartments?"}
                )
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]["stage"], "agent_start")
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["report"]["answer"], "Five.")

    def test_invalid_stream_request_returns_400(self):
        with TestClient(app) as client:
            response = client.post("/api/chat/stream", json={"question": ""})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
