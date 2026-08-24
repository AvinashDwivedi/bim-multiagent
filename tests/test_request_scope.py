import os
import unittest
from unittest.mock import patch

from bim_context import Settings
from bim_context import BimContext
from bim_agents.graph_contract import load_graph_contract


class RequestScopeTests(unittest.TestCase):
    def test_explicit_scope_replaces_environment_scope(self):
        env = {
            "NEO4J_URI": "neo4j://localhost",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
            "BIM_CLIENT_ID": "11111111-1111-1111-1111-111111111111",
            "BIM_PROJECT_ID": "22222222-2222-2222-2222-222222222222",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env(
                client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
                project_id="858ef0f0-454a-11f1-8957-1fe1b101e373",
            )
        self.assertEqual(settings.client_id, "653fbe80-e4c5-11ed-95e8-fdb8a484b2c4")
        self.assertEqual(settings.project_id, "858ef0f0-454a-11f1-8957-1fe1b101e373")

    def test_scope_ids_are_required_as_a_pair(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            Settings.from_env(client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4")

    def test_scope_ids_must_be_safe_uuids(self):
        with self.assertRaisesRegex(ValueError, "must be a UUID"):
            Settings.from_env(client_id="../other", project_id="also-invalid")

    def test_source_resolution_falls_back_to_exact_source_node_scope(self):
        context = object.__new__(BimContext)
        context.settings = Settings(
            neo4j_uri="neo4j://localhost", neo4j_username="neo4j",
            neo4j_password="password", neo4j_database="neo4j",
            client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
            project_id="858ef0f0-454a-11f1-8957-1fe1b101e373",
        )
        calls = []

        def query(cypher, parameters):
            calls.append((cypher, parameters))
            return [] if len(calls) == 1 else [{"source": "scoped.ifc"}]

        context.query = query
        self.assertEqual(
            context.resolve_allowed_sources(load_graph_contract()), ["scoped.ifc"]
        )
        self.assertIn("source.client_id = $client_id", calls[1][0])
        self.assertEqual(calls[1][1]["project_id"], context.settings.project_id)


if __name__ == "__main__":
    unittest.main()
