import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from bim_agents.graph_contract import load_graph_contract
from bim_agents.knowledge import LearnedKnowledgeStore
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.tools import build_compact_model_profile


class ModelProfileTests(unittest.TestCase):
    def test_compact_profile_is_reused_without_live_rediscovery(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(
                client_id="profile-client", project_id="profile-project",
                allowed_sources=["model.ifc"],
            ),
            graph_contract=load_graph_contract(),
            schema_fingerprint="schema-v1",
        )
        structure = {
            "authorized_source_count": 1,
            "node_types": [{
                "labels": ["IfcFlowSegment"], "node_count": 12,
                "properties": ["GlobalID", "Length"],
            }],
            "relationship_types": [{
                "type": "CONTAINS", "from_labels": ["IfcBuildingStorey"],
                "to_labels": ["IfcFlowSegment"], "relationship_count": 12,
            }],
            "scope_note": "authorized",
        }
        with patch("bim_agents.tools._project_graph_structure", return_value=structure) as inspect:
            first = build_compact_model_profile(context)
            second = build_compact_model_profile(context)

        self.assertIs(first, second)
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(first.node_types[0].node_count, 12)
        self.assertIn("live exhaustive query", first.freshness_note)

    def test_compact_profile_persists_across_run_contexts(self):
        database = Path.cwd() / "tests" / f"profile-{uuid4().hex}.sqlite3"
        try:
            store = LearnedKnowledgeStore(database)
            structure = {
                "authorized_source_count": 1,
                "node_types": [{
                    "labels": ["IfcSpace"], "node_count": 2,
                    "properties": ["GlobalID", "source"],
                }],
                "relationship_types": [],
                "scope_note": "authorized",
            }
            contexts = [
                BimRunContext(
                    bim=object(),
                    scope=ProjectScope(
                        client_id="persistent-profile-client",
                        project_id="persistent-profile-project",
                        allowed_sources=["persistent.ifc"],
                    ),
                    graph_contract=load_graph_contract(),
                    schema_fingerprint="persistent-schema-v1",
                    knowledge_store=store,
                )
                for _ in range(2)
            ]
            with patch("bim_agents.tools._project_graph_structure", return_value=structure) as inspect:
                first = build_compact_model_profile(contexts[0])
                from bim_agents import tools
                tools._MODEL_PROFILE_CACHE.clear()
                second = build_compact_model_profile(contexts[1])

            self.assertEqual(inspect.call_count, 1)
            self.assertEqual(first, second)
        finally:
            database.unlink(missing_ok=True)

    def test_profile_without_fingerprint_is_not_reused_across_runs(self):
        def context():
            return BimRunContext(
                bim=object(),
                scope=ProjectScope(
                    client_id="uncached-client", project_id="uncached-project",
                    allowed_sources=["model.ifc"],
                ),
                graph_contract=load_graph_contract(),
            )

        structure = {
            "authorized_source_count": 1,
            "node_types": [],
            "relationship_types": [],
            "scope_note": "authorized",
        }
        with patch("bim_agents.tools._project_graph_structure", return_value=structure) as inspect:
            build_compact_model_profile(context())
            build_compact_model_profile(context())

        self.assertEqual(inspect.call_count, 2)


if __name__ == "__main__":
    unittest.main()
