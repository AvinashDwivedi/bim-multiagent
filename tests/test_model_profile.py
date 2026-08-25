import unittest
from unittest.mock import patch

from bim_agents.graph_contract import load_graph_contract
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


if __name__ == "__main__":
    unittest.main()
