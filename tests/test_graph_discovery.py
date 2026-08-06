import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.tools import _project_graph_structure


class FakeBim:
    def query(self, cypher, parameters):
        if "MATCH (a)-[r]->(b)" in cypher:
            return [
                {
                    "from_labels": ["BIMHub"],
                    "relationship_type": "CONTAINS",
                    "to_labels": ["BIMElement", "IfcProject"],
                    "relationship_count": 1,
                    "property_key_groups": [],
                    "sample_names": [],
                },
                {
                    "from_labels": ["BIMElement", "IfcBuilding"],
                    "relationship_type": "CONTAINS",
                    "to_labels": ["BIMElement", "IfcBuildingStorey"],
                    "relationship_count": 3,
                    "property_key_groups": [["source"]],
                    "sample_names": [],
                },
            ]
        properties = [["name", "source"]] if "keys(n)" in cypher else []
        return [{
            "labels": ["BIMElement", "IfcProduct", "IfcWall"],
            "node_count": 4,
            "sample_names": ["External wall"],
            "property_key_groups": properties,
        }]


class GraphDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.context = BimRunContext(
            bim=FakeBim(),
            scope=ProjectScope(
                client_id="client",
                project_id="project",
                allowed_sources=["model.ifc"],
            ),
            graph_contract=load_graph_contract(),
        )

    def test_returns_unique_scoped_types_and_endpoint_aware_contract_mapping(self):
        result = _project_graph_structure(self.context, sample_limit=3)

        self.assertIn("IfcWall", result["unique_node_labels"])
        self.assertEqual(result["unique_relationship_names"], ["CONTAINS"])
        hub_pattern, building_pattern = result["relationship_types"]
        self.assertEqual(hub_pattern["contract_relationship_types"], ["hub_contains_ifc_project"])
        self.assertEqual(building_pattern["contract_relationship_types"], [])
        self.assertEqual(result["node_types"][0]["properties"], [])

    def test_can_focus_and_include_property_names(self):
        result = _project_graph_structure(
            self.context,
            include_properties=True,
            focus="wall",
        )

        self.assertEqual(result["node_type_count"], 1)
        self.assertEqual(result["node_types"][0]["properties"], ["name", "source"])
        self.assertEqual(result["relationship_pattern_count"], 0)


if __name__ == "__main__":
    unittest.main()
