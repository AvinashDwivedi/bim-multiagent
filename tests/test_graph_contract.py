import unittest

import yaml
from pydantic import ValidationError

from bim_agents.graph_contract import GraphSchemaContract, load_graph_contract


class GraphContractTests(unittest.TestCase):
    def test_default_contract_maps_count_capability(self):
        contract = load_graph_contract()
        capability = contract.capability("count_elements_by_type_and_level")
        node = contract.node(capability.node_type)
        self.assertEqual(node.label, "BIMElement")
        self.assertEqual(capability.identity_property, "object_id")
        self.assertEqual(capability.source_property, "source")
        self.assertEqual(capability.type_property, "canonical_type")
        self.assertEqual(capability.level_property, "canonical_level")

        node_count = contract.capability("count_project_nodes")
        self.assertEqual(node_count.executor, "count_project_nodes")
        self.assertEqual(
            node_count.relationships,
            ["client_has_project", "project_has_bim", "hub_contains_ifc_project"],
        )

    def test_rejects_unsafe_schema_identifier(self):
        bad_contract = """
version: 1
node_types:
  bad: {label: "BIMElement`) DELETE n //", properties: []}
relationship_types: {}
authorization_path: {start_node: bad, relationships: [], source_node: bad, source_property: source}
capabilities: {}
"""
        with self.assertRaises(ValidationError):
            GraphSchemaContract.model_validate(yaml.safe_load(bad_contract))


if __name__ == "__main__":
    unittest.main()
