import unittest

import yaml
from pydantic import ValidationError

from bim_agents.graph_contract import GraphSchemaContract, load_graph_contract


class GraphContractTests(unittest.TestCase):
    def test_default_contract_maps_general_query_entities(self):
        contract = load_graph_contract()
        elements = contract.query_entity("elements")
        node = contract.node(elements.node_type)
        self.assertEqual(node.label, "IfcProduct")
        self.assertEqual(elements.identity_property, "object_id")
        self.assertEqual(elements.source_property, "source")
        self.assertEqual(elements.fields["type"].property, "canonical_type")
        self.assertEqual(elements.fields["level"].ontology_kind, "level")
        self.assertEqual(elements.fields["area_m2"].data_type, "number")

        spaces = contract.query_entity("spaces")
        self.assertEqual(contract.node(spaces.node_type).label, "IfcSpace")
        self.assertEqual(spaces.fields["type"].ontology_kind, "canonical_type")
        self.assertEqual(spaces.fields["area_basis"].property, "Bepalingsmethode")
        self.assertNotIn("segment", spaces.fields)
        self.assertIn("object_id", spaces.default_select)

        levels = contract.query_entity("levels")
        self.assertEqual(contract.node(levels.node_type).label, "IfcBuildingStorey")
        self.assertEqual(levels.fields["elevation_m"].property, "placement_z")

        graph = contract.query_entity("project_graph")
        self.assertEqual(graph.kind, "project_graph")

    def test_client_overlay_adds_deployment_specific_fields(self):
        contract = load_graph_contract(client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4")
        spaces = contract.query_entity("spaces")
        self.assertEqual(spaces.fields["segment"].property, "Segment")
        self.assertEqual(spaces.fields["owner"].property, "Afnemer")
        self.assertEqual(spaces.fields["room_count"].property, "Programma")

    def test_rejects_unsafe_schema_identifier(self):
        bad_contract = """
version: 1
node_types:
  bad: {label: "BIMElement`) DELETE n //", properties: []}
relationship_types: {}
authorization_path: {start_node: bad, relationships: [], source_node: bad, source_property: source}
query_entities: {}
"""
        with self.assertRaises(ValidationError):
            GraphSchemaContract.model_validate(yaml.safe_load(bad_contract))


if __name__ == "__main__":
    unittest.main()
