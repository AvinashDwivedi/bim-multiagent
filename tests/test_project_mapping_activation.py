from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.tools import PipelineContext, activate_project_knowledge_mapping


class FakeBim:
    def __init__(self, observed_values):
        self.observed_values = observed_values
        self.ontology = SimpleNamespace(bim_query_knowledge={
            "lighting_switches": {
                "entity_concept": "lighting switches",
                "label": "IfcBuildingElementProxy",
                "source_property": "source",
                "identity_property": "GlobalID",
                "category_property": "Category",
                "category_value": "Lighting Devices",
                "classification_property": "Family",
                "exact_family_values": ["Switch A", "Switch B"],
                "type_property": "Type",
                "counting_unit": "physical switch",
                "counting_unit_semantics": "GlobalID is unique per physical switch.",
            }
        })

    def query(self, cypher, parameters):
        return [{
            "node_count": 10,
            "populated_count": 10,
            "distinct_count": 10,
            "observed_values": self.observed_values,
        }]


class ProjectMappingActivationTests(unittest.TestCase):
    def context(self, observed_values):
        return BimRunContext(
            bim=FakeBim(observed_values),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )

    def test_governed_mapping_is_live_validated_and_registered(self):
        context = self.context(["Switch A", "Switch B"])
        observed = {
            "properties": ["source", "GlobalID", "Category", "Family", "Type"]
        }
        with patch("bim_agents.tools._observed_node_type", return_value=observed):
            registered = activate_project_knowledge_mapping(
                PipelineContext(context), "lighting_switches"
            )

        self.assertIn("mapping-", registered)
        mapping = next(iter(context.schema_mappings.values()))
        self.assertEqual(mapping.proposal.identity_property, "GlobalID")
        family_binding = next(
            item for item in mapping.proposal.value_bindings
            if item.semantic_name == "family"
        )
        self.assertEqual(
            [item.value for item in family_binding.matches],
            ["Switch A", "Switch B"],
        )

    def test_governed_mapping_fails_closed_when_exact_values_change(self):
        context = self.context(["Switch A"])
        observed = {
            "properties": ["source", "GlobalID", "Category", "Family", "Type"]
        }
        with patch("bim_agents.tools._observed_node_type", return_value=observed):
            with self.assertRaisesRegex(ValueError, "classification boundary changed"):
                activate_project_knowledge_mapping(
                    PipelineContext(context), "lighting_switches"
                )


if __name__ == "__main__":
    unittest.main()
