from types import SimpleNamespace
import json
import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, Evidence, ProjectScope
from bim_agents.schema_mapping import (
    RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal,
    SchemaValueBinding, SchemaValueMatch,
)
from bim_agents.tools import BimFilter, BimQueryPlan, _execute_plan, _verify_query_evidence


class _Bim:
    def __init__(self):
        self.queries = []
        self.ontology = SimpleNamespace()

    def query(self, cypher, parameters):
        self.queries.append((cypher, parameters))
        if "AS candidate_count" in cypher:
            return [{"candidate_count": 4, "populated_count": 0}]
        if "AS group_0" in cypher:
            return [{
                "group_0": "Tray", "group_1": "300.0", "group_2": "not modelled",
                "count": 4, "metric_value": 2.5, "metric_records": 4,
            }]
        if "AS total_records" in cypher:
            return [{"total_records": 4, "metric_records": 4}]
        if "AS evaluated_records" in cypher:
            return [{"value": 2.5, "matched_records": 1, "evaluated_records": 1}]
        raise AssertionError(cypher)


def _context():
    bim = _Bim()
    context = BimRunContext(
        bim=bim,
        scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
        graph_contract=load_graph_contract(),
    )
    proposal = SchemaMappingProposal(
        entity_name="tray_segments", label="IfcFlowSegment",
        identity_property="GlobalID", source_property="source",
        fields=[
            SchemaFieldMapping(
                semantic_name="category", property="Category", ontology_kind="canonical_type",
            ),
            SchemaFieldMapping(
                semantic_name="type", property="Type", ontology_kind="canonical_type",
            ),
            SchemaFieldMapping(
                semantic_name="length", property="Length", data_type="number",
                source_unit="cm", unit="m", conversion_factor=0.01,
                conversion_basis="centimetres to metres",
            ),
            SchemaFieldMapping(
                semantic_name="width", property="Width", data_type="number",
                source_unit="cm", unit="mm", conversion_factor=10,
                conversion_basis="centimetres to millimetres",
            ),
            SchemaFieldMapping(semantic_name="material", property="Material"),
        ],
        value_bindings=[
            SchemaValueBinding(
                semantic_name="category", property="Category", user_concept="tray category",
                matches=[SchemaValueMatch(value="Cable Trays", similarity=1)],
            ),
            SchemaValueBinding(
                semantic_name="type", property="Type", user_concept="tray type",
                matches=[SchemaValueMatch(value="Tray", similarity=1)],
            ),
        ],
        counting_unit="tray segment", counting_unit_evidence="GlobalID is unique.",
        reasoning_summary="test mapping",
    )
    mapping = RegisteredSchemaMapping(
        mapping_id="mapping-test", proposal=proposal, node_count=4,
        populated_identity_count=4, distinct_identity_count=4,
    )
    context.schema_mappings[mapping.mapping_id] = mapping
    return context, bim


def _filters():
    return [
        BimFilter(field="category", operator="equals", value="Cable Trays"),
        BimFilter(field="type", operator="equals", value="Tray"),
    ]


class MeasurementContractTests(unittest.TestCase):
    def test_aggregate_converts_once_and_retains_source_measurement(self):
        context, bim = _context()
        result = _execute_plan(context, BimQueryPlan(
            entity="tray_segments", mapping_id="mapping-test", operation="sum",
            metric="length", filters=_filters(),
        ))

        self.assertIn("* 0.01", bim.queries[0][0])
        self.assertEqual(result["claim"]["value"], 2.5)
        self.assertEqual(result["claim"]["unit"], "m")
        self.assertEqual(result["claim"]["measurement"]["source_value"], 250)
        self.assertEqual(result["claim"]["measurement"]["source_unit"], "cm")
        self.assertEqual(result["claim"]["measurement"]["canonical_unit"], "m")

    def test_coverage_proves_property_absence_over_full_population(self):
        context, _ = _context()
        result = _execute_plan(context, BimQueryPlan(
            entity="tray_segments", mapping_id="mapping-test", operation="coverage",
            coverage_field="material", filters=_filters(),
        ))

        coverage = result["claim"]["coverage"]
        self.assertEqual(coverage["candidate_count"], 4)
        self.assertEqual(coverage["matched_count"], 0)
        self.assertEqual(coverage["missing_count"], 4)
        self.assertTrue(coverage["exhaustive"])
        self.assertIn("0 of 4", result["claim"]["statement"])

    def test_nullable_group_dimension_becomes_explicit_bucket(self):
        context, bim = _context()
        result = _execute_plan(context, BimQueryPlan(
            entity="tray_segments", mapping_id="mapping-test",
            operation="multi_group_summary",
            group_by_fields=["type", "width", "material"], metric="length",
            filters=_filters(),
        ))

        grouping_query = bim.queries[0][0]
        self.assertNotIn("n.`Material` IS NOT NULL", grouping_query)
        self.assertIn("toFloat(n.`Width`) * 10.0", grouping_query)
        self.assertIn("width=300.0 mm", result["claim"]["details"][0])
        self.assertIn("material=not modelled", result["claim"]["details"][0])

    def test_exhaustive_zero_property_coverage_verifies_as_absence(self):
        context, _ = _context()
        plan = BimQueryPlan(
            entity="tray_segments", mapping_id="mapping-test", operation="coverage",
            coverage_field="material", filters=_filters(),
        )
        original = _execute_plan(context, plan)
        context.add_evidence(Evidence(
            evidence_id="coverage-material", kind="query", summary="coverage",
            payload=json.dumps(original),
        ))

        verification = _verify_query_evidence(context, ["coverage-material"])

        self.assertTrue(verification["verified"])
        self.assertEqual(verification["checks"][0]["diagnostics"], ["verified_absence"])
        population_check = next(
            item for item in verification["checks"][0]["semantic_checks"]
            if item["name"] == "population_coverage"
        )
        self.assertTrue(population_check["passed"])


if __name__ == "__main__":
    unittest.main()
