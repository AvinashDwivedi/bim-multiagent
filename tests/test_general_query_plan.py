import unittest

from pydantic import ValidationError

from bim_agents.graph_contract import load_graph_contract
from bim_agents.tools import (
    BimFilter,
    BimQueryPlan,
    _format_space_rows,
    _space_list_headline,
    _validate_plan,
)


class GeneralQueryPlanTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_graph_contract()

    def test_accepts_general_filtered_aggregate(self):
        plan = BimQueryPlan(
            entity="elements",
            operation="sum",
            metric="area_m2",
            filters=[BimFilter(field="ifc_class", operator="equals", value="wall")],
        )
        _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_accepts_space_function_group_summary(self):
        plan = BimQueryPlan(
            entity="spaces",
            operation="group_summary",
            group_by="type",
            metric="area_m2",
            filters=[BimFilter(field="level", operator="equals", value="ground floor")],
        )
        _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_accepts_maximum_summed_area_by_level_with_gross_basis(self):
        plan = BimQueryPlan(
            entity="spaces",
            operation="maximum_group_sum",
            group_by="level",
            metric="area_m2",
            filters=[BimFilter(field="area_basis", operator="equals", value="BVO")],
        )
        _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_group_summary_requires_numeric_metric(self):
        plan = BimQueryPlan(
            entity="spaces",
            operation="group_summary",
            group_by="type",
            metric="name",
        )
        with self.assertRaisesRegex(ValueError, "not numeric"):
            _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_formats_filtered_spaces_as_technical_record_blocks(self):
        headline = _space_list_headline(
            8,
            8,
            [
                {"field": "type", "values": ["apartment"], "requested_values": ["apartments"]},
                {"field": "level", "values": ["07 zevende verdieping"], "requested_values": ["7th floor"]},
            ],
        )
        details = _format_space_rows([{
            "object_id": "6891",
            "ifc_class": "IfcSpace",
            "name": "Area:1952114",
            "level": "07 zevende verdieping",
            "area_m2": 58.261,
            "segment": "MSH",
            "owner": "Belegger",
            "room_count": "3 kamer",
        }])

        self.assertEqual(headline, "There are 8 apartments on the 7th floor.")
        self.assertEqual(
            details[0],
            "IfcSpace 6891\n"
            "  - Name: Area:1952114\n"
            "  - Level: 07 zevende verdieping\n"
            "  - Area: 58.26 m²\n"
            "  - Segment: MSH\n"
            "  - Owner: Belegger\n"
            "  - Room Count: 3 kamer",
        )

    def test_rejects_unknown_field_instead_of_accepting_cypher(self):
        plan = BimQueryPlan(
            entity="elements",
            operation="count",
            filters=[BimFilter(field="n.name) DELETE n", operator="equals", value="x")],
        )
        with self.assertRaisesRegex(ValueError, "Unknown field"):
            _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_rejects_numeric_aggregate_on_text(self):
        plan = BimQueryPlan(entity="elements", operation="sum", metric="name")
        with self.assertRaisesRegex(ValueError, "not numeric"):
            _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_project_graph_only_allows_count(self):
        plan = BimQueryPlan(entity="project_graph", operation="list")
        with self.assertRaisesRegex(ValueError, "only an unfiltered count"):
            _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_result_limit_is_bounded(self):
        with self.assertRaises(ValidationError):
            BimQueryPlan(entity="elements", operation="list", limit=1000)

    def test_record_details_are_an_explicit_plan_choice(self):
        self.assertFalse(BimQueryPlan(entity="spaces", operation="count").include_details)
        self.assertTrue(
            BimQueryPlan(entity="spaces", operation="count", include_details=True).include_details
        )


if __name__ == "__main__":
    unittest.main()
