import unittest

from pydantic import ValidationError

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, BimTaskContract, ProjectScope, TaskConstraint
from bim_agents.tools import (
    BimFilter,
    BimQueryPlan,
    PipelineContext,
    _format_space_rows,
    _space_list_headline,
    _validate_answer_metadata,
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

    def test_accepts_distinct_physical_dwelling_identity(self):
        contract = load_graph_contract(
            client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
            project_id="858ef0f0-454a-11f1-8957-1fe1b101e373",
        )
        plan = BimQueryPlan(
            entity="spaces", operation="count_distinct", group_by="dwelling_unit_number",
            filters=[BimFilter(field="type", operator="equals", value="apartment")],
        )
        _validate_plan(contract.query_entity("spaces"), plan)

    def test_accepts_space_function_group_summary(self):
        plan = BimQueryPlan(
            entity="spaces",
            operation="group_summary",
            group_by="type",
            metric="area_m2",
            filters=[BimFilter(field="level", operator="equals", value="ground floor")],
        )
        _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_accepts_multi_dimensional_group_summary(self):
        plan = BimQueryPlan(
            entity="elements", operation="multi_group_summary",
            group_by_fields=["ifc_class", "level"], metric="area_m2",
            role="answer_producing", answer_key="area_by_class_and_level",
            satisfies=["class", "level", "total area"],
        )
        _validate_plan(self.contract.query_entity(plan.entity), plan)

    def test_multi_grouping_rejects_duplicate_dimensions(self):
        plan = BimQueryPlan(
            entity="elements", operation="multi_group_count",
            group_by_fields=["ifc_class", "ifc_class"],
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
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

    def test_answer_plan_cannot_drop_ground_floor_task_constraint(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=self.contract,
            task_contract=BimTaskContract(
                goal="Summarize ground-floor functions", operation="group_summary",
                entity_concept="spaces",
                constraints=[TaskConstraint(concept="floor", requested_value="ground floor")],
                required_outputs=["function counts and areas"],
                success_criteria=["Ground-floor filter is explicit"],
            ),
        )
        unfiltered = BimQueryPlan(
            entity="spaces", operation="group_summary", group_by="type", metric="area_m2",
            answer_key="ground-functions", satisfies=["function counts and areas"],
        )
        with self.assertRaisesRegex(ValueError, "omits required task constraints"):
            _validate_answer_metadata(PipelineContext(context), unfiltered)

        filtered = unfiltered.model_copy(update={
            "filters": [BimFilter(field="level", operator="equals", value="ground floor")]
        })
        _validate_answer_metadata(PipelineContext(context), filtered)

    def test_authorized_scope_is_governance_not_a_query_filter(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=self.contract,
            task_contract=BimTaskContract(
                goal="Count spaces", operation="count", entity_concept="spaces",
                constraints=[TaskConstraint(
                    concept="authorized scope", requested_value="configured BIM project"
                )],
                required_outputs=["space count"],
                success_criteria=["Scoped count"],
            ),
        )
        plan = BimQueryPlan(
            entity="spaces", operation="count", answer_key="spaces",
            satisfies=["space count"],
        )

        _validate_answer_metadata(PipelineContext(context), plan)


if __name__ == "__main__":
    unittest.main()
