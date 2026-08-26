import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import (
    BimRunContext, BimTaskContract, ConstraintBinding, OutputSpec, ProjectScope,
    SemanticBoundary, SemanticIntent,
)
from bim_agents.schema_mapping import (
    RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal,
)
from bim_agents.tools import (
    BimFilter, BimQueryPlan, GeometryQueryPlan, _semantic_adequacy_checks,
)


def _mapping(*, counting_unit="residential floor plate"):
    return RegisteredSchemaMapping(
        mapping_id="mapping-test",
        proposal=SchemaMappingProposal(
            entity_name="model_records", label="IfcProduct",
            identity_property="GlobalID", source_property="source",
            fields=[
                SchemaFieldMapping(
                    semantic_name="type", property="Type",
                    ontology_kind="canonical_type",
                ),
                SchemaFieldMapping(
                    semantic_name="area_bvo", property="Area", data_type="number",
                    unit="m2",
                ),
                SchemaFieldMapping(semantic_name="material", property="Material"),
                SchemaFieldMapping(
                    semantic_name="level", property="Level", ontology_kind="level",
                ),
            ],
            counting_unit=counting_unit,
            counting_unit_evidence="One unique GlobalID represents one governed record.",
            reasoning_summary="Test mapping.",
        ),
        node_count=4, populated_identity_count=4, distinct_identity_count=4,
    )


def _context(intent: SemanticIntent):
    return BimRunContext(
        bim=object(),
        scope=ProjectScope(
            client_id="client", project_id="project", allowed_sources=["model.ifc"],
        ),
        graph_contract=load_graph_contract(),
        task_contract=BimTaskContract(
            goal="Answer one typed BIM output", operation="sum",
            entity_concept="model records", required_outputs=["answer"],
            output_specs=[OutputSpec(key="answer", semantic_intent=intent)],
            success_criteria=["The typed semantic intent and replay must pass."],
        ),
    )


class SemanticAdequacyTests(unittest.TestCase):
    def test_replayed_related_aggregate_cannot_answer_individual_measurement(self):
        context = _context(SemanticIntent(
            entity_grain="individual home",
            measurement_basis="gross floor area",
            population_boundary=[SemanticBoundary(
                semantic_field="type", exact_values=["apartment"],
            )],
            value_origin="planned",
            requested_projection=["comparison"],
        ))
        mapping = _mapping()
        plan = BimQueryPlan(
            entity="model_records", mapping_id=mapping.mapping_id,
            operation="sum", metric="area_bvo",
            filters=[BimFilter(field="type", operator="equals", value="residential_zone")],
            answer_key="answer", satisfies=["answer"],
            semantic_intent=SemanticIntent(value_origin="actual"),
        )

        checks = _semantic_adequacy_checks(
            context, plan,
            {"statement": "Fifteen floor plates.", "value": 15, "unit": "records"},
            registered=mapping, matched_count=15,
        )
        by_name = {item["name"]: item["passed"] for item in checks}

        self.assertFalse(by_name["entity_grain_matches"])
        # BVO is a gross-area metric, so the physical measurement dimension is
        # compatible even though the entity grain and classification are wrong.
        self.assertTrue(by_name["measurement_basis_matches"])
        self.assertFalse(by_name["population_complete"])
        self.assertFalse(by_name["planned_actual_distinguished"])
        self.assertFalse(by_name["projection_answers_question"])
        self.assertTrue(by_name["requested_outputs_present"])

    def test_existing_population_with_unpopulated_property_is_verified_as_missing_data(self):
        context = _context(SemanticIntent(
            entity_grain="tray segment",
            measurement_basis="material",
            population_boundary=[SemanticBoundary(
                semantic_field="type", exact_values=["cable tray"],
            )],
            absence_semantics="property_missing",
            requested_projection=["value", "unit", "coverage"],
        ))
        mapping = _mapping(counting_unit="tray segment")
        plan = BimQueryPlan(
            entity="model_records", mapping_id=mapping.mapping_id,
            operation="coverage", coverage_field="material",
            filters=[BimFilter(field="type", operator="equals", value="cable tray")],
            answer_key="answer", satisfies=["answer"],
        )
        claim = {
            "statement": "Four tray segments exist, but material is unpopulated.",
            "value": 0, "unit": "records",
            "coverage": {
                "candidate_count": 4, "evaluated_count": 4, "matched_count": 0,
                "missing_count": 4, "unknown_count": 0, "excluded_count": 0,
                "exhaustive": True,
            },
        }

        checks = _semantic_adequacy_checks(
            context, plan, claim, registered=mapping, matched_count=0,
        )

        self.assertTrue(all(item["passed"] for item in checks), checks)

    def test_legacy_contract_remains_backward_compatible(self):
        context = _context(SemanticIntent())
        mapping = _mapping()
        plan = BimQueryPlan(
            entity="model_records", mapping_id=mapping.mapping_id,
            operation="count", answer_key="answer", satisfies=["answer"],
        )

        checks = _semantic_adequacy_checks(
            context, plan, {"statement": "Four records.", "value": 4, "unit": "records"},
            registered=mapping, matched_count=4,
        )

        self.assertTrue(all(item["passed"] for item in checks))

    def test_supporting_calculation_may_intentionally_satisfy_no_direct_output(self):
        context = _context(SemanticIntent(value_origin="planned"))
        context.task_contract.semantic_intent = SemanticIntent(value_origin="planned")
        plan = GeometryQueryPlan(
            calculation="panel_assignment_coverage",
            role="supporting", include_in_answer=False, satisfies=[],
            semantic_intent=SemanticIntent(value_origin="planned"),
        )

        checks = _semantic_adequacy_checks(
            context, plan,
            {
                "statement": "Related assignment coverage was measured.",
                "value": 4,
                "unit": "records missing assignment",
                "coverage": {
                    "candidate_count": 10, "evaluated_count": 10,
                    "matched_count": 6, "missing_count": 4,
                    "unknown_count": 0, "excluded_count": 0,
                    "exhaustive": True,
                },
            },
            matched_count=1,
        )

        by_name = {item["name"]: item for item in checks}
        self.assertTrue(by_name["requested_outputs_present"]["passed"])
        self.assertIn("supporting-only", by_name["requested_outputs_present"]["explanation"])

    def test_architect_prose_matches_canonical_floor_area_evidence(self):
        context = _context(SemanticIntent(
            entity_grain=(
                "One aggregate area measurement for one individual physical floor/storey instance"
            ),
            measurement_basis="The governed gross floor area basis",
            population_boundary=[SemanticBoundary(
                semantic_field="floor", exact_values=["6th floor"],
            )],
            value_origin="actual",
            requested_projection=["value", "unit", "details", "coverage"],
        ))
        mapping = _mapping(counting_unit="residential BVO floor-plate record")
        plan = BimQueryPlan(
            entity="model_records", mapping_id=mapping.mapping_id,
            operation="sum", metric="area_bvo",
            filters=[BimFilter(field="level", operator="equals", value="06 zesde verdieping")],
            answer_key="answer", satisfies=["answer"],
            constraint_bindings=[ConstraintBinding(
                concept="floor", requested_value="6th floor", semantic_field="level",
                exact_values=["06 zesde verdieping"], mapping_id=mapping.mapping_id,
            )],
        )
        claim = {
            "statement": "The sixth-floor BVO area is 586.248 m².",
            "value": 586.248, "unit": "m²", "basis": "canonical gross area",
            "method": "sum", "coverage": {
                "candidate_count": 1, "evaluated_count": 1, "matched_count": 1,
                "missing_count": 0, "unknown_count": 0, "excluded_count": 0,
                "exhaustive": True,
            },
        }

        checks = _semantic_adequacy_checks(
            context, plan, claim, registered=mapping, matched_count=1,
        )

        self.assertTrue(all(item["passed"] for item in checks), checks)

    def test_grouping_field_changes_output_grain_from_segment_to_type(self):
        context = _context(SemanticIntent(
            entity_grain="Distinct cable-tray type identities",
            measurement_basis="classification values",
            requested_projection=["groups"],
        ))
        mapping = _mapping(counting_unit="physical cable-tray segment instance")
        plan = BimQueryPlan(
            entity="model_records", mapping_id=mapping.mapping_id,
            operation="distinct", group_by="type", answer_key="answer",
            satisfies=["answer"],
        )

        checks = _semantic_adequacy_checks(
            context, plan,
            {"statement": "Two tray types.", "value": 2, "details": ["A", "B"]},
            registered=mapping, matched_count=4,
        )

        by_name = {item["name"]: item["passed"] for item in checks}
        self.assertTrue(by_name["entity_grain_matches"])
        self.assertTrue(by_name["measurement_basis_matches"])
        self.assertTrue(by_name["projection_answers_question"])


if __name__ == "__main__":
    unittest.main()
