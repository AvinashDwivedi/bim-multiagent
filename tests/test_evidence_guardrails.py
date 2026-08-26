import json
import unittest
from unittest.mock import patch

from bim_agents.graph_contract import load_graph_contract
from bim_agents.guardrails import pipeline_report_from_evidence, verification_report_from_evidence
from bim_agents.models import (
    BimRunContext, BimTaskContract, ConstraintBinding, Evidence,
    EvidenceWorkPackage, OutputSpec, ProjectScope, RunArtifact,
    WorkstreamDiagnostic,
)
from bim_agents.schema_mapping import (
    RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal,
    SchemaValueBinding, SchemaValueMatch,
)
from bim_agents.tools import (
    BimFilter, BimQueryPlan, _verify_query_evidence, ensure_bim_verification,
    ensure_compliance_evidence,
)


class EvidenceGuardrailTests(unittest.TestCase):
    def test_wrong_work_package_cannot_satisfy_another_packages_output(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Count and classify switches", operation="count",
                entity_concept="switches", required_outputs=["count", "types"],
                output_specs=[
                    OutputSpec(key="count", kind="count"),
                    OutputSpec(key="types", kind="grouped_summary"),
                ],
                work_packages=[
                    EvidenceWorkPackage(
                        package_id="counting", objective="Count switches",
                        required_outputs=["count"],
                    ),
                    EvidenceWorkPackage(
                        package_id="classification", objective="Classify switches",
                        required_outputs=["types"],
                    ),
                ],
                success_criteria=["Both outputs are verified"],
            ),
        )
        context.add_evidence(Evidence(
            evidence_id="q-wrong-owner", kind="query", summary="types from count worker",
            workstream_id="counting", work_package_id="counting",
        ))
        context.add_evidence(Evidence(
            evidence_id="verification-package-owner", kind="verification", summary="review",
            payload=json.dumps({"checks": [{
                "evidence_id": "q-wrong-owner", "verified": True,
                "plan": {"role": "answer_producing", "answer_key": "types", "satisfies": ["types"]},
                "claim": {"statement": "Two types.", "value": 2, "basis": "test"},
                "semantic_checks": [],
            }]}),
        ))

        report = pipeline_report_from_evidence(context)

        statuses = {item.output: item for item in report.output_statuses}
        self.assertEqual(report.verification_status, "insufficient_evidence")
        self.assertEqual(statuses["types"].status, "unsupported")
        self.assertEqual(statuses["types"].package_id, "classification")
        self.assertEqual(statuses["types"].spec.kind, "grouped_summary")

    def test_verified_claim_retains_package_constraint_bindings(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        binding = ConstraintBinding(
            concept="level", requested_value="ground floor", semantic_field="level",
            exact_values=["Ground Floor"], mapping_id="spaces", package_id="ground_floor",
        )
        context.add_evidence(Evidence(
            evidence_id="q-ground", kind="query", summary="ground floor count",
            workstream_id="ground_floor", work_package_id="ground_floor",
            constraint_bindings=[binding],
        ))
        context.add_evidence(Evidence(
            evidence_id="verification-ground", kind="verification", summary="review",
            payload=json.dumps({"checks": [{
                "evidence_id": "q-ground", "verified": True,
                "plan": {"role": "answer_producing", "answer_key": "count", "satisfies": []},
                "claim": {"statement": "Five spaces.", "value": 5, "basis": "test"},
                "semantic_checks": [],
            }]}),
        ))

        claim = verification_report_from_evidence(context).verified_claims[0]

        self.assertEqual(claim.work_package_id, "ground_floor")
        self.assertEqual(claim.constraint_bindings[0].semantic_field, "level")
        self.assertEqual(claim.constraint_bindings[0].exact_values, ["Ground Floor"])

    def test_zero_from_exact_live_mapping_fails_classification_purity(self):
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-switches",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcBuildingElementProxy",
                identity_property="GlobalID", source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="classification", property="OmniClass",
                    ontology_kind="canonical_type",
                )],
                value_bindings=[SchemaValueBinding(
                    semantic_name="classification", property="OmniClass",
                    user_concept="switches",
                    matches=[SchemaValueMatch(value="Switches", similarity=1.0)],
                )],
                counting_unit="physical switch",
                counting_unit_evidence="GlobalID is unique per instance.",
                reasoning_summary="Live mapping.",
            ),
            node_count=20, populated_identity_count=20, distinct_identity_count=20,
        )
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(), schema_mappings={mapping.mapping_id: mapping},
        )
        plan = BimQueryPlan(
            entity="switches", mapping_id=mapping.mapping_id, operation="count",
            filters=[BimFilter(field="classification", operator="equals", value="Switches")],
            answer_key="switch-count",
        )
        context.add_evidence(Evidence(
            evidence_id="query-zero", kind="query", summary="zero",
            payload=json.dumps({"plan": plan.model_dump(mode="json"), "result_digest": "same"}),
        ))
        rerun = {
            "plan": plan.model_dump(mode="json"), "result_digest": "same",
            "claim": {"statement": "0 switches", "value": 0, "basis": "test"},
            "matched_count": 0, "limitations": [],
        }

        with patch("bim_agents.tools._execute_plan", return_value=rerun):
            result = _verify_query_evidence(context, ["query-zero"])

        self.assertFalse(result["verified"])
        purity = next(
            check for check in result["checks"][0]["semantic_checks"]
            if check["name"] == "classification_purity"
        )
        self.assertFalse(purity["passed"])
        self.assertIn("unresolved mapping contradiction", purity["explanation"])
    def test_runtime_recovers_verification_from_valid_query_evidence(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )
        plan = BimQueryPlan(entity="spaces", operation="count")
        context.add_evidence(Evidence(
            evidence_id="query-valid",
            kind="query",
            summary="There are 5 apartments.",
            payload=json.dumps({
                "plan": plan.model_dump(mode="json"),
                "result_digest": "stable-digest",
                "claim": {
                    "statement": "There are 5 apartments.",
                    "value": 5,
                    "unit": "apartments",
                    "basis": "test",
                },
                "limitations": [],
            }),
        ))
        rerun = {
            "plan": plan.model_dump(mode="json"),
            "result_digest": "stable-digest",
            "claim": {
                "statement": "There are 5 apartments.",
                "value": 5,
                "unit": "apartments",
                "basis": "test",
            },
            "limitations": [],
        }

        with patch("bim_agents.tools._execute_plan", return_value=rerun):
            verification_id = ensure_bim_verification(context)

        self.assertIsNotNone(verification_id)
        self.assertEqual(pipeline_report_from_evidence(context).verification_status, "verified")
        self.assertEqual(pipeline_report_from_evidence(context).claims[0].value, 5)
        self.assertIn(f"review-{verification_id}", context.artifacts)

    def test_verification_replays_all_query_evidence_when_model_passes_no_ids(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )
        plan = BimQueryPlan(
            entity="spaces",
            operation="group_summary",
            group_by="type",
            metric="area_m2",
            filters=[BimFilter(field="level", operator="equals", value="ground floor")],
        )
        context.add_evidence(Evidence(
            evidence_id="query-functions",
            kind="query",
            summary="ground-floor function schedule",
            payload=json.dumps({
                "plan": plan.model_dump(mode="json"),
                "result_digest": "stable-digest",
            }),
        ))
        rerun = {
            "plan": plan.model_dump(mode="json"),
            "result_digest": "stable-digest",
            "claim": {"statement": "There are 10 function types.", "basis": "test"},
            "limitations": [],
        }

        with patch("bim_agents.tools._execute_plan", return_value=rerun):
            result = _verify_query_evidence(context, [])

        self.assertTrue(result["verified"])
        self.assertEqual(result["requested_evidence_ids"], [])
        self.assertEqual(result["verified_evidence_ids"], ["query-functions"])
        self.assertEqual(len(result["checks"]), 1)
        self.assertTrue(all(item["passed"] for item in result["checks"][0]["semantic_checks"]))

    def test_builds_general_verified_claims_from_replayed_evidence(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        claim = {
            "statement": "The sum area_m2 for project-scoped elements is 100.0 m².",
            "value": 100.0,
            "unit": "m²",
            "basis": "sum of populated project-scoped values",
            "details": ["IfcSpace 6891\n  - Area: 58.26 m²"],
        }
        context.add_evidence(Evidence(
            evidence_id="verification-general",
            kind="verification",
            summary="test general verification",
            payload=json.dumps({
                "verified": True,
                "checks": [{
                    "evidence_id": "query-general",
                    "verified": True,
                    "claim": claim,
                    "limitations": [],
                }],
            }),
        ))

        report = verification_report_from_evidence(context)

        self.assertEqual(report.status, "verified")
        self.assertEqual(report.verified_claims[0].statement, claim["statement"])
        self.assertEqual(report.verified_claims[0].value, 100.0)
        self.assertEqual(
            report.verified_claims[0].evidence_ids,
            ["query-general", "verification-general"],
        )
        pipeline_report = pipeline_report_from_evidence(context)
        self.assertEqual(pipeline_report.verification_status, "verified")
        self.assertEqual(pipeline_report.claims[0].value, 100.0)
        self.assertIn("IfcSpace 6891\n  - Area: 58.26 m²", pipeline_report.answer)

    def test_verified_exploration_is_not_rendered_as_an_answer_claim(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        exploration = BimQueryPlan(
            entity="spaces", operation="distinct", group_by="level", include_in_answer=False
        )
        answer = BimQueryPlan(
            entity="spaces",
            operation="count",
            filters=[BimFilter(field="level", operator="equals", value="7")],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-focused",
            kind="verification",
            summary="test focused verification",
            payload=json.dumps({
                "verified": True,
                "checks": [
                    {
                        "evidence_id": "query-explore",
                        "verified": True,
                        "plan": exploration.model_dump(mode="json"),
                        "claim": {"statement": "All 13 levels...", "basis": "test"},
                        "limitations": [],
                    },
                    {
                        "evidence_id": "query-answer",
                        "verified": True,
                        "plan": answer.model_dump(mode="json"),
                        "claim": {
                            "statement": "There are 8 apartments on the 7th floor.",
                            "value": 8,
                            "basis": "test",
                        },
                        "limitations": [],
                    },
                ],
            }),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(len(report.claims), 1)
        self.assertEqual(report.claims[0].value, 8)
        self.assertNotIn("All 13 levels", report.answer)

    def test_verified_claims_survive_a_rejected_supporting_query(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        context.add_evidence(Evidence(
            evidence_id="verification-partial",
            kind="verification",
            summary="one valid and one rejected query",
            payload=json.dumps({
                "verified": False,
                "checks": [
                    {
                        "evidence_id": "query-valid",
                        "verified": True,
                        "plan": BimQueryPlan(
                            entity="spaces", operation="group_count", group_by="type"
                        ).model_dump(mode="json"),
                        "claim": {"statement": "Ten function types.", "value": 10, "basis": "test"},
                        "semantic_checks": [],
                    },
                    {
                        "evidence_id": "query-support",
                        "verified": False,
                        "plan": BimQueryPlan(entity="spaces", operation="count").model_dump(mode="json"),
                        "semantic_checks": [{
                            "name": "classification_purity",
                            "passed": False,
                            "explanation": "test failure",
                        }],
                    },
                ],
            }),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(report.claims[0].value, 10)
        self.assertIn("omitted", " ".join(report.limitations))

    def test_conflicting_answer_key_values_are_not_rendered(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        plans = [
            BimQueryPlan(entity="elements", operation="count", answer_key="switch_count"),
            BimQueryPlan(entity="elements", operation="count", answer_key="switch_count"),
        ]
        context.add_evidence(Evidence(
            evidence_id="verification-conflict", kind="verification", summary="conflict",
            payload=json.dumps({"checks": [
                {"evidence_id": f"q-{value}", "verified": True,
                 "plan": plan.model_dump(mode="json"),
                 "claim": {"statement": f"{value} switches", "value": value,
                           "unit": "switches", "basis": "test"}, "semantic_checks": []}
                for value, plan in zip((21, 37), plans)
            ]}),
        ))
        report = pipeline_report_from_evidence(context)
        self.assertEqual(report.verification_status, "insufficient_evidence")
        self.assertNotIn("21 switches", report.answer)
        self.assertIn("Conflicting", " ".join(report.limitations))

    def test_grouped_summary_dominates_raw_list_for_same_typed_output(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Summarize functions", operation="group_summary",
                entity_concept="spaces", required_outputs=["function summary"],
                output_specs=[OutputSpec(
                    key="function summary", kind="grouped_summary",
                    metric="area_m2", grouping_dimensions=["type"],
                )],
                success_criteria=["The grouped summary is verified"],
            ),
        )
        raw_plan = BimQueryPlan(
            entity="spaces", operation="list", select=["type", "area_m2"],
            answer_key="raw-functions", satisfies=["function summary"],
        )
        summary_plan = BimQueryPlan(
            entity="spaces", operation="group_summary", group_by="type",
            metric="area_m2", answer_key="grouped-functions",
            satisfies=["function summary"],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-group-dominance", kind="verification", summary="review",
            payload=json.dumps({"checks": [
                {
                    "evidence_id": "query-raw-functions", "verified": True,
                    "plan": raw_plan.model_dump(mode="json"),
                    "claim": {
                        "statement": "There are 380 raw function records.",
                        "value": 380, "unit": "records", "basis": "test",
                    },
                    "semantic_checks": [],
                },
                {
                    "evidence_id": "query-grouped-functions", "verified": True,
                    "plan": summary_plan.model_dump(mode="json"),
                    "claim": {
                        "statement": "Ten function groups summarize the scoped spaces.",
                        "value": 10, "unit": "groups", "basis": "test",
                        "details": ["Office: 100 m2", "Retail: 80 m2"],
                    },
                    "semantic_checks": [],
                },
            ]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(
            [claim.statement for claim in report.claims],
            ["Ten function groups summarize the scoped spaces."],
        )
        self.assertNotIn("380 raw function records", report.answer)
        self.assertEqual(report.output_statuses[0].evidence_ids[0], "query-grouped-functions")

    def test_measurement_dominates_raw_count_for_same_typed_output(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Measure total length", operation="sum", entity_concept="elements",
                required_outputs=["total length"],
                output_specs=[OutputSpec(
                    key="total length", kind="measurement", metric="length",
                    required_unit="m",
                )],
                success_criteria=["The canonical measurement is verified"],
            ),
        )
        count_plan = BimQueryPlan(
            entity="elements", operation="count", answer_key="length-record-count",
            satisfies=["total length"],
        )
        sum_plan = BimQueryPlan(
            entity="elements", operation="sum", metric="length",
            answer_key="canonical-total-length", satisfies=["total length"],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-measurement-dominance", kind="verification", summary="review",
            payload=json.dumps({"checks": [
                {
                    "evidence_id": "query-length-count", "verified": True,
                    "plan": count_plan.model_dump(mode="json"),
                    "claim": {
                        "statement": "There are 106 length records.",
                        "value": 106, "unit": "records", "basis": "test",
                    },
                    "semantic_checks": [],
                },
                {
                    "evidence_id": "query-length-sum", "verified": True,
                    "plan": sum_plan.model_dump(mode="json"),
                    "claim": {
                        "statement": "The total length is 556.98 m.",
                        "value": 556.98, "unit": "m", "basis": "test",
                        "measurement": {
                            "source_value": 556.98, "source_unit": "m",
                            "canonical_unit": "m", "conversion_factor": 1.0,
                            "conversion_basis": "identity conversion",
                            "source_property": "Length",
                        },
                    },
                    "semantic_checks": [],
                },
            ]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(len(report.claims), 1)
        self.assertEqual(report.claims[0].value, 556.98)
        self.assertEqual(report.claims[0].measurement.canonical_unit, "m")
        self.assertNotIn("106 length records", report.answer)

    def test_dominance_preserves_claim_that_satisfies_a_distinct_output(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="List and summarize functions", operation="group_summary",
                entity_concept="spaces",
                required_outputs=["function summary", "function records"],
                output_specs=[
                    OutputSpec(
                        key="function summary", kind="grouped_summary",
                        grouping_dimensions=["type"],
                    ),
                    OutputSpec(key="function records", kind="list"),
                ],
                success_criteria=["Both outputs are verified"],
            ),
        )
        raw_plan = BimQueryPlan(
            entity="spaces", operation="list", select=["type"],
            answer_key="function-records",
            satisfies=["function summary", "function records"],
        )
        summary_plan = BimQueryPlan(
            entity="spaces", operation="group_count", group_by="type",
            answer_key="function-summary", satisfies=["function summary"],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-distinct-output", kind="verification", summary="review",
            payload=json.dumps({"checks": [
                {
                    "evidence_id": "query-function-records", "verified": True,
                    "plan": raw_plan.model_dump(mode="json"),
                    "claim": {
                        "statement": "The function records are listed.",
                        "value": 380, "unit": "records", "basis": "test",
                    },
                    "semantic_checks": [],
                },
                {
                    "evidence_id": "query-function-summary", "verified": True,
                    "plan": summary_plan.model_dump(mode="json"),
                    "claim": {
                        "statement": "The records form ten function groups.",
                        "value": 10, "unit": "groups", "basis": "test",
                    },
                    "semantic_checks": [],
                },
            ]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(len(report.claims), 2)
        self.assertEqual(
            {item.output: item.status for item in report.output_statuses},
            {"function summary": "verified", "function records": "verified"},
        )
        self.assertIn("The function records are listed.", report.answer)
        self.assertIn("The records form ten function groups.", report.answer)

    def test_dominance_does_not_mask_an_atomic_answer_conflict(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Summarize functions", operation="group_summary",
                entity_concept="spaces", required_outputs=["function summary"],
                output_specs=[OutputSpec(
                    key="function summary", kind="grouped_summary",
                    grouping_dimensions=["type"],
                )],
                success_criteria=["The summary is verified"],
            ),
        )
        plans = [
            BimQueryPlan(
                entity="spaces", operation="list", answer_key="function-summary",
                satisfies=["function summary"],
            ),
            BimQueryPlan(
                entity="spaces", operation="group_count", group_by="type",
                answer_key="function-summary", satisfies=["function summary"],
            ),
        ]
        context.add_evidence(Evidence(
            evidence_id="verification-typed-conflict", kind="verification", summary="review",
            payload=json.dumps({"checks": [
                {
                    "evidence_id": f"query-conflict-{value}", "verified": True,
                    "plan": plan.model_dump(mode="json"),
                    "claim": {
                        "statement": f"The answer is {value} groups.",
                        "value": value, "unit": "groups", "basis": "test",
                    },
                    "semantic_checks": [],
                }
                for value, plan in zip((9, 10), plans)
            ]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "insufficient_evidence")
        self.assertEqual(report.claims, [])
        self.assertIn("Conflicting", " ".join(report.limitations))
        self.assertNotIn("10 groups", report.answer)

    def test_missing_required_output_marks_partial_answer_insufficient(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Tray schedule", operation="group_summary", entity_concept="cable trays",
                required_outputs=["type", "width", "material availability"],
                success_criteria=["All requested dimensions are verified"],
            ),
        )
        plan = BimQueryPlan(
            entity="elements", operation="group_count", group_by="type",
            answer_key="tray_types", satisfies=["type"],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-partial-required", kind="verification", summary="partial",
            payload=json.dumps({"checks": [{
                "evidence_id": "q-types", "verified": True,
                "plan": plan.model_dump(mode="json"),
                "claim": {"statement": "Two tray types.", "value": 2, "basis": "test"},
                "semantic_checks": [],
            }]}),
        ))
        report = pipeline_report_from_evidence(context)
        self.assertEqual(report.verification_status, "insufficient_evidence")
        self.assertIn("Two tray types", report.answer)
        self.assertIn("material availability", " ".join(report.limitations))
        statuses = {item.output: item for item in report.output_statuses}
        self.assertEqual(statuses["type"].status, "verified")
        self.assertTrue(statuses["type"].evidence_ids)
        self.assertEqual(statuses["material availability"].status, "unsupported")
        self.assertIn("No replay-verified evidence", statuses["material availability"].limitation)

    def test_verified_supporting_probe_can_satisfy_missing_data_output(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Count apartments", operation="count", entity_concept="apartments",
                required_outputs=["apartment count", "missing identity status"],
                success_criteria=["Count and missing data are replayed"],
            ),
        )
        answer = BimQueryPlan(
            entity="spaces", operation="count", answer_key="apartment-count",
            satisfies=["apartment count"],
        )
        missing = BimQueryPlan(
            entity="spaces", operation="count", role="supporting",
            include_in_answer=False, satisfies=["missing identity status"],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-complete-with-probe", kind="verification", summary="complete",
            payload=json.dumps({"checks": [
                {
                    "evidence_id": "q-answer", "verified": True,
                    "plan": answer.model_dump(mode="json"),
                    "claim": {"statement": "There are 8 apartments.", "value": 8, "basis": "test"},
                    "semantic_checks": [],
                },
                {
                    "evidence_id": "q-missing", "verified": True,
                    "plan": missing.model_dump(mode="json"),
                    "claim": {"statement": "No identities are missing.", "value": 0, "basis": "test"},
                    "semantic_checks": [],
                },
            ]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(report.answer, "There are 8 apartments.")
        self.assertEqual(len(report.supporting_claims), 1)
        self.assertEqual(report.supporting_claims[0].statement, "No identities are missing.")

    def test_supporting_only_evidence_replaces_generic_insufficient_answer(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            completion_status="insufficient_evidence",
            task_contract=BimTaskContract(
                goal="Assess electrical connectivity", operation="coverage",
                entity_concept="electrical components",
                required_outputs=["panel assignment coverage", "physical connectivity"],
                success_criteria=["Report assignment and relationship coverage separately"],
            ),
            runtime_limitations=[
                "Physical port-to-panel continuity was not represented by an executable path."
            ],
        )
        supporting = BimQueryPlan(
            entity="elements", operation="coverage", coverage_field="panel",
            role="supporting", include_in_answer=False,
            satisfies=["panel assignment coverage"],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-supporting-only", kind="verification",
            summary="supporting coverage verified",
            payload=json.dumps({"checks": [{
                "evidence_id": "q-panel-coverage", "verified": True,
                "plan": supporting.model_dump(mode="json"),
                "claim": {
                    "statement": "Panel assignment is populated for 183 of 488 components.",
                    "value": 183, "unit": "components", "basis": "replay-verified coverage",
                },
                "semantic_checks": [],
            }]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "insufficient_evidence")
        self.assertTrue(report.answer.startswith("Verified supporting evidence:"))
        self.assertIn("183 of 488", report.answer)
        self.assertIn("A complete direct answer remains unresolved", report.answer)
        self.assertIn("Physical port-to-panel continuity", report.answer)
        self.assertEqual(report.claims, [])
        self.assertEqual(len(report.supporting_claims), 1)
        statuses = {item.output: item.status for item in report.output_statuses}
        self.assertEqual(statuses["panel assignment coverage"], "verified")
        self.assertEqual(statuses["physical connectivity"], "unsupported")

    def test_no_support_fallback_explains_gap_instead_of_old_generic_sentence(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            runtime_limitations=["No authorized project-program source was available."],
        )

        report = pipeline_report_from_evidence(context)

        self.assertNotEqual(
            report.answer,
            "The BIM question could not be verified from the available scoped evidence.",
        )
        self.assertIn("no replay-verified supporting fact", report.answer)
        self.assertIn("No authorized project-program source", report.answer)

    def test_pipeline_report_exposes_ordered_artifact_trace(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        context.add_artifact(RunArtifact(
            artifact_id="discovery-1",
            kind="graph_discovery",
            producer="Graph Inspector",
            summary="Inspected scoped labels and relationships.",
        ))
        context.add_artifact(RunArtifact(
            artifact_id="mapping-1",
            kind="mapping",
            producer="Schema Mapper",
            summary="Mapped apartment and ground-floor values.",
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.investigation_trace, [
            "Graph Inspector: Inspected scoped labels and relationships.",
            "Schema Mapper: Mapped apartment and ground-floor values.",
        ])

    def test_pipeline_report_preserves_runtime_limitations(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            runtime_limitations=["The investigation reached its configured turn limit."],
        )

        report = pipeline_report_from_evidence(context)

        self.assertIn("turn limit", " ".join(report.limitations))

    def test_measurement_gap_leads_with_not_assessable_diagnostic(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            completion_status="insufficient_evidence",
            runtime_limitations=["Gross floor area is not assessable from the modelled basis."],
        )
        context.add_evidence(Evidence(
            evidence_id="verification-basis-gap",
            kind="verification",
            summary="verified diagnostic",
            payload=json.dumps({
                "verified": True,
                "checks": [{
                    "evidence_id": "query-bvo",
                    "verified": True,
                    "claim": {"statement": "There are 0 BVO apartment records.", "value": 0, "basis": "test"},
                    "semantic_checks": [],
                }],
            }),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.verification_status, "insufficient_evidence")
        self.assertTrue(report.answer.startswith("There are 0 BVO apartment records"))
        self.assertIn("Unresolved verification notes", report.answer)
        self.assertIn("Gross floor area is not assessable", report.answer)

    def test_compliance_evidence_is_added_when_requirements_query_is_missing(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Assess compliance",
                operation="list",
                entity_concept="ground-floor functions",
                required_outputs=["applicable compliance requirements"],
                success_criteria=["Check scoped requirements"],
            ),
        )
        with patch(
            "bim_agents.tools.query_bim",
            return_value=json.dumps({"evidence_id": "query-requirements"}),
        ) as query:
            evidence_id = ensure_compliance_evidence(context)

        self.assertEqual(evidence_id, "query-requirements")
        plan = query.call_args.args[1]
        self.assertEqual(plan.entity, "permit_knowledge")
        self.assertEqual(plan.operation, "list")

    def test_report_discloses_bounded_details_and_actual_stages(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            workstream_diagnostics=[WorkstreamDiagnostic(
                package_id="quantity-1", specialist="quantity",
                status="partial_completed", attempts=2, typed_output_failures=1,
                recovery_strategy="evidence_checkpoint",
                required_outputs=["socket count"],
                satisfied_outputs=["socket count"],
            )],
        )
        context.add_artifact(RunArtifact(
            artifact_id="query-1", kind="cypher_query", producer="Cypher Query Handler",
            summary="Executed a bounded list.",
        ))
        context.add_artifact(RunArtifact(
            artifact_id="verification-1", kind="semantic_review", producer="Verifier",
            summary="Replayed the bounded list.",
        ))
        context.add_evidence(Evidence(
            evidence_id="verification-bounded-list", kind="verification", summary="verified",
            payload=json.dumps({"checks": [{
                "evidence_id": "query-list", "verified": True,
                "plan": {"role": "answer_producing", "include_in_answer": True},
                "claim": {
                    "statement": "Found 61 socket boxes.", "value": 61,
                    "basis": "test", "details": ["first", "second"],
                    "total_count": 61, "displayed_count": 2,
                },
                "semantic_checks": [],
            }]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertIn("shows 2 of 61", report.answer)
        self.assertEqual(report.stages_used, ["Cypher Query Handler", "Verifier"])
        self.assertEqual(report.workstream_diagnostics[0].specialist, "quantity")
        self.assertEqual(report.workstream_diagnostics[0].typed_output_failures, 1)
        self.assertEqual(report.workstream_diagnostics[0].satisfied_outputs, ["socket count"])

    def test_replay_safe_semantic_mismatch_is_retained_as_supporting_fact(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        context.add_evidence(Evidence(
            evidence_id="verification-related", kind="verification", summary="related",
            payload=json.dumps({"checks": [{
                "evidence_id": "query-related", "verified": False,
                "plan": {"role": "answer_producing", "answer_key": "floor-area"},
                "claim": {
                    "statement": "The governed floor-plate area is 586.248 m².",
                    "value": 586.248, "unit": "m²", "basis": "scoped query",
                    "method": "sum",
                },
                "semantic_checks": [
                    {"name": "replay_stability", "passed": True, "explanation": "stable"},
                    {"name": "authorized_scope", "passed": True, "explanation": "scoped"},
                    {"name": "measurement_plausibility", "passed": True, "explanation": "plausible"},
                    {"name": "entity_grain_matches", "passed": False,
                     "explanation": "The exact direct-answer grain remains unresolved."},
                ],
            }]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.claims, [])
        self.assertEqual(len(report.supporting_claims), 1)
        self.assertIn("586.248 m²", report.answer)
        self.assertIn("replay-stable related BIM fact", report.answer)
        self.assertIn("How this was derived", report.answer)

    def test_failed_safety_check_suppresses_related_value(self):
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        context.add_evidence(Evidence(
            evidence_id="verification-unsafe", kind="verification", summary="unsafe",
            payload=json.dumps({"checks": [{
                "evidence_id": "query-unsafe", "verified": False,
                "plan": {"role": "answer_producing", "answer_key": "unsafe"},
                "claim": {"statement": "Unsafe value 999.", "value": 999, "basis": "test"},
                "semantic_checks": [
                    {"name": "replay_stability", "passed": True, "explanation": "stable"},
                    {"name": "authorized_scope", "passed": True, "explanation": "scoped"},
                    {"name": "measurement_plausibility", "passed": False,
                     "explanation": "implausible"},
                ],
            }]}),
        ))

        report = pipeline_report_from_evidence(context)

        self.assertEqual(report.supporting_claims, [])
        self.assertNotIn("Unsafe value 999", report.answer)


if __name__ == "__main__":
    unittest.main()
