import json
import unittest
from unittest.mock import patch

from bim_agents.graph_contract import load_graph_contract
from bim_agents.guardrails import pipeline_report_from_evidence, verification_report_from_evidence
from bim_agents.models import (
    BimRunContext, BimTaskContract, Evidence, ProjectScope, RunArtifact,
)
from bim_agents.tools import (
    BimFilter, BimQueryPlan, _verify_query_evidence, ensure_bim_verification,
    ensure_compliance_evidence,
)


class EvidenceGuardrailTests(unittest.TestCase):
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

    def test_compliance_evidence_is_added_when_requirements_query_is_missing(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Assess compliance",
                operation="list",
                entity_concept="ground-floor functions",
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


if __name__ == "__main__":
    unittest.main()
