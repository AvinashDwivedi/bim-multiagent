import json
import unittest
from unittest.mock import patch

from bim_agents.graph_contract import load_graph_contract
from bim_agents.guardrails import supervisor_report_from_evidence, verification_report_from_evidence
from bim_agents.models import BimRunContext, Evidence, ProjectScope
from bim_agents.tools import BimFilter, BimQueryPlan, _verify_query_evidence


class EvidenceGuardrailTests(unittest.TestCase):
    def test_verification_replays_all_query_evidence_when_model_passes_no_ids(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
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
        supervisor_report = supervisor_report_from_evidence(context)
        self.assertEqual(supervisor_report.verification_status, "verified")
        self.assertEqual(supervisor_report.claims[0].value, 100.0)
        self.assertIn("IfcSpace 6891\n  - Area: 58.26 m²", supervisor_report.answer)

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

        report = supervisor_report_from_evidence(context)

        self.assertEqual(len(report.claims), 1)
        self.assertEqual(report.claims[0].value, 8)
        self.assertNotIn("All 13 levels", report.answer)


if __name__ == "__main__":
    unittest.main()
