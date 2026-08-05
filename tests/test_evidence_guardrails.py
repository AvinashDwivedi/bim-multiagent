import json
import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.guardrails import supervisor_report_from_evidence, verification_report_from_evidence
from bim_agents.models import BimRunContext, Evidence, ProjectScope


class EvidenceGuardrailTests(unittest.TestCase):
    def test_builds_verified_zero_from_deterministic_evidence(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        payload = {
            "verified": True,
            "count": 0,
            "expected_count": 0,
            "element_term": "apartments",
            "requested_level": "ground floor",
            "identity": "DISTINCT BIMElement.object_id",
            "classification_issues": ["loft classification is inconsistent"],
        }
        context.add_evidence(Evidence(
            evidence_id="verification-1",
            kind="verification",
            summary="test",
            payload=json.dumps(payload),
        ))
        report = verification_report_from_evidence(context)
        self.assertEqual(report.status, "verified")
        self.assertEqual(report.verified_claims[0].value, 0)
        self.assertIn("loft classification", report.limitations[0])
        supervisor_report = supervisor_report_from_evidence(context)
        self.assertEqual(supervisor_report.verification_status, "verified")
        self.assertEqual(supervisor_report.claims[0].value, 0)

    def test_builds_project_node_claim_from_deterministic_evidence(self):
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        payload = {
            "verified": True,
            "count": 3369,
            "expected_count": 3369,
            "element_term": "project-scoped graph nodes",
            "requested_level": "the authorized project",
            "identity": "DISTINCT Neo4j nodes in the authorized project scope",
            "claim_text": "The authorized project graph contains 3369 nodes.",
            "unit": "nodes",
            "capability": "count_project_nodes",
        }
        context.add_evidence(Evidence(
            evidence_id="verification-nodes",
            kind="verification",
            summary="test node count",
            payload=json.dumps(payload),
        ))

        report = verification_report_from_evidence(context)

        self.assertEqual(report.status, "verified")
        self.assertEqual(report.verified_claims[0].statement, payload["claim_text"])
        self.assertEqual(report.verified_claims[0].unit, "nodes")
        self.assertIn("not a count of the entire Neo4j database", report.limitations[-1])


if __name__ == "__main__":
    unittest.main()
