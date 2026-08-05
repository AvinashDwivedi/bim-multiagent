import unittest

from bim_agents.models import Claim, ProjectScope, SupervisorReport


class ContractTests(unittest.TestCase):
    def test_project_scope(self):
        scope = ProjectScope(client_id="client", project_id="project", allowed_sources=["a.ifc"])
        self.assertEqual(scope.allowed_sources, ["a.ifc"])

    def test_supervisor_report_uses_verified_contract(self):
        claim = Claim(statement="There are 2 spaces.", value=2, unit="spaces", basis="distinct IDs")
        report = SupervisorReport(
            answer=claim.statement,
            claims=[claim],
            agents_used=["BIM Query Agent", "Verification Agent"],
            verification_status="verified",
        )
        self.assertEqual(report.claims[0].value, 2)


if __name__ == "__main__":
    unittest.main()
