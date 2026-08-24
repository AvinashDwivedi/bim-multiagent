import unittest

from bim_context import Settings
from bim_agents.models import Claim, PipelineReport
from bim_agents.graph_contract import load_graph_contract
from bim_agents.runtime import _capability_gap, redact_pipeline_report


class RuntimeRedactionTests(unittest.TestCase):
    def test_capability_gap_does_not_conflate_elevation_with_height(self):
        contract = load_graph_contract()
        self.assertIsNone(_capability_gap(
            "What are the various section heights of the building?", contract
        ))
        self.assertIn("vertical positions", _capability_gap(
            "Whats the ground floor space height?", contract
        ))
        self.assertIsNone(_capability_gap(
            "How high is the model?", contract
        ))

    def test_capability_gap_requires_both_facade_ratio_operands(self):
        gap = _capability_gap(
            "What is the opening percentage of the tower facade?",
            load_graph_contract(),
        )
        self.assertIn("opening-area and façade-area", gap)

    def test_removes_scope_ids_and_credentials_from_every_report_field(self):
        settings = Settings(
            neo4j_uri="neo4j+s://private.example",
            neo4j_username="private-user",
            neo4j_password="private-password",
            neo4j_database="neo4j",
            client_id="client-12345678",
            project_id="project-12345678",
        )
        report = PipelineReport(
            answer="client_id=client-12345678 at neo4j+s://private.example",
            claims=[Claim(
                statement="Project project-12345678",
                basis="neo4j_password=private-password",
            )],
            investigation_trace=["Connected as private-user"],
            verification_status="verified",
        )

        redacted = redact_pipeline_report(report, settings)
        rendered = redacted.model_dump_json()

        for secret in (
            settings.client_id, settings.project_id, settings.neo4j_uri,
            settings.neo4j_username, settings.neo4j_password,
        ):
            self.assertNotIn(secret, rendered)
        self.assertIn("[REDACTED]", rendered)


if __name__ == "__main__":
    unittest.main()
