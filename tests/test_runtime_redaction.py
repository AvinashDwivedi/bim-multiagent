import unittest

from bim_context import Settings
from bim_agents.models import Claim, PipelineReport
from bim_agents.runtime import redact_pipeline_report


class RuntimeRedactionTests(unittest.TestCase):
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
