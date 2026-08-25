import unittest

from pydantic import ValidationError

from bim_agents.models import (
    BimRunContext, BimTaskContract, Claim, Evidence, EvidenceWorkPackage,
    PipelineReport, ProjectScope, RunArtifact, TaskConstraint,
)


class ContractTests(unittest.TestCase):
    def test_run_context_records_append_only_workflow_artifacts(self):
        from bim_agents.graph_contract import load_graph_contract
        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="client", project_id="project"),
            graph_contract=load_graph_contract(),
        )
        context.task_contract = BimTaskContract(
            goal="Count apartments on the ground floor.", operation="count",
            entity_concept="apartment",
            constraints=[TaskConstraint(concept="level", requested_value="ground floor")],
            questions_to_resolve=["What record represents one apartment?"],
            required_outputs=["ground-floor apartment count"],
            success_criteria=["Use a unique apartment identity."],
        )
        artifact = RunArtifact(
            artifact_id="task-contract", kind="task_contract", producer="Pipeline",
            summary=context.task_contract.goal,
            payload=context.task_contract.model_dump(mode="json"),
        )
        context.add_artifact(artifact)
        self.assertEqual(context.artifacts["task-contract"].payload["operation"], "count")
        with self.assertRaisesRegex(ValueError, "already exists"):
            context.add_artifact(artifact)

    def test_project_scope(self):
        scope = ProjectScope(client_id="client", project_id="project", allowed_sources=["a.ifc"])
        self.assertEqual(scope.allowed_sources, ["a.ifc"])

    def test_work_packages_must_partition_required_outputs(self):
        with self.assertRaises(ValidationError):
            BimTaskContract(
                goal="Count and list", operation="count", entity_concept="switches",
                required_outputs=["count", "types"],
                work_packages=[EvidenceWorkPackage(
                    package_id="count", objective="Count switches",
                    required_outputs=["count"],
                )],
                success_criteria=["Both outputs are verified"],
            )

    def test_evidence_ledger_rejects_duplicate_ids(self):
        from bim_agents.graph_contract import load_graph_contract

        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        context.add_evidence(Evidence(
            evidence_id="q1", kind="query", summary="first",
        ))
        with self.assertRaisesRegex(ValueError, "already exists"):
            context.add_evidence(Evidence(
                evidence_id="q1", kind="query", summary="second",
            ))

    def test_pipeline_report_uses_verified_contract(self):
        claim = Claim(statement="There are 2 spaces.", value=2, unit="spaces", basis="distinct IDs")
        report = PipelineReport(
            answer=claim.statement, claims=[claim], stages_used=["Query Planner", "Verifier"],
            verification_status="verified",
        )
        self.assertEqual(report.claims[0].value, 2)


if __name__ == "__main__":
    unittest.main()
