import unittest

from pydantic import ValidationError

from bim_agents.models import (
    BimRunContext, BimTaskContract, Claim, ConstraintBinding, Evidence,
    EvidenceWorkPackage, OutputSpec, PipelineReport, ProjectScope, RunArtifact,
    SemanticIntent, TaskConstraint,
)


class ContractTests(unittest.TestCase):
    def test_contract_boundary_normalizes_requirements_and_coverage_absence(self):
        from bim_agents.graph_contract import load_graph_contract
        from bim_agents.tools import PipelineContext, define_bim_task

        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        outputs = ["function assignment coverage", "applicable requirements"]
        specs = [
            OutputSpec(
                key="coverage", kind="coverage",
                semantic_intent=SemanticIntent(absence_semantics="property_missing"),
            ),
            OutputSpec(
                key="requirements", kind="list",
                semantic_intent=SemanticIntent(absence_semantics="property_missing"),
            ),
        ]
        contract = BimTaskContract(
            goal="Report function coverage and assess applicable requirements",
            operation="list", entity_concept="functions",
            required_outputs=outputs, output_specs=specs,
            work_packages=[
                EvidenceWorkPackage(
                    package_id="facts", objective="Report function coverage",
                    required_outputs=[outputs[0]], output_specs=[specs[0]],
                ),
                EvidenceWorkPackage(
                    package_id="requirements", objective="Find applicable requirements",
                    required_outputs=[outputs[1]], output_specs=[specs[1]],
                    route_hint="requirements",
                ),
            ],
            success_criteria=["Return facts and requirements availability."],
        )

        define_bim_task(PipelineContext(context), contract)

        normalized = context.task_contract
        self.assertEqual(
            normalized.output_specs[0].semantic_intent.absence_semantics,
            "unspecified",
        )
        self.assertEqual(
            normalized.output_specs[1].semantic_intent.absence_semantics,
            "population_missing",
        )
        self.assertEqual(
            normalized.work_packages[1].output_specs[0].semantic_intent.absence_semantics,
            "population_missing",
        )

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

    def test_work_package_dependency_cycles_are_rejected_at_contract_boundary(self):
        with self.assertRaisesRegex(ValidationError, "acyclic"):
            BimTaskContract(
                goal="Resolve two dependent facts", operation="list", entity_concept="elements",
                required_outputs=["first", "second"],
                work_packages=[
                    EvidenceWorkPackage(
                        package_id="first", objective="Resolve first", required_outputs=["first"],
                        depends_on=["second"],
                    ),
                    EvidenceWorkPackage(
                        package_id="second", objective="Resolve second", required_outputs=["second"],
                        depends_on=["first"],
                    ),
                ],
                success_criteria=["Both facts are verified"],
            )

    def test_legacy_outputs_gain_typed_specs(self):
        contract = BimTaskContract(
            goal="Count switches", operation="count", entity_concept="switches",
            required_outputs=["switch count"], success_criteria=["Count is verified"],
        )

        self.assertEqual(contract.output_specs, [OutputSpec(key="switch count")])

    def test_work_package_specialist_contract_rejects_preflight_role(self):
        package = EvidenceWorkPackage(
            package_id="job", objective="Count objects", required_outputs=["count"],
        )
        for role in ("auto", "quantity", "relationship", "geometry", "requirements"):
            with self.subTest(role=role):
                self.assertEqual(package.model_copy(update={"specialist": role}).specialist, role)
        with self.assertRaises(ValidationError):
            EvidenceWorkPackage(
                package_id="map", objective="Map schema", required_outputs=["mapping"],
                specialist="schema_mapping",
            )

    def test_typed_outputs_can_define_required_output_labels(self):
        contract = BimTaskContract(
            goal="Measure tray length", operation="sum", entity_concept="cable trays",
            output_specs=[OutputSpec(
                key="total tray length", kind="measurement", metric="length",
                required_unit="m",
            )],
            work_packages=[EvidenceWorkPackage(
                package_id="tray_length", objective="Measure all trays",
                required_outputs=["total tray length"],
            )],
            success_criteria=["Length is verified"],
        )

        self.assertEqual(contract.required_outputs, ["total tray length"])
        self.assertEqual(contract.work_packages[0].output_specs[0].required_unit, "m")

    def test_machine_output_keys_may_differ_from_human_labels(self):
        contract = BimTaskContract(
            goal="Count seventh-floor apartments", operation="count",
            entity_concept="apartment",
            required_outputs=["Count of apartments on the 7th floor"],
            output_specs=[OutputSpec(
                key="apartment_count_7th_floor", kind="count",
                metric="apartment count", scope="Apartments on the 7th floor",
                required_unit="apartments",
            )],
            work_packages=[EvidenceWorkPackage(
                package_id="wp1", objective="Count apartments on the resolved floor",
                required_outputs=["Count of apartments on the 7th floor"],
            )],
            success_criteria=["Return one verified integer count."],
        )

        self.assertEqual(contract.required_outputs, ["Count of apartments on the 7th floor"])
        self.assertEqual(contract.output_specs[0].key, "apartment_count_7th_floor")
        self.assertEqual(contract.work_packages[0].output_specs[0].key, "apartment_count_7th_floor")

    def test_branch_evidence_records_package_and_constraint_provenance(self):
        from bim_agents.graph_contract import load_graph_contract

        context = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), workstream_id="ground_floor",
        )
        binding = ConstraintBinding(
            concept="level", requested_value="ground floor", semantic_field="level",
            exact_values=["Ground Floor"], mapping_id="spaces", package_id="ground_floor",
        )
        context.add_evidence(Evidence(
            evidence_id="q-ground", kind="query", summary="ground floor count",
            constraint_bindings=[binding],
        ))

        evidence = context.evidence["q-ground"]
        self.assertEqual(evidence.workstream_id, "ground_floor")
        self.assertEqual(evidence.work_package_id, "ground_floor")
        self.assertEqual(evidence.constraint_bindings[0].exact_values, ["Ground Floor"])

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

    def test_claim_rejects_inconsistent_display_population(self):
        with self.assertRaises(ValidationError):
            Claim(
                statement="Three records.", value=3, unit="records", basis="test",
                total_count=3, displayed_count=4,
            )


if __name__ == "__main__":
    unittest.main()
