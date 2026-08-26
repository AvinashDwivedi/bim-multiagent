import json
import unittest
from pathlib import Path
from uuid import uuid4

from bim_agents.graph_contract import load_graph_contract
from bim_agents.knowledge import (
    KnowledgeProposal, LearnedKnowledgeStore, activate_promoted_knowledge,
    curate_verified_mappings, save_knowledge_candidate, verify_and_promote_candidate,
)
from bim_agents.models import BimRunContext, BimTaskContract, Evidence, ProjectScope
from bim_agents.schema_mapping import (
    RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal,
    SchemaMatchEvidence,
)


class _Bim:
    def query(self, cypher, parameters=None):
        if "property_key_groups" in cypher:
            return [{
                "node_count": 2, "populated_count": 2, "distinct_count": 2,
                "property_key_groups": [["source", "GlobalID", "object_family_type"]],
            }]
        raise AssertionError(f"Unexpected query: {cypher}")


def _mapping(mapping_id="mapping-switches"):
    return RegisteredSchemaMapping(
        mapping_id=mapping_id,
        proposal=SchemaMappingProposal(
            entity_name="electrical_switches",
            label="IfcProduct",
            identity_property="GlobalID",
            source_property="source",
            fields=[SchemaFieldMapping(
                semantic_name="family_type", property="object_family_type"
            )],
            counting_unit="physical electrical switch",
            counting_unit_evidence="GlobalID is populated and unique for each switch instance.",
            reasoning_summary="Live family and identity evidence supports the mapping.",
        ),
        node_count=2,
        populated_identity_count=2,
        distinct_identity_count=2,
    )


def _semantic_checks(passed=True):
    return [
        {"name": name, "passed": passed, "explanation": "test semantic gate"}
        for name in sorted({
            "replay_stability", "authorized_scope", "identity_integrity",
            "constraint_binding", "counting_unit", "boundary_exactness",
            "source_deduplication", "classification_purity", "constraint_coverage",
            "entity_grain_matches", "measurement_basis_matches", "population_complete",
            "planned_actual_distinguished", "absence_semantics_correct",
            "projection_answers_question", "requested_outputs_present",
        })
    ]


class LearnedKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.database_path = Path.cwd() / "tests" / f"knowledge-{uuid4().hex}.sqlite3"
        self.store = LearnedKnowledgeStore(self.database_path)

    def tearDown(self):
        self.database_path.unlink(missing_ok=True)

    def context(self):
        mapping = _mapping()
        context = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(client_id="client-a", project_id="project-a", allowed_sources=["m.ifc"]),
            graph_contract=load_graph_contract(),
            schema_mappings={mapping.mapping_id: mapping},
            knowledge_store=self.store,
            schema_fingerprint="schema-a",
            completion_status="ready_for_verification",
            task_contract=BimTaskContract(
                goal="Count switches", operation="count_distinct",
                entity_concept="electrical switches", required_outputs=["switch count"],
                success_criteria=["A scoped switch count is verified."],
            ),
        )
        plan = {
            "mapping_id": mapping.mapping_id,
            "role": "answer_producing",
            "include_in_answer": True,
            "answer_key": "switch-count",
            "satisfies": ["switch count"],
        }
        context.add_evidence(Evidence(
            evidence_id="query-1", kind="query", summary="Scoped switch count.",
            payload=json.dumps({
                "plan": plan,
                "claim": {"statement": "There are two switches.", "value": 2},
            }),
        ))
        context.add_evidence(Evidence(
            evidence_id="verification-1", kind="verification", summary="Replay passed.",
            payload=json.dumps({
                "checks": [{
                    "evidence_id": "query-1", "verified": True, "plan": plan,
                    "claim": {"statement": "There are two switches.", "value": 2},
                    "matched_count": 2, "diagnostics": [],
                    "semantic_checks": _semantic_checks(),
                }],
                "verified_evidence_ids": ["query-1"],
            }),
        ))
        return context

    @staticmethod
    def _replace_plan(context, plan, **check_updates):
        context.evidence["query-1"] = Evidence(
            evidence_id="query-1", kind="query", summary="Scoped switch count.",
            payload=json.dumps({
                "plan": plan,
                "claim": {"statement": "There are two switches.", "value": 2},
            }),
        )
        check = {
            "evidence_id": "query-1", "verified": True, "plan": plan,
            "claim": {"statement": "There are two switches.", "value": 2},
            "matched_count": 2, "diagnostics": [],
            "semantic_checks": _semantic_checks(),
        }
        check.update(check_updates)
        context.evidence["verification-1"] = Evidence(
            evidence_id="verification-1", kind="verification", summary="Replay passed.",
            payload=json.dumps({
                "checks": [check], "verified_evidence_ids": ["query-1"],
            }),
        )

    def test_incomplete_run_cannot_curate_mapping(self):
        context = self.context()
        context.completion_status = "insufficient_evidence"

        report = curate_verified_mappings(context)

        self.assertEqual(report.status, "no_change")
        self.assertEqual(self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a"
        ), [])

    def test_low_level_promotion_waits_for_completed_failure_free_run(self):
        context = self.context()
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.9,
        ))
        context.completion_status = "insufficient_evidence"

        incomplete = verify_and_promote_candidate(context, candidate.knowledge_id)

        self.assertEqual(incomplete.status, "candidate")
        context.completion_status = "ready_for_verification"
        context.failure_categories.append("worker_failure")
        failed = verify_and_promote_candidate(context, candidate.knowledge_id)
        self.assertEqual(failed.status, "candidate")

    def test_supporting_only_evidence_cannot_create_or_promote_mapping(self):
        context = self.context()
        plan = {
            "mapping_id": "mapping-switches", "role": "supporting",
            "include_in_answer": False, "answer_key": "switch-count",
            "satisfies": ["switch count"],
        }
        self._replace_plan(context, plan)

        report = curate_verified_mappings(context)

        self.assertEqual(report.status, "no_change")
        self.assertEqual(self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a",
            statuses=("candidate", "verified", "promoted"),
        ), [])

    def test_verified_flag_cannot_override_failed_semantic_adequacy(self):
        context = self.context()
        plan = json.loads(context.evidence["query-1"].payload)["plan"]
        failed_checks = _semantic_checks()
        failed_checks[0]["passed"] = False
        self._replace_plan(context, plan, semantic_checks=failed_checks)
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.9,
        ))

        result = verify_and_promote_candidate(context, candidate.knowledge_id)

        self.assertEqual(result.status, "rejected")
        self.assertIn("semantic adequacy", result.validation_note)

    def test_answer_evidence_must_satisfy_a_required_output(self):
        context = self.context()
        plan = json.loads(context.evidence["query-1"].payload)["plan"]
        plan["satisfies"] = ["unrequested diagnostic"]
        self._replace_plan(context, plan)
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.9,
        ))

        result = verify_and_promote_candidate(context, candidate.knowledge_id)

        self.assertEqual(result.status, "candidate")

    def test_answer_evidence_must_use_the_candidate_mapping(self):
        context = self.context()
        other = _mapping("mapping-other")
        other.proposal.fields[0].property = "different_type_property"
        context.schema_mappings[other.mapping_id] = other
        plan = json.loads(context.evidence["query-1"].payload)["plan"]
        plan["mapping_id"] = other.mapping_id
        self._replace_plan(context, plan)
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.9,
        ))

        result = verify_and_promote_candidate(context, candidate.knowledge_id)

        self.assertEqual(result.status, "candidate")

    def test_diagnostic_answer_evidence_is_rejected(self):
        context = self.context()
        plan = json.loads(context.evidence["query-1"].payload)["plan"]
        self._replace_plan(context, plan, diagnostics=["missing_metric"])
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.9,
        ))

        result = verify_and_promote_candidate(context, candidate.knowledge_id)

        self.assertEqual(result.status, "rejected")

    def test_verified_mapping_is_promoted_without_storing_answer_values(self):
        context = self.context()

        report = curate_verified_mappings(context)

        self.assertEqual(report.status, "promoted")
        records = self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "promoted")
        rendered = json.dumps(records[0].payload, sort_keys=True)
        self.assertNotIn('"answer"', rendered)
        self.assertNotIn('"count"', rendered)
        self.assertNotIn("node_count", rendered)

    def test_direct_answers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot store answer/result"):
            self.store.save_candidate(
                client_id="client-a", project_id="project-a", schema_fingerprint="schema-a",
                proposal=KnowledgeProposal(
                    concept="switches", mapping_id="mapping-switches",
                    evidence_ids=["query-1"],
                ),
                payload={"answer": 21},
            )

    def test_scope_and_schema_fingerprint_isolate_promoted_knowledge(self):
        context = self.context()
        curate_verified_mappings(context)

        self.assertEqual(self.store.compatible(
            client_id="client-a", project_id="project-b", schema_fingerprint="schema-a"
        ), [])
        changed = self.store.deprecate_incompatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-b"
        )
        self.assertEqual(changed, 1)
        self.assertEqual(self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a"
        ), [])

    def test_promoted_mapping_is_live_revalidated_before_activation(self):
        original = self.context()
        curate_verified_mappings(original)
        fresh = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(client_id="client-a", project_id="project-a", allowed_sources=["m.ifc"]),
            graph_contract=load_graph_contract(), knowledge_store=self.store,
            schema_fingerprint="schema-a",
        )

        active = activate_promoted_knowledge(fresh)

        self.assertEqual(len(active), 1)
        self.assertIn("learned-" + active[0].knowledge_id.removeprefix("knowledge-"), fresh.schema_mappings)
        self.assertTrue(any(
            artifact.producer == "Knowledge Curator" for artifact in fresh.artifacts.values()
        ))

    def test_non_mapping_rules_are_saved_but_not_autonomously_promoted(self):
        context = self.context()
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            kind="measurement_rule", concept="tray length basis",
            definition="Use the modeled segment length property and list fittings separately.",
            unit="m", basis="Revit segment centerline length",
            evidence_ids=["query-1"], confidence=0.95,
        ))

        result = verify_and_promote_candidate(context, candidate.knowledge_id)

        self.assertEqual(result.status, "verified")
        self.assertEqual(self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a"
        ), [])
        verified = self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a",
            statuses=("verified",),
        )
        self.assertEqual([item.kind for item in verified], ["measurement_rule"])

    def test_semantically_identical_mapping_candidates_share_one_record(self):
        context = self.context()
        first_mapping = context.schema_mappings["mapping-switches"]
        first_mapping.proposal.match_evidence = [SchemaMatchEvidence(
            semantic_name="family_type", property="object_family_type",
            concept="switch type", similarity=0.81,
        )]
        second_mapping = _mapping("mapping-switches-repeat")
        second_mapping.proposal.match_evidence = [SchemaMatchEvidence(
            semantic_name="family_type", property="object_family_type",
            concept="lighting switch family", similarity=0.97,
        )]
        context.schema_mappings[second_mapping.mapping_id] = second_mapping
        context.add_evidence(Evidence(
            evidence_id="query-2", kind="query", summary="Repeated scoped discovery.",
            payload=json.dumps({"plan": {"mapping_id": second_mapping.mapping_id}}),
        ))

        first = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id=first_mapping.mapping_id,
            evidence_ids=["query-1"], aliases=["switch"], confidence=0.81,
        ))
        second = save_knowledge_candidate(context, KnowledgeProposal(
            concept="lighting switches", mapping_id=second_mapping.mapping_id,
            evidence_ids=["query-2"], aliases=["lighting switch"], confidence=0.97,
        ))

        self.assertEqual(first.knowledge_id, second.knowledge_id)
        self.assertEqual(second.evidence_ids, ["query-1", "query-2"])
        self.assertEqual(second.aliases, ["switch", "lighting switch"])
        compatible = self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a",
            statuses=("candidate",),
        )
        self.assertEqual([item.knowledge_id for item in compatible], [second.knowledge_id])

    def test_compatible_records_collapse_legacy_duplicates_and_preserve_lifecycle(self):
        context = self.context()
        candidate = save_knowledge_candidate(context, KnowledgeProposal(
            concept="electrical switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.8,
        ))
        verified = self.store.transition(
            candidate.knowledge_id, client_id="client-a", project_id="project-a",
            from_statuses=("candidate",), to_status="verified", note="test verification",
        )
        promoted = self.store.transition(
            verified.knowledge_id, client_id="client-a", project_id="project-a",
            from_statuses=("verified",), to_status="promoted", note="test promotion",
        )
        legacy_id = "knowledge-legacyduplicate"
        connection = self.store._connect()
        try:
            connection.execute(
                "UPDATE learned_knowledge SET knowledge_id=? WHERE knowledge_id=?",
                (legacy_id, promoted.knowledge_id),
            )
            connection.commit()
        finally:
            connection.close()
        repeated = save_knowledge_candidate(context, KnowledgeProposal(
            concept="lighting switches", mapping_id="mapping-switches",
            evidence_ids=["query-1"], confidence=0.99,
        ))

        compatible = self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a",
            statuses=("promoted", "candidate"),
        )

        self.assertNotEqual(repeated.knowledge_id, legacy_id)
        self.assertEqual(len(compatible), 1)
        self.assertEqual(compatible[0].knowledge_id, legacy_id)
        self.assertEqual(compatible[0].status, "promoted")
        fresh = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(
                client_id="client-a", project_id="project-a", allowed_sources=["m.ifc"]
            ),
            graph_contract=load_graph_contract(), knowledge_store=self.store,
            schema_fingerprint="schema-a",
        )

        active = activate_promoted_knowledge(fresh)

        self.assertEqual([item.knowledge_id for item in active], [legacy_id])
        self.assertEqual(len(fresh.schema_mappings), 1)


if __name__ == "__main__":
    unittest.main()
