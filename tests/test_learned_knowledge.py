import json
import unittest
from pathlib import Path
from uuid import uuid4

from bim_agents.graph_contract import load_graph_contract
from bim_agents.knowledge import (
    KnowledgeProposal, LearnedKnowledgeStore, activate_promoted_knowledge,
    curate_verified_mappings, save_knowledge_candidate, verify_and_promote_candidate,
)
from bim_agents.models import BimRunContext, Evidence, ProjectScope
from bim_agents.schema_mapping import (
    RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal,
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
        )
        plan = {
            "mapping_id": mapping.mapping_id,
            "role": "answer_producing",
            "include_in_answer": True,
        }
        context.add_evidence(Evidence(
            evidence_id="query-1", kind="query", summary="Scoped switch count.",
            payload=json.dumps({"plan": plan}),
        ))
        context.add_evidence(Evidence(
            evidence_id="verification-1", kind="verification", summary="Replay passed.",
            payload=json.dumps({
                "checks": [{"evidence_id": "query-1", "verified": True, "plan": plan}],
                "verified_evidence_ids": ["query-1"],
            }),
        ))
        return context

    def test_incomplete_run_cannot_curate_mapping(self):
        context = self.context()
        context.completion_status = "insufficient_evidence"

        report = curate_verified_mappings(context)

        self.assertEqual(report.status, "no_change")
        self.assertEqual(self.store.compatible(
            client_id="client-a", project_id="project-a", schema_fingerprint="schema-a"
        ), [])

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


if __name__ == "__main__":
    unittest.main()
