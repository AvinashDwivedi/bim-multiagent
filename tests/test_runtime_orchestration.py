import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bim_context import Settings
from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import (
    BimRunContext,
    BimTaskContract,
    Evidence,
    EvidenceHandoff,
    EvidenceWorkPackage,
    EvidenceWorkstreamResult,
    ProjectScope,
)
from bim_agents.observability import PipelineEvents
from bim_agents.orchestration import (
    _checkpoint_handoff, evidence_work_packages, package_contract, run_evidence_workstreams,
)
from bim_agents.registry import build_agent_registry
from bim_agents.runtime import _merge_isolated_workstream
from bim_agents.schema_mapping import RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal


class FakeBimContext:
    def __init__(self):
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True


class RuntimeOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def contract(self):
        return BimTaskContract(
            goal="Count and classify switches", operation="count",
            entity_concept="switches",
            required_outputs=["switch count", "switch types"],
            work_packages=[
                EvidenceWorkPackage(
                    package_id="count", objective="Count physical switches",
                    required_outputs=["switch count"],
                ),
                EvidenceWorkPackage(
                    package_id="types", objective="Group switches by type",
                    required_outputs=["switch types"],
                ),
            ],
            success_criteria=["Both results are replayable"],
        )

    def test_package_projection_contains_only_assigned_outputs(self):
        contract = self.contract()
        packages = evidence_work_packages(contract)

        projected = package_contract(contract, packages[0])

        self.assertEqual(projected.required_outputs, ["switch count"])
        self.assertEqual(projected.work_packages, [])

    def test_registered_mapping_checkpoint_recovers_scout_turn_limit(self):
        contract = self.contract()
        root = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question="Count switches",
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=object(), scope=root.scope, graph_contract=root.graph_contract,
            question=root.question, task_contract=package_contract(
                contract, contract.work_packages[0]
            ),
        )
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-switches",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcProduct",
                identity_property="GlobalID", source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="type", property="family_and_type",
                )],
                counting_unit="physical switch",
                counting_unit_evidence="GlobalID is unique per switch.",
                reasoning_summary="Validated live mapping.",
            ),
            node_count=21, populated_identity_count=21, distinct_identity_count=21,
        )
        branch.schema_mappings[mapping.mapping_id] = mapping

        handoff = _checkpoint_handoff(root, branch, contract.work_packages[0])

        self.assertEqual(handoff.route, "live_mapping")
        self.assertEqual(handoff.mapping_id, mapping.mapping_id)
        self.assertIn("type", handoff.evidence_summary)

    async def test_parallel_workers_use_distinct_contexts_and_compact_inputs(self):
        contract = self.contract()
        root = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question="Count and classify switches",
            task_contract=contract,
        )
        settings = Settings("bolt://test", "u", "p", "neo4j", "c", "p")
        registry = build_agent_registry()
        instances = [FakeBimContext(), FakeBimContext()]
        seen_contexts = {}
        seen_inputs = {}

        async def fake_run(agent, input_text, *, context, **kwargs):
            seen_contexts.setdefault(context.workstream_id, id(context))
            seen_inputs.setdefault(context.workstream_id, []).append(input_text)
            if agent.name == "BIM Schema Scout":
                return SimpleNamespace(final_output=EvidenceHandoff(
                    package_id=context.workstream_id,
                    status="ready_for_query", route="contract",
                    entity="elements", operation="count",
                    evidence_summary="Trusted contract route.",
                ))
            evidence_id = "query-" + context.workstream_id
            output = context.task_contract.required_outputs[0]
            context.add_evidence(Evidence(
                evidence_id=evidence_id, kind="query", summary=output,
                payload=json.dumps({
                    "plan": {
                        "role": "answer_producing", "include_in_answer": True,
                        "satisfies": [output], "answer_key": context.workstream_id,
                    },
                    "claim": {"value": 1},
                }),
            ))
            return SimpleNamespace(final_output=EvidenceWorkstreamResult(
                status="query_completed", package_id=context.workstream_id,
                evidence_ids=[evidence_id],
            ))

        with patch("bim_agents.orchestration.BimContext", side_effect=instances), patch(
            "bim_agents.orchestration.Runner.run", side_effect=fake_run
        ):
            outcomes = await run_evidence_workstreams(
                settings=settings, root=root, registry=registry,
                events=PipelineEvents(),
                merge=lambda authoritative, branch: _merge_isolated_workstream(
                    authoritative, branch, workstream_id=branch.workstream_id
                ),
            )

        self.assertEqual({item.status for item in outcomes}, {"query_completed"})
        self.assertEqual(set(root.evidence), {"query-count", "query-types"})
        self.assertEqual(len(set(seen_contexts.values())), 2)
        self.assertTrue(all(item.connected and item.closed for item in instances))
        self.assertNotIn("switch types", seen_inputs["count"][0])
        self.assertNotIn("switch count", seen_inputs["types"][0])
        self.assertEqual(root.evidence["query-count"].workstream_id, "count")


if __name__ == "__main__":
    unittest.main()
