import asyncio
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.observability import AgentRunHooks
from bim_agents.runtime import (
    _agent_work_timeout, _answer_query_ids, _budget_exhaustion_diagnostic,
    _has_relevant_promoted_mapping, _is_budget_exhaustion, _merge_isolated_workstream,
    _run_parallel_workstreams, _should_run_parallel_challenger,
)
from bim_agents.models import BimTaskContract, Evidence, RunArtifact
from bim_agents.schema_mapping import RegisteredSchemaMapping, SchemaMappingProposal


class RuntimeBudgetTests(unittest.IsolatedAsyncioTestCase):
    def context(self):
        return BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            max_tool_calls=1,
        )

    async def test_control_loop_tools_do_not_consume_evidence_budget(self):
        context = self.context()
        events = SimpleNamespace(stage=lambda *args, **kwargs: None)
        hooks = AgentRunHooks(events)
        wrapper = SimpleNamespace(context=context)
        agent = SimpleNamespace(name="Supervisor")
        await hooks.on_tool_start(wrapper, agent, SimpleNamespace(name="select_next_action"))
        self.assertEqual(context.tool_calls, 0)
        await hooks.on_tool_start(wrapper, agent, SimpleNamespace(name="query_bim_graph"))
        self.assertEqual(context.tool_calls, 1)

    async def test_isolated_workstreams_really_start_in_parallel(self):
        both_started = asyncio.Event()
        starts = []

        async def workstream(name):
            starts.append(name)
            if len(starts) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            return name

        results = await _run_parallel_workstreams(
            workstream("primary"), workstream("challenger")
        )

        self.assertEqual(results, ["primary", "challenger"])
        self.assertCountEqual(starts, ["primary", "challenger"])

    def test_wrapped_budget_error_is_recognized(self):
        try:
            try:
                raise RuntimeError(
                    "BIM run guardrail stopped execution: tool_calls exceeded 40."
                )
            except RuntimeError as cause:
                raise ValueError("Error running tool") from cause
        except ValueError as wrapped:
            self.assertTrue(_is_budget_exhaustion(wrapped))
        self.assertFalse(_is_budget_exhaustion(ValueError("unrelated tool error")))

    def test_budget_diagnostic_identifies_the_exhausted_counter(self):
        context = self.context()
        context.tool_calls = 2

        category, detail = _budget_exhaustion_diagnostic(context)

        self.assertEqual(category, "tool_call_budget_exhausted")
        self.assertIn("2 calls; limit 1", detail)

    def test_budget_diagnostic_identifies_per_agent_limit(self):
        context = self.context()
        context.agent_starts_by_name["Graph Explorer"] = context.max_starts_per_agent + 1

        category, detail = _budget_exhaustion_diagnostic(context)

        self.assertEqual(category, "agent_start_budget_exhausted")
        self.assertIn("Graph Explorer", detail)

    def test_public_timeout_reserves_time_for_deterministic_finalization(self):
        self.assertEqual(_agent_work_timeout(600, reserve_seconds=30), 570)
        self.assertEqual(_agent_work_timeout(10, reserve_seconds=30), 8)

    def test_nonpositive_public_timeout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _agent_work_timeout(0)

    def test_relevant_promoted_mapping_triggers_fresh_challenger(self):
        context = self.context()
        context.question = "כמה מפסקים בפרוייקט?"
        context.task_contract = BimTaskContract(
            goal="Count switches", operation="count", entity_concept="switches",
            required_outputs=["switch count"], success_criteria=["Verified count"],
        )
        context.bim = SimpleNamespace(ontology=SimpleNamespace(
            keyword_synonyms_for=lambda value: ["switch", "switches"],
            retrieval_terms_for=lambda value: ["lighting switch"],
        ))
        context.learned_knowledge["k"] = SimpleNamespace(
            status="promoted", concept="switches", aliases=[],
            payload={"proposal": {"entity_name": "switches", "value_bindings": []}},
        )

        self.assertTrue(_has_relevant_promoted_mapping(context))

    def test_parallel_challenger_runs_by_default_without_learned_knowledge(self):
        context = self.context()
        context.task_contract = BimTaskContract(
            goal="Count switches", operation="count", entity_concept="switches",
            required_outputs=["switch count"], success_criteria=["Verified count"],
        )
        with patch("bim_agents.runtime.os.getenv", side_effect=lambda key, default=None: default):
            self.assertTrue(_should_run_parallel_challenger(context))

    def test_auto_parallel_mode_requires_relevant_promoted_knowledge(self):
        context = self.context()
        context.task_contract = BimTaskContract(
            goal="Count switches", operation="count", entity_concept="switches",
            required_outputs=["switch count"], success_criteria=["Verified count"],
        )
        context.question = "Count switches"
        context.bim = SimpleNamespace(ontology=SimpleNamespace(
            keyword_synonyms_for=lambda value: [], retrieval_terms_for=lambda value: [],
        ))
        def environment(key, default=None):
            return {"BIM_PARALLEL_CHALLENGER_MODE": "auto"}.get(key, default)

        with patch("bim_agents.runtime.os.getenv", side_effect=environment):
            self.assertFalse(_should_run_parallel_challenger(context))

    def test_mapping_only_isolated_workstream_is_not_committed(self):
        root = self.context()
        branch = self.context()
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-only",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcProduct",
                identity_property="GlobalID", source_property="source",
                counting_unit="switch", counting_unit_evidence="Unique GlobalID.",
                reasoning_summary="Live mapping.",
            ),
            node_count=2, populated_identity_count=2, distinct_identity_count=2,
        )
        branch.schema_mappings[mapping.mapping_id] = mapping

        merged = _merge_isolated_workstream(root, branch, workstream_id="challenger")

        self.assertEqual(merged, [])
        self.assertNotIn(mapping.mapping_id, root.schema_mappings)

    def test_isolated_answer_workstream_commits_mapping_evidence_and_trace(self):
        root = self.context()
        branch = self.context()
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-switches",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcProduct",
                identity_property="GlobalID", source_property="source",
                counting_unit="switch", counting_unit_evidence="Unique GlobalID.",
                reasoning_summary="Live mapping.",
            ),
            node_count=21, populated_identity_count=21, distinct_identity_count=21,
        )
        branch.schema_mappings[mapping.mapping_id] = mapping
        branch.add_evidence(Evidence(
            evidence_id="query-switches", kind="query", summary="21 switches",
            payload='{"plan":{"role":"answer_producing","include_in_answer":true}}',
        ))
        branch.add_artifact(RunArtifact(
            artifact_id="graph-discovery", kind="graph_discovery",
            producer="Graph Inspector", summary="fresh graph",
        ))
        root.add_artifact(RunArtifact(
            artifact_id="graph-discovery", kind="graph_discovery",
            producer="Graph Inspector", summary="primary graph",
        ))

        merged = _merge_isolated_workstream(root, branch, workstream_id="challenger")

        self.assertEqual(merged, ["query-switches"])
        self.assertIn(mapping.mapping_id, root.schema_mappings)
        self.assertIn("query-switches", root.evidence)
        self.assertIn("challenger-graph-discovery", root.artifacts)
        self.assertEqual(_answer_query_ids(root), ["query-switches"])

    def test_conflicting_branch_merge_is_atomic(self):
        root = self.context()
        branch = self.context()
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-new",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcProduct",
                identity_property="GlobalID", source_property="source",
                counting_unit="switch", counting_unit_evidence="Unique GlobalID.",
                reasoning_summary="Live mapping.",
            ),
            node_count=2, populated_identity_count=2, distinct_identity_count=2,
        )
        branch.schema_mappings[mapping.mapping_id] = mapping
        root.add_evidence(Evidence(
            evidence_id="query-conflict", kind="query", summary="root",
            payload='{"plan":{"role":"answer_producing","include_in_answer":true},"claim":{"value":1}}',
        ))
        branch.add_evidence(Evidence(
            evidence_id="query-conflict", kind="query", summary="branch",
            payload='{"plan":{"role":"answer_producing","include_in_answer":true},"claim":{"value":2}}',
        ))

        merged = _merge_isolated_workstream(root, branch, workstream_id="worker-a")

        self.assertEqual(merged, [])
        self.assertNotIn(mapping.mapping_id, root.schema_mappings)
        self.assertEqual(root.evidence["query-conflict"].summary, "root")
        self.assertIn("parallel_evidence_conflict", root.failure_categories)


if __name__ == "__main__":
    unittest.main()
