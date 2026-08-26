import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.observability import AgentRunHooks
from bim_agents.claude_runtime import MaxTurnsExceeded, RunResult
from bim_agents.runtime import (
    ArchitectContractError, _agent_work_timeout, _answer_query_ids,
    _architect_repair_limit, _budget_exhaustion_diagnostic,
    _architect_capability_hints, _architect_contract_issues,
    _normalize_architect_capability_metadata,
    _has_relevant_promoted_mapping, _is_budget_exhaustion, _merge_isolated_workstream,
    _recover_active_workstream_checkpoints,
    _run_parallel_workstreams, _run_task_architect, _should_run_parallel_challenger,
)
from bim_agents.models import (
    BimTaskContract, Evidence, EvidenceWorkPackage, OutputSpec, RunArtifact,
)
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

    async def test_task_architect_retries_a_turn_limit_with_fresh_input(self):
        context = self.context()
        contract = BimTaskContract(
            goal="Count elements", operation="count", entity_concept="elements",
            required_outputs=["count"], success_criteria=["Return a verified count."],
        )
        events_seen = []
        events = SimpleNamespace(stage=lambda name, **details: events_seen.append((name, details)))
        runner = AsyncMock(side_effect=[
            MaxTurnsExceeded("bounded attempt ended"),
            SimpleNamespace(final_output=contract),
        ])

        with patch("bim_agents.runtime.Runner.run", runner), patch.dict(
            "os.environ", {"BIM_ARCHITECT_MAX_TURNS": "2", "BIM_ARCHITECT_REPAIR_ATTEMPTS": "1"},
        ):
            result = await _run_task_architect(
                SimpleNamespace(task_architect=SimpleNamespace(name="Task Architect")),
                "Question: count elements",
                context=context,
                hooks=AgentRunHooks(events),
                events=events,
            )

        self.assertEqual(result.final_output, contract)
        self.assertEqual(runner.await_count, 2)
        self.assertTrue(all(call.kwargs["max_turns"] == 2 for call in runner.await_args_list))
        self.assertTrue(any(name == "architect_typed_output_retry" for name, _ in events_seen))

    def test_task_architect_defaults_to_two_bounded_repairs(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_architect_repair_limit(), 2)

    def test_architect_hint_preserves_governed_group_metric(self):
        context = self.context()
        context.question = (
            "what type of functions are in ground floor and does it comply with requirements?"
        )
        context.bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
            "functional_spaces": {
                "entity_concept": "functional spaces",
                "retrieval_terms": [
                    "types of functions on the ground floor", "ground floor function areas",
                ],
                "query_profile": {
                    "operation": "group_summary", "metric": "area_m2",
                    "grouping_fields": ["function_type"],
                },
                "counting_unit": "physical functional-space record",
            },
        }))

        hints = _architect_capability_hints(context.question, context)
        contract = BimTaskContract(
            goal="List and count ground-floor function types",
            operation="group_summary", entity_concept="ground-floor functions",
            required_outputs=["function types and counts"],
            output_specs=[OutputSpec(
                key="functions", kind="grouped_summary", metric="count",
                grouping_dimensions=["function_type"],
            )],
            success_criteria=["Return each function group."],
        )

        self.assertEqual(hints[0]["concept"], "functional spaces")
        self.assertIn("area_m2", " ".join(_architect_contract_issues(contract, hints)))

        normalized = _normalize_architect_capability_metadata(contract, hints)

        self.assertEqual(normalized.output_specs[0].metric, "area_m2")
        self.assertEqual(normalized.output_specs[0].kind, "grouped_summary")
        self.assertIn("function_type", normalized.output_specs[0].grouping_dimensions)
        self.assertEqual(_architect_contract_issues(normalized, hints), [])

    def test_function_route_synonym_satisfies_concept_fidelity(self):
        contract = BimTaskContract(
            goal="Summarize function types on the ground floor",
            operation="group_summary", entity_concept="ground-floor functions",
            required_outputs=["function type summary"],
            output_specs=[OutputSpec(
                key="functions", kind="grouped_summary", metric="area_m2",
                grouping_dimensions=["function_type"],
            )],
            success_criteria=["Return the function groups."],
        )
        hint = {
            "concept": "functional spaces",
            "anchor_tokens": ["functional", "space"],
            "matched_tokens": ["function", "ground", "floor", "area", "type"],
            "metric": "area_m2",
            "required_grouping_dimensions": ["function_type"],
        }

        self.assertEqual(_architect_contract_issues(contract, [hint]), [])

    def test_generic_type_token_does_not_equate_breakers_with_switches(self):
        contract = BimTaskContract(
            goal="List circuit breaker types", operation="list",
            entity_concept="circuit breakers", required_outputs=["breaker types"],
            output_specs=[OutputSpec(key="breakers", kind="list")],
            success_criteria=["Return the breaker types."],
        )
        hint = {
            "concept": "broad switch inventory",
            "anchor_tokens": ["broad", "switch", "inventory"],
            "matched_tokens": ["switch", "type", "project"],
        }

        issues = " ".join(_architect_contract_issues(contract, [hint]))
        self.assertIn("drifted away", issues)

    async def test_task_architect_returns_new_result_for_frozen_runtime_envelope(self):
        context = self.context()
        context.question = "List function types on the ground floor"
        context.bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
            "functional_spaces": {
                "entity_concept": "functional spaces",
                "retrieval_terms": ["function types on the ground floor"],
                "query_profile": {
                    "operation": "group_summary", "metric": "area_m2",
                    "grouping_fields": ["function_type"],
                },
            },
        }))
        contract = BimTaskContract(
            goal="List ground-floor function types",
            operation="group_summary", entity_concept="functional spaces",
            required_outputs=["function summary"],
            output_specs=[OutputSpec(
                key="functions", kind="grouped_summary", metric="count",
                grouping_dimensions=["function_type"],
            )],
            success_criteria=["Return the grouped summary."],
        )
        runner = AsyncMock(return_value=RunResult(final_output=contract))
        events = SimpleNamespace(stage=lambda *args, **kwargs: None)

        with patch("bim_agents.runtime.Runner.run", runner):
            result = await _run_task_architect(
                SimpleNamespace(task_architect=SimpleNamespace(name="Task Architect")),
                f"Question: {context.question}", context=context,
                hooks=AgentRunHooks(events), events=events,
            )

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.final_output.output_specs[0].metric, "area_m2")
        self.assertEqual(contract.output_specs[0].metric, "count")

    def test_architect_hint_requires_generic_floor_as_grouping_dimension(self):
        context = self.context()
        context.question = "How many electrical endpoints are planned on the floor?"
        context.bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
            "electrical_endpoints_by_level": {
                "calculation": "electrical_endpoints_by_level",
                "route_terms": ["electrical endpoints planned on the floor"],
                "recipe": "composite_grouped_count",
                "counting_unit": "physical electrical endpoints",
                "group_unit": "modelled floor",
                "semantic_intent": {"value_origin": "planned"},
            },
        }))
        hints = _architect_capability_hints(context.question, context)
        contract = BimTaskContract(
            goal="Count endpoints on a specified floor",
            operation="group_count", entity_concept="electrical endpoints",
            required_outputs=["endpoint count on specified floor"],
            output_specs=[OutputSpec(
                key="endpoints", kind="grouped_summary",
                grouping_dimensions=["endpoint type"],
            )],
            success_criteria=["Use an exact floor."],
        )

        issues = " ".join(_architect_contract_issues(contract, hints))
        self.assertIn("floor", issues)
        self.assertIn("grouping", issues)

    def test_architect_hint_requires_governed_height_semantics(self):
        context = self.context()
        context.question = "what is the height of the building?"
        context.bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
            "massing_sections": {
                "calculation": "section_heights",
                "route_terms": ["height of the building", "building height"],
                "semantic_intent": {
                    "entity_grain": "building massing sections",
                    "measurement_basis": "height above project datum",
                    "value_origin": "actual",
                },
            },
        }))
        hints = _architect_capability_hints(context.question, context)
        contract = BimTaskContract(
            goal="Measure building height", operation="maximum",
            entity_concept="building height", required_outputs=["height"],
            output_specs=[OutputSpec(key="height", kind="measurement", metric="height")],
            success_criteria=["Return height."],
        )

        issues = " ".join(_architect_contract_issues(contract, hints))
        self.assertIn("measurement_basis", issues)
        self.assertIn("entity_grain", issues)
        self.assertIn("value_origin", issues)

        normalized = _normalize_architect_capability_metadata(contract, hints)
        intent = normalized.output_specs[0].semantic_intent
        self.assertEqual(intent.entity_grain, "building massing sections")
        self.assertEqual(intent.measurement_basis, "height above project datum")
        self.assertEqual(intent.value_origin, "actual")
        self.assertEqual(_architect_contract_issues(normalized, hints), [])

    async def test_task_architect_repairs_multilingual_concept_drift(self):
        context = self.context()
        context.question = "אילו סוגי מפסקים בפרויקט?"
        context.bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
            "broad_switch_inventory": {
                "calculation": "broad_switch_inventory",
                "route_terms": ["סוגי מפסקים בפרויקט"],
                "recipe": "composite_grouped_count",
                "counting_unit": "physical switch-related instances",
                "group_unit": "governed switch type",
            },
        }))
        bad = BimTaskContract(
            goal="List circuit breaker types", operation="list",
            entity_concept="circuit breakers", required_outputs=["breaker types"],
            output_specs=[OutputSpec(key="breakers", kind="list")],
            success_criteria=["List circuit breaker types."],
        )
        output = "switch types"
        spec = OutputSpec(
            key="switches", kind="grouped_summary",
            grouping_dimensions=["governed switch type"],
        )
        good = BimTaskContract(
            goal="List the broad switch inventory", operation="group_count",
            entity_concept="broad switch inventory", required_outputs=[output],
            output_specs=[spec], work_packages=[EvidenceWorkPackage(
                package_id="switches", objective="Group the broad switch inventory by type",
                required_outputs=[output], output_specs=[spec],
            )],
            success_criteria=["Preserve the governed switch scope."],
        )
        runner = AsyncMock(side_effect=[
            SimpleNamespace(final_output=bad), SimpleNamespace(final_output=good),
        ])
        events = SimpleNamespace(stage=lambda *args, **kwargs: None)

        with patch("bim_agents.runtime.Runner.run", runner), patch.dict(
            "os.environ", {"BIM_ARCHITECT_REPAIR_ATTEMPTS": "1"},
        ):
            result = await _run_task_architect(
                SimpleNamespace(task_architect=SimpleNamespace(name="Task Architect")),
                f"Question: {context.question}", context=context,
                hooks=AgentRunHooks(events), events=events,
            )

        self.assertEqual(result.final_output.entity_concept, "broad switch inventory")
        self.assertEqual(runner.await_count, 2)
        self.assertIn("semantic capability hints", runner.await_args_list[0].args[1])
        self.assertIn("drifted away", runner.await_args_list[1].args[1])

    async def test_task_architect_exhaustion_is_explicit_after_three_attempts(self):
        context = self.context()
        context.question = "Which switch types are in the project?"
        context.bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
            "broad_switch_inventory": {
                "calculation": "broad_switch_inventory",
                "route_terms": ["switch types in the project"],
                "recipe": "composite_grouped_count",
                "counting_unit": "physical switch-related instances",
                "group_unit": "governed switch type",
            },
        }))
        drifting = BimTaskContract(
            goal="List circuit breaker types", operation="list",
            entity_concept="circuit breakers", required_outputs=["breaker types"],
            output_specs=[OutputSpec(key="breakers", kind="list")],
            success_criteria=["List circuit breaker types."],
        )
        runner = AsyncMock(side_effect=[
            SimpleNamespace(final_output=drifting),
            SimpleNamespace(final_output=drifting),
            SimpleNamespace(final_output=drifting),
        ])
        events_seen = []
        events = SimpleNamespace(
            stage=lambda name, **details: events_seen.append((name, details)),
        )

        with patch("bim_agents.runtime.Runner.run", runner), patch.dict(
            "os.environ", {"BIM_ARCHITECT_REPAIR_ATTEMPTS": "2"},
        ):
            with self.assertRaises(ArchitectContractError) as raised:
                await _run_task_architect(
                    SimpleNamespace(task_architect=SimpleNamespace(name="Task Architect")),
                    f"Question: {context.question}", context=context,
                    hooks=AgentRunHooks(events), events=events,
                )

        self.assertEqual(runner.await_count, 3)
        self.assertEqual(raised.exception.last_error_type, "ValueError")
        exhausted = [
            details for name, details in events_seen
            if name == "architect_contract_exhausted"
        ]
        self.assertEqual(exhausted[-1]["attempts"], 3)

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

    def test_isolated_supporting_population_evidence_is_committed(self):
        root = self.context()
        branch = self.context()
        branch.add_evidence(Evidence(
            evidence_id="query-related-population",
            kind="query",
            summary="100 governed dwelling identities",
            payload=(
                '{"plan":{"role":"supporting","include_in_answer":false,'
                '"satisfies":[]},"claim":{"value":100}}'
            ),
        ))

        merged = _merge_isolated_workstream(
            root, branch, workstream_id="sector-count",
        )

        self.assertEqual(merged, ["query-related-population"])
        self.assertIn("query-related-population", root.evidence)
        self.assertEqual(_answer_query_ids(root), ["query-related-population"])

    def test_timeout_recovery_commits_active_branch_query_checkpoint(self):
        root = self.context()
        branch = self.context()
        branch.workstream_id = "homes"
        branch.add_evidence(Evidence(
            evidence_id="query-homes", kind="query", summary="five candidate homes",
            payload=(
                '{"plan":{"role":"answer_producing","include_in_answer":true,'
                '"satisfies":["home count"]},"claim":{"value":5}}'
            ),
        ))
        active = {"homes": branch}
        emitted = []

        recovered = _recover_active_workstream_checkpoints(
            root, active,
            events=SimpleNamespace(stage=lambda name, **details: emitted.append((name, details))),
        )

        self.assertEqual(recovered, ["query-homes"])
        self.assertIn("query-homes", root.evidence)
        self.assertEqual(active, {})
        self.assertEqual(emitted[0][0], "workstream_timeout_checkpoint_committed")

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
