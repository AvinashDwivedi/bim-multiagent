import json
import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import (
    BimRunContext, BimTaskContract, Evidence, InvestigationAction,
    InvestigationHypothesis, InvestigationObservation, ProjectScope,
)
from bim_agents.tools import (
    PipelineContext, define_bim_task, inspect_completion_gates,
    record_investigation_observation, register_investigation_hypothesis,
    select_investigation_action,
)


class InvestigationLoopTests(unittest.TestCase):
    def context(self):
        return BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )

    def test_action_requires_observation_before_next_action(self):
        context = self.context()
        define_bim_task(PipelineContext(context), BimTaskContract(
            goal="Count physical conduits", operation="count",
            entity_concept="conduits", required_outputs=["physical conduit count"],
            success_criteria=["Verified physical identity and count"],
        ))
        first = InvestigationAction(
            action_id="profile-identity", phase="explore", objective="Resolve identity",
            unresolved_question="Which field identifies a physical segment?",
            proposed_action="Profile identity candidates",
            expected_information_gain="Reveal uniqueness and duplicate rates",
        )
        select_investigation_action(PipelineContext(context), first)
        with self.assertRaisesRegex(ValueError, "Record an observation"):
            select_investigation_action(PipelineContext(context), first.model_copy(
                update={"action_id": "count"}
            ))
        record_investigation_observation(PipelineContext(context), InvestigationObservation(
            action_id="profile-identity", result_summary="GlobalID is duplicated.",
            unexpected_findings=["50 records cover 25 GlobalID values"],
            new_questions=["Is object_id unique?"],
        ))
        self.assertIsNone(context.notebook.next_action)
        self.assertIn("Is object_id unique?", context.notebook.unresolved_questions)

    def test_simple_task_never_reduces_configured_nested_workflow_budget(self):
        context = self.context()
        context.max_llm_calls = 48
        context.max_tool_calls = 60

        define_bim_task(PipelineContext(context), BimTaskContract(
            goal="Count physical switches", operation="count",
            entity_concept="switches", required_outputs=["physical switch count"],
            success_criteria=["Verified classification, identity, count, and replay"],
            complexity="simple",
        ))

        self.assertEqual(context.max_llm_calls, 48)
        self.assertEqual(context.max_tool_calls, 60)

    def test_observation_updates_falsifiable_hypothesis(self):
        context = self.context()
        hypothesis = InvestigationHypothesis(
            hypothesis_id="h1", statement="GlobalID is the physical identity",
            expected_observation="One unique value per conduit record", status="testing",
        )
        register_investigation_hypothesis(PipelineContext(context), hypothesis)
        record_investigation_observation(PipelineContext(context), InvestigationObservation(
            action_id="identity-test", result_summary="Identity collision found.",
            contradicts_hypotheses=["h1"],
        ))
        self.assertEqual(context.notebook.hypotheses[0].status, "rejected")
        self.assertIn(hypothesis.statement, context.notebook.rejected_interpretations)

    def test_completion_gates_prevent_premature_response(self):
        context = self.context()
        define_bim_task(PipelineContext(context), BimTaskContract(
            goal="Tray schedule", operation="group_summary", entity_concept="trays",
            required_outputs=["type", "width", "height", "length"],
            success_criteria=["Every dimension verified"], complexity="complex",
        ))
        gates = json.loads(inspect_completion_gates(PipelineContext(context)))
        self.assertFalse(gates["ready_to_respond"])
        self.assertIn("required output: height", gates["missing"])
        self.assertGreaterEqual(context.max_tool_calls, 55)

    def test_phase_machine_covers_action_observation_recovery_and_blocked_paths(self):
        context = self.context()
        define_bim_task(PipelineContext(context), BimTaskContract(
            goal="Count physical conduits", operation="count",
            entity_concept="conduits", required_outputs=["conduit count"],
            success_criteria=["Verified count"],
        ))
        self.assertEqual(context.notebook.phase, "explore")

        register_investigation_hypothesis(PipelineContext(context), InvestigationHypothesis(
            hypothesis_id="identity", statement="One record is one conduit",
            expected_observation="Stable unique identity", status="testing",
        ))
        self.assertEqual(context.notebook.phase, "analyze")

        select_investigation_action(PipelineContext(context), InvestigationAction(
            action_id="inspect", phase="act", objective="Inspect identity",
            unresolved_question="What is the identity?", proposed_action="Profile IDs",
            expected_information_gain="Identify a stable key",
        ))
        self.assertEqual(context.notebook.phase, "act")
        record_investigation_observation(PipelineContext(context), InvestigationObservation(
            action_id="inspect", phase="observe", result_summary="Identity observed",
        ))
        self.assertEqual(context.notebook.phase, "analyze")
        self.assertIn("observe", context.notebook.phase_history)

        select_investigation_action(PipelineContext(context), InvestigationAction(
            action_id="verify", phase="verify", objective="Verify identity",
            unresolved_question="Is the identity stable?", proposed_action="Replay query",
            expected_information_gain="Confirm replay stability",
        ))
        self.assertEqual(context.notebook.phase, "verify")
        # A failed verification may return to analysis for a correction.
        gates = json.loads(inspect_completion_gates(PipelineContext(context)))
        self.assertFalse(gates["ready_to_respond"])
        self.assertEqual(context.notebook.phase, "analyze")

        select_investigation_action(PipelineContext(context), InvestigationAction(
            action_id="stop", phase="blocked", objective="Stop safely",
            unresolved_question="Can the result be verified?", proposed_action="Report limitation",
            expected_information_gain="None",
        ))
        self.assertEqual(context.notebook.phase, "blocked")
        with self.assertRaisesRegex(ValueError, "Invalid investigation phase transition"):
            select_investigation_action(PipelineContext(context), InvestigationAction(
                action_id="illegal", phase="respond", objective="Respond",
                unresolved_question="", proposed_action="Answer", expected_information_gain="None",
            ))

    def test_verified_completion_routes_through_verify_to_respond(self):
        context = self.context()
        define_bim_task(PipelineContext(context), BimTaskContract(
            goal="Count physical conduits", operation="count",
            entity_concept="conduits", required_outputs=["conduit count"],
            success_criteria=["Verified count"],
        ))
        context.add_evidence(Evidence(
            evidence_id="query-1", kind="query", summary="count",
            payload=json.dumps({
                "plan": {"role": "answer_producing", "answer_key": "count",
                         "satisfies": ["conduit count"]},
                "claim": {"value": 2, "unit": "conduits"},
            }),
        ))
        context.add_evidence(Evidence(
            evidence_id="verification-1", kind="verification", summary="verified",
            payload=json.dumps({
                "checks": [{
                    "verified": True,
                    "plan": {"role": "answer_producing", "satisfies": ["conduit count"]},
                    "semantic_checks": [
                        {"name": "identity_integrity", "passed": True},
                        {"name": "classification_purity", "passed": True},
                        {"name": "replay_stability", "passed": True},
                    ],
                }],
            }),
        ))

        gates = json.loads(inspect_completion_gates(PipelineContext(context)))
        self.assertTrue(gates["ready_to_respond"])
        self.assertEqual(context.notebook.phase, "respond")
        self.assertEqual(context.notebook.phase_history[-3:], ["analyze", "verify", "respond"])


if __name__ == "__main__":
    unittest.main()
