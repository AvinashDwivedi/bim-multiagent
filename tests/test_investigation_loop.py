import json
import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import (
    BimRunContext, BimTaskContract, InvestigationAction,
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


if __name__ == "__main__":
    unittest.main()
