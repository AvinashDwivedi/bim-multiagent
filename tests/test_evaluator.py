import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bim_agents.evaluator import (
    EvaluationCase, EvaluationJudgment, evaluate_cases, load_cases,
)


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    def test_loads_supported_json_shapes(self):
        payloads = [
            {"question": "Q1", "answer": "A1"},
            [{"question": "Q1", "answer": "A1"}],
            {"cases": [{"question": "Q1", "answer": "A1"}]},
            {"Q1": "A1", "Q2": "A2"},
        ]
        for payload in payloads:
            with patch.object(Path, "read_text", return_value=json.dumps(payload)):
                cases = load_cases(Path("eval-cases.json"))
                self.assertGreaterEqual(len(cases), 1)
                self.assertEqual(cases[0].question, "Q1")

    async def test_exact_evaluation_builds_aggregate_report(self):
        async def answerer(question):
            answer = "Five apartments." if question == "How many?" else "Wrong"
            return SimpleNamespace(answer=answer, verification_status="verified")

        report = await evaluate_cases(
            [
                EvaluationCase(question="How many?", answer="  five   apartments. "),
                EvaluationCase(question="Other?", answer="Expected"),
            ],
            judge_mode="exact",
            answerer=answerer,
        )

        self.assertEqual(report.summary.total, 2)
        self.assertEqual(report.summary.passed, 1)
        self.assertEqual(report.summary.pass_rate, 0.5)

    async def test_semantic_evaluation_accepts_injected_structured_judge(self):
        async def answerer(question):
            return SimpleNamespace(answer="There are five.", verification_status="verified")

        async def judge(case, actual, status, threshold):
            return EvaluationJudgment(
                score=0.9,
                passed=True,
                correctness=1.0,
                completeness=0.8,
                groundedness=0.9,
                reason="Semantically equivalent.",
            )

        report = await evaluate_cases(
            [EvaluationCase(question="How many?", answer="5")],
            judge_mode="semantic",
            answerer=answerer,
            judge=judge,
        )

        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.results[0].score, 0.9)
        self.assertEqual(report.summary.average_score, 0.9)

    async def test_case_errors_do_not_expose_exception_messages(self):
        async def answerer(question):
            raise RuntimeError("password=secret-value")

        report = await evaluate_cases(
            [EvaluationCase(question="Q", answer="A")],
            judge_mode="exact",
            answerer=answerer,
        )

        self.assertEqual(report.results[0].error, "RuntimeError")
        self.assertNotIn("secret-value", report.model_dump_json())


if __name__ == "__main__":
    unittest.main()
