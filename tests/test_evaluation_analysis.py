import json
import unittest

from bim_agents.evaluation_analysis import analyze_report, compare_reports


class EvaluationAnalysisTests(unittest.TestCase):
    def test_aggregate_analysis_excludes_case_content_and_scope(self):
        report = [{
            "question": "How many rooms?",
            "expected_answer": 42,
            "actual_answer": "42 rooms",
            "client_id": "653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
            "is_correct": False,
            "pipeline_report": {
                "verification_status": "insufficient_evidence",
                "failure_categories": ["retrieval_failure"],
                "semantic_checks": [
                    {"name": "population_complete", "passed": False, "explanation": "private"},
                    {"name": "authorized_scope", "passed": True},
                ],
                "workstream_diagnostics": [
                    {"specialist": "quantity", "status": "turn_limit", "package_id": "project-id"}
                ],
            },
        }]
        result = analyze_report(report)
        encoded = json.dumps(result)
        self.assertEqual(result["total_cases"], 1)
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(result["pipeline_status_counts"]["insufficient_evidence"], 1)
        self.assertEqual(result["semantic_check_counts"]["population_complete"]["failed"], 1)
        self.assertEqual(result["implicated_specialist_counts"]["quantity"], 1)
        for secret in ("How many rooms", "42", "653fbe80", "private", "project-id"):
            self.assertNotIn(secret, encoded)

    def test_list_and_nested_shapes_and_safe_allow_lists(self):
        result = analyze_report({"results": [
            {"evaluation": {"correct": True}, "status": "verified",
             "semantic_checks": [{"name": "measurement_basis_matches", "passed": True}]},
            {"evaluation": {"correct": False, "category": "made-up-project-value"},
             "verification_status": "conflict", "stages_used": ["verify", "not-a-stage"],
             "specialist": "geometry"},
        ]})
        self.assertEqual(result["passed_cases"], 1)
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(result["pipeline_status_counts"]["verified"], 1)
        self.assertEqual(result["pipeline_status_counts"]["conflict"], 1)
        self.assertEqual(result["failure_category_counts"]["conflict"], 1)
        self.assertEqual(result["failure_category_counts"]["other"], 1)
        self.assertEqual(result["implicated_stage_counts"]["verify"], 1)

    def test_regression_comparison_is_aggregate_and_anonymous(self):
        baseline = [{"is_correct": True, "verification_status": "verified",
                     "semantic_checks": [{"name": "population_complete", "passed": True}]}]
        current = [{"is_correct": False, "verification_status": "conflict",
                    "semantic_checks": [{"name": "population_complete", "passed": False}],
                    "question": "secret question", "actual_answer": 987654}]
        result = compare_reports(current, baseline)
        self.assertEqual(result["deltas"]["failed_cases"], 1)
        self.assertEqual(result["outcome_transitions"]["passed_to_failed"], 1)
        self.assertEqual(result["semantic_check_deltas"]["population_complete"]["failed"], 1)
        encoded = json.dumps(result)
        self.assertNotIn("secret question", encoded)
        self.assertNotIn("987654", encoded)

    def test_evaluator_report_envelope_and_runtime_stage_names(self):
        result = analyze_report({
            "run_id": "private-run-id",
            "summary": {"correct": 1, "total": 2},
            "results": [
                {"question": "private", "correct": True,
                 "pipeline_status": "insufficient_evidence",
                 "failure_categories": ["workstream_error"],
                 "stages_used": ["Schema Mapper", "Verifier"]},
                {"question": "private", "correct": False,
                 "pipeline_status": "insufficient_evidence",
                 "failure_categories": ["workstream_turn_limit"],
                 "stages_used": ["Query Planner", "Knowledge Curator"]},
            ],
        })
        self.assertEqual(result["total_cases"], 2)
        self.assertEqual(result["implicated_stage_counts"], {
            "knowledge_curator": 1, "query_planner": 1,
            "schema_mapper": 1, "verifier": 1,
        })
        encoded = json.dumps(result)
        self.assertNotIn("private", encoded)
        self.assertNotIn("private-run-id", encoded)


if __name__ == "__main__":
    unittest.main()
