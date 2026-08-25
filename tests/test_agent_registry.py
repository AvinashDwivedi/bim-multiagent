import unittest

from pydantic import ValidationError

from agents.strict_schema import ensure_strict_json_schema

from bim_agents.registry import build_agent_registry
from bim_agents.models import EvidenceWorkstreamResult


class AgentRegistryTests(unittest.TestCase):
    def test_registry_contains_isolated_scout_and_query_workers(self):
        registry = build_agent_registry(project_knowledge={"large": "not serialized"})
        self.assertEqual(
            [
                registry.task_architect.name,
                registry.schema_scout.name,
                registry.query_executor.name,
            ],
            ["Task Architect", "BIM Schema Scout", "BIM Query Worker"],
        )

    def test_every_output_schema_is_strict(self):
        registry = build_agent_registry()
        for item in (
            registry.task_architect, registry.schema_scout, registry.query_executor
        ):
            with self.subTest(agent=item.name):
                ensure_strict_json_schema(item.output_type.model_json_schema())

    def test_scout_can_discover_and_map_but_cannot_execute(self):
        registry = build_agent_registry()
        tool_names = {tool.name for tool in registry.schema_scout.tools}
        self.assertIn("inspect_evidence_capabilities", tool_names)
        self.assertIn("activate_project_mapping", tool_names)
        self.assertIn("inspect_graph_structure", tool_names)
        self.assertIn("register_schema_mapping", tool_names)
        self.assertNotIn("inspect_query_capabilities", tool_names)
        self.assertNotIn("submit_query_plan", tool_names)
        self.assertNotIn("submit_geometry_query", tool_names)

    def test_query_worker_has_no_discovery_or_mapping_tools(self):
        registry = build_agent_registry()
        self.assertEqual(
            {tool.name for tool in registry.query_executor.tools},
            {"inspect_query_capabilities", "submit_query_plan", "submit_geometry_query"},
        )

    def test_project_knowledge_is_not_duplicated_in_system_prompts(self):
        marker = "SECRET-LARGE-PROJECT-KNOWLEDGE"
        registry = build_agent_registry(project_knowledge={"marker": marker})
        for agent in (
            registry.task_architect, registry.schema_scout, registry.query_executor
        ):
            self.assertNotIn(marker, str(agent.instructions))

    def test_model_agents_cannot_issue_parallel_calls_against_one_context(self):
        registry = build_agent_registry()
        for agent in (
            registry.task_architect, registry.schema_scout, registry.query_executor
        ):
            with self.subTest(agent=agent.name):
                self.assertFalse(agent.model_settings.parallel_tool_calls)

    def test_mapping_only_workstream_cannot_claim_completion(self):
        with self.assertRaises(ValidationError):
            EvidenceWorkstreamResult(status="query_completed", package_id="p")
        unsupported = EvidenceWorkstreamResult(
            status="unsupported", package_id="p",
            limitations=["No exact classification boundary was proven."],
        )
        self.assertEqual(unsupported.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
