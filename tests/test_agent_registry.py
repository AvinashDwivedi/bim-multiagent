import unittest

from agents.strict_schema import ensure_strict_json_schema

from bim_agents.registry import build_agent_registry


class AgentRegistryTests(unittest.TestCase):
    def test_registry_contains_exactly_four_agents(self):
        registry = build_agent_registry()
        self.assertEqual(
            [
                registry.supervisor.name,
                registry.graph_explorer.name,
                registry.query_planner.name,
                registry.verifier.name,
            ],
            ["Supervisor", "Graph Explorer", "Query Planner", "Verifier"],
        )

    def test_every_output_schema_is_strict(self):
        registry = build_agent_registry()
        for item in (
            registry.supervisor,
            registry.graph_explorer,
            registry.query_planner,
            registry.verifier,
        ):
            with self.subTest(agent=item.name):
                ensure_strict_json_schema(item.output_type.model_json_schema())

    def test_supervisor_has_iterative_investigation_tools(self):
        registry = build_agent_registry()
        tool_names = {tool.name for tool in registry.supervisor.tools}
        self.assertEqual(
            tool_names,
            {
                "create_task_contract",
                "inspect_query_capabilities",
                "inspect_project_knowledge",
                "inspect_graph_structure",
                "inspect_node_inventory",
                "profile_node_properties",
                "search_nodes",
                "search_properties",
                "search_property_values",
                "register_schema_mapping",
                "submit_query_plan",
                "submit_project_knowledge_query",
                "submit_geometry_query",
                "replay_and_verify",
            },
        )
        self.assertNotIn("shell", tool_names)
        self.assertNotIn("apply_patch", tool_names)


if __name__ == "__main__":
    unittest.main()
