import unittest

from agents.strict_schema import ensure_strict_json_schema

from bim_agents.registry import build_agent_registry


class AgentSchemaTests(unittest.TestCase):
    def test_every_agent_output_is_strict_json_schema_compatible(self):
        registry = build_agent_registry()
        agents = [
            registry.supervisor,
            registry.query_agent,
            registry.verifier,
        ]
        for agent in agents:
            with self.subTest(agent=agent.name):
                ensure_strict_json_schema(agent.output_type.model_json_schema())


if __name__ == "__main__":
    unittest.main()
