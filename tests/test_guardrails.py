import asyncio
import unittest
from types import SimpleNamespace

from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.observability import BimRunHooks, configure_logging
from bim_agents.graph_contract import load_graph_contract


class GuardrailTests(unittest.TestCase):
    def test_llm_budget_stops_the_run(self):
        state = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            max_llm_calls=1,
        )
        context = SimpleNamespace(context=state)
        agent = SimpleNamespace(name="test-agent")
        hooks = BimRunHooks(configure_logging(verbose=False))
        asyncio.run(hooks.on_llm_start(context, agent, None, []))
        with self.assertRaisesRegex(RuntimeError, "llm_calls exceeded 1"):
            asyncio.run(hooks.on_llm_start(context, agent, None, []))


if __name__ == "__main__":
    unittest.main()
