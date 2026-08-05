import asyncio
import unittest
from types import SimpleNamespace

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.observability import BimRunHooks, configure_logging


class RealtimeEventTests(unittest.TestCase):
    def test_repeated_agent_invocations_get_unique_ordered_events(self):
        events = []
        state = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        context = SimpleNamespace(context=state)
        agent = SimpleNamespace(name="Verification Agent")
        hooks = BimRunHooks(configure_logging(verbose=False), event_sink=events.append)

        async def exercise():
            await hooks.on_agent_start(context, agent)
            await hooks.on_agent_end(context, agent, None)
            await hooks.on_agent_start(context, agent)
            await hooks.on_agent_end(context, agent, None)

        asyncio.run(exercise())
        starts = [event for event in events if event["type"] == "agent_start"]
        ends = [event for event in events if event["type"] == "agent_end"]
        self.assertEqual([event["sequence"] for event in starts], [1, 2])
        self.assertEqual([event["id"] for event in starts], ["agent-1", "agent-2"])
        self.assertEqual([event["id"] for event in ends], ["agent-1", "agent-2"])


if __name__ == "__main__":
    unittest.main()
