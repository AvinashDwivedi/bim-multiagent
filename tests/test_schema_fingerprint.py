import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.knowledge import compute_live_schema_fingerprint
from bim_agents.models import BimRunContext, ProjectScope


class _Graph:
    def __init__(self, relationship_type="PORT_OF"):
        self.relationship_type = relationship_type

    def query(self, statement, parameters):
        if "UNWIND labels(n)" in statement:
            return [{"label": "IfcFlowSegment", "properties": ["source", "GlobalID"]}]
        if "MATCH (a)-[r]->(b)" in statement:
            return [{
                "from_labels": ["IfcDistributionPort"],
                "relationship_type": self.relationship_type,
                "to_labels": ["IfcFlowSegment"],
                "property_key_groups": [["connection_type"]],
            }]
        raise AssertionError(statement)


def _context(graph, sources=None):
    return BimRunContext(
        bim=graph,
        scope=ProjectScope(
            client_id="client", project_id="project",
            allowed_sources=sources or ["model.ifc"],
        ),
        graph_contract=load_graph_contract(),
    )


class SchemaFingerprintTests(unittest.TestCase):
    def test_relationship_topology_changes_fingerprint(self):
        first = compute_live_schema_fingerprint(_context(_Graph("PORT_OF")))
        second = compute_live_schema_fingerprint(_context(_Graph("CONNECTED_TO")))
        self.assertNotEqual(first, second)

    def test_authorized_source_set_changes_fingerprint(self):
        first = compute_live_schema_fingerprint(_context(_Graph(), ["a.ifc"]))
        second = compute_live_schema_fingerprint(_context(_Graph(), ["a.ifc", "b.ifc"]))
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
