import unittest

from bim_agents.cypher_handler import CypherQueryHandler


class CypherQueryHandlerTests(unittest.TestCase):
    def test_executes_parameterized_scoped_calculation_and_records_it(self):
        calls = []

        def query(statement, parameters):
            calls.append((statement, parameters))
            return [{"count": 5}]

        handler = CypherQueryHandler(query, ["model.ifc"])
        rows = handler.execute(
            "MATCH (n) WHERE n.source IN $allowed_sources RETURN count(DISTINCT n.id) AS count",
            {},
        )

        self.assertEqual(rows, [{"count": 5}])
        self.assertEqual(calls[0][1]["allowed_sources"], ["model.ifc"])
        self.assertEqual(handler.audit_log()[0]["row_count"], 1)

    def test_rejects_writes_and_unscoped_queries(self):
        handler = CypherQueryHandler(lambda statement, parameters: [], ["model.ifc"])

        with self.assertRaises(PermissionError):
            handler.execute("MATCH (n) SET n.changed = true RETURN n", {})
        with self.assertRaises(PermissionError):
            handler.execute("MATCH (n) RETURN count(n)", {})


if __name__ == "__main__":
    unittest.main()
