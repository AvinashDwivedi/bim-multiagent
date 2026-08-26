import json
import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, BimTaskContract, ProjectScope
from bim_agents.schema_mapping import RegisteredSchemaMapping, SchemaMappingProposal
from bim_agents.tools import BimQueryPlan, PipelineContext, query_bim


class _CountingBim:
    def __init__(self):
        self.queries = []

    def query(self, cypher, parameters=None):
        self.queries.append((cypher, parameters or {}))
        if "count(DISTINCT n.`GlobalID`) AS count" in cypher:
            return [{"count": 7}]
        raise AssertionError(f"Unexpected query: {cypher}")


def _mapping():
    return RegisteredSchemaMapping(
        mapping_id="mapping-switches",
        proposal=SchemaMappingProposal(
            entity_name="switches", label="IfcBuildingElementProxy",
            identity_property="GlobalID", source_property="source",
            counting_unit="physical switch",
            counting_unit_evidence="GlobalID is populated and unique per modeled switch.",
            reasoning_summary="Live identity evidence supports this mapping.",
        ),
        node_count=7, populated_identity_count=7, distinct_identity_count=7,
    )


def _context(bim, *, sources=None):
    mapping = _mapping()
    return BimRunContext(
        bim=bim,
        scope=ProjectScope(
            client_id="client-a", project_id="project-a",
            allowed_sources=sources or ["electrical.ifc"],
        ),
        graph_contract=load_graph_contract(),
        task_contract=BimTaskContract(
            goal="Count switches", operation="count", entity_concept="switches",
            required_outputs=["switch count"], success_criteria=["Exact scoped count"],
        ),
        schema_mappings={mapping.mapping_id: mapping},
        schema_fingerprint="schema-a",
    )


def _plan(**updates):
    values = {
        "entity": "switches", "mapping_id": "mapping-switches",
        "operation": "count", "answer_key": "switch-count",
        "satisfies": ["switch count"],
    }
    values.update(updates)
    return BimQueryPlan(**values)


class QueryResultCacheTests(unittest.TestCase):
    def test_fully_defaulted_exact_plan_reuses_original_evidence_and_bim_result(self):
        bim = _CountingBim()
        context = _context(bim)

        first = query_bim(PipelineContext(context), _plan())
        second = query_bim(PipelineContext(context), _plan(
            limit=10, include_details=False, include_in_answer=True,
            role="answer_producing", sort_order="ascending",
        ))

        self.assertEqual(second, first)
        self.assertEqual(len(bim.queries), 1)
        self.assertEqual(len(context.evidence), 1)
        self.assertEqual(len(context.artifacts), 2)
        payload = json.loads(first)
        self.assertEqual(payload["evidence_id"], next(iter(context.evidence)))
        self.assertEqual(payload["claim"]["value"], 7)

    def test_authorized_source_scope_change_invalidates_cache(self):
        bim = _CountingBim()
        context = _context(bim)

        first = json.loads(query_bim(PipelineContext(context), _plan()))
        context.scope.allowed_sources.append("linked-electrical.ifc")
        second = json.loads(query_bim(PipelineContext(context), _plan()))

        self.assertEqual(len(bim.queries), 2)
        self.assertEqual(len(context.evidence), 2)
        self.assertNotEqual(first["evidence_id"], second["evidence_id"])
        self.assertEqual(
            bim.queries[-1][1]["allowed_sources"],
            ["electrical.ifc", "linked-electrical.ifc"],
        )

    def test_cache_is_isolated_between_run_contexts(self):
        bim = _CountingBim()
        first_context = _context(bim)
        second_context = _context(bim)

        first = json.loads(query_bim(PipelineContext(first_context), _plan()))
        second = json.loads(query_bim(PipelineContext(second_context), _plan()))

        self.assertEqual(len(bim.queries), 2)
        self.assertNotEqual(first["evidence_id"], second["evidence_id"])
        self.assertEqual(len(first_context.evidence), 1)
        self.assertEqual(len(second_context.evidence), 1)

    def test_active_mapping_change_invalidates_cache(self):
        bim = _CountingBim()
        context = _context(bim)

        first = json.loads(query_bim(PipelineContext(context), _plan()))
        context.schema_mappings["mapping-switches"] = _mapping().model_copy(update={
            "node_count": 8,
        })
        second = json.loads(query_bim(PipelineContext(context), _plan()))

        self.assertEqual(len(bim.queries), 2)
        self.assertNotEqual(first["evidence_id"], second["evidence_id"])


if __name__ == "__main__":
    unittest.main()
