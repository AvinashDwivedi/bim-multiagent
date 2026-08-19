import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, ProjectScope
from bim_agents.schema_mapping import (
    RegisteredSchemaMapping,
    SchemaFieldMapping,
    SchemaMappingProposal,
    SchemaRelationshipStep,
    SchemaValueBinding,
    SchemaValueMatch,
)
from bim_agents.tools import (
    BimFilter,
    BimQueryPlan,
    _entity_from_registered_mapping,
    _execute_plan,
    _rank_embedding_candidates,
)


class _Ontology:
    def resolve_space_function(self, term):
        return "apartment" if "apartment" in term else None

    def resolve_element_classes(self, term):
        return []

    def retrieval_terms_for(self, term):
        return []

    def absence_notes_for_query(self, term):
        return []


class _Bim:
    ontology = _Ontology()

    def __init__(self):
        self.queries = []

    def query(self, cypher, parameters=None):
        self.queries.append((cypher, parameters or {}))
        if "RETURN DISTINCT n.`floor_value` AS value" in cypher:
            return [{"value": "00 begane grond"}]
        if "count(DISTINCT n.`space_key`) AS count" in cypher:
            return [{"count": 5}]
        raise AssertionError(f"Unexpected query: {cypher}")


class DynamicSchemaMappingTests(unittest.TestCase):
    def test_permit_knowledge_absence_can_be_queried_after_space_mapping(self):
        context = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            schema_mappings={"mapping-spaces": object()},
        )
        plan = BimQueryPlan(entity="permit_knowledge", operation="list")

        entity, live_label = _entity_from_registered_mapping(context, plan)

        self.assertEqual(entity.node_type, "permit_knowledge")
        self.assertIsNone(live_label)
        self.assertEqual(plan.mapping_id, "")

    def test_embedding_candidates_are_ranked_by_type_and_name_similarity(self):
        candidates = [
            {"node_ref": "wall", "embedding_text": "type: IfcWall; name: exterior wall"},
            {"node_ref": "space", "embedding_text": "type: IfcSpace; name: apartment"},
        ]

        ranked = _rank_embedding_candidates(
            [1.0, 0.0],
            [[0.0, 1.0], [0.9, 0.1]],
            candidates,
        )

        self.assertEqual(ranked[0]["node_ref"], "space")
        self.assertGreater(ranked[0]["similarity"], ranked[1]["similarity"])

    def test_count_executes_from_registered_live_mapping_without_contract_entity(self):
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-live",
            proposal=SchemaMappingProposal(
                entity_name="live_spaces",
                label="ObservedSpace",
                identity_property="space_key",
                source_property="source",
                counting_unit="apartment",
                counting_unit_evidence="The live space key is the populated unique apartment identifier.",
                fields=[
                    SchemaFieldMapping(
                        semantic_name="type", property="usage_value", ontology_kind="canonical_type"
                    ),
                    SchemaFieldMapping(
                        semantic_name="level", property="floor_value", ontology_kind="level"
                    ),
                ],
                reasoning_summary="Observed unique IDs and representative apartment/floor values.",
            ),
            node_count=10,
            populated_identity_count=10,
            distinct_identity_count=10,
        )
        bim = _Bim()
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question="how many apartments are there on ground floor?",
            schema_mappings={mapping.mapping_id: mapping},
        )
        plan = BimQueryPlan(
            entity="live_spaces",
            mapping_id=mapping.mapping_id,
            operation="count",
            filters=[
                BimFilter(field="type", operator="equals", value="apartments"),
                BimFilter(field="level", operator="equals", value="ground floor"),
            ],
        )

        result = _execute_plan(context, plan)

        self.assertEqual(result["claim"]["value"], 5)
        count_query, parameters = bim.queries[-1]
        self.assertIn("MATCH (n:`ObservedSpace`)", count_query)
        self.assertIn("n.`usage_value`", count_query)
        self.assertIn("n.`floor_value`", count_query)
        self.assertIn("n.`space_key`", count_query)
        self.assertEqual(parameters["filter_0"], ["apartment"])
        self.assertEqual(parameters["filter_1"], ["00 begane grond"])

    def test_count_traverses_registered_relationship_path(self):
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-related",
            proposal=SchemaMappingProposal(
                entity_name="related_spaces",
                label="ObservedSpace",
                identity_property="space_key",
                source_property="source",
                relationship_path=[SchemaRelationshipStep(
                    from_label="BuildingStorey",
                    relationship_type="CONTAINS",
                    to_label="ObservedSpace",
                    direction="outgoing",
                    purpose="Reach spaces contained by a storey.",
                )],
                counting_unit="space",
                counting_unit_evidence="Each space_key identifies one reached space.",
                reasoning_summary="Observed storey-to-space containment.",
            ),
            node_count=10,
            populated_identity_count=10,
            distinct_identity_count=10,
        )
        bim = _Bim()
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question="How many contained spaces are there?",
            schema_mappings={mapping.mapping_id: mapping},
        )

        result = _execute_plan(context, BimQueryPlan(
            entity="related_spaces",
            mapping_id=mapping.mapping_id,
            operation="count",
        ))

        self.assertEqual(result["claim"]["value"], 5)
        self.assertIn(
            "MATCH (p0:`BuildingStorey`)-[:`CONTAINS`]->(n:`ObservedSpace`)",
            bim.queries[-1][0],
        )

    def test_registered_user_constraint_requires_exact_value_filter(self):
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-category",
            proposal=SchemaMappingProposal(
                entity_name="live_spaces",
                label="ObservedSpace",
                identity_property="space_key",
                source_property="source",
                counting_unit="apartment",
                counting_unit_evidence="The live space key is the populated unique apartment identifier.",
                fields=[
                    SchemaFieldMapping(
                        semantic_name="category", property="usage_value"
                    ),
                    SchemaFieldMapping(
                        semantic_name="level", property="floor_value", ontology_kind="level"
                    ),
                ],
                value_bindings=[SchemaValueBinding(
                    semantic_name="category",
                    property="usage_value",
                    user_concept="requested dwelling category",
                    matches=[SchemaValueMatch(value="apartment", similarity=0.72)],
                )],
                reasoning_summary="Live evidence.",
            ),
            node_count=10,
            populated_identity_count=10,
            distinct_identity_count=10,
        )
        context = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question="count records for the requested category",
            schema_mappings={mapping.mapping_id: mapping},
        )
        plan = BimQueryPlan(
            entity="live_spaces",
            mapping_id=mapping.mapping_id,
            operation="count",
            filters=[BimFilter(field="level", operator="equals", value="7")],
        )

        with self.assertRaisesRegex(ValueError, "registered exact value binding"):
            _execute_plan(context, plan)

    def test_single_registered_mapping_is_authoritative_over_plan_display_name(self):
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-only",
            proposal=SchemaMappingProposal(
                entity_name="discovered_apartment_spaces",
                label="ObservedSpace",
                identity_property="space_key",
                source_property="source",
                counting_unit="apartment",
                counting_unit_evidence="The live space key is the populated unique apartment identifier.",
                fields=[SchemaFieldMapping(semantic_name="type", property="usage_value")],
                reasoning_summary="Live evidence.",
            ),
            node_count=10,
            populated_identity_count=10,
            distinct_identity_count=10,
        )
        context = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            schema_mappings={mapping.mapping_id: mapping},
        )
        plan = BimQueryPlan(entity="spaces", operation="count")

        entity, label = _entity_from_registered_mapping(context, plan)

        self.assertEqual(plan.mapping_id, "mapping-only")
        self.assertEqual(label, "ObservedSpace")
        self.assertEqual(entity.identity_property, "space_key")


if __name__ == "__main__":
    unittest.main()
