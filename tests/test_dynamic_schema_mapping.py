import json
import unittest

from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import BimRunContext, BimTaskContract, ProjectScope
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
    PipelineContext,
    _entity_from_registered_mapping,
    _execute_plan,
    _resolve_canonical_type,
    _validate_hierarchy_binding_coverage,
    _rank_hybrid_candidates,
    _rank_embedding_candidates,
    inspect_queryable_node_types,
    profile_project_classification_hierarchy,
)


class _Ontology:
    def resolve_space_function(self, term):
        return "apartment" if "apartment" in term else None

    def resolve_element_classes(self, term):
        return []

    def retrieval_terms_for(self, term):
        return ["lighting switch", "SwitchButton"] if "switch" in term else []

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


class _DiscoveryBim:
    ontology = _Ontology()

    def query(self, cypher, parameters=None):
        if "RETURN labels(n) AS labels" in cypher:
            return [{
                "labels": ["BIMElement", "IfcBuildingElementProxy"],
                "node_count": 37,
                "property_key_groups": [[
                    "source", "GlobalID", "name", "Category", "Family", "Type",
                    *[f"Verbose Property {index}" for index in range(100)],
                ]],
                "sample_names": ["LD_Lighting Switch Water Proof:Single"],
            }]
        if "count(DISTINCT n) AS record_count" in cypher:
            return [
                {"value_0": "Lighting Devices", "value_1": "Emergency Exit Sign", "value_2": "Exit Sign", "record_count": 16},
                {"value_0": "Lighting Devices", "value_1": "M_Lighting Switches", "value_2": "Single Pole", "record_count": 6},
                {"value_0": "Lighting Devices", "value_1": "ED_SwitchButton", "value_2": "Button", "record_count": 4},
            ]
        raise AssertionError(f"Unexpected query: {cypher}")


class _SplitSignatureDiscoveryBim(_DiscoveryBim):
    def query(self, cypher, parameters=None):
        if "RETURN labels(n) AS labels" in cypher:
            return [
                {
                    "labels": ["BIMElement", "IfcBuildingElementProxy"],
                    "node_count": 100,
                    "property_key_groups": [["source", "GlobalID", "name"]],
                    "sample_names": ["Generic proxy"],
                },
                {
                    "labels": ["IfcBuildingElementProxy", "RevitElement"],
                    "node_count": 37,
                    "property_key_groups": [[
                        "source", "GlobalID", "Category", "Family", "Type"
                    ]],
                    "sample_names": ["Lighting switch"],
                },
            ]
        return super().query(cypher, parameters)


class DynamicSchemaMappingTests(unittest.TestCase):
    def _switch_mapping_proposal(self, classification_property):
        classification_name = {
            "Category": "category", "Family": "family",
        }.get(classification_property, "switch_classification")
        selected_values = {
            "Category": ["Lighting Devices"],
            "Family": ["M_Lighting Switches", "ED_SwitchButton"],
        }.get(classification_property, ["Switches"])
        return SchemaMappingProposal(
            entity_name="switches", label="IfcBuildingElementProxy",
            identity_property="GlobalID", source_property="source",
            fields=[
                SchemaFieldMapping(
                    semantic_name=classification_name, property=classification_property,
                    ontology_kind="canonical_type",
                ),
                *([SchemaFieldMapping(semantic_name="category", property="Category")]
                  if classification_name != "category" else []),
                *([SchemaFieldMapping(semantic_name="family", property="Family")]
                  if classification_name != "family" else []),
                SchemaFieldMapping(semantic_name="type", property="Type"),
            ],
            value_bindings=[SchemaValueBinding(
                semantic_name=classification_name, property=classification_property,
                user_concept="physical switches",
                matches=[SchemaValueMatch(value=value, similarity=1.0) for value in selected_values],
            )],
            counting_unit="physical switch",
            counting_unit_evidence="One populated unique GlobalID per modeled instance.",
            reasoning_summary="Live classification and identity evidence.",
        )

    def _hierarchy_context(self):
        context = BimRunContext(
            bim=_DiscoveryBim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )
        json.loads(profile_project_classification_hierarchy(
            PipelineContext(context), "IfcBuildingElementProxy",
            ["Category", "Family", "Type"], concept="switch", limit=10,
        ))
        return context

    def test_secondary_classification_binding_cannot_omit_profiled_hierarchy(self):
        with self.assertRaisesRegex(ValueError, "outside the profiled"):
            _validate_hierarchy_binding_coverage(
                self._hierarchy_context(),
                self._switch_mapping_proposal("Identity Data_OmniClass Title"),
            )

    def test_polluted_broad_category_binding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unrelated family/type"):
            _validate_hierarchy_binding_coverage(
                self._hierarchy_context(), self._switch_mapping_proposal("Category")
            )

    def test_complete_profiled_family_binding_passes_hierarchy_guard(self):
        _validate_hierarchy_binding_coverage(
            self._hierarchy_context(), self._switch_mapping_proposal("Family")
        )

    def test_classification_mapping_requires_hierarchy_profile(self):
        context = BimRunContext(
            bim=_DiscoveryBim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )
        with self.assertRaisesRegex(ValueError, "Profile the observed"):
            _validate_hierarchy_binding_coverage(
                context, self._switch_mapping_proposal("Family")
            )

    def test_unresolved_plural_is_not_corrupted_into_invented_canonical_value(self):
        context = BimRunContext(
            bim=_Bim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )

        canonical, _ = _resolve_canonical_type(context, "Switches")

        self.assertEqual(canonical, "switches")
        self.assertNotEqual(canonical, "switche")

    def test_compact_inventory_omits_verbose_property_surface(self):
        context = BimRunContext(
            bim=_DiscoveryBim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )

        result = json.loads(inspect_queryable_node_types(PipelineContext(context)))
        item = result["node_types"][0]

        self.assertEqual(item["property_count"], 106)
        self.assertNotIn("properties", item)
        self.assertIn("Category", item["suggested_properties"])
        self.assertIn("Family", item["suggested_properties"])
        self.assertLessEqual(len(item["suggested_properties"]), 24)

    def test_classification_hierarchy_ranks_leaf_match_above_broad_category_noise(self):
        context = BimRunContext(
            bim=_DiscoveryBim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )

        result = json.loads(profile_project_classification_hierarchy(
            PipelineContext(context), "IfcBuildingElementProxy",
            ["Category", "Family", "Type"], concept="switch", limit=10,
        ))

        self.assertEqual(result["candidate_branch_count"], 3)
        self.assertIn(
            result["branches"][0]["path"]["Family"],
            {"M_Lighting Switches", "ED_SwitchButton"},
        )
        self.assertEqual(result["branches"][-1]["path"]["Family"], "Emergency Exit Sign")
        self.assertFalse(result["branches"][0]["broad_only_match"])
        self.assertTrue(result["branches"][-1]["broad_only_match"])
        self.assertGreater(
            result["branches"][0]["lexical_similarity"],
            result["branches"][-1]["lexical_similarity"],
        )

    def test_classification_profiler_unions_properties_across_label_signatures(self):
        context = BimRunContext(
            bim=_SplitSignatureDiscoveryBim(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
        )

        result = json.loads(profile_project_classification_hierarchy(
            PipelineContext(context), "IfcBuildingElementProxy",
            ["Category", "Family", "Type"], concept="switch", limit=10,
        ))

        self.assertEqual(result["candidate_branch_count"], 3)

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

    def test_hybrid_ranking_uses_expanded_cross_language_vocabulary(self):
        candidates = [
            {"node_ref": "wall", "embedding_text": "type: IfcWall; name: exterior wall"},
            {"node_ref": "switch", "embedding_text": "category: Lighting Devices; family: lighting switch"},
        ]

        ranked = _rank_hybrid_candidates(
            "user concept: מפסקים; BIM vocabulary: lighting switch | Lighting Devices",
            [1.0, 0.0],
            [[0.9, 0.1], [0.1, 0.9]],
            candidates,
        )

        self.assertEqual(ranked[0]["node_ref"], "switch")
        self.assertEqual(ranked[0]["lexical_similarity"], 1.0)
        self.assertGreater(ranked[0]["similarity"], ranked[0]["embedding_similarity"])

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
                    SchemaFieldMapping(
                        semantic_name="dwelling_unit_number",
                        property="dwelling_number",
                        ontology_kind="aggregate_identity",
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

    def test_distinct_numeric_property_returns_converted_measurement_values(self):
        class NumericBim(_Bim):
            def query(self, cypher, parameters=None):
                self.queries.append((cypher, parameters or {}))
                if "AS candidate_count" in cypher:
                    return [{"candidate_count": 3}]
                if "AS total_groups" in cypher:
                    return [{"total_groups": 2, "total_records": 3}]
                if "AS value" in cypher and "ORDER BY count DESC" in cypher:
                    return [{"value": 14.5, "count": 2}, {"value": 4.0, "count": 1}]
                raise AssertionError(f"Unexpected query: {cypher}")

        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-elevations",
            proposal=SchemaMappingProposal(
                entity_name="socket_boxes",
                label="ObservedSocket",
                identity_property="GlobalID",
                source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="elevation_from_level",
                    property="Elevation from Level",
                    data_type="number",
                    source_unit="cm",
                    unit="mm",
                    conversion_factor=10,
                    conversion_basis="centimetres to millimetres",
                )],
                counting_unit="physical socket box",
                counting_unit_evidence="GlobalID is unique per socket box.",
                reasoning_summary="The observed numeric property is the requested elevation.",
            ),
            node_count=3,
            populated_identity_count=3,
            distinct_identity_count=3,
        )
        bim = NumericBim()
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            schema_mappings={mapping.mapping_id: mapping},
        )

        result = _execute_plan(context, BimQueryPlan(
            entity="socket_boxes",
            mapping_id=mapping.mapping_id,
            operation="distinct",
            group_by="elevation_from_level",
        ))

        self.assertEqual(result["claim"]["details"], ["145 mm: 2 records", "40 mm: 1 records"])
        self.assertEqual(result["claim"]["measurement"]["conversion_factor"], 10)
        self.assertIn("toFloat(n.`Elevation from Level`)", bim.queries[-1][0])

    def test_registered_exact_value_binding_bypasses_canonical_singularization(self):
        class SwitchBim(_Bim):
            def query(self, cypher, parameters=None):
                self.queries.append((cypher, parameters or {}))
                if "count(DISTINCT n.`GlobalID`) AS count" in cypher:
                    return [{"count": 21}]
                raise AssertionError(f"Unexpected query: {cypher}")

        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-switches",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcBuildingElementProxy",
                identity_property="GlobalID", source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="classification",
                    property="Identity Data_OmniClass Title",
                    ontology_kind="canonical_type",
                )],
                value_bindings=[SchemaValueBinding(
                    semantic_name="classification",
                    property="Identity Data_OmniClass Title",
                    user_concept="physical switches",
                    matches=[SchemaValueMatch(value="Switches", similarity=1.0)],
                )],
                counting_unit="physical switch",
                counting_unit_evidence="GlobalID is unique per modeled switch.",
                reasoning_summary="Observed exact live classification and identity.",
            ),
            node_count=459, populated_identity_count=459, distinct_identity_count=459,
        )
        bim = SwitchBim()
        context = BimRunContext(
            bim=bim,
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Count switches", operation="count", entity_concept="switches",
                required_outputs=["switch count"], success_criteria=["Verified count"],
            ),
            schema_mappings={mapping.mapping_id: mapping},
        )
        plan = BimQueryPlan(
            entity="switches", mapping_id=mapping.mapping_id, operation="count",
            filters=[BimFilter(field="classification", operator="equals", value="Switches")],
            answer_key="switch-count", satisfies=["switch count"],
        )

        result = _execute_plan(context, plan)

        self.assertEqual(result["claim"]["value"], 21)
        self.assertEqual(bim.queries[-1][1]["filter_0"], ["switches"])

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

    def test_contract_entity_is_preferred_without_explicit_mapping_id(self):
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

        self.assertEqual(plan.mapping_id, "")
        self.assertIsNone(label)
        self.assertEqual(entity.identity_property, "object_id")


if __name__ == "__main__":
    unittest.main()
