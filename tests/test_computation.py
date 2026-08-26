import unittest

from pydantic import ValidationError

from bim_agents.computation import (
    DEFAULT_COMPUTATION_REGISTRY,
    CompositeGroupedCountRecipe,
    PropertyCoverageRecipe,
    execute_governed_computation,
)


class RecordingGraph:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def query(self, statement, parameters):
        self.calls.append((statement, parameters))
        return self.responses.pop(0)


class GovernedComputationTests(unittest.TestCase):
    def test_composite_grouped_count_is_scoped_parameterized_and_deduplicated(self):
        graph = RecordingGraph([
            [
                {"group_value": "Level 01", "count": 8},
                {"group_value": "Level 02", "count": 5},
            ],
            [{"total_count": 13}],
        ])
        config = {
            "recipe": "composite_grouped_count",
            "components": [
                {
                    "key": "fixtures",
                    "label": "IfcBuildingElementProxy",
                    "identity_property": "GlobalID",
                    "source_property": "source",
                    "group_property": "canonical_level",
                    "filters": [{"property": "Category", "values": ["Electrical Fixtures"]}],
                },
                {
                    "key": "lights",
                    "label": "IfcLightFixture",
                    "identity_property": "GlobalID",
                    "source_property": "source",
                    "group_property": "canonical_level",
                },
            ],
            "counting_unit": "physical endpoint instances",
            "group_unit": "modelled level",
            "semantics": "Combine the governed endpoint populations and deduplicate GlobalID values.",
        }

        result = execute_governed_computation(
            calculation_key="endpoint-counts-by-level",
            governed_config=config,
            query=graph.query,
            allowed_sources=["authorized.ifc"],
        )

        self.assertEqual(result.total_count, 13)
        self.assertEqual(result.rows[0], {"group": "Level 01", "count": 8})
        self.assertEqual(len(graph.calls), 2)
        for statement, parameters in graph.calls:
            self.assertIn("$allowed_sources", statement)
            self.assertNotIn("Electrical Fixtures", statement)
            self.assertEqual(parameters["allowed_sources"], ["authorized.ifc"])
            self.assertEqual(
                parameters["component_0_filter_0"], ["Electrical Fixtures"]
            )
        self.assertIn("count(DISTINCT identity_key)", graph.calls[0][0])
        self.assertEqual(result.provenance["identity_namespace"], "global")
        self.assertEqual(len(result.config_digest), 64)
        self.assertEqual(len(result.result_digest), 64)

    def test_scope_comparison_returns_baseline_deltas_and_percentages(self):
        graph = RecordingGraph([[
            {"scope_key": "all_units", "count": 20},
            {"scope_key": "ground_floor", "count": 5},
        ]])
        config = {
            "recipe": "scope_comparison",
            "entity": {
                "label": "IfcSpace",
                "identity_property": "EenheidNummer",
                "filters": [{"property": "canonical_type", "values": ["apartment"]}],
            },
            "scopes": [
                {"key": "all_units"},
                {
                    "key": "ground_floor",
                    "filters": [{"property": "canonical_level", "values": ["Ground"]}],
                },
            ],
            "baseline_scope": "all_units",
            "counting_unit": "physical apartments",
            "semantics": "Compare one physical-unit identity under governed additive scopes.",
        }

        result = execute_governed_computation(
            calculation_key="unit-scope-comparison",
            governed_config=config,
            query=graph.query,
            allowed_sources=["authorized.ifc"],
        )

        rows = {row["scope"]: row for row in result.rows}
        self.assertEqual(result.total_count, 20)
        self.assertEqual(rows["ground_floor"]["difference_from_baseline"], -15)
        self.assertEqual(rows["ground_floor"]["percent_of_baseline"], 25.0)
        statement, parameters = graph.calls[0]
        self.assertNotIn("Ground", statement)
        self.assertEqual(parameters["scope_1_filter_0"], ["Ground"])
        self.assertEqual(parameters["scope_0_base_filter_0"], ["apartment"])
        self.assertEqual(parameters["scope_1_base_filter_0"], ["apartment"])

    def test_zero_baseline_has_no_invented_percentage(self):
        graph = RecordingGraph([[
            {"scope_key": "all", "count": 0},
            {"scope_key": "subset", "count": 0},
        ]])
        config = {
            "recipe": "scope_comparison",
            "entity": {"label": "IfcSpace", "identity_property": "GlobalID"},
            "scopes": [{"key": "all"}, {"key": "subset"}],
            "baseline_scope": "all",
            "counting_unit": "spaces",
            "semantics": "Compare two governed scopes.",
        }

        result = execute_governed_computation(
            calculation_key="empty-scope-comparison",
            governed_config=config,
            query=graph.query,
            allowed_sources=["authorized.ifc"],
        )

        self.assertIsNone(result.rows[1]["percent_of_baseline"])
        self.assertIn("undefined", result.limitations[0])

    def test_property_coverage_is_generic_scoped_and_does_not_claim_connectivity(self):
        graph = RecordingGraph([[{
            "candidate_count": 488,
            "populated_count": 183,
        }]])
        config = {
            "recipe": "property_coverage",
            "entity": {
                "label": "BIMElement",
                "identity_property": "GlobalID",
                "filters": [{"property": "discipline", "values": ["electrical"]}],
            },
            "property": "feeding_panel",
            "missing_values": ["Unassigned", "-"],
            "counting_unit": "electrical components",
            "assignment_unit": "feeding-panel assignment",
            "semantics": "Measure explicit logical panel assignment on the governed population.",
        }

        result = execute_governed_computation(
            calculation_key="panel_assignment_coverage",
            governed_config=config,
            query=graph.query,
            allowed_sources=["authorized.ifc"],
        )

        self.assertEqual(result.total_count, 488)
        self.assertEqual(result.rows, [
            {"status": "populated", "count": 183},
            {"status": "missing", "count": 305},
        ])
        self.assertEqual(result.provenance["fill_rate_percent"], 37.5)
        self.assertIn("does not establish", result.limitations[0])
        statement, parameters = graph.calls[0]
        self.assertIn("$allowed_sources", statement)
        self.assertIn("$missing_values", statement)
        self.assertNotIn("electrical", statement)
        self.assertEqual(parameters["coverage_filter_0"], ["electrical"])

    def test_property_coverage_rejects_unsafe_property(self):
        with self.assertRaises(ValidationError):
            PropertyCoverageRecipe.model_validate({
                "recipe": "property_coverage",
                "entity": {"label": "BIMElement", "identity_property": "GlobalID"},
                "property": "panel`) MATCH (secret",
                "counting_unit": "components",
                "assignment_unit": "panel assignment",
                "semantics": "Unsafe properties must fail before compilation.",
            })

    def test_composite_recipe_supports_governed_exact_exclusions(self):
        graph = RecordingGraph([[], [{"total_count": 0}]])
        config = {
            "recipe": "composite_grouped_count",
            "components": [{
                "key": "eligible",
                "label": "IfcBuildingElementProxy",
                "identity_property": "GlobalID",
                "group_property": "canonical_level",
                "filters": [{
                    "property": "Family", "operator": "not_in",
                    "values": ["Opening", "Cover"],
                }],
            }],
            "counting_unit": "eligible instances",
            "group_unit": "modelled level",
            "semantics": "Exclude governed non-endpoint families.",
        }

        execute_governed_computation(
            calculation_key="eligible_by_level", governed_config=config,
            query=graph.query, allowed_sources=["authorized.ifc"],
        )

        self.assertIn("NOT (n.`Family` IN $component_0_filter_0)", graph.calls[0][0])
        self.assertEqual(graph.calls[0][1]["component_0_filter_0"], ["Opening", "Cover"])

    def test_composite_group_can_fall_back_to_scoped_spatial_relationship(self):
        graph = RecordingGraph([[{"group_value": "GF", "count": 2}], [{"total_count": 2}]])
        config = {
            "recipe": "composite_grouped_count",
            "components": [{
                "key": "orphaned_switches", "label": "IfcBuildingElementProxy",
                "identity_property": "GlobalID", "group_property": "canonical_level",
                "group_relationship": {
                    "relationship_type": "SPATIALLY_CONTAINS",
                    "target_label": "IfcBuildingStorey",
                    "target_property": "canonical_level",
                    "direction": "either",
                },
            }],
            "counting_unit": "switches", "group_unit": "storey",
            "semantics": "Use direct level first and scoped IFC containment as fallback.",
        }

        result = execute_governed_computation(
            calculation_key="switches_by_storey", governed_config=config,
            query=graph.query, allowed_sources=["authorized.ifc"],
        )

        statement = graph.calls[0][0]
        self.assertIn("OPTIONAL MATCH (n)-[:`SPATIALLY_CONTAINS`]-(group_node:`IfcBuildingStorey`)", statement)
        self.assertIn("coalesce(n.`canonical_level`, group_node.`canonical_level`)", statement)
        self.assertIn("group_node.`source` IN $allowed_sources", statement)
        self.assertEqual(result.rows, [{"group": "GF", "count": 2}])

    def test_composite_group_fallback_skips_blank_exported_text(self):
        graph = RecordingGraph([[{"group_value": "GF", "count": 1}], [{"total_count": 1}]])
        config = {
            "recipe": "composite_grouped_count",
            "components": [{
                "key": "fixtures", "label": "IfcFlowTerminal",
                "identity_property": "GlobalID", "group_property": "Level",
                "group_fallback_properties": ["Schedule Level", "canonical_level"],
            }],
            "counting_unit": "fixtures", "group_unit": "storey",
            "semantics": "Use the first populated governed level property.",
        }

        execute_governed_computation(
            calculation_key="fixtures_by_storey", governed_config=config,
            query=graph.query, allowed_sources=["authorized.ifc"],
        )

        statement = graph.calls[0][0]
        self.assertIn("trim(toString(n.`Level`)) <> ''", statement)
        self.assertIn("trim(toString(n.`Schedule Level`)) <> ''", statement)
        self.assertIn("THEN n.`canonical_level` END", statement)

    def test_composite_group_can_join_authorized_sources_by_stable_identity(self):
        graph = RecordingGraph([[{"group_value": "GF", "count": 2}], [{"total_count": 2}]])
        config = {
            "recipe": "composite_grouped_count",
            "components": [{
                "key": "switches", "label": "IfcBuildingElementProxy",
                "identity_property": "GlobalID", "group_property": "Level",
                "cross_source_identity_join": True,
                "group_relationship": {
                    "relationship_type": "SPATIALLY_CONTAINS",
                    "target_label": "IfcBuildingStorey",
                    "target_property": "canonical_level",
                    "direction": "either",
                },
            }],
            "counting_unit": "switches", "group_unit": "storey",
            "semantics": "Join authoring and IFC records on stable GlobalID.",
        }

        result = execute_governed_computation(
            calculation_key="joined_switches", governed_config=config,
            query=graph.query, allowed_sources=["authoring.json", "model.ifc"],
        )

        statement = graph.calls[0][0]
        self.assertIn("OPTIONAL MATCH (peer:`IfcBuildingElementProxy`)", statement)
        self.assertIn("peer.`GlobalID` = n.`GlobalID`", statement)
        self.assertIn("peer_group_node.`source` IN $allowed_sources", statement)
        self.assertEqual(
            result.provenance["cross_source_identity_join_components"], ["switches"]
        )

    def test_missing_aggregate_rows_are_not_reinterpreted_as_verified_zeroes(self):
        grouped_graph = RecordingGraph([[], []])
        grouped_config = {
            "recipe": "composite_grouped_count",
            "components": [{
                "key": "spaces",
                "label": "IfcSpace",
                "identity_property": "GlobalID",
                "group_property": "level",
            }],
            "counting_unit": "spaces",
            "group_unit": "levels",
            "semantics": "Count spaces by level.",
        }
        with self.assertRaisesRegex(RuntimeError, "omitted its total population"):
            execute_governed_computation(
                calculation_key="spaces-by-level",
                governed_config=grouped_config,
                query=grouped_graph.query,
                allowed_sources=["authorized.ifc"],
            )

        comparison_graph = RecordingGraph([[
            {"scope_key": "all", "count": 0},
        ]])
        comparison_config = {
            "recipe": "scope_comparison",
            "entity": {"label": "IfcSpace", "identity_property": "GlobalID"},
            "scopes": [{"key": "all"}, {"key": "subset"}],
            "baseline_scope": "all",
            "counting_unit": "spaces",
            "semantics": "Compare two governed scopes.",
        }
        with self.assertRaisesRegex(RuntimeError, "omitted configured scopes: subset"):
            execute_governed_computation(
                calculation_key="space-comparison",
                governed_config=comparison_config,
                query=comparison_graph.query,
                allowed_sources=["authorized.ifc"],
            )

    def test_unknown_recipe_and_empty_scope_fail_before_query(self):
        graph = RecordingGraph([])
        with self.assertRaisesRegex(ValueError, "Unknown governed computation recipe"):
            execute_governed_computation(
                calculation_key="unknown-calculation",
                governed_config={"recipe": "raw_cypher"},
                query=graph.query,
                allowed_sources=["authorized.ifc"],
            )
        with self.assertRaisesRegex(PermissionError, "non-empty authorized source"):
            execute_governed_computation(
                calculation_key="unknown-calculation",
                governed_config={"recipe": "scope_comparison"},
                query=graph.query,
                allowed_sources=[],
            )
        self.assertEqual(graph.calls, [])

    def test_unsafe_identifiers_and_duplicate_component_keys_are_rejected(self):
        with self.assertRaises(ValidationError):
            CompositeGroupedCountRecipe.model_validate({
                "recipe": "composite_grouped_count",
                "components": [{
                    "key": "bad",
                    "label": "IfcSpace`) DELETE n",
                    "identity_property": "GlobalID",
                    "group_property": "level",
                }],
                "counting_unit": "spaces",
                "group_unit": "levels",
                "semantics": "Unsafe identifiers must fail before compilation.",
            })
        with self.assertRaisesRegex(ValidationError, "component keys must be unique"):
            CompositeGroupedCountRecipe.model_validate({
                "recipe": "composite_grouped_count",
                "components": [
                    {
                        "key": "spaces",
                        "label": "IfcSpace",
                        "identity_property": "GlobalID",
                        "group_property": "level",
                    },
                    {
                        "key": "spaces",
                        "label": "IfcSpace",
                        "identity_property": "GlobalID",
                        "group_property": "level",
                    },
                ],
                "counting_unit": "spaces",
                "group_unit": "levels",
                "semantics": "Duplicate component keys are ambiguous.",
            })

    def test_default_registry_is_frozen_and_allowlisted(self):
        self.assertEqual(
            DEFAULT_COMPUTATION_REGISTRY.available_recipes(),
            {
                "composite_grouped_count": 1,
                "property_coverage": 1,
                "scope_comparison": 1,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            DEFAULT_COMPUTATION_REGISTRY.register(
                "another_recipe", CompositeGroupedCountRecipe, lambda *_: None
            )


if __name__ == "__main__":
    unittest.main()
