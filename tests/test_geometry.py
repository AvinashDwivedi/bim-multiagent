import json
from types import SimpleNamespace
import unittest

from bim_agents.geometry import calculate_project_geometry
from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import (
    BimRunContext, BimTaskContract, ModelNodeSignature, OutputSpec,
    ProjectModelProfile, ProjectScope,
)
from bim_agents.tools import (
    GeometryQueryPlan, PipelineContext, _validate_answer_metadata,
    _validate_governed_calculation_schema,
)
from ontology.concept_ontology import load_ontology


KNOWLEDGE = {
    "massing_sections": {
        "slab_label": "IfcSlab", "predefined_type_property": "PredefinedType",
        "predefined_type_value": "ROOF", "quantity_set": "Qto_SlabBaseQuantities",
        "area_quantity": "GrossArea", "primary_plate_min_largest_fraction": 0.5,
        "semantics": "Test massing rule.",
    },
    "tower_facade": {
        "wall_label": "IfcWall", "wall_name_property": "name",
        "wall_name_value": "Klimaatgevel", "orientation_property": "Orientatie",
        "wall_quantity_set": "Qto_WallBaseQuantities",
        "opaque_area_quantity": "GrossSideArea", "opening_labels": ["IfcWindow", "IfcDoor"],
        "external_property": "IsExternal", "window_quantity_set": "Qto_WindowBaseQuantities",
        "door_quantity_set": "Qto_DoorBaseQuantities", "opening_area_quantity": "Area",
        "repeated_floor_min_count": 2, "semantics": "Test facade rule.",
    },
    "tower_floor_area": {
        "space_label": "IfcSpace", "type_property": "canonical_type",
        "floor_plate_type": "residential_zone", "basis_property": "Bepalingsmethode",
        "floor_plate_basis": "BVO", "area_property": "canonical_area_m2",
        "level_property": "canonical_level", "repeated_floor_min_count": 2,
        "repeated_area_precision": 3, "semantics": "Test tower rule.",
    },
}


class FakeBim:
    def query(self, cypher, parameters):
        if "IfcSlab" in cypher:
            def slab(area, elevation, level):
                return {
                    "roof_id": f"roof-{level}",
                    "roof_name": "Dakvloer",
                    "quantities": json.dumps({"Qto_SlabBaseQuantities": {"GrossArea": area}}),
                    "level": level,
                    "elevation_m": elevation,
                }

            return [
                slab(100, 5, "01"), slab(90, 8, "02"), slab(380, 11, "03"),
                slab(381, 14, "04"), slab(562, 38, "12"),
            ]
        if "IfcBuildingStorey" in cypher:
            return [{"highest_storey_elevation_m": 41.0}]
        if "IfcSpace" in cypher:
            return [
                {"level": "05", "area_m2": 586.248, "records": 1},
                {"level": "06", "area_m2": 586.248, "records": 1},
                {"level": "ground", "area_m2": 234.34, "records": 1},
            ]
        if "IfcWall" in cypher:
            return [
                {
                    "id": f"wall-{level}",
                    "level": level,
                    "quantities": json.dumps({"Qto_WallBaseQuantities": {"GrossSideArea": 269.836}}),
                }
                for level in ("05", "06", "07")
            ]
        return [
            {
                "id": f"window-{level}",
                "level": level,
                "ifc_class": "IfcWindow",
                "quantities": json.dumps({"Qto_WindowBaseQuantities": {"Area": 208.512}}),
            }
            for level in ("05", "06", "07")
        ]


class GeometryTests(unittest.TestCase):
    def _validate_typed_geometry_outputs(self, calculation, specs):
        knowledge = load_ontology(
            client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
            project_id="858ef0f0-454a-11f1-8957-1fe1b101e373",
        ).bim_query_knowledge
        outputs = [spec.key for spec in specs]
        context = BimRunContext(
            bim=SimpleNamespace(
                ontology=SimpleNamespace(bim_query_knowledge=knowledge),
            ),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            task_contract=BimTaskContract(
                goal="Validate governed geometry outputs",
                operation="list",
                entity_concept="geometry",
                required_outputs=outputs,
                output_specs=specs,
                success_criteria=["Every output is calculation-backed."],
            ),
        )
        _validate_answer_metadata(
            PipelineContext(context),
            GeometryQueryPlan(
                calculation=calculation,
                answer_key=calculation,
                satisfies=outputs,
            ),
        )

    def test_derives_three_primary_section_heights(self):
        report = calculate_project_geometry(
            "section_heights", FakeBim(), ["source"], KNOWLEDGE
        )
        self.assertEqual(report.verification_status, "verified")
        self.assertIn("11 m, 14 m, 38 m", report.answer)
        self.assertIn("highest defined storey reference elevation is 41 m", report.answer)

    def test_section_height_recipe_accepts_measurement_group_and_basis_outputs(self):
        self._validate_typed_geometry_outputs("section_heights", [
            OutputSpec(key="section heights", kind="grouped_summary"),
            OutputSpec(key="height basis", kind="fact"),
        ])

    def test_derives_repeated_tower_facade_ratio(self):
        report = calculate_project_geometry(
            "facade_opening_percentage", FakeBim(), ["source"], KNOWLEDGE
        )
        self.assertEqual(report.verification_status, "verified")
        self.assertIn("43.6% (approximately 45%)", report.answer)
        self.assertIn("Qualifying opening-area numerator", report.claims[0].details[0])
        self.assertEqual(report.claims[0].coverage.candidate_count, 6)
        self.assertTrue(report.claims[0].coverage.exhaustive)
        self.assertEqual(report.claims[0].method, "typical_facade_opening_ratio")

    def test_facade_recipe_accepts_measurement_coverage_and_basis_outputs(self):
        self._validate_typed_geometry_outputs("facade_opening_percentage", [
            OutputSpec(key="opening percentage", kind="measurement"),
            OutputSpec(key="input coverage", kind="coverage"),
            OutputSpec(key="calculation basis", kind="fact"),
        ])

    def test_derives_governed_tower_maximum_floor_area(self):
        report = calculate_project_geometry(
            "tower_max_floor_area", FakeBim(), ["source"], KNOWLEDGE
        )
        self.assertEqual(report.verification_status, "verified")
        self.assertIn("586.25 m² BVO", report.answer)
        self.assertIn("05, 06", report.answer)
        self.assertEqual(
            report.claims[0].details,
            ["Attaining floor/storey: 05.", "Attaining floor/storey: 06.", "Area measurement basis: BVO."],
        )
        self.assertEqual(report.claims[0].measurement.canonical_unit, "m²")

    def test_tower_area_recipe_accepts_measurement_identity_and_basis_outputs(self):
        self._validate_typed_geometry_outputs("tower_max_floor_area", [
            OutputSpec(key="maximum area", kind="measurement"),
            OutputSpec(key="attaining floors", kind="list"),
            OutputSpec(key="area basis", kind="fact"),
        ])

    def test_dispatches_allowlisted_governed_scope_comparison(self):
        class ScopeBim:
            def query(self, cypher, parameters):
                return [
                    {"scope_key": "all_units", "count": 100},
                    {"scope_key": "ground_floor", "count": 5},
                ]

        knowledge = {"unit_scopes": {
            "calculation": "physical_unit_scope_comparison",
            "recipe": "scope_comparison",
            "entity": {
                "label": "IfcSpace", "identity_property": "EenheidNummer",
                "filters": [{"property": "canonical_type", "values": ["apartment"]}],
            },
            "scopes": [
                {"key": "all_units"},
                {"key": "ground_floor", "filters": [
                    {"property": "canonical_level", "values": ["00 begane grond"]},
                ]},
            ],
            "baseline_scope": "all_units",
            "counting_unit": "physical apartments",
            "semantics": "Count shared duplex identity only once.",
        }}

        report = calculate_project_geometry(
            "physical_unit_scope_comparison", ScopeBim(), ["source"], knowledge,
        )

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(report.claims[0].value, "all_units=100; ground_floor=5")
        self.assertIn("all_units: 100 physical apartments", report.claims[0].details)
        self.assertEqual(report.claims[0].method, "scope_comparison")

    def test_dispatches_domain_neutral_property_coverage_without_connectivity_claim(self):
        class CoverageBim:
            def query(self, cypher, parameters):
                return [{"candidate_count": 20, "populated_count": 12}]

        knowledge = {"asset_tags": {
            "calculation": "asset_tag_coverage",
            "recipe": "property_coverage",
            "entity": {"label": "IfcProduct", "identity_property": "GlobalID"},
            "property": "AssetTag",
            "counting_unit": "maintainable assets",
            "assignment_unit": "asset-tag assignment",
            "semantics": "Measure explicit asset-tag completeness without inferring identity links.",
        }}

        report = calculate_project_geometry(
            "asset_tag_coverage", CoverageBim(), ["source"], knowledge,
        )

        self.assertEqual(report.verification_status, "verified")
        self.assertEqual(report.claims[0].value, 8)
        self.assertEqual(report.claims[0].coverage.missing_count, 8)
        self.assertIn("12 have an explicit asset-tag assignment", report.answer)
        self.assertIn("does not establish", report.limitations[0])

    def test_optional_group_relationship_may_be_absent_from_current_ingest(self):
        knowledge = {"endpoint_counts": {
            "calculation": "endpoint_counts",
            "recipe": "composite_grouped_count",
            "components": [{
                "key": "lights", "label": "IfcFlowTerminal",
                "identity_property": "GlobalID", "source_property": "source",
                "group_property": "Level",
                "filters": [{"property": "Category", "values": ["Lighting Fixtures"]}],
                "group_relationship": {
                    "relationship_type": "SPATIALLY_CONTAINS",
                    "target_label": "IfcBuildingStorey",
                    "target_property": "canonical_level",
                    "direction": "either",
                },
            }],
            "counting_unit": "endpoints", "group_unit": "floor",
            "semantics": "Use an optional relationship and retain unassigned records.",
        }}
        context = BimRunContext(
            bim=SimpleNamespace(
                ontology=SimpleNamespace(bim_query_knowledge=knowledge),
            ),
            scope=ProjectScope(
                client_id="c", project_id="p", allowed_sources=["model.ifc"],
            ),
            graph_contract=load_graph_contract(),
            model_profile=ProjectModelProfile(
                authorized_source_count=1,
                node_types=[ModelNodeSignature(
                    labels=["BIMElement", "IfcFlowTerminal"], node_count=10,
                    properties=["GlobalID", "source", "Level", "Category"],
                )],
                relationship_types=[],
                scope_note="Authorized source only.", freshness_note="Test profile.",
            ),
        )

        _validate_governed_calculation_schema(context, "endpoint_counts")


if __name__ == "__main__":
    unittest.main()
