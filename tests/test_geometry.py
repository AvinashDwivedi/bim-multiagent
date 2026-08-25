import json
import unittest

from bim_agents.geometry import calculate_project_geometry


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
    def test_derives_three_primary_section_heights(self):
        report = calculate_project_geometry(
            "section_heights", FakeBim(), ["source"], KNOWLEDGE
        )
        self.assertEqual(report.verification_status, "verified")
        self.assertIn("11 m, 14 m, 38 m", report.answer)
        self.assertIn("highest defined storey reference elevation is 41 m", report.answer)

    def test_derives_repeated_tower_facade_ratio(self):
        report = calculate_project_geometry(
            "facade_opening_percentage", FakeBim(), ["source"], KNOWLEDGE
        )
        self.assertEqual(report.verification_status, "verified")
        self.assertIn("43.6% (approximately 45%)", report.answer)

    def test_derives_governed_tower_maximum_floor_area(self):
        report = calculate_project_geometry(
            "tower_max_floor_area", FakeBim(), ["source"], KNOWLEDGE
        )
        self.assertEqual(report.verification_status, "verified")
        self.assertIn("586.25 m² BVO", report.answer)
        self.assertIn("05, 06", report.answer)


if __name__ == "__main__":
    unittest.main()
