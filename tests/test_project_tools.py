from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from bim_agent.ifc_analysis import IfcAnalysisEngine
from bim_agent.project_tools import ProjectError, RawProjectTools


def test_raw_tools_discover_exact_three_file_contract(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    inspection = tools.execute("inspect_project", {})
    assert inspection["record_count"] == 22
    assert {item["kind"] for item in inspection["source_files"]} == {"tree", "properties", "ifc"}
    assert any(item["key"] == "Dimensions.Length" for item in inspection["property_keys"])


def test_model_selected_ids_can_be_aggregated_without_semantic_rules(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    result = tools.execute("aggregate_records", {
        "object_ids": ["43", "44"],
        "operation": "sum",
        "field": "Dimensions.Length",
        "output_unit": "m",
    })
    assert result["value"] == 5.5
    assert result["unit"] == "m"


def test_raw_search_returns_paths_and_matching_properties(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    result = tools.execute("search_records", {
        "terms": ["Alpha Switch"], "match": "all", "limit": 100,
    })
    assert result["total_matches"] >= 3
    assert any("Lighting Devices" in item["path"] for item in result["records"])


def test_ifc_tool_searches_sqlite_exports_generically(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    result = tools.execute("search_ifc", {"terms": ["Water Flow Alarm"], "limit": 20})
    assert result["format"] == "SQLite"
    assert result["total_matches"] == 1
    assert result["matches"][0]["table"] == "_objects_val"


def test_model_authored_workspace_joins_and_aggregates_project_data(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    description = tools.execute("describe_bim_workspace", {})
    assert description["read_only"] is True
    assert {item["name"] for item in description["tables"]} >= {
        "records", "properties", "tree_nodes", "tree_edges",
    }
    assert "_objects_val" in description["attached_ifc_tables"]
    assert any("compound UNION" in item and "outer SELECT" in item for item in description["guidance"])

    result = tools.execute("query_bim_workspace", {
        "sql": """
            SELECT p.property_value AS level, COUNT(DISTINCT r.object_id) AS instances
            FROM records AS r
            JOIN properties AS p ON p.object_id = r.object_id
            WHERE p.property_key = ? AND r.name LIKE ?
            GROUP BY p.property_value
            ORDER BY p.property_value
        """,
        "parameters": ["Constraints.Level", "Pipe [%"],
        "row_limit": 20,
    })
    assert result["rows"] == [{"level": "GF", "instances": 2}]
    assert result["source_files"] == ["model-properties.json"]


def test_workspace_exposes_ifc_database_read_only(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    result = tools.execute("query_bim_workspace", {
        "sql": "SELECT value FROM ifc_source._objects_val WHERE id = ?",
        "parameters": [1],
        "row_limit": 1000,
    })
    assert "Water Flow Alarm" in result["rows"][0]["value"]
    assert result["source_files"] == ["model.ifc"]

    with pytest.raises(ProjectError, match="read-only"):
        tools.execute("query_bim_workspace", {
            "sql": "DELETE FROM records", "parameters": [], "row_limit": 1000,
        })


def test_workspace_indexes_step_ifc_entities_and_relationships(sample_data: Path) -> None:
    ifc_path = sample_data / "model.ifc"
    ifc_path.write_text(
        """ISO-10303-21;
DATA;
#1=IFCPROJECT('project');
#2=IFCRELAGGREGATES('relationship',$,$,$,#1,(#3));
#3=IFCSITE('site');
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )
    tools = RawProjectTools(sample_data)
    result = tools.execute("query_bim_workspace", {
        "sql": """
            SELECT e.entity_type, r.target_step_id
            FROM ifc_entities AS e
            JOIN ifc_references AS r ON r.source_step_id = e.step_id
            WHERE e.step_id = 2
            ORDER BY r.target_step_id
        """,
        "parameters": [],
        "row_limit": 1000,
    })
    assert result["rows"] == [
        {"entity_type": "IFCRELAGGREGATES", "target_step_id": 1},
        {"entity_type": "IFCRELAGGREGATES", "target_step_id": 3},
    ]


def test_workspace_tool_is_available_to_the_model(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    definitions = {item["name"]: item for item in tools.definitions()}
    assert {
        "describe_bim_workspace", "query_bim_workspace", "analyze_ifc_geometry",
        "rank_ifc_geometry", "reconcile_populations", "analyze_ifc_graph",
    } <= set(definitions)
    assert "read-only SQLite" in definitions["query_bim_workspace"]["description"]
    assert all(definitions[name]["strict"] is True for name in definitions)
    assert set(definitions["query_bim_workspace"]["parameters"]["required"]) == {
        "sql", "parameters", "row_limit",
    }


def test_scope_profile_exposes_complete_hierarchy_without_fixed_depth(sample_data: Path) -> None:
    profile = RawProjectTools(sample_data).scope_profile()
    lighting = next(item for item in profile["hierarchy_nodes"] if item["name"] == "Lighting Devices")
    assert lighting["depth"] == 1
    assert lighting["descendant_count"] == 6
    assert lighting["leaf_descendant_count"] == 3
    assert any("Alpha Switch [1001]" in path for path in lighting["leaf_path_samples"])


def test_geometry_tool_reports_unavailable_for_sqlite_ifc_export(sample_data: Path) -> None:
    result = RawProjectTools(sample_data).execute("analyze_ifc_geometry", {
        "record_object_ids": [],
        "ifc_step_ids": [],
        "global_ids": [],
        "entity_types": [],
        "name_terms": ["Switch"],
        "metrics": ["bounding_box"],
        "max_results": 20,
    })
    assert result["available"] is False
    assert result["source_file"] == "model.ifc"


def test_available_geometry_path_returns_a_valid_observation(
    sample_data: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEntity:
        def id(self) -> int:
            return 7

    class FakeModel:
        def by_id(self, step_id: int) -> Any:
            return FakeEntity() if step_id == 7 else None

        def by_type(self, _entity_type: str) -> list[Any]:
            return []

    engine = IfcAnalysisEngine(sample_data / "model.ifc", [])
    monkeypatch.setattr(IfcAnalysisEngine, "available", property(lambda _self: True))
    engine._model = FakeModel()
    monkeypatch.setattr(engine, "_geometry_row", lambda entity, metrics, scale: {
        "ifc_step_id": entity.id(),
        "bounding_box": {"size": {"x": 1.0, "y": 2.0, "z": 3.0}, "volume_m3": 6.0},
    })

    result = engine.analyze_geometry({
        "record_object_ids": [], "ifc_step_ids": [7], "global_ids": [],
        "entity_types": [], "name_terms": [], "metrics": ["bounding_box"], "max_results": 20,
    })

    assert result["available"] is True
    assert result["returned_entities"] == 1
    assert result["rows"][0]["ifc_step_id"] == 7


def test_rank_geometry_compares_complete_filtered_inventory(
    sample_data: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = IfcAnalysisEngine(sample_data / "model.ifc", [])
    monkeypatch.setattr(IfcAnalysisEngine, "available", property(lambda _self: True))
    engine._geometry_inventory = [
        {"ifc_step_id": 1, "entity_type": "IfcWall", "record_object_ids": [], "solid_volume_m3": 2.0},
        {"ifc_step_id": 2, "entity_type": "IfcWall", "record_object_ids": [], "solid_volume_m3": 5.0},
        {"ifc_step_id": 3, "entity_type": "IfcDoor", "record_object_ids": [], "solid_volume_m3": 9.0},
    ]
    result = engine.rank_geometry({
        "record_object_ids": [], "ifc_step_ids": [], "global_ids": [],
        "entity_types": ["IfcWall"], "name_terms": [],
        "metric": "solid_volume", "order": "descending", "max_results": 10,
    })
    assert result["matching_entities"] == 2
    assert [item["ifc_step_id"] for item in result["rows"]] == [2, 1]


def test_population_reconciliation_exposes_duplicates_missing_ids_and_overlaps(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    result = tools.execute("reconcile_populations", {
        "populations": [
            {"label": "pipes", "object_ids": ["43", "43", "44"]},
            {"label": "review", "object_ids": ["44", "missing"]},
        ],
    })
    assert result["union_unique_object_ids"] == 3
    assert result["union_found_records"] == 2
    assert result["populations"][0]["duplicate_occurrences"] == 1
    assert result["populations"][1]["missing_object_ids"] == ["missing"]
    assert result["pairwise_overlaps"][0]["sample_object_ids"] == ["44"]


def test_ifc_graph_traversal_retains_relationship_evidence(
    sample_data: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = RawProjectTools(sample_data)
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE ifc_relationships(relationship_step_id INTEGER, relationship_type TEXT, "
        "relating_step_id INTEGER, related_step_id INTEGER, relating_role TEXT, related_role TEXT)"
    )
    connection.executemany("INSERT INTO ifc_relationships VALUES (?, ?, ?, ?, ?, ?)", [
        (100, "IfcRelConnectsPorts", 1, 2, "RelatingPort", "RelatedPort"),
        (101, "IfcRelConnectsPorts", 2, 3, "RelatingPort", "RelatedPort"),
        (102, "IfcRelAssignsToGroup", 9, 3, "RelatingGroup", "RelatedObjects"),
    ])
    monkeypatch.setattr(tools, "_workspace_database", lambda: connection)
    result = tools.execute("analyze_ifc_graph", {
        "start_step_ids": [1], "target_step_ids": [3],
        "relationship_types": ["IfcRelConnectsPorts"], "direction": "forward",
        "max_depth": 4, "max_paths": 20,
    })
    assert result["returned_paths"] == 1
    assert result["paths"][0]["depth"] == 2
    assert [edge["relationship_step_id"] for edge in result["paths"][0]["edges"]] == [100, 101]


def test_rejects_incomplete_project_contract(tmp_path: Path) -> None:
    with pytest.raises(ProjectError):
        RawProjectTools(tmp_path)
