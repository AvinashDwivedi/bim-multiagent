from __future__ import annotations

from pathlib import Path

import pytest

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


def test_rejects_incomplete_project_contract(tmp_path: Path) -> None:
    with pytest.raises(ProjectError):
        RawProjectTools(tmp_path)
