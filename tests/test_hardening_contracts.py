from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bim_agent.agent_loop import (
    _claim_plan_scope_issue,
    _claim_matches_planned_metric,
    _completion_issues,
    _ensure_interpretation_disclosure,
    _exhaustive_output_coverage_issues,
    _materialize_geometry_selection,
    _materialize_record_selection,
    _merge_pagination_result,
    _population_observation_complete,
    _project_derived_identity_output_issue,
    _render_verified_project_claims,
    _sql_execution_scope_issue,
    _sql_filter_source_lineage_issue,
    _sql_query_shape_issue,
    _structured_claim_coverage_issues,
    _tool_execution_scope_issue,
    _validate_tool_arguments,
    ToolObservationError,
)
from bim_agent.claim_verification import verify_structured_claims
from bim_agent.ifc_analysis import _mesh_metrics, _verified_unit_scale
from bim_agent.project_tools import ProjectError, RawProjectTools


def _filter(field: str, value: str, source: str = "records") -> dict:
    return {"source": source, "field": field, "operator": "equals", "value": value}


def test_sql_grouping_rejects_distinct_paths_with_the_same_terminal_column() -> None:
    plan = {
        "metrics": [{
            "group_by": ["tree.parent.name", "tree.child.name"],
        }],
    }

    issue = _sql_query_shape_issue(
        plan,
        {"sql": "SELECT name, COUNT(*) AS n FROM tree_nodes GROUP BY name"},
    )

    assert issue is not None
    assert "collapse to the same SQL terminal column" in issue


def test_sql_grouping_allows_the_same_exact_group_path_for_multiple_metrics() -> None:
    plan = {
        "metrics": [
            {"group_by": ["records.path_depth"]},
            {"group_by": ["records.path_depth"]},
        ],
    }

    issue = _sql_query_shape_issue(
        plan,
        {
            "sql": (
                "SELECT path_depth, COUNT(*) AS n, MAX(object_id) AS last_id "
                "FROM records GROUP BY path_depth"
            )
        },
    )

    assert issue is None


def test_sql_grouping_cannot_substitute_a_records_field_for_a_property_group() -> None:
    plan = {
        "metrics": [{
            "group_by": ["properties.floor"],
        }],
    }
    arguments = {
        "sql": "SELECT floor, COUNT(DISTINCT object_id) AS count FROM records GROUP BY floor",
        "parameters": [],
    }
    result = {
        "project_output_lineage": [
            {
                "output_column": "floor",
                "operation": "direct",
                "input_table": "records",
                "input_column": "floor",
                "project_derived": True,
            },
            {
                "output_column": "count",
                "operation": "distinct_count",
                "input_table": "records",
                "input_column": "object_id",
                "project_derived": True,
            },
        ],
    }

    issue = _sql_query_shape_issue(plan, arguments, result=result)

    assert issue is not None
    assert "exact owning SQL source" in issue


def test_complete_plan_scoped_sql_can_supersede_an_exploratory_cursor_only_with_claim_gate() -> None:
    plan = {
        "answer_shape": "count",
        "population": {
            "description": "All records.",
            "identity_basis": "records.object_id",
            "universe": "all_records",
            "filters": [],
        },
        "metrics": [{
            "name": "instance count",
            "aggregation": "distinct_count",
            "value_field": "records.object_id",
            "unit": "count",
            "source_basis": "derived",
            "null_policy": "not_applicable",
            "group_by": [],
            "result_limit": 0,
        }],
        "relationship": {"direction": "not_applicable", "relationship_types": []},
    }
    evidence = {
        "sql": {
            "tool": "query_bim_workspace",
            "arguments": {
                "sql": "SELECT COUNT(DISTINCT object_id) AS value FROM records",
                "parameters": [],
                "row_limit": 100,
            },
            "result": {
                "columns": ["value"],
                "rows": [{"value": 2}],
                "returned_rows": 1,
                "truncated": False,
                "source_tables": ["records"],
                "source_columns": [{"table": "records", "column": "object_id"}],
                    "project_output_lineage": [{
                        "output_column": "value", "operation": "distinct_count",
                        "input_table": "records", "input_column": "object_id",
                        "project_derived": True,
                    }],
            },
        },
    }
    common = {
        "question": "How many records are in the project?",
        "tool_categories": {"schema", "sql", "review"},
        "outstanding_cursors": {"opaque-next-page"},
        "interpretation_plan": plan,
        "evidence": evidence,
        "reconciliation_required": False,
        "reconciliation_status": "not_required",
        "route": {
            "answer_shape": "count",
            "required_capabilities": ["schema_inspection", "record_query"],
            "required_sources": ["properties"],
            "requirements_enforced": True,
        },
        "observed_sources": {"properties"},
    }

    guarded = _completion_issues(
        **common,
        structured_claim_verification_enabled=True,
    )
    unguarded = _completion_issues(
        **common,
        structured_claim_verification_enabled=False,
    )
    identity_only = copy.deepcopy(common)
    identity_only["evidence"]["sql"]["arguments"]["sql"] = "SELECT object_id FROM records"
    identity_only["evidence"]["sql"]["result"].update({
        "columns": ["object_id"],
        "rows": [{"object_id": "1"}],
        "project_output_lineage": [{
            "output_column": "object_id", "operation": "direct",
            "input_table": "records", "input_column": "object_id", "project_derived": True,
        }],
    })
    identity_only_guarded = _completion_issues(
        **identity_only,
        structured_claim_verification_enabled=True,
    )

    assert "all remaining result pages via fetch_more" not in guarded
    assert "all remaining result pages via fetch_more" in unguarded
    assert "all remaining result pages via fetch_more" in identity_only_guarded


def test_nullable_attribute_count_cannot_certify_a_population_count() -> None:
    plan = {
        "answer_shape": "count",
        "population": {
            "description": "All records.",
            "identity_basis": "records.object_id",
            "universe": "all_records",
            "filters": [],
        },
        "metrics": [{
            "name": "instance count",
            "aggregation": "count",
            "value_field": "records.object_id",
            "unit": "count",
            "source_basis": "derived",
            "null_policy": "not_applicable",
            "group_by": [],
            "result_limit": 0,
        }],
        "relationship": {"direction": "not_applicable", "relationship_types": []},
    }
    evidence = {"sql": {
        "tool": "query_bim_workspace",
        "evidence_class": "project",
        "arguments": {
            "sql": "SELECT MIN(object_id) AS object_id, COUNT(name) AS n FROM records",
            "parameters": [],
            "row_limit": 100,
        },
        "result": {
            "columns": ["object_id", "n"],
            "rows": [{"object_id": "first", "n": 1}],
            "returned_rows": 1,
            "truncated": False,
            "source_tables": ["records"],
            "source_kinds": ["properties"],
            "source_columns": [
                {"table": "records", "column": "object_id"},
                {"table": "records", "column": "name"},
            ],
            "project_output_lineage": [
                {
                    "output_column": "object_id", "operation": "minimum",
                    "input_table": "records", "input_column": "object_id",
                    "project_derived": True,
                },
                {
                    "output_column": "n", "operation": "count",
                    "input_table": "records", "input_column": "name",
                    "project_derived": True,
                },
            ],
        },
    }}
    claim = {
        "claim_id": "count", "evidence_ref": "sql", "kind": "field",
        "row_identity": [{"field": "object_id", "operator": "eq", "value": "first"}],
        "filters": [], "field": "n", "expected_value": 1, "expected_unit": "count",
    }

    issue = _claim_plan_scope_issue(
        claim,
        interpretation_plan=plan,
        evidence=evidence,
        evidence_aliases={},
    )

    assert issue is not None
    assert "nullable attributes" in issue


def test_local_tool_schema_rejects_undeclared_private_arguments(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    definition = next(item for item in tools.definitions() if item["name"] == "get_records")
    arguments = {
        "object_ids": [tools.records[0]["object_id"]],
        "selection_evidence_refs": [],
        "_selection_provenance_validated": True,
    }

    with pytest.raises(ToolObservationError, match="undeclared fields"):
        _validate_tool_arguments("get_records", arguments, definition["parameters"])
    with pytest.raises(ToolObservationError, match="runtime-private"):
        _materialize_record_selection(
            arguments,
            evidence={},
            evidence_aliases={},
            tools=tools,
            interpretation_plan=None,
        )


def test_population_transfer_rejects_a_tree_subset_for_all_records(
    sample_data: Path,
) -> None:
    tools = RawProjectTools(sample_data)
    plan = {
        "population": {
            "description": "All property records.",
            "identity_basis": "records.object_id",
            "universe": "all_records",
            "filters": [],
        },
    }
    evidence = {"tree_subset": {
        "tool": "query_bim_workspace",
        "evidence_class": "project",
        "arguments": {
            "sql": "SELECT object_id FROM tree_nodes", "parameters": [], "row_limit": 100,
        },
        "result": {
            "columns": ["object_id"],
            "rows": [{"object_id": "tree-only"}],
            "returned_rows": 1,
            "truncated": False,
            "source_tables": ["tree_nodes"],
            "source_columns": [{"table": "tree_nodes", "column": "object_id"}],
            "project_output_lineage": [{
                "output_column": "object_id", "operation": "direct",
                "input_table": "tree_nodes", "input_column": "object_id",
                "project_derived": True,
            }],
        },
    }}

    with pytest.raises(ToolObservationError, match="records table population"):
        _materialize_record_selection(
            {"object_ids": [], "selection_evidence_refs": ["tree_subset"]},
            evidence=evidence,
            evidence_aliases={},
            tools=tools,
            interpretation_plan=plan,
        )
    with pytest.raises(ToolObservationError, match="records table population"):
        _materialize_geometry_selection(
            {
                "population_mode": "explicit_selectors",
                "record_object_ids": [], "ifc_step_ids": [], "global_ids": [],
                "entity_types": [], "name_terms": [],
                "selection_evidence_refs": ["tree_subset"],
            },
            evidence=evidence,
            evidence_aliases={},
            tools=tools,
            interpretation_plan=plan,
        )


def test_fetch_more_preserves_original_selector_scope_for_plan_validation() -> None:
    plan = {
        "answer_shape": "list",
        "population": {
            "identity_basis": "records.object_id",
            "universe": "all_records",
            "filters": [],
        },
        "metrics": [{
            "name": "record name",
            "aggregation": "none",
            "value_field": "records.name",
            "unit": "",
            "source_basis": "property",
            "null_policy": "fail",
            "group_by": [],
            "result_limit": 0,
        }],
        "relationship": {"direction": "not_applicable", "relationship_types": []},
    }
    claim = {
        "kind": "field",
        "evidence_ref": "call_1",
        "row_identity": [{"field": "object_id", "operator": "eq", "value": "43"}],
        "filters": [],
        "field": "name",
        "expected_value": "Pipe [4001]",
        "expected_unit": None,
    }
    evidence = {
        "call_1": {
            "tool": "fetch_more",
            "evidence_class": "project",
            "arguments": {"cursor": "opaque"},
            "result": {
                "results": [{"object_id": "43", "name": "Pipe [4001]"}],
                "source_kinds": ["tree", "properties"],
                "pagination_scope_tool": "search_records",
                "pagination_scope_arguments": {"terms": ["Pipe"], "limit": 1},
                "page_tool": "search_records",
                "pagination_complete": True,
                "total_count": 1,
                "returned_count": 1,
                "offset": 0,
                "cursor": None,
            },
        },
    }

    issue = _claim_plan_scope_issue(
        claim,
        interpretation_plan=plan,
        evidence=evidence,
        evidence_aliases={},
    )

    assert issue is not None
    assert "adds a population restriction" in issue


def test_non_sql_property_metric_requires_exact_nested_property_path() -> None:
    metric = {
        "name": "specified name",
        "aggregation": "none",
        "value_field": "Some.Pset.Name",
        "unit": "",
        "source_basis": "property",
        "null_policy": "fail",
        "group_by": [],
        "result_limit": 0,
    }
    entry = {"tool": "get_records"}
    common = {
        "metric": metric,
        "entry": entry,
        "result": {},
        "arguments": {},
        "source_columns": set(),
        "source_kinds": {"properties"},
    }
    top_level = {
        "kind": "field", "field": "name", "expected_unit": None,
        "row_identity": [], "filters": [],
    }
    exact_property = {
        **top_level,
        "field": ["properties", "Some.Pset.Name"],
    }
    different_case_property = {
        **top_level,
        "field": ["properties", "some.pset.name"],
    }

    assert _claim_matches_planned_metric(top_level, **common) is False
    assert _claim_matches_planned_metric(exact_property, **common) is True
    assert _claim_matches_planned_metric(different_case_property, **common) is False
    assert _claim_matches_planned_metric(
        {**exact_property, "expected_unit": "m"}, **common,
    ) is False


def test_filtered_records_cannot_be_certified_from_tree_only_sql() -> None:
    plan = {
        "answer_shape": "list",
        "population": {
            "identity_basis": "records.object_id",
            "universe": "filtered_records",
            "filters": [_filter("name", "Pipe", source="tree")],
        },
        "metrics": [{
            "name": "tree depth", "aggregation": "none", "value_field": "tree.path_depth",
            "unit": "", "source_basis": "tree", "null_policy": "fail", "group_by": [],
        }],
        "relationship": {"direction": "not_applicable", "relationship_types": []},
    }
    evidence = {"tree_sql": {
        "tool": "query_bim_workspace",
        "evidence_class": "project",
        "arguments": {
            "sql": "SELECT object_id, path_depth FROM tree_nodes WHERE name = ?",
            "parameters": ["Pipe"], "row_limit": 100,
        },
        "result": {
            "columns": ["object_id", "path_depth"],
            "rows": [{"object_id": "43", "path_depth": 4}],
            "returned_rows": 1, "truncated": False,
            "source_tables": ["tree_nodes"], "source_kinds": ["tree"],
            "source_columns": [
                {"table": "tree_nodes", "column": "object_id"},
                {"table": "tree_nodes", "column": "path_depth"},
                {"table": "tree_nodes", "column": "name"},
            ],
            "project_output_lineage": [
                {"output_column": "object_id", "input_table": "tree_nodes", "input_column": "object_id", "operation": "direct", "project_derived": True},
                {"output_column": "path_depth", "input_table": "tree_nodes", "input_column": "path_depth", "operation": "direct", "project_derived": True},
            ],
        },
    }}
    claim = {
        "kind": "field", "evidence_ref": "tree_sql",
        "row_identity": [{"field": "object_id", "operator": "eq", "value": "43"}],
        "filters": [{"field": "name", "operator": "eq", "value": "Pipe"}],
        "field": "path_depth", "expected_value": 4, "expected_unit": None,
    }

    issue = _claim_plan_scope_issue(
        claim, interpretation_plan=plan, evidence=evidence, evidence_aliases={},
    )

    assert issue is not None
    assert "records table population" in issue


def test_certifying_sql_rejects_compound_and_hidden_group_scope() -> None:
    planned = [_filter("name", "Door")]

    assert _sql_execution_scope_issue(planned, {
        "sql": "SELECT object_id FROM records WHERE name=? UNION ALL SELECT object_id FROM records",
        "parameters": ["Door"],
    })
    assert _sql_execution_scope_issue(planned, {
        "sql": "SELECT category, COUNT(*) FROM records WHERE name=? GROUP BY category",
        "parameters": ["Door"],
    }) is None  # Grouping is checked against the metric contract, not population parsing.


def test_text_filter_certification_has_one_case_sensitive_execution_semantics() -> None:
    planned = [{
        "source": "records", "field": "records.name",
        "operator": "contains", "value": "Door",
    }]

    like_issue = _sql_execution_scope_issue(planned, {
        "sql": "SELECT object_id FROM records WHERE name LIKE ?",
        "parameters": ["%Door%"],
    })
    glob_issue = _sql_execution_scope_issue(planned, {
        "sql": "SELECT object_id FROM records WHERE name GLOB ?",
        "parameters": ["*Door*"],
    })
    different_case_issue = _sql_execution_scope_issue(planned, {
        "sql": "SELECT object_id FROM records WHERE name GLOB ?",
        "parameters": ["*door*"],
    })

    assert like_issue is not None
    assert glob_issue is None
    assert different_case_issue is not None


def test_raw_and_normalized_eav_columns_are_not_interchangeable() -> None:
    planned = [{
        "source": "properties", "field": "Identity.Name",
        "operator": "contains", "value": "Door",
    }]

    raw_issue = _sql_execution_scope_issue(planned, {
        "sql": (
            "SELECT object_id FROM properties "
            "WHERE property_key = ? AND property_value GLOB ?"
        ),
        "parameters": ["Identity.Name", "*Door*"],
    })
    normalized_issue = _sql_execution_scope_issue(planned, {
        "sql": (
            "SELECT object_id FROM properties "
            "WHERE normalized_key = ? AND normalized_value GLOB ?"
        ),
        "parameters": ["Identity.Name", "*Door*"],
    })

    assert raw_issue is None
    assert normalized_issue is not None


def test_eav_population_filter_requires_key_and_value_and_rejects_multi_property_scan() -> None:
    fire_rating = [_filter("FireRating", "60", source="properties")]

    assert _sql_execution_scope_issue(fire_rating, {
        "sql": "SELECT object_id FROM properties WHERE property_key=?",
        "parameters": ["FireRating"],
    })
    assert _sql_execution_scope_issue(fire_rating, {
        "sql": "SELECT object_id FROM properties WHERE property_value=?",
        "parameters": ["60"],
    })
    assert _sql_execution_scope_issue(fire_rating, {
        "sql": "SELECT object_id FROM properties WHERE property_key=? AND property_value=?",
        "parameters": ["FireRating", "60"],
    }) is None
    assert _sql_execution_scope_issue([
        *fire_rating,
        _filter("Status", "Active", source="properties"),
    ], {
        "sql": (
            "SELECT object_id FROM properties WHERE property_key=? AND property_value=? "
            "AND property_key=? AND property_value=?"
        ),
        "parameters": ["FireRating", "Active", "Status", "60"],
    })


def test_pagination_merge_produces_one_complete_population_observation() -> None:
    evidence = {
        "first": {
            "tool": "search_records",
            "arguments": {"terms": [], "limit": 1, "offset": 0},
            "result": {
                "results": [{"object_id": "1"}],
                "total_count": 2,
                "returned_count": 1,
                "cursor": "next",
                "offset": 0,
            },
        },
    }
    merged = _merge_pagination_result(
        {"cursor": "next"},
        {
            "results": [{"object_id": "2"}],
            "total_count": 2,
            "returned_count": 1,
            "cursor": None,
            "offset": 1,
            "page_tool": "search_records",
        },
        evidence=evidence,
    )

    assert [row["object_id"] for row in merged["results"]] == ["1", "2"]
    assert merged["pagination_complete"] is True
    assert _population_observation_complete({
        "tool": "fetch_more", "arguments": {"cursor": "next"}, "result": merged,
    }) is True


def test_identity_materialization_rejects_any_constant_identity_alias() -> None:
    result = {
        "columns": ["object_id", "record_object_id"],
        "project_output_lineage": [
            {
                "output_column": "object_id", "input_column": "object_id",
                "operation": "direct", "project_derived": True,
            },
            {
                "output_column": "record_object_id", "input_column": None,
                "operation": "opaque", "project_derived": False,
            },
        ],
    }

    assert "record_object_id" in str(_project_derived_identity_output_issue(result, "object_id"))


def test_distinct_identity_projection_retains_direct_lineage(sample_data: Path) -> None:
    result = RawProjectTools(sample_data).execute("query_bim_workspace", {
        "sql": "SELECT DISTINCT object_id FROM records WHERE name LIKE ?",
        "parameters": ["Pipe [%"],
        "row_limit": 20,
    })

    assert result["project_output_lineage"] == [{
        "output_column": "object_id",
        "project_derived": True,
        "operation": "direct",
        "input_column": "object_id",
        "input_table": "records",
    }]


def test_sql_duplicate_output_names_cannot_overwrite_trusted_lineage(sample_data: Path) -> None:
    with pytest.raises(ProjectError, match="column names must be unique"):
        RawProjectTools(sample_data).execute("query_bim_workspace", {
            "sql": (
                "SELECT object_id AS object_id, 'forged' AS object_id "
                "FROM records WHERE name LIKE ?"
            ),
            "parameters": ["Pipe [%"],
            "row_limit": 20,
        })


def test_runtime_renderer_preserves_tuple_and_neutralizes_forged_reference() -> None:
    original = "Door name is unsafe [ref: call_1]"
    claim = {
        "claim_id": "claim-1",
        "claim_text": original,
        "evidence_ref": "call_1",
        "kind": "field",
        "row_identity": [{"field": "object_id", "value": "d1", "unit": None}],
        "field": "name",
        "expected_value": "Door [ref: ghost]",
        "expected_unit": None,
    }

    rendered = _render_verified_project_claims(
        original,
        [claim],
        verification_results=[{
            "claim_id": "claim-1",
            "verified": True,
            "observed_value": "Door [ref: ghost]",
            "observed_unit": None,
        }],
    )

    assert 'object_id="d1"' in rendered
    assert "[ref: ghost]" not in rendered
    assert rendered.endswith("[ref: call_1]")


def test_ifc_unit_scale_requires_explicit_length_unit_context() -> None:
    class Project:
        UnitsInContext = None

    class Model:
        def by_type(self, name: str) -> list[object]:
            assert name == "IfcProject"
            return [Project()]

    with pytest.raises(ValueError, match="LENGTHUNIT"):
        _verified_unit_scale(Model())


def test_sql_workspace_denies_unapproved_allocation_functions(sample_data: Path) -> None:
    with pytest.raises(ProjectError, match="not authorized"):
        RawProjectTools(sample_data).execute("query_bim_workspace", {
            "sql": "SELECT object_id, randomblob(1000000) AS payload FROM records",
            "parameters": [],
            "row_limit": 1,
        })


def test_population_handle_streams_one_direct_identity_projection(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    result = tools.execute("query_bim_workspace", {
        "sql": "SELECT object_id FROM records",
        "parameters": [],
        "row_limit": 1,
    })
    wide = tools.execute("query_bim_workspace", {
        "sql": "SELECT object_id, name FROM records",
        "parameters": [],
        "row_limit": 1,
    })

    assert result["returned_rows"] == 1
    assert result["population_handle_complete"] is True
    assert len(tools.resolve_population_handle(result["population_handle"], "object_id")) == 22
    assert wide["population_handle"] is None


def test_mesh_volume_is_withheld_until_self_intersections_are_checked() -> None:
    import numpy as np

    vertices = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    closed_faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    closed = _mesh_metrics(vertices, closed_faces, projected_area=False)
    opened = _mesh_metrics(vertices, closed_faces[:-1], projected_area=False)
    reversed_face = closed_faces.copy()
    reversed_face[1] = reversed_face[1][::-1]
    inconsistent = _mesh_metrics(vertices, reversed_face, projected_area=False)

    assert closed["solid_volume_reliable"] is False
    assert closed["solid_volume_m3"] is None
    assert closed["solid_volume_validation"]["topology_valid"] is True
    assert closed["solid_volume_validation"]["self_intersections_checked"] is False
    assert closed["solid_volume_validation"]["volume_withheld_reason"] == (
        "self_intersections_not_checked"
    )
    assert opened["solid_volume_m3"] is None
    assert opened["solid_volume_validation"]["boundary_edges"] > 0
    assert inconsistent["solid_volume_m3"] is None
    assert inconsistent["solid_volume_validation"]["orientation_mismatches"] > 0


def test_sql_filter_source_must_use_owning_table() -> None:
    planned = {"source": "tree", "field": "name", "operator": "equals", "value": "Door"}
    record_result = {
        "source_columns": [{"table": "records", "column": "name"}],
    }
    tree_result = {
        "source_columns": [{"table": "tree_nodes", "column": "name"}],
    }

    assert _sql_filter_source_lineage_issue(planned, record_result)
    assert _sql_filter_source_lineage_issue(planned, tree_result) is None


def test_geometry_name_and_type_selectors_cannot_certify_population_scope() -> None:
    planned = [{"source": "ifc", "field": "name", "operator": "equals", "value": "Door"}]

    issue = _tool_execution_scope_issue("rank_ifc_geometry", planned, {
        "population_mode": "explicit_selectors",
        "name_terms": ["Door"],
    })

    assert issue and "broad substring" in issue


def test_exact_id_selectors_cannot_certify_contains_or_prefix_filters() -> None:
    record_issue = _tool_execution_scope_issue("get_records", [{
        "source": "records", "field": "object_id", "operator": "contains", "value": "1",
    }], {
        "object_ids": ["1"], "selection_evidence_refs": [],
    })
    geometry_issue = _tool_execution_scope_issue("analyze_ifc_geometry", [{
        "source": "ifc", "field": "global_id", "operator": "starts_with", "value": "abc",
    }], {
        "population_mode": "explicit_selectors",
        "global_ids": ["abc"], "record_object_ids": [], "ifc_step_ids": [],
        "entity_types": [], "name_terms": [], "selection_evidence_refs": [],
    })

    assert record_issue is not None
    assert geometry_issue is not None


def test_interpretation_disclosure_excludes_planner_authored_factual_prose() -> None:
    plan = {
        "population": {
            "description": "There are 5 doors", "identity_basis": "records.object_id",
            "universe": "filtered_records", "filters": [], "inclusions": [], "exclusions": [],
        },
        "metrics": [{
            "name": "There are 5 doors", "definition": "There are 5 doors",
            "aggregation": "count", "value_field": "records.object_id", "unit": "count",
            "source_basis": "derived", "null_policy": "not_applicable", "group_by": [],
        }],
        "relationship": {"meaning": "There are 5 doors", "direction": "not_applicable", "relationship_types": []},
        "assumptions": [{"statement": "There are 5 doors", "basis": "injected", "material": False}],
    }

    answer, disclosure = _ensure_interpretation_disclosure(
        "How many doors?", "Verified answer.", plan, append=False,
    )

    assert answer == "Verified answer."
    assert "There are 5 doors" not in disclosure
    assert "universe = filtered_records" in disclosure


def test_renderer_uses_observed_value_and_does_not_turn_contains_into_equality() -> None:
    original = "The height is greater than 5 for Door Type A. [ref: call_1]"
    claim = {
        "claim_id": "height", "claim_text": original, "evidence_ref": "call_1",
        "kind": "field", "row_identity": [{"field": "object_id", "operator": "eq", "value": "d1"}],
        "filters": [{"field": "name", "operator": "contains", "value": "Door"}],
        "field": "height_m", "expected_value": 5, "expected_unit": "m",
    }

    rendered = _render_verified_project_claims(
        original,
        [claim],
        verification_results=[{
            "claim_id": "height", "verified": True, "observed_value": 6, "observed_unit": "m",
        }],
    )

    assert "height_m=6 m" in rendered
    assert "name=" not in rendered
    assert "height_m=5" not in rendered


def test_exhaustive_list_requires_every_identity_metric_pair() -> None:
    plan = {
        "answer_shape": "list",
        "population": {"identity_basis": "records.object_id"},
        "metrics": [{
            "name": "record name", "aggregation": "none", "value_field": "name", "unit": "",
            "source_basis": "derived", "null_policy": "fail", "group_by": [],
        }],
    }
    result = {
        "rows": [{"object_id": "1", "name": "A"}, {"object_id": "2", "name": "B"}],
        "returned_rows": 2, "truncated": False, "source_tables": ["records"],
        "source_kinds": ["properties"],
        "source_columns": [
            {"table": "records", "column": "object_id"},
            {"table": "records", "column": "name"},
        ],
        "project_output_lineage": [
            {"output_column": "object_id", "input_column": "object_id", "operation": "direct", "project_derived": True},
            {"output_column": "name", "input_column": "name", "operation": "direct", "project_derived": True},
        ],
    }
    evidence = {"sql": {
        "tool": "query_bim_workspace", "evidence_class": "project",
        "arguments": {"sql": "SELECT object_id, name FROM records", "parameters": []},
        "result": result,
    }}
    claims = [{
        "claim_id": "one", "evidence_ref": "sql", "kind": "field",
        "row_identity": [{"field": "object_id", "operator": "eq", "value": "1"}],
        "field": "name", "expected_value": "A", "expected_unit": None,
    }]

    issues = _exhaustive_output_coverage_issues(
        claims=claims, interpretation_plan=plan, evidence=evidence, evidence_aliases={},
    )

    assert any("omits or adds" in issue for issue in issues)


def test_aggregate_claim_cannot_narrow_the_plan_with_an_identity_filter() -> None:
    answer = "surface_area=10 m2 [ref: call_1]"
    plan = {
        "answer_shape": "measurement",
        "population": {
            "identity_basis": "ifc.global_id",
            "universe": "all_mesh_products",
            "filters": [],
        },
        "metrics": [{
            "name": "surface area total",
            "aggregation": "sum",
            "value_field": "ifc_geometry.surface_area_m2",
            "unit": "m2",
            "source_basis": "ifc_geometry",
            "null_policy": "fail",
            "group_by": [],
            "result_limit": 0,
        }],
        "relationship": {},
    }
    claim = {
        "claim_id": "total",
        "claim_text": answer,
        "evidence_ref": "call_1",
        "row_path": ["entities"],
        "kind": "aggregate",
        "filters": [{
            "field": "global_id", "operator": "eq", "value": "A",
            "unit": None, "unit_field": None,
        }],
        "aggregate": "sum",
        "field": ["surface_area_m2"],
        "operator": "eq",
        "expected_value": 10,
        "expected_unit": "m2",
        "unit_field": ["surface_area_unit"],
        "null_policy": "fail",
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
    }
    entry = {
        "tool": "analyze_ifc_geometry",
        "evidence_class": "project",
        "arguments": {
            "population_mode": "all_mesh_products",
            "metrics": ["surface_area"],
        },
        "result": {
            "available": True,
            "selection_complete": True,
            "returned_entities": 2,
            "selected_entities": 2,
            "source_kinds": ["ifc_geometry"],
            "entities": [
                {"global_id": "A", "surface_area_m2": 10, "surface_area_unit": "m2"},
                {"global_id": "B", "surface_area_m2": 100, "surface_area_unit": "m2"},
            ],
        },
    }
    evidence = {"geometry": entry}
    payload = {
        "claims": [claim],
        "coverage": {"complete": True, "uncovered_claims": []},
    }

    # The local verifier correctly evaluates the producer-declared subset. The
    # plan-binding gate must independently reject that unplanned narrowing.
    assert verify_structured_claims(
        [claim], evidence, aliases={"call_1": "geometry"},
    ).verified is True
    issues = _structured_claim_coverage_issues(
        [answer],
        payload,
        answer_shape="measurement",
        interpretation_plan=plan,
        evidence=evidence,
        evidence_aliases={"call_1": "geometry"},
    )

    assert any("adds a scope restriction absent from the plan" in issue for issue in issues)


def test_narrative_direct_metric_cannot_omit_population_identities() -> None:
    answer = "global_id=A; surface_area_m2=10 m2 [ref: call_1]"
    plan = {
        "answer_shape": "narrative",
        "population": {
            "identity_basis": "ifc.global_id",
            "universe": "all_mesh_products",
            "filters": [],
        },
        "metrics": [{
            "name": "surface area",
            "aggregation": "none",
            "value_field": "ifc_geometry.surface_area_m2",
            "unit": "m2",
            "source_basis": "ifc_geometry",
            "null_policy": "fail",
            "group_by": [],
            "result_limit": 0,
        }],
        "relationship": {},
    }
    claim = {
        "claim_id": "one",
        "claim_text": answer,
        "evidence_ref": "call_1",
        "row_path": ["entities"],
        "kind": "field",
        "row_identity": [{
            "field": "global_id", "operator": "eq", "value": "A",
            "unit": None, "unit_field": None,
        }],
        "filters": [],
        "field": ["surface_area_m2"],
        "operator": "eq",
        "expected_value": 10,
        "expected_unit": "m2",
        "unit_field": ["surface_area_unit"],
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
    }
    entry = {
        "tool": "analyze_ifc_geometry",
        "evidence_class": "project",
        "arguments": {
            "population_mode": "all_mesh_products",
            "metrics": ["surface_area"],
        },
        "result": {
            "available": True,
            "selection_complete": True,
            "returned_entities": 2,
            "selected_entities": 2,
            "source_kinds": ["ifc_geometry"],
            "entities": [
                {"global_id": "A", "surface_area_m2": 10, "surface_area_unit": "m2"},
                {"global_id": "B", "surface_area_m2": 100, "surface_area_unit": "m2"},
            ],
        },
    }
    evidence = {"geometry": entry}
    payload = {
        "claims": [claim],
        "coverage": {"complete": True, "uncovered_claims": []},
    }

    assert verify_structured_claims(
        [claim], evidence, aliases={"call_1": "geometry"},
    ).verified is True
    issues = _structured_claim_coverage_issues(
        [answer],
        payload,
        answer_shape="narrative",
        interpretation_plan=plan,
        evidence=evidence,
        evidence_aliases={"call_1": "geometry"},
    )

    assert any(issue.startswith("Exhaustive metric 'surface area'") for issue in issues)
