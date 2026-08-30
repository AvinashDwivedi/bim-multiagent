from __future__ import annotations

import copy

import pytest

from bim_agent.question_planning import normalize_planning_payload


def _schema_context(*, hierarchy_values: list[str] | None = None) -> dict:
    values = hierarchy_values or ["Lighting Devices", "Cable Trays", "Doors"]
    return {
        "schema_version": "1.0",
        "read_only": True,
        "tables": [
            {
                "name": "records",
                "definition": (
                    "CREATE TABLE records(object_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                    "path_text TEXT NOT NULL)"
                ),
            },
            {
                "name": "properties",
                "definition": (
                    "CREATE TABLE properties(object_id TEXT NOT NULL, property_key TEXT NOT NULL, "
                    "property_value TEXT NOT NULL, number_value REAL, unit TEXT)"
                ),
            },
        ],
        "property_keys": [
            {"key": "Identity Data.Type Name", "record_count": 10},
            {"key": "Dimensions.Length", "record_count": 10},
            {"key": "Dimensions.Width", "record_count": 10},
            {"key": "Materials and Finishes.Material", "record_count": 10},
        ],
        "hierarchy_shape_samples": [
            {
                "path": ["Project", value],
                "path_depth": 2,
                "path_truncated": False,
                "direct_children": 0,
                "descendants": 0,
                "leaf_descendants": 0,
                "direct_child_names": [],
            }
            for value in values
        ],
        "hierarchy_samples_truncated": False,
    }


def _count_metric() -> dict:
    return {
        "name": "instance_count",
        "definition": "COUNT(DISTINCT records.object_id)",
        "aggregation": "distinct_count",
        "value_field": "records.object_id",
        "unit": "count",
        "source_basis": "derived",
        "null_policy": "not_applicable",
        "group_by": [],
        "result_limit": 0,
    }


def _payload(
    *,
    answer_shape: str = "count",
    question_term: str = "מפסקים",
    schema_value: str = "Lighting Devices",
    decision: str = "inspect_then_execute",
    metrics: list[dict] | None = None,
    ambiguities: list[dict] | None = None,
) -> dict:
    return {
        "route": {
            "answer_shape": answer_shape,
            "required_capabilities": ["schema_inspection", "hierarchy", "record_query"],
            "required_sources": ["tree", "properties"],
            "preferred_compute": "sql",
            "confidence": 0.92,
            "uncertainties": [],
        },
        "interpretation_plan": {
            "objective": "Answer over the schema-observed project category.",
            "population": {
                "description": "Installed records in the schema-observed category.",
                "identity_basis": "records.object_id",
                "universe": "filtered_records",
                "filters": [
                    {
                        "source": "records",
                        "field": "path_text",
                        "operator": "contains",
                        "value": schema_value,
                        "binding": "observed_schema_mapping",
                        "question_term": question_term,
                    }
                ],
                "inclusions": [],
                "exclusions": [],
            },
            "metrics": copy.deepcopy(metrics) if metrics is not None else [_count_metric()],
            "relationship": {
                "meaning": "",
                "direction": "not_applicable",
                "relationship_types": [],
            },
            "inclusion_exclusion_rationale": (
                "The category mapping is provisional until runtime scope inspection confirms it."
            ),
            "assumptions": [],
            "ambiguities": copy.deepcopy(ambiguities or []),
            "execution_decision": decision,
            "clarification_question": "",
        },
    }


def test_observed_schema_mapping_allows_exact_hebrew_term_to_enter_inspection() -> None:
    _, plan, _ = normalize_planning_payload(
        _payload(),
        question="כמה מפסקים בפרויקט?",
        schema_context=_schema_context(),
    )

    assert plan["execution_decision"] == "inspect_then_execute"
    assert plan["population"]["filters"] == [
        {
            "source": "records",
            "field": "path_text",
            "operator": "contains",
            "value": "Lighting Devices",
            "binding": "observed_schema_mapping",
            "question_term": "מפסקים",
        }
    ]


@pytest.mark.parametrize(
    ("schema_context", "schema_value"),
    [
        (None, "Lighting Devices"),
        (_schema_context(), "Mechanical Equipment"),
    ],
    ids=["schema-context-absent", "schema-value-not-observed"],
)
def test_observed_schema_mapping_without_exact_schema_evidence_clarifies(
    schema_context: dict | None,
    schema_value: str,
) -> None:
    _, plan, _ = normalize_planning_payload(
        _payload(schema_value=schema_value),
        question="כמה מפסקים בפרויקט?",
        schema_context=schema_context,
    )

    assert plan["execution_decision"] == "clarify"
    assert any(
        item["term"] == "population question binding"
        for item in plan["ambiguities"]
    )


def test_observed_schema_mapping_rejects_a_forged_question_term() -> None:
    _, plan, _ = normalize_planning_payload(
        _payload(question_term="מגשים"),
        question="כמה מפסקים בפרויקט?",
        schema_context=_schema_context(),
    )

    assert plan["execution_decision"] == "clarify"
    assert any(
        item["term"] == "population question binding"
        for item in plan["ambiguities"]
    )


def test_observed_schema_mapping_cannot_skip_runtime_inspection() -> None:
    _, plan, _ = normalize_planning_payload(
        _payload(decision="execute"),
        question="כמה מפסקים בפרויקט?",
        schema_context=_schema_context(),
    )

    assert plan["execution_decision"] == "clarify"
    assert any(
        item["term"] == "population question binding"
        for item in plan["ambiguities"]
    )


def test_schema_mapping_does_not_silence_material_biggest_ambiguity() -> None:
    ranking_metric = {
        "name": "ranked_volume",
        "definition": "Rank selected records by their authored volume.",
        "aggregation": "rank_descending",
        "value_field": "Dimensions.Volume",
        "unit": "m3",
        "source_basis": "property",
        "null_policy": "exclude",
        "group_by": [],
        "result_limit": 1,
    }
    ambiguity = {
        "term": "biggest",
        "alternatives": ["volume", "area", "length"],
        "material": True,
        "resolution_basis": "The requested adjective does not select a measurement dimension.",
    }

    _, plan, _ = normalize_planning_payload(
        _payload(
            answer_shape="ranking",
            question_term="door",
            schema_value="Doors",
            metrics=[ranking_metric],
            ambiguities=[ambiguity],
        ),
        question="Which door is biggest?",
        schema_context=_schema_context(),
    )

    assert plan["execution_decision"] == "clarify"
    assert any(item["term"] == "biggest" and item["material"] for item in plan["ambiguities"])


def test_grouped_total_uses_group_keys_without_redundant_none_metrics() -> None:
    grouped_length = {
        "name": "total_length",
        "definition": "SUM of authored cable-tray length for each type, width, and material group.",
        "aggregation": "sum",
        "value_field": "Dimensions.Length",
        "unit": "m",
        "source_basis": "property",
        "null_policy": "exclude",
        "group_by": [
            "Identity Data.Type Name",
            "Dimensions.Width",
            "Materials and Finishes.Material",
        ],
        "result_limit": 0,
    }

    _, plan, _ = normalize_planning_payload(
        _payload(
            answer_shape="grouped_total",
            question_term="מגשי הכבלים",
            schema_value="Cable Trays",
            metrics=[grouped_length],
        ),
        question="מהו האורך הכולל של מגשי הכבלים לפי סוג, רוחב וחומר?",
        schema_context=_schema_context(),
    )

    assert plan["execution_decision"] == "inspect_then_execute"
    assert len(plan["metrics"]) == 1
    assert plan["metrics"][0]["aggregation"] == "sum"
    assert plan["metrics"][0]["group_by"] == grouped_length["group_by"]
