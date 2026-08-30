from __future__ import annotations

import json

import pytest

from bim_agent.claim_verification import (
    CLAIM_CONTRACT_VERSION,
    StructuredClaimProducer,
    StructuredClaimVerifier,
    serialize_cited_evidence,
    structured_claims_json_schema,
    verify_structured_claims,
)
from conftest import final_response


def _field_claim(**overrides):
    claim = {
        "claim_id": "door-count",
        "evidence_ref": "call_1",
        "row_path": ["rows"],
        "kind": "field",
        "row_identity": [
            {"field": "category", "operator": "eq", "value": "doors"},
        ],
        "field": "count",
        "operator": "eq",
        "expected_value": 3,
    }
    claim.update(overrides)
    return claim


def _aggregate_claim(**overrides):
    claim = {
        "claim_id": "door-count",
        "evidence_ref": "call_1",
        "row_path": ["rows"],
        "kind": "aggregate",
        "filters": [
            {"field": "category", "operator": "eq", "value": "door"},
        ],
        "aggregate": "count",
        "operator": "eq",
        "expected_value": 2,
        "null_policy": "fail",
    }
    claim.update(overrides)
    return claim


def _ranking_claim(**overrides):
    claim = {
        "claim_id": "largest-door",
        "evidence_ref": "call_1",
        "row_path": ["rows"],
        "kind": "ranking",
        "filters": [
            {"field": "category", "operator": "eq", "value": "door"},
        ],
        "row_identity": [
            {"field": "object_id", "operator": "eq", "value": "d1"},
        ],
        "field": "volume_m3",
        "ranking": "maximum",
        "operator": "eq",
        "expected_value": 5,
        "expected_unit": "m3",
        "null_policy": "fail",
    }
    claim.update(overrides)
    return claim


def test_field_claim_binds_number_to_row_identity_and_field() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"category": "walls", "count": 5},
                {"category": "doors", "count": 3},
            ],
            "returned_rows": 2,
            "truncated": False,
        }
    }

    correct = StructuredClaimVerifier(evidence).verify(_field_claim())
    crossed = StructuredClaimVerifier(evidence).verify(_field_claim(expected_value=5))

    assert correct.verified is True
    assert correct.observed_value == 3
    assert crossed.verified is False
    assert crossed.code == "predicate_failed"
    assert crossed.observed_value == 3


@pytest.mark.parametrize(("raw_value", "expected_value"), [
    ("1,234 m", 1234),
    ("1.234 m", 1.234),
])
def test_field_claim_rejects_locale_ambiguous_raw_measurements(
    raw_value: str,
    expected_value: float,
) -> None:
    evidence = {
        "call_1": {
            "rows": [{"category": "doors", "height": raw_value}],
            "returned_rows": 1,
            "truncated": False,
        },
    }
    claim = _field_claim(
        field="height",
        expected_value=expected_value,
        expected_unit="m",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code in {
        "predicate_failed", "unit_not_evidenced", "unit_unproven", "unit_mismatch",
    }


def test_field_claim_accepts_unambiguous_grouped_decimal_measurement() -> None:
    evidence = {
        "call_1": {
            "rows": [{"category": "doors", "height": "1,234.56 m"}],
            "returned_rows": 1,
            "truncated": False,
        },
    }
    claim = _field_claim(
        field="height",
        expected_value=1234.56,
        expected_unit="m",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is True


def test_ranking_claim_proves_selected_row_is_extreme_over_complete_population() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"object_id": "d1", "category": "door", "volume_m3": 5},
                {"object_id": "d2", "category": "door", "volume_m3": 9},
                {"object_id": "w1", "category": "wall", "volume_m3": 100},
            ],
            "returned_rows": 3,
            "truncated": False,
            "complete": True,
        }
    }

    wrong = StructuredClaimVerifier(evidence).verify(_ranking_claim())
    correct = StructuredClaimVerifier(evidence).verify(_ranking_claim(
        row_identity=[{"field": "object_id", "operator": "eq", "value": "d2"}],
        expected_value=9,
    ))

    assert wrong.verified is False
    assert wrong.code == "ranking_failed"
    assert correct.verified is True
    assert correct.matched_rows == 2


def test_ranking_claim_rejects_incomplete_population() -> None:
    evidence = {
        "call_1": {
            "rows": [{"object_id": "d1", "category": "door", "volume_m3": 5}],
            "returned_rows": 1,
            "total_count": 2,
            "returned_count": 1,
            "truncated": True,
        }
    }

    result = StructuredClaimVerifier(evidence).verify(_ranking_claim())

    assert result.verified is False
    assert result.code == "incomplete_evidence"


def test_ranking_claim_rejects_a_tied_singular_winner() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"object_id": "d1", "category": "door", "volume_m3": 5},
                {"object_id": "d2", "category": "door", "volume_m3": 5},
            ],
            "returned_rows": 2,
            "truncated": False,
            "complete": True,
        }
    }

    result = StructuredClaimVerifier(evidence).verify(_ranking_claim())

    assert result.verified is False
    assert result.code == "ranking_tie"


def test_trusted_geometry_ranking_can_verify_one_returned_winner_over_full_population() -> None:
    evidence = {
        "call_1": {
            "tool": "rank_ifc_geometry",
            "evidence_class": "project",
            "arguments": {"metric": "surface_area", "order": "descending"},
            "result": {
                "rows": [{
                    "object_id": "d2", "surface_area_m2": 9,
                    "ranking_metric": "surface_area", "ranking_value": 9,
                }],
                "metric": "surface_area", "order": "descending",
                "complete_ranking": True, "selection_complete": True,
                "available": True, "units_verified": True,
                "unrankable_entities": 0, "selected_entities": 3,
                "matching_entities": 3, "rankable_entities": 3,
                "returned_entities": 1, "extreme_tie_count": 1,
                "truncated": True,
            },
        },
    }
    claim = _ranking_claim(
        filters=[],
        row_identity=[{"field": "object_id", "operator": "eq", "value": "d2"}],
        field="surface_area_m2",
        expected_value=9,
        expected_unit="m2",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is True
    assert result.matched_rows == 3


def test_nonfinite_numeric_equality_never_verifies() -> None:
    evidence = {
        "call_1": {
            "rows": [{"category": "doors", "count": float("inf")}],
            "returned_rows": 1,
            "truncated": False,
        },
    }

    result = StructuredClaimVerifier(evidence).verify(_field_claim(
        expected_value=float("inf"),
    ))

    assert result.verified is False
    assert result.code == "comparison_error"


def test_schema_valid_bad_predicate_fails_closed_without_crashing() -> None:
    evidence = {"call_1": {"rows": [{"category": "door"}], "truncated": False}}
    claim = _aggregate_claim(filters=[{
        "field": "category", "operator": "contains", "value": "door",
    }])

    result = StructuredClaimVerifier({
        "call_1": {"rows": [{"category": 12}], "truncated": False, "complete": True}
    }).verify(claim)

    assert result.verified is False
    assert result.code == "comparison_error"


def test_field_claim_requires_a_unique_row_identity() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"category": "doors", "floor": "GF", "count": 2},
                {"category": "doors", "floor": "L1", "count": 1},
            ]
        }
    }

    result = StructuredClaimVerifier(evidence).verify(_field_claim())

    assert result.verified is False
    assert result.code == "identity_not_unique"
    assert result.matched_rows == 2


def test_field_claim_identity_cannot_be_the_claimed_field_itself() -> None:
    evidence = {"call_1": {"rows": [{"count": 3}]}}
    claim = _field_claim(
        row_identity=[{"field": "count", "operator": "eq", "value": 3}],
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code == "invalid_claim"
    assert "independent of the claimed field" in result.message


def test_field_claim_cannot_smuggle_claimed_value_into_a_mixed_identity() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"population": "doors", "count": 3},
                {"population": "doors", "count": 5},
            ]
        }
    }
    claim = _field_claim(row_identity=[
        {"field": "population", "operator": "eq", "value": "doors"},
        {"field": "count", "operator": "eq", "value": 5},
    ], expected_value=5)

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code == "invalid_claim"


def test_equivalent_dotted_and_segmented_paths_cannot_self_identify() -> None:
    evidence = {"call_1": {"rows": [{"metrics": {"count": 3}}]}}
    claim = _field_claim(
        row_identity=[{
            "field": ["metrics", "count"], "operator": "eq", "value": 3,
        }],
        field="metrics.count",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code == "invalid_claim"


def test_claim_producer_cannot_choose_a_tolerance_that_makes_a_wrong_value_pass() -> None:
    evidence = {"call_1": {"rows": [{"category": "doors", "count": 3}]}}
    claim = _field_claim(expected_value=5, absolute_tolerance=2)

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code == "invalid_claim"
    assert "tolerances are not trusted" in result.message


def test_field_claim_converts_explicit_units_before_comparison() -> None:
    evidence = {
        "call_1": {
            "rows": [{"object_id": "socket-7", "height": "145 cm"}],
            "returned_rows": 1,
            "truncated": False,
        }
    }
    claim = _field_claim(
        claim_id="socket-height",
        row_identity=[{"field": "object_id", "operator": "eq", "value": "socket-7"}],
        field="height",
        expected_value=1.45,
        expected_unit="m",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is True
    assert result.observed_value == 1.45
    assert result.observed_unit == "m"


def test_unit_must_be_evidenced_and_dimensionally_compatible() -> None:
    evidence = {"call_1": {"rows": [{"object_id": "x", "value": 5}]}}
    base = _field_claim(
        row_identity=[{"field": "object_id", "operator": "eq", "value": "x"}],
        field="value",
        expected_value=5,
        expected_unit="m",
    )

    missing = StructuredClaimVerifier(evidence).verify(base)
    incompatible = StructuredClaimVerifier({
        "call_1": {"rows": [{"object_id": "x", "value": "5 m2"}]}
    }).verify(base)

    assert missing.code == "unit_not_evidenced"
    assert incompatible.code == "unit_mismatch"


def test_exact_flattened_property_key_is_not_split_on_dots() -> None:
    evidence = {
        "call_1": {
            "records": [{
                "object_id": "43",
                "properties": {"IFC Parameters.IfcGUID": "guid-43"},
                "IFC Parameters.IfcGUID": "flattened-guid-43",
            }]
        }
    }
    claim = _field_claim(
        row_path=["records"],
        row_identity=[{"field": "object_id", "operator": "eq", "value": "43"}],
        field="IFC Parameters.IfcGUID",
        expected_value="flattened-guid-43",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is True


def test_aggregate_is_recomputed_over_the_declared_population() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"object_id": "d1", "category": "door"},
                {"object_id": "d2", "category": "door"},
                {"object_id": "w1", "category": "wall"},
            ],
            "returned_rows": 3,
            "truncated": False,
            "complete": True,
        }
    }

    result = StructuredClaimVerifier(evidence).verify(_aggregate_claim())
    wrong_population = StructuredClaimVerifier(evidence).verify(
        _aggregate_claim(filters=[{"field": "category", "operator": "eq", "value": "wall"}])
    )

    assert result.verified is True
    assert result.matched_rows == 2
    assert result.observed_value == 2
    assert wrong_population.code == "aggregate_failed"
    assert wrong_population.observed_value == 1


def test_aggregate_requires_affirmative_complete_population_metadata() -> None:
    evidence = {"call_1": {"rows": [{"category": "door"}, {"category": "door"}]}}

    result = StructuredClaimVerifier(evidence).verify(_aggregate_claim())

    assert result.verified is False
    assert result.code == "incomplete_evidence"
    assert "does not affirm" in result.message


def test_field_identity_is_not_unique_over_truncated_sql_evidence() -> None:
    evidence = {
        "call_1": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {
                "sql": "SELECT category, COUNT(*) AS count FROM records GROUP BY category LIMIT 1",
            },
            "result": {
                "rows": [{"category": "doors", "count": 3}],
                "returned_rows": 1,
                "truncated": True,
                "source_tables": ["records"],
                "project_derived_columns": ["category", "count"],
            },
        },
    }

    result = StructuredClaimVerifier(evidence).verify(_field_claim())

    assert result.verified is False
    assert result.code == "incomplete_evidence"


def test_row_identity_cannot_resolve_from_root_metadata() -> None:
    claim = _field_claim(
        row_identity=[{"field": ["$", "complete"], "operator": "eq", "value": True}],
        field="value",
        expected_value=3,
    )

    result = StructuredClaimVerifier({
        "call_1": {"rows": [{"value": 3}], "complete": True},
    }).verify(claim)

    assert result.verified is False
    assert result.code == "invalid_claim"


def test_identity_equality_does_not_coerce_numeric_strings() -> None:
    result = StructuredClaimVerifier({
        "call_1": {"rows": [{"object_id": "001", "value": 3}]},
    }).verify(_field_claim(
        row_identity=[{"field": "object_id", "operator": "eq", "value": 1}],
        field="value",
        expected_value=3,
    ))

    assert result.verified is False
    assert result.code == "identity_not_found"


def test_declared_unit_cannot_override_tool_schema_unit() -> None:
    result = StructuredClaimVerifier({
        "call_1": {"rows": [{"object_id": "A", "length_m": 5, "unit": "ft"}]},
    }).verify(_field_claim(
        row_identity=[{"field": "object_id", "operator": "eq", "value": "A"}],
        field="length_m",
        expected_value=5,
        expected_unit="ft",
        unit_field="unit",
    ))

    assert result.verified is False
    assert result.code == "unit_mismatch"


def test_distinct_count_rejects_unit_bearing_measurements() -> None:
    evidence = {
        "call_1": {
            "rows": [{"length": "1000 mm"}, {"length": "1 m"}],
            "returned_rows": 2,
            "complete": True,
        },
    }
    claim = _aggregate_claim(
        filters=[],
        aggregate="distinct_count",
        field="length",
        distinct_by=None,
        expected_value=2,
        expected_unit="count",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code == "unsupported_unit_distinct_count"


def test_geometry_rank_claim_is_bound_to_tool_metric_and_order() -> None:
    evidence = {
        "call_1": {
            "tool": "rank_ifc_geometry",
            "evidence_class": "project",
            "arguments": {"metric": "solid_volume", "order": "descending"},
            "result": {
                "rows": [
                    {
                        "object_id": "A", "length_m": 5, "ranking_metric": "solid_volume",
                        "ranking_value": 100,
                    },
                ],
                "metric": "solid_volume", "order": "descending", "complete_ranking": True,
                "selection_complete": True, "available": True, "units_verified": True,
                "unrankable_entities": 0, "selected_entities": 2, "matching_entities": 2,
                "rankable_entities": 2, "returned_entities": 1, "extreme_tie_count": 1,
                "truncated": True,
            },
        },
    }
    claim = _ranking_claim(
        filters=[],
        row_identity=[{"field": "object_id", "operator": "eq", "value": "A"}],
        field="length_m",
        expected_value=5,
        expected_unit="m",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is False
    assert result.code == "ranking_provenance_mismatch"


def test_distinct_count_uses_explicit_identity_field() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"category": "door", "object_id": "d1"},
                {"category": "door", "object_id": "d1"},
                {"category": "door", "object_id": "d2"},
            ],
            "returned_rows": 3,
            "truncated": False,
            "complete": True,
        }
    }
    claim = _aggregate_claim(
        aggregate="distinct_count",
        field="object_id",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is True
    assert result.matched_rows == 3
    assert result.observed_value == 2


def test_numeric_aggregate_normalizes_each_evidence_unit() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"category": "tray", "length": "100 cm"},
                {"category": "tray", "length": "2 m"},
            ],
            "returned_rows": 2,
            "truncated": False,
            "complete": True,
        }
    }
    claim = _aggregate_claim(
        claim_id="tray-length",
        filters=[{"field": "category", "operator": "eq", "value": "tray"}],
        aggregate="sum",
        field="length",
        expected_value=3,
        expected_unit="m",
    )

    result = StructuredClaimVerifier(evidence).verify(claim)

    assert result.verified is True
    assert result.observed_value == 3
    assert result.observed_unit == "m"


def test_aggregate_rejects_paginated_or_compacted_evidence() -> None:
    paginated = {
        "call_1": {
            "rows": [{"category": "door"}, {"category": "door"}],
            "total_count": 10,
            "returned_count": 2,
            "cursor": "next-page",
        }
    }
    compacted = {
        "call_1": {
            "rows": [{"category": "door"}, {"category": "door"}],
            "returned_rows": 2,
            "model_rows_omitted": 8,
        }
    }

    first = StructuredClaimVerifier(paginated).verify(_aggregate_claim())
    second = StructuredClaimVerifier(compacted).verify(_aggregate_claim())

    assert first.code == "incomplete_evidence"
    assert second.code == "incomplete_evidence"


def test_complete_cumulative_fetch_more_chain_can_verify_an_aggregate() -> None:
    evidence = {
        "call_1": {
            "tool": "fetch_more",
            "evidence_class": "project",
            "arguments": {"cursor": "opaque-page-2"},
            "result": {
                "results": [{"category": "door"}, {"category": "door"}],
                "total_count": 2,
                "returned_count": 2,
                "offset": 0,
                "cursor": None,
                "pagination_complete": True,
                "pagination_pages": 2,
                "page_tool": "search_records",
                "pagination_scope_tool": "search_records",
                "pagination_scope_arguments": {"terms": ["door"], "limit": 1},
            },
        },
    }

    result = StructuredClaimVerifier(evidence).verify(_aggregate_claim(row_path=["results"]))

    assert result.verified is True
    assert result.observed_value == 2


def test_fetch_more_cannot_claim_completeness_without_originating_scope_metadata() -> None:
    evidence = {
        "call_1": {
            "tool": "fetch_more",
            "evidence_class": "project",
            "arguments": {"cursor": "opaque-page-2"},
            "result": {
                "results": [{"category": "door"}, {"category": "door"}],
                "total_count": 2,
                "returned_count": 2,
                "offset": 0,
                "cursor": None,
                "pagination_complete": True,
                "pagination_pages": 2,
                "page_tool": "search_records",
            },
        },
    }

    result = StructuredClaimVerifier(evidence).verify(_aggregate_claim(row_path=["results"]))

    assert result.verified is False
    assert result.code == "incomplete_evidence"


def test_missing_filter_field_fails_instead_of_silently_excluding_rows() -> None:
    evidence = {
        "call_1": {
            "rows": [
                {"category": "door"},
                {"name": "unclassified"},
            ],
            "returned_rows": 2,
            "truncated": False,
            "complete": True,
        }
    }

    result = StructuredClaimVerifier(evidence).verify(_aggregate_claim(expected_value=1))

    assert result.verified is False
    assert result.code == "filter_field_not_found"


def test_ledger_alias_uses_unabridged_result_not_compact_output() -> None:
    evidence = {
        "opaque-call-id": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {
                "sql": "SELECT category, COUNT(*) AS count FROM records GROUP BY category",
                "parameters": [],
            },
            "output": '{"rows":[{"category":"wall","count":5}]}',
            "result": {
                "rows": [{"category": "doors", "count": 3}],
                "returned_rows": 1,
                "truncated": False,
                "source_tables": ["records"],
                "source_columns": [
                    {"table": "records", "column": "category"},
                ],
                "source_kinds": ["properties"],
                "project_derived_columns": ["category", "count"],
            },
        }
    }

    result = verify_structured_claims(
        [_field_claim()],
        evidence,
        aliases={"call_1": "opaque-call-id"},
    )

    assert result.verified is True
    assert result.results[0].evidence_ref == "opaque-call-id"
    assert result.results[0].observed_value == 3


def test_model_review_cannot_verify_a_project_claim() -> None:
    evidence = {
        "call_1": {
            "tool": "review_scope_and_evidence",
            "evidence_class": "model_review",
            "output": '{"rows":[{"category":"doors","count":3}]}',
            "result": {"rows": [{"category": "doors", "count": 3}]},
        }
    }

    result = StructuredClaimVerifier(evidence).verify(_field_claim())

    assert result.verified is False
    assert result.code == "wrong_evidence_class"


def test_batch_requires_at_least_one_verified_claim() -> None:
    report = verify_structured_claims([], {})

    assert report.contract_version == CLAIM_CONTRACT_VERSION
    assert report.verified is False
    assert report.results == ()


def test_json_schema_is_strict_and_versioned() -> None:
    schema = structured_claims_json_schema()

    assert schema["properties"]["contract_version"]["const"] == CLAIM_CONTRACT_VERSION
    assert schema["additionalProperties"] is False
    assert "coverage" in schema["required"]
    variants = schema["properties"]["claims"]["items"]["anyOf"]
    assert {item["properties"]["kind"]["const"] for item in variants} == {
        "field", "aggregate", "ranking",
    }
    assert all(item["additionalProperties"] is False for item in variants)
    assert all("claim_text" in item["required"] for item in variants)


def _produced_field_claim(answer: str) -> dict:
    return {
        "claim_id": "door-count",
        "claim_text": answer,
        "evidence_ref": "call_1",
        "row_path": ["rows"],
        "kind": "field",
        "row_identity": [{
            "field": "category",
            "operator": "eq",
            "value": "doors",
            "unit": None,
            "unit_field": None,
        }],
        "field": "count",
        "operator": "eq",
        "expected_value": 3,
        "expected_unit": None,
        "unit_field": None,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
    }


def test_cited_evidence_serializer_keeps_full_result_and_arguments_only() -> None:
    evidence = {
        "opaque-1": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {"sql": "SELECT category, COUNT(*) AS count FROM records GROUP BY category"},
            "output": '{"rows":[],"model_rows_omitted":100}',
            "result": {"rows": [{"category": "doors", "count": 3}], "returned_rows": 1},
        },
        "opaque-unused": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {"sql": "SELECT 99"},
            "result": {"rows": [{"value": 99}]},
        },
    }

    serialized = serialize_cited_evidence(
        "There are 3 doors. [ref: call_1]",
        evidence,
        aliases={"call_1": "opaque-1", "call_2": "opaque-unused"},
    )

    assert len(serialized) == 1
    assert serialized[0]["evidence_ref"] == "call_1"
    assert serialized[0]["resolved_ref"] == "opaque-1"
    assert serialized[0]["arguments"]["sql"].startswith("SELECT category")
    assert serialized[0]["result"]["rows"] == [{"category": "doors", "count": 3}]
    assert "output" not in serialized[0]


def test_cited_evidence_serializer_bounds_model_projection_but_not_local_ledger() -> None:
    rows = [{"object_id": str(index), "name": "x" * 100} for index in range(200)]
    evidence = {
        "opaque-1": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {"sql": "SELECT object_id, name FROM records"},
            "result": {"rows": rows, "returned_rows": 200, "truncated": False},
        }
    }

    serialized = serialize_cited_evidence(
        "Objects were listed. [ref: call_1]",
        evidence,
        aliases={"call_1": "opaque-1"},
        max_total_chars=8_000,
    )

    projected = serialized[0]["result"]
    assert len(projected.get("rows", [])) < len(rows)
    assert projected["_producer_projection"]["full_result_retained_for_local_verification"] is True
    assert len(evidence["opaque-1"]["result"]["rows"]) == 200


def test_cited_evidence_serializer_preserves_compact_result_over_fifty_rows() -> None:
    rows = [{"object_id": index} for index in range(60)]
    evidence = {
        "opaque-1": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {"sql": "SELECT object_id FROM records"},
            "result": {"rows": rows, "returned_rows": 60, "truncated": False},
        }
    }

    serialized = serialize_cited_evidence(
        "Objects were listed. [ref: call_1]",
        evidence,
        aliases={"call_1": "opaque-1"},
        max_total_chars=8_000,
    )

    assert serialized[0]["result"]["rows"] == rows
    assert "_producer_projection" not in serialized[0]["result"]


def test_cited_evidence_serializer_enforces_one_cumulative_character_limit() -> None:
    evidence = {
        f"opaque-{index}": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "result": {"blob": str(index) * 1_300},
        }
        for index in range(3)
    }
    answer = " ".join(
        f"Claim {index}. [ref: call_{index}]" for index in range(3)
    )
    aliases = {f"call_{index}": f"opaque-{index}" for index in range(3)}
    character_limit = 4_000

    serialized = serialize_cited_evidence(
        answer,
        evidence,
        aliases=aliases,
        max_total_chars=character_limit,
    )

    assert len(serialized) == 3
    assert len(json.dumps(serialized, ensure_ascii=False)) <= character_limit


def test_cited_evidence_serializer_bounds_fallback_key_summaries() -> None:
    pathological_key = "key" * 40_000
    evidence = {
        "opaque-1": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "result": {pathological_key: "value"},
        }
    }
    character_limit = 2_000

    serialized = serialize_cited_evidence(
        "A value was observed. [ref: call_1]",
        evidence,
        aliases={"call_1": "opaque-1"},
        max_total_chars=character_limit,
    )

    assert len(json.dumps(serialized, ensure_ascii=False)) <= character_limit
    result = serialized[0]["result"]
    assert result["_producer_projection"]["summary_only"] is True
    assert all(len(key) <= 256 for key in result["available_top_level_keys"])


def test_raw_mapping_cannot_bypass_evidence_class_restriction() -> None:
    result = StructuredClaimVerifier({
        "call_1": {
            "evidence_class": "model_review",
            "rows": [{"category": "doors", "count": 3}],
        }
    }).verify(_field_claim())

    assert result.verified is False
    assert result.code == "wrong_evidence_class"


def test_model_claim_producer_uses_strict_contract_and_exact_answer_coverage(
    fake_client_factory,
) -> None:
    answer = "There are 3 doors. [ref: call_1]"
    model_payload = {
        "contract_version": CLAIM_CONTRACT_VERSION,
        "claims": [_produced_field_claim(answer)],
        "coverage": {
            "project_claims_found": 1,
            "structured_claims_produced": 1,
            "complete": True,
            "uncovered_claims": [],
        },
    }
    client = fake_client_factory(final_response(1, json.dumps(model_payload)))
    evidence = {
        "opaque-1": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {
                "sql": "SELECT category, COUNT(*) AS count FROM records GROUP BY category",
                "parameters": [],
            },
            "output": "compacted output that must not be sent",
            "result": {
                "rows": [{"category": "doors", "count": 3}],
                "returned_rows": 1,
                "truncated": False,
                "source_tables": ["records"],
                "source_columns": [
                    {"table": "records", "column": "category"},
                ],
                "source_kinds": ["properties"],
                "project_derived_columns": ["category", "count"],
            },
        },
        "unused": {
            "tool": "query_bim_workspace",
            "evidence_class": "project",
            "arguments": {"sql": "SELECT 99"},
            "result": {"rows": [{"value": 99}]},
        },
    }
    producer = StructuredClaimProducer(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    production = producer.produce(
        question="How many doors?",
        final_answer=answer,
        interpretation_plan={"population": {"description": "Door rows"}},
        evidence=evidence,
        aliases={"call_1": "opaque-1"},
    )

    assert production.contract_valid is True
    assert production.coverage_complete is True
    assert production.cited_evidence_refs == ("call_1",)
    request = client.responses.requests[0]
    assert request["text"]["format"]["name"] == "bim_structured_claims"
    assert request["text"]["format"]["strict"] is True
    assert "complete factual sentence or table row" in request["instructions"]
    sent = json.loads(request["input"])
    assert [item["evidence_ref"] for item in sent["cited_evidence"]] == ["call_1"]
    assert sent["cited_evidence"][0]["arguments"]["parameters"] == []
    assert sent["cited_evidence"][0]["result"] == evidence["opaque-1"]["result"]
    assert "output" not in sent["cited_evidence"][0]

    verified = verify_structured_claims(
        production.claims,
        evidence,
        aliases={"call_1": "opaque-1"},
    )
    assert verified.verified is True


def test_model_claim_producer_rejects_self_identifying_preaggregated_field(
    fake_client_factory,
) -> None:
    answer = "There are 3 doors. [ref: call_1]"
    claim = _produced_field_claim(answer)
    claim["row_identity"] = [{
        "field": "count",
        "operator": "eq",
        "value": 3,
        "unit": None,
        "unit_field": None,
    }]
    payload = {
        "contract_version": CLAIM_CONTRACT_VERSION,
        "claims": [claim],
        "coverage": {
            "project_claims_found": 1,
            "structured_claims_produced": 1,
            "complete": True,
            "uncovered_claims": [],
        },
    }
    client = fake_client_factory(final_response(1, json.dumps(payload)))
    producer = StructuredClaimProducer(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    result = producer.produce(
        question="How many doors?",
        final_answer=answer,
        interpretation_plan={},
        evidence={
            "call_1": {
                "tool": "query_bim_workspace",
                "evidence_class": "project",
                "arguments": {},
                "result": {"rows": [{"count": 3}]},
            }
        },
    )

    assert result.contract_valid is False
    assert "independently of the claimed field" in result.contract_error


def test_model_claim_producer_rejects_non_exact_claim_coverage(fake_client_factory) -> None:
    answer = "There are 3 doors. [ref: call_1]"
    malformed = {
        "contract_version": CLAIM_CONTRACT_VERSION,
        "claims": [_produced_field_claim("There are three doors. [ref: call_1]")],
        "coverage": {
            "project_claims_found": 1,
            "structured_claims_produced": 1,
            "complete": True,
            "uncovered_claims": [],
        },
    }
    client = fake_client_factory(final_response(1, json.dumps(malformed)))
    producer = StructuredClaimProducer(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    result = producer.produce(
        question="How many doors?",
        final_answer=answer,
        interpretation_plan={},
        evidence={
            "call_1": {
                "tool": "query_bim_workspace",
                "evidence_class": "project",
                "arguments": {},
                "result": {"rows": [{"category": "doors", "count": 3}]},
            }
        },
    )

    assert result.contract_valid is False
    assert result.coverage_complete is False
    assert "exact final-answer substring" in result.contract_error


def test_model_claim_producer_preserves_explicit_uncovered_claims(fake_client_factory) -> None:
    answer = "The three files are fully synchronized. [ref: call_1]"
    payload = {
        "contract_version": CLAIM_CONTRACT_VERSION,
        "claims": [],
        "coverage": {
            "project_claims_found": 1,
            "structured_claims_produced": 0,
            "complete": False,
            "uncovered_claims": [{
                "claim_text": answer,
                "reason": "unstructured_evidence",
            }],
        },
    }
    client = fake_client_factory(final_response(1, json.dumps(payload)))
    producer = StructuredClaimProducer(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    result = producer.produce(
        question="Are the files synchronized?",
        final_answer=answer,
        interpretation_plan={},
        evidence={
            "call_1": {
                "tool": "inspect_project",
                "evidence_class": "project",
                "arguments": {},
                "result": {"source_files": []},
            }
        },
    )

    assert result.contract_valid is True
    assert result.coverage_complete is False
    assert result.payload["coverage"]["uncovered_claims"][0]["reason"] == "unstructured_evidence"
