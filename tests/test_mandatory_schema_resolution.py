from __future__ import annotations

import json
from pathlib import Path

import pytest

from bim_agent import BimAgent
from bim_agent.config import Settings
from bim_agent.project_tools import RawProjectTools
from bim_agent.question_planning import (
    QuestionPlanner,
    _candidate_provenance_validation,
    _normalize_schema_resolution_payload,
    _requires_schema_resolution_retry,
    _schema_resolution_candidate_inventory,
    _schema_resolution_public_inventory,
)

from conftest import final_response


QUESTION = "כמה מפסקים בפרויקט?"
QUESTION_TERM = "מפסקים"


def test_public_schema_inventory_keeps_taxonomy_and_drops_unmentioned_leaf_noise(
    sample_data: Path,
) -> None:
    inventory = _schema_resolution_candidate_inventory(RawProjectTools(sample_data))
    lighting = _candidate(inventory, label="Lighting Devices", kind="hierarchy_category")

    public = _schema_resolution_public_inventory(
        inventory,
        forced_candidate_ids={lighting["candidate_id"]},
        question=QUESTION,
    )

    assert any(item["candidate_id"] == lighting["candidate_id"] for item in public)
    assert any(item["label"] == "Door Switch" for item in public)
    assert all(item["kind"] != "named_element" for item in public)


@pytest.mark.parametrize("answer_shape", ["count", "list"])
def test_partial_population_gate_is_shape_generic_for_count_and_list(answer_shape: str) -> None:
    interpretation = {
        "answer_shape": answer_shape,
        "population": {
            "universe": "filtered_records",
            "filters": [
                {"binding": "observed_schema_mapping"},
                {"binding": "verbatim"},
            ],
        },
        "schema_grounded_mappings": [{"binding": "observed_schema_mapping"}],
        "ambiguities": [{"term": "population question binding", "material": True}],
        "execution_decision": "clarify",
    }

    assert _requires_schema_resolution_retry(interpretation) is True


def test_runtime_candidate_reconciliation_detects_missing_source_identity(
    sample_data: Path,
) -> None:
    tools = RawProjectTools(sample_data)

    validation = _candidate_provenance_validation(
        tools,
        ["13"],
        expected_tree_ids=["13", "tree-only-id"],
    )

    assert validation["status"] == "mismatched"
    assert validation["provenance_validated"] is True
    assert validation["missing_property_identity_count"] == 1
    assert validation["sample_missing_property_ids"] == ["tree-only-id"]


def _empty_count_plan() -> dict:
    """Planner output with a raw ambiguous population and no injected mapping."""

    return {
        "route": {
            "answer_shape": "count",
            "required_capabilities": ["schema_inspection", "hierarchy", "record_query"],
            "required_sources": ["tree", "properties"],
            "preferred_compute": "sql",
            "confidence": 0.72,
            "uncertainties": ["The project taxonomy uses different terminology."],
        },
        "interpretation_plan": {
            "objective": "Count the requested installed elements.",
            "population": {
                "description": "Installed records matching the user's raw category term.",
                "identity_basis": "records.object_id",
                "universe": "filtered_records",
                "filters": [],
                "inclusions": [],
                "exclusions": [],
            },
            "metrics": [{
                "name": "instance_count",
                "definition": "COUNT(DISTINCT records.object_id)",
                "aggregation": "distinct_count",
                "value_field": "records.object_id",
                "unit": "count",
                "source_basis": "derived",
                "null_policy": "not_applicable",
                "group_by": [],
                "result_limit": 0,
            }],
            "relationship": {
                "meaning": "",
                "direction": "not_applicable",
                "relationship_types": [],
            },
            "inclusion_exclusion_rationale": "No population mapping was chosen silently.",
            "assumptions": [],
            "ambiguities": [{
                "term": QUESTION_TERM,
                "alternatives": ["Lighting Devices", "Door Switch"],
                "material": True,
                "resolution_basis": "The raw term can map to multiple observed populations.",
            }],
            "execution_decision": "clarify",
            "clarification_question": "Which project population do you mean?",
        },
    }


def _partially_grounded_count_plan(
    *,
    unrelated_material_ambiguity: bool = False,
    invalid_binding: str = "observed_schema_mapping",
) -> dict:
    payload = _empty_count_plan()
    plan = payload["interpretation_plan"]
    plan["population"]["filters"] = [
        {
            "source": "records",
            "field": "path_text",
            "operator": "contains",
            "value": "Lighting Devices",
            "binding": "observed_schema_mapping",
            "question_term": QUESTION_TERM,
        },
        {
            "source": "tree",
            "field": "child_count",
            "operator": "equals",
            "value": "0",
            "binding": invalid_binding,
            "question_term": "0" if invalid_binding == "verbatim" else "בפרוייקט",
        },
    ]
    plan["execution_decision"] = "inspect_then_execute"
    plan["clarification_question"] = ""
    plan["ambiguities"][0]["material"] = False
    if unrelated_material_ambiguity:
        plan["ambiguities"].append({
            "term": "unrelated relationship meaning",
            "alternatives": ["physical connection", "system assignment"],
            "material": True,
            "resolution_basis": "Schema population matching cannot resolve relationship semantics.",
        })
    return payload


def _fully_grounded_count_plan() -> dict:
    payload = _partially_grounded_count_plan()
    plan = payload["interpretation_plan"]
    plan["population"]["filters"] = plan["population"]["filters"][:1]
    plan["ambiguities"] = []
    return payload


def _candidate(inventory: list[dict], *, label: str, kind: str) -> dict:
    return next(
        item for item in inventory
        if item["label"] == label and item["kind"] == kind
    )


def test_raw_ambiguous_keyword_forces_schema_mapping_without_injection(
    sample_data: Path,
    fake_client_factory,
) -> None:
    tools = RawProjectTools(sample_data)
    inventory = _schema_resolution_candidate_inventory(tools)
    lighting = _candidate(inventory, label="Lighting Devices", kind="hierarchy_category")
    door_switch = _candidate(inventory, label="Door Switch", kind="hierarchy_type")
    retry_payload = {
        "question_terms": [QUESTION_TERM],
        "resolved_ambiguity_terms": [QUESTION_TERM],
        "candidates": [
            {
                "candidate_id": door_switch["candidate_id"],
                "match_relation": "related_optional_element",
                "selection_reason": "A separately authored door-opening switch also matches.",
            },
            {
                "candidate_id": lighting["candidate_id"],
                "match_relation": "direct_category_match",
                "selection_reason": "This is the direct observed project category.",
            },
        ],
    }
    client = fake_client_factory(
        final_response(1, json.dumps(_empty_count_plan(), ensure_ascii=False)),
        final_response(2, json.dumps(retry_payload, ensure_ascii=False)),
    )
    planner = QuestionPlanner(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
        project_tools=tools,
    )

    initial = planner.plan(QUESTION)
    assert initial.requires_schema_resolution_retry is True
    assert initial.interpretation_plan["schema_grounded_mappings"] == []

    resolved = planner.resolve_schema_population(QUESTION, initial)

    assert resolved.requires_schema_resolution_retry is False
    assert resolved.requires_clarification is False
    mappings = resolved.interpretation_plan["schema_grounded_mappings"]
    assert mappings[0]["question_term"] == QUESTION_TERM
    assert mappings[0]["value"] == "Lighting Devices"
    candidates = resolved.interpretation_plan["schema_resolution_candidates"]
    assert [(item["role"], item["label"]) for item in candidates] == [
        ("primary", "Lighting Devices"),
        ("alternate", "Door Switch"),
    ]
    observation = resolved.schema_resolution_observation
    assert observation is not None
    assert observation["population_size_used_for_ranking"] is False
    assert [item["instance_count"] for item in observation["candidates"]] == [3, 1]
    assert observation["candidates"][1]["additional_unique_count"] == 1
    assert observation["candidates"][1]["combined_with_primary_count"] == 4
    assert len(client.responses.requests) == 2
    assert client.responses.requests[1]["text"]["format"]["name"] == (
        "bim_schema_population_resolution"
    )
    sent_inventory = json.loads(client.responses.requests[1]["input"])["observed_candidates"]
    assert {item["candidate_id"] for item in sent_inventory}.issuperset({
        lighting["candidate_id"], door_switch["candidate_id"],
    })
    assert all("instance_count" not in item for item in sent_inventory)


def test_zero_validated_candidates_is_the_only_population_clarification_fallback(
    sample_data: Path,
    fake_client_factory,
) -> None:
    client = fake_client_factory(
        final_response(1, json.dumps(_empty_count_plan(), ensure_ascii=False)),
        final_response(2, json.dumps({
            "question_terms": [QUESTION_TERM],
            "resolved_ambiguity_terms": [],
            "candidates": [],
        }, ensure_ascii=False)),
    )
    planner = QuestionPlanner(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
        project_tools=RawProjectTools(sample_data),
    )

    resolved = planner.resolve_schema_population(QUESTION, planner.plan(QUESTION))

    assert resolved.requires_clarification is True
    assert resolved.schema_resolution_observation is None
    assert any("validated zero candidates" in item for item in resolved.normalization_warnings)


def test_accepted_mapping_still_requires_runtime_population_validation(
    sample_data: Path,
    fake_client_factory,
) -> None:
    tools = RawProjectTools(sample_data)
    inventory = _schema_resolution_candidate_inventory(tools)
    contained_type = _candidate(inventory, label="Alpha Switch", kind="hierarchy_type")
    client = fake_client_factory(
        final_response(1, json.dumps(_fully_grounded_count_plan(), ensure_ascii=False)),
        final_response(2, json.dumps({
            "question_terms": [QUESTION_TERM],
            "resolved_ambiguity_terms": [],
            "candidates": [{
                "candidate_id": contained_type["candidate_id"],
                "match_relation": "direct_type_match",
                "selection_reason": "A matching authored type inside the primary category.",
            }],
        }, ensure_ascii=False)),
    )
    planner = QuestionPlanner(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
        project_tools=tools,
    )

    initial = planner.plan(QUESTION)
    assert initial.interpretation_plan["schema_grounded_mappings"]
    assert initial.requires_schema_resolution_retry is True

    resolved = planner.resolve_schema_population(QUESTION, initial)

    assert resolved.requires_clarification is False
    assert resolved.schema_resolution_observation is not None
    assert resolved.schema_resolution_observation["reconciliation_status"] == "matched"
    assert resolved.schema_resolution_observation["provenance_validated"] is True
    assert resolved.interpretation_plan["schema_resolution_candidates"][0]["label"] == (
        "Lighting Devices"
    )
    assert [
        item["role"] for item in resolved.interpretation_plan["schema_resolution_candidates"]
    ] == ["primary", "component"]
    assert resolved.schema_resolution_observation["candidates"][1]["additional_unique_count"] == 0


@pytest.mark.parametrize("invalid_binding", ["observed_schema_mapping", "verbatim"])
def test_partially_grounded_population_is_rebuilt_instead_of_poisoning_execution(
    sample_data: Path,
    fake_client_factory,
    invalid_binding: str,
) -> None:
    tools = RawProjectTools(sample_data)
    inventory = _schema_resolution_candidate_inventory(tools)
    lighting = _candidate(inventory, label="Lighting Devices", kind="hierarchy_category")
    client = fake_client_factory(
        final_response(1, json.dumps(
            _partially_grounded_count_plan(invalid_binding=invalid_binding),
            ensure_ascii=False,
        )),
        final_response(2, json.dumps({
            "question_terms": [QUESTION_TERM],
            "resolved_ambiguity_terms": ["population question binding"],
            "candidates": [{
                "candidate_id": lighting["candidate_id"],
                "match_relation": "direct_category_match",
                "selection_reason": "The accepted category candidate replaces the rejected extra predicate.",
            }],
        }, ensure_ascii=False)),
    )
    planner = QuestionPlanner(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
        project_tools=tools,
    )

    initial = planner.plan(QUESTION)
    assert len(initial.interpretation_plan["population"]["filters"]) == 2
    assert len(initial.interpretation_plan["schema_grounded_mappings"]) == 1
    assert initial.requires_schema_resolution_retry is True

    resolved = planner.resolve_schema_population(QUESTION, initial)

    assert resolved.requires_clarification is False
    assert resolved.interpretation_plan["population"]["filters"] == [
        resolved.interpretation_plan["schema_resolution_candidates"][0]["filter"]
    ]
    assert resolved.interpretation_plan["population"]["filters"][0]["value"] == (
        "Lighting Devices"
    )
    assert not any(
        item.get("material") for item in resolved.interpretation_plan["ambiguities"]
    )


def test_population_retry_does_not_override_unrelated_material_ambiguity(
    sample_data: Path,
    fake_client_factory,
) -> None:
    client = fake_client_factory(
        final_response(
            1,
            json.dumps(
                _partially_grounded_count_plan(unrelated_material_ambiguity=True),
                ensure_ascii=False,
            ),
        ),
    )
    planner = QuestionPlanner(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="low",
        project_tools=RawProjectTools(sample_data),
    )

    initial = planner.plan(QUESTION)
    resolved = planner.resolve_schema_population(QUESTION, initial)

    assert initial.requires_schema_resolution_retry is True
    assert resolved.requires_clarification is True
    assert any(
        item.get("term") == "unrelated relationship meaning" and item.get("material")
        for item in resolved.interpretation_plan["ambiguities"]
    )
    assert len(client.responses.requests) == 2


def test_agent_loop_cannot_skip_the_mandatory_retry(
    sample_data: Path,
    tmp_path: Path,
    fake_client_factory,
) -> None:
    inventory = _schema_resolution_candidate_inventory(RawProjectTools(sample_data))
    lighting = _candidate(inventory, label="Lighting Devices", kind="hierarchy_category")
    door_switch = _candidate(inventory, label="Door Switch", kind="hierarchy_type")
    retry_payload = {
        "question_terms": [QUESTION_TERM],
        "resolved_ambiguity_terms": [QUESTION_TERM],
        "candidates": [
            {
                "candidate_id": lighting["candidate_id"],
                "match_relation": "direct_category_match",
                "selection_reason": "Direct category match.",
            },
            {
                "candidate_id": door_switch["candidate_id"],
                "match_relation": "related_optional_element",
                "selection_reason": "Optional related element.",
            },
        ],
    }
    client = fake_client_factory(
        final_response(1, json.dumps(_empty_count_plan(), ensure_ascii=False)),
        final_response(2, json.dumps(retry_payload, ensure_ascii=False)),
        final_response(3, "נמצאו 3 מפסקים; מפסק הדלת הוא חלופה נוספת. [ref: call_1]"),
    )
    report = BimAgent(
        settings=Settings(
            data_dir=sample_data,
            trace_dir=tmp_path / "traces",
            model="gpt-5.6-sol",
            planning_model="gpt-5.6-sol",
            reasoning_effort="low",
            max_agent_iterations=1,
            enable_question_planning=True,
            enable_structured_claim_verification=False,
            enable_local_python=False,
        ),
        client=client,
    ).ask(QUESTION)

    assert report.agent_loop["iterations_used"] == 1
    assert len(client.responses.requests) == 3
    planning = report.agent_loop["planning"]
    assert planning["interpretation_plan"]["schema_grounded_mappings"][0]["value"] == (
        "Lighting Devices"
    )
    assert planning["schema_resolution"]["primary_candidate_id"] == lighting["candidate_id"]
    assert report.agent_loop["reconciliation_status"] == "matched"
    execution_input = client.responses.requests[2]["input"]
    assert any(
        "Mandatory runtime schema-resolution observation" in str(item.get("content") or "")
        for item in execution_input if isinstance(item, dict)
    )


def test_specificity_ranking_cannot_be_overridden_by_population_size() -> None:
    inventory = [
        {
            "candidate_id": "small-category",
            "label": "Switches",
            "kind": "hierarchy_category",
            "path": ["Model", "Switches"],
            "filter": {},
            "record_object_ids": ["1"],
        },
        {
            "candidate_id": "large-related-type",
            "label": "Related controls",
            "kind": "type_name",
            "path": ["Type Name", "Related controls"],
            "filter": {},
            "record_object_ids": [str(index) for index in range(100)],
        },
    ]
    payload = {
        "question_terms": ["Switches"],
        "resolved_ambiguity_terms": ["Switches"],
        "candidates": [
            {
                "candidate_id": "large-related-type",
                "match_relation": "related_optional_element",
                "selection_reason": "Related but broad.",
            },
            {
                "candidate_id": "small-category",
                "match_relation": "exact_text_match",
                "selection_reason": "Exact category wording.",
            },
        ],
    }

    normalized = _normalize_schema_resolution_payload(
        payload,
        question="How many Switches?",
        inventory=inventory,
        ambiguity_terms={"Switches"},
    )

    assert [item["candidate_id"] for item in normalized["selected"]] == [
        "small-category", "large-related-type",
    ]


def test_model_cannot_forge_exact_or_category_specificity() -> None:
    inventory = [{
        "candidate_id": "door-type",
        "label": "Door Switch",
        "kind": "hierarchy_type",
        "path": ["Model", "Fixtures", "Door Switch"],
        "filter": {},
        "record_object_ids": ["23"],
    }]
    base = {
        "question_terms": [QUESTION_TERM],
        "resolved_ambiguity_terms": [QUESTION_TERM],
        "candidates": [{
            "candidate_id": "door-type",
            "selection_reason": "Attempted rank escalation.",
            "match_relation": "exact_text_match",
        }],
    }

    with pytest.raises(ValueError, match="exact text match"):
        _normalize_schema_resolution_payload(
            base,
            question=QUESTION,
            inventory=inventory,
            ambiguity_terms={QUESTION_TERM},
        )

    base["candidates"][0]["match_relation"] = "direct_category_match"
    with pytest.raises(ValueError, match="not a category"):
        _normalize_schema_resolution_payload(
            base,
            question=QUESTION,
            inventory=inventory,
            ambiguity_terms={QUESTION_TERM},
        )
