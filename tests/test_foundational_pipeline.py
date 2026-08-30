from __future__ import annotations

import json
from pathlib import Path

from bim_agent import BimAgent
from bim_agent.agent_loop import (
    _completion_issues,
    _run_reconciliation_status,
    _structured_claim_coverage_issues,
)
from bim_agent.config import Settings
from bim_agent.project_tools import RawProjectTools

from conftest import final_response, tool_response


def _settings(sample_data: Path, tmp_path: Path, **overrides) -> Settings:
    values = {
        "data_dir": sample_data,
        "trace_dir": tmp_path / "traces",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "max_agent_iterations": 8,
        "enable_local_python": False,
    }
    values.update(overrides)
    return Settings(**values)


def _ambiguous_ranking_plan() -> dict:
    return {
        "route": {
            "answer_shape": "ranking",
            "required_capabilities": ["schema_inspection", "record_query", "geometry", "reconciliation"],
            "required_sources": ["properties", "ifc_geometry"],
            "preferred_compute": "specialized_ifc",
            "confidence": 0.99,
            "uncertainties": [],
        },
        "interpretation_plan": {
            "objective": "Identify the biggest element.",
            "population": {
                "description": "All physical project elements.",
                "identity_basis": "ifc.global_id",
                "universe": "all_ifc_products",
                "filters": [],
                "inclusions": ["physical instances"],
                "exclusions": ["spatial and definition entities"],
            },
            "metrics": [{
                "name": "ranking metric",
                "definition": "The engineer must choose solid volume, maximum extent, or surface area.",
                "aggregation": "rank_descending",
                "value_field": "",
                "unit": "",
                "source_basis": "ifc_geometry",
                "null_policy": "fail",
                "group_by": [],
            }],
            "relationship": {
                "meaning": "",
                "direction": "not_applicable",
                "relationship_types": [],
            },
            "inclusion_exclusion_rationale": "Definitions are not installed elements.",
            "assumptions": [],
            "ambiguities": [{
                "term": "biggest",
                "alternatives": ["solid volume", "maximum extent", "surface area"],
                "material": True,
                "resolution_basis": "Different metrics can select different elements.",
            }],
            "execution_decision": "clarify",
            "clarification_question": (
                "For ‘biggest’, should I rank by solid volume, maximum extent, or surface area?"
            ),
        },
    }


def _executable_count_plan() -> dict:
    return {
        "route": {
            "answer_shape": "count",
            "required_capabilities": ["schema_inspection", "hierarchy", "record_query", "reconciliation"],
            "required_sources": ["tree", "properties"],
            "preferred_compute": "sql",
            "confidence": 0.99,
            "uncertainties": [],
        },
        "interpretation_plan": {
            "objective": "Count installed pipe records.",
            "population": {
                "description": "Installed pipe records",
                "identity_basis": "records.object_id",
                "universe": "filtered_records",
                "filters": [{
                    "source": "records", "field": "name", "operator": "starts_with", "value": "Pipe [",
                }],
                "inclusions": [],
                "exclusions": [],
            },
            "metrics": [{
                "name": "instance count",
                "definition": "COUNT(DISTINCT records.object_id)",
                "aggregation": "distinct_count",
                "value_field": "records.object_id",
                "unit": "count",
                "source_basis": "derived",
                "null_policy": "not_applicable",
                "group_by": [],
            }],
            "relationship": {
                "meaning": "", "direction": "not_applicable", "relationship_types": [],
            },
            "inclusion_exclusion_rationale": "Definitions are not installed instances.",
            "assumptions": [],
            "ambiguities": [],
            "execution_decision": "execute",
            "clarification_question": "",
        },
    }


def _claim_payload(answer: str, *, evidence_ref: str = "call_2") -> dict:
    return {
        "contract_version": "1.0",
        "claims": [{
            "claim_id": "pipe-count",
            "claim_text": answer,
            "evidence_ref": evidence_ref,
            "row_path": ["rows"],
            "kind": "field",
            "row_identity": [
                {
                    "field": "population", "operator": "eq", "value": "Pipe",
                    "unit": None, "unit_field": None,
                },
                {
                    "field": "metric", "operator": "eq", "value": "instance_count",
                    "unit": None, "unit_field": None,
                },
            ],
            "field": "count",
            "operator": "eq",
            "expected_value": 2,
            "expected_unit": "count",
            "unit_field": None,
            "absolute_tolerance": 0,
            "relative_tolerance": 0,
        }],
        "coverage": {
            "project_claims_found": 1,
            "structured_claims_produced": 1,
            "complete": True,
            "uncovered_claims": [],
        },
    }


def test_material_ambiguity_allows_read_only_inspection_before_clarification(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        final_response(1, json.dumps(_ambiguous_ranking_plan())),
        tool_response(2, "describe_bim_workspace", {}),
        final_response(
            3,
            "Should I rank biggest by solid volume, maximum extent, or surface area?",
        ),
    )
    report = BimAgent(
        settings=_settings(
            sample_data,
            tmp_path,
            enable_question_planning=True,
            planning_model="gpt-5.6-sol",
        ),
        client=client,
    ).ask("Which element is biggest?")

    assert report.status == "limited"
    assert report.answer.startswith("Should I rank")
    assert report.agent_loop["iterations_used"] == 2
    assert report.agent_loop["termination_reason"] == "clarification_requested"
    assert report.status_dimensions["ambiguity"] == "clarification_required"
    assert report.status_dimensions["reconciliation"] == "not_evaluated"
    assert len(client.responses.requests) == 3
    assert client.responses.requests[0]["text"]["format"]["name"] == "bim_question_plan"
    assert report.agent_loop["iterations"][0]["calls"][0]["tool"] == "describe_bim_workspace"


def test_structured_claim_gate_requires_a_validated_interpretation_plan(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    answer_1 = "population=Pipe; metric=instance_count; count=2 count [ref: call_1]"
    answer_2 = "population=Pipe; metric=instance_count; count=2 count [ref: call_2]"
    incomplete = {
        "contract_version": "1.0",
        "claims": [],
        "coverage": {
            "project_claims_found": 1,
            "structured_claims_produced": 0,
            "complete": False,
            "uncovered_claims": [{
                "claim_text": answer_1,
                "reason": "insufficient_row_identity",
            }],
        },
    }
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"],
            "row_limit": 20,
        }),
        final_response(2, answer_1),
        final_response(3, json.dumps(incomplete)),
        tool_response(4, "query_bim_workspace", {
            "sql": (
                "SELECT MIN(substr(name, 1, 4)) AS population, 'instance_count' AS metric, "
                "COUNT(DISTINCT object_id) AS count FROM records WHERE name LIKE ?"
            ),
            "parameters": ["Pipe [%"],
            "row_limit": 20,
        }),
        final_response(5, answer_2),
        final_response(6, json.dumps(_claim_payload(answer_2))),
    )
    report = BimAgent(
        settings=_settings(
            sample_data,
            tmp_path,
            enable_structured_claim_verification=True,
            claim_verification_model="gpt-5.6-sol",
        ),
        client=client,
    ).ask("How many pipes?")

    assert report.status == "limited"
    assert report.agent_loop["structured_claim_forced"] is True
    assert len(report.agent_loop["structured_claim_attempts"]) == 2
    assert report.agent_loop["structured_claim_verification"]["verified"] is True
    assert report.status_dimensions["evidence"] == "structured_claim_verification_rejected"
    assert any(
        item.get("action") == "structured_claim_continue"
        for item in report.agent_loop["iterations"]
    )


def test_structured_coverage_rejects_value_bound_to_wrong_population() -> None:
    answer = "There are 5 doors. [ref: call_1]"
    payload = _claim_payload(answer, evidence_ref="call_1")
    payload["claims"][0]["expected_value"] = 5
    payload["claims"][0]["row_identity"][0]["value"] = "walls"

    issues = _structured_claim_coverage_issues([answer], payload)

    assert any("interpretation plan" in item for item in issues)


def test_structured_coverage_rejects_wrong_aggregate_population() -> None:
    answer = "There are 5 doors. [ref: call_1]"
    payload = {
        "claims": [{
            "claim_text": answer,
            "kind": "aggregate",
            "expected_value": 5,
            "filters": [{"field": "category", "operator": "eq", "value": "walls"}],
        }],
        "coverage": {"complete": True, "uncovered_claims": []},
    }

    issues = _structured_claim_coverage_issues([answer], payload)

    assert any("interpretation plan" in item for item in issues)


def test_answer_quantity_cannot_bind_to_an_identity_number() -> None:
    answer = "There are 5 doors. [ref: call_1]"
    payload = _claim_payload(answer, evidence_ref="call_1")
    payload["claims"][0]["row_identity"] = [
        {"field": "floor", "operator": "eq", "value": 5},
        {"field": "category", "operator": "eq", "value": "doors"},
    ]
    payload["claims"][0]["expected_value"] = 3

    issues = _structured_claim_coverage_issues([answer], payload)

    assert any("expected value" in item.casefold() and "absent" in item.casefold() for item in issues)


def test_sql_tool_cannot_self_certify_cursor_supersession(sample_data: Path) -> None:
    definition = next(
        item for item in RawProjectTools(sample_data).definitions()
        if item["name"] == "query_bim_workspace"
    )

    properties = definition["parameters"]["properties"]
    assert "supersedes_cursors" not in properties
    assert definition["parameters"]["required"] == ["sql", "parameters", "row_limit"]


def test_required_reconciliation_allows_later_match_to_supersede_exploration() -> None:
    assert _run_reconciliation_status(
        required=True,
        outcomes=[
            {"automatic": False, "status": "identity_validated", "provenance_validated": True},
            {"automatic": False, "status": "matched", "provenance_validated": True},
        ],
    ) == "matched"
    assert _run_reconciliation_status(
        required=True,
        outcomes=[{"automatic": True, "status": "identity_validated"}],
    ) == "incomplete"


def test_voluntary_incomplete_reconciliation_is_not_reported_as_identity_validated() -> None:
    assert _run_reconciliation_status(
        required=False,
        outcomes=[{
            "automatic": False,
            "status": "incomplete",
            "provenance_validated": True,
        }],
    ) == "incomplete"
    assert _run_reconciliation_status(
        required=False,
        outcomes=[
            {"automatic": False, "status": "incomplete", "provenance_validated": True},
            {"automatic": False, "status": "matched", "provenance_validated": True},
        ],
    ) == "matched"


def test_executable_project_plan_rejects_scaffolding_only_answer(
    sample_data: Path, fake_client_factory,
) -> None:
    agent = BimAgent(sample_data, client=fake_client_factory()).agent
    plan = _executable_count_plan()["interpretation_plan"]
    plan = {**plan, "answer_shape": "count"}

    issues, contract, verification, _, _ = agent._verify_structured_candidate(
        question="How many pipes?",
        candidate_answer="Summary",
        interpretation_plan=plan,
        evidence={},
        evidence_aliases={},
        trusted_claims=set(),
    )

    assert any("no cited project claim" in issue.casefold() for issue in issues)
    assert contract is not None and contract["contract_valid"] is False
    assert verification is not None and verification["verified"] is False


def test_route_requirements_are_enforced_but_invalid_fallback_requirements_are_not() -> None:
    base = {
        "answer_shape": "measurement",
        "required_capabilities": ["schema_inspection", "geometry"],
        "required_sources": ["ifc_geometry"],
        "requirements_enforced": True,
    }
    issues = _completion_issues(
        "Measure the door",
        tool_categories={"schema", "sql"},
        outstanding_cursors=set(),
        reconciliation_required=False,
        reconciliation_status="not_required",
        route=base,
        observed_sources={"tree", "properties"},
    )
    fallback = {**base, "requirements_enforced": False}
    fallback_issues = _completion_issues(
        "Measure the door",
        tool_categories={"schema", "sql"},
        outstanding_cursors=set(),
        reconciliation_required=False,
        reconciliation_status="not_required",
        route=fallback,
        observed_sources={"tree", "properties"},
    )

    assert "the planned geometry capability" in issues
    assert "the planned ifc_geometry evidence source" in issues
    assert not any(item.startswith("the planned ") for item in fallback_issues)


def test_planning_execution_reconciliation_review_and_claim_verification_work_together(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    question = 'How many records have name starting with "Pipe ["?'
    answer = "population=Pipe; metric=instance_count; count=2 count [ref: call_2]"
    review = {
        "can_finalize": True,
        "summary": "The identified SQL population and count are consistent.",
        "required_follow_up": [],
        "required_disclosures": [],
        "unsupported_claims": [],
    }
    client = fake_client_factory(
        final_response(1, json.dumps(_executable_count_plan())),
        tool_response(2, "describe_bim_workspace", {}),
        tool_response(3, "query_bim_workspace", {
            "sql": (
                "SELECT MIN(substr(name, 1, 4)) AS population, 'instance_count' AS metric, "
                "COUNT(DISTINCT object_id) AS count FROM records WHERE name GLOB ?"
            ),
            "parameters": ["Pipe [[]*"],
            "row_limit": 20,
        }),
        tool_response(4, "query_bim_workspace", {
            "sql": "SELECT object_id FROM tree_nodes WHERE name GLOB ?",
            "parameters": ["Pipe [[]*"],
            "row_limit": 20,
        }),
        tool_response(5, "query_bim_workspace", {
            "sql": "SELECT DISTINCT object_id FROM records WHERE name GLOB ?",
            "parameters": ["Pipe [[]*"],
            "row_limit": 20,
        }),
        tool_response(6, "reconcile_populations", {
            "populations": [
                {"label": "tree", "evidence_refs": ["call_3"]},
                {"label": "properties", "evidence_refs": ["call_4"]},
            ],
            "require_equal_populations": True,
            "require_ifc_mapping": False,
        }),
        tool_response(7, "review_scope_and_evidence", {
            "question": question,
            "selected_scope": "Installed pipe records",
            "exclusions": "Hierarchy definitions",
            "evidence": "Identified aggregate row and equal source populations",
            "reconciliation": "tree and properties object IDs match",
            "draft_answer": answer,
        }),
        final_response(8, json.dumps(review)),
        final_response(9, answer),
        final_response(10, json.dumps(_claim_payload(answer, evidence_ref="call_2"))),
    )
    report = BimAgent(
        settings=_settings(
            sample_data,
            tmp_path,
            enable_question_planning=True,
            planning_model="gpt-5.6-sol",
            enable_structured_claim_verification=True,
            claim_verification_model="gpt-5.6-sol",
        ),
        client=client,
    ).ask(question)

    assert report.status == "completed"
    assert report.agent_loop["planning"]["contract_valid"] is True
    assert report.agent_loop["reconciliation_status"] == "matched"
    assert report.agent_loop["review_seen"] is True
    assert report.agent_loop["structured_claim_verification"]["verified"] is True
    assert report.status_dimensions["evidence"] == "structured_claim_verification_passed"
    assert report.status_dimensions["reconciliation"] == "matched"
