from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bim_agent import BimAgent
from bim_agent.config import Settings
from bim_agent.question_planning import QuestionPlan

from conftest import final_response, function_call, tool_response


class _StaticPlanner:
    def __init__(self, plan: QuestionPlan):
        self.plan_result = plan

    def plan(self, question: str) -> QuestionPlan:
        return self.plan_result

    def resolve_schema_population(
        self, question: str, initial_plan: QuestionPlan,
    ) -> QuestionPlan:
        return initial_plan


def test_model_chooses_tools_and_owns_completion(
    sample_data: Path, tmp_path: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(
        tool_response(1, "inspect_project", {}),
        tool_response(2, "search_records", {
            "terms": ["Pipe ["], "match": "all", "limit": 100,
        }),
        tool_response(3, "query_bim_workspace", {
            "sql": (
                "SELECT SUM(CAST(REPLACE(property_value, ' m', '') AS REAL)) AS total_m "
                "FROM properties WHERE property_key = ? AND object_id IN (?, ?)"
            ),
            "parameters": ["Dimensions.Length", "43", "44"], "row_limit": 20,
        }),
        final_response(4, "The total pipe length is 5.5 m. [ref: call-3]"),
    )
    report = BimAgent(sample_data, client=client).ask("What is the total pipe length?")

    assert report.status == "completed"
    assert report.answer.startswith("The total pipe length is 5.5 m. [ref: call-3]")
    assert "cross-file version alignment are unverified" in report.answer
    assert report.agent_loop["pattern"] == "model_directed_tool_loop"
    assert [
        call["tool"]
        for turn in report.agent_loop["iterations"]
        for call in turn.get("calls", [])
        if not call.get("automatic")
    ] == ["inspect_project", "search_records", "query_bim_workspace"]
    assert len(client.responses.requests) == 4
    assert all(request["tool_choice"] == "auto" for request in client.responses.requests)
    assert all(request["parallel_tool_calls"] is False for request in client.responses.requests)
    assert Path(report.trace_path).is_file()


def test_no_fixed_tool_order_or_required_pipeline(
    sample_data: Path, tmp_path: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(
        tool_response(1, "calculate", {"expression": "21 + 1"}),
        final_response(2, "The calculation result is 22. [ref: call-1]"),
    )
    report = BimAgent(sample_data, client=client).ask("Calculate 21 + 1")
    assert report.answer.endswith("[ref: call-1]")
    assert report.agent_loop["iterations"][0]["calls"][0]["tool"] == "calculate"
    assert report.agent_loop["iterations_used"] == 2


def test_tool_error_is_observed_by_model_instead_of_triggering_fallback(
    sample_data: Path, tmp_path: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(
        tool_response(1, "not_a_tool", {}),
        final_response(2, "I could not use that tool, so no project claim is made."),
    )
    report = BimAgent(sample_data, client=client).ask("Use an unavailable capability")
    call = report.agent_loop["iterations"][0]["calls"][0]
    assert call["outcome"] == "error"
    second_input = client.responses.requests[1]["input"]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in second_input
    )


def test_invalid_null_tool_observation_is_rejected_and_returned_as_an_error(
    sample_data: Path, tmp_path: Path, fake_client_factory, monkeypatch,
) -> None:
    client = fake_client_factory(
        tool_response(1, "inspect_project", {}),
        final_response(2, "The tool failed, so I make no project claim."),
    )
    agent = BimAgent(sample_data, client=client)
    monkeypatch.setattr(agent.tools, "execute", lambda _name, _arguments: None)
    report = agent.ask("Use the inspect capability")
    call = report.agent_loop["iterations"][0]["calls"][0]
    assert call["outcome"] == "error"
    observation = client.responses.requests[1]["input"][-1]["output"]
    assert "ToolObservationError" in observation


def test_trace_contains_tool_telemetry_not_private_reasoning(
    sample_data: Path, tmp_path: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(final_response(1, "No tools were needed."))
    report = BimAgent(sample_data, client=client).ask("Say whether tools are needed")
    events = [json.loads(line) for line in Path(report.trace_path).read_text(encoding="utf-8").splitlines()]
    assert [item["stage"] for item in events][-1] == "model_agent_finish"
    assert all("thought" not in item and "reasoning" not in item for item in events)


def test_gpt_56_enables_programmatic_tool_calling_with_scoped_callers(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        max_agent_iterations=4,
    )
    client = fake_client_factory(final_response(1, "Done."))
    report = BimAgent(settings=settings, client=client).ask("Say done")

    request_tools = client.responses.requests[0]["tools"]
    assert {item.get("type") for item in request_tools} >= {"function", "programmatic_tool_calling"}
    project_tool = next(item for item in request_tools if item.get("name") == "query_bim_workspace")
    reviewer = next(item for item in request_tools if item.get("name") == "review_scope_and_evidence")
    assert project_tool["allowed_callers"] == ["direct", "programmatic"]
    assert reviewer["allowed_callers"] == ["direct"]
    assert report.agent_loop["programmatic_tool_calling"] is True


def test_programmatic_function_caller_is_preserved_on_tool_output(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        max_agent_iterations=4,
    )
    call = function_call("calculate", {"expression": "2 + 3"})
    call.caller = SimpleNamespace(type="program", caller_id="program-123")
    client = fake_client_factory(
        SimpleNamespace(id="resp-1", output=[call], output_text=""),
        final_response(2, "5 [ref: call-1]"),
    )
    BimAgent(settings=settings, client=client).ask("Calculate 2 + 3")

    call_output = next(
        item for item in client.responses.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert call_output["caller"] == {"type": "program", "caller_id": "program-123"}


def test_primary_agent_can_invoke_model_backed_evidence_review(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=5,
    )
    client = fake_client_factory(
        tool_response(1, "review_scope_and_evidence", {
            "question": "How many switches?",
            "selected_scope": "Lighting Devices",
            "exclusions": "Switchboards",
            "evidence": "Three leaf records",
            "reconciliation": "3 unique object IDs = 3 found records",
            "draft_answer": "3",
        }),
        final_response(2, json.dumps({
            "can_finalize": False,
            "summary": "Check the Electrical Fixtures category as a plausible broader scope.",
            "required_follow_up": ["Inspect the broader fixture category."],
            "required_disclosures": [],
            "unsupported_claims": [],
        })),
        tool_response(3, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Alpha Switch [%"], "row_limit": 20,
        }),
        tool_response(4, "reconcile_populations", {
            "populations": [{"label": "lighting switches", "object_ids": ["13", "14", "16"]}],
        }),
        tool_response(5, "review_scope_and_evidence", {
            "question": "How many switches?",
            "selected_scope": "Lighting Devices",
            "exclusions": "Switchboards and door switches",
            "evidence": "Three leaf records",
            "reconciliation": "3 unique object IDs = 3 found records",
            "draft_answer": "3 lighting switches",
        }),
        final_response(6, json.dumps({
            "can_finalize": True,
            "summary": "The selected scope is sufficient.",
            "required_follow_up": [],
            "required_disclosures": [],
            "unsupported_claims": [],
        })),
        final_response(7, "There are 3 lighting switches. [ref: call_2] [ref: call_3]"),
    )
    report = BimAgent(settings=settings, client=client).ask("How many switches?")

    assert report.status == "completed"
    assert report.agent_loop["iterations"][0]["calls"][0]["tool"] == "review_scope_and_evidence"
    nested_request = client.responses.requests[1]
    assert "critical BIM evidence-review tool" in nested_request["instructions"]
    assert '"hierarchy_nodes":' not in nested_request["input"]
    assert '"source_inventory_complete": true' in nested_request["input"]
    assert nested_request["max_output_tokens"] == 1000
    assert nested_request["text"]["format"]["name"] == "bim_evidence_review"
    incremental_request = client.responses.requests[5]
    incremental_payload = json.loads(incremental_request["input"])
    assert incremental_payload["review_mode"] == "delta"
    assert "project_summary" not in incremental_payload
    assert sorted(incremental_payload["changed_inputs"]) == ["draft_answer", "exclusions"]
    assert incremental_request["max_output_tokens"] == 750
    model_observation = next(
        item["output"] for item in client.responses.requests[2]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert "Electrical Fixtures" in model_observation


def test_structured_review_blocks_finalization_until_follow_up_is_re_reviewed(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=8,
        enable_local_python=False,
    )
    review_args = {
        "question": "How many pipes?",
        "selected_scope": "Pipes",
        "exclusions": "Definitions",
        "evidence": "Two records",
        "reconciliation": "Routine SQL population",
        "draft_answer": "2 pipes",
    }
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(2, "review_scope_and_evidence", review_args),
        final_response(3, json.dumps({
            "can_finalize": False,
            "summary": "Verify the selected leaves.",
            "required_follow_up": ["Return the selected object IDs."],
            "required_disclosures": [],
            "unsupported_claims": [],
        })),
        final_response(4, "There are 2 pipes. [ref: call_1]"),
        tool_response(5, "query_bim_workspace", {
            "sql": "SELECT object_id FROM records WHERE name LIKE ? ORDER BY object_id",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(6, "review_scope_and_evidence", {
            **review_args,
            "evidence": "Two selected records: 43 and 44",
        }),
        final_response(7, json.dumps({
            "can_finalize": True,
            "summary": "The selected scope is supported.",
            "required_follow_up": [],
            "required_disclosures": [],
            "unsupported_claims": [],
        })),
        final_response(8, "There are 2 pipes. [ref: call_1] [ref: call_3]"),
    )

    report = BimAgent(settings=settings, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert report.agent_loop["completeness_forced"] is True
    assert report.agent_loop["review_can_finalize"] is True
    assert report.agent_loop["review_stale"] is False
    assert any(
        item.get("action") == "completeness_continue"
        and "required follow-up" in " ".join(item.get("issues", []))
        for item in report.agent_loop["iterations"]
    )


def test_structured_review_disclosure_codes_are_appended_with_project_and_review_refs(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=4,
        enable_local_python=False,
    )
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(2, "review_scope_and_evidence", {
            "question": "How many pipes?",
            "selected_scope": "Pipes",
            "exclusions": "Definitions",
            "evidence": "Two records",
            "reconciliation": "Routine SQL population",
            "draft_answer": "2 pipes",
        }),
        final_response(3, json.dumps({
            "can_finalize": True,
            "summary": "The count is sufficient with scope caveats.",
            "required_follow_up": [],
            "required_disclosures": [
                "loaded_snapshot", "record_count_not_physical_uniqueness",
            ],
            "unsupported_claims": [],
        })),
        final_response(4, "There are 2 pipe records. [ref: call_1]"),
    )

    report = BimAgent(settings=settings, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert "source freshness and cross-file version alignment are unverified" in report.answer
    assert "not proof of unique physical entities" in report.answer
    assert "[ref: call_1] [ref: call_2]" in report.answer
    assert report.agent_loop["completeness_forced"] is False


def test_runtime_executes_with_planned_tool_scope_and_records_contract(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    plan = QuestionPlan(
        route={
            "answer_shape": "narrative",
            "required_capabilities": ["hierarchy"],
            "required_sources": ["tree"],
            "preferred_compute": "none",
            "confidence": 0.8,
            "uncertainties": ["The requested population is ambiguous."],
            "tool_policy": "safe_superset",
            "exposed_tool_names": ["inspect_project"],
            "project_data_operation": True,
            "requirements_enforced": True,
        },
        interpretation_plan={
            "objective": "Resolve an ambiguous population.",
            "population": {
                "description": "Unresolved project population.",
                "identity_basis": "Unresolved until inspection.",
                "universe": "not_applicable",
                "filters": [],
                "inclusions": [],
                "exclusions": [],
            },
            "metrics": [],
            "relationship": {
                "meaning": "",
                "direction": "not_applicable",
                "relationship_types": [],
            },
            "inclusion_exclusion_rationale": "No scope was silently selected.",
            "assumptions": [],
            "ambiguities": [{
                "term": "requested population",
                "alternatives": ["candidate one", "candidate two"],
                "material": True,
                "resolution_basis": "The runtime schema does not distinguish user intent.",
            }],
            "execution_decision": "clarify",
            "clarification_question": "Which population should I use?",
        },
        schema_fingerprint="schema-fingerprint",
        response_id="plan-response",
        usage_record={"usage_available": False, "purpose": "question_planning"},
        contract_valid=True,
        contract_error=None,
    )
    client = fake_client_factory(
        tool_response(1, "inspect_project", {}),
        final_response(2, "Which population should I use?"),
    )
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        max_agent_iterations=2,
        enable_question_planning=True,
        enable_local_python=False,
    )
    agent = BimAgent(settings=settings, client=client)
    agent.planner = _StaticPlanner(plan)

    report = agent.ask("Inspect the requested project population.")

    assert report.agent_loop["planning"]["contract_valid"] is True
    assert report.agent_loop["tools_available"] == ["inspect_project", "programmatic_tool_calling"]
    assert report.agent_loop["termination_reason"] == "clarification_requested"
    assert any(
        "Schema-aware preflight contract" in str(item.get("content") or "")
        for item in client.responses.requests[0]["input"]
        if isinstance(item, dict)
    )
