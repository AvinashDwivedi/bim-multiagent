from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bim_agent import BimAgent
from bim_agent.config import Settings

from conftest import final_response, function_call, tool_response


def test_model_chooses_tools_and_owns_completion(
    sample_data: Path, tmp_path: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(
        tool_response(1, "inspect_project", {}),
        tool_response(2, "search_records", {
            "terms": ["Pipe ["], "match": "all", "limit": 100,
        }),
        tool_response(3, "aggregate_records", {
            "object_ids": ["43", "44"], "operation": "sum",
            "field": "Dimensions.Length", "output_unit": "m",
        }),
        final_response(4, "The total pipe length is 5.5 m."),
    )
    report = BimAgent(sample_data, client=client).ask("What is the total pipe length?")

    assert report.status == "completed"
    assert report.answer == "The total pipe length is 5.5 m."
    assert report.agent_loop["pattern"] == "model_directed_tool_loop"
    assert [
        call["tool"]
        for turn in report.agent_loop["iterations"]
        for call in turn.get("calls", [])
    ] == ["inspect_project", "search_records", "aggregate_records"]
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
        final_response(2, "The calculation result is 22."),
    )
    report = BimAgent(sample_data, client=client).ask("Calculate 21 + 1")
    assert report.answer.endswith("22.")
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
    report = agent.ask("Inspect the project")
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
    report = BimAgent(settings=settings, client=client).ask("Inspect this project")

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
        final_response(2, "5"),
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
        max_agent_iterations=4,
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
        final_response(2, "Check the Electrical Fixtures category as a plausible broader scope."),
        final_response(3, "There are 3 lighting switches; a broader definition may include another device."),
    )
    report = BimAgent(settings=settings, client=client).ask("How many switches?")

    assert report.status == "completed"
    assert report.agent_loop["iterations"][0]["calls"][0]["tool"] == "review_scope_and_evidence"
    nested_request = client.responses.requests[1]
    assert "critical BIM evidence-review tool" in nested_request["instructions"]
    model_observation = client.responses.requests[2]["input"][-1]["output"]
    assert "Electrical Fixtures" in model_observation
