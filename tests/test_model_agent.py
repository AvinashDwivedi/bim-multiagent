from __future__ import annotations

import json
from pathlib import Path

from bim_agent import BimAgent

from conftest import final_response, tool_response


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


def test_trace_contains_tool_telemetry_not_private_reasoning(
    sample_data: Path, tmp_path: Path, monkeypatch, fake_client_factory
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(final_response(1, "No tools were needed."))
    report = BimAgent(sample_data, client=client).ask("Say whether tools are needed")
    events = [json.loads(line) for line in Path(report.trace_path).read_text(encoding="utf-8").splitlines()]
    assert [item["stage"] for item in events][-1] == "model_agent_finish"
    assert all("thought" not in item and "reasoning" not in item for item in events)
