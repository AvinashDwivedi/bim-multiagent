from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bim_agent import BimAgent
from bim_agent.agent_loop import (
    _disclosure_present,
    _ensure_review_disclosures,
    _grounding_issues,
    _model_tool_result,
    _review_completion_issues,
)
from bim_agent.model_tools import _review_disclosure_text
from bim_agent.config import Settings
from bim_agent.model_tools import UNVERIFIED_STANDARDS_DISCLAIMER
from bim_agent.project_tools import RawProjectTools

from conftest import final_response, function_call, tool_response


def _with_usage(
    response: SimpleNamespace,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    model: str = "gpt-5.4",
) -> SimpleNamespace:
    response.model = model
    response.usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return response


def test_pageable_tools_return_cursor_and_fetch_more(sample_data: Path) -> None:
    tools = RawProjectTools(sample_data)
    first = tools.execute("search_records", {"terms": [], "limit": 2})

    assert first["total_count"] == 22
    assert first["returned_count"] == 2
    assert first["cursor"]
    assert first["records"] == first["results"]

    second = tools.execute("fetch_more", {"cursor": first["cursor"]})
    assert second["offset"] == 2
    assert second["returned_count"] == 2
    assert {row["object_id"] for row in first["results"]}.isdisjoint(
        row["object_id"] for row in second["results"]
    )


def test_model_observations_compact_large_inventories_without_changing_trace_results() -> None:
    inspection = _model_tool_result("inspect_project", {
        "source_files": [{"kind": "tree", "path": "C:/private/project/model-tree.json"}],
        "property_keys": [{"key": f"Key {index}", "record_count": 1} for index in range(100)],
    })
    query = _model_tool_result("query_bim_workspace", {
        "columns": ["object_id"],
        "rows": [{"object_id": str(index)} for index in range(100)],
        "returned_rows": 100,
        "truncated": False,
    })

    assert len(inspection["property_keys"]) == 40
    assert inspection["property_keys_omitted"] == 60
    assert inspection["source_files"][0]["file_name"] == "model-tree.json"
    assert "path" not in inspection["source_files"][0]
    assert len(query["rows"]) == 30
    assert query["model_rows_omitted"] == 70
    assert query["returned_rows"] == 100


def test_semantic_review_disclosures_do_not_get_duplicated() -> None:
    answer = (
        "בצילום הפרויקט שנטען התוצאה חלה על Cable Trays בלבד ואינה כוללת אביזרים. "
        "ספירת הרשומות אינה בהכרח מוכיחה ייחודיות פיזית בין המקורות; "
        "עדכניות המקורות והתאמת הגרסאות בין הקבצים לא אומתו."
    )

    assert _disclosure_present("loaded_snapshot", answer)
    assert _disclosure_present("record_count_not_physical_uniqueness", answer)
    assert _disclosure_present("selected_scope_only", answer)


def test_standard_hebrew_scope_disclosure_satisfies_code_gate_without_duplication() -> None:
    disclosure = _review_disclosure_text("selected_scope_only", True)
    answer = f"נמצאו שני סוגים. [ref: call_1]\n\n{disclosure} [ref: call_1] [ref: call_2]"
    evidence = {
        "project": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"type_count":2}]}',
            "evidence_class": "project",
        },
        "review": {
            "tool": "review_scope_and_evidence",
            "output": '{"required_disclosures":["selected_scope_only"]}',
            "evidence_class": "model_review",
        },
    }

    updated, satisfied = _ensure_review_disclosures(
        "אילו סוגים קיימים?",
        answer,
        evidence,
        evidence_aliases={"call_1": "project", "call_2": "review"},
        disclosure_codes=["selected_scope_only"],
    )
    issues = _review_completion_issues(
        updated,
        review_seen=True,
        review_can_finalize=True,
        review_stale=False,
        required_follow_up=[],
        required_disclosures=["selected_scope_only"],
        satisfied_disclosures=satisfied,
        unsupported_claims=[],
    )

    assert updated == answer
    assert satisfied == {"selected_scope_only"}
    assert issues == []


def test_non_english_project_claim_cannot_rely_only_on_model_review() -> None:
    issues = _grounding_issues(
        "אילו סוגי מגשים קיימים?",
        "קיימים שני סוגי מגשים. [ref: review_1]",
        {
            "project_1": {
                "tool": "query_bim_workspace",
                "output": '{"rows":[{"type_count":2}]}',
                "evidence_class": "project",
            },
            "review_1": {
                "tool": "review_scope_and_evidence",
                "output": '{"review":"קיימים שני סוגי מגשים"}',
                "evidence_class": "model_review",
            },
        },
        require_disclaimer=False,
    )

    assert any("non-project evidence" in issue for issue in issues)


def test_duplicate_tool_calls_are_executed_once_per_run(
    sample_data: Path, fake_client_factory, monkeypatch,
) -> None:
    client = fake_client_factory(
        tool_response(1, "calculate", {"expression": "21 + 1"}),
        tool_response(2, "calculate", {"expression": "21 + 1"}),
        final_response(3, "The result is 22. [ref: call-2]"),
    )
    agent = BimAgent(sample_data, client=client)
    original = agent.tools.execute
    executions: list[str] = []

    def counted(name: str, arguments: dict) -> dict:
        executions.append(name)
        return original(name, arguments)

    monkeypatch.setattr(agent.tools, "execute", counted)
    report = agent.ask("Calculate 21 + 1")

    assert executions.count("calculate") == 1
    assert report.agent_loop["iterations"][1]["calls"][0]["cached"] is True


def test_grounding_gate_rejects_a_cited_value_absent_from_tool_output(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "calculate", {"expression": "2 + 3"}),
        final_response(2, "The result is 99. [ref: call-1]"),
        final_response(3, "The result is 5. [ref: call-1]"),
    )
    report = BimAgent(sample_data, client=client).ask("Calculate 2 + 3")

    assert report.status == "completed"
    assert report.answer.startswith("The result is 5")
    assert report.agent_loop["grounding_forced"] is True
    assert report.agent_loop["iterations"][1]["action"] == "grounding_continue"
    assert "99" in client.responses.requests[2]["input"][-1]["content"]


def test_grounding_gate_withholds_answer_after_failed_retry(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "calculate", {"expression": "2 + 3"}),
        final_response(2, "The result is 99. [ref: call-1]"),
        final_response(3, "The result is still 99. [ref: call-1]"),
    )
    report = BimAgent(sample_data, client=client).ask("Calculate 2 + 3")

    assert report.status == "limited"
    assert report.agent_loop["termination_reason"] == "grounding_rejected"
    assert report.answer.startswith("The answer was withheld")


def test_completeness_gate_forces_project_data_verification(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        final_response(1, "There are 2 pipes."),
        tool_response(2, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(3, "reconcile_populations", {
            "populations": [{"label": "pipes", "object_ids": ["43", "44"]}],
        }),
        final_response(4, "There are 2 pipes. [ref: call-2] [ref: call-3]"),
    )
    report = BimAgent(sample_data, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert report.agent_loop["completeness_forced"] is True
    assert report.agent_loop["iterations"][0]["action"] == "completeness_continue"


def test_data_observation_automatically_injects_population_reconciliation(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "search_records", {"terms": ["Pipe"], "limit": 20}),
        final_response(2, "Pipe records were found. [ref: call-1]"),
    )
    report = BimAgent(sample_data, client=client).ask("Find all pipe records")

    calls = report.agent_loop["iterations"][0]["calls"]
    automatic = next(call for call in calls if call.get("automatic"))
    assert automatic["tool"] == "reconcile_populations"
    assert automatic["call_id"] == "auto_reconcile_call-1"
    assert "Automatic population reconciliation observation" in str(
        client.responses.requests[1]["input"][-1]["content"]
    )
    population_ids = automatic["arguments"]["populations"][0]["object_ids"]
    assert len(population_ids) == len(set(population_ids))
    assert automatic["mismatch"] is False
    model_output = next(
        item["output"] for item in client.responses.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert "records" not in json.loads(model_output)


def test_aggregate_question_hides_redundant_compute_tools(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(2, "reconcile_populations", {
            "populations": [{"label": "pipes", "object_ids": ["43", "44"]}],
        }),
        final_response(3, "There are 2 pipes. [ref: call-1] [ref: call-2]"),
    )
    BimAgent(sample_data, client=client).ask("How many pipe objects are in the project?")
    names = {item.get("name") for item in client.responses.requests[0]["tools"]}

    assert "query_bim_workspace" in names
    assert "aggregate_records" not in names
    assert "calculate" not in names
    assert "run_local_python" not in names


def test_correctly_grounded_routine_sql_count_passes_without_reconciliation(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        final_response(2, "There are 2 pipes. [ref: call-1]"),
    )

    report = BimAgent(sample_data, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert report.agent_loop["grounding_forced"] is False
    assert report.agent_loop["completeness_forced"] is False
    assert not any(
        call["tool"] == "reconcile_populations"
        for iteration in report.agent_loop["iterations"]
        for call in iteration.get("calls", [])
    )


def test_ordinal_citation_alias_resolves_to_opaque_api_call_id(
    sample_data: Path, fake_client_factory,
) -> None:
    call = function_call("query_bim_workspace", {
        "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
        "parameters": ["Pipe [%"], "row_limit": 20,
    })
    call.call_id = "call_opaque_7f4ab9"
    client = fake_client_factory(
        SimpleNamespace(id="resp-1", output=[call], output_text=""),
        final_response(2, "There are 2 pipes. [ref: call_1]"),
    )

    report = BimAgent(sample_data, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert report.agent_loop["evidence_aliases"] == {"call_1": "call_opaque_7f4ab9"}
    assert report.agent_loop["iterations"][0]["calls"][0]["citation_alias"] == "call_1"
    observation = next(
        item["output"] for item in client.responses.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert json.loads(observation)["_citation_reference"] == "call_1"


def test_hebrew_number_word_matches_numeric_cited_value(
    sample_data: Path, fake_client_factory,
) -> None:
    call = function_call("query_bim_workspace", {
        "sql": "SELECT 'דו-קוטבי' AS family, 1 AS instance_count",
        "parameters": [],
        "row_limit": 20,
    })
    call.call_id = "call_opaque_hebrew"
    client = fake_client_factory(
        SimpleNamespace(id="resp-1", output=[call], output_text=""),
        final_response(2, "מפסק דו־קוטבי אחד. [ref: call_1]"),
    )

    report = BimAgent(sample_data, client=client).ask("בדיקת מפסק")

    assert report.status == "completed"
    assert report.agent_loop["grounding_forced"] is False


def test_unsourced_standards_research_propagates_mandatory_disclaimer(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        _with_usage(
            tool_response(1, "research_standards", {
                "query": "clearance requirement", "jurisdiction": "unknown", "standard_hint": "",
            }),
            input_tokens=100, output_tokens=20,
        ),
        _with_usage(
            final_response(2, "The requirement may be 10 m."),
            input_tokens=200, output_tokens=40,
        ),
        _with_usage(
            final_response(
                3,
                f"{UNVERIFIED_STANDARDS_DISCLAIMER}\nThe requirement may be 10 m. [ref: call-1]",
            ),
            input_tokens=300, output_tokens=60,
        ),
    )
    report = BimAgent(sample_data, client=client).ask("What standard clearance applies?")

    assert report.status == "completed"
    assert report.answer.startswith(UNVERIFIED_STANDARDS_DISCLAIMER)
    assert report.cost["api_requests"] == 3
    assert report.cost["excluded_tool_fees"] == ["web_search"]
    assert report.cost["is_complete"] is False
    observation = next(
        item["output"] for item in client.responses.requests[2]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert json.loads(observation)["verified"] is False


def test_trace_persists_full_tool_output_for_replay(
    sample_data: Path, tmp_path: Path, fake_client_factory, monkeypatch,
) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    client = fake_client_factory(
        tool_response(1, "inspect_project", {}),
        final_response(2, "The project contains 22 records. [ref: call-1]"),
    )
    report = BimAgent(sample_data, client=client).ask("Inspect the project")
    events = [
        json.loads(line)
        for line in Path(report.trace_path).read_text(encoding="utf-8").splitlines()
    ]
    tool_item = next(
        item for item in events
        if item.get("stage") == "transcript" and item.get("tool") == "inspect_project"
    )

    assert tool_item["output"]["record_count"] == 22
    assert len(tool_item["output"]["property_keys"]) > 1


def test_answer_report_calculates_total_cost_from_all_response_usage(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=4,
    )
    client = fake_client_factory(
        _with_usage(
            tool_response(1, "calculate", {"expression": "21 + 1"}),
            input_tokens=1000, cached_tokens=200, output_tokens=100,
        ),
        _with_usage(
            final_response(2, "The result is 22. [ref: call-1]"),
            input_tokens=2000, cached_tokens=500, output_tokens=200,
        ),
    )
    report = BimAgent(settings=settings, client=client).ask("Calculate 21 + 1")

    assert report.cost["status"] == "calculated"
    assert report.cost["is_complete"] is True
    assert report.cost["api_requests"] == 2
    assert report.cost["tokens"] == {
        "input_tokens": 3000,
        "cached_input_tokens": 700,
        "cache_write_input_tokens": 0,
        "output_tokens": 300,
        "total_tokens": 3300,
    }
    assert report.cost["estimated_cost_usd"] == 0.010425
    events = [
        json.loads(line)
        for line in Path(report.trace_path).read_text(encoding="utf-8").splitlines()
    ]
    assert any(item.get("role") == "cost" for item in events if item.get("stage") == "transcript")


def test_cost_guard_stops_before_another_tool_execution(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=4,
        max_answer_cost_usd=0.001,
    )
    client = fake_client_factory(_with_usage(
        tool_response(1, "calculate", {"expression": "2 + 3"}),
        input_tokens=1000,
        output_tokens=100,
    ))

    report = BimAgent(settings=settings, client=client).ask("Calculate 2 + 3")

    assert report.status == "limited"
    assert report.agent_loop["termination_reason"] == "cost_budget_exceeded"
    assert report.cost["budget_usd"] == 0.001
    assert report.cost["budget_exceeded"] is True
    assert report.agent_loop["iterations"][0]["action"] == "cost_budget_exceeded"


def test_markdown_table_headers_do_not_trigger_a_grounding_retry(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "calculate", {"expression": "2 + 3"}),
        final_response(2, "| Item | Value |\n| --- | ---: |\n| Sum | 5 [ref: call-1] |"),
    )

    report = BimAgent(sample_data, client=client).ask("Calculate 2 + 3")

    assert report.status == "completed"
    assert report.agent_loop["grounding_forced"] is False


def test_one_clarifying_question_is_returned_instead_of_guessing_scope(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(final_response(1, "Which room do you mean?"))

    report = BimAgent(sample_data, client=client).ask("What is in the big room?")

    assert report.status == "completed"
    assert report.answer == "Which room do you mean?"
    assert report.agent_loop["termination_reason"] == "clarification_requested"
