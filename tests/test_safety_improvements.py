from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bim_agent import BimAgent
from bim_agent.agent_loop import (
    COST_BUDGET_NOTICE,
    _classify_route,
    _disclosure_present,
    _ensure_review_disclosures,
    _grounding_issues,
    _model_tool_result,
    _needs_reconciliation,
    _population_ids,
    _repair_citation_placement,
    _requires_population_reconciliation,
    _review_completion_issues,
    _tool_cache_key,
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

    updated, disclosure_state = _ensure_review_disclosures(
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
        disclosure_state=disclosure_state,
        unsupported_claims=[],
    )

    assert updated.count(disclosure) == 1
    assert disclosure_state["selected_scope_only"]["citation_valid"] is True
    assert disclosure_state["selected_scope_only"]["references"] == ["call_1", "call_2"]
    assert disclosure_state["loaded_snapshot"]["citation_valid"] is True
    assert issues == []


def test_multiple_model_authored_disclosures_are_replaced_by_separately_cited_canonical_sentences() -> None:
    snapshot = _review_disclosure_text("loaded_snapshot", False)
    uniqueness = _review_disclosure_text("record_count_not_physical_uniqueness", False)
    answer = f"There are 2 pipe records. [ref: call_1]\n\n{snapshot} {uniqueness} [ref: call_2]"
    evidence = {
        "project": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"count":2}]}',
            "evidence_class": "project",
        },
        "review": {
            "tool": "review_scope_and_evidence",
            "output": '{"required_disclosures":["loaded_snapshot","record_count_not_physical_uniqueness"]}',
            "evidence_class": "model_review",
        },
    }

    updated, state = _ensure_review_disclosures(
        "How many pipes?",
        answer,
        evidence,
        evidence_aliases={"call_1": "project", "call_2": "review"},
        disclosure_codes=["loaded_snapshot", "record_count_not_physical_uniqueness"],
    )

    assert updated.count(snapshot) == 1
    assert updated.count(uniqueness) == 1
    assert f"{snapshot} [ref: call_1] [ref: call_2]" in updated
    assert f"{uniqueness} [ref: call_1] [ref: call_2]" in updated
    assert all(item["citation_valid"] for item in state.values())


def test_disclosure_combined_with_factual_sentence_keeps_fact_and_rebuilds_disclosure() -> None:
    snapshot = _review_disclosure_text("loaded_snapshot", False)
    answer = f"{snapshot} There are 2 pipe records. [ref: call_1]"
    evidence = {
        "project": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"count":2}]}',
            "evidence_class": "project",
        },
    }

    updated, state = _ensure_review_disclosures(
        "How many pipes?",
        answer,
        evidence,
        evidence_aliases={"call_1": "project"},
        disclosure_codes=[],
    )

    assert "There are 2 pipe records. [ref: call_1]" in updated
    assert updated.count(snapshot) == 1
    assert f"{snapshot} [ref: call_1]" in updated
    assert state["loaded_snapshot"]["citation_valid"] is True


def test_semantic_uncited_disclosure_is_replaced_with_canonical_citations() -> None:
    answer = (
        "There are 2 pipe records. [ref: call_1]\n\n"
        "This result covers only the selected scope and excludes objects outside it."
    )
    evidence = {
        "project": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"count":2}]}',
            "evidence_class": "project",
        },
        "review": {
            "tool": "review_scope_and_evidence",
            "output": '{"required_disclosures":["selected_scope_only"]}',
            "evidence_class": "model_review",
        },
    }

    updated, state = _ensure_review_disclosures(
        "How many pipes are in the selected scope?",
        answer,
        evidence,
        evidence_aliases={"call_1": "project", "call_2": "review"},
        disclosure_codes=["selected_scope_only"],
    )

    canonical = _review_disclosure_text("selected_scope_only", False)
    assert "This result covers only" not in updated
    assert f"{canonical} [ref: call_1] [ref: call_2]" in updated
    assert state["selected_scope_only"]["citation_valid"] is True


def test_safe_citation_placement_repair_reuses_only_supporting_trailing_refs() -> None:
    evidence = {
        "project": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"pipe_count":2,"population":"pipes"}]}',
            "evidence_class": "project",
        },
    }
    repaired, repairs = _repair_citation_placement(
        "How many pipes?",
        "There are 2 pipes. The pipe population count is 2. [ref: call_1]",
        evidence,
        evidence_aliases={"call_1": "project"},
    )
    not_repaired, rejected_repairs = _repair_citation_placement(
        "How many pipes?",
        "There are 9 pipes. The pipe population count is 2. [ref: call_1]",
        evidence,
        evidence_aliases={"call_1": "project"},
    )

    assert "There are 2 pipes. [ref: call_1]" in repaired
    assert len(repairs) == 1
    assert "There are 9 pipes. [ref: call_1]" not in not_repaired
    assert rejected_repairs == []


def test_sentence_grounding_does_not_share_references_across_adjacent_claims() -> None:
    evidence = {
        "pipes": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"pipe_count":2}]}',
            "evidence_class": "project",
        },
        "ducts": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"duct_count":99}]}',
            "evidence_class": "project",
        },
    }

    correct = _grounding_issues(
        "How many pipes and ducts are in the project?",
        "There are 2 pipes. [ref: pipes] There are 99 ducts. [ref: ducts]",
        evidence,
        require_disclaimer=False,
    )
    crossed = _grounding_issues(
        "How many pipes and ducts are in the project?",
        "There are 2 pipes. [ref: ducts] There are 99 ducts. [ref: pipes]",
        evidence,
        require_disclaimer=False,
    )

    assert correct == []
    assert len(crossed) == 2
    assert all("do not occur" in issue for issue in crossed)


def test_plain_markdown_heading_is_scaffolding_but_factual_heading_is_grounded() -> None:
    evidence = {
        "project": {
            "tool": "query_bim_workspace",
            "output": '{"rows":[{"count":2}]}',
            "evidence_class": "project",
        },
    }

    issues = _grounding_issues(
        "How many project records?",
        "## Result\n\nThere are 2 records. [ref: project]",
        evidence,
        require_disclaimer=False,
    )
    factual_heading_issues = _grounding_issues(
        "How many project records?",
        "## Data quality: low",
        evidence,
        require_disclaimer=False,
    )

    assert issues == []
    assert any("no inline tool reference" in issue for issue in factual_heading_issues)


def test_hebrew_per_floor_aggregate_routes_to_sql_and_requires_reconciliation() -> None:
    question = "מהי צפיפות ההספק התאורטי (W/m²) לכל קומה, ומה איכות נתוני הפוטומטריה?"

    route = _classify_route(question)

    assert route["project_data_operation"] is True
    assert route["general_compute_path"] == "sql_only"
    assert route["calculate_exposed"] is False
    assert route["local_python_exposed"] is False
    assert _requires_population_reconciliation(question) is True


def test_sql_cache_key_ignores_only_a_terminal_semicolon() -> None:
    without_semicolon = _tool_cache_key("query_bim_workspace", {
        "sql": "SELECT record_object_id FROM record_ifc_candidates ORDER BY record_object_id",
        "parameters": [],
        "row_limit": 200,
    })
    with_semicolon = _tool_cache_key("query_bim_workspace", {
        "sql": " SELECT record_object_id FROM record_ifc_candidates ORDER BY record_object_id; ",
        "parameters": [],
        "row_limit": 200,
    })

    assert without_semicolon == with_semicolon


def test_complete_identity_query_reconciles_all_rows_before_model_row_compaction(sample_data: Path) -> None:
    object_ids = [str(index) for index in range(1, 111)]
    result = {
        "columns": ["record_object_id"],
        "rows": [{"record_object_id": object_id} for object_id in object_ids],
        "returned_rows": 110,
        "truncated": False,
    }

    assert _needs_reconciliation("query_bim_workspace", result) is True
    assert _population_ids(
        "query_bim_workspace",
        {"sql": "SELECT record_object_id FROM selected", "parameters": [], "row_limit": 200},
        result,
        RawProjectTools(sample_data),
    ) == object_ids
    assert len(_model_tool_result("query_bim_workspace", result)["rows"]) == 30


def test_malformed_structured_review_fails_closed(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(final_response(1, '{"can_finalize":true,"summary":"truncated'))
    agent = BimAgent(sample_data, client=client)

    result = agent.agent.model_tools.execute("review_scope_and_evidence", {
        "question": "How many pipes?",
        "selected_scope": "Pipes",
        "exclusions": "Definitions",
        "evidence": "Two records",
        "reconciliation": "Two unique IDs",
        "draft_answer": "Two pipes",
    })

    assert result["can_finalize"] is False
    assert result["review_parse_valid"] is False
    assert result["required_follow_up"]
    assert "Invalid or truncated" in result["review"]
    assert agent.agent.model_tools._last_review_result is None


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


def test_content_and_formatting_grounding_retries_have_independent_budgets(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        final_response(2, "There are 99 pipes. [ref: call-1]"),
        final_response(3, "There are 2 pipes."),
        final_response(4, "There are 2 pipes. [ref: call-1]"),
    )

    report = BimAgent(sample_data, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert report.agent_loop["grounding_content_forced"] is True
    assert report.agent_loop["grounding_format_forced"] is True
    retries = [
        item["grounding_issue_category"]
        for item in report.agent_loop["iterations"]
        if item.get("action") == "grounding_continue"
    ]
    assert retries == ["content", "formatting"]


def test_sentence_level_citation_repair_avoids_a_model_retry(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        final_response(
            2,
            "There are 2 pipes. The selected query returned a count of 2. [ref: call-1]",
        ),
    )

    report = BimAgent(sample_data, client=client).ask("How many pipes?")

    assert report.status == "completed"
    assert "There are 2 pipes. [ref: call-1]" in report.answer
    assert report.agent_loop["grounding_forced"] is False
    assert len(report.agent_loop["citation_repairs"]) == 1


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
    sample_data: Path, tmp_path: Path, fake_client_factory, monkeypatch,
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=4,
        max_answer_cost_usd=0.001,
    )
    client = fake_client_factory(
        _with_usage(
            tool_response(1, "calculate", {"expression": "2 + 3"}),
            input_tokens=1000,
            output_tokens=100,
        ),
        _with_usage(
            final_response(2, "The available evidence does not yet establish the result."),
            input_tokens=200,
            output_tokens=40,
        ),
    )
    agent = BimAgent(settings=settings, client=client)
    executions: list[str] = []
    original_execute = agent.tools.execute

    def tracked_execute(name: str, arguments: dict) -> dict:
        executions.append(name)
        return original_execute(name, arguments)

    monkeypatch.setattr(agent.tools, "execute", tracked_execute)
    report = agent.ask("Calculate 2 + 3")

    assert report.status == "limited"
    assert report.agent_loop["termination_reason"] == "cost_budget_exceeded"
    assert report.cost["budget_usd"] == 0.001
    assert report.cost["budget_exceeded"] is True
    assert report.agent_loop["iterations"][0]["action"] == "cost_budget_answer"
    assert report.agent_loop["iterations"][0]["budget_finalization"]["trigger"] == "before_tool_execution"
    assert executions == []
    assert "tools" not in client.responses.requests[1]
    assert report.answer.endswith(COST_BUDGET_NOTICE)
    assert report.answer.count(COST_BUDGET_NOTICE) == 1
    assert [item["purpose"] for item in report.cost["request_breakdown"]] == [
        "agent_turn", "budget_finalization",
    ]


def test_final_candidate_that_crosses_budget_is_preserved_without_an_extra_request(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    settings = Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.4",
        reasoning_effort="medium",
        max_agent_iterations=2,
        max_answer_cost_usd=0.001,
    )
    client = fake_client_factory(_with_usage(
        final_response(1, f"Best available answer.\n\n{COST_BUDGET_NOTICE}"),
        input_tokens=1000,
        output_tokens=100,
    ))

    report = BimAgent(settings=settings, client=client).ask("Explain the available information")

    assert report.status == "limited"
    assert len(client.responses.requests) == 1
    assert report.answer.startswith("Best available answer.")
    assert report.answer.endswith(COST_BUDGET_NOTICE)
    assert report.answer.count(COST_BUDGET_NOTICE) == 1
    metadata = report.agent_loop["iterations"][0]["budget_finalization"]
    assert metadata["trigger"] == "after_answer_generation"
    assert metadata["finalization_attempted"] is False


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
