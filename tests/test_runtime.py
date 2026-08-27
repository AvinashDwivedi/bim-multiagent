from __future__ import annotations

from pathlib import Path

from bim_agent import BimAgent


def test_agent_counts_primary_switch_population_and_audits_related(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    agent = BimAgent(sample_data, use_llm=False)
    report = agent.ask("how many switches are in the project?")
    assert report.status == "verified"
    assert report.evidence.distinct_identity_count == 3
    assert report.plan.categories == ["Lighting Devices"]
    assert sum(group["count"] for group in report.evidence.related_groups) == 3
    assert Path(report.trace_path).is_file()


def test_agent_lists_types(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    report = BimAgent(sample_data, use_llm=False).ask("what types of switches are there?")
    assert report.plan.intent == "list"
    assert {(group["type_name"], group["count"]) for group in report.evidence.groups} == {
        ("Single", 2), ("Double", 1)
    }


def test_agent_sums_numeric_property(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    report = BimAgent(sample_data, use_llm=False).ask("what is the total length of pipes?")
    assert report.status == "verified"
    assert report.evidence.operation_value == 5.5
    assert report.evidence.unit == "m"


def test_hebrew_question_uses_cross_language_vocabulary(sample_data: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_TRACE_DIR", str(tmp_path / "traces"))
    report = BimAgent(sample_data, use_llm=False).ask("כמה מפסקים בפרויקט?")
    assert report.evidence.distinct_identity_count == 3
    assert "3" in report.answer

