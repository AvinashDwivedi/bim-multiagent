from __future__ import annotations

from pathlib import Path

import pytest

import bim_agent.config as config
from bim_agent.config import Settings


def test_dotenv_managed_values_refresh_without_overriding_explicit_environment(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BIM_MAX_AGENT_ITERATIONS", raising=False)
    dotenv = tmp_path / ".env"
    try:
        dotenv.write_text("BIM_MAX_AGENT_ITERATIONS=21\n", encoding="utf-8")
        assert Settings.from_env(data_dir=sample_data).max_agent_iterations == 21

        dotenv.write_text("BIM_MAX_AGENT_ITERATIONS=40\n", encoding="utf-8")
        assert Settings.from_env(data_dir=sample_data).max_agent_iterations == 40

        monkeypatch.setenv("BIM_MAX_AGENT_ITERATIONS", "55")
        assert Settings.from_env(data_dir=sample_data).max_agent_iterations == 55
    finally:
        config._DOTENV_MANAGED.pop("BIM_MAX_AGENT_ITERATIONS", None)


def test_model_pricing_can_be_overridden_from_environment(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "BIM_MODEL_PRICING_JSON",
        '{"private-model":{"input":1.5,"cached_input":0.2,"output":8}}',
    )
    rates = Settings.from_env(data_dir=sample_data).pricing_overrides["private-model"]
    assert rates == {"input": 1.5, "cached_input": 0.2, "output": 8.0}

    monkeypatch.setenv("BIM_MODEL_PRICING_JSON", "[]")
    with pytest.raises(ValueError, match="model-to-rates JSON object"):
        Settings.from_env(data_dir=sample_data)


def test_cost_guard_and_local_python_limits_are_loaded_safely(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIM_MAX_ANSWER_COST_USD", "0.42")
    monkeypatch.setenv("BIM_LOCAL_PYTHON_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("BIM_LOCAL_PYTHON_CPUS", "1.5")

    settings = Settings.from_env(data_dir=sample_data)

    assert settings.max_answer_cost_usd == 0.42
    assert settings.local_python_timeout_seconds == 75
    assert settings.local_python_cpus == 1.5


def test_question_planning_can_be_enabled_and_inherits_the_execution_model(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIM_ENABLE_QUESTION_PLANNING", "true")
    monkeypatch.setenv("BIM_MODEL", "schema-capable-model")
    monkeypatch.delenv("BIM_PLANNING_MODEL", raising=False)

    settings = Settings.from_env(data_dir=sample_data)

    assert settings.enable_question_planning is True
    assert settings.planning_model == "schema-capable-model"


def test_reasoning_effort_defaults_to_high(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BIM_REASONING_EFFORT", raising=False)

    settings = Settings.from_env(data_dir=sample_data)

    assert settings.reasoning_effort == "high"
    assert settings.planning_reasoning_effort == "medium"
    assert settings.model_tool_reasoning_effort == "medium"
    assert settings.finalization_reasoning_effort == "low"
