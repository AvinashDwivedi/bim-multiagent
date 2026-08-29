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


def test_cost_guard_and_container_expiry_are_loaded_safely(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIM_MAX_ANSWER_COST_USD", "0.42")
    monkeypatch.setenv("BIM_PYTHON_EXPIRY_MINUTES", "120")

    settings = Settings.from_env(data_dir=sample_data)

    assert settings.max_answer_cost_usd == 0.42
    assert settings.python_expiry_minutes == 20
