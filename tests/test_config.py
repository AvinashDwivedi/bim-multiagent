from __future__ import annotations

from pathlib import Path

import pytest

from bim_agent.config import Settings


def test_builtin_runtime_settings_are_loaded(sample_data: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_MAX_AGENT_ITERATIONS", "21")
    monkeypatch.setenv("BIM_MAX_TOOL_OUTPUT_CHARS", "9000")
    monkeypatch.setenv("BIM_OPENAI_MAX_RETRIES", "2")
    monkeypatch.setenv("BIM_OPENAI_TIMEOUT_SECONDS", "75")

    settings = Settings.from_env(data_dir=sample_data)

    assert settings.data_dir == sample_data.resolve()
    assert settings.max_agent_iterations == 21
    assert settings.max_tool_output_chars == 9000
    assert settings.openai_max_retries == 2
    assert settings.openai_timeout_seconds == 75


def test_pricing_override_requires_an_object(sample_data: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIM_MODEL_PRICING_JSON", "[]")

    with pytest.raises(ValueError, match="top-level value must be an object"):
        Settings.from_env(data_dir=sample_data)
