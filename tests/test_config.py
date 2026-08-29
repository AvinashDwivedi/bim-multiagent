from __future__ import annotations

from pathlib import Path

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
