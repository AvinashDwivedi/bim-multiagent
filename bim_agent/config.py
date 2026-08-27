from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    trace_dir: Path
    model: str
    reasoning_effort: str
    use_llm: bool
    max_profile_values: int = 160

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: str | Path | None = None,
        use_llm: bool | None = None,
    ) -> "Settings":
        load_dotenv(Path.cwd() / ".env")
        selected_data = Path(data_dir or os.getenv("BIM_DATA_DIR", "test-project-data"))
        selected_llm = _boolean("BIM_USE_LLM", True) if use_llm is None else use_llm
        return cls(
            data_dir=selected_data.resolve(),
            trace_dir=Path(os.getenv("BIM_TRACE_DIR", "logs/traces")).resolve(),
            model=os.getenv("BIM_MODEL", os.getenv("BIM_OPENAI_AGENT_MODEL", "gpt-5.4")),
            reasoning_effort=os.getenv("BIM_REASONING_EFFORT", "medium"),
            use_llm=selected_llm and bool(os.getenv("OPENAI_API_KEY")),
            max_profile_values=max(40, int(os.getenv("BIM_MAX_PROFILE_VALUES", "160"))),
        )

