from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path


_DOTENV_MANAGED: dict[str, str] = {}
_DOTENV_LOCK = threading.Lock()


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries while preserving explicit process overrides.

    Values previously injected by this loader are refreshed when the file changes,
    which lets the cached web application adopt runtime configuration updates.
    """
    if not path.is_file():
        return
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            parsed[key] = value
    with _DOTENV_LOCK:
        for key in list(_DOTENV_MANAGED):
            if key not in parsed and os.environ.get(key) == _DOTENV_MANAGED[key]:
                os.environ.pop(key, None)
                _DOTENV_MANAGED.pop(key, None)
        for key, value in parsed.items():
            previous_managed = _DOTENV_MANAGED.get(key)
            if key not in os.environ or (
                previous_managed is not None and os.environ.get(key) == previous_managed
            ):
                os.environ[key] = value
                _DOTENV_MANAGED[key] = value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    trace_dir: Path
    model: str
    reasoning_effort: str
    max_agent_iterations: int = 20
    max_tool_output_chars: int = 80000
    enable_hosted_python: bool = False
    python_memory_limit: str = "4g"
    python_expiry_minutes: int = 120
    openai_max_retries: int = 4
    openai_timeout_seconds: float = 180.0

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: str | Path | None = None,
    ) -> "Settings":
        load_dotenv(Path.cwd() / ".env")
        selected_data = Path(data_dir or os.getenv("BIM_DATA_DIR", "test-project-data"))
        memory_limit = os.getenv("BIM_PYTHON_MEMORY_LIMIT", "4g").casefold()
        if memory_limit not in {"1g", "4g", "16g", "64g"}:
            raise ValueError("BIM_PYTHON_MEMORY_LIMIT must be one of: 1g, 4g, 16g, 64g.")
        return cls(
            data_dir=selected_data.resolve(),
            trace_dir=Path(os.getenv("BIM_TRACE_DIR", "logs/traces")).resolve(),
            model=os.getenv("BIM_MODEL", os.getenv("BIM_OPENAI_AGENT_MODEL", "gpt-5.4")),
            reasoning_effort=os.getenv("BIM_REASONING_EFFORT", "medium"),
            max_agent_iterations=max(1, int(os.getenv("BIM_MAX_AGENT_ITERATIONS", "20"))),
            max_tool_output_chars=max(2000, int(os.getenv("BIM_MAX_TOOL_OUTPUT_CHARS", "80000"))),
            enable_hosted_python=_env_bool("BIM_ENABLE_HOSTED_PYTHON", False),
            python_memory_limit=memory_limit,
            python_expiry_minutes=max(20, int(os.getenv("BIM_PYTHON_EXPIRY_MINUTES", "120"))),
            openai_max_retries=max(0, int(os.getenv("BIM_OPENAI_MAX_RETRIES", "4"))),
            openai_timeout_seconds=max(10.0, float(os.getenv("BIM_OPENAI_TIMEOUT_SECONDS", "180"))),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() not in {"0", "false", "no", "off"}
