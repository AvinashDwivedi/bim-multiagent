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


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    trace_dir: Path
    model: str
    reasoning_effort: str
    max_agent_iterations: int = 20
    max_tool_output_chars: int = 80000

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: str | Path | None = None,
    ) -> "Settings":
        load_dotenv(Path.cwd() / ".env")
        selected_data = Path(data_dir or os.getenv("BIM_DATA_DIR", "test-project-data"))
        return cls(
            data_dir=selected_data.resolve(),
            trace_dir=Path(os.getenv("BIM_TRACE_DIR", "logs/traces")).resolve(),
            model=os.getenv("BIM_MODEL", os.getenv("BIM_OPENAI_AGENT_MODEL", "gpt-5.4")),
            reasoning_effort=os.getenv("BIM_REASONING_EFFORT", "medium"),
            max_agent_iterations=max(1, int(os.getenv("BIM_MAX_AGENT_ITERATIONS", "20"))),
            max_tool_output_chars=max(2000, int(os.getenv("BIM_MAX_TOOL_OUTPUT_CHARS", "80000"))),
        )
