from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AnswerReport:
    answer: str
    status: str
    sources: list[dict[str, Any]]
    trace_path: str
    cost: dict[str, Any] = field(default_factory=dict)
    agent_loop: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
