from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 4_000 else value[:4_000] + "...[truncated]"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in list(value.items())[:80]}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value[:80]]
    return _safe(str(value))


class AgentTraceLog:
    """Request-scoped JSONL trace safe for concurrent async requests."""

    def __init__(self, *, enabled: bool, directory: Path) -> None:
        self.enabled = enabled
        self.directory = directory
        self._path: ContextVar[Path | None] = ContextVar("bim_trace_path", default=None)
        self._request_id: ContextVar[str | None] = ContextVar("bim_trace_request_id", default=None)
        self._lock = threading.Lock()

    def start(self, **fields: Any) -> str:
        request_id = uuid4().hex
        if not self.enabled:
            return request_id
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path.set(self.directory / f"trace-{timestamp}-{request_id}.jsonl")
        self._request_id.set(request_id)
        self.log("request_start", **fields)
        return request_id

    def log(self, event: str, **fields: Any) -> None:
        path = self._path.get()
        if not self.enabled or path is None:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": self._request_id.get(),
            "event": event,
            **{key: _safe(value) for key, value in fields.items()},
        }
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            # Observability must never make the BIM answer pipeline fail.
            return

    def finish(self, **fields: Any) -> None:
        self.log("request_end", **fields)

    @property
    def active_path(self) -> Path | None:
        return self._path.get()
