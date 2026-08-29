from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class TraceLog:
    def __init__(self, directory: Path, question: str):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_id = uuid4().hex
        self.path = directory / f"trace-{stamp}-{self.session_id[:10]}.jsonl"
        self._write("request", {"question": question})

    def event(self, stage: str, **payload: Any) -> None:
        self._write(stage, payload, full=False)

    def transcript(self, role: str, **payload: Any) -> None:
        """Persist an untrimmed, replayable transcript item with secret-key redaction."""
        self._write("transcript", {"role": role, **payload}, full=True)

    def _write(self, stage: str, payload: dict[str, Any], *, full: bool = False) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "stage": stage,
            **_safe(payload, full=full),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _safe(value: Any, *, full: bool = False) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if any(secret in str(key).lower() for secret in ("api_key", "password", "token")):
                output[str(key)] = "<redacted>"
            else:
                output[str(key)] = _safe(item, full=full)
        return output
    if isinstance(value, (list, tuple)):
        selected = value if full else value[:100]
        return [_safe(item, full=full) for item in selected]
    if not full and isinstance(value, str) and len(value) > 1000:
        return value[:1000] + "…"
    return value
