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
        self.path = directory / f"trace-{stamp}-{uuid4().hex[:10]}.jsonl"
        self._write("request", {"question": question})

    def event(self, stage: str, **payload: Any) -> None:
        self._write(stage, payload)

    def _write(self, stage: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **_safe(payload),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if any(secret in str(key).lower() for secret in ("api_key", "password", "token")):
                output[str(key)] = "<redacted>"
            else:
                output[str(key)] = _safe(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value[:100]]
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + "…"
    return value
