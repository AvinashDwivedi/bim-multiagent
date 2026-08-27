from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bim_agent import BimAgent
from bim_agent.dataset import DatasetError

from .adapter import evaluator_payload


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    client_id: str | None = None
    project_id: str | None = None
    request_id: str | None = None
    evaluation_run_id: str | None = None
    evaluation_case_index: int | None = None


app = FastAPI(title="Local BIM Agent", version="0.1.0")


def _project_directory(project_id: str | None = None) -> Path:
    configured = Path(os.getenv("BIM_DATA_DIR", "test-project-data")).resolve()
    projects_root = os.getenv("BIM_PROJECTS_ROOT")
    if not projects_root or not project_id:
        return configured
    root = Path(projects_root).resolve()
    candidate = (root / project_id).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise DatasetError(f"Project is not available under BIM_PROJECTS_ROOT: {project_id}")
    return candidate


def _fingerprint(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    files = [*data_dir.glob("*-tree.json"), *data_dir.glob("*-properties.json"), *data_dir.glob("*.ifc")]
    return tuple(sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files))


@lru_cache(maxsize=16)
def _cached_agent(data_dir: str, fingerprint: tuple[tuple[str, int, int], ...]) -> BimAgent:
    del fingerprint  # It is part of the cache key and intentionally forces reload on source changes.
    return BimAgent(data_dir)


def _agent(project_id: str | None = None) -> BimAgent:
    data_dir = _project_directory(project_id)
    return _cached_agent(str(data_dir), _fingerprint(data_dir))


_agent.cache_clear = _cached_agent.cache_clear  # type: ignore[attr-defined]


@app.get("/api/health")
def health() -> dict:
    try:
        agent = _agent()
        return {
            "status": "ok",
            "physical_instances": agent.profile["physical_instance_count"],
            "llm_planner_enabled": agent.settings.use_llm,
        }
    except DatasetError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    try:
        report = _agent(request.project_id).ask(request.question)
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return evaluator_payload(report, client_id=request.client_id, project_id=request.project_id)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def generate():
        yield json.dumps({"type": "stage", "stage": "started"}) + "\n"
        report = _agent(request.project_id).ask(request.question)
        payload = evaluator_payload(report, client_id=request.client_id, project_id=request.project_id)
        yield json.dumps({"type": "result", "report": payload}, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
