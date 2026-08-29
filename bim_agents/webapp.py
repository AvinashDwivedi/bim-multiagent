from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bim_agent import BimAgent
from bim_agent.agent_loop import _is_retryable_api_error
from bim_agent.project_tools import ProjectError

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
        raise ProjectError(f"Project is not available under BIM_PROJECTS_ROOT: {project_id}")
    return candidate


def _fingerprint(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    files = [*data_dir.glob("*-tree.json"), *data_dir.glob("*-properties.json"), *data_dir.glob("*.ifc")]
    return tuple(sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files))


def _configuration_fingerprint() -> tuple[tuple[str, str], ...]:
    names = (
        "BIM_MODEL", "BIM_OPENAI_AGENT_MODEL", "BIM_REASONING_EFFORT", "BIM_MAX_AGENT_ITERATIONS",
        "BIM_MAX_TOOL_OUTPUT_CHARS", "BIM_ENABLE_LOCAL_PYTHON", "BIM_LOCAL_PYTHON_IMAGE",
        "BIM_PYTHON_MEMORY_LIMIT", "BIM_LOCAL_PYTHON_CPUS", "BIM_LOCAL_PYTHON_TIMEOUT_SECONDS",
        "BIM_LOCAL_PYTHON_OUTPUT_CHARS", "BIM_OPENAI_MAX_RETRIES", "BIM_OPENAI_TIMEOUT_SECONDS",
    )
    values = [(name, os.getenv(name, "")) for name in names]
    dotenv = Path.cwd() / ".env"
    if dotenv.is_file():
        stat = dotenv.stat()
        values.append((".env", f"{stat.st_size}:{stat.st_mtime_ns}"))
    return tuple(values)


@lru_cache(maxsize=16)
def _cached_agent(
    data_dir: str,
    fingerprint: tuple[tuple[str, int, int], ...],
    configuration_fingerprint: tuple[tuple[str, str], ...],
) -> BimAgent:
    del fingerprint, configuration_fingerprint  # Cache-key-only invalidation signals.
    return BimAgent(data_dir)


def _agent(project_id: str | None = None) -> BimAgent:
    data_dir = _project_directory(project_id)
    return _cached_agent(str(data_dir), _fingerprint(data_dir), _configuration_fingerprint())


_agent.cache_clear = _cached_agent.cache_clear  # type: ignore[attr-defined]


@app.get("/api/health")
def health() -> dict:
    try:
        agent = _agent()
        return {
            "status": "ok",
            "raw_records": len(agent.tools.records),
            "model_directed": True,
            "local_python": agent.agent.python_sandbox.status(),
        }
    except (ProjectError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    try:
        report = _agent(request.project_id).ask(request.question)
    except ProjectError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise _upstream_http_exception(exc) from exc
    return evaluator_payload(report, client_id=request.client_id, project_id=request.project_id)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def generate():
        yield json.dumps({"type": "stage", "stage": "started"}) + "\n"
        try:
            report = _agent(request.project_id).ask(request.question)
            payload = evaluator_payload(report, client_id=request.client_id, project_id=request.project_id)
            yield json.dumps({"type": "result", "report": payload}, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({
                "type": "error",
                "code": "upstream_unavailable" if _is_retryable_api_error(exc) else "agent_error",
                "retryable": _is_retryable_api_error(exc),
                "message": str(exc),
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _upstream_http_exception(exc: Exception) -> HTTPException:
    retryable = _is_retryable_api_error(exc)
    return HTTPException(
        status_code=503,
        detail={
            "code": "upstream_unavailable" if retryable else "agent_error",
            "retryable": retryable,
            "message": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
