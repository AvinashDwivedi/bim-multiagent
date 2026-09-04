from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bim_agent import BimAgent
from bim_agent.builtin_agent import tool_policy
from bim_agent.config import load_dotenv
from bim_agent.errors import AgentRunCancelled, is_retryable_api_error
from bim_agent.project_data import (
    ProjectDataError,
    configured_projects_root,
    discover_projects,
    project_manifest,
    resolve_project_id,
)

from .adapter import evaluator_payload


# Project routing happens before BimAgent/Settings is constructed, so load the
# application environment here instead of relying on Settings.from_env later.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    client_id: str | None = None
    project_id: str | None = None
    request_id: str | None = None
    evaluation_run_id: str | None = None
    evaluation_case_index: int | None = None


app = FastAPI(title="Local BIM Agent", version="0.1.0")
_active_requests: dict[str, dict] = {}
_active_requests_lock = threading.Lock()


def _register_request(request_id: str) -> dict:
    state = {"cancel": threading.Event(), "progress": None}
    with _active_requests_lock:
        _active_requests[request_id] = state
    return state


def _update_request_progress(request_id: str, progress: dict) -> None:
    with _active_requests_lock:
        state = _active_requests.get(request_id)
        if state is not None:
            state["progress"] = progress


def _finish_request(request_id: str) -> None:
    with _active_requests_lock:
        _active_requests.pop(request_id, None)


def _project_directory(
    client_id: str | None = None, project_id: str | None = None,
) -> Path:
    configured = Path(os.getenv("BIM_DATA_DIR", "test-project-data")).resolve()
    if not project_id:
        return configured
    projects_root = configured_projects_root()
    if projects_root.is_dir():
        if not client_id:
            raise ProjectDataError("client_id is required when project_id is selected")
        return resolve_project_id(
            client_id, project_id, projects_root=projects_root,
        ).project_dir
    single_project_id = os.getenv("BIM_SINGLE_PROJECT_ID", "").strip()
    if not single_project_id:
        raise ProjectDataError(
            "A request project_id cannot be mapped safely. Configure BIM_PROJECTS_ROOT for multiple "
            "projects or BIM_SINGLE_PROJECT_ID for the BIM_DATA_DIR snapshot."
        )
    if project_id != single_project_id:
        raise ProjectDataError(f"Project is not the configured BIM_DATA_DIR snapshot: {project_id}")
    return configured


def _fingerprint(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    files = list(data_dir.glob("*.ifc"))
    return tuple(sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files))


def _configuration_fingerprint() -> tuple[tuple[str, str], ...]:
    names = (
        "BIM_MODEL", "BIM_OPENAI_AGENT_MODEL", "BIM_REASONING_EFFORT", "BIM_MAX_AGENT_ITERATIONS",
        "BIM_MAX_TOOL_OUTPUT_CHARS", "BIM_BASH_PATH", "BIM_SHELL_TIMEOUT_SECONDS",
        "BIM_OPENAI_MAX_RETRIES", "BIM_OPENAI_TIMEOUT_SECONDS",
        "BIM_PROJECTS_ROOT", "BIM_SINGLE_PROJECT_ID",
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


def _agent(client_id: str | None = None, project_id: str | None = None) -> BimAgent:
    data_dir = _project_directory(client_id, project_id)
    return _cached_agent(str(data_dir), _fingerprint(data_dir), _configuration_fingerprint())


_agent.cache_clear = _cached_agent.cache_clear  # type: ignore[attr-defined]


@app.get("/api/projects")
def projects() -> dict:
    return {
        "status": "ok",
        "contract_version": "fastapi-builtin-agent-1",
        **discover_projects(),
    }


@app.get("/api/health")
def health(project_id: str | None = None, client_id: str = "local") -> dict:
    try:
        if not project_id:
            return {
                "status": "ok",
                "contract_version": "fastapi-builtin-agent-1",
                **discover_projects(),
            }
        agent = _agent(client_id, project_id)
        project = project_manifest(client_id, project_id, agent.project)
        return {
            "status": "ok",
            "contract_version": "fastapi-builtin-agent-1",
            "client_id": client_id,
            **project,
            "project_dir": str(agent.project.project_dir),
            "model_directed": True,
            "tools": tool_policy(),
            "shell": agent.shell.status(),
        }
    except (ProjectDataError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/tools")
def tools() -> dict:
    return tool_policy()


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    request_id = request.request_id or str(uuid4())
    state = _register_request(request_id)
    try:
        report = _agent(request.client_id, request.project_id).ask(
            request.question,
            should_cancel=state["cancel"].is_set,
            progress_callback=lambda progress: _update_request_progress(request_id, progress),
        )
    except AgentRunCancelled as exc:
        raise HTTPException(status_code=499, detail="BIM request was cancelled.") from exc
    except ProjectDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise _upstream_http_exception(exc) from exc
    finally:
        _finish_request(request_id)
    return evaluator_payload(report, client_id=request.client_id, project_id=request.project_id)


@app.post("/api/chat/cancel/{request_id}")
def cancel_chat(request_id: str) -> dict:
    with _active_requests_lock:
        state = _active_requests.get(request_id)
        if state is None:
            return {"cancelled": False, "active": False, "partial_cost": None}
        state["cancel"].set()
        progress = state.get("progress") or {}
    return {
        "cancelled": True,
        "active": True,
        "partial_cost": progress.get("cost"),
        "iteration": progress.get("iteration"),
        "phase": progress.get("phase"),
    }


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def generate():
        yield json.dumps({"type": "stage", "stage": "started"}) + "\n"
        try:
            report = _agent(request.client_id, request.project_id).ask(request.question)
            payload = evaluator_payload(report, client_id=request.client_id, project_id=request.project_id)
            yield json.dumps({"type": "result", "report": payload}, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({
                "type": "error",
                "code": "upstream_unavailable" if is_retryable_api_error(exc) else "agent_error",
                "retryable": is_retryable_api_error(exc),
                "message": str(exc),
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _upstream_http_exception(exc: Exception) -> HTTPException:
    retryable = is_retryable_api_error(exc)
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
