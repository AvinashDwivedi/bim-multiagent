from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .builtin_agent import tool_policy
from .errors import is_retryable_api_error
from .project_data import ProjectDataError
from .runtime import BimAgent


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Local BIM Agent", version="0.1.0")

    @lru_cache(maxsize=1)
    def get_agent() -> BimAgent:
        return BimAgent(data_dir=data_dir)

    @app.get("/api/health")
    def health() -> dict:
        try:
            agent = get_agent()
            return {
                "status": "ok",
                "model_directed": True,
                "source_count": len(agent.project.paths()),
                "tools": tool_policy(),
                "shell": agent.shell.status(),
            }
        except (ProjectDataError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/tools")
    def tools() -> dict:
        return tool_policy()

    @app.get("/api/inspect")
    def inspect() -> dict:
        try:
            return get_agent().inspect()
        except (ProjectDataError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        try:
            return get_agent().ask(request.question).to_dict()
        except ProjectDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            retryable = is_retryable_api_error(exc)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "upstream_unavailable" if retryable else "agent_error",
                    "retryable": retryable,
                    "message": str(exc),
                },
            ) from exc

    return app


app = create_app()
