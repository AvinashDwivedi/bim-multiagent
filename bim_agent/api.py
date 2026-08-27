from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .project_tools import ProjectError
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
                "raw_records": len(agent.tools.records),
                "model_directed": True,
            }
        except (ProjectError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/inspect")
    def inspect() -> dict:
        try:
            return get_agent().inspect()
        except (ProjectError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        try:
            return get_agent().ask(request.question).to_dict()
        except (ProjectError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
