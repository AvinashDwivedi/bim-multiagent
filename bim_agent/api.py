from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dataset import DatasetError
from .runtime import BimAgent


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def create_app(data_dir: str | Path | None = None, use_llm: bool | None = None) -> FastAPI:
    app = FastAPI(title="Local BIM Agent", version="0.1.0")

    @lru_cache(maxsize=1)
    def get_agent() -> BimAgent:
        return BimAgent(data_dir=data_dir, use_llm=use_llm)

    @app.get("/api/health")
    def health() -> dict:
        try:
            agent = get_agent()
            return {
                "status": "ok",
                "physical_instances": agent.profile["physical_instance_count"],
                "llm_planner_enabled": agent.settings.use_llm,
            }
        except DatasetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/inspect")
    def inspect() -> dict:
        try:
            return get_agent().inspect()
        except DatasetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        try:
            return get_agent().ask(request.question).to_dict()
        except DatasetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
