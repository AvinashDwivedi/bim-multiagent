from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query as QueryParam, Request
from fastapi.responses import StreamingResponse

from .anthropic_agent import BIMAgent
from .config import Settings
from .contracts import AnswerReport, ChatRequest, HealthReport
from .runtime import answer_safely


def create_app(*, agent: BIMAgent | None = None, settings: Settings | None = None) -> FastAPI:
    configured_settings = settings
    configured_agent = agent

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal configured_settings, configured_agent
        configured_settings = configured_settings or Settings.from_env()
        configured_agent = configured_agent or BIMAgent(configured_settings)
        app.state.settings = configured_settings
        app.state.agent = configured_agent
        try:
            yield
        finally:
            if agent is None and configured_agent is not None:
                await configured_agent.close()

    app = FastAPI(
        title="Agentic BIM Graph Assistant",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/api/health", response_model=HealthReport)
    async def health(
        request: Request,
        client_id: str = QueryParam(min_length=1),
        project_id: str = QueryParam(min_length=1),
    ) -> HealthReport:
        active: BIMAgent = request.app.state.agent
        await asyncio.to_thread(active.graph.verify_connectivity)
        summary = await asyncio.to_thread(active.graph.scope_summary, client_id, project_id)
        return HealthReport(
            model=request.app.state.settings.agent_model,
            provider=getattr(request.app.state.settings, "llm_provider", "unknown"),
            client_id=client_id,
            project_id=project_id,
            source_count=int(summary["source_count"]),
            element_count=int(summary["element_count"]),
        )

    @app.post("/api/chat", response_model=AnswerReport)
    async def chat(payload: ChatRequest, request: Request) -> AnswerReport:
        return await answer_safely(
            request.app.state.agent,
            question=payload.question,
            client_id=str(payload.client_id),
            project_id=str(payload.project_id),
            session_id=payload.session_id,
            request_id=payload.request_id,
            evaluation_run_id=payload.evaluation_run_id,
            evaluation_case_index=payload.evaluation_case_index,
        )

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        async def generate():
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            async def sink(event: dict[str, Any]) -> None:
                await queue.put(event)

            async def run_agent() -> None:
                report = await answer_safely(
                    request.app.state.agent,
                    question=payload.question,
                    client_id=str(payload.client_id),
                    project_id=str(payload.project_id),
                    sink=sink,
                    session_id=payload.session_id,
                    request_id=payload.request_id,
                    evaluation_run_id=payload.evaluation_run_id,
                    evaluation_case_index=payload.evaluation_case_index,
                )
                await queue.put({"type": "result", "report": report.model_dump(mode="json")})

            task = asyncio.create_task(run_agent())
            try:
                while True:
                    event = await queue.get()
                    yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
                    if event.get("type") in {"result", "error"}:
                        break
                    if await request.is_disconnected():
                        break
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run("bim_agents.webapp:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
