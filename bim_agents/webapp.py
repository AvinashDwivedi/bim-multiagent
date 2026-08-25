from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bim_context import BimContext, Settings

from .graph_contract import load_graph_contract, validate_live_schema
from .observability import PipelineEvents, configure_logging
from .runtime import answer_bim_question

logger = configure_logging(
    verbose=os.getenv("BIM_API_QUIET", os.getenv("BIM_WEB_QUIET", "0")) != "1",
    log_file=os.getenv("BIM_API_LOG_FILE") or os.getenv("BIM_WEB_LOG_FILE") or None,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    client_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)


app = FastAPI(
    title="BIM Multiagent API",
    description="Project-scoped, read-only BIM investigation service.",
    version="1.0.0",
)
origins = [
    value.strip()
    for value in os.getenv(
        "BIM_CORS_ORIGINS", "http://127.0.0.1:8090,http://localhost:8090"
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/")
async def root() -> dict:
    return {"name": "BIM Multiagent API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
async def health(client_id: str, project_id: str) -> dict:
    def inspect() -> dict:
        bim = BimContext(Settings.from_env(client_id=client_id, project_id=project_id))
        try:
            bim.connect()
            contract = load_graph_contract(
                os.getenv("BIM_GRAPH_SCHEMA_PATH") or None,
                client_id=bim.settings.client_id,
                project_id=bim.settings.project_id,
            )
            summary = bim.get_project_summary(contract)
            validate_live_schema(bim, contract, authorization_only=True)
            return {
                "status": "ok",
                "source_count": len(summary.get("allowed_sources") or []),
                "element_count": summary.get("element_count", 0),
                "contract_version": contract.version,
            }
        finally:
            bim.close()

    try:
        return await run_in_threadpool(inspect)
    except Exception as exc:
        logger.error("health.failed | %s: %s", type(exc).__name__, exc)
        raise HTTPException(status_code=503, detail="BIM service is unavailable.") from exc


async def _run_until_disconnect(request: Request, awaitable):
    """Cancel and await backend work when the HTTP client abandons the request."""
    task = asyncio.create_task(awaitable)
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=0.25)
        if done:
            break
        if await request.is_disconnected():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise asyncio.CancelledError
    return await task


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict:
    try:
        logger.info("chat.request | mode=json | chars=%d", len(payload.question))
        report = await _run_until_disconnect(
            request,
            answer_bim_question(
                payload.question, client_id=payload.client_id, project_id=payload.project_id,
                hooks=PipelineEvents(logger),
            ),
        )
        return report.model_dump(mode="json")
    except asyncio.CancelledError:
        logger.info("chat.cancelled | client disconnected")
        raise
    except Exception as exc:
        logger.error("chat.failed | %s: %s", type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail="The BIM workflow could not complete.") from exc


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """Stream ordered runtime lifecycle events and the final report as NDJSON."""
    logger.info("chat.request | mode=stream | chars=%d", len(payload.question))
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def publish(event: dict) -> None:
        queue.put_nowait(event)

    async def execute() -> None:
        try:
            report = await answer_bim_question(
                payload.question, client_id=payload.client_id, project_id=payload.project_id,
                hooks=PipelineEvents(logger, event_sink=publish),
            )
            await queue.put({"type": "result", "report": report.model_dump(mode="json")})
        except Exception as exc:
            logger.error("chat.stream.failed | %s: %s", type(exc).__name__, exc)
            await queue.put({"type": "error", "message": "The BIM workflow could not complete."})

    task = asyncio.create_task(execute())

    async def events():
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn
    uvicorn.run(
        "bim_agents.webapp:app",
        host=os.getenv("BIM_API_HOST", os.getenv("BIM_WEB_HOST", "127.0.0.1")),
        port=int(os.getenv("BIM_API_PORT", os.getenv("BIM_WEB_PORT", "8000"))),
        reload=False,
    )


if __name__ == "__main__":
    main()
