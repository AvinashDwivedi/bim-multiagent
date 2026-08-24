from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from bim_context import BimContext, Settings

from .graph_contract import load_graph_contract, validate_live_schema
from .observability import PipelineEvents, configure_logging
from .runtime import answer_bim_question


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
logger = configure_logging(
    verbose=os.getenv("BIM_WEB_QUIET", "0") != "1",
    log_file=os.getenv("BIM_WEB_LOG_FILE") or None,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    client_id: str | None = None
    project_id: str | None = None


async def index(request: Request) -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


async def asset(request: Request) -> FileResponse:
    filename = request.path_params["filename"]
    if filename not in {"styles.css", "app.js"}:
        return JSONResponse({"detail": "Asset not found."}, status_code=404)
    return FileResponse(WEB_DIR / filename, headers={"Cache-Control": "no-cache"})


async def health(request: Request) -> JSONResponse:
    def inspect() -> dict:
        bim = BimContext(Settings.from_env())
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
        return JSONResponse(await run_in_threadpool(inspect))
    except Exception as exc:
        logger.error("health.failed | %s: %s", type(exc).__name__, exc)
        return JSONResponse({"detail": "BIM service is unavailable."}, status_code=503)


async def chat(request: Request) -> JSONResponse:
    try:
        payload = ChatRequest.model_validate(await request.json())
        logger.info("chat.request | mode=json | chars=%d", len(payload.question))
        report = await answer_bim_question(
            payload.question, client_id=payload.client_id, project_id=payload.project_id,
            hooks=PipelineEvents(logger),
        )
        return JSONResponse(report.model_dump(mode="json"))
    except Exception as exc:
        logger.error("chat.failed | %s: %s", type(exc).__name__, exc)
        return JSONResponse({"detail": "The BIM workflow could not complete."}, status_code=500)


async def chat_stream(request: Request) -> StreamingResponse:
    """Stream ordered runtime lifecycle events and the final report as NDJSON."""
    try:
        payload = ChatRequest.model_validate(await request.json())
    except Exception:
        return JSONResponse({"detail": "A valid question is required."}, status_code=400)
    logger.info("chat.request | mode=stream | chars=%d", len(payload.question))

    queue: asyncio.Queue[dict] = asyncio.Queue()

    def publish(event: dict) -> None:
        queue.put_nowait(event)

    async def execute() -> None:
        try:
            report = await answer_bim_question(
                payload.question,
                client_id=payload.client_id,
                project_id=payload.project_id,
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

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app = Starlette(
    debug=False,
    routes=[
        Route("/", index),
        Route("/assets/{filename}", asset),
        Route("/api/health", health),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/chat/stream", chat_stream, methods=["POST"]),
    ],
)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "bim_agents.webapp:app",
        host=os.getenv("BIM_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("BIM_WEB_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
