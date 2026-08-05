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
from .observability import BimRunHooks, configure_logging
from .runtime import answer_bim_question


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
logger = configure_logging(
    verbose=os.getenv("BIM_WEB_QUIET", "0") != "1",
    log_file=os.getenv("BIM_WEB_LOG_FILE") or None,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


async def index(request: Request) -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


async def asset(request: Request) -> FileResponse:
    filename = request.path_params["filename"]
    if filename not in {"styles.css", "app.js"}:
        return JSONResponse({"detail": "Asset not found."}, status_code=404)
    return FileResponse(WEB_DIR / filename)


async def health(request: Request) -> JSONResponse:
    def inspect() -> dict:
        bim = BimContext(Settings.from_env())
        try:
            bim.connect()
            summary = bim.get_project_summary()
            contract = load_graph_contract(os.getenv("BIM_GRAPH_SCHEMA_PATH") or None)
            validate_live_schema(bim, contract)
            return {
                "status": "ok",
                "project_id": bim.settings.project_id,
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
        report = await answer_bim_question(payload.question, hooks=BimRunHooks(logger))
        return JSONResponse(report.model_dump(mode="json"))
    except Exception as exc:
        logger.error("chat.failed | %s: %s", type(exc).__name__, exc)
        return JSONResponse(
            {"detail": f"{type(exc).__name__}: {exc}"},
            status_code=500,
        )


async def chat_stream(request: Request) -> StreamingResponse:
    """Stream ordered runtime lifecycle events and the final report as NDJSON."""
    try:
        payload = ChatRequest.model_validate(await request.json())
    except Exception:
        return JSONResponse({"detail": "A valid question is required."}, status_code=400)

    queue: asyncio.Queue[dict] = asyncio.Queue()

    def publish(event: dict) -> None:
        queue.put_nowait(event)

    async def execute() -> None:
        try:
            report = await answer_bim_question(
                payload.question,
                hooks=BimRunHooks(logger, event_sink=publish),
            )
            await queue.put({"type": "result", "report": report.model_dump(mode="json")})
        except Exception as exc:
            logger.error("chat.stream.failed | %s: %s", type(exc).__name__, exc)
            await queue.put({
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
            })

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
