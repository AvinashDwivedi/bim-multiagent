from __future__ import annotations

import argparse
import asyncio
import json

from .anthropic_agent import BIMAgent
from .config import Settings
from .contracts import ChatRequest
from .runtime import answer_safely


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Ask the authorized BIM graph a question.")
    value.add_argument("question")
    value.add_argument("--client-id", required=True)
    value.add_argument("--project-id", required=True)
    value.add_argument("--session-id")
    value.add_argument("--request-id")
    value.add_argument("--evaluation-run-id")
    value.add_argument("--evaluation-case-index", type=int)
    value.add_argument("--quiet", action="store_true", help="Emit only the JSON answer contract.")
    return value


async def run(args: argparse.Namespace) -> int:
    request = ChatRequest(
        question=args.question,
        client_id=args.client_id,
        project_id=args.project_id,
        session_id=args.session_id,
        request_id=args.request_id,
        evaluation_run_id=args.evaluation_run_id,
        evaluation_case_index=args.evaluation_case_index,
    )
    agent = BIMAgent(Settings.from_env())
    try:
        report = await answer_safely(
            agent,
            question=request.question,
            client_id=str(request.client_id),
            project_id=str(request.project_id),
            session_id=request.session_id,
            request_id=request.request_id,
            evaluation_run_id=request.evaluation_run_id,
            evaluation_case_index=request.evaluation_case_index,
        )
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))
        return 0
    finally:
        await agent.close()


def main() -> None:
    raise SystemExit(asyncio.run(run(parser().parse_args())))


if __name__ == "__main__":
    main()
