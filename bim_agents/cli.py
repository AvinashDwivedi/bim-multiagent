from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .observability import PipelineEvents, configure_logging
from .runtime import answer_bim_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a project-scoped BIM calculation pipeline.")
    parser.add_argument("question", help="The BIM question to answer")
    parser.add_argument("--client-id", help="Client UUID for this request (overrides BIM_CLIENT_ID)")
    parser.add_argument("--project-id", help="Project UUID for this request (overrides BIM_PROJECT_ID)")
    parser.add_argument("--timeout", type=float, default=None, help="Maximum total runtime in seconds (default: 240)")
    parser.add_argument("--log-file", help="Also write detailed lifecycle logs to this file")
    parser.add_argument("--quiet", action="store_true", help="Hide progress logs")
    args = parser.parse_args()
    logger = configure_logging(verbose=not args.quiet, log_file=args.log_file)
    try:
        report = asyncio.run(
            answer_bim_question(
                args.question,
                client_id=args.client_id,
                project_id=args.project_id,
                timeout_seconds=args.timeout,
                hooks=PipelineEvents(logger),
            )
        )
    except Exception as exc:
        logger.error("run.failed  | %s: %s", type(exc).__name__, exc)
        if args.log_file:
            logger.debug("failure traceback", exc_info=True)
        raise SystemExit(1) from exc
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
