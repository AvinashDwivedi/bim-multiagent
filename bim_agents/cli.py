from __future__ import annotations

import argparse
import json
import sys

from bim_agent import BimAgent

from .adapter import evaluator_payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Evaluator-compatible BIM agent CLI")
    value.add_argument("question")
    value.add_argument("--client-id")
    value.add_argument("--project-id")
    value.add_argument("--data-dir", default=None)
    value.add_argument("--quiet", action="store_true")
    value.add_argument("--no-llm", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    # `None` lets BIM_USE_LLM and API-key availability control normal runs.
    # Only --no-llm is an explicit CLI override.
    report = BimAgent(args.data_dir, use_llm=False if args.no_llm else None).ask(args.question)
    payload = evaluator_payload(report, client_id=args.client_id, project_id=args.project_id)
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.quiet else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
