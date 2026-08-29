from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runtime import BimAgent, report_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="bim-agent", description="Ask evidence-backed questions over three BIM export files.")
    value.add_argument("--data-dir", default=None, help="Directory containing tree JSON, properties JSON, and .ifc")
    commands = value.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Ask a BIM question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", dest="as_json")

    inspect = commands.add_parser("inspect", help="Print discovered model vocabulary and source metadata")
    inspect.add_argument("--json", action="store_true", dest="as_json")

    commands.add_parser("chat", help="Open an interactive session and reuse the loaded model index")

    serve = commands.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return value


def _cost_label(cost: dict) -> str:
    value = cost.get("estimated_cost_usd")
    if value is None:
        return "unavailable (API usage or model pricing missing)"
    qualifier = "estimated" if cost.get("is_complete") else "estimated subtotal"
    return f"${float(value):.6f} USD ({qualifier})"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        app = create_app(data_dir=args.data_dir)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    agent = BimAgent(data_dir=args.data_dir)
    if args.command == "inspect":
        result = agent.inspect()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "chat":
        print("BIM agent ready. Enter a question, or type :quit to exit.")
        while True:
            try:
                question = input("bim> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {":q", ":quit", "quit", "exit"}:
                return 0
            if not question:
                continue
            report = agent.ask(question)
            print(report.answer)
            print(
                f"Status: {report.status} | Cost: {_cost_label(report.cost)} | "
                f"Trace: {report.trace_path}\n"
            )
    report = agent.ask(args.question)
    print(report_json(report) if args.as_json else report.answer)
    if not args.as_json:
        print(f"\nStatus: {report.status}")
        print(f"Cost: {_cost_label(report.cost)}")
        print(f"Trace: {Path(report.trace_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
