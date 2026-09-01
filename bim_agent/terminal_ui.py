from __future__ import annotations

from datetime import datetime
import os
import sys
import threading
import textwrap
from typing import Any


_OUTPUT_LOCK = threading.Lock()
_RESET = "\033[0m"
_COLORS = {
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "dim": "\033[2m",
    "bold": "\033[1m",
}
_ASCII_TRANSLATION = str.maketrans({
    "╭": "+", "├": "|-", "╰": "\\-", "◆": "*", "│": "|",
    "└": "\\-", "✓": "OK", "✗": "X", "●": "*", "•": "|", "…": "...",
})


class TerminalStepLogger:
    """Compact terminal rendering for observable agent and tool events."""

    def __init__(self, session_id: str) -> None:
        raw = os.getenv("BIM_PRETTY_LOGS", "true").strip().casefold()
        self.enabled = raw not in {"0", "false", "no", "off"}
        self.color = (
            self.enabled
            and os.getenv("NO_COLOR") is None
            and (sys.stdout.isatty() or os.getenv("FORCE_COLOR") == "1")
        )
        self.unicode = os.name != "nt" and _supports_unicode(sys.stdout)
        self.session = session_id[:8]

    def start(self, *, question: str, model: str, tools: list[str], trace_path: str) -> None:
        self._line("╭", "AGENT", f"session {self.session}", "cyan", bold=True)
        self._line("├", "QUESTION", _single_line(question, 500), "blue")
        self._line("├", "MODEL", model, "dim")
        self._line("├", "TOOLS", "  •  ".join(tools), "dim")
        self._line("╰", "TRACE", trace_path, "dim")

    def iteration(self, number: int, maximum: int) -> None:
        self._line(
            "◆", f"STEP {number:02d}/{maximum:02d}",
            "asking the model what to do next", "cyan", bold=True,
        )

    def model_response(
        self, *, response_id: str, elapsed: float, tools: list[str], has_answer: bool,
    ) -> None:
        action = ", ".join(tools) if tools else ("final answer" if has_answer else "continue")
        self._line("│", "MODEL", f"{response_id or 'no-id'}  •  {elapsed:.2f}s  •  {action}", "blue")

    def tool_start(self, *, name: str, command: str, backend: str | None = None) -> None:
        suffix = f"  [{backend}]" if backend else ""
        self._line("├", f"TOOL {name}", _single_line(command, 600) + suffix, "yellow")

    def tool_result(
        self, *, name: str, outcome: str, elapsed: float, characters: int, preview: str = "",
    ) -> None:
        color = "green" if outcome == "success" else "red"
        symbol = "✓" if outcome == "success" else "✗"
        detail = f"{symbol} {outcome}  •  {elapsed:.2f}s  •  {characters:,} chars"
        if preview.strip():
            detail += "  •  " + _single_line(preview, 240)
        self._line("└", f"TOOL {name}", detail, color)

    def web_search(self, *, status: str, call_id: str) -> None:
        self._line("├", "WEB SEARCH", f"{status}  •  {call_id or 'no-id'}", "yellow")

    def error(self, *, stage: str, error: BaseException) -> None:
        self._line("✗", stage.upper(), f"{type(error).__name__}: {error}", "red", bold=True)

    def finish(
        self, *, status: str, iterations: int, elapsed: float, cost: Any, answer: str,
    ) -> None:
        color = "green" if status == "completed" else "yellow"
        amount = cost if isinstance(cost, (int, float)) else None
        cost_text = f"${amount:.6f}" if amount is not None else "cost unavailable"
        self._line(
            "●", "DONE", f"{status}  •  {iterations} steps  •  {elapsed:.2f}s  •  {cost_text}",
            color, bold=True,
        )
        self._line(" ", "ANSWER", _single_line(answer, 700), color)

    def _line(
        self, symbol: str, label: str, detail: str, color: str, *, bold: bool = False,
    ) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {symbol} {label:<12}"
        if self.color:
            style = _COLORS.get(color, "") + (_COLORS["bold"] if bold else "")
            rendered = f"{_COLORS['dim']}[{timestamp}]{_RESET} {style}{symbol} {label:<12}{_RESET}{detail}"
        else:
            rendered = prefix + detail
        if not self.unicode:
            rendered = rendered.translate(_ASCII_TRANSLATION)
        rendered = _terminal_safe(rendered, sys.stdout)
        with _OUTPUT_LOCK:
            print(rendered, flush=True)


def _single_line(value: str, limit: int) -> str:
    compact = " ".join(str(value).split())
    return textwrap.shorten(compact, width=limit, placeholder=" …")


def _supports_unicode(stream: Any) -> bool:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "╭✓•…".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def _terminal_safe(value: str, stream: Any) -> str:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        return value.encode(encoding, errors="backslashreplace").decode(encoding)
    except LookupError:
        return value
