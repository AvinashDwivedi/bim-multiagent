from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False

    def as_api_output(self) -> dict[str, Any]:
        outcome: dict[str, Any] = (
            {"type": "timeout"}
            if self.timed_out
            else {"type": "exit", "exit_code": int(self.exit_code or 0)}
        )
        return {"stdout": self.stdout, "stderr": self.stderr, "outcome": outcome}


class ReadonlyProjectShell:
    """Project-local native Bash executor, matching Claude Code's Bash model."""

    def __init__(
        self,
        project_dir: Path,
        *,
        bash_path: str | Path | None = None,
        timeout_seconds: float = 120.0,
        max_output_chars: int = 80_000,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.bash_path = _find_bash_executable(bash_path)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.max_output_chars = max(1000, max_output_chars)
        self.last_backend = "native_bash"

    def status(self) -> dict[str, Any]:
        if self.bash_path is None:
            return {
                "available": False,
                "mode": "native_bash",
                "reason": (
                    "Bash is unavailable. On Windows, install Git for Windows or set BIM_BASH_PATH "
                    "to bash.exe. Docker is not used."
                ),
                "docker_required": False,
            }
        return {
            "available": True,
            "mode": "native_bash",
            "executable": str(self.bash_path),
            "working_directory": str(self.project_dir),
            "permission_policy": "read_only_by_agent_instruction",
            "docker_required": False,
        }

    def run_action(
        self,
        action: Any,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        commands = list(_value(action, "commands") or [])
        if not commands:
            return [CommandResult("", "No shell command was supplied.", 2).as_api_output()], 1000
        commands = commands[:8]
        requested_timeout_ms = _integer(_value(action, "timeout_ms"))
        timeout = min(
            self.timeout_seconds,
            requested_timeout_ms / 1000 if requested_timeout_ms else self.timeout_seconds,
        )
        requested_output = _integer(_value(action, "max_output_length"))
        max_output = min(self.max_output_chars, requested_output or self.max_output_chars)
        results: list[dict[str, Any]] = []
        for command in commands:
            if should_cancel is not None and should_cancel():
                results.append(CommandResult(
                    "", "Request cancelled before command execution.", 130
                ).as_api_output())
                continue
            results.append(self.run(
                str(command), timeout=timeout, max_output_chars=max_output
            ).as_api_output())
        return results, max_output

    def run(self, command: str, *, timeout: float, max_output_chars: int) -> CommandResult:
        if not command.strip():
            return CommandResult("", "Empty shell command.", 2)
        if len(command) > 100_000:
            return CommandResult("", "Shell command exceeds the 100,000-character limit.", 2)
        if self.bash_path is None:
            return CommandResult("", self.status()["reason"], 127)

        self.last_backend = "native_bash"
        prepared = _python_alias_prelude() + _map_legacy_project_path(command, self.project_dir)
        try:
            result = subprocess.run(
                [str(self.bash_path), "--noprofile", "--norc", "-c", prepared],
                cwd=self.project_dir,
                env=_shell_environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=_creation_flags(),
            )
            return CommandResult(
                _bounded(result.stdout, max_output_chars),
                _bounded(result.stderr, max_output_chars),
                result.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                _bounded(_decoded(exc.stdout), max_output_chars),
                _bounded(_decoded(exc.stderr), max_output_chars),
                None,
                True,
            )
        except OSError as exc:
            return CommandResult("", str(exc), 127)


def _find_bash_executable(configured: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    selected = configured or os.getenv("BIM_BASH_PATH")
    if selected:
        candidates.append(Path(selected).expanduser())
    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
            root = os.getenv(variable)
            if root:
                candidates.extend((
                    Path(root) / "Git" / "bin" / "bash.exe",
                    Path(root) / "Programs" / "Git" / "bin" / "bash.exe",
                ))
        git = shutil.which("git")
        if git:
            git_root = Path(git).resolve().parent.parent
            candidates.extend((git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"))
    else:
        bash = shutil.which("bash")
        if bash:
            candidates.append(Path(bash))

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _python_alias_prelude() -> str:
    executable = Path(sys.executable).resolve().as_posix().replace("'", "'\\''")
    return (
        f"python3() {{ '{executable}' \"$@\"; }}\n"
        f"python() {{ '{executable}' \"$@\"; }}\n"
        f"py() {{ '{executable}' \"$@\"; }}\n"
    )


def _map_legacy_project_path(command: str, project_dir: Path) -> str:
    """Keep older model turns using /project working after switching to native Bash."""
    if "/project" not in command:
        return command
    native = project_dir.as_posix()
    return command.replace("/project", native)


def _shell_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not _is_secret_environment_key(key)
    }
    environment.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return environment


def _is_secret_environment_key(key: str) -> bool:
    normalized = key.casefold()
    return any(part in normalized for part in (
        "api_key", "authorization", "password", "secret", "token",
    ))


def _bounded(value: str | None, limit: int) -> str:
    if value is None:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[output truncated]"


def _decoded(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
