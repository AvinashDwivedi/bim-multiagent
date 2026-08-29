from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .project_tools import RawProjectTools


WORKSPACE_FORMAT_VERSION = "1"
MAX_CODE_CHARACTERS = 30_000
_PROBE_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_PROBE_LOCK = threading.Lock()


class LocalPythonSandbox:
    """Execute model-authored Python in a constrained local Docker container.

    Project files are exposed only through local read-only bind mounts. This
    class has no OpenAI client and never calls an upload, file, or container API.
    """

    def __init__(
        self,
        *,
        project_tools: RawProjectTools,
        cache_root: Path,
        enabled: bool,
        image: str,
        memory_limit: str,
        cpus: float,
        execution_timeout_seconds: float,
        max_output_characters: int,
    ):
        self.project_tools = project_tools
        self.cache_root = cache_root
        self.enabled = bool(enabled)
        self.image = image.strip() or "python:3.12-slim"
        self.memory_limit = memory_limit
        self.cpus = max(0.1, float(cpus))
        self.execution_timeout_seconds = max(1.0, float(execution_timeout_seconds))
        self.max_output_characters = max(1000, int(max_output_characters))
        self.docker_executable = _find_docker()
        self._last_error: str | None = None

    def tool_definition(self) -> dict[str, Any] | None:
        runtime = self._runtime_status()
        if not runtime["ready"]:
            return None
        return {
            "type": "function",
            "name": "run_local_python",
            "description": (
                "Run Python inside the on-device Docker sandbox for custom BIM calculations that are not "
                "reasonably expressible as one read-only SQL query. Networking is disabled. Raw sources are "
                "read-only under /project; bim_workspace.sqlite and bim_workspace_guide.json are read-only "
                "under /workspace. Use sqlite3 URI mode=ro and print compact JSON/text evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "maxLength": MAX_CODE_CHARACTERS},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
            "allowed_callers": ["direct"],
        }

    def status(self) -> dict[str, Any]:
        runtime = self._runtime_status()
        return {
            "enabled": self.enabled,
            "runtime": "local_docker",
            "ready": runtime["ready"],
            "docker_cli": self.docker_executable,
            "docker_engine_available": runtime["engine_available"],
            "image": self.image,
            "image_available": runtime["image_available"],
            "network_access": "disabled (--network none)",
            "project_inputs": "local read-only bind mounts; project files are never uploaded",
            "filesystem": "read-only root and project/workspace mounts; temporary /tmp only",
            "memory_limit": self.memory_limit,
            "cpus": self.cpus,
            "pids_limit": 64,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "max_output_characters": self.max_output_characters,
            "last_error": self._last_error or runtime.get("error"),
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        code = str(arguments.get("code") or "")
        if not code.strip():
            raise ValueError("run_local_python requires non-empty code.")
        if len(code) > MAX_CODE_CHARACTERS:
            raise ValueError(
                f"run_local_python code exceeds {MAX_CODE_CHARACTERS:,} characters."
            )
        runtime = self._runtime_status(force=True)
        if not runtime["ready"]:
            message = str(runtime.get("error") or "the local Docker sandbox is not ready")
            self._last_error = message
            raise RuntimeError(message)

        bundle_dir = self._ensure_workspace()
        project_dir = self.project_tools.files.tree.parent.resolve()
        runs_dir = self.cache_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        container_name = f"bim-python-{uuid.uuid4().hex[:12]}"

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="run-", dir=runs_dir) as temporary_name:
            run_dir = Path(temporary_name).resolve()
            script_path = run_dir / "analysis.py"
            script_path.write_text(code, encoding="utf-8")
            command = self._docker_command(
                container_name=container_name,
                project_dir=project_dir,
                bundle_dir=bundle_dir,
                run_dir=run_dir,
            )
            execution = _run_bounded(
                command,
                timeout_seconds=self.execution_timeout_seconds,
                max_output_characters=self.max_output_characters,
                cleanup_command=[self.docker_executable or "docker", "rm", "-f", container_name],
            )

        result = {
            "available": True,
            "runtime": "local_docker",
            "success": execution["exit_code"] == 0 and not execution["timed_out"],
            "exit_code": execution["exit_code"],
            "timed_out": execution["timed_out"],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
            "stdout_characters": execution["stdout_characters"],
            "stderr_characters": execution["stderr_characters"],
            "output_truncated": execution["output_truncated"],
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "network_access": "disabled",
            "project_mount": "/project (read-only)",
            "workspace_mount": "/workspace (read-only)",
            "temporary_writes": "/tmp only",
        }
        if result["success"]:
            self._last_error = None
        else:
            self._last_error = (
                "Local Python timed out." if result["timed_out"] else
                f"Local Python exited with code {result['exit_code']}."
            )
        return result

    def _runtime_status(self, *, force: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {
                "ready": False,
                "engine_available": False,
                "image_available": False,
                "error": "local Python is disabled",
            }
        if not self.docker_executable:
            return {
                "ready": False,
                "engine_available": False,
                "image_available": False,
                "error": "Docker CLI was not found",
            }

        key = (self.docker_executable, self.image)
        now = time.monotonic()
        with _PROBE_LOCK:
            cached = _PROBE_CACHE.get(key)
            if cached and not force and now - cached[0] < 30.0:
                return dict(cached[1])

        engine = _probe_command(
            [self.docker_executable, "version", "--format", "{{.Server.Version}}"],
            timeout_seconds=5.0,
        )
        engine_available = engine["exit_code"] == 0 and bool(engine["stdout"].strip())
        image_available = False
        error = None
        if engine_available:
            image_probe = _probe_command(
                [self.docker_executable, "image", "inspect", self.image, "--format", "{{.Id}}"],
                timeout_seconds=5.0,
            )
            image_available = image_probe["exit_code"] == 0 and bool(image_probe["stdout"].strip())
            if not image_available:
                error = (
                    f"Docker image {self.image!r} is not installed locally. "
                    f"Install it explicitly with: docker pull {self.image}"
                )
        else:
            error = engine["stderr"].strip() or "Docker engine is not running"
        status = {
            "ready": engine_available and image_available,
            "engine_available": engine_available,
            "image_available": image_available,
            "error": error,
        }
        with _PROBE_LOCK:
            _PROBE_CACHE[key] = (now, dict(status))
        return status

    def _ensure_workspace(self) -> Path:
        fingerprint = self._project_fingerprint()
        bundle_dir = (self.cache_root / fingerprint).resolve()
        self.project_tools.export_python_workspace(bundle_dir)
        return bundle_dir

    def _project_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(WORKSPACE_FORMAT_VERSION.encode("ascii"))
        for item in self.project_tools.manifest():
            digest.update(str(item["kind"]).encode("utf-8"))
            digest.update(str(item["sha256"]).encode("ascii"))
        return digest.hexdigest()[:24]

    def _docker_command(
        self,
        *,
        container_name: str,
        project_dir: Path,
        bundle_dir: Path,
        run_dir: Path,
    ) -> list[str]:
        docker = self.docker_executable or "docker"
        return [
            docker,
            "run",
            "--rm",
            "--name", container_name,
            "--network", "none",
            "--read-only",
            "--memory", self.memory_limit,
            "--cpus", f"{self.cpus:g}",
            "--pids-limit", "64",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--workdir", "/workspace",
            "--mount", f"type=bind,source={project_dir},target=/project,readonly",
            "--mount", f"type=bind,source={bundle_dir},target=/workspace,readonly",
            "--mount", f"type=bind,source={run_dir},target=/runner,readonly",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=128m",
            "--env", "HOME=/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "python",
            "-I",
            "-B",
            "/runner/analysis.py",
        ]


def _find_docker() -> str | None:
    located = shutil.which("docker")
    if located:
        return located
    if os.name == "nt":
        candidate = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
        if candidate.is_file():
            return str(candidate)
    return None


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _probe_command(command: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            creationflags=_creation_flags(),
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout.decode("utf-8", errors="replace"),
            "stderr": completed.stderr.decode("utf-8", errors="replace"),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _run_bounded(
    command: list[str],
    *,
    timeout_seconds: float,
    max_output_characters: int,
    cleanup_command: list[str],
) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_creation_flags(),
    )
    limit = max_output_characters * 4
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_total = [0]
    stderr_total = [0]

    def drain(stream: Any, chunks: list[bytes], total: list[int]) -> None:
        stored = 0
        while True:
            block = stream.read(8192)
            if not block:
                break
            total[0] += len(block)
            if stored < limit:
                kept = block[:limit - stored]
                chunks.append(kept)
                stored += len(kept)

    threads = [
        threading.Thread(target=drain, args=(process.stdout, stdout_chunks, stdout_total), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr_chunks, stderr_total), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        exit_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _probe_command(cleanup_command, timeout_seconds=10.0)
        process.kill()
        exit_code = process.wait(timeout=5.0)
    for thread in threads:
        thread.join(timeout=5.0)

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")[:max_output_characters]
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")[:max_output_characters]
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_characters": stdout_total[0],
        "stderr_characters": stderr_total[0],
        "output_truncated": (
            stdout_total[0] > len(b"".join(stdout_chunks))
            or stderr_total[0] > len(b"".join(stderr_chunks))
            or len(stdout) >= max_output_characters
            or len(stderr) >= max_output_characters
        ),
    }
