from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from .project_tools import RawProjectTools


WORKSPACE_FORMAT_VERSION = "3"


class HostedPythonWorkspaceError(RuntimeError):
    pass


class HostedPythonWorkspace:
    """Lazily stage project evidence in an isolated OpenAI Code Interpreter container."""

    def __init__(
        self,
        *,
        client: Any,
        project_tools: RawProjectTools,
        cache_root: Path,
        enabled: bool,
        memory_limit: str,
        expiry_minutes: int,
    ):
        self.client = client
        self.project_tools = project_tools
        self.cache_root = cache_root
        self.enabled = enabled
        self.memory_limit = memory_limit
        self.expiry_minutes = expiry_minutes
        self._container_id: str | None = None
        self._lock = threading.RLock()
        self._last_error: str | None = None

    def tool_definition(self) -> dict[str, Any] | None:
        if not self.enabled or not hasattr(self.client, "containers"):
            return None
        try:
            container_id = self._ensure_container()
        except HostedPythonWorkspaceError:
            # Preserve the core read-only agent if the optional hosted runtime
            # is unavailable; inspect() exposes the concrete preparation error.
            return None
        return {
            "type": "code_interpreter",
            "container": container_id,
            "allowed_callers": ["direct"],
        }

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available_on_client": hasattr(self.client, "containers"),
            "container_id": self._container_id,
            "network_access": "disabled",
            "memory_limit": self.memory_limit,
            "expiry_minutes_after_last_activity": self.expiry_minutes,
            "last_error": self._last_error,
        }

    def _ensure_container(self) -> str:
        with self._lock:
            if self._container_id:
                try:
                    current = self.client.containers.retrieve(self._container_id)
                    if str(getattr(current, "status", "active")).casefold() not in {"deleted", "expired"}:
                        return self._container_id
                except Exception:
                    self._container_id = None

            fingerprint = self._project_fingerprint()
            bundle_dir = self.cache_root / fingerprint
            try:
                files = self.project_tools.export_python_workspace(bundle_dir)
                container = self.client.containers.create(
                    name=f"bim-analysis-{fingerprint[:12]}",
                    expires_after={"anchor": "last_active_at", "minutes": self.expiry_minutes},
                    memory_limit=self.memory_limit,
                    network_policy={"type": "disabled"},
                )
                container_id = str(getattr(container, "id", "") or "")
                if not container_id:
                    raise HostedPythonWorkspaceError("Container API returned no container ID.")
                for path in files:
                    with path.open("rb") as stream:
                        self.client.containers.files.create(container_id, file=stream)
                self._container_id = container_id
                self._last_error = None
                return container_id
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise HostedPythonWorkspaceError(
                    f"Could not prepare the sandboxed Python workspace: {self._last_error}"
                ) from exc

    def _project_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(WORKSPACE_FORMAT_VERSION.encode("ascii"))
        for item in self.project_tools.manifest():
            digest.update(str(item["kind"]).encode("utf-8"))
            digest.update(str(item["sha256"]).encode("ascii"))
        return digest.hexdigest()[:24]
