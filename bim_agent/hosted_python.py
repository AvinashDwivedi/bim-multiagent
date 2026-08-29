from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

from .project_tools import RawProjectTools


WORKSPACE_FORMAT_VERSION = "4"
# The API rejects individual files above 50 MB. Keep transport artifacts below
# that boundary to avoid decimal/binary-unit ambiguity and request overhead.
MAX_CONTAINER_FILE_BYTES = 49_000_000
UPLOAD_MANIFEST_NAME = "upload_manifest.json"


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
        execution_timeout_seconds: float,
    ):
        self.client = client
        self.project_tools = project_tools
        self.cache_root = cache_root
        self.enabled = enabled
        self.memory_limit = memory_limit
        # The Containers API accepts 1-20 minutes after last activity.
        self.expiry_minutes = min(20, max(1, int(expiry_minutes)))
        self.execution_timeout_seconds = execution_timeout_seconds
        self._container_id: str | None = None
        self._lock = threading.RLock()
        self._last_error: str | None = None
        self._upload_summary: dict[str, Any] | None = None

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
            "project_inputs": (
                "lossless uploaded snapshots isolated from source; oversized artifacts use manifest-described "
                "gzip transport; no source-data mount or write-back path"
            ),
            "memory_limit": self.memory_limit,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "expiry_minutes_after_last_activity": self.expiry_minutes,
            "max_upload_file_bytes": MAX_CONTAINER_FILE_BYTES,
            "upload_summary": self._upload_summary,
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
                source_files = self.project_tools.export_python_workspace(bundle_dir)
                files, upload_summary = self._prepare_upload_files(source_files, bundle_dir)
                self._upload_summary = upload_summary
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

    def _prepare_upload_files(
        self, source_files: list[Path], bundle_dir: Path,
    ) -> tuple[list[Path], dict[str, Any]]:
        """Create lossless, size-bounded transport artifacts for a container upload."""
        bundle_dir.mkdir(parents=True, exist_ok=True)
        upload_files: list[Path] = []
        entries: list[dict[str, Any]] = []
        for index, source in enumerate(source_files, start=1):
            original_size = source.stat().st_size
            entry: dict[str, Any] = {
                "source_name": source.name,
                "original_size_bytes": original_size,
                "sha256": _file_sha256(source),
                "restore_as": source.name,
            }
            if original_size <= MAX_CONTAINER_FILE_BYTES:
                entry.update({
                    "transport": "raw",
                    "uploaded_files": [source.name],
                })
                upload_files.append(source)
                entries.append(entry)
                continue

            gzip_path = bundle_dir / f"transport-{index:02d}-{source.name}.gz"
            _gzip_source(source, gzip_path)
            compressed_size = gzip_path.stat().st_size
            entry["compressed_size_bytes"] = compressed_size
            if compressed_size <= MAX_CONTAINER_FILE_BYTES:
                entry.update({
                    "transport": "gzip",
                    "uploaded_files": [gzip_path.name],
                })
                upload_files.append(gzip_path)
                entries.append(entry)
                continue

            parts = _split_file(gzip_path, bundle_dir, MAX_CONTAINER_FILE_BYTES)
            entry.update({
                "transport": "gzip_parts",
                "uploaded_files": [path.name for path in parts],
                "part_count": len(parts),
            })
            upload_files.extend(parts)
            entries.append(entry)

        manifest_path = bundle_dir / UPLOAD_MANIFEST_NAME
        manifest = {
            "version": 1,
            "lossless": True,
            "max_upload_file_bytes": MAX_CONTAINER_FILE_BYTES,
            "entries": entries,
            "restore_instructions": {
                "raw": "Use the uploaded file directly.",
                "gzip": (
                    "Use Python gzip.open(uploaded_files[0], 'rb') and write the bytes to restore_as."
                ),
                "gzip_parts": (
                    "Concatenate uploaded_files in the listed order, then decompress the combined gzip stream "
                    "to restore_as."
                ),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if manifest_path.stat().st_size > MAX_CONTAINER_FILE_BYTES:
            raise HostedPythonWorkspaceError("The generated upload manifest exceeds the file-size limit.")
        upload_files.append(manifest_path)
        summary = {
            "source_file_count": len(source_files),
            "uploaded_file_count": len(upload_files),
            "compressed_source_count": sum(
                entry["transport"] != "raw" for entry in entries
            ),
            "multipart_source_count": sum(
                entry["transport"] == "gzip_parts" for entry in entries
            ),
            "manifest": manifest_path.name,
        }
        return upload_files, summary

    def _project_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(WORKSPACE_FORMAT_VERSION.encode("ascii"))
        for item in self.project_tools.manifest():
            digest.update(str(item["kind"]).encode("utf-8"))
            digest.update(str(item["sha256"]).encode("ascii"))
        return digest.hexdigest()[:24]


def _gzip_source(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".building")
    with source.open("rb") as input_stream, gzip.open(temporary, "wb", compresslevel=6) as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    temporary.replace(destination)


def _split_file(source: Path, directory: Path, part_size: int) -> list[Path]:
    parts: list[Path] = []
    with source.open("rb") as stream:
        index = 1
        while True:
            chunk = stream.read(part_size)
            if not chunk:
                break
            part = directory / f"{source.name}.part{index:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    if not parts:
        raise HostedPythonWorkspaceError(f"Could not split empty transport artifact: {source.name}")
    return parts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
