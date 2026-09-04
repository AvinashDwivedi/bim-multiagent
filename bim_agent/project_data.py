from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIM_DATA_DIR = ROOT / "bim-data"


class ProjectDataError(ValueError):
    """The selected directory does not contain one unambiguous IFC artifact."""


@dataclass(frozen=True)
class ProjectFiles:
    project_dir: Path
    ifc: Path

    def paths(self) -> dict[str, Path]:
        return {"ifc": self.ifc}

    def manifest(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for role, path in self.paths().items():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            output.append({
                "role": role,
                "name": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
        return output


def resolve_project_files(data_dir: str | Path) -> ProjectFiles:
    project_dir = Path(data_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise ProjectDataError(f"Project directory does not exist: {project_dir}")
    files = [path for path in project_dir.iterdir() if path.is_file()]
    ifc = _one_file(
        [path for path in files if path.suffix.casefold() == ".ifc"],
        role="IFC",
        project_dir=project_dir,
    )
    return ProjectFiles(project_dir, ifc)


def configured_projects_root() -> Path:
    """Return the multi-project collection root used by the evaluator API."""
    configured = os.getenv("BIM_PROJECTS_ROOT")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_BIM_DATA_DIR.resolve()


def _plain_folder_name(value: str, *, field: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ProjectDataError(f"{field} must be one plain folder name")
    return normalized


def resolve_project_id(
    client_id: str, project_id: str, *, projects_root: str | Path | None = None,
) -> ProjectFiles:
    """Resolve one client/project pair inside the configured collection."""
    client_name = _plain_folder_name(client_id, field="client_id")
    project_name = _plain_folder_name(project_id, field="project_id")
    root = Path(projects_root or configured_projects_root()).expanduser().resolve()
    client_dir = (root / client_name).resolve()
    project_dir = (client_dir / project_name).resolve()
    try:
        project_dir.relative_to(root)
    except ValueError as exc:
        raise ProjectDataError("project_id resolves outside the BIM data directory") from exc
    if not project_dir.is_dir():
        raise ProjectDataError(f"Project folder does not exist: {project_dir}")
    return resolve_project_files(project_dir)


def project_manifest(
    client_id: str, project_id: str, project: ProjectFiles,
) -> dict[str, Any]:
    """Return the evaluator-facing description of one project snapshot."""
    return {
        "client_id": client_id,
        "project_id": project_id,
        "source_count": len(project.paths()),
        "artifacts": {
            role: {"name": path.name, "bytes": path.stat().st_size}
            for role, path in project.paths().items()
        },
    }


def discover_projects(*, projects_root: str | Path | None = None) -> dict[str, Any]:
    """Scan the collection on every request and separate valid from invalid folders."""
    root = Path(projects_root or configured_projects_root()).expanduser().resolve()
    projects: list[dict[str, Any]] = []
    invalid_projects: list[dict[str, str]] = []
    clients: list[dict[str, Any]] = []
    if root.is_dir():
        client_folders = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for client_folder in client_folders:
            client_projects: list[dict[str, Any]] = []
            project_folders = sorted(
                (path for path in client_folder.iterdir() if path.is_dir()),
                key=lambda path: (path.name.casefold(), path.name),
            )
            for project_folder in project_folders:
                try:
                    project = resolve_project_id(
                        client_folder.name, project_folder.name, projects_root=root,
                    )
                    manifest = project_manifest(
                        client_folder.name, project_folder.name, project,
                    )
                    projects.append(manifest)
                    client_projects.append(manifest)
                except (OSError, ProjectDataError) as exc:
                    invalid_projects.append({
                        "client_id": client_folder.name,
                        "project_id": project_folder.name,
                        "error": str(exc),
                    })
            clients.append({
                "client_id": client_folder.name,
                "project_count": len(client_projects),
                "projects": client_projects,
            })
    return {
        "bim_data_dir": str(root),
        "client_count": len(clients),
        "clients": clients,
        "project_count": len(projects),
        "projects": projects,
        "invalid_projects": invalid_projects,
    }


def _one_file(files: list[Path], *, role: str, project_dir: Path) -> Path:
    if len(files) == 1:
        resolved = files[0].resolve()
        try:
            resolved.relative_to(project_dir)
        except ValueError as exc:
            raise ProjectDataError(f"The {role} file resolves outside the project directory.") from exc
        return resolved
    if not files:
        raise ProjectDataError(f"Project directory has no {role} file.")
    names = ", ".join(sorted(path.name for path in files))
    raise ProjectDataError(f"Project directory has ambiguous {role} files: {names}")
