from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bim_agent import BimAgent
from bim_agents.adapter import evaluator_payload
from bim_agents.webapp import app

from conftest import FakeShell, final_response, shell_response


def test_evaluator_payload_reports_only_builtin_tools(
    sample_data: Path, fake_client_factory,
) -> None:
    report = BimAgent(
        sample_data,
        client=fake_client_factory(final_response(1, "Answer from the project snapshot.")),
    ).ask("What is available?")

    payload = evaluator_payload(report, client_id="client", project_id="project")

    assert payload["verification_status"] == "completed"
    assert payload["answer"] == "Answer from the project snapshot."
    assert payload["semantic_checks"] == []
    assert payload["artifact_ids"]
    assert not any("Tool:" in stage for stage in payload["stages_used"])


def test_evaluator_payload_surfaces_shell_usage(
    sample_data: Path, fake_client_factory,
) -> None:
    report = BimAgent(
        sample_data,
        client=fake_client_factory(
            shell_response(1), final_response(2, "Two matching elements.")
        ),
        shell=FakeShell(),
    ).ask("How many match?")

    payload = evaluator_payload(report)

    assert "Tool: shell" in payload["stages_used"]
    assert all("query_bim_workspace" not in stage for stage in payload["stages_used"])


def test_evaluator_api_discovers_and_resolves_projects(
    tmp_path: Path, sample_data: Path, monkeypatch,
) -> None:
    projects_root = tmp_path / "bim-data"
    project = projects_root / "client-one" / "project-one"
    project.mkdir(parents=True)
    for source in sample_data.iterdir():
        (project / source.name).write_bytes(source.read_bytes())
    monkeypatch.setenv("BIM_PROJECTS_ROOT", str(projects_root))

    with TestClient(app) as client:
        listing = client.get("/api/projects")
        health = client.get(
            "/api/health", params={"client_id": "client-one", "project_id": "project-one"},
        )

    assert listing.status_code == 200
    assert listing.json()["client_count"] == 1
    assert listing.json()["clients"][0]["client_id"] == "client-one"
    assert [item["project_id"] for item in listing.json()["projects"]] == ["project-one"]
    assert listing.json()["projects"][0]["source_count"] == 1
    assert list(listing.json()["projects"][0]["artifacts"]) == ["ifc"]
    assert health.status_code == 200
    assert health.json()["client_id"] == "client-one"
    assert health.json()["project_id"] == "project-one"
    assert health.json()["source_count"] == 1
    assert list(health.json()["artifacts"]) == ["ifc"]


def test_evaluator_api_rejects_project_path_traversal(tmp_path: Path, monkeypatch) -> None:
    projects_root = tmp_path / "bim-data"
    projects_root.mkdir()
    monkeypatch.setenv("BIM_PROJECTS_ROOT", str(projects_root))

    response = TestClient(app).get(
        "/api/health", params={"client_id": "client", "project_id": "../escape"},
    )

    assert response.status_code == 503
    assert "plain folder name" in response.json()["detail"]


def test_evaluator_api_rejects_client_path_traversal(tmp_path: Path, monkeypatch) -> None:
    projects_root = tmp_path / "bim-data"
    projects_root.mkdir()
    monkeypatch.setenv("BIM_PROJECTS_ROOT", str(projects_root))

    response = TestClient(app).get(
        "/api/health", params={"client_id": "../escape", "project_id": "project"},
    )

    assert response.status_code == 503
    assert "client_id must be one plain folder name" in response.json()["detail"]
