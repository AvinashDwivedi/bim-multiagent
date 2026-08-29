from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bim_agent import BimAgent
from bim_agent.config import Settings

from conftest import final_response


class FakeContainerFiles:
    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []

    def create(self, container_id: str, *, file: Any) -> Any:
        self.uploaded.append((container_id, Path(file.name).name))
        return SimpleNamespace(id=f"file-{len(self.uploaded)}")


class FakeContainers:
    def __init__(self) -> None:
        self.files = FakeContainerFiles()
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(id="container-bim", status="active")

    def retrieve(self, container_id: str) -> Any:
        return SimpleNamespace(id=container_id, status="active")


def _settings(sample_data: Path, tmp_path: Path) -> Settings:
    return Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        max_agent_iterations=4,
        enable_hosted_python=True,
        python_memory_limit="4g",
        python_expiry_minutes=120,
    )


def test_hosted_python_uploads_sources_and_normalized_workspace(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    client = fake_client_factory(final_response(1, "Done."))
    client.containers = FakeContainers()
    report = BimAgent(settings=_settings(sample_data, tmp_path), client=client).ask("Analyze the data")

    code_tool = next(
        item for item in client.responses.requests[0]["tools"]
        if item.get("type") == "code_interpreter"
    )
    assert code_tool == {
        "type": "code_interpreter",
        "container": "container-bim",
        "allowed_callers": ["direct"],
    }
    assert client.containers.created[0]["network_policy"] == {"type": "disabled"}
    assert client.containers.created[0]["memory_limit"] == "4g"
    assert {name for _, name in client.containers.files.uploaded} == {
        "model-tree.json",
        "model-properties.json",
        "model.ifc",
        "bim_workspace.sqlite",
        "bim_workspace_guide.json",
    }
    assert report.agent_loop["tools_available"][-2:] == ["code_interpreter", "programmatic_tool_calling"]


def test_code_interpreter_execution_is_preserved_in_trace(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    python_item = SimpleNamespace(
        type="code_interpreter_call",
        id="ci-1",
        container_id="container-bim",
        status="completed",
        code="import sqlite3\nprint(42)",
        outputs=[SimpleNamespace(type="logs", logs="42\n")],
    )
    response = SimpleNamespace(id="resp-1", output=[python_item], output_text="The result is 42.")
    client = fake_client_factory(response)
    client.containers = FakeContainers()
    report = BimAgent(settings=_settings(sample_data, tmp_path), client=client).ask("Run an analysis")

    record = report.agent_loop["iterations"][0]["python_calls"][0]
    assert record["status"] == "completed"
    assert record["code"] == "import sqlite3\nprint(42)"
    assert record["outputs"][0]["preview"] == "42\n"
    assert client.responses.requests[0]["include"] == ["code_interpreter_call.outputs"]
