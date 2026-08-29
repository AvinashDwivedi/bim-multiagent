from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bim_agent import BimAgent
from bim_agent.config import Settings
from bim_agent.hosted_python import HostedPythonWorkspace
import bim_agent.hosted_python as hosted_python

from conftest import final_response, tool_response


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
    agent = BimAgent(settings=_settings(sample_data, tmp_path), client=client)
    report = agent.ask("Analyze the data")

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
    assert client.containers.created[0]["expires_after"]["minutes"] == 20
    hosted_status = agent.inspect()["agentic_flow"]["hosted_python"]
    assert hosted_status["execution_timeout_seconds"] == 180.0
    assert "no source-data mount or write-back path" in hosted_status["project_inputs"]
    assert report.agent_loop["route"]["hosted_python_exposed"] is True
    assert {name for _, name in client.containers.files.uploaded} == {
        "model-tree.json",
        "model-properties.json",
        "model.ifc",
        "bim_workspace.sqlite",
        "bim_workspace_guide.json",
        "upload_manifest.json",
    }
    assert hosted_status["upload_summary"]["compressed_source_count"] == 0
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
    response = SimpleNamespace(
        id="resp-1", output=[python_item], output_text="The result is 42. [ref: ci-1]"
    )
    client = fake_client_factory(response)
    client.containers = FakeContainers()
    report = BimAgent(settings=_settings(sample_data, tmp_path), client=client).ask("Run an analysis")

    record = report.agent_loop["iterations"][0]["python_calls"][0]
    assert record["status"] == "completed"
    assert record["code"] == "import sqlite3\nprint(42)"
    assert record["outputs"][0]["preview"] == "42\n"
    assert client.responses.requests[0]["include"] == ["code_interpreter_call.outputs"]


def test_sql_routing_does_not_create_or_expose_hosted_python(
    sample_data: Path, tmp_path: Path, fake_client_factory
) -> None:
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        tool_response(2, "reconcile_populations", {
            "populations": [{"label": "pipes", "object_ids": ["43", "44"]}],
        }),
        final_response(3, "There are 2 pipes. [ref: call-1] [ref: call-2]"),
    )
    client.containers = FakeContainers()
    BimAgent(settings=_settings(sample_data, tmp_path), client=client).ask(
        "How many pipe objects are in the project?"
    )

    assert client.containers.created == []
    assert not any(
        item.get("type") == "code_interpreter"
        for item in client.responses.requests[0]["tools"]
    )


def test_oversized_hosted_python_file_is_uploaded_as_lossless_gzip(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(hosted_python, "MAX_CONTAINER_FILE_BYTES", 1024)
    source = tmp_path / "large.ifc"
    original = b"IFC-DATA\n" * 2000
    source.write_bytes(original)
    workspace = HostedPythonWorkspace(
        client=SimpleNamespace(),
        project_tools=SimpleNamespace(),
        cache_root=tmp_path,
        enabled=True,
        memory_limit="4g",
        expiry_minutes=20,
        execution_timeout_seconds=180,
    )

    uploads, summary = workspace._prepare_upload_files([source], tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "upload_manifest.json").read_text("utf-8"))
    transport = next(path for path in uploads if path.suffix == ".gz")

    assert transport.stat().st_size <= 1024
    assert gzip.decompress(transport.read_bytes()) == original
    assert manifest["entries"][0]["transport"] == "gzip"
    assert manifest["entries"][0]["restore_as"] == "large.ifc"
    assert summary["compressed_source_count"] == 1


def test_incompressible_hosted_python_file_is_split_into_bounded_gzip_parts(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(hosted_python, "MAX_CONTAINER_FILE_BYTES", 1024)
    source = tmp_path / "large.sqlite"
    original = os.urandom(5000)
    source.write_bytes(original)
    workspace = HostedPythonWorkspace(
        client=SimpleNamespace(),
        project_tools=SimpleNamespace(),
        cache_root=tmp_path,
        enabled=True,
        memory_limit="4g",
        expiry_minutes=20,
        execution_timeout_seconds=180,
    )

    uploads, summary = workspace._prepare_upload_files([source], tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "upload_manifest.json").read_text("utf-8"))
    part_names = manifest["entries"][0]["uploaded_files"]
    parts = [next(path for path in uploads if path.name == name) for name in part_names]

    assert len(parts) > 1
    assert all(path.stat().st_size <= 1024 for path in parts)
    assert gzip.decompress(b"".join(path.read_bytes() for path in parts)) == original
    assert manifest["entries"][0]["transport"] == "gzip_parts"
    assert summary["multipart_source_count"] == 1
