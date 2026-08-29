from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bim_agent import BimAgent
from bim_agent.config import Settings
from bim_agent.local_python import LocalPythonSandbox
import bim_agent.local_python as local_python

from conftest import final_response, tool_response


def _settings(sample_data: Path, tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        data_dir=sample_data,
        trace_dir=tmp_path / "traces",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        max_agent_iterations=4,
        enable_local_python=enabled,
        local_python_image="python:3.12-slim",
        python_memory_limit="4g",
        local_python_cpus=1.25,
        local_python_timeout_seconds=30,
        local_python_output_chars=5000,
    )


def test_local_python_is_a_function_tool_and_never_uses_openai_container_apis(
    sample_data: Path, tmp_path: Path, fake_client_factory, monkeypatch,
) -> None:
    monkeypatch.setattr(
        LocalPythonSandbox,
        "_runtime_status",
        lambda self, force=False: {
            "ready": True, "engine_available": True, "image_available": True, "error": None,
        },
    )
    monkeypatch.setattr(
        LocalPythonSandbox,
        "execute",
        lambda self, arguments: {
            "available": True, "success": True, "stdout": "42\n", "stderr": "",
        },
    )
    client = fake_client_factory(
        tool_response(1, "run_local_python", {"code": "print(42)"}),
        final_response(2, "The local result is 42. [ref: call_1]"),
    )
    client.containers = SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI container API used")),
        files=SimpleNamespace(
            create=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI file API used"))
        ),
    )

    agent = BimAgent(settings=_settings(sample_data, tmp_path), client=client)
    report = agent.ask("Run a custom Python analysis")

    definition = next(
        item for item in client.responses.requests[0]["tools"]
        if item.get("name") == "run_local_python"
    )
    assert definition["type"] == "function"
    assert definition["allowed_callers"] == ["direct"]
    assert "include" not in client.responses.requests[0]
    assert report.status == "completed"
    assert report.agent_loop["route"]["local_python_exposed"] is True
    assert report.agent_loop["iterations"][0]["calls"][0]["tool"] == "run_local_python"


def test_local_docker_command_enforces_network_mount_and_resource_boundaries(
    sample_data: Path, tmp_path: Path, monkeypatch,
) -> None:
    project_tools = SimpleNamespace(
        files=SimpleNamespace(tree=sample_data / "model-tree.json"),
        manifest=lambda: [
            {"kind": "tree", "sha256": "a"},
            {"kind": "properties", "sha256": "b"},
            {"kind": "ifc", "sha256": "c"},
        ],
    )

    def export(directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bim_workspace.sqlite").write_bytes(b"sqlite")
        (directory / "bim_workspace_guide.json").write_text("{}", encoding="utf-8")
        return []

    project_tools.export_python_workspace = export
    sandbox = LocalPythonSandbox(
        project_tools=project_tools,
        cache_root=tmp_path / "local-python",
        enabled=True,
        image="python:3.12-slim",
        memory_limit="4g",
        cpus=1.25,
        execution_timeout_seconds=30,
        max_output_characters=5000,
    )
    sandbox.docker_executable = "docker"
    monkeypatch.setattr(
        sandbox,
        "_runtime_status",
        lambda force=False: {
            "ready": True, "engine_available": True, "image_available": True, "error": None,
        },
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        commands.append(command)
        return {
            "exit_code": 0,
            "timed_out": False,
            "stdout": "{\"count\": 2}\n",
            "stderr": "",
            "stdout_characters": 13,
            "stderr_characters": 0,
            "output_truncated": False,
        }

    monkeypatch.setattr(local_python, "_run_bounded", fake_run)
    result = sandbox.execute({"code": "print('ok')"})
    command = commands[0]

    assert result["success"] is True
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--memory") + 1] == "4g"
    assert command[command.index("--cpus") + 1] == "1.25"
    assert command[command.index("--pids-limit") + 1] == "64"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any("target=/project,readonly" in item for item in mounts)
    assert any("target=/workspace,readonly" in item for item in mounts)
    assert any("target=/runner,readonly" in item for item in mounts)
    assert command[-5:] == ["python:3.12-slim", "python", "-I", "-B", "/runner/analysis.py"]


def test_local_python_is_hidden_when_docker_is_unavailable(
    sample_data: Path, tmp_path: Path, fake_client_factory, monkeypatch,
) -> None:
    monkeypatch.setattr(
        LocalPythonSandbox,
        "_runtime_status",
        lambda self, force=False: {
            "ready": False,
            "engine_available": False,
            "image_available": False,
            "error": "Docker engine is not running",
        },
    )
    client = fake_client_factory(final_response(1, "Could not run local Python."))
    agent = BimAgent(settings=_settings(sample_data, tmp_path), client=client)
    agent.ask("Run a custom Python analysis")

    assert not any(
        item.get("name") == "run_local_python"
        for item in client.responses.requests[0]["tools"]
    )


def test_sql_routing_does_not_expose_local_python(
    sample_data: Path, tmp_path: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        tool_response(1, "query_bim_workspace", {
            "sql": "SELECT COUNT(*) AS count FROM records WHERE name LIKE ?",
            "parameters": ["Pipe [%"], "row_limit": 20,
        }),
        final_response(2, "There are 2 pipes. [ref: call_1]"),
    )
    BimAgent(settings=_settings(sample_data, tmp_path, enabled=False), client=client).ask(
        "How many pipe objects are in the project?"
    )

    assert not any(
        item.get("name") == "run_local_python"
        for item in client.responses.requests[0]["tools"]
    )
