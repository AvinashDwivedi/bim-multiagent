from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bim_agent import BimAgent
from bim_agent.api import create_app
from bim_agent.builtin_agent import builtin_tool_definitions, build_system_prompt, tool_policy
from bim_agent.project_data import resolve_project_files
from bim_agent.readonly_shell import ReadonlyProjectShell
from bim_agent.tracing import _safe

from conftest import FakeShell, final_response, shell_response


def test_policy_contains_only_reference_equivalent_builtins() -> None:
    assert builtin_tool_definitions() == [
        {"type": "shell", "environment": {"type": "local"}},
        {"type": "web_search"},
    ]
    policy = tool_policy()
    assert policy["built_in"] == ["shell", "web_search"]
    assert policy["custom"] == []
    assert policy["mcp_servers"] == []
    assert policy["reference_tool_equivalents"] == {
        "shell": ["Read", "Grep", "Glob", "Bash"],
        "web_search": ["WebSearch", "WebFetch"],
    }


def test_runtime_exposes_no_custom_tools_and_returns_shell_output(
    sample_data: Path, fake_client_factory,
) -> None:
    client = fake_client_factory(
        shell_response(1),
        final_response(2, "The project contains 2 matching elements."),
    )
    shell = FakeShell()

    report = BimAgent(sample_data, client=client, shell=shell).ask("How many match?")

    assert report.status == "completed"
    assert report.agent_loop["tools_available"] == ["shell", "web_search"]
    assert report.agent_loop["custom_tools"] == []
    assert len(shell.actions) == 1
    assert client.responses.requests[0]["tools"] == builtin_tool_definitions()
    assert all(tool.get("type") != "function" for tool in client.responses.requests[0]["tools"])
    second_input = client.responses.requests[1]["input"]
    assert any(item.get("type") == "shell_call_output" for item in second_input if isinstance(item, dict))


def test_tools_endpoint_does_not_construct_the_agent(sample_data: Path) -> None:
    response = TestClient(create_app(sample_data)).get("/api/tools")

    assert response.status_code == 200
    assert response.json()["built_in"] == ["shell", "web_search"]
    assert response.json()["custom"] == []


def test_web_search_is_the_only_external_tool(
    sample_data: Path, fake_client_factory,
) -> None:
    response = SimpleNamespace(
        id="resp-web",
        output=[SimpleNamespace(type="web_search_call", id="web-1", status="completed")],
        output_text="External evidence with a cited URL.",
    )

    report = BimAgent(sample_data, client=fake_client_factory(response), shell=FakeShell()).ask(
        "What does the current standard require?"
    )

    assert report.agent_loop["iterations"][0]["calls"][0]["tool"] == "web_search"
    assert report.cost["excluded_tool_fees"] == ["web_search"]


def test_shell_invokes_native_bash_with_project_cwd_and_no_docker(
    sample_data: Path, monkeypatch,
) -> None:
    bash = sample_data / "bash.exe"
    bash.write_bytes(b"")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-shell")
    monkeypatch.setattr("bim_agent.readonly_shell.subprocess.run", fake_run)
    shell = ReadonlyProjectShell(sample_data, bash_path=bash)

    result = shell.run(
        "python3 - <<'PY'\nimport pprint\npprint.pp({'ok': True})\nPY",
        timeout=5,
        max_output_chars=1000,
    )

    assert result.exit_code == 0
    argv, kwargs = calls[0]
    assert argv[:4] == [str(bash.resolve()), "--noprofile", "--norc", "-c"]
    assert "python3()" in argv[4]
    assert "import pprint" in argv[4]
    assert kwargs["cwd"] == sample_data.resolve()
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert all("docker" not in item.casefold() for item in argv)
    assert shell.status()["mode"] == "native_bash"
    assert shell.status()["docker_required"] is False


def test_native_bash_runs_unrestricted_analysis_syntax_used_in_failed_trace(
    sample_data: Path,
) -> None:
    shell = ReadonlyProjectShell(sample_data)
    if shell.bash_path is None:
        pytest.skip("Native Bash is not installed on this test host.")
    command = (
        "python3 - <<'PY'\n"
        "import json,pprint\n"
        "counter=0\n"
        "def inspect_data():\n"
        " global counter\n"
        " counter += 1\n"
        " data=json.load(open('/project/model-tree.json',encoding='utf-8'))\n"
        " pprint.pp({'type':type(data).__name__,'length':len(data),'counter':counter})\n"
        "inspect_data()\n"
        "PY\n"
        "printf 'objects='\n"
        "grep -o 'objectid' model-tree.json | awk '{n++} END {print n}'"
    )

    result = shell.run(command, timeout=15, max_output_chars=4000)

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "'type': 'dict'" in result.stdout
    assert "objects=" in result.stdout


def test_system_prompt_matches_claude_native_bash_contract(sample_data: Path) -> None:
    prompt = build_system_prompt(resolve_project_files(sample_data))

    assert "native Bash" in prompt
    assert "current working directory" in prompt
    assert "Never create, edit, move, delete" in prompt
    assert "Docker" not in prompt
    assert "portable fallback" not in prompt


def test_pretty_terminal_log_shows_each_observable_step(
    sample_data: Path, fake_client_factory, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("BIM_PRETTY_LOGS", "true")
    monkeypatch.setenv("NO_COLOR", "1")
    client = fake_client_factory(shell_response(1), final_response(2, "Two elements."))

    BimAgent(sample_data, client=client, shell=FakeShell()).ask("How many?")

    output = capsys.readouterr().out
    assert "AGENT" in output
    assert "STEP 01/" in output
    assert "TOOL shell" in output
    assert "success" in output
    assert "DONE" in output


def test_trace_redacts_credentials_but_preserves_usage_counts() -> None:
    safe = _safe({
        "api_key": "secret-value",
        "access_token": "secret-token",
        "input_tokens": 123,
        "total_tokens": 456,
    })

    assert safe["api_key"] == "<redacted>"
    assert safe["access_token"] == "<redacted>"
    assert safe["input_tokens"] == 123
    assert safe["total_tokens"] == 456


def test_agent_marks_unverified_answer_limited_when_all_shell_commands_fail(
    sample_data: Path, fake_client_factory,
) -> None:
    class FailedShell(FakeShell):
        def run_action(self, action, *, should_cancel=None):
            self.actions.append(action)
            return ([{
                "stdout": "",
                "stderr": "shell unavailable",
                "outcome": {"type": "exit", "exit_code": 127},
            }], 4096)

    client = fake_client_factory(
        shell_response(1),
        final_response(2, "I could not inspect the project files."),
    )

    report = BimAgent(sample_data, client=client, shell=FailedShell()).ask("How many?")

    assert report.status == "limited"
    assert report.agent_loop["termination_reason"] == "tool_unavailable"
    assert "could not be verified" in report.limitations[0]
