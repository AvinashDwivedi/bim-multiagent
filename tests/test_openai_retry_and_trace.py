import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from bim_agents.anthropic_agent import ANSWER_TOOL
from bim_agents.llm.provider import OpenAIProvider, _OpenAISession
from bim_agents.trace import AgentTraceLog


class OutputItem:
    type = "function_call"
    name = "submit_answer"
    call_id = "call-1"

    def __init__(self, arguments: str):
        self.arguments = arguments

    def model_dump(self, **_kwargs):
        return {
            "type": self.type,
            "name": self.name,
            "call_id": self.call_id,
            "arguments": self.arguments,
        }


class FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                output=[OutputItem('{"answer":"unterminated')],
                output_text="",
            )
        return SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output=[OutputItem('{"answer":"Complete","limitations":[]}')],
            output_text="",
        )


@pytest.mark.asyncio
async def test_structured_openai_call_repairs_truncated_function_json():
    responses = FakeResponses()
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.client = SimpleNamespace(responses=responses)
    result = await provider.structured(
        model="gpt-5.6-sol", instructions="Return an answer", payload={"synthetic": True},
        tool=ANSWER_TOOL, max_output_tokens=1_000, reasoning_effort="high",
    )
    assert result["answer"] == "Complete"
    assert len(responses.calls) == 2
    assert responses.calls[1]["reasoning"] == {"effort": "low"}
    assert responses.calls[1]["max_output_tokens"] == 8_000


@pytest.mark.asyncio
async def test_tool_session_repairs_truncated_function_json():
    responses = FakeResponses()
    session = _OpenAISession(
        SimpleNamespace(responses=responses), model="gpt-5.6-sol", instructions="Investigate",
        payload={"synthetic": True}, tools=[ANSWER_TOOL], max_output_tokens=2_000,
        reasoning_effort="high",
    )
    turn = await session.next()
    assert turn.tool_calls[0].arguments["answer"] == "Complete"
    assert len(responses.calls) == 2
    assert responses.calls[1]["max_output_tokens"] == 10_000


def test_trace_is_jsonl_and_truncates_long_values():
    directory = Path(f"pytest-cache-files-trace-{uuid4().hex}")
    trace = AgentTraceLog(enabled=True, directory=directory)
    try:
        request_id = trace.start(question="synthetic trace")
        trace.log("tool_arguments", agent="Test", value="x" * 5_000)
        trace.finish(status="verified")
        path = trace.active_path
        assert path is not None
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [record["event"] for record in records] == [
            "request_start", "tool_arguments", "request_end"
        ]
        assert all(record["request_id"] == request_id for record in records)
        assert records[1]["value"].endswith("...[truncated]")
    finally:
        if trace.active_path and trace.active_path.exists():
            trace.active_path.unlink()
        if directory.exists():
            directory.rmdir()
