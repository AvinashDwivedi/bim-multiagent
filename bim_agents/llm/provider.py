from __future__ import annotations

import json
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ..config import Settings


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolSession(Protocol):
    async def next(self, results: list[ToolResult] | None = None) -> ModelTurn: ...


class LLMProvider(Protocol):
    async def structured(
        self,
        *,
        model: str,
        instructions: str,
        payload: dict[str, Any],
        tool: dict[str, Any],
        max_output_tokens: int,
        reasoning_effort: str,
    ) -> dict[str, Any]: ...

    def tool_session(
        self,
        *,
        model: str,
        instructions: str,
        payload: dict[str, Any],
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        reasoning_effort: str,
    ) -> ToolSession: ...

    async def close(self) -> None: ...


def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return tool


def _openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["input_schema"],
        "strict": tool.get("strict", True),
    }


def _response_metadata(response: Any, elapsed_ms: float) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "elapsed_ms": round(elapsed_ms, 1),
        "usage": usage.model_dump(mode="json", exclude_none=True)
            if hasattr(usage, "model_dump") else usage,
        "output_types": [getattr(item, "type", None) for item in response.output],
    }


class _AnthropicSession:
    def __init__(self, client: AsyncAnthropic, **kwargs: Any) -> None:
        self.client = client
        self.kwargs = kwargs
        self.messages: list[dict[str, Any]] = [
            {"role": "user", "content": json.dumps(kwargs.pop("payload"), ensure_ascii=False)}
        ]

    async def next(self, results: list[ToolResult] | None = None) -> ModelTurn:
        if results:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.call_id,
                            "content": item.content,
                            "is_error": item.is_error,
                        }
                        for item in results
                    ],
                }
            )
        response = await self.client.messages.create(
            model=self.kwargs["model"],
            max_tokens=self.kwargs["max_output_tokens"],
            system=self.kwargs["instructions"],
            messages=self.messages,
            tools=[_anthropic_tool(t) for t in self.kwargs["tools"]],
            tool_choice={"type": "auto", "disable_parallel_tool_use": False},
        )
        content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
        self.messages.append({"role": "assistant", "content": content})
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        calls = [
            ToolCall(block.id, block.name, dict(block.input))
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        return ModelTurn(text=text, tool_calls=calls)


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)

    async def structured(self, **kwargs: Any) -> dict[str, Any]:
        tool = kwargs["tool"]
        response = await self.client.messages.create(
            model=kwargs["model"],
            max_tokens=kwargs["max_output_tokens"],
            system=kwargs["instructions"],
            messages=[
                {"role": "user", "content": json.dumps(kwargs["payload"], ensure_ascii=False)}
            ],
            tools=[_anthropic_tool(tool)],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                return dict(block.input)
        raise RuntimeError(f"Anthropic did not call required tool: {tool['name']}")

    def tool_session(self, **kwargs: Any) -> _AnthropicSession:
        return _AnthropicSession(self.client, **kwargs)

    async def close(self) -> None:
        await self.client.close()


class _OpenAISession:
    def __init__(self, client: AsyncOpenAI, **kwargs: Any) -> None:
        self.client = client
        self.kwargs = kwargs
        self.trace_hook: Callable[[str, dict[str, Any]], None] | None = kwargs.pop(
            "trace_hook", None
        )
        self.input: list[Any] = [
            {"role": "user", "content": json.dumps(kwargs.pop("payload"), ensure_ascii=False)}
        ]

    async def next(self, results: list[ToolResult] | None = None) -> ModelTurn:
        if results:
            self.input.extend(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.content,
                }
                for item in results
            )
        diagnostics = ""
        for attempt in range(2):
            retry_note = [] if attempt == 0 else [{
                "role": "user",
                "content": (
                    "Retry the last requested operation. The prior function arguments were incomplete. "
                    "Return compact valid JSON, omit narrative, and keep claims and limitations concise."
                ),
            }]
            started = time.perf_counter()
            response = await self.client.responses.create(
                model=self.kwargs["model"],
                instructions=self.kwargs["instructions"],
                input=[*self.input, *retry_note],
                tools=[_openai_tool(t) for t in self.kwargs["tools"]],
                tool_choice="auto",
                parallel_tool_calls=True,
                reasoning={"effort": self.kwargs["reasoning_effort"] if attempt == 0 else "low"},
                max_output_tokens=(
                    self.kwargs["max_output_tokens"]
                    if attempt == 0
                    else min(max(self.kwargs["max_output_tokens"] * 2, 10_000), 20_000)
                ),
                store=False,
            )
            metadata = _response_metadata(
                response, (time.perf_counter() - started) * 1000
            )
            if self.trace_hook:
                self.trace_hook("provider_response", {
                    "provider": "openai", "operation": "tool_session", **metadata,
                })
            calls: list[ToolCall] = []
            parse_error: json.JSONDecodeError | None = None
            for item in response.output:
                if getattr(item, "type", None) != "function_call":
                    continue
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError as exc:
                    parse_error = exc
                    diagnostics = (
                        f"tool={item.name}, argument_chars={len(item.arguments)}, "
                        f"json_error={exc.msg} at character {exc.pos}"
                    )
                    break
                calls.append(ToolCall(item.call_id, item.name, arguments))
            incomplete = getattr(response, "status", None) == "incomplete"
            if not parse_error and not incomplete:
                self.input.extend(
                    item.model_dump(mode="json", exclude_none=True) for item in response.output
                )
                return ModelTurn(
                    text=response.output_text or "", tool_calls=calls, metadata=metadata
                )
            if incomplete:
                details = getattr(response, "incomplete_details", None)
                diagnostics = f"response_status=incomplete, details={details}"
            trace_hook = getattr(self, "trace_hook", None)
            if attempt == 0 and trace_hook:
                trace_hook("provider_retry", {
                    "provider": "openai", "operation": "tool_session",
                    "reason": diagnostics, "retry_reasoning_effort": "low",
                    "retry_max_output_tokens": min(
                        max(self.kwargs["max_output_tokens"] * 2, 10_000), 20_000
                    ),
                })
        raise RuntimeError(
            "OpenAI returned incomplete function-call output after one compact retry: " + diagnostics
        )


class OpenAIProvider:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.trace_hook: Callable[[str, dict[str, Any]], None] | None = None

    def set_trace_hook(self, hook: Callable[[str, dict[str, Any]], None]) -> None:
        self.trace_hook = hook

    async def structured(self, **kwargs: Any) -> dict[str, Any]:
        tool = kwargs["tool"]
        original_input = {
            "role": "user", "content": json.dumps(kwargs["payload"], ensure_ascii=False)
        }
        diagnostics = ""
        for attempt in range(2):
            model_input = [original_input]
            if attempt:
                model_input.append({
                    "role": "user",
                    "content": (
                        "Return compact valid function arguments. The prior attempt was incomplete; "
                        "omit narrative and keep arrays concise."
                    ),
                })
            started = time.perf_counter()
            response = await self.client.responses.create(
                model=kwargs["model"], instructions=kwargs["instructions"], input=model_input,
                tools=[_openai_tool(tool)],
                tool_choice={"type": "function", "name": tool["name"]},
                reasoning={"effort": kwargs["reasoning_effort"] if attempt == 0 else "low"},
                max_output_tokens=(
                    kwargs["max_output_tokens"]
                    if attempt == 0
                    else min(max(kwargs["max_output_tokens"] * 2, 8_000), 20_000)
                ),
                store=False,
            )
            metadata = _response_metadata(
                response, (time.perf_counter() - started) * 1000
            )
            trace_hook = getattr(self, "trace_hook", None)
            if trace_hook:
                trace_hook("provider_response", {
                    "provider": "openai", "operation": "structured",
                    "tool": tool["name"], **metadata,
                })
            for item in response.output:
                if getattr(item, "type", None) != "function_call" or item.name != tool["name"]:
                    continue
                try:
                    return json.loads(item.arguments)
                except json.JSONDecodeError as exc:
                    diagnostics = (
                        f"argument_chars={len(item.arguments)}, {exc.msg} at character {exc.pos}"
                    )
            if getattr(response, "status", None) == "incomplete":
                diagnostics = f"response_status=incomplete, details={response.incomplete_details}"
            trace_hook = getattr(self, "trace_hook", None)
            if attempt == 0 and trace_hook:
                trace_hook("provider_retry", {
                    "provider": "openai", "operation": "structured",
                    "tool": tool["name"], "reason": diagnostics,
                    "retry_reasoning_effort": "low",
                    "retry_max_output_tokens": min(
                        max(kwargs["max_output_tokens"] * 2, 8_000), 20_000
                    ),
                })
        raise RuntimeError(
            f"OpenAI did not return complete valid arguments for {tool['name']} after one retry: "
            f"{diagnostics or 'required tool call missing'}"
        )

    def tool_session(self, **kwargs: Any) -> _OpenAISession:
        kwargs["trace_hook"] = getattr(self, "trace_hook", None)
        return _OpenAISession(self.client, **kwargs)

    async def close(self) -> None:
        await self.client.close()


def create_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider.")
        return OpenAIProvider(settings.openai_api_key)
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic provider.")
    return AnthropicProvider(settings.anthropic_api_key)
