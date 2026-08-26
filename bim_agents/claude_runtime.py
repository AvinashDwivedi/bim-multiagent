from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, get_type_hints

from pydantic import BaseModel, create_model


TContext = TypeVar("TContext")


class MaxTurnsExceeded(RuntimeError):
    """The Claude agent reached its configured turn limit."""


class ModelTimeoutError(RuntimeError):
    """The Anthropic request timed out."""


class ModelRateLimitError(RuntimeError):
    """Anthropic rate-limited the request."""


class ModelConnectionError(RuntimeError):
    """Claude Code or the Anthropic API could not be reached."""


class ModelAccessError(RuntimeError):
    """Anthropic rejected the configured credentials or billing state."""


@dataclass(frozen=True)
class ModelSettings:
    effort: str = "high"
    parallel_tool_calls: bool = False


@dataclass
class Agent:
    name: str
    instructions: str
    model: str
    model_settings: ModelSettings
    tools: list["FunctionTool"] = field(default_factory=list)
    output_type: type[BaseModel] | None = None


@dataclass(frozen=True)
class RunContextWrapper(Generic[TContext]):
    context: TContext


class RunHooks(Generic[TContext]):
    """Lifecycle interface implemented by the BIM observability hooks."""


@dataclass(frozen=True)
class RunResult:
    final_output: Any


@dataclass
class FunctionTool:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    _function: Callable[..., Any]
    _arguments_model: type[BaseModel]
    _context_parameter: str | None

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
        validated = self._arguments_model.model_validate(arguments)
        kwargs = {
            name: getattr(validated, name)
            for name in self._arguments_model.model_fields
        }
        if self._context_parameter is not None:
            kwargs[self._context_parameter] = RunContextWrapper(context)
        result = self._function(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def function_tool(function: Callable[..., Any]) -> FunctionTool:
    """Expose a typed Python function as an in-process Claude MCP tool."""
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    fields: dict[str, tuple[Any, Any]] = {}
    context_parameter: str | None = None
    for name, parameter in signature.parameters.items():
        annotation = hints.get(name, parameter.annotation)
        if name in {"ctx", "context"} and get_origin_name(annotation) == "RunContextWrapper":
            context_parameter = name
            continue
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (annotation, default)
    arguments_model = create_model(f"{function.__name__.title()}Arguments", **fields)
    return FunctionTool(
        name=function.__name__,
        description=(inspect.getdoc(function) or function.__name__).strip(),
        params_json_schema=arguments_model.model_json_schema(),
        _function=function,
        _arguments_model=arguments_model,
        _context_parameter=context_parameter,
    )


def get_origin_name(annotation: Any) -> str:
    origin = getattr(annotation, "__origin__", None)
    return getattr(origin or annotation, "__name__", "")


def _strict_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare Pydantic JSON Schema for Claude structured output.

    Claude limits the total number of optional structured-output fields. The
    BIM contracts deliberately contain many defaults, so the wire schema makes
    every property explicit while Pydantic remains the authoritative validator.
    """
    try:
        from anthropic import transform_schema

        prepared = transform_schema(schema)
    except (ImportError, AttributeError):
        prepared = json.loads(json.dumps(schema))

    unsupported = {
        "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "minItems", "maxItems", "multipleOf",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in list(value):
                if key in unsupported:
                    value.pop(key, None)
            properties = value.get("properties")
            if value.get("type") == "object" and isinstance(properties, dict):
                value["additionalProperties"] = False
                value["required"] = list(properties)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(prepared)
    return prepared


def ensure_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper used by schema-regression tests."""
    return _strict_output_schema(schema)


def _error_from_status(status: int | None, message: str) -> Exception:
    normalized = message.casefold()
    if status in {401, 402, 403} or any(
        marker in normalized
        for marker in ("authentication", "billing", "credit balance", "api key")
    ):
        return ModelAccessError(message)
    if status == 429:
        return ModelRateLimitError(message)
    if status in {408, 504}:
        return ModelTimeoutError(message)
    return ModelConnectionError(message)


def _raise_failed_result(result: Any) -> None:
    reason = getattr(result, "terminal_reason", None)
    subtype = getattr(result, "subtype", "")
    message = getattr(result, "result", None) or "; ".join(
        getattr(result, "errors", None) or []
    ) or f"Claude agent failed: {subtype or reason or 'unknown error'}"
    if reason == "max_turns" or subtype == "error_max_turns":
        raise MaxTurnsExceeded(message)
    raise _error_from_status(getattr(result, "api_error_status", None), message)


class Runner:
    @staticmethod
    async def run(
        agent: Agent,
        input: str,
        *,
        context: TContext,
        max_turns: int,
        hooks: RunHooks[TContext] | None = None,
    ) -> RunResult:
        """Run one isolated worker through Claude Code's Agent SDK."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ModelAccessError(
                "Missing required environment variable: ANTHROPIC_API_KEY"
            )

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            CLIConnectionError,
            ProcessError,
            ResultError,
            ResultMessage,
            create_sdk_mcp_server,
            tool as claude_tool,
        )

        wrapper = RunContextWrapper(context)
        if hooks is not None:
            await hooks.on_agent_start(wrapper, agent)

        fatal_tool_error: list[Exception] = []
        sdk_tools = []
        for function in agent.tools:
            async def handler(
                arguments: dict[str, Any], *, _function: FunctionTool = function,
            ) -> dict[str, Any]:
                if fatal_tool_error:
                    return {
                        "content": [{"type": "text", "text": str(fatal_tool_error[0])}],
                        "is_error": True,
                    }
                try:
                    if hooks is not None:
                        await hooks.on_tool_start(wrapper, agent, _function)
                    result = await _function.invoke(context, arguments)
                    if hooks is not None:
                        await hooks.on_tool_end(wrapper, agent, _function, result)
                    text = result if isinstance(result, str) else json.dumps(
                        result, ensure_ascii=False, default=str,
                    )
                    return {"content": [{"type": "text", "text": text}]}
                except Exception as exc:
                    if "budget" in str(exc).casefold():
                        fatal_tool_error.append(exc)
                    return {
                        "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                        "is_error": True,
                    }

            sdk_tools.append(
                claude_tool(
                    function.name,
                    function.description,
                    function.params_json_schema,
                )(handler)
            )

        server_name = "bim"
        mcp_servers = {}
        allowed_tools = []
        if sdk_tools:
            mcp_servers[server_name] = create_sdk_mcp_server(
                server_name, version="1.0.0", tools=sdk_tools,
            )
            allowed_tools = [f"mcp__{server_name}__{item.name}" for item in agent.tools]

        options = ClaudeAgentOptions(
            model=agent.model,
            system_prompt=agent.instructions,
            tools=[],
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            allowed_tools=allowed_tools,
            disallowed_tools=[
                "Agent", "Bash", "Edit", "Glob", "Grep", "Read", "WebFetch",
                "WebSearch", "Write",
            ],
            permission_mode="dontAsk",
            setting_sources=[],
            skills=[],
            max_turns=max_turns,
            effort=agent.model_settings.effort,
            output_format=(
                {
                    "type": "json_schema",
                    "schema": _strict_output_schema(agent.output_type.model_json_schema()),
                }
                if agent.output_type is not None else None
            ),
            cwd=Path.cwd(),
            env={"CLAUDE_AGENT_SDK_CLIENT_APP": "bim-multiagent/1.0"},
        )

        final_result = None
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(input)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage) and hooks is not None:
                        await hooks.on_llm_start(
                            wrapper, agent, agent.instructions, [input],
                        )
                    if isinstance(message, ResultMessage):
                        final_result = message
        except ResultError as exc:
            if exc.terminal_reason == "max_turns" or exc.subtype == "error_max_turns":
                raise MaxTurnsExceeded(str(exc)) from exc
            raise _error_from_status(exc.api_error_status, str(exc)) from exc
        except CLIConnectionError as exc:
            raise ModelConnectionError(str(exc)) from exc
        except ProcessError as exc:
            raise ModelConnectionError(str(exc)) from exc

        if fatal_tool_error:
            raise fatal_tool_error[0]
        if final_result is None:
            raise ModelConnectionError("Claude Agent SDK returned no terminal result.")
        if final_result.is_error:
            _raise_failed_result(final_result)

        output: Any = final_result.structured_output
        if output is None:
            output = final_result.result
            if agent.output_type is not None and isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    pass
        if agent.output_type is not None:
            output = agent.output_type.model_validate(output)
        if hooks is not None:
            await hooks.on_agent_end(wrapper, agent, output)
        return RunResult(final_output=output)


@contextmanager
def trace(*_args: Any, **_kwargs: Any):
    """Claude Code emits native telemetry; retain the former call-site shape."""
    yield
