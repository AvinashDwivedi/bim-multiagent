"""Project-scoped multi-agent Neo4j calculation system for BIM questions."""

from typing import Any


async def answer_bim_question(*args: Any, **kwargs: Any):
    from .runtime import answer_bim_question as _answer

    return await _answer(*args, **kwargs)


def build_agent_registry(*args: Any, **kwargs: Any):
    from .registry import build_agent_registry as _build

    return _build(*args, **kwargs)


__all__ = ["answer_bim_question", "build_agent_registry"]
