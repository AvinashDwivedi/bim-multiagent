"""Hierarchical, read-only multi-agent orchestration for BIM questions.

Runtime imports stay lazy so contracts and safety policy can be tested without loading
the network-facing Agents SDK.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .registry import BimAgentRegistry


def build_agent_registry(*args: Any, **kwargs: Any):
    from .registry import build_agent_registry as _build

    return _build(*args, **kwargs)


async def answer_bim_question(*args: Any, **kwargs: Any):
    from .runtime import answer_bim_question as _answer

    return await _answer(*args, **kwargs)


__all__ = ["answer_bim_question", "build_agent_registry"]
