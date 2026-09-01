from __future__ import annotations


class AgentRunCancelled(RuntimeError):
    """Raised when a caller cooperatively cancels an in-flight agent run."""


def is_retryable_api_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}:
        return True
    return getattr(exc, "status_code", None) in {408, 409, 429, 500, 502, 503, 504}
