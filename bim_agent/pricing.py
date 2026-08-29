from __future__ import annotations

import re
from typing import Any


PRICING_AS_OF = "2026-08-29"
PRICING_SOURCES = {
    "gpt-5.4": "https://developers.openai.com/api/docs/models/gpt-5.4",
    "gpt-5.6-sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
}

# Standard text-token list prices in USD per one million tokens. Runtime
# overrides can replace any model entry through BIM_MODEL_PRICING_JSON.
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "cache_write": 2.50,
        "output": 15.00,
        "large_context_threshold": 272_000,
        "large_context_input_multiplier": 2.0,
        "large_context_output_multiplier": 1.5,
    },
    "gpt-5.6-sol": {
        "input": 4.00,
        "cached_input": 0.40,
        "cache_write": 5.00,
        "output": 20.00,
        "large_context_threshold": 272_000,
        "large_context_input_multiplier": 2.0,
        "large_context_output_multiplier": 1.5,
    },
    "gpt-5.6": {
        "input": 4.00,
        "cached_input": 0.40,
        "cache_write": 5.00,
        "output": 20.00,
        "large_context_threshold": 272_000,
        "large_context_input_multiplier": 2.0,
        "large_context_output_multiplier": 1.5,
    },
}


def response_usage_record(
    response: Any,
    *,
    configured_model: str,
    purpose: str,
    excluded_tool_fees: list[str] | None = None,
) -> dict[str, Any]:
    usage = _value(response, "usage")
    actual_model = str(_value(response, "model") or configured_model)
    record = {
        "response_id": str(_value(response, "id") or ""),
        "model": actual_model,
        "purpose": purpose,
        "usage_available": usage is not None,
        "excluded_tool_fees": list(excluded_tool_fees or []),
    }
    if usage is None:
        return record
    details = _value(usage, "input_tokens_details")
    cached = _integer(_value(details, "cached_tokens"))
    cache_write = _integer(
        _value(details, "cache_write_tokens")
        or _value(details, "input_cache_write_tokens")
        or _value(usage, "input_cache_write_tokens")
    )
    input_tokens = _integer(_value(usage, "input_tokens"))
    return {
        **record,
        "input_tokens": input_tokens,
        "cached_input_tokens": min(input_tokens, cached),
        "cache_write_input_tokens": min(input_tokens, cache_write),
        "output_tokens": _integer(_value(usage, "output_tokens")),
        "total_tokens": _integer(_value(usage, "total_tokens")) or (
            input_tokens + _integer(_value(usage, "output_tokens"))
        ),
    }


def estimate_run_cost(
    records: list[dict[str, Any]],
    *,
    pricing_overrides: dict[str, dict[str, float]] | None = None,
    extra_excluded_tool_fees: set[str] | None = None,
) -> dict[str, Any]:
    catalog = {key: dict(value) for key, value in DEFAULT_MODEL_PRICING.items()}
    for model, rates in (pricing_overrides or {}).items():
        merged = dict(catalog.get(str(model), {}))
        merged.update({str(key): float(value) for key, value in rates.items()})
        catalog[str(model)] = merged

    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    request_breakdown: list[dict[str, Any]] = []
    priced_requests = 0
    usage_requests = 0
    estimated_cost = 0.0
    excluded_fees = set(extra_excluded_tool_fees or set())

    for record in records:
        excluded_fees.update(str(item) for item in record.get("excluded_tool_fees", []))
        if not record.get("usage_available"):
            request_breakdown.append({**record, "estimated_cost_usd": None, "pricing_status": "usage_unavailable"})
            continue
        usage_requests += 1
        for key in totals:
            totals[key] += _integer(record.get(key))
        rates = _rates_for_model(str(record.get("model", "")), catalog)
        if rates is None:
            request_breakdown.append({**record, "estimated_cost_usd": None, "pricing_status": "model_unpriced"})
            continue

        input_tokens = _integer(record.get("input_tokens"))
        cached_tokens = min(input_tokens, _integer(record.get("cached_input_tokens")))
        cache_write_tokens = min(
            max(0, input_tokens - cached_tokens),
            _integer(record.get("cache_write_input_tokens")),
        )
        uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
        output_tokens = _integer(record.get("output_tokens"))
        large_context = input_tokens > int(rates.get("large_context_threshold", 10**18))
        input_multiplier = rates.get("large_context_input_multiplier", 1.0) if large_context else 1.0
        output_multiplier = rates.get("large_context_output_multiplier", 1.0) if large_context else 1.0
        request_cost = (
            uncached_tokens * rates["input"] * input_multiplier
            + cached_tokens * rates.get("cached_input", rates["input"]) * input_multiplier
            + cache_write_tokens * rates.get("cache_write", rates["input"]) * input_multiplier
            + output_tokens * rates["output"] * output_multiplier
        ) / 1_000_000
        priced_requests += 1
        estimated_cost += request_cost
        request_breakdown.append({
            **record,
            "uncached_input_tokens": uncached_tokens,
            "large_context_pricing": large_context,
            "estimated_cost_usd": round(request_cost, 10),
            "pricing_status": "calculated",
        })

    all_priced = bool(records) and priced_requests == len(records)
    any_priced = priced_requests > 0
    status = "calculated" if all_priced else ("partial" if any_priced else "unavailable")
    return {
        "currency": "USD",
        "status": status,
        "estimated_cost_usd": round(estimated_cost, 10) if any_priced else None,
        "is_complete": all_priced and not excluded_fees,
        "api_requests": len(records),
        "requests_with_usage": usage_requests,
        "priced_requests": priced_requests,
        "tokens": totals,
        "request_breakdown": request_breakdown,
        "excluded_tool_fees": sorted(excluded_fees),
        "pricing_basis": "Standard API text-token list prices; taxes, contractual discounts, Scale Tier, Batch, and Flex pricing are not applied.",
        "pricing_as_of": PRICING_AS_OF,
        "pricing_sources": PRICING_SOURCES,
    }


def _rates_for_model(
    model: str, catalog: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    if model in catalog:
        rates = catalog[model]
        return rates if {"input", "output"} <= rates.keys() else None
    for key, rates in catalog.items():
        if re.fullmatch(re.escape(key) + r"-\d{4}-\d{2}-\d{2}", model):
            return rates if {"input", "output"} <= rates.keys() else None
    return None


def _value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
