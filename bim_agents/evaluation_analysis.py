"""Privacy-safe aggregate analysis for evaluator reports.

The evaluator is deliberately a separate application and its report formats have
changed over time.  This module accepts the common JSON shapes (a list of cases,
or a mapping containing ``results``/``evaluations``/``runs``) and keeps only
aggregate, operational signals.  It never returns or stores questions, expected
answers, responses, answer values, scope identifiers, or free-form explanations.

The functions are pure: a path is read for the duration of the call and no
report is written to disk.  The returned dictionaries are suitable for a
dashboard or a small, deliberately safe persisted summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


_CONTAINER_KEYS = (
    "results", "evaluations", "evaluation_results", "runs", "cases", "items",
    "records", "entries", "report", "reports",
)
_EVALUATION_KEYS = (
    "evaluation", "judgement", "judgment", "grading", "grade", "assessment",
    "review", "review_result",
)
_SEMANTIC_CHECK_NAMES = {
    "replay_stability", "authorized_scope", "identity_integrity",
    "constraint_binding", "counting_unit", "boundary_exactness",
    "source_deduplication", "classification_purity", "constraint_coverage",
    "population_coverage", "relationship_binding", "measurement_plausibility",
    "entity_grain_matches", "measurement_basis_matches", "population_complete",
    "planned_actual_distinguished", "absence_semantics_correct",
    "projection_answers_question", "requested_outputs_present",
}
_SPECIALISTS = {
    "auto", "quantity", "relationship", "geometry", "requirements",
    "schema", "schema_scout", "schema_mapping", "query", "query_worker",
    "task_architect", "architect", "verification", "replay", "curator",
}
_STAGES = {
    "understand", "explore", "analyze", "act", "observe", "verify", "respond",
    "blocked", "architecture", "planning", "discovery", "schema_mapping",
    "query", "calculation", "geometry", "requirements", "replay",
    "verification", "completion", "curation", "rendering", "merge",
    "capability_guard", "cypher_query_handler", "knowledge_curator",
    "query_planner", "schema_mapper", "verifier",
}
_PIPELINE_STATUS_ALIASES = {
    "success": "verified", "passed": "verified", "pass": "verified",
    "verified": "verified", "ready_for_verification": "ready_for_verification",
    "insufficient": "insufficient_evidence", "insufficient_evidence": "insufficient_evidence",
    "unsupported": "insufficient_evidence", "conflict": "conflict",
    "failed": "failed", "failure": "failed", "error": "error",
    "timeout": "timeout", "timed_out": "timeout", "cancelled": "cancelled",
    "canceled": "cancelled", "blocked": "blocked",
    "query_completed": "verified", "completed": "verified",
    "partial_completed": "failed", "turn_limit": "timeout",
    "merge_rejected": "failed",
}
_CATEGORY_ALIASES = {
    "semantic": "semantic_check_failed", "semantic_check": "semantic_check_failed",
    "semantic_check_failed": "semantic_check_failed", "incorrect": "evaluator_incorrect",
    "wrong": "evaluator_incorrect", "wrong_answer": "evaluator_incorrect",
    "evaluator_incorrect": "evaluator_incorrect", "insufficient": "insufficient_evidence",
    "insufficient_evidence": "insufficient_evidence", "conflict": "conflict",
    "timeout": "timeout", "timed_out": "timeout", "error": "pipeline_error",
    "pipeline_error": "pipeline_error", "malformed": "malformed_evaluation",
    "malformed_evaluation": "malformed_evaluation", "scope": "scope_failure",
    "scope_failure": "scope_failure", "retrieval": "retrieval_failure",
    "retrieval_failure": "retrieval_failure", "mapping": "mapping_failure",
    "mapping_failure": "mapping_failure", "query": "query_failure",
    "query_failure": "query_failure", "verification": "verification_failure",
    "verification_failure": "verification_failure", "other": "other",
    "workstream_turn_limit": "timeout", "workstream_error": "pipeline_error",
    "workstream_merge_rejected": "pipeline_error",
    "workstream_partial_completed": "pipeline_error",
}


def _as_payload(source: Any) -> Any:
    """Load JSON from a path/string or accept an already decoded payload."""
    if isinstance(source, Path):
        return json.loads(source.read_text(encoding="utf-8"))
    if isinstance(source, (Mapping, list, tuple)):
        return source
    if isinstance(source, str):
        text = source.lstrip()
        # JSON documents begin with an object/array; avoid treating a large
        # inline document as a Windows path (which can raise ``OSError``).
        if text.startswith(("{", "[")):
            return json.loads(source)
        candidate = Path(source)
        try:
            if candidate.exists() and candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except OSError:
            pass
        return json.loads(source)
    raise TypeError("report must be a JSON path, JSON string, mapping, or list")


def _dicts(value: Any):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _dicts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _dicts(item)


def _case_like(item: Mapping[str, Any]) -> bool:
    keys = {str(k).lower() for k in item}
    markers = {
        "question", "expected", "expected_answer", "actual", "actual_answer",
        "answer", "correct", "is_correct", "passed", "semantic_checks",
        "verification_status", "pipeline_status", "workstream_diagnostics",
    }
    return bool(keys & markers)


def _cases(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    # Prefer an explicit report collection.  This avoids treating summary
    # metadata next to the collection as another evaluation case.
    for key in _CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    if _case_like(payload):
        return [payload]
    nested = [item for key in _CONTAINER_KEYS if isinstance(payload.get(key), Mapping)
              for item in _cases(payload[key])]
    return nested or [payload]


def _normal(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _safe_category(value: Any) -> str | None:
    token = _normal(value)
    if not token:
        return None
    return _CATEGORY_ALIASES.get(token, "other")


def _safe_status(value: Any) -> str:
    token = _normal(value)
    return _PIPELINE_STATUS_ALIASES.get(token, "unknown")


def _safe_check(value: Any) -> str | None:
    token = _normal(value)
    return token if token in _SEMANTIC_CHECK_NAMES else None


def _safe_specialist(value: Any) -> str | None:
    token = _normal(value)
    return token if token in _SPECIALISTS else None


def _safe_stage(value: Any) -> str | None:
    token = _normal(value)
    return token if token in _STAGES else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = _normal(value)
        if token in {"true", "yes", "pass", "passed", "correct", "success"}:
            return True
        if token in {"false", "no", "fail", "failed", "incorrect", "wrong"}:
            return False
    return None


def _explicit_outcome(case: Mapping[str, Any]) -> str:
    # Correctness belongs to the evaluator envelope, not semantic check items.
    # Check the case envelope and explicitly named evaluation envelope first;
    # this prevents a check's own ``passed`` flag from becoming a case grade.
    nodes: list[Mapping[str, Any]] = [case]
    for key in _EVALUATION_KEYS:
        value = case.get(key)
        if isinstance(value, Mapping):
            nodes.extend(_dicts(value))
    for node in nodes:
        for key in ("correct", "is_correct", "answer_correct", "passed", "pass", "success"):
            if key in node:
                value = _bool(node[key])
                if value is not None:
                    return "passed" if value else "failed"
        for key in ("outcome", "result", "grade", "judgement", "judgment"):
            if key in node and isinstance(node[key], str):
                token = _normal(node[key])
                if token in {"pass", "passed", "correct", "success"}:
                    return "passed"
                if token in {"fail", "failed", "incorrect", "wrong"}:
                    return "failed"
    # Some older reports put correctness one level below a named result object.
    # Still skip semantic-check and workstream records, whose ``passed``/status
    # fields are not evaluator grades.
    for node in _dicts(case):
        if ("name" in node and "passed" in node) or ("specialist" in node and "status" in node):
            continue
        for key in ("correct", "is_correct", "answer_correct"):
            if key in node:
                value = _bool(node[key])
                if value is not None:
                    return "passed" if value else "failed"
    return "unknown"


def _pipeline_status(case: Mapping[str, Any]) -> str:
    for node in _dicts(case):
        for key in ("verification_status", "pipeline_status", "completion_status", "pipeline_state"):
            if key in node:
                status = _safe_status(node[key])
                if status != "unknown":
                    return status
        # Generic status is only used in pipeline/workstream-shaped objects.
        if "status" in node and any(k in node for k in ("workstream_diagnostics", "semantic_checks", "claims")):
            status = _safe_status(node["status"])
            if status != "unknown":
                return status
    return "unknown"


def _semantic_checks(case: Mapping[str, Any]) -> dict[str, Counter]:
    found: dict[str, Counter] = defaultdict(Counter)
    for node in _dicts(case):
        checks = node.get("semantic_checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            name = _safe_check(check.get("name") or check.get("check"))
            if not name:
                continue
            passed = _bool(check.get("passed", check.get("pass", check.get("ok"))))
            found[name]["passed" if passed is True else "failed" if passed is False else "unknown"] += 1
    return found


def _diagnostics(case: Mapping[str, Any]) -> tuple[Counter, Counter]:
    specialists: Counter = Counter()
    stages: Counter = Counter()
    for node in _dicts(case):
        diagnostics = node.get("workstream_diagnostics")
        if not isinstance(diagnostics, list):
            continue
        for item in diagnostics:
            if not isinstance(item, Mapping):
                continue
            specialist = _safe_specialist(item.get("specialist") or item.get("role"))
            status = _safe_status(item.get("status"))
            # A diagnostic is implicated when it was not a normal completion.
            if specialist and status not in {"verified", "unknown"}:
                specialists[specialist] += 1
            elif specialist and item.get("failure_categories"):
                specialists[specialist] += 1
            stage = _safe_stage(item.get("stage") or item.get("failed_stage"))
            if stage and status not in {"verified", "unknown"}:
                stages[stage] += 1
    return stages, specialists


def _case_record(case: Mapping[str, Any]) -> dict[str, Any]:
    checks = _semantic_checks(case)
    failed_checks = {name for name, counts in checks.items() if counts["failed"]}
    status = _pipeline_status(case)
    outcome = _explicit_outcome(case)
    categories: set[str] = set()
    for node in _dicts(case):
        for key in ("failure_category", "category", "failure_categories", "failure_types"):
            value = node.get(key)
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                category = _safe_category(item)
                if category:
                    categories.add(category)
    if failed_checks:
        categories.add("semantic_check_failed")
    if outcome == "failed" and not categories:
        categories.add("evaluator_incorrect")
    if status == "insufficient_evidence":
        categories.add("insufficient_evidence")
    elif status == "conflict":
        categories.add("conflict")
    elif status == "timeout":
        categories.add("timeout")
    elif status in {"failed", "error"}:
        categories.add("pipeline_error")
    stages, specialists = _diagnostics(case)
    # Explicit stage lists are safe only after allow-listing.  They describe
    # pipeline shape, never project content.
    for node in _dicts(case):
        if not ("status" in node and "specialist" in node):
            for key in ("implicated_stage", "failed_stage", "stage"):
                stage = _safe_stage(node.get(key))
                if stage:
                    stages[stage] += 1
        value = node.get("stages_used")
        if isinstance(value, list) and (outcome == "failed" or categories or status not in {"verified", "unknown"}):
            for item in value:
                stage = _safe_stage(item)
                if stage:
                    stages[stage] += 1
        specialist = _safe_specialist(node.get("specialist") or node.get("role"))
        if specialist and (outcome == "failed" or categories) and not (
            "status" in node and "specialist" in node
        ):
            specialists[specialist] += 1
    return {"outcome": outcome, "status": status, "categories": categories,
            "checks": checks, "stages": stages, "specialists": specialists}


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def analyze_report(source: Any) -> dict[str, Any]:
    """Return aggregate-only analysis for an evaluator report.

    The input may be a decoded JSON value, a JSON string, or a path.  The output
    intentionally contains no case-level identifiers or free-form report text.
    """
    payload = _as_payload(source)
    records = [_case_record(case) for case in _cases(payload)]
    outcomes = Counter(record["outcome"] for record in records)
    statuses = Counter(record["status"] for record in records)
    categories: Counter = Counter()
    stages: Counter = Counter()
    specialists: Counter = Counter()
    checks: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        categories.update(record["categories"])
        stages.update(record["stages"])
        specialists.update(record["specialists"])
        for name, values in record["checks"].items():
            checks[name].update(values)
    total = len(records)
    passed = outcomes["passed"]
    judged = passed + outcomes["failed"]
    return {
        "schema_version": 1,
        "total_cases": total,
        "outcomes": _counter_dict(outcomes),
        "passed_cases": passed,
        "failed_cases": outcomes["failed"],
        "unknown_cases": outcomes["unknown"],
        "pass_rate": round(passed / judged, 6) if judged else None,
        "pipeline_status_counts": _counter_dict(statuses),
        "failure_category_counts": _counter_dict(categories),
        "semantic_check_counts": {
            name: _counter_dict(values) for name, values in sorted(checks.items())
        },
        "implicated_stage_counts": _counter_dict(stages),
        "implicated_specialist_counts": _counter_dict(specialists),
    }


def compare_reports(current: Any, baseline: Any) -> dict[str, Any]:
    """Compare two reports using aggregate counts and anonymous outcome transitions."""
    current_payload, baseline_payload = _as_payload(current), _as_payload(baseline)
    current_cases = [_case_record(case) for case in _cases(current_payload)]
    baseline_cases = [_case_record(case) for case in _cases(baseline_payload)]
    current_summary = analyze_report(current_payload)
    baseline_summary = analyze_report(baseline_payload)

    def delta(field: str) -> int:
        return current_summary[field] - baseline_summary[field]

    transitions = Counter()
    for before, after in zip(baseline_cases, current_cases):
        transitions[f"{before['outcome']}_to_{after['outcome']}"] += 1
    current_categories = Counter(current_summary["failure_category_counts"])
    baseline_categories = Counter(baseline_summary["failure_category_counts"])
    current_checks = current_summary["semantic_check_counts"]
    baseline_checks = baseline_summary["semantic_check_counts"]
    check_deltas: dict[str, dict[str, int]] = {}
    for name in sorted(set(current_checks) | set(baseline_checks)):
        current_values, baseline_values = current_checks.get(name, {}), baseline_checks.get(name, {})
        values = {
            key: current_values.get(key, 0) - baseline_values.get(key, 0)
            for key in {"passed", "failed", "unknown"}
        }
        if any(values.values()):
            check_deltas[name] = {key: values[key] for key in sorted(values)}
    return {
        "schema_version": 1,
        "baseline": baseline_summary,
        "current": current_summary,
        "deltas": {
            "total_cases": delta("total_cases"),
            "passed_cases": delta("passed_cases"),
            "failed_cases": delta("failed_cases"),
            "unknown_cases": delta("unknown_cases"),
            "pass_rate": (
                round((current_summary["pass_rate"] or 0) - (baseline_summary["pass_rate"] or 0), 6)
                if current_summary["pass_rate"] is not None or baseline_summary["pass_rate"] is not None else None
            ),
        },
        "outcome_transitions": _counter_dict(transitions),
        "newly_failing_categories": _counter_dict(
            Counter({key: value - baseline_categories.get(key, 0)
                     for key, value in current_categories.items()
                     if value > baseline_categories.get(key, 0)})
        ),
        "resolved_failure_categories": _counter_dict(
            Counter({key: value - current_categories.get(key, 0)
                     for key, value in baseline_categories.items()
                     if value > current_categories.get(key, 0)})
        ),
        "semantic_check_deltas": check_deltas,
    }


# Descriptive aliases for callers that prefer an explicit API name.
analyze_evaluation_report = analyze_report
compare_evaluation_reports = compare_reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluator outcomes without exposing BIM answers.")
    parser.add_argument("report", type=Path, help="Evaluator JSON report")
    parser.add_argument("--baseline", type=Path, help="Optional earlier report for regression comparison")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    args = parser.parse_args(argv)
    result = compare_reports(args.report, args.baseline) if args.baseline else analyze_report(args.report)
    print(json.dumps(result, ensure_ascii=False, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
