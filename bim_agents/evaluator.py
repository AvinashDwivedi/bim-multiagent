from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Awaitable, Callable, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter

from .runtime import answer_bim_question


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1, description="Reference/expected answer.")


class EvaluationJudgment(BaseModel):
    score: float = Field(ge=0, le=1)
    passed: bool
    correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    reason: str


class CaseResult(BaseModel):
    index: int
    question: str
    expected_answer: str
    actual_answer: str = ""
    pipeline_status: str = "error"
    elapsed_seconds: float = 0
    exact_match: bool = False
    score: float = 0
    passed: bool = False
    correctness: float = 0
    completeness: float = 0
    groundedness: float = 0
    reason: str = ""
    error: str | None = None


class EvaluationSummary(BaseModel):
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    average_score: float
    average_elapsed_seconds: float


class EvaluationReport(BaseModel):
    judge_mode: Literal["semantic", "exact"]
    judge_model: str | None = None
    pass_threshold: float
    summary: EvaluationSummary
    results: list[CaseResult]


Answerer = Callable[[str], Awaitable[object]]
Judge = Callable[[EvaluationCase, str, str, float], Awaitable[EvaluationJudgment]]


def load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and set(payload) >= {"question", "answer"}:
        records = [payload]
    elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        records = payload["cases"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and all(isinstance(value, str) for value in payload.values()):
        records = [{"question": question, "answer": answer} for question, answer in payload.items()]
    else:
        raise ValueError(
            "Evaluation JSON must be a {question, answer} object, a list of those objects, "
            "a {cases: [...]} object, or a question-to-answer mapping."
        )
    cases = TypeAdapter(list[EvaluationCase]).validate_python(records)
    if not cases:
        raise ValueError("Evaluation dataset cannot be empty.")
    return cases


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


async def semantic_judge(
    case: EvaluationCase,
    actual_answer: str,
    pipeline_status: str,
    threshold: float,
    *,
    model: str,
) -> EvaluationJudgment:
    prompt = f"""Evaluate a BIM system answer against the reference answer.

Question:
{case.question}

Reference answer:
{case.answer}

System answer:
{actual_answer}

Pipeline status: {pipeline_status}

Judge semantic equivalence, not wording. Penalize wrong numbers, units, scope, measurement basis,
unsupported compliance conclusions, contradictions, and claims that confuse missing data with zero.
An honest 'cannot assess' passes only when it agrees with the reference. Set passed=true only when
score is at least {threshold}. Keep the reason concise and do not include credentials or identifiers.
"""

    def call() -> EvaluationJudgment:
        response = OpenAI().responses.parse(
            model=model,
            input=[
                {"role": "developer", "content": "You are a strict BIM evaluation grader."},
                {"role": "user", "content": prompt},
            ],
            text_format=EvaluationJudgment,
        )
        judgment = response.output_parsed
        if judgment is None:
            raise RuntimeError("The semantic judge returned no structured result.")
        judgment.passed = judgment.score >= threshold
        return judgment

    return await asyncio.to_thread(call)


async def evaluate_cases(
    cases: list[EvaluationCase],
    *,
    judge_mode: Literal["semantic", "exact"] = "semantic",
    judge_model: str | None = None,
    pass_threshold: float = 0.8,
    timeout_seconds: float | None = None,
    answerer: Answerer = answer_bim_question,
    judge: Judge | None = None,
) -> EvaluationReport:
    if not 0 <= pass_threshold <= 1:
        raise ValueError("pass_threshold must be between 0 and 1.")
    model = judge_model or os.getenv("BIM_EVALUATOR_MODEL", "gpt-5.6-terra")
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        started = time.monotonic()
        result = CaseResult(index=index, question=case.question, expected_answer=case.answer)
        try:
            report = await answerer(case.question) if timeout_seconds is None else await asyncio.wait_for(
                answerer(case.question), timeout=timeout_seconds
            )
            result.elapsed_seconds = round(time.monotonic() - started, 3)
            result.actual_answer = str(report.answer)
            result.pipeline_status = str(report.verification_status)
            result.exact_match = _normalized(result.actual_answer) == _normalized(case.answer)
            if judge_mode == "exact" or result.exact_match:
                score = 1.0 if result.exact_match else 0.0
                judgment = EvaluationJudgment(
                    score=score,
                    passed=score >= pass_threshold,
                    correctness=score,
                    completeness=score,
                    groundedness=score,
                    reason="Normalized exact match." if result.exact_match else "Answers do not exactly match.",
                )
            else:
                judge_fn = judge or (
                    lambda c, a, s, t: semantic_judge(c, a, s, t, model=model)
                )
                judgment = await judge_fn(
                    case, result.actual_answer, result.pipeline_status, pass_threshold
                )
            for field in ("score", "passed", "correctness", "completeness", "groundedness", "reason"):
                setattr(result, field, getattr(judgment, field))
        except Exception as exc:
            result.elapsed_seconds = round(time.monotonic() - started, 3)
            result.error = type(exc).__name__
            result.reason = "Evaluation case could not complete."
        results.append(result)

    total = len(results)
    passed = sum(item.passed for item in results)
    errors = sum(item.error is not None for item in results)
    summary = EvaluationSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        errors=errors,
        pass_rate=round(passed / total, 4),
        average_score=round(sum(item.score for item in results) / total, 4),
        average_elapsed_seconds=round(sum(item.elapsed_seconds for item in results) / total, 3),
    )
    return EvaluationReport(
        judge_mode=judge_mode,
        judge_model=model if judge_mode == "semantic" else None,
        pass_threshold=pass_threshold,
        summary=summary,
        results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BIM answers from a question/answer JSON dataset.")
    parser.add_argument("input", type=Path, help="Path to the evaluation JSON file")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--judge-mode", choices=["semantic", "exact"], default="semantic")
    parser.add_argument("--judge-model", default=None, help="Semantic grader model")
    parser.add_argument("--pass-threshold", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=None, help="Per-question timeout in seconds")
    args = parser.parse_args()

    try:
        report = asyncio.run(evaluate_cases(
            load_cases(args.input),
            judge_mode=args.judge_mode,
            judge_model=args.judge_model,
            pass_threshold=args.pass_threshold,
            timeout_seconds=args.timeout,
        ))
    except Exception as exc:
        raise SystemExit(f"Evaluator failed: {type(exc).__name__}: {exc}") from exc
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
