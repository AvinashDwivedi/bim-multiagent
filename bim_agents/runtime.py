from __future__ import annotations

from .anthropic_agent import BIMAgent, EventSink
from .contracts import AnswerReport


async def answer_safely(
    agent: BIMAgent,
    *,
    question: str,
    client_id: str,
    project_id: str,
    sink: EventSink | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    evaluation_run_id: str | None = None,
    evaluation_case_index: int | None = None,
) -> AnswerReport:
    try:
        correlation = {
            key: value for key, value in {
                "request_id": request_id,
                "evaluation_run_id": evaluation_run_id,
                "evaluation_case_index": evaluation_case_index,
            }.items() if value is not None
        }
        return await agent.answer(
            question=question,
            client_id=client_id,
            project_id=project_id,
            sink=sink,
            session_id=session_id,
            **correlation,
        )
    except Exception as exc:
        agent.record_pipeline_error(exc)
        return AnswerReport(
            answer=(
                "The BIM investigation could not complete. No project fact was asserted because the "
                "available evidence pipeline failed."
            ),
            verification_status="error",
            limitations=[f"{type(exc).__name__}: {str(exc)[:500]}"],
            stages_used=[],
            artifact_ids=[],
            investigation_trace=["The request ended with a controlled pipeline error."],
            failure_categories=["pipeline_error"],
            semantic_checks=[],
        )
