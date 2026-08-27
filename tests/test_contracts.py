from uuid import uuid4

import pytest
from pydantic import ValidationError

from bim_agents.contracts import AnswerReport, ChatRequest


def test_chat_request_validates_scope_ids():
    request = ChatRequest(
        question="Count walls", client_id=uuid4(), project_id=uuid4(),
        request_id="request-1", evaluation_run_id="run-1", evaluation_case_index=7,
    )
    assert request.question == "Count walls"
    assert request.evaluation_case_index == 7


def test_chat_request_rejects_invalid_scope():
    with pytest.raises(ValidationError):
        ChatRequest(question="Count walls", client_id="not-a-uuid", project_id=uuid4())


def test_answer_contract_matches_evaluator_fields():
    report = AnswerReport(answer="Five.", verification_status="verified")
    assert set(report.model_dump()) == {
        "answer",
        "verification_status",
        "limitations",
        "stages_used",
        "artifact_ids",
        "investigation_trace",
        "failure_categories",
        "semantic_checks",
    }
