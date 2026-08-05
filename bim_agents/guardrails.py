from __future__ import annotations

import json

from agents import GuardrailFunctionOutput, RunContextWrapper, output_guardrail

from .models import BimRunContext, Claim, SupervisorReport, VerificationReport


def _latest_verification(context: BimRunContext) -> tuple[str, dict] | None:
    for evidence_id, evidence in reversed(list(context.evidence.items())):
        if evidence.kind == "verification":
            return evidence_id, json.loads(evidence.payload)
    return None


def verification_report_from_evidence(context: BimRunContext) -> VerificationReport:
    latest = _latest_verification(context)
    if latest is None:
        return VerificationReport(
            status="insufficient_evidence",
            limitations=["The Verification Agent produced no deterministic verification evidence."],
        )
    evidence_id, payload = latest
    limitations = list(payload.get("classification_issues") or [])
    if payload.get("verified") is True:
        count = int(payload["count"])
        element_term = str(payload.get("element_term") or "elements")
        level = str(payload.get("requested_level") or "requested level")
        claim = Claim(
            statement=str(
                payload.get("claim_text")
                or f"The BIM contains {count} {element_term} on {level}."
            ),
            value=count,
            unit=str(payload.get("unit") or element_term),
            basis=str(payload.get("identity") or "distinct project-scoped BIM element identities"),
            evidence_ids=[evidence_id],
            confidence=1.0,
        )
        if payload.get("capability") == "count_project_nodes":
            limitations.append(
                "This is an authorized project-scoped graph count, not a count of the entire Neo4j database."
            )
        else:
            limitations.append(
                "This is a BIM-element count, not an inference about unmodelled real-world units."
            )
        return VerificationReport(
            status="verified",
            verified_claims=[claim],
            limitations=limitations,
        )
    return VerificationReport(
        status="insufficient_evidence",
        rejected_claims=[
            f"Expected {payload.get('expected_count')}, independently obtained {payload.get('count')}."
        ],
        limitations=[
            *limitations,
            "The requested level was not recognized or the independent count differed.",
        ],
    )


def supervisor_report_from_evidence(context: BimRunContext) -> SupervisorReport:
    verification = verification_report_from_evidence(context)
    if verification.status == "verified" and verification.verified_claims:
        claim = verification.verified_claims[0]
        answer = claim.statement
        if verification.limitations:
            answer += " " + verification.limitations[-1]
        return SupervisorReport(
            answer=answer,
            claims=verification.verified_claims,
            limitations=verification.limitations,
            agents_used=["BIM Query Agent", "Verification Agent"],
            verification_status="verified",
        )
    return SupervisorReport(
        answer="The BIM question could not be verified from the available scoped evidence.",
        limitations=verification.limitations,
        agents_used=["BIM Query Agent", "Verification Agent"],
        verification_status="insufficient_evidence",
    )


@output_guardrail(name="verification_matches_deterministic_evidence")
def verification_matches_evidence(
    ctx: RunContextWrapper[BimRunContext], agent, output: VerificationReport
) -> GuardrailFunctionOutput:
    expected = verification_report_from_evidence(ctx.context)
    output_values = [claim.value for claim in output.verified_claims]
    expected_values = [claim.value for claim in expected.verified_claims]
    valid = output.status == expected.status and output_values == expected_values
    return GuardrailFunctionOutput(
        output_info={
            "expected_status": expected.status,
            "expected_values": expected_values,
            "actual_status": output.status,
            "actual_values": output_values,
        },
        tripwire_triggered=not valid,
    )


async def verification_failure_from_evidence(
    ctx: RunContextWrapper[BimRunContext], error: Exception
) -> str:
    """Fail closed to deterministic evidence if the verifier contradicts its own tool."""
    return verification_report_from_evidence(ctx.context).model_dump_json()


@output_guardrail(name="supervisor_matches_verified_evidence")
def supervisor_matches_evidence(
    ctx: RunContextWrapper[BimRunContext], agent, output: SupervisorReport
) -> GuardrailFunctionOutput:
    expected = verification_report_from_evidence(ctx.context)
    expected_status = "verified" if expected.status == "verified" else "insufficient_evidence"
    output_values = [claim.value for claim in output.claims]
    expected_values = [claim.value for claim in expected.verified_claims]
    valid = (
        output.verification_status == expected_status
        and (expected_status != "verified" or output_values == expected_values)
    )
    return GuardrailFunctionOutput(
        output_info={
            "expected_status": expected_status,
            "expected_values": expected_values,
            "actual_status": output.verification_status,
            "actual_values": output_values,
        },
        tripwire_triggered=not valid,
    )
