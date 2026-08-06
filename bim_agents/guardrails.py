from __future__ import annotations

import json

from agents import GuardrailFunctionOutput, RunContextWrapper, output_guardrail

from .models import BimQueryReport, BimRunContext, Claim, SupervisorReport, VerificationReport


def _latest_verification(context: BimRunContext) -> tuple[str, dict] | None:
    for evidence_id, evidence in reversed(list(context.evidence.items())):
        if evidence.kind == "verification":
            return evidence_id, json.loads(evidence.payload)
    return None


def _claim_from_payload(payload: dict, evidence_ids: list[str]) -> Claim:
    claim_payload = payload.get("claim") or {}
    return Claim(
        statement=str(claim_payload.get("statement") or "No BIM claim was produced."),
        value=claim_payload.get("value"),
        unit=claim_payload.get("unit"),
        basis=str(claim_payload.get("basis") or "project-scoped BIM query evidence"),
        evidence_ids=evidence_ids,
        details=[str(item) for item in claim_payload.get("details") or []],
        total_count=claim_payload.get("total_count"),
        displayed_count=claim_payload.get("displayed_count"),
        confidence=1.0,
    )


def _claim_signatures(claims: list[Claim]) -> list[tuple]:
    return [
        (
            claim.statement,
            claim.value,
            claim.unit,
            tuple(claim.details),
            claim.total_count,
            claim.displayed_count,
        )
        for claim in claims
    ]


def _render_claim_answer(claim: Claim) -> str:
    if not claim.details:
        return claim.statement
    return claim.statement + "\n\n- " + "\n\n- ".join(claim.details)


def _consolidate_claims(claims: list[Claim]) -> list[Claim]:
    """Collapse duplicate conclusions, retaining the version with the richer details."""
    consolidated: dict[tuple, Claim] = {}
    for claim in claims:
        key = (claim.statement, claim.value, claim.unit)
        existing = consolidated.get(key)
        if existing is None:
            consolidated[key] = claim
            continue
        preferred = claim if len(claim.details) > len(existing.details) else existing
        preferred.evidence_ids = list(dict.fromkeys(existing.evidence_ids + claim.evidence_ids))
        consolidated[key] = preferred
    return list(consolidated.values())


def query_report_from_evidence(context: BimRunContext) -> BimQueryReport:
    claims: list[Claim] = []
    evidence_ids: list[str] = []
    limitations: list[str] = []
    for evidence_id, evidence in context.evidence.items():
        if evidence.kind != "query":
            continue
        payload = json.loads(evidence.payload)
        include_in_answer = (payload.get("plan") or {}).get("include_in_answer", True)
        if include_in_answer:
            claims.append(_claim_from_payload(payload, [evidence_id]))
            limitations.extend(payload.get("limitations") or [])
        evidence_ids.append(evidence_id)
    if not claims:
        limitations.append("The BIM Analyst produced no deterministic query evidence.")
    return BimQueryReport(
        claims=_consolidate_claims(claims),
        evidence_ids=evidence_ids,
        limitations=list(dict.fromkeys(limitations)),
    )


def verification_report_from_evidence(context: BimRunContext) -> VerificationReport:
    latest = _latest_verification(context)
    if latest is None:
        return VerificationReport(
            status="insufficient_evidence",
            limitations=["The Verification Agent produced no deterministic verification evidence."],
        )
    evidence_id, payload = latest
    limitations: list[str] = []
    claims: list[Claim] = []
    rejected: list[str] = []
    for check in payload.get("checks") or []:
        include_in_answer = (check.get("plan") or {}).get("include_in_answer", True)
        if include_in_answer:
            limitations.extend(check.get("limitations") or [])
        if check.get("verified") is True and include_in_answer:
            claims.append(_claim_from_payload(
                {"claim": check.get("claim") or {}},
                [str(check.get("evidence_id")), evidence_id],
            ))
        else:
            rejected.append(
                f"Query evidence {check.get('evidence_id')} did not reproduce the same result digest."
            )
    if payload.get("verified") is True and claims:
        limitations.append(
            "Every result is restricted to the configured client, project, and authorized BIM sources."
        )
        return VerificationReport(
            status="verified",
            verified_claims=_consolidate_claims(claims),
            limitations=list(dict.fromkeys(limitations)),
        )
    return VerificationReport(
        status="insufficient_evidence",
        rejected_claims=rejected or ["No independently reproducible BIM query result was produced."],
        limitations=list(dict.fromkeys(limitations)),
    )


def supervisor_report_from_evidence(context: BimRunContext) -> SupervisorReport:
    verification = verification_report_from_evidence(context)
    if verification.status == "verified" and verification.verified_claims:
        answer = "\n\n".join(_render_claim_answer(claim) for claim in verification.verified_claims)
        return SupervisorReport(
            answer=answer,
            claims=verification.verified_claims,
            limitations=verification.limitations,
            agents_used=["BIM Analyst", "Verification Agent"],
            verification_status="verified",
        )
    return SupervisorReport(
        answer="The BIM question could not be verified from the available scoped evidence.",
        limitations=verification.limitations,
        agents_used=["BIM Analyst", "Verification Agent"],
        verification_status="insufficient_evidence",
    )


@output_guardrail(name="query_matches_deterministic_evidence")
def query_matches_evidence(
    ctx: RunContextWrapper[BimRunContext], agent, output: BimQueryReport
) -> GuardrailFunctionOutput:
    expected = query_report_from_evidence(ctx.context)
    valid = (
        _claim_signatures(output.claims) == _claim_signatures(expected.claims)
        and output.evidence_ids == expected.evidence_ids
    )
    return GuardrailFunctionOutput(
        output_info={
            "expected_claims": _claim_signatures(expected.claims),
            "actual_claims": _claim_signatures(output.claims),
        },
        tripwire_triggered=not valid,
    )


async def query_failure_from_evidence(
    ctx: RunContextWrapper[BimRunContext], error: Exception
) -> str:
    """Fall back to the deterministic query evidence if the analyst changes its result."""
    return query_report_from_evidence(ctx.context).model_dump_json()


@output_guardrail(name="verification_matches_deterministic_evidence")
def verification_matches_evidence(
    ctx: RunContextWrapper[BimRunContext], agent, output: VerificationReport
) -> GuardrailFunctionOutput:
    expected = verification_report_from_evidence(ctx.context)
    output_claims = _claim_signatures(output.verified_claims)
    expected_claims = _claim_signatures(expected.verified_claims)
    valid = output.status == expected.status and output_claims == expected_claims
    return GuardrailFunctionOutput(
        output_info={
            "expected_status": expected.status,
            "expected_claims": expected_claims,
            "actual_status": output.status,
            "actual_claims": output_claims,
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
    output_claims = _claim_signatures(output.claims)
    expected_claims = _claim_signatures(expected.verified_claims)
    valid = (
        output.verification_status == expected_status
        and (expected_status != "verified" or output_claims == expected_claims)
        and (
            expected_status != "verified"
            or all(claim.statement in output.answer for claim in expected.verified_claims)
        )
        and (
            expected_status != "verified"
            or all(
                detail in output.answer
                for claim in expected.verified_claims
                for detail in claim.details
            )
        )
    )
    return GuardrailFunctionOutput(
        output_info={
            "expected_status": expected_status,
            "expected_claims": expected_claims,
            "actual_status": output.verification_status,
            "actual_claims": output_claims,
        },
        tripwire_triggered=not valid,
    )
