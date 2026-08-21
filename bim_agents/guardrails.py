from __future__ import annotations

import json

from .models import (
    BimQueryReport, BimRunContext, Claim, PipelineReport, SemanticCheck,
    VerificationReport,
)


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


def _semantic_check_signatures(checks: list[SemanticCheck]) -> list[tuple[str, bool, str]]:
    return [(check.name, check.passed, check.explanation) for check in checks]


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
        limitations.append("The query stage produced no deterministic evidence.")
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
            limitations=["The verifier produced no deterministic verification evidence."],
        )
    evidence_id, payload = latest
    limitations: list[str] = []
    claims: list[Claim] = []
    rejected: list[str] = []
    semantic_checks: list[SemanticCheck] = []
    for check in payload.get("checks") or []:
        include_in_answer = (check.get("plan") or {}).get("include_in_answer", True)
        if include_in_answer:
            limitations.extend(check.get("limitations") or [])
        semantic_checks.extend(
            SemanticCheck.model_validate(item)
            for item in check.get("semantic_checks") or []
        )
        if check.get("verified") is True and include_in_answer:
            claims.append(_claim_from_payload(
                {"claim": check.get("claim") or {}},
                [str(check.get("evidence_id")), evidence_id],
            ))
        else:
            failed_checks = [
                item.get("name", "unknown_check")
                for item in check.get("semantic_checks") or []
                if item.get("passed") is not True
            ]
            rejected.append(
                f"Query evidence {check.get('evidence_id')} was rejected"
                + (f" by: {', '.join(failed_checks)}." if failed_checks else ".")
            )
    if claims:
        limitations.append(
            "Every displayed result is restricted to the configured authorized BIM scope."
        )
        if rejected:
            limitations.append(
                "Some supporting query evidence failed verification and was omitted: "
                + " ".join(rejected)
            )
        return VerificationReport(
            status="verified",
            verified_claims=_consolidate_claims(claims),
            limitations=list(dict.fromkeys(limitations)),
            semantic_checks=semantic_checks,
        )
    return VerificationReport(
        status="insufficient_evidence",
        rejected_claims=rejected or ["No independently reproducible BIM query result was produced."],
        limitations=list(dict.fromkeys(limitations)),
        semantic_checks=semantic_checks,
    )


def pipeline_report_from_evidence(context: BimRunContext) -> PipelineReport:
    verification = verification_report_from_evidence(context)
    limitations = list(dict.fromkeys(
        verification.limitations + context.runtime_limitations
    ))
    investigation_trace = [
        f"{artifact.producer}: {artifact.summary}"
        for artifact in context.artifacts.values()
    ]
    if verification.status == "verified" and verification.verified_claims:
        verified_answer = "\n\n".join(
            _render_claim_answer(claim) for claim in verification.verified_claims
        )
        if context.completion_status == "insufficient_evidence" and context.runtime_limitations:
            answer = "\n\n".join(context.runtime_limitations + ["Verified diagnostic facts:\n" + verified_answer])
            status = "insufficient_evidence"
        else:
            answer = verified_answer
            status = "verified"
        return PipelineReport(
            answer=answer,
            claims=verification.verified_claims,
            limitations=limitations,
            stages_used=["Graph Inspector", "Schema Mapper", "Query Planner", "Cypher Query Handler", "Verifier"],
            artifact_ids=list(context.artifacts),
            investigation_trace=investigation_trace,
            semantic_checks=verification.semantic_checks,
            verification_status=status,
        )
    return PipelineReport(
        answer="The BIM question could not be verified from the available scoped evidence.",
        limitations=limitations,
        stages_used=["Graph Inspector", "Schema Mapper", "Query Planner", "Cypher Query Handler", "Verifier"],
        artifact_ids=list(context.artifacts),
        investigation_trace=investigation_trace,
        semantic_checks=verification.semantic_checks,
        verification_status="insufficient_evidence",
    )
