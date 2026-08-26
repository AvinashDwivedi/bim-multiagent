from __future__ import annotations

import json

from .models import (
    BimQueryReport, BimRunContext, Claim, OutputSpec, PipelineReport,
    SemanticCheck, PlausibilityFlag, VerificationReport,
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
        coverage=claim_payload.get("coverage"),
        measurement=claim_payload.get("measurement"),
        source_tags=[str(item) for item in claim_payload.get("source_tags") or []],
        method=str(claim_payload.get("method") or ""),
        caveats=[str(item) for item in claim_payload.get("caveats") or []],
        plausibility_flags=claim_payload.get("plausibility_flags") or [],
        answer_key=str(claim_payload.get("answer_key") or ""),
        satisfies=[str(item) for item in claim_payload.get("satisfies") or []],
        work_package_id=str(payload.get("work_package_id") or ""),
        constraint_bindings=claim_payload.get("constraint_bindings") or [],
        confidence=1.0,
    )


def _package_for_output(context: BimRunContext, output: str) -> str:
    if not context.task_contract:
        return ""
    for package in context.task_contract.work_packages:
        if output in package.required_outputs:
            return package.package_id
    return ""


def _verified_outputs_for_check(
    context: BimRunContext, check: dict, plan: dict,
) -> tuple[set[str], str]:
    """Accept output satisfaction only from the package that owns the output.

    Legacy/root evidence remains valid.  Isolated workstream evidence, however, may
    satisfy only its declared package outputs, preventing global constraints and
    outputs from leaking between parallel workers.
    """
    outputs = {str(item) for item in plan.get("satisfies") or []}
    evidence = context.evidence.get(str(check.get("evidence_id") or ""))
    if evidence is None:
        return outputs, ""
    package_id = evidence.work_package_id or (
        evidence.workstream_id if evidence.workstream_id != "root" else ""
    )
    if not package_id or not context.task_contract or not context.task_contract.work_packages:
        return outputs, package_id
    package = next(
        (item for item in context.task_contract.work_packages if item.package_id == package_id),
        None,
    )
    if package is None:
        return set(), package_id
    return outputs & set(package.required_outputs), package_id


def _claim_signatures(claims: list[Claim]) -> list[tuple]:
    return [
        (
            claim.statement,
            claim.value,
            claim.unit,
            tuple(claim.details),
            claim.total_count,
            claim.displayed_count,
            claim.coverage.model_dump_json() if claim.coverage else None,
            claim.measurement.model_dump_json() if claim.measurement else None,
        )
        for claim in claims
    ]


def _semantic_check_signatures(checks: list[SemanticCheck]) -> list[tuple[str, bool, str]]:
    return [(check.name, check.passed, check.explanation) for check in checks]


def _render_claim_answer(claim: Claim) -> str:
    sections = [claim.statement]
    if claim.details:
        sections.append("- " + "\n\n- ".join(claim.details))
    caveats = list(claim.caveats)
    if (
        claim.total_count is not None
        and claim.displayed_count is not None
        and claim.displayed_count < claim.total_count
    ):
        disclosure = (
            f"The detailed list shows {claim.displayed_count} of {claim.total_count} results; "
            "the summary total covers the complete verified population."
        )
        if disclosure not in caveats:
            caveats.append(disclosure)
    if caveats:
        sections.append("Caveats:\n- " + "\n- ".join(caveats))
    if claim.method:
        derivation = f"Method: {claim.method.replace('_', ' ')}."
        if claim.coverage is not None:
            derivation += (
                f" Evaluated {claim.coverage.evaluated_count} of "
                f"{claim.coverage.candidate_count} candidate records"
                + (" exhaustively." if claim.coverage.exhaustive else ".")
            )
        if claim.measurement is not None and claim.measurement.source_property:
            derivation += f" Source quantity: {claim.measurement.source_property}."
        if claim.source_tags:
            derivation += " Source tags: " + ", ".join(claim.source_tags[:5]) + "."
        sections.append("How this was derived:\n" + derivation)
    return "\n\n".join(sections)


_ADEQUACY_ONLY_CHECKS = {
    "entity_grain_matches", "measurement_basis_matches", "population_complete",
    "planned_actual_distinguished", "absence_semantics_correct",
    "projection_answers_question", "requested_outputs_present",
}


def _is_safe_replay_support(check: dict) -> bool:
    """Retain a computed fact when only answer-fit semantics remain unresolved.

    This never satisfies a required output and can never promote knowledge. It merely
    prevents a replay-stable, scoped calculation from disappearing behind a generic
    failure sentence. Any failed safety, identity, classification, scope, constraint,
    relationship, or plausibility check still suppresses the value completely.
    """
    checks = check.get("semantic_checks") or []
    failed = {
        str(item.get("name") or "unknown_check")
        for item in checks if item.get("passed") is not True
    }
    passed = {
        str(item.get("name") or "")
        for item in checks if item.get("passed") is True
    }
    required_safety = {"replay_stability", "authorized_scope", "measurement_plausibility"}
    return bool(failed) and failed.issubset(_ADEQUACY_ONLY_CHECKS) and required_safety.issubset(passed)


def _stages_used(context: BimRunContext) -> list[str]:
    """Report stages that actually produced artifacts, in causal ledger order."""
    return list(dict.fromkeys(
        artifact.producer for artifact in context.artifacts.values()
        if artifact.producer != "Pipeline"
    ))


def _is_answer_plan(plan: dict) -> bool:
    """Honor explicit evidence roles while retaining legacy include_in_answer plans."""
    if plan.get("include_in_answer") is False:
        return False
    role = plan.get("role")
    if role is not None:
        return role == "answer_producing"
    return plan.get("include_in_answer", True) is True


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
        preferred.satisfies = list(dict.fromkeys(existing.satisfies + claim.satisfies))
        preferred.source_tags = list(dict.fromkeys(existing.source_tags + claim.source_tags))
        preferred.caveats = list(dict.fromkeys(existing.caveats + claim.caveats))
        bindings = {
            item.model_dump_json(): item
            for item in [*existing.constraint_bindings, *claim.constraint_bindings]
        }
        preferred.constraint_bindings = list(bindings.values())
        consolidated[key] = preferred
    return list(consolidated.values())


def _typed_output_score(claim: Claim, plan: dict, spec: OutputSpec) -> tuple[int, ...]:
    """Rank alternative claims only where the output type defines richer structure."""
    operation = str(plan.get("operation") or "")
    if spec.kind == "grouped_summary":
        operation_score = {
            "multi_group_summary": 5,
            "group_summary": 5,
            "multi_group_count": 4,
            "group_count": 4,
            "distinct": 2,
            "list": 1,
            "count_distinct": 1,
            "count": 0,
        }.get(operation, 0)
        actual_dimensions = {
            str(item) for item in plan.get("group_by_fields") or [] if str(item)
        }
        if plan.get("group_by"):
            actual_dimensions.add(str(plan["group_by"]))
        expected_dimensions = set(spec.grouping_dimensions)
        dimensions_match = int(
            bool(expected_dimensions) and expected_dimensions.issubset(actual_dimensions)
        )
        metric_match = int(bool(spec.metric) and plan.get("metric") == spec.metric)
        return (
            operation_score,
            dimensions_match,
            metric_match,
            int(bool(claim.details)),
            len(claim.details),
        )
    if spec.kind == "measurement":
        operation_score = {
            "maximum_group_sum": 6,
            "multi_group_summary": 5,
            "group_summary": 5,
            "sum": 4,
            "average": 4,
            "minimum": 4,
            "maximum": 4,
            "list": 1,
            "count_distinct": 0,
            "count": 0,
        }.get(operation, 0)
        unit_match = int(
            bool(spec.required_unit)
            and (
                claim.unit == spec.required_unit
                or (
                    claim.measurement is not None
                    and claim.measurement.canonical_unit == spec.required_unit
                )
            )
        )
        metric_match = int(bool(spec.metric) and plan.get("metric") == spec.metric)
        return (
            operation_score,
            metric_match,
            unit_match,
            int(claim.measurement is not None),
            len(claim.details),
        )
    return (0,)


def _dominant_verified_entries(
    context: BimRunContext, entries: list[tuple[Claim, dict]],
) -> list[tuple[Claim, dict]]:
    """Keep the richest result per typed output without dropping other obligations.

    The verifier has already removed conflicting atomic answers before this helper is
    called. Tied results remain visible, and a claim that supports any distinct output
    is retained even when a richer claim dominates one of its other outputs.
    """
    if not context.task_contract:
        return entries
    ranked_specs = {
        output: spec
        for output, spec in zip(
            context.task_contract.required_outputs,
            context.task_contract.output_specs,
            strict=True,
        )
        if spec.kind in {"grouped_summary", "measurement"}
    }
    if not ranked_specs:
        return entries

    winners: set[int] = set()
    considered: set[int] = set()
    for output, spec in ranked_specs.items():
        candidates = [
            index
            for index, (claim, _plan) in enumerate(entries)
            if output in claim.satisfies
        ]
        if not candidates:
            continue
        considered.update(candidates)
        best_score = max(
            _typed_output_score(entries[index][0], entries[index][1], spec)
            for index in candidates
        )
        winners.update(
            index
            for index in candidates
            if _typed_output_score(entries[index][0], entries[index][1], spec) == best_score
        )

    retained: list[tuple[Claim, dict]] = []
    for index, entry in enumerate(entries):
        claim, _plan = entry
        supports_distinct_output = any(
            output not in ranked_specs for output in claim.satisfies
        )
        if index not in considered or index in winners or supports_distinct_output:
            retained.append(entry)
    return retained


def query_report_from_evidence(context: BimRunContext) -> BimQueryReport:
    claims: list[Claim] = []
    evidence_ids: list[str] = []
    limitations: list[str] = []
    for evidence_id, evidence in context.evidence.items():
        if evidence.kind != "query":
            continue
        payload = json.loads(evidence.payload)
        payload.setdefault(
            "work_package_id",
            evidence.work_package_id or (
                evidence.workstream_id if evidence.workstream_id != "root" else ""
            ),
        )
        claim_payload = payload.setdefault("claim", {})
        if evidence.constraint_bindings and not claim_payload.get("constraint_bindings"):
            claim_payload["constraint_bindings"] = [
                item.model_dump(mode="json") for item in evidence.constraint_bindings
            ]
        include_in_answer = _is_answer_plan(payload.get("plan") or {})
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
    verified_entries: list[tuple[Claim, dict]] = []
    supporting_entries: list[tuple[Claim, dict]] = []
    satisfied_outputs: set[str] = set()
    plausibility_flags: list[PlausibilityFlag] = []
    for check in payload.get("checks") or []:
        plan = check.get("plan") or {}
        include_in_answer = _is_answer_plan(plan)
        if (
            check.get("verified") is True
            and plan.get("role", "answer_producing") in {"answer_producing", "supporting"}
        ):
            owned_outputs, _ = _verified_outputs_for_check(context, check, plan)
            satisfied_outputs.update(owned_outputs)
        role = plan.get("role", "answer_producing")
        if include_in_answer or role == "supporting":
            limitations.extend(check.get("limitations") or [])
        semantic_checks.extend(
            SemanticCheck.model_validate(item)
            for item in check.get("semantic_checks") or []
        )
        for item in check.get("plausibility_flags") or []:
            flag = PlausibilityFlag.model_validate(item)
            if not flag.evidence_ids:
                flag = flag.model_copy(update={"evidence_ids": [str(check.get("evidence_id"))]})
            plausibility_flags.append(flag)
        if check.get("verified") is True and (
            include_in_answer or role == "supporting"
        ):
            source_evidence = context.evidence.get(str(check.get("evidence_id") or ""))
            owned_outputs, package_id = _verified_outputs_for_check(context, check, plan)
            claim_payload = dict(check.get("claim") or {})
            if (
                source_evidence is not None
                and source_evidence.constraint_bindings
                and not claim_payload.get("constraint_bindings")
            ):
                claim_payload["constraint_bindings"] = [
                    item.model_dump(mode="json")
                    for item in source_evidence.constraint_bindings
                ]
            claim = _claim_from_payload(
                {
                    "claim": claim_payload,
                    "work_package_id": package_id,
                },
                [str(check.get("evidence_id")), evidence_id],
            )
            if not claim.answer_key:
                claim.answer_key = str(plan.get("answer_key") or "")
            claim.satisfies = sorted(owned_outputs)
            if include_in_answer:
                verified_entries.append((claim, plan))
            else:
                supporting_entries.append((claim, plan))
        elif include_in_answer:
            failed_checks = [
                item.get("name", "unknown_check")
                for item in check.get("semantic_checks") or []
                if item.get("passed") is not True
            ]
            if _is_safe_replay_support(check) and check.get("claim"):
                source_evidence = context.evidence.get(str(check.get("evidence_id") or ""))
                _, package_id = _verified_outputs_for_check(context, check, plan)
                claim_payload = dict(check.get("claim") or {})
                if (
                    source_evidence is not None
                    and source_evidence.constraint_bindings
                    and not claim_payload.get("constraint_bindings")
                ):
                    claim_payload["constraint_bindings"] = [
                        item.model_dump(mode="json")
                        for item in source_evidence.constraint_bindings
                    ]
                claim = _claim_from_payload(
                    {"claim": claim_payload, "work_package_id": package_id},
                    [str(check.get("evidence_id")), evidence_id],
                )
                claim.answer_key = str(plan.get("answer_key") or claim.answer_key)
                claim.satisfies = []
                claim.confidence = min(claim.confidence, 0.5)
                explanations = [
                    str(item.get("explanation") or item.get("name") or "semantic fit unresolved")
                    for item in check.get("semantic_checks") or []
                    if item.get("passed") is not True
                ]
                claim.caveats = list(dict.fromkeys([
                    *claim.caveats,
                    "This is a replay-stable related BIM fact, but it did not satisfy the direct-answer semantic contract.",
                    *explanations,
                ]))
                supporting_entries.append((claim, plan))
            rejected.append(
                f"Query evidence {check.get('evidence_id')} was rejected"
                + (f" by: {', '.join(failed_checks)}." if failed_checks else ".")
            )
    conflicts: set[str] = set()
    by_answer_key: dict[str, set[tuple]] = {}
    for claim, plan in verified_entries:
        answer_key = str(plan.get("answer_key") or "").strip()
        if answer_key:
            by_answer_key.setdefault(answer_key, set()).add((str(claim.value), claim.unit))
    conflicts = {key for key, values in by_answer_key.items() if len(values) > 1}
    if conflicts:
        limitations.append(
            "Conflicting verified values were produced for: " + ", ".join(sorted(conflicts)) + "."
        )
    nonconflicting_entries: list[tuple[Claim, dict]] = []
    for claim, plan in verified_entries:
        answer_key = str(plan.get("answer_key") or "").strip()
        if answer_key in conflicts:
            rejected.append(
                f"Conflicting verified values were produced for atomic answer {answer_key!r}."
            )
        else:
            nonconflicting_entries.append((claim, plan))
    claims.extend(
        claim for claim, _plan in _dominant_verified_entries(context, nonconflicting_entries)
    )
    required_outputs = set(context.task_contract.required_outputs if context.task_contract else [])
    missing_outputs = sorted(required_outputs - satisfied_outputs)
    if missing_outputs:
        limitations.append(
            "The investigation did not verify every required output: " + ", ".join(missing_outputs) + "."
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
            status="insufficient_evidence" if conflicts or missing_outputs else "verified",
            verified_claims=_consolidate_claims(claims),
            supporting_claims=_consolidate_claims([
                claim for claim, _plan in supporting_entries
            ]),
            limitations=list(dict.fromkeys(limitations)),
            semantic_checks=semantic_checks,
            plausibility_flags=plausibility_flags,
        )
    return VerificationReport(
        status="insufficient_evidence",
        supporting_claims=_consolidate_claims([
            claim for claim, _plan in supporting_entries
        ]),
        rejected_claims=rejected or ["No independently reproducible BIM query result was produced."],
        limitations=list(dict.fromkeys(limitations)),
        semantic_checks=semantic_checks,
        plausibility_flags=plausibility_flags,
    )


def pipeline_report_from_evidence(context: BimRunContext) -> PipelineReport:
    verification = verification_report_from_evidence(context)
    limitations = list(dict.fromkeys(
        verification.limitations + context.runtime_limitations
    ))
    answer_limitations = [
        item for item in limitations
        if item != "Every displayed result is restricted to the configured authorized BIM scope."
    ]
    investigation_trace = [
        f"{artifact.producer}: {artifact.summary}"
        for artifact in context.artifacts.values()
    ]
    plausibility_flags = verification.plausibility_flags
    required_outputs = list(context.task_contract.required_outputs if context.task_contract else [])
    output_statuses = []
    specs_by_output = dict(zip(
        required_outputs,
        context.task_contract.output_specs if context.task_contract else [],
        strict=True,
    ))
    for output in required_outputs:
        supporting_claims = [
            claim for claim in [
                *verification.verified_claims, *verification.supporting_claims,
            ]
            if output in claim.satisfies
        ]
        output_spec = specs_by_output.get(output)
        output_statuses.append({
            "output": output,
            "status": "verified" if supporting_claims else "unsupported",
            "evidence_ids": list(dict.fromkeys(
                evidence_id for claim in supporting_claims for evidence_id in claim.evidence_ids
            )),
            "limitation": "" if supporting_claims else f"No replay-verified evidence satisfied {output!r}.",
            "spec": output_spec.model_dump(mode="json") if output_spec else None,
            "package_id": _package_for_output(context, output),
        })
    if verification.verified_claims:
        verified_answer = "\n\n".join(
            _render_claim_answer(claim) for claim in verification.verified_claims
        )
        if verification.status != "verified" or (
            context.completion_status == "insufficient_evidence" and context.runtime_limitations
        ):
            # Lead with useful replay-verified facts. Open secondary gates are
            # caveats, not a reason to bury the answer behind pipeline prose.
            answer = verified_answer
            if verification.supporting_claims:
                answer += "\n\nVerified supporting evidence:\n\n" + "\n\n".join(
                    _render_claim_answer(claim) for claim in verification.supporting_claims
                )
            if answer_limitations:
                answer += "\n\nUnresolved verification notes:\n- " + "\n- ".join(
                    answer_limitations
                )
            status = "insufficient_evidence"
        else:
            answer = verified_answer
            status = "verified"
        return PipelineReport(
            answer=answer,
            claims=verification.verified_claims,
            supporting_claims=verification.supporting_claims,
            limitations=limitations,
            stages_used=_stages_used(context),
            artifact_ids=list(context.artifacts),
            investigation_trace=investigation_trace,
            semantic_checks=verification.semantic_checks,
            failure_categories=list(dict.fromkeys(context.failure_categories)),
            plausibility_flags=plausibility_flags,
            output_statuses=output_statuses,
            workstream_diagnostics=context.workstream_diagnostics,
            verification_status=status,
        )
    if verification.supporting_claims:
        answer = "Verified supporting evidence:\n\n" + "\n\n".join(
            _render_claim_answer(claim) for claim in verification.supporting_claims
        )
        answer += "\n\nA complete direct answer remains unresolved."
    else:
        answer = (
            "A direct answer could not be verified, and no replay-verified supporting "
            "fact was available."
        )
    if answer_limitations:
        answer += "\n\nUnresolved verification notes:\n- " + "\n- ".join(answer_limitations)
    return PipelineReport(
        answer=answer,
        supporting_claims=verification.supporting_claims,
        limitations=limitations,
        stages_used=_stages_used(context),
        artifact_ids=list(context.artifacts),
        investigation_trace=investigation_trace,
        semantic_checks=verification.semantic_checks,
        failure_categories=list(dict.fromkeys(context.failure_categories)),
        output_statuses=output_statuses,
        workstream_diagnostics=context.workstream_diagnostics,
        verification_status="insufficient_evidence",
    )
