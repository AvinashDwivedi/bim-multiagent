from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Callable

from .claude_runtime import MaxTurnsExceeded, Runner

from bim_context import BimContext, Settings

from .models import (
    BimRunContext,
    BimTaskContract,
    ConstraintBinding,
    EvidenceHandoff,
    EvidenceWorkPackage,
    EvidenceWorkstreamResult,
    ResolvedConstraint,
    SemanticIntent,
    SpecialistRole,
)
from .observability import AgentRunHooks, PipelineEvents
from .registry import BimAgentRegistry
from .schema_mapping import RegisteredSchemaMapping
from .tools import (
    BimFilter,
    BimQueryPlan,
    GeometryQueryPlan,
    PipelineContext,
    _constraint_field,
    _level_matches,
    _measurement_dimension,
    activate_project_knowledge_mapping,
    define_bim_task,
    ensure_bim_verification,
    query_bim,
    query_project_geometry,
)


@dataclass
class WorkstreamOutcome:
    package: EvidenceWorkPackage
    branch: BimRunContext | None
    status: str
    evidence_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    error: Exception | None = None
    specialist: str = ""
    specialist_attempts: int = 0
    typed_output_failures: int = 0
    verification_rejections: int = 0
    recovery_strategy: str = ""
    satisfied_outputs: list[str] = field(default_factory=list)


def _answer_evidence_snapshot(
    branch: BimRunContext,
    package: EvidenceWorkPackage,
) -> tuple[list[str], list[str]]:
    """Return valid package-local answer evidence and the outputs it covers.

    The evidence store is the durable checkpoint: a malformed model handoff must
    not erase query work that the worker already completed.  Invalid/non-answer
    artifacts are ignored, and foreign-package evidence is never recovered.
    """
    evidence_ids: list[str] = []
    satisfied: set[str] = set()
    owned_outputs = set(package.required_outputs)
    for evidence_id, evidence in branch.evidence.items():
        if evidence.kind != "query":
            continue
        if evidence.work_package_id and evidence.work_package_id != package.package_id:
            continue
        try:
            payload = json.loads(evidence.payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        plan = payload.get("plan") or {}
        if not isinstance(plan, dict):
            continue
        if plan.get("role", "answer_producing") != "answer_producing":
            continue
        if plan.get("include_in_answer", True) is False:
            continue
        evidence_ids.append(evidence_id)
        satisfied.update(
            str(item) for item in (plan.get("satisfies") or [])
            if str(item) in owned_outputs
        )
    return evidence_ids, sorted(satisfied)


def _validate_workstream_completion(
    value: Any,
    *,
    branch: BimRunContext,
    package: EvidenceWorkPackage,
) -> EvidenceWorkstreamResult:
    """Validate both the typed envelope and its references to durable evidence."""
    completion = EvidenceWorkstreamResult.model_validate(value)
    if completion.package_id != package.package_id:
        raise ValueError("Query worker returned a result for a different work package.")
    unknown_ids = []
    referenced_outputs: set[str] = set()
    for evidence_id in completion.evidence_ids:
        evidence = branch.evidence.get(evidence_id)
        if (
            evidence is None
            or evidence.kind != "query"
            or evidence.work_package_id not in {"", package.package_id}
        ):
            unknown_ids.append(evidence_id)
            continue
        try:
            plan = json.loads(evidence.payload).get("plan") or {}
        except (TypeError, ValueError, json.JSONDecodeError):
            unknown_ids.append(evidence_id)
            continue
        if (
            plan.get("role", "answer_producing") != "answer_producing"
            or plan.get("include_in_answer", True) is False
        ):
            unknown_ids.append(evidence_id)
            continue
        referenced_outputs.update(
            str(item) for item in plan.get("satisfies") or []
            if str(item) in package.required_outputs
        )
    if unknown_ids:
        raise ValueError(
            "Query worker returned unknown, foreign-package, or non-answer evidence IDs."
        )
    if completion.status == "query_completed":
        missing = sorted(set(package.required_outputs) - referenced_outputs)
        if missing:
            raise ValueError(
                "Completed workstream evidence does not satisfy outputs: "
                + ", ".join(missing)
                + "."
            )
    return completion


def _typed_repair_limit() -> int:
    """Read a deliberately small repair budget; retries still consume branch budgets."""
    try:
        configured = int(os.getenv("BIM_TYPED_OUTPUT_REPAIR_ATTEMPTS", "1"))
    except ValueError:
        configured = 1
    return min(max(configured, 0), 2)


def _verification_repair_limit() -> int:
    """Return the bounded number of evidence-driven query correction attempts."""
    try:
        configured = int(os.getenv("BIM_VERIFICATION_REPAIR_ATTEMPTS", "1"))
    except ValueError:
        configured = 1
    return min(max(configured, 0), 2)


def _repair_step_timeout() -> float:
    """Bound one corrective model step so finalization can retain its checkpoint.

    The public timeout is a limit for the whole investigation, not a useful budget
    for any single remapping attempt.  A correction that stalls must yield to the
    deterministic evidence checkpoint instead of consuming the finalization reserve.
    """
    try:
        configured = float(os.getenv("BIM_REPAIR_STEP_TIMEOUT_SECONDS", "90"))
    except ValueError:
        configured = 90.0
    return min(max(configured, 5.0), 180.0)


def _package_verification_snapshot(
    branch: BimRunContext,
    package: EvidenceWorkPackage,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Verify branch evidence and return verified IDs, outputs, and compact failures.

    Verification is performed while the worker is still recoverable.  The compact
    failure records contain deterministic check names and explanations only; they
    never expose graph credentials or sibling workstream state.
    """
    verification_id = ensure_bim_verification(branch)
    if verification_id is None:
        return [], [], [{"failed_checks": ["no_query_evidence"]}]
    payload = json.loads(branch.evidence[verification_id].payload)
    owned_outputs = set(package.required_outputs)
    verified_ids: list[str] = []
    verified_outputs: set[str] = set()
    failures: list[dict[str, Any]] = []
    for check in payload.get("checks") or []:
        plan = check.get("plan") or {}
        satisfies = {
            str(item) for item in plan.get("satisfies") or []
            if str(item) in owned_outputs
        }
        if not satisfies or plan.get("role", "answer_producing") != "answer_producing":
            continue
        evidence_id = str(check.get("evidence_id") or "")
        if check.get("verified") is True:
            if evidence_id:
                verified_ids.append(evidence_id)
            verified_outputs.update(satisfies)
            continue
        failed_checks = [
            {
                "name": str(item.get("name") or "semantic_check"),
                "explanation": " ".join(str(item.get("explanation") or "").split())[:500],
            }
            for item in check.get("semantic_checks") or []
            if item.get("passed") is not True
        ]
        failures.append({
            "evidence_id": evidence_id,
            "outputs": sorted(satisfies),
            "failed_checks": failed_checks,
            "diagnostics": list(check.get("diagnostics") or [])[:5],
        })
    absence_ids, absence_outputs = _verified_requirements_absence_snapshot(
        branch, package, payload,
    )
    verified_ids.extend(absence_ids)
    verified_outputs.update(absence_outputs)
    if absence_outputs:
        # The allowed measurement-basis failure is vacuous for an empty population,
        # not a live semantic contradiction. Remove its provisional failure record so
        # callers do not launch a pointless remapping repair after accepting absence.
        failures = [
            failure for failure in failures
            if not set(failure.get("outputs") or []) <= set(absence_outputs)
        ]
    return list(dict.fromkeys(verified_ids)), sorted(verified_outputs), failures


def _verified_requirements_absence_snapshot(
    branch: BimRunContext,
    package: EvidenceWorkPackage,
    verification_payload: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Recognize a replay-verified empty requirements population as a typed result.

    A zero-row requirements query cannot exhibit a requested requirement measurement
    basis because there is no record from which to observe that basis. That one check
    is therefore vacuous after replay, authorization, exhaustive population, and
    projection checks have independently passed. Other semantic failures remain fatal.
    """
    if package.route_hint != "requirements" and not _is_compliance_package(package):
        return [], []
    owned = set(package.required_outputs)
    evidence_ids: list[str] = []
    outputs: set[str] = set()
    required_checks = {
        "replay_stability", "authorized_scope", "population_coverage",
        "population_complete", "projection_answers_question",
        "requested_outputs_present",
    }
    vacuous_on_empty = {"measurement_basis_matches"}
    for check in verification_payload.get("checks") or []:
        evidence_id = str(check.get("evidence_id") or "")
        evidence = branch.evidence.get(evidence_id)
        if not evidence or evidence.kind != "query":
            continue
        try:
            query_payload = json.loads(evidence.payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        plan = query_payload.get("plan") or {}
        claim = query_payload.get("claim") or {}
        coverage = claim.get("coverage") or {}
        satisfies = {
            str(item) for item in plan.get("satisfies") or []
            if str(item) in owned
        }
        if not satisfies or plan.get("role", "answer_producing") != "answer_producing":
            continue
        if not (
            plan.get("entity") == "permit_knowledge"
            and int(claim.get("value") or 0) == 0
            and bool(coverage.get("exhaustive"))
            and int(coverage.get("candidate_count") or 0) == 0
        ):
            continue
        semantic_checks = check.get("semantic_checks") or []
        passed = {
            str(item.get("name") or "") for item in semantic_checks
            if item.get("passed") is True
        }
        failed = {
            str(item.get("name") or "") for item in semantic_checks
            if item.get("passed") is not True
        }
        if not required_checks <= passed or failed - vacuous_on_empty:
            continue
        evidence_ids.append(evidence_id)
        outputs.update(satisfies)
    return list(dict.fromkeys(evidence_ids)), sorted(outputs)


def _verification_repair_input(
    original_input: str,
    *,
    package: EvidenceWorkPackage,
    verified_ids: list[str],
    verified_outputs: list[str],
    failures: list[dict[str, Any]],
) -> str:
    """Give a specialist deterministic verification feedback for one bounded retry."""
    missing = [
        output for output in package.required_outputs if output not in verified_outputs
    ]
    return (
        original_input
        + "\nEvidence-verification correction:\n"
        + "Independent replay and semantic checks rejected or did not cover part of the "
        + "previous answer. Preserve verified evidence, inspect the failed check feedback, "
        + "and submit a revised declarative plan only for unresolved outputs. Do not repeat "
        + "an identical rejected plan. If the schema handoff cannot support the correction, "
        + "return unsupported with that precise mapping gap.\n"
        + json.dumps({
            "verified_evidence_ids": verified_ids,
            "verified_outputs": verified_outputs,
            "missing_outputs": missing,
            "verification_failures": failures[-4:],
            "package_id": package.package_id,
        }, ensure_ascii=False, sort_keys=True)
    )


def _verification_requires_remapping(failures: list[dict[str, Any]]) -> bool:
    """Classify verifier failures owned by discovery rather than query planning."""
    mapping_checks = {
        "entity_grain_matches", "measurement_basis_matches", "population_complete",
        "classification_purity", "constraint_binding", "constraint_coverage",
        "relationship_binding", "counting_unit", "identity_integrity",
    }
    return any(
        str(check.get("name") or "") in mapping_checks
        for failure in failures
        for check in failure.get("failed_checks") or []
        if isinstance(check, dict)
    )


def _repair_input(
    original_input: str,
    *,
    package: EvidenceWorkPackage,
    error: Exception,
    evidence_ids: list[str],
    satisfied_outputs: list[str],
) -> str:
    missing = [
        output for output in package.required_outputs if output not in satisfied_outputs
    ]
    validation_error = " ".join(str(error).split())[:800]
    return (
        original_input
        + "\nTerminal-output repair:\n"
        + "Your previous terminal result did not satisfy the EvidenceWorkstreamResult contract. "
        + "Do not repeat completed query work. Return one valid typed terminal result and perform "
        + "only work still required for missing outputs.\n"
        + json.dumps({
            "validation_error": validation_error,
            "retained_evidence_ids": evidence_ids,
            "satisfied_outputs": satisfied_outputs,
            "missing_outputs": missing,
            "package_id": package.package_id,
        }, ensure_ascii=False, sort_keys=True)
    )


def evidence_work_packages(contract: BimTaskContract) -> list[EvidenceWorkPackage]:
    """Return the architect DAG, with a safe single-package legacy fallback."""
    if contract.work_packages:
        return list(contract.work_packages)
    return [EvidenceWorkPackage(
        package_id="primary",
        objective=contract.goal,
        required_outputs=list(contract.required_outputs),
        output_specs=[item.model_copy(deep=True) for item in contract.output_specs],
    )]


def package_contract(
    contract: BimTaskContract, package: EvidenceWorkPackage,
) -> BimTaskContract:
    """Project the global definition of done into one worker-sized contract."""
    return BimTaskContract(
        goal=package.objective,
        operation=contract.operation,
        entity_concept=contract.entity_concept,
        constraints=_package_constraints(contract, package),
        # Root-level unresolved questions may describe sibling packages. The package
        # objective and typed outputs are the complete specialist contract.
        questions_to_resolve=[],
        required_outputs=list(package.required_outputs),
        output_specs=[item.model_copy(deep=True) for item in package.output_specs],
        success_criteria=[
            criterion for criterion in contract.success_criteria
            if any(output.casefold() in criterion.casefold() for output in package.required_outputs)
        ] or [f"Produce replayable evidence for {output}." for output in package.required_outputs],
        complexity=contract.complexity,
    )


def _package_constraints(
    contract: BimTaskContract, package: EvidenceWorkPackage,
) -> list:
    """Project global constraints onto the output package that actually owns them."""
    if package.constraints is not None:
        return list(package.constraints)
    packages = evidence_work_packages(contract)
    if len(packages) == 1:
        return list(contract.constraints)
    package_text = " ".join([
        package.objective, *package.required_outputs,
    ]).casefold()
    selected = []
    for constraint in contract.constraints:
        concept = constraint.concept.casefold().strip()
        value = constraint.requested_value.casefold().strip()
        if (value and value in package_text) or (concept and concept in package_text):
            selected.append(constraint)
    return selected


def _branch_limit(total: int, consumed: int, workstream_count: int, slot: int) -> int:
    """Allocate a disjoint quotient/remainder budget; branch totals never exceed root."""
    remaining = max(total - consumed, 0)
    count = max(workstream_count, 1)
    quotient, remainder = divmod(remaining, count)
    return quotient + (1 if slot < remainder else 0)


def resolve_package_specialist(package: EvidenceWorkPackage) -> SpecialistRole:
    """Resolve one terminal specialist from typed semantics, failing closed on conflict."""
    inferred: set[str] = set()
    kinds = {item.kind for item in package.output_specs}
    if "relationship_coverage" in kinds:
        inferred.add("relationship")
    if "compliance" in kinds or package.route_hint == "requirements":
        inferred.add("requirements")
    if package.route_hint == "geometry":
        inferred.add("geometry")
    if len(inferred) > 1:
        raise ValueError(
            f"Work package {package.package_id!r} mixes incompatible specialist jobs: "
            + ", ".join(sorted(inferred))
            + "."
        )
    semantic_role = next(iter(inferred), "")
    if package.specialist != "auto":
        if semantic_role and package.specialist != semantic_role:
            raise ValueError(
                f"Work package {package.package_id!r} assigns {package.specialist!r} "
                f"but its typed route requires {semantic_role!r}."
            )
        return package.specialist
    return semantic_role or "quantity"


def _is_compliance_package(package: EvidenceWorkPackage) -> bool:
    """Return whether a package produces a compliance comparison.

    Compliance is identified from the typed output rather than package prose. This
    keeps the dependency safety rule stable across construction domains and client
    vocabulary.
    """
    return any(item.kind == "compliance" for item in package.output_specs)


def _dependency_policy(package: EvidenceWorkPackage) -> str:
    """Resolve a package's dependency policy at the scheduling boundary.

    Requirements availability is an independent evidence route: it must remain
    runnable when a model-fact sibling fails. A compliance comparison is different:
    it may only run after every declared input succeeds, even if a malformed contract
    explicitly asks for a more permissive policy.
    """
    if _is_compliance_package(package):
        return "all_success"
    if package.dependency_policy == "auto":
        return "independent" if package.route_hint == "requirements" else "all_success"
    return package.dependency_policy


def _failed_dependencies(
    package: EvidenceWorkPackage,
    finished: dict[str, WorkstreamOutcome],
) -> list[str]:
    """Return dependencies that are not admissible for this package's policy."""
    policy = _dependency_policy(package)
    if policy == "independent":
        return []
    accepted = {"query_completed"}
    if policy == "allow_partial":
        accepted.add("partial_completed")
    failed = []
    for dependency in package.depends_on:
        outcome = finished[dependency]
        if outcome.status in accepted:
            continue
        # Defensive recovery for an older/partial outcome envelope: verified typed
        # absence is a successful requirements-availability result, not incomplete
        # evidence. The normal path now emits query_completed before reaching here.
        dependency_package = getattr(outcome, "package", None)
        dependency_branch = getattr(outcome, "branch", None)
        satisfied = set(getattr(outcome, "satisfied_outputs", []) or [])
        if (
            dependency_package is not None
            and dependency_package.route_hint == "requirements"
            and set(dependency_package.required_outputs) <= satisfied
            and dependency_branch is not None
            and _scoped_requirements_are_absent(dependency_branch)
        ):
            continue
        failed.append(dependency)
    if _is_compliance_package(package):
        # A comparison is meaningful only when it has two independently produced
        # inputs: model facts and applicable requirements. Keep this check typed and
        # role-based so it generalizes beyond electrical/BIM vocabulary.
        dependency_packages = [
            getattr(finished[item], "package", None) for item in package.depends_on
        ]
        if not package.depends_on:
            failed.extend(["model-facts input", "requirements input"])
        elif all(dependency is not None for dependency in dependency_packages):
            has_requirements = any(
                dependency.route_hint == "requirements"
                or any(spec.kind == "compliance" for spec in dependency.output_specs)
                for dependency in dependency_packages
            )
            has_model_facts = any(not (
                dependency.route_hint == "requirements"
                or any(spec.kind == "compliance" for spec in dependency.output_specs)
            ) for dependency in dependency_packages)
            if not has_model_facts:
                failed.append("model-facts input")
            if not has_requirements:
                failed.append("requirements input")
    return list(dict.fromkeys(failed))


def _worker_input(
    contract: BimTaskContract,
    package: EvidenceWorkPackage,
    specialist: str,
    dependency_evidence: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize only package-owned facts; root/sibling text is intentionally absent."""
    payload = {
        "specialist": specialist,
        "objective": package.objective,
        "entity_concept": contract.entity_concept,
        "operation": contract.operation,
        "constraints": [item.model_dump(mode="json") for item in contract.constraints],
        "required_outputs": list(package.required_outputs),
        "output_specs": [item.model_dump(mode="json") for item in package.output_specs],
        "route_hint": package.route_hint,
        "dependency_evidence": dependency_evidence or [],
    }
    return (
        "Resolve only this immutable work package. Use only the configured authorized BIM scope. "
        "Do not expose scope identifiers or credentials.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _dependency_evidence_bundle(
    package: EvidenceWorkPackage,
    finished: dict[str, WorkstreamOutcome],
) -> list[dict[str, Any]]:
    """Build an immutable, verified-only bundle for a dependent workstream."""
    bundle: list[dict[str, Any]] = []
    for dependency_id in package.depends_on:
        outcome = finished.get(dependency_id)
        if outcome is None or outcome.branch is None:
            continue
        verification_payloads = [
            json.loads(evidence.payload)
            for evidence in outcome.branch.evidence.values()
            if evidence.kind == "verification"
        ]
        latest = verification_payloads[-1] if verification_payloads else {}
        for check in latest.get("checks") or []:
            if check.get("verified") is not True:
                continue
            plan = check.get("plan") or {}
            claim = check.get("claim") or {}
            bundle.append({
                "source_package_id": dependency_id,
                "evidence_id": str(check.get("evidence_id") or ""),
                "satisfies": list(plan.get("satisfies") or []),
                "claim": {
                    "value": claim.get("value"),
                    "unit": claim.get("unit"),
                    "basis": claim.get("basis"),
                    "method": claim.get("method"),
                    "coverage": claim.get("coverage"),
                    "details": list(claim.get("details") or [])[:20],
                },
            })
    return bundle


def _execution_context(
    discovery: BimRunContext,
    contract: BimTaskContract,
    package: EvidenceWorkPackage,
    handoff: EvidenceHandoff,
    specialist: SpecialistRole | None = None,
) -> BimRunContext:
    """Create a transcript-free context for exactly one terminal specialist."""
    # Mapping and query evidence are immutable package-local checkpoints.  Preserve
    # them across the transcript-free scout -> specialist handoff so a later model
    # failure, repair timeout, or cancellation cannot erase already computed facts.
    selected_mappings = dict(discovery.schema_mappings)
    task_artifact = discovery.artifacts.get("task-contract")
    return BimRunContext(
        bim=discovery.bim,
        scope=discovery.scope,
        graph_contract=discovery.graph_contract,
        question=package.objective,
        run_id=discovery.run_id,
        workstream_id=discovery.workstream_id,
        parent_workstream_id=discovery.parent_workstream_id,
        active_specialist=specialist or resolve_package_specialist(package),
        task_contract=contract,
        evidence=dict(discovery.evidence),
        artifacts={
            artifact_id: artifact
            for artifact_id, artifact in discovery.artifacts.items()
            if artifact_id != "task-contract"
        } | ({"task-contract": task_artifact} if task_artifact is not None else {}),
        schema_mappings=selected_mappings,
        schema_discovery={
            "active_constraint_bindings": [
                item.model_dump(mode="json") for item in handoff.constraint_bindings
            ],
        },
        model_profile=discovery.model_profile,
        knowledge_store=discovery.knowledge_store,
        schema_fingerprint=discovery.schema_fingerprint,
        llm_calls=discovery.llm_calls,
        tool_calls=discovery.tool_calls,
        agent_starts=discovery.agent_starts,
        agent_starts_by_name=dict(discovery.agent_starts_by_name),
        max_llm_calls=discovery.max_llm_calls,
        max_tool_calls=discovery.max_tool_calls,
        max_agent_starts=discovery.max_agent_starts,
        max_starts_per_agent=discovery.max_starts_per_agent,
    )


def _checkpoint_handoff(
    root: BimRunContext,
    branch: BimRunContext,
    package: EvidenceWorkPackage,
) -> EvidenceHandoff | None:
    """Recover a scout that registered a complete mapping at its turn boundary."""
    new_mappings = [
        mapping for mapping_id, mapping in branch.schema_mappings.items()
        if mapping_id not in root.schema_mappings
        and isinstance(mapping, RegisteredSchemaMapping)
    ]
    if len(new_mappings) != 1:
        return None
    return _registered_mapping_handoff(
        branch, new_mappings[0], root.task_contract, package, root.question,
    )


def _numeric_filter_request(text: str) -> tuple[str, str] | None:
    """Parse one explicit numeric comparison from a typed user constraint."""
    normalized = _normalized_route_text(text)
    match = re.search(r"-?\d+(?:[.,]\d+)?", normalized)
    if not match:
        return None
    value = match.group(0).replace(",", ".")
    patterns = [
        ("less_or_equal", (r"<=", r"at most", r"no more than", r"maximum of")),
        ("greater_or_equal", (r">=", r"at least", r"no less than", r"minimum of")),
        ("less_than", (r"(?<!<)<(?![=])", r"less than", r"smaller than", r"below", r"under")),
        ("greater_than", (r"(?<!>)>(?![=])", r"greater than", r"larger than", r"above", r"over")),
    ]
    for operator, terms in patterns:
        if any(re.search(term, normalized) for term in terms):
            return operator, value
    return None


def _registered_mapping_handoff(
    context: BimRunContext,
    mapping: RegisteredSchemaMapping,
    contract: BimTaskContract,
    package: EvidenceWorkPackage,
    question: str,
) -> EvidenceHandoff:
    """Project one validated mapping into the worker's compact typed input."""
    proposal = mapping.proposal
    # The immutable work package is the narrow worker contract. Prefer its typed
    # outputs so a sibling/global output cannot erase this worker's metric or grouping
    # requirements. Legacy packages fall back to the root contract.
    output_specs = package.output_specs or contract.output_specs
    numeric_fields = [
        field.semantic_name for field in proposal.fields if field.data_type == "number"
    ]
    requested_text = " ".join([
        package.objective, *package.required_outputs, question,
    ]).casefold()
    typed_metric_text = _normalized_route_text(" ".join(
        spec.metric for spec in output_specs if spec.metric
    ))
    typed_metric_matches = [
        field.semantic_name
        for field in proposal.fields
        if field.data_type == "number"
        and (
            _route_term_matches(typed_metric_text, field.semantic_name)
            or _route_term_matches(typed_metric_text, field.property)
            or any(_route_term_matches(typed_metric_text, alias) for alias in field.aliases)
        )
    ]
    metric = typed_metric_matches[0] if len(typed_metric_matches) == 1 else ""
    if not metric:
        requested_metric_dimensions = [
            _measurement_dimension(spec.metric)
            for spec in output_specs
            if spec.metric
        ]
        requested_metric_dimensions = [value for value in requested_metric_dimensions if value]
        dimension_matches = [
            field.semantic_name
            for field in proposal.fields
            if field.data_type == "number"
            and _measurement_dimension(
                f"{field.semantic_name} {field.property} {field.unit or ''}"
            ) in requested_metric_dimensions
        ]
        if len(dimension_matches) == 1:
            metric = dimension_matches[0]
    if not metric:
        metric = next(
            (field for field in numeric_fields if field.replace("_", " ") in requested_text),
            "",
        )
    constraints = [
        ResolvedConstraint(
            semantic_field=binding.semantic_name,
            requested_value=binding.user_concept,
            exact_values=[match.value for match in binding.matches],
            mapping_id=mapping.mapping_id,
            applied_as="entity_boundary",
        )
        for binding in proposal.value_bindings
    ]
    constraints.extend(
        ResolvedConstraint(
            semantic_field=binding.semantic_name,
            requested_value=binding.purpose,
            mapping_id=mapping.mapping_id,
            applied_as="relationship",
        )
        for binding in proposal.relationship_bindings
    )
    constraint_bindings: list[ConstraintBinding] = []
    for constraint in contract.constraints:
        level_fields = [
            field for field in proposal.fields
            if field.ontology_kind == "level" or field.semantic_name == "level"
        ]
        constraint_text = _normalized_route_text(
            f"{constraint.concept} {constraint.requested_value}"
        )
        requests_level = any(
            re.search(rf"(?<!\w){term}(?!\w)", constraint_text)
            for term in ("floor", "level", "storey", "story")
        )
        if requests_level and len(level_fields) == 1:
            level_field = level_fields[0]
            rows = context.bim.query(
                f"MATCH (n:`{proposal.label}`) "
                f"WHERE n.`{proposal.source_property}` IN $allowed_sources "
                f"AND n.`{level_field.property}` IS NOT NULL "
                f"RETURN DISTINCT n.`{level_field.property}` AS value "
                "ORDER BY toString(value)",
                {"allowed_sources": context.scope.allowed_sources},
            )
            exact_levels = list(dict.fromkeys(
                str(row["value"])
                for row in rows
                if row.get("value") is not None
                and _level_matches(
                    constraint.requested_value,
                    str(row["value"]),
                    context.graph_contract.level_aliases,
                )
            ))
            if exact_levels:
                constraints.append(ResolvedConstraint(
                    semantic_field=level_field.semantic_name,
                    requested_value=constraint.requested_value,
                    exact_values=exact_levels,
                    mapping_id=mapping.mapping_id,
                    applied_as="filter",
                ))
                constraint_bindings.append(ConstraintBinding(
                    concept=constraint.concept,
                    requested_value=constraint.requested_value,
                    semantic_field=level_field.semantic_name,
                    exact_values=exact_levels,
                    mapping_id=mapping.mapping_id,
                    applied_as="filter",
                    package_id=package.package_id,
                ))
                continue
        target_field = _constraint_field(
            f"{constraint.concept} {constraint.requested_value}"
        )
        field_matches = [
            field for field in proposal.fields
            if any(
                _normalized_route_text(term) == _normalized_route_text(target_field)
                or _normalized_route_text(target_field) in _normalized_route_text(term)
                or _route_term_matches(constraint_text, term)
                for term in [field.semantic_name, *field.aliases]
            )
        ]
        numeric_constraint = _numeric_filter_request(constraint_text)
        if numeric_constraint is not None:
            field_matches = [field for field in field_matches if field.data_type == "number"]
        if numeric_constraint is not None and metric:
            metric_field = next(
                (
                    field for field in proposal.fields
                    if field.semantic_name == metric and field.data_type == "number"
                ),
                None,
            )
            if metric_field is not None:
                field_matches = [metric_field]
        if numeric_constraint is not None and not field_matches:
            requested_dimension = _measurement_dimension(constraint_text)
            dimension_fields = [
                field for field in proposal.fields
                if field.data_type == "number"
                and requested_dimension
                and _measurement_dimension(
                    f"{field.semantic_name} {field.property} {field.unit or ''}"
                ) == requested_dimension
            ]
            if len(dimension_fields) == 1:
                field_matches = dimension_fields
        grouping_request = bool(re.search(
            r"(?<!\w)(?:by|per|each|group|grouping|dimension|breakdown|לפי|בכל)(?!\w)",
            _normalized_route_text(f"{constraint.concept} {constraint.requested_value}"),
            flags=re.UNICODE,
        ))
        if grouping_request and len(field_matches) == 1:
            field = field_matches[0]
            constraints.append(ResolvedConstraint(
                semantic_field=field.semantic_name,
                requested_value=constraint.requested_value,
                mapping_id=mapping.mapping_id,
                applied_as="grouping",
            ))
            constraint_bindings.append(ConstraintBinding(
                concept=constraint.concept,
                requested_value=constraint.requested_value,
                semantic_field=field.semantic_name,
                mapping_id=mapping.mapping_id,
                applied_as="grouping",
                package_id=package.package_id,
            ))
            continue
        if len(field_matches) == 1:
            field = field_matches[0]
            numeric_request = numeric_constraint if field.data_type == "number" else None
            if numeric_request is not None:
                operator, numeric_value = numeric_request
                constraints.append(ResolvedConstraint(
                    semantic_field=field.semantic_name,
                    requested_value=constraint.requested_value,
                    exact_values=[numeric_value],
                    operator=operator,
                    mapping_id=mapping.mapping_id,
                    applied_as="filter",
                ))
                constraint_bindings.append(ConstraintBinding(
                    concept=constraint.concept,
                    requested_value=constraint.requested_value,
                    semantic_field=field.semantic_name,
                    exact_values=[numeric_value],
                    operator=operator,
                    mapping_id=mapping.mapping_id,
                    applied_as="filter",
                    package_id=package.package_id,
                ))
                continue
            governed_bindings = [
                binding for binding in proposal.value_bindings
                if binding.semantic_name == field.semantic_name
            ]
            field_terms = [field.semantic_name, *field.aliases]
            if len(governed_bindings) == 1 and any(
                _route_term_matches(constraint_text, term) for term in field_terms
            ):
                binding = governed_bindings[0]
                constraint_bindings.append(ConstraintBinding(
                    concept=constraint.concept,
                    requested_value=constraint.requested_value,
                    semantic_field=binding.semantic_name,
                    exact_values=[match.value for match in binding.matches],
                    mapping_id=mapping.mapping_id,
                    applied_as="entity_boundary",
                    package_id=package.package_id,
                ))
                continue
        requested = _normalized_route_text(
            f"{constraint.concept} {constraint.requested_value}"
        )
        relationship_candidates = []
        for binding in proposal.relationship_bindings:
            description = _normalized_route_text(
                f"{binding.semantic_name} {binding.purpose}"
            )
            tokens = {
                token for token in re.findall(r"[^\W_]+", requested, flags=re.UNICODE)
                if len(token) >= 3
            }
            score = sum(len(token) for token in tokens if token in description)
            if score:
                relationship_candidates.append((score, binding))
        if relationship_candidates:
            best_relationship_score = max(score for score, _ in relationship_candidates)
            relationship_winners = [
                binding for score, binding in relationship_candidates
                if score == best_relationship_score
            ]
            if len(relationship_winners) == 1:
                binding = relationship_winners[0]
                constraint_bindings.append(ConstraintBinding(
                    concept=constraint.concept,
                    requested_value=constraint.requested_value,
                    semantic_field=binding.semantic_name,
                    mapping_id=mapping.mapping_id,
                    applied_as="relationship",
                    package_id=package.package_id,
                ))
                continue
        candidates: list[tuple[int, object]] = []
        for binding in proposal.value_bindings:
            description = _normalized_route_text(
                f"{binding.semantic_name} {binding.user_concept} {proposal.entity_name}"
            )
            tokens = {
                token for token in re.findall(r"[^\W_]+", requested, flags=re.UNICODE)
                if len(token) >= 3
            }
            score = sum(len(token) for token in tokens if token in description)
            if score:
                candidates.append((score, binding))
        if not candidates:
            continue
        best = max(score for score, _ in candidates)
        winners = [binding for score, binding in candidates if score == best]
        if len(winners) > 1:
            # Governed mappings commonly expose both a broad category boundary
            # and a narrower family/type boundary with overlapping aliases. An
            # entity-type constraint belongs to the narrower classification.
            classification_winners = [
                binding for binding in winners if binding.semantic_name != "category"
            ]
            if len(classification_winners) == 1:
                winners = classification_winners
        if len(winners) != 1:
            continue
        binding = winners[0]
        constraint_bindings.append(ConstraintBinding(
            concept=constraint.concept,
            requested_value=constraint.requested_value,
            semantic_field=binding.semantic_name,
            exact_values=[match.value for match in binding.matches],
            mapping_id=mapping.mapping_id,
            applied_as="entity_boundary",
            package_id=package.package_id,
        ))
    field_summary = {
        field.semantic_name: {"type": field.data_type, "unit": field.unit}
        for field in proposal.fields
    }
    grouping_fields = list(dict.fromkeys(
        item.semantic_field for item in constraint_bindings
        if item.applied_as == "grouping"
    ))
    for spec in output_specs:
        if spec.kind != "grouped_summary":
            continue
        for dimension in spec.grouping_dimensions:
            normalized_dimension = _normalized_route_text(dimension)
            candidates = [
                field.semantic_name for field in proposal.fields
                if any(
                    _normalized_route_text(term) == normalized_dimension
                    or normalized_dimension in _normalized_route_text(term)
                    or _route_term_matches(normalized_dimension, term)
                    for term in [field.semantic_name, *field.aliases]
                )
            ]
            if len(candidates) == 1 and candidates[0] not in grouping_fields:
                grouping_fields.append(candidates[0])
    operation = contract.operation
    aggregate_identity_fields = [
        field.semantic_name for field in proposal.fields
        if field.ontology_kind == "aggregate_identity"
    ]
    requests_aggregate_identity = bool(re.search(
        r"(?<!\w)(?:apartment|apartments|home|homes|dwelling|dwellings|housing unit|housing units|residential unit|residential units|physical unit|physical units)(?!\w)",
        _normalized_route_text(" ".join([
            contract.entity_concept, package.objective, *package.required_outputs,
        ])),
        flags=re.UNICODE,
    ))
    if (
        operation in {"count", "count_distinct"}
        and requests_aggregate_identity
        and len(aggregate_identity_fields) == 1
    ):
        operation = "count_distinct"
        grouping_fields = [aggregate_identity_fields[0]]
    if any(spec.kind == "grouped_summary" for spec in output_specs) and grouping_fields:
        if metric:
            operation = "multi_group_summary" if len(grouping_fields) > 1 else "group_summary"
        else:
            operation = "multi_group_count" if len(grouping_fields) > 1 else "group_count"
    return EvidenceHandoff(
        package_id=package.package_id,
        status="ready_for_query",
        route="live_mapping",
        entity=proposal.entity_name,
        mapping_id=mapping.mapping_id,
        operation=operation,
        metric=metric,
        grouping_fields=grouping_fields,
        constraints=constraints,
        constraint_bindings=constraint_bindings,
        evidence_summary=json.dumps({
            "registered_mapping": mapping.mapping_id,
            "fields": field_summary,
            "counting_unit": proposal.counting_unit,
            "counting_unit_evidence": proposal.counting_unit_evidence,
            "relationships": {
                binding.semantic_name: binding.purpose
                for binding in proposal.relationship_bindings
            },
        }, ensure_ascii=False, sort_keys=True),
    )


def _normalized_route_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", value.casefold())).strip()


def _route_term_matches(text: str, term: str) -> bool:
    normalized_term = _normalized_route_text(term)
    if not normalized_term:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_term)}(?!\w)", text,
        flags=re.UNICODE,
    ) is not None


def _apply_governed_query_profile(
    handoff: EvidenceHandoff,
    mapping: RegisteredSchemaMapping,
    profile: object,
) -> EvidenceHandoff:
    """Apply a validated project query profile to a deterministic mapping route.

    A profile carries domain/project interpretation (for example, a functional-space
    inventory is normally useful as count plus area by function) without embedding any
    result values. The live mapping remains the source of labels, fields, units, and data.
    """
    if profile in (None, {}):
        return handoff
    if not isinstance(profile, dict):
        raise ValueError("A governed query_profile must be an object.")
    allowed_keys = {"operation", "metric", "grouping_fields", "apply_when_operations"}
    unknown_keys = set(profile) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown governed query_profile keys: {sorted(unknown_keys)}")
    operation = str(profile.get("operation") or handoff.operation)
    valid_operations = {
        "count", "count_distinct", "list", "group_count", "group_summary", "distinct",
        "sum", "average", "minimum", "maximum", "maximum_group_sum", "coverage",
        "multi_group_count", "multi_group_summary", "relationship_coverage",
        "project_graph_count",
    }
    if operation not in valid_operations:
        raise ValueError(f"Unsupported governed query_profile operation: {operation!r}.")
    apply_when = [str(item) for item in profile.get("apply_when_operations") or []]
    if any(item not in valid_operations for item in apply_when):
        raise ValueError("A governed query_profile names an invalid triggering operation.")
    if apply_when and handoff.operation not in apply_when:
        return handoff
    fields = {field.semantic_name: field for field in mapping.proposal.fields}
    metric = str(profile.get("metric") or handoff.metric)
    if metric:
        field = fields.get(metric)
        if field is None or field.data_type != "number":
            raise ValueError("A governed query_profile metric must name a mapped numeric field.")
    grouping_fields = [str(item) for item in profile.get("grouping_fields") or handoff.grouping_fields]
    if not grouping_fields or len(grouping_fields) > 3 or len(set(grouping_fields)) != len(grouping_fields):
        raise ValueError("A governed query_profile needs one to three unique grouping fields.")
    if any(item not in fields for item in grouping_fields):
        raise ValueError("A governed query_profile grouping field is absent from the live mapping.")
    if operation in {"group_summary", "multi_group_summary", "maximum_group_sum"} and not metric:
        raise ValueError("A governed grouped-summary query_profile requires a numeric metric.")
    if operation in {"multi_group_count", "multi_group_summary"} and len(grouping_fields) < 2:
        raise ValueError("A multi-group governed query_profile requires multiple grouping fields.")
    if operation in {"group_count", "group_summary", "maximum_group_sum"} and len(grouping_fields) != 1:
        raise ValueError("A single-group governed query_profile requires exactly one grouping field.")
    profile_constraints = [
        ResolvedConstraint(
            semantic_field=item,
            requested_value="governed query profile grouping",
            mapping_id=mapping.mapping_id,
            applied_as="grouping",
        )
        for item in grouping_fields
    ]
    if metric:
        profile_constraints.append(ResolvedConstraint(
            semantic_field=metric,
            requested_value="governed query profile metric",
            mapping_id=mapping.mapping_id,
            applied_as="metric",
        ))
    return handoff.model_copy(update={
        "operation": operation,
        "metric": metric,
        "grouping_fields": grouping_fields,
        "constraints": [*handoff.constraints, *profile_constraints],
    })


def _mapping_field_for_text(
    mapping: RegisteredSchemaMapping,
    text: str,
    *,
    numeric: bool | None = None,
) -> str:
    normalized = _normalized_route_text(text)
    fields = [
        field for field in mapping.proposal.fields
        if numeric is None or (field.data_type == "number") == numeric
    ]
    exact = [
        field.semantic_name for field in fields
        if any(
            _route_term_matches(normalized, term)
            for term in [field.semantic_name, field.property, *field.aliases]
        )
    ]
    if len(exact) == 1:
        return exact[0]
    target = _constraint_field(text)
    semantic = [
        field.semantic_name for field in fields
        if _constraint_field(field.semantic_name) == target
        or target in _normalized_route_text(field.semantic_name).replace(" ", "_")
    ]
    if len(semantic) == 1:
        return semantic[0]
    if numeric:
        dimension = _measurement_dimension(text)
        dimensional = [
            field.semantic_name for field in fields
            if _measurement_dimension(
                f"{field.semantic_name} {field.property} {field.unit or ''}"
            ) == dimension
        ]
        if dimension and len(dimensional) == 1:
            return dimensional[0]
    return ""


def _deterministic_mapping_plans(
    branch: BimRunContext,
    handoff: EvidenceHandoff,
    package: EvidenceWorkPackage,
) -> list[BimQueryPlan] | None:
    """Compile a complete typed mapping handoff without another model turn."""
    mapping = branch.schema_mappings.get(handoff.mapping_id)
    if not isinstance(mapping, RegisteredSchemaMapping):
        return None
    contract = branch.task_contract
    if contract is None:
        return None
    specs = list(contract.output_specs)
    if not specs:
        return None
    filters = []
    seen_filters: set[tuple[str, str, tuple[str, ...]]] = set()
    for constraint in handoff.constraints:
        if constraint.applied_as not in {"filter", "entity_boundary"} or not constraint.exact_values:
            continue
        values = tuple(dict.fromkeys(str(value) for value in constraint.exact_values))
        key = (constraint.semantic_field, constraint.operator, values)
        if key in seen_filters:
            continue
        seen_filters.add(key)
        operator = constraint.operator
        if operator == "in" and len(values) == 1:
            operator = "equals"
        elif operator == "equals" and len(values) > 1:
            operator = "in"
        filters.append(BimFilter(
            field=constraint.semantic_field,
            operator=operator,
            value=values[0] if len(values) == 1 else "",
            values=list(values) if len(values) > 1 else [],
        ))
    bound_constraints = {
        (
            _normalized_route_text(binding.concept),
            _normalized_route_text(binding.requested_value),
        )
        for binding in handoff.constraint_bindings
    }
    unresolved_constraints = [
        constraint for constraint in contract.constraints
        if (
            _normalized_route_text(constraint.concept),
            _normalized_route_text(constraint.requested_value),
        ) not in bound_constraints
    ]
    if unresolved_constraints:
        # The governed entity mapping is still useful evidence, but it must not
        # masquerade as the requested classified/filter population. Return one
        # replayable related-population probe that satisfies no direct output.
        aggregate_identity = next((
            field.semantic_name for field in mapping.proposal.fields
            if field.ontology_kind == "aggregate_identity"
        ), "")
        return [BimQueryPlan(
            entity=handoff.entity,
            mapping_id=handoff.mapping_id,
            operation="count_distinct" if aggregate_identity else "count",
            group_by=aggregate_identity,
            filters=[item.model_copy(deep=True) for item in filters],
            role="supporting",
            include_in_answer=False,
            satisfies=[],
            constraint_bindings=[
                item.model_copy(deep=True) for item in handoff.constraint_bindings
            ],
            work_package_id=package.package_id,
        )]
    plans: list[BimQueryPlan] = []
    numeric_operations = {"sum", "average", "minimum", "maximum", "maximum_group_sum"}
    for output_label, spec in zip(contract.required_outputs, specs, strict=True):
        operation = handoff.operation
        metric = handoff.metric
        grouping_fields = list(handoff.grouping_fields)
        coverage_field = ""
        if spec.kind == "coverage":
            coverage_field = _mapping_field_for_text(mapping, spec.key, numeric=None)
            if not coverage_field:
                return None
            operation = "coverage"
            metric = ""
            grouping_fields = []
        elif spec.kind == "relationship_coverage" or spec.kind == "compliance":
            return None
        elif spec.kind == "count":
            # A typed count describes the shape of the answer, not necessarily
            # the physical identity to count. Preserve a governed aggregate
            # identity (for example dwelling number) instead of falling back to
            # child BIM records such as IfcSpace rows.
            operation = (
                "count_distinct"
                if handoff.operation == "count_distinct" and len(grouping_fields) == 1
                else "count"
            )
            metric = ""
            if operation == "count":
                grouping_fields = []
        elif spec.kind == "grouped_summary":
            resolved_groups = [
                _mapping_field_for_text(mapping, dimension)
                for dimension in spec.grouping_dimensions
            ]
            if spec.grouping_dimensions and any(not item for item in resolved_groups):
                return None
            grouping_fields = list(dict.fromkeys(resolved_groups or grouping_fields))
            if not grouping_fields:
                return None
            if spec.metric:
                metric = _mapping_field_for_text(mapping, spec.metric, numeric=True)
                if not metric:
                    return None
            operation = (
                "multi_group_summary" if metric and len(grouping_fields) > 1 else
                "group_summary" if metric else
                "multi_group_count" if len(grouping_fields) > 1 else
                "group_count"
            )
        elif spec.kind == "measurement":
            if spec.metric:
                metric = _mapping_field_for_text(mapping, spec.metric, numeric=True)
            if not metric:
                return None
            if operation == "distinct":
                # Set-valued numeric measurements (elevations, offsets,
                # diameters, and similar properties) use the numeric field as
                # the distinct value rather than as an aggregate operand.
                grouping_fields = [metric]
                metric = ""
            elif operation in numeric_operations:
                grouping_fields = []
            else:
                return None
        elif spec.kind in {"list", "fact"}:
            if operation in {"group_count", "group_summary", "distinct", "maximum_group_sum"}:
                if len(grouping_fields) != 1:
                    return None
            elif operation in {"multi_group_count", "multi_group_summary"}:
                if len(grouping_fields) < 2:
                    return None
            elif operation not in {"list", "count", "count_distinct", *numeric_operations}:
                operation = "list"
                metric = ""
                grouping_fields = []
        else:
            return None
        plan_values = {
            "entity": handoff.entity,
            "mapping_id": handoff.mapping_id,
            "operation": operation,
            "filters": [item.model_copy(deep=True) for item in filters],
            "metric": metric,
            "coverage_field": coverage_field,
            "limit": 20,
            "answer_key": f"{package.package_id}:{spec.key}",
            "satisfies": [output_label],
            "constraint_bindings": [
                item.model_copy(deep=True) for item in handoff.constraint_bindings
            ],
            "work_package_id": package.package_id,
        }
        if operation in {"group_count", "group_summary", "distinct", "count_distinct", "maximum_group_sum"}:
            plan_values["group_by"] = grouping_fields[0] if grouping_fields else ""
        if operation in {"multi_group_count", "multi_group_summary"}:
            plan_values["group_by_fields"] = grouping_fields
        plans.append(BimQueryPlan(**plan_values))
    return plans


def _deterministic_geometry_plan(
    branch: BimRunContext,
    handoff: EvidenceHandoff,
    package: EvidenceWorkPackage,
) -> GeometryQueryPlan | None:
    """Compile an allowlisted calculation handoff when its typed outputs fit."""
    contract = branch.task_contract
    if contract is None:
        return None
    configs = [
        item for item in branch.bim.ontology.bim_query_knowledge.values()
        if isinstance(item, dict) and item.get("calculation") == handoff.calculation
    ]
    if len(configs) != 1:
        return None
    config = configs[0]
    support_only = bool(config.get("support_only"))
    allowed_kinds = set(config.get("output_kinds") or [])

    configured_dimensions: set[str] | None = None
    if config.get("grouping_dimensions"):
        configured_dimensions = {
            _constraint_field(str(item))
            for item in config.get("grouping_dimensions") or []
        }
    elif config.get("recipe") == "composite_grouped_count":
        configured_dimensions = {
            _constraint_field(str(item.get("group_property") or ""))
            for item in config.get("components") or []
            if isinstance(item, dict) and item.get("group_property")
        }

    def dimensions_fit(spec) -> bool:
        if configured_dimensions is None or not spec.grouping_dimensions:
            return True
        requested = {_constraint_field(item) for item in spec.grouping_dimensions}
        return requested.issubset(configured_dimensions)

    satisfied_outputs = [
        output
        for output, spec in zip(
            package.required_outputs, package.output_specs, strict=True,
        )
        if (not allowed_kinds or spec.kind in allowed_kinds) and dimensions_fit(spec)
    ]
    if not support_only and not satisfied_outputs:
        return None
    configured_intent = config.get("semantic_intent") or {}
    if not isinstance(configured_intent, dict):
        return None
    if "value_origin" not in configured_intent:
        configured_intent = {**configured_intent, "value_origin": "actual"}
    return GeometryQueryPlan(
        calculation=handoff.calculation,
        answer_key=package.package_id,
        satisfies=[] if support_only else satisfied_outputs,
        role="supporting" if support_only else "answer_producing",
        include_in_answer=not support_only,
        semantic_intent=SemanticIntent.model_validate(configured_intent),
    )


def trusted_governed_mapping_handoff(
    root: BimRunContext,
    branch: BimRunContext,
    contract: BimTaskContract,
    package: EvidenceWorkPackage,
) -> EvidenceHandoff | None:
    """Activate one unambiguous project-governed mapping without an LLM scout.

    Package-local language is authoritative for routing. The full question is only a
    fallback for legacy single-package tasks. Ties fail closed to schema discovery.
    Live graph validation remains mandatory inside the activation tool.
    """
    if package.route_hint == "requirements":
        return None
    ontology = getattr(root.bim, "ontology", None)
    knowledge = getattr(ontology, "bim_query_knowledge", {})
    if not isinstance(knowledge, dict):
        return None
    local_text = _normalized_route_text(" ".join([
        contract.entity_concept, package.objective, *package.required_outputs,
    ]))
    fallback_text = _normalized_route_text(root.question)
    matches: list[tuple[int, int, str]] = []
    for key, config in knowledge.items():
        if not isinstance(config, dict) or config.get("calculation"):
            continue
        # Only complete governed mappings may enter the deterministic route.
        if not config.get("entity_concept") or not config.get("exact_family_values"):
            continue
        requested_grain_text = _normalized_route_text(" ".join([
            contract.entity_concept,
            package.objective,
            *package.required_outputs,
            contract.semantic_intent.entity_grain,
            package.semantic_intent.entity_grain,
            *(spec.semantic_intent.entity_grain for spec in package.output_specs),
        ]))
        configured_counting_unit = _normalized_route_text(
            str(config.get("counting_unit") or "")
        )
        requests_individual_dwelling = bool(re.search(
            r"(?<!\w)(?:apartment|home|dwelling|housing unit|residential unit)(?!\w)",
            requested_grain_text,
            flags=re.UNICODE,
        ))
        configured_as_floor_plate = bool(re.search(
            r"(?<!\w)(?:floor plate|floorplate)(?!\w)",
            configured_counting_unit,
            flags=re.UNICODE,
        ))
        if requests_individual_dwelling and configured_as_floor_plate:
            # A lexically attractive area-plan mapping is still the wrong entity
            # for per-home thresholds and counts. Let the apartment/unit mapping
            # compete instead of returning a replayable answer at the wrong grain.
            continue
        terms = {
            str(config.get("entity_concept") or "").strip(),
            *(str(item).strip() for item in config.get("aliases") or []),
            *(str(item).strip() for item in config.get("retrieval_terms") or []),
        }
        local_hits = [term for term in terms if _route_term_matches(local_text, term)]
        fallback_hits = [term for term in terms if _route_term_matches(fallback_text, term)]
        if local_hits:
            matches.append((2, max(len(_normalized_route_text(x)) for x in local_hits), str(key)))
        elif fallback_hits:
            matches.append((1, max(len(_normalized_route_text(x)) for x in fallback_hits), str(key)))
    if not matches:
        return None
    best_rank = max((tier, term_length) for tier, term_length, _ in matches)
    winners = [key for tier, term_length, key in matches if (tier, term_length) == best_rank]
    if len(winners) != 1:
        return None
    try:
        activated = activate_project_knowledge_mapping(PipelineContext(branch), winners[0])
        mapping = RegisteredSchemaMapping.model_validate_json(activated)
    except (TypeError, ValueError, json.JSONDecodeError):
        # Stale or invalid governed knowledge is never silently trusted. The scout may
        # investigate the changed schema but cannot receive a partially validated route.
        return None
    handoff = _registered_mapping_handoff(
        branch, mapping, contract, package, root.question,
    )
    try:
        return _apply_governed_query_profile(
            handoff, mapping, knowledge[winners[0]].get("query_profile"),
        )
    except (TypeError, ValueError):
        # A stale profile never enters execution; schema discovery remains the safe fallback.
        return None


def trusted_geometry_handoff(
    root: BimRunContext, package: EvidenceWorkPackage,
) -> EvidenceHandoff | None:
    """Resolve an exact project-governed calculation before open-ended discovery."""
    if package.route_hint not in {"auto", "geometry", "live_schema"}:
        return None
    local_text = _normalized_route_text(" ".join([
        package.objective, *package.required_outputs,
    ]))
    fallback_text = _normalized_route_text(root.question)
    matches: list[tuple[int, int, str, dict]] = []
    ontology = getattr(root.bim, "ontology", None)
    knowledge = getattr(ontology, "bim_query_knowledge", {})
    for config in knowledge.values():
        if not isinstance(config, dict) or not config.get("calculation"):
            continue
        terms = [str(term).strip() for term in config.get("route_terms") or []]
        local_hits = [term for term in terms if _route_term_matches(local_text, term)]
        fallback_hits = [term for term in terms if _route_term_matches(fallback_text, term)]
        if local_hits:
            matches.append((
                2, max(len(_normalized_route_text(term)) for term in local_hits),
                str(config["calculation"]), config,
            ))
        elif fallback_hits:
            matches.append((
                1, max(len(_normalized_route_text(term)) for term in fallback_hits),
                str(config["calculation"]), config,
            ))
    if not matches:
        return None
    best_rank = max((tier, length) for tier, length, _, _ in matches)
    winners = [
        (calculation, config)
        for tier, length, calculation, config in matches
        if (tier, length) == best_rank
    ]
    if len({calculation for calculation, _ in winners}) != 1:
        return None
    calculation, config = winners[0]
    return EvidenceHandoff(
        package_id=package.package_id,
        status="ready_for_query",
        route="geometry",
        calculation=calculation,
        operation=root.task_contract.operation if root.task_contract else "",
        evidence_summary=str(
            config.get("semantics") or "Project-governed deterministic geometry calculation."
        ),
    )


def _scoped_requirements_are_absent(root: BimRunContext) -> bool:
    """Return true only after a committed exhaustive requirements query found no rows."""
    for evidence in root.evidence.values():
        if evidence.kind != "query":
            continue
        try:
            payload = json.loads(evidence.payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        plan = payload.get("plan") or {}
        claim = payload.get("claim") or {}
        coverage = claim.get("coverage") or {}
        if (
            plan.get("entity") == "permit_knowledge"
            and int(claim.get("value") or 0) == 0
            and bool(coverage.get("exhaustive"))
            and int(coverage.get("candidate_count") or 0) == 0
        ):
            return True
    return False


def _deterministic_requirements_plan(
    root: BimRunContext,
    contract: BimTaskContract,
    package: EvidenceWorkPackage,
) -> BimQueryPlan | None:
    """Compile requirements availability and absence-based compliance outcomes."""
    compliance = _is_compliance_package(package)
    if compliance:
        if not _scoped_requirements_are_absent(root):
            return None
    elif package.route_hint != "requirements":
        return None
    if any(spec.kind not in {"fact", "list", "compliance"} for spec in package.output_specs):
        return None
    return BimQueryPlan(
        entity="permit_knowledge",
        operation="list",
        select=["name", "summary", "knowledge"],
        limit=20,
        answer_key=f"{package.package_id}:requirements_availability",
        satisfies=list(package.required_outputs),
        work_package_id=package.package_id,
        semantic_intent=SemanticIntent(
            value_origin="comparison" if compliance else "planned",
        ),
    )


async def _run_one_workstream(
    *,
    settings: Settings,
    root: BimRunContext,
    registry: BimAgentRegistry,
    package: EvidenceWorkPackage,
    dependency_evidence: list[dict[str, Any]],
    package_count: int,
    budget_slot: int,
    events: PipelineEvents,
    semaphore: asyncio.Semaphore,
    active_branches: dict[str, BimRunContext] | None = None,
) -> WorkstreamOutcome:
    async with semaphore:
        branch_bim = BimContext(settings)
        workstream_id = package.package_id
        branch: BimRunContext | None = None
        specialist_role = ""
        specialist_attempts = 0
        typed_output_failures = 0
        verification_rejections = 0
        try:
            specialist_role = resolve_package_specialist(package)
            await asyncio.to_thread(branch_bim.connect)
            branch = BimRunContext(
                bim=branch_bim,
                scope=root.scope.model_copy(deep=True),
                graph_contract=root.graph_contract,
                question=package.objective,
                run_id=root.run_id,
                workstream_id=workstream_id,
                parent_workstream_id=root.workstream_id,
                active_specialist=specialist_role,
                schema_mappings=dict(root.schema_mappings),
                schema_fingerprint=root.schema_fingerprint,
                model_profile=root.model_profile,
                learned_knowledge=dict(root.learned_knowledge),
                max_llm_calls=_branch_limit(
                    root.max_llm_calls, root.llm_calls, package_count, budget_slot
                ),
                max_tool_calls=_branch_limit(
                    root.max_tool_calls, root.tool_calls, package_count, budget_slot
                ),
                max_agent_starts=_branch_limit(
                    root.max_agent_starts, root.agent_starts, package_count, budget_slot
                ),
                max_starts_per_agent=root.max_starts_per_agent,
            )
            if active_branches is not None:
                active_branches[workstream_id] = branch
            worker_contract = package_contract(root.task_contract, package)
            define_bim_task(PipelineContext(branch), worker_contract)
            hooks = AgentRunHooks(events)

            async def run_scout(correction: dict[str, Any] | None = None) -> EvidenceHandoff:
                scout_input = _worker_input(
                    worker_contract, package, "schema_mapping", dependency_evidence,
                )
                if correction:
                    scout_input += (
                        "\nCorrective exploration request:\n"
                        "A prior governed route produced only supporting or rejected evidence. "
                        "Independently inspect alternative fields, values, relationships, or "
                        "measurement bases for the unresolved outputs. Do not repeat a rejected "
                        "mapping without new evidence.\n"
                        + json.dumps(correction, ensure_ascii=False, sort_keys=True)
                    )
                try:
                    if correction:
                        async with asyncio.timeout(_repair_step_timeout()):
                            result = await Runner.run(
                                registry.schema_scout,
                                scout_input,
                                context=branch,
                                max_turns=int(os.getenv("BIM_SCOUT_MAX_TURNS", "18")),
                                hooks=hooks,
                            )
                    else:
                        result = await Runner.run(
                            registry.schema_scout,
                            scout_input,
                            context=branch,
                            max_turns=int(os.getenv("BIM_SCOUT_MAX_TURNS", "18")),
                            hooks=hooks,
                        )
                    resolved = EvidenceHandoff.model_validate(result.final_output)
                    if resolved.package_id != package.package_id:
                        raise ValueError(
                            "Schema scout returned a handoff for a different work package."
                        )
                    return resolved
                except MaxTurnsExceeded:
                    checkpoint = _checkpoint_handoff(root, branch, package)
                    if checkpoint is None:
                        raise
                    events.stage(
                        "scout_checkpoint_recovered", run_id=root.run_id,
                        workstream_id=workstream_id, mapping_id=checkpoint.mapping_id,
                    )
                    return checkpoint

            events.stage(
                "workstream_start", run_id=root.run_id,
                workstream_id=workstream_id, outputs=len(package.required_outputs),
                specialist=specialist_role,
            )
            requirements_plan = _deterministic_requirements_plan(
                root, worker_contract, package,
            )
            if requirements_plan is not None:
                evidence_id = json.loads(query_bim(
                    PipelineContext(branch), requirements_plan,
                ))["evidence_id"]
                verified_ids, satisfied_outputs, _ = _package_verification_snapshot(
                    branch, package,
                )
                missing_outputs = [
                    output for output in package.required_outputs
                    if output not in satisfied_outputs
                ]
                events.stage(
                    "trusted_requirements_executed", run_id=root.run_id,
                    workstream_id=workstream_id,
                    outcome=(
                        "compliance_not_assessable"
                        if _is_compliance_package(package)
                        else "requirements_availability"
                    ),
                )
                return WorkstreamOutcome(
                    package, branch,
                    "partial_completed" if missing_outputs else "query_completed",
                    evidence_ids=verified_ids or [evidence_id],
                    limitations=(
                        [
                            "The scoped requirements query did not semantically verify: "
                            + ", ".join(missing_outputs)
                            + "."
                        ] if missing_outputs else []
                    ),
                    specialist=specialist_role,
                    satisfied_outputs=satisfied_outputs,
                )
            handoff = trusted_geometry_handoff(root, package)
            if handoff is not None:
                events.stage(
                    "trusted_route_selected", run_id=root.run_id,
                    workstream_id=workstream_id, route="geometry",
                    calculation=handoff.calculation,
                )
            else:
                handoff = trusted_governed_mapping_handoff(
                    root, branch, worker_contract, package,
                )
            if handoff is not None and handoff.route == "live_mapping":
                events.stage(
                    "trusted_route_selected", run_id=root.run_id,
                    workstream_id=workstream_id, route="live_mapping",
                    mapping_id=handoff.mapping_id,
                )
            if handoff is None:
                handoff = await run_scout()
            if handoff.package_id != package.package_id:
                raise ValueError("Schema scout returned a handoff for a different work package.")
            if handoff.status == "unsupported":
                events.stage(
                    "workstream_unsupported", run_id=root.run_id,
                    workstream_id=workstream_id,
                )
                return WorkstreamOutcome(
                    package, branch, "unsupported", limitations=list(handoff.limitations),
                    specialist=specialist_role,
                )
            branch.schema_discovery["active_constraint_bindings"] = [
                item.model_dump(mode="json") for item in handoff.constraint_bindings
            ]
            deterministic_geometry = (
                _deterministic_geometry_plan(branch, handoff, package)
                if handoff.route == "geometry" else None
            )
            if deterministic_geometry is not None:
                evidence_id = json.loads(query_project_geometry(
                    PipelineContext(branch), deterministic_geometry,
                ))["evidence_id"]
                verified_ids, satisfied_outputs, verification_failures = (
                    _package_verification_snapshot(branch, package)
                )
                missing_outputs = [
                    output for output in package.required_outputs
                    if output not in satisfied_outputs
                ]
                events.stage(
                    "trusted_geometry_executed", run_id=root.run_id,
                    workstream_id=workstream_id, calculation=handoff.calculation,
                    status="partial_completed" if missing_outputs else "query_completed",
                )
                if not missing_outputs:
                    return WorkstreamOutcome(
                        package, branch, "query_completed", evidence_ids=verified_ids,
                        specialist=specialist_role,
                        satisfied_outputs=satisfied_outputs,
                    )
                handoff = await run_scout({
                    "prior_route": "geometry",
                    "retained_evidence_ids": [evidence_id],
                    "verified_outputs": satisfied_outputs,
                    "missing_outputs": missing_outputs,
                    "verification_failures": verification_failures[-4:],
                })
                events.stage(
                    "partial_route_reopened", run_id=root.run_id,
                    workstream_id=workstream_id, prior_route="geometry",
                    missing_outputs=len(missing_outputs),
                )
                if handoff.status == "unsupported":
                    return WorkstreamOutcome(
                        package, branch, "partial_completed", evidence_ids=verified_ids,
                        limitations=[
                            f"Workstream {workstream_id} preserved a governed calculation "
                            "but corrective exploration could not establish: "
                            + ", ".join(missing_outputs)
                            + ".",
                            *handoff.limitations,
                        ],
                        specialist=specialist_role,
                        recovery_strategy="supporting_calculation_checkpoint",
                        satisfied_outputs=satisfied_outputs,
                    )
            deterministic_plans = (
                _deterministic_mapping_plans(branch, handoff, package)
                if handoff.route == "live_mapping" else None
            )
            if deterministic_plans is not None:
                # The trusted route is more specific than an architect's advisory
                # hint. A stored numeric offset may be described as "geometry" by
                # the user/model, but once a live mapping compiles it is an ordinary
                # quantity query and must execute under that policy boundary.
                specialist_role = "quantity"
                branch.active_specialist = "quantity"
                evidence_ids = [
                    json.loads(query_bim(PipelineContext(branch), plan))["evidence_id"]
                    for plan in deterministic_plans
                ]
                verified_ids, satisfied_outputs, verification_failures = (
                    _package_verification_snapshot(branch, package)
                )
                missing_outputs = [
                    output for output in package.required_outputs
                    if output not in satisfied_outputs
                ]
                events.stage(
                    "trusted_mapping_executed", run_id=root.run_id,
                    workstream_id=workstream_id, plans=len(deterministic_plans),
                    status="partial_completed" if missing_outputs else "query_completed",
                )
                if not missing_outputs:
                    return WorkstreamOutcome(
                        package, branch, "query_completed", evidence_ids=verified_ids,
                        specialist=specialist_role,
                        satisfied_outputs=satisfied_outputs,
                    )
                handoff = await run_scout({
                    "prior_route": "live_mapping",
                    "prior_mapping_id": handoff.mapping_id,
                    "retained_evidence_ids": evidence_ids,
                    "verified_outputs": satisfied_outputs,
                    "missing_outputs": missing_outputs,
                    "verification_failures": verification_failures[-4:],
                })
                events.stage(
                    "partial_route_reopened", run_id=root.run_id,
                    workstream_id=workstream_id, prior_route="live_mapping",
                    missing_outputs=len(missing_outputs),
                )
                if handoff.status == "unsupported":
                    unbound = [
                        f"{item.concept}={item.requested_value}"
                        for item in worker_contract.constraints
                        if (
                            _normalized_route_text(item.concept),
                            _normalized_route_text(item.requested_value),
                        ) not in {
                            (
                                _normalized_route_text(binding.concept),
                                _normalized_route_text(binding.requested_value),
                            )
                            for binding in handoff.constraint_bindings
                        }
                    ]
                    limitation = (
                        f"Workstream {workstream_id} retained verified supporting evidence, "
                        "but corrective exploration could not establish: "
                        + ", ".join(missing_outputs)
                        + "."
                    )
                    if unbound:
                        limitation += " Unresolved constraints: " + ", ".join(unbound) + "."
                    return WorkstreamOutcome(
                        package, branch, "partial_completed", evidence_ids=verified_ids,
                        limitations=[limitation, *handoff.limitations],
                        specialist=specialist_role,
                        recovery_strategy="supporting_population_checkpoint",
                        satisfied_outputs=satisfied_outputs,
                    )
            if handoff.route != "geometry" and specialist_role == "geometry":
                specialist_role = "quantity"
                branch.active_specialist = "quantity"
            query_input = (
                _worker_input(
                    worker_contract, package, specialist_role, dependency_evidence,
                )
                + "\nSchema handoff:\n"
                + handoff.model_dump_json()
            )
            branch = _execution_context(
                branch, worker_contract, package, handoff,
                specialist=specialist_role,
            )
            if active_branches is not None:
                active_branches[workstream_id] = branch
            specialist_agent = registry.execution_specialist(specialist_role)
            events.stage(
                "specialist_selected", run_id=root.run_id,
                workstream_id=workstream_id, specialist=specialist_role,
                agent=specialist_agent.name,
            )
            completion: EvidenceWorkstreamResult | None = None
            last_validation_error: Exception | None = None
            current_input = query_input
            verification_repairs = 0
            max_attempts = 1 + _typed_repair_limit() + _verification_repair_limit()
            for attempt in range(max_attempts):
                specialist_attempts += 1
                query_result = await Runner.run(
                    specialist_agent,
                    current_input,
                    context=branch,
                    max_turns=int(os.getenv(
                        f"BIM_{specialist_role.upper()}_MAX_TURNS",
                        os.getenv("BIM_QUERY_WORKER_MAX_TURNS", "6"),
                    )),
                    hooks=hooks,
                )
                try:
                    completion = _validate_workstream_completion(
                        query_result.final_output, branch=branch, package=package,
                    )
                    if completion.status == "unsupported":
                        break
                    verified_ids, verified_outputs, verification_failures = (
                        _package_verification_snapshot(branch, package)
                    )
                    missing_verified_outputs = [
                        output for output in package.required_outputs
                        if output not in verified_outputs
                    ]
                    if not missing_verified_outputs:
                        completion = completion.model_copy(update={
                            "evidence_ids": verified_ids,
                        })
                        break
                    events.stage(
                        "workstream_verification_rejected",
                        run_id=root.run_id,
                        workstream_id=workstream_id,
                        specialist=specialist_role,
                        attempt=attempt + 1,
                        verified_outputs=len(verified_outputs),
                        missing_outputs=len(missing_verified_outputs),
                        failed_evidence=len(verification_failures),
                    )
                    verification_rejections += 1
                    if verification_repairs >= _verification_repair_limit():
                        last_validation_error = ValueError(
                            "Evidence verification did not satisfy outputs: "
                            + ", ".join(missing_verified_outputs)
                            + "."
                        )
                        completion = None
                        break
                    verification_repairs += 1
                    if _verification_requires_remapping(verification_failures):
                        corrected_handoff = await run_scout({
                            "prior_route": handoff.route,
                            "prior_mapping_id": handoff.mapping_id,
                            "verified_outputs": verified_outputs,
                            "missing_outputs": missing_verified_outputs,
                            "verification_failures": verification_failures[-4:],
                        })
                        if corrected_handoff.status == "unsupported":
                            completion = EvidenceWorkstreamResult(
                                status="unsupported",
                                package_id=package.package_id,
                                limitations=list(corrected_handoff.limitations),
                            )
                            events.stage(
                                "workstream_remapping_unsupported", run_id=root.run_id,
                                workstream_id=workstream_id,
                                missing_outputs=len(missing_verified_outputs),
                            )
                            break
                        handoff = corrected_handoff
                        branch.schema_discovery["active_constraint_bindings"] = [
                            item.model_dump(mode="json")
                            for item in handoff.constraint_bindings
                        ]
                        if handoff.route != "geometry" and specialist_role == "geometry":
                            specialist_role = "quantity"
                            branch.active_specialist = "quantity"
                            specialist_agent = registry.execution_specialist(specialist_role)
                        query_input = (
                            _worker_input(
                                worker_contract, package, specialist_role,
                                dependency_evidence,
                            )
                            + "\nSchema handoff:\n"
                            + handoff.model_dump_json()
                        )
                        events.stage(
                            "workstream_remapped_after_verification", run_id=root.run_id,
                            workstream_id=workstream_id,
                            route=handoff.route,
                            mapping_id=handoff.mapping_id,
                        )
                    current_input = _verification_repair_input(
                        query_input,
                        package=package,
                        verified_ids=verified_ids,
                        verified_outputs=verified_outputs,
                        failures=verification_failures,
                    )
                    completion = None
                    events.stage(
                        "workstream_verification_retry",
                        run_id=root.run_id,
                        workstream_id=workstream_id,
                        specialist=specialist_role,
                        next_attempt=attempt + 2,
                    )
                    continue
                except Exception as validation_error:
                    typed_output_failures += 1
                    last_validation_error = validation_error
                    retained_ids, satisfied_outputs = _answer_evidence_snapshot(branch, package)
                    missing_outputs = [
                        output for output in package.required_outputs
                        if output not in satisfied_outputs
                    ]
                    events.stage(
                        "workstream_typed_output_invalid", run_id=root.run_id,
                        workstream_id=workstream_id, specialist=specialist_role,
                        attempt=attempt + 1, error_type=type(validation_error).__name__,
                        retained_evidence=len(retained_ids),
                        satisfied_outputs=len(satisfied_outputs),
                        missing_outputs=len(missing_outputs),
                    )
                    if retained_ids and not missing_outputs:
                        events.stage(
                            "workstream_typed_output_recovered", run_id=root.run_id,
                            workstream_id=workstream_id,
                            strategy="deterministic_evidence_checkpoint",
                            evidence=len(retained_ids), attempts=specialist_attempts,
                        )
                        return WorkstreamOutcome(
                            package, branch, "query_completed", evidence_ids=retained_ids,
                            specialist=specialist_role,
                            specialist_attempts=specialist_attempts,
                            typed_output_failures=typed_output_failures,
                            verification_rejections=verification_rejections,
                            recovery_strategy="deterministic_evidence_checkpoint",
                            satisfied_outputs=satisfied_outputs,
                        )
                    if typed_output_failures > _typed_repair_limit():
                        break
                    current_input = _repair_input(
                        query_input,
                        package=package,
                        error=validation_error,
                        evidence_ids=retained_ids,
                        satisfied_outputs=satisfied_outputs,
                    )
                    events.stage(
                        "workstream_typed_output_retry", run_id=root.run_id,
                        workstream_id=workstream_id, specialist=specialist_role,
                        next_attempt=attempt + 2,
                    )
            if completion is None:
                retained_ids, satisfied_outputs, _ = _package_verification_snapshot(
                    branch, package,
                )
                if retained_ids:
                    missing_outputs = [
                        output for output in package.required_outputs
                        if output not in satisfied_outputs
                    ]
                    status = "partial_completed" if missing_outputs else "query_completed"
                    limitations = []
                    if missing_outputs:
                        limitations.append(
                            f"Workstream {workstream_id} retained completed evidence but did not "
                            "complete outputs: " + ", ".join(missing_outputs) + "."
                        )
                    events.stage(
                        "workstream_typed_output_recovered", run_id=root.run_id,
                        workstream_id=workstream_id,
                        strategy="partial_evidence_checkpoint", status=status,
                        evidence=len(retained_ids), attempts=specialist_attempts,
                    )
                    return WorkstreamOutcome(
                        package, branch, status, evidence_ids=retained_ids,
                        limitations=limitations, error=last_validation_error,
                        specialist=specialist_role,
                        specialist_attempts=specialist_attempts,
                        typed_output_failures=typed_output_failures,
                        verification_rejections=verification_rejections,
                        recovery_strategy="partial_evidence_checkpoint",
                        satisfied_outputs=satisfied_outputs,
                    )
                assert last_validation_error is not None
                raise last_validation_error
            retained_ids, satisfied_outputs, _ = _package_verification_snapshot(
                branch, package,
            )
            events.stage(
                "workstream_end", run_id=root.run_id, workstream_id=workstream_id,
                status=completion.status, evidence=len(completion.evidence_ids),
                specialist=specialist_role, attempts=specialist_attempts,
                typed_output_failures=typed_output_failures,
                verification_rejections=verification_rejections,
                satisfied_outputs=len(satisfied_outputs),
            )
            return WorkstreamOutcome(
                package,
                branch,
                completion.status,
                evidence_ids=list(completion.evidence_ids),
                limitations=list(completion.limitations),
                specialist=specialist_role,
                specialist_attempts=specialist_attempts,
                typed_output_failures=typed_output_failures,
                verification_rejections=verification_rejections,
                satisfied_outputs=satisfied_outputs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if branch is not None:
                retained_ids, satisfied_outputs, _ = _package_verification_snapshot(
                    branch, package,
                )
                if retained_ids:
                    missing_outputs = [
                        output for output in package.required_outputs
                        if output not in satisfied_outputs
                    ]
                    status = "partial_completed" if missing_outputs else "query_completed"
                    strategy = "exception_evidence_checkpoint"
                    limitation = (
                        f"Workstream {workstream_id} retained completed evidence after "
                        f"{type(exc).__name__}."
                    )
                    if missing_outputs:
                        limitation += " Incomplete outputs: " + ", ".join(missing_outputs) + "."
                    events.stage(
                        "workstream_exception_recovered", run_id=root.run_id,
                        workstream_id=workstream_id, specialist=specialist_role,
                        status=status, error_type=type(exc).__name__,
                        evidence=len(retained_ids), satisfied_outputs=len(satisfied_outputs),
                    )
                    return WorkstreamOutcome(
                        package, branch, status, evidence_ids=retained_ids,
                        limitations=[limitation], error=exc,
                        specialist=specialist_role,
                        specialist_attempts=specialist_attempts,
                        typed_output_failures=typed_output_failures,
                        verification_rejections=verification_rejections,
                        recovery_strategy=strategy,
                        satisfied_outputs=satisfied_outputs,
                    )
            category = "turn_limit" if isinstance(exc, MaxTurnsExceeded) else "error"
            events.stage(
                "workstream_failed", run_id=root.run_id,
                workstream_id=workstream_id, category=category,
                error_type=type(exc).__name__,
            )
            limitation = (
                f"Workstream {workstream_id} reached its bounded model-turn limit; "
                "any completed evidence was retained."
                if isinstance(exc, MaxTurnsExceeded) else
                f"Workstream {workstream_id} failed before producing a valid typed result."
            )
            return WorkstreamOutcome(
                package, branch, category, limitations=[limitation], error=exc,
                specialist=specialist_role,
                specialist_attempts=specialist_attempts,
                typed_output_failures=typed_output_failures,
                verification_rejections=verification_rejections,
            )
        finally:
            await asyncio.to_thread(branch_bim.close)


async def run_evidence_workstreams(
    *,
    settings: Settings,
    root: BimRunContext,
    registry: BimAgentRegistry,
    events: PipelineEvents,
    merge: Callable[[BimRunContext, BimRunContext], list[str]],
    active_branches: dict[str, BimRunContext] | None = None,
) -> list[WorkstreamOutcome]:
    """Schedule the architect DAG with isolated state and deterministic commits."""
    packages = evidence_work_packages(root.task_contract)
    maximum = max(1, int(os.getenv("BIM_MAX_PARALLEL_WORKERS", "3")))
    semaphore = asyncio.Semaphore(min(maximum, len(packages)))
    pending = {item.package_id: item for item in packages}
    budget_slots = {item.package_id: index for index, item in enumerate(packages)}
    finished: dict[str, WorkstreamOutcome] = {}
    ordered_outcomes: list[WorkstreamOutcome] = []

    while pending:
        ready = [
            item for item in packages
            if item.package_id in pending
            and all(dependency in finished for dependency in item.depends_on)
        ]
        if not ready:
            raise ValueError("The evidence work-package graph contains a dependency cycle.")
        runnable: list[EvidenceWorkPackage] = []
        for item in ready:
            failed_dependencies = _failed_dependencies(item, finished)
            if failed_dependencies:
                outcome = WorkstreamOutcome(
                    item, None, "dependency_blocked",
                    limitations=[
                        f"Workstream {item.package_id} was blocked by: "
                        + ", ".join(failed_dependencies)
                        + f" (dependency policy: {_dependency_policy(item)})."
                    ],
                )
                finished[item.package_id] = outcome
                ordered_outcomes.append(outcome)
                pending.pop(item.package_id)
            else:
                runnable.append(item)
        results = await asyncio.gather(*(
            _run_one_workstream(
                settings=settings,
                root=root,
                registry=registry,
                package=item,
                dependency_evidence=_dependency_evidence_bundle(item, finished),
                package_count=len(packages),
                budget_slot=budget_slots[item.package_id],
                events=events,
                semaphore=semaphore,
                active_branches=active_branches,
            )
            for item in runnable
        ))
        for outcome in results:
            if outcome.branch is not None:
                outcome.evidence_ids = merge(root, outcome.branch)
                if outcome.status in {"query_completed", "partial_completed"} and not outcome.evidence_ids:
                    outcome.status = "merge_rejected"
                    outcome.limitations.append(
                        f"Workstream {outcome.package.package_id} produced no committable answer evidence."
                    )
                events.stage(
                    "workstream_commit", run_id=root.run_id,
                    workstream_id=outcome.package.package_id,
                    status=outcome.status, evidence=len(outcome.evidence_ids),
                    specialist=outcome.specialist,
                    attempts=outcome.specialist_attempts,
                    typed_output_failures=outcome.typed_output_failures,
                    recovery=outcome.recovery_strategy or "none",
                    satisfied_outputs=len(outcome.satisfied_outputs),
                    required_outputs=len(outcome.package.required_outputs),
                )
                if active_branches is not None:
                    active_branches.pop(outcome.package.package_id, None)
            finished[outcome.package.package_id] = outcome
            ordered_outcomes.append(outcome)
            pending.pop(outcome.package.package_id)
    return ordered_outcomes
