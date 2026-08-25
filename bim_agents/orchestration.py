from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from typing import Callable

from agents import MaxTurnsExceeded, Runner

from bim_context import BimContext, Settings

from .models import (
    BimRunContext,
    BimTaskContract,
    EvidenceHandoff,
    EvidenceWorkPackage,
    EvidenceWorkstreamResult,
    ResolvedConstraint,
)
from .observability import AgentRunHooks, PipelineEvents
from .registry import BimAgentRegistry
from .schema_mapping import RegisteredSchemaMapping
from .tools import PipelineContext, define_bim_task


@dataclass
class WorkstreamOutcome:
    package: EvidenceWorkPackage
    branch: BimRunContext | None
    status: str
    evidence_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    error: Exception | None = None


def evidence_work_packages(contract: BimTaskContract) -> list[EvidenceWorkPackage]:
    """Return the architect DAG, with a safe single-package legacy fallback."""
    if contract.work_packages:
        return list(contract.work_packages)
    return [EvidenceWorkPackage(
        package_id="primary",
        objective=contract.goal,
        required_outputs=list(contract.required_outputs),
    )]


def package_contract(
    contract: BimTaskContract, package: EvidenceWorkPackage,
) -> BimTaskContract:
    """Project the global definition of done into one worker-sized contract."""
    return BimTaskContract(
        goal=package.objective,
        operation=contract.operation,
        entity_concept=contract.entity_concept,
        constraints=list(contract.constraints),
        questions_to_resolve=list(contract.questions_to_resolve),
        required_outputs=list(package.required_outputs),
        success_criteria=[
            criterion for criterion in contract.success_criteria
            if any(output.casefold() in criterion.casefold() for output in package.required_outputs)
        ] or [f"Produce replayable evidence for {output}." for output in package.required_outputs],
        complexity=contract.complexity,
    )


def _branch_limit(total: int, consumed: int, workstream_count: int, minimum: int) -> int:
    remaining = max(total - consumed, workstream_count)
    return max(minimum, remaining // max(workstream_count, 1))


def _worker_input(
    question: str,
    contract: BimTaskContract,
    package: EvidenceWorkPackage,
) -> str:
    payload = {
        "question": question,
        "entity_concept": contract.entity_concept,
        "operation": contract.operation,
        "constraints": [item.model_dump(mode="json") for item in contract.constraints],
        "work_package": package.model_dump(mode="json"),
    }
    return (
        "Resolve only this immutable work package. Use only the configured authorized BIM scope. "
        "Do not expose scope identifiers or credentials.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
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
    mapping = new_mappings[0]
    proposal = mapping.proposal
    numeric_fields = [
        field.semantic_name for field in proposal.fields if field.data_type == "number"
    ]
    requested_text = " ".join([
        package.objective, *package.required_outputs, root.question,
    ]).casefold()
    metric = next(
        (field for field in numeric_fields if field.replace("_", " ") in requested_text),
        "",
    )
    constraints = [
        ResolvedConstraint(
            semantic_field=binding.semantic_name,
            requested_value=binding.user_concept,
            exact_values=[match.value for match in binding.matches],
        )
        for binding in proposal.value_bindings
    ]
    field_summary = {
        field.semantic_name: {"type": field.data_type, "unit": field.unit}
        for field in proposal.fields
    }
    return EvidenceHandoff(
        package_id=package.package_id,
        status="ready_for_query",
        route="live_mapping",
        entity=proposal.entity_name,
        mapping_id=mapping.mapping_id,
        operation=root.task_contract.operation,
        metric=metric,
        constraints=constraints,
        evidence_summary=json.dumps({
            "registered_mapping": mapping.mapping_id,
            "fields": field_summary,
            "counting_unit": proposal.counting_unit,
            "counting_unit_evidence": proposal.counting_unit_evidence,
        }, ensure_ascii=False, sort_keys=True),
    )


async def _run_one_workstream(
    *,
    settings: Settings,
    root: BimRunContext,
    registry: BimAgentRegistry,
    package: EvidenceWorkPackage,
    package_count: int,
    events: PipelineEvents,
    semaphore: asyncio.Semaphore,
) -> WorkstreamOutcome:
    async with semaphore:
        branch_bim = BimContext(settings)
        workstream_id = package.package_id
        branch: BimRunContext | None = None
        try:
            await asyncio.to_thread(branch_bim.connect)
            branch = BimRunContext(
                bim=branch_bim,
                scope=root.scope.model_copy(deep=True),
                graph_contract=root.graph_contract,
                question=root.question,
                run_id=root.run_id,
                workstream_id=workstream_id,
                parent_workstream_id=root.workstream_id,
                schema_mappings=dict(root.schema_mappings),
                schema_fingerprint=root.schema_fingerprint,
                learned_knowledge=dict(root.learned_knowledge),
                max_llm_calls=_branch_limit(
                    root.max_llm_calls, root.llm_calls, package_count, 6
                ),
                max_tool_calls=_branch_limit(
                    root.max_tool_calls, root.tool_calls, package_count, 10
                ),
                max_agent_starts=_branch_limit(
                    root.max_agent_starts, root.agent_starts, package_count, 3
                ),
                max_starts_per_agent=root.max_starts_per_agent,
            )
            worker_contract = package_contract(root.task_contract, package)
            define_bim_task(PipelineContext(branch), worker_contract)
            hooks = AgentRunHooks(events)
            events.stage(
                "workstream_start", run_id=root.run_id,
                workstream_id=workstream_id, outputs=len(package.required_outputs),
            )
            try:
                scout_result = await Runner.run(
                    registry.schema_scout,
                    _worker_input(root.question, root.task_contract, package),
                    context=branch,
                    max_turns=int(os.getenv("BIM_SCOUT_MAX_TURNS", "18")),
                    hooks=hooks,
                )
                handoff = EvidenceHandoff.model_validate(scout_result.final_output)
            except MaxTurnsExceeded:
                handoff = _checkpoint_handoff(root, branch, package)
                if handoff is None:
                    raise
                events.stage(
                    "scout_checkpoint_recovered", run_id=root.run_id,
                    workstream_id=workstream_id, mapping_id=handoff.mapping_id,
                )
            if handoff.package_id != package.package_id:
                raise ValueError("Schema scout returned a handoff for a different work package.")
            if handoff.status == "unsupported":
                events.stage(
                    "workstream_unsupported", run_id=root.run_id,
                    workstream_id=workstream_id,
                )
                return WorkstreamOutcome(
                    package, branch, "unsupported", limitations=list(handoff.limitations)
                )
            query_input = (
                _worker_input(root.question, root.task_contract, package)
                + "\nSCOUT_HANDOFF:\n"
                + handoff.model_dump_json()
            )
            query_result = await Runner.run(
                registry.query_executor,
                query_input,
                context=branch,
                max_turns=int(os.getenv("BIM_QUERY_WORKER_MAX_TURNS", "6")),
                hooks=hooks,
            )
            completion = EvidenceWorkstreamResult.model_validate(query_result.final_output)
            if completion.package_id != package.package_id:
                raise ValueError("Query worker returned a result for a different work package.")
            unknown_ids = [
                evidence_id for evidence_id in completion.evidence_ids
                if evidence_id not in branch.evidence
                or branch.evidence[evidence_id].kind != "query"
            ]
            if unknown_ids:
                raise ValueError("Query worker returned unknown evidence IDs.")
            events.stage(
                "workstream_end", run_id=root.run_id, workstream_id=workstream_id,
                status=completion.status, evidence=len(completion.evidence_ids),
            )
            return WorkstreamOutcome(
                package,
                branch,
                completion.status,
                evidence_ids=list(completion.evidence_ids),
                limitations=list(completion.limitations),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
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
                package, branch, category, limitations=[limitation], error=exc
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
) -> list[WorkstreamOutcome]:
    """Schedule the architect DAG with isolated state and deterministic commits."""
    packages = evidence_work_packages(root.task_contract)
    maximum = max(1, int(os.getenv("BIM_MAX_PARALLEL_WORKERS", "3")))
    semaphore = asyncio.Semaphore(min(maximum, len(packages)))
    pending = {item.package_id: item for item in packages}
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
            failed_dependencies = [
                dependency for dependency in item.depends_on
                if finished[dependency].status != "query_completed"
            ]
            if failed_dependencies:
                outcome = WorkstreamOutcome(
                    item, None, "dependency_blocked",
                    limitations=[
                        f"Workstream {item.package_id} was blocked by: "
                        + ", ".join(failed_dependencies) + "."
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
                package_count=len(packages),
                events=events,
                semaphore=semaphore,
            )
            for item in runnable
        ))
        for outcome in results:
            if outcome.branch is not None:
                outcome.evidence_ids = merge(root, outcome.branch)
                if outcome.status == "query_completed" and not outcome.evidence_ids:
                    outcome.status = "merge_rejected"
                    outcome.limitations.append(
                        f"Workstream {outcome.package.package_id} produced no committable answer evidence."
                    )
            finished[outcome.package.package_id] = outcome
            ordered_outcomes.append(outcome)
            pending.pop(outcome.package.package_id)
    return ordered_outcomes
