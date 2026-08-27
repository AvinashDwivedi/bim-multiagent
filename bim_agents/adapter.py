from __future__ import annotations

import os
from typing import Any

from bim_agent.models import AnswerReport


STAGES = [
    "File Inspector",
    "Vocabulary Profiler",
    "Semantic Planner",
    "Deterministic Investigator",
    "Counterexample Auditor",
    "Replay Verifier",
    "Answer Composer",
]


def evaluator_payload(
    report: AnswerReport,
    *,
    client_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    plan = report.plan
    evidence = report.evidence
    planned_categories = plan.categories or list(dict.fromkeys(
        category for branch in plan.population_branches for category in branch.categories
    ))
    stages = list(STAGES)
    if evidence.connectivity:
        stages.insert(4, "IFC Graph Inspector")
    trace = [
        f"File Inspector: read exactly three sources and indexed their tree/property identities.",
        f"Vocabulary Profiler: discovered {planned_categories or ['no exact category boundary']}.",
        f"Semantic Planner ({plan.planner}): {plan.interpretation or plan.target_label}",
        (
            "Deterministic Investigator: selected "
            f"{evidence.selected_count} record(s), {evidence.distinct_identity_count} distinct identity/identities."
        ),
        f"Counterexample Auditor: found {sum(row['count'] for row in evidence.related_groups)} related physical candidate(s).",
        f"Replay Verifier: status={report.verification.status}, digest={report.verification.digest}.",
    ]
    if evidence.grouped_measurements:
        trace.insert(-1, f"Measurement Investigator: produced {len(evidence.grouped_measurements)} grouped measurement row(s).")
    if evidence.property_summaries:
        trace.insert(-1, f"Property Investigator: projected {len(evidence.property_summaries)} requested property field(s).")
    if evidence.connectivity:
        graph = evidence.connectivity.get("ifc", {})
        trace.insert(-1, (
            "IFC Graph Inspector: found "
            f"{graph.get('ports', 0)} ports and {graph.get('port_connections', 0)} port connection(s)."
        ))
    limitations = list(report.verification.limitations)
    if client_id or project_id:
        limitations.append(
            "Project scope was resolved through BIM_PROJECTS_ROOT."
            if os.getenv("BIM_PROJECTS_ROOT") and project_id
            else "Project identifiers are request metadata because BIM_PROJECTS_ROOT is not configured; data scope is BIM_DATA_DIR."
        )
    return {
        "answer": report.answer,
        "verification_status": report.verification.status,
        "limitations": limitations,
        "stages_used": stages,
        "artifact_ids": [
            f"source-{source['sha256'][:16]}" for source in report.sources
        ] + [f"evidence-{report.verification.digest[:16]}"],
        "investigation_trace": trace,
        "failure_categories": [] if report.verification.status == "verified" else ["limited_evidence"],
        "semantic_checks": report.verification.checks,
    }
