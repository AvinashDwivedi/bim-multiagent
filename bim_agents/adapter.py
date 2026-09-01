from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from bim_agent.models import AnswerReport


def evaluator_payload(
    report: AnswerReport,
    *,
    client_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    loop = report.agent_loop
    tools_used = []
    trace = []
    for iteration in loop.get("iterations", []):
        if iteration.get("action") == "finish":
            trace.append(
                f"Model response {iteration.get('response_id', '')}: finished with a user answer."
            )
            continue
        for call in iteration.get("calls", []):
            name = str(call.get("tool", ""))
            if name and name not in tools_used:
                tools_used.append(name)
            trace.append(
                f"Model iteration {iteration.get('iteration')}: called {name}; "
                f"outcome={call.get('outcome')}; output_characters={call.get('output_characters')}; "
                f"truncated={call.get('truncated')}."
            )
    limitations = list(report.limitations)
    if client_id or project_id:
        limitations.append(
            "Project scope was resolved through the selected client/project folder."
            if client_id and project_id and (
                os.getenv("BIM_PROJECTS_ROOT")
                or (Path(__file__).resolve().parents[1] / "bim-data").is_dir()
            )
            else "Project scope was matched to BIM_SINGLE_PROJECT_ID and resolved through BIM_DATA_DIR."
        )
    response_artifacts = [
        "response-" + hashlib.sha256(item.encode("utf-8")).hexdigest()[:16]
        for item in loop.get("response_ids", []) if item
    ]
    return {
        "answer": report.answer,
        "cost": report.cost,
        "verification_status": report.status,
        "limitations": limitations,
        "stages_used": ["Model-Directed BIM Agent", *[f"Tool: {name}" for name in tools_used]],
        "artifact_ids": [
            f"source-{source['sha256'][:16]}" for source in report.sources
        ] + response_artifacts,
        "investigation_trace": trace,
        "failure_categories": [] if report.status == "completed" else ["model_incomplete"],
        "semantic_checks": [],
    }
