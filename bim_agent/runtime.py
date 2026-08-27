from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .dataset import ProjectDataset, normalize
from .executor import DeterministicExecutor
from .models import AnswerReport, QueryEvidence, QueryPlan
from .planner import SemanticPlanner
from .tracing import TraceLog
from .verifier import EvidenceVerifier


class BimAgent:
    """Agentic orchestration with deterministic execution and replay verification."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        use_llm: bool | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or Settings.from_env(data_dir=data_dir, use_llm=use_llm)
        self.dataset = ProjectDataset(self.settings.data_dir)
        self.profile = self.dataset.profile(self.settings.max_profile_values)
        self.planner = SemanticPlanner(
            model=self.settings.model,
            reasoning_effort=self.settings.reasoning_effort,
            use_llm=self.settings.use_llm,
        )

    def ask(self, question: str) -> AnswerReport:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        trace = TraceLog(self.settings.trace_dir, question)
        trace.event(
            "file_inspector",
            data_dir=str(self.settings.data_dir),
            profile=self.profile,
            sqlite_inventory=self.dataset.sqlite_inventory(),
        )

        question_profile = self.dataset.profile(self.settings.max_profile_values, focus_terms=[question])
        plan = self.planner.plan(question, question_profile)
        trace.event("semantic_planner", plan=plan.to_dict())

        executor = DeterministicExecutor(self.dataset)
        evidence = executor.execute(plan)
        trace.event("investigator", evidence=evidence.to_dict())

        sqlite_matches = self.dataset.search_sqlite_metadata(plan.search_terms)
        trace.event(
            "counterexample_auditor",
            related_groups=evidence.related_groups,
            sqlite_metadata_matches=sqlite_matches[:50],
        )

        verifier = EvidenceVerifier(self.dataset)
        verification = verifier.verify(plan, evidence, question)
        trace.event("verifier", verification=verification.to_dict())

        failed_checks = [check for check in verification.checks if not check["passed"]]
        if failed_checks and self.planner.use_llm:
            refined = self.planner.refine(
                question,
                question_profile,
                plan,
                {
                    "selected_count": evidence.distinct_identity_count,
                    "groups": evidence.groups[:20],
                    "grouped_measurements": evidence.grouped_measurements[:30],
                    "property_summaries": evidence.property_summaries,
                    "connectivity": evidence.connectivity,
                },
                failed_checks,
            )
            if refined.to_dict() != plan.to_dict():
                plan = refined
                trace.event("semantic_refinement", plan=plan.to_dict(), failed_checks=failed_checks)
                evidence = executor.execute(plan)
                trace.event("refined_investigator", evidence=evidence.to_dict())
                verification = verifier.verify(plan, evidence, question)
                trace.event("refined_verifier", verification=verification.to_dict())

        answer = compose_answer(question, plan, evidence, verification.limitations)
        trace.event("answer_composer", answer=answer, status=verification.status)
        return AnswerReport(
            answer=answer,
            status=verification.status,
            plan=plan,
            evidence=evidence,
            verification=verification,
            sources=self.dataset.source_manifest(),
            trace_path=str(trace.path),
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.settings.data_dir),
            "profile": self.profile,
            "sqlite": self.dataset.sqlite_inventory(),
            "sources": self.dataset.source_manifest(),
            "llm_planner_enabled": self.settings.use_llm,
            "model": self.settings.model if self.settings.use_llm else None,
        }


def compose_answer(
    question: str, plan: QueryPlan, evidence: QueryEvidence, limitations: list[str]
) -> str:
    hebrew = bool(re.search(r"[\u0590-\u05ff]", question))
    if plan.unsupported_requirements:
        heading = (
            "לא ניתן לאמת תשובה מלאה לכל דרישות השאלה."
            if hebrew else "A complete answer cannot be verified for every requirement in the question."
        )
        label = "נדרש:" if hebrew else "Required:"
        return "\n".join([heading, *[f"- {label} {item}" for item in plan.unsupported_requirements]])
    if evidence.connectivity:
        logical = evidence.connectivity.get("logical", {})
        graph = evidence.connectivity.get("ifc", {})
        paths = graph.get("panel_path_analysis", {})
        if hebrew:
            lead = (
                f"נבדקו בנפרד שיוך לוגי והמשכיות פיזית עבור {logical.get('elements', 0)} רכיבים: "
                f"{logical.get('without_panel', 0)} ללא לוח מזין ו-"
                f"{logical.get('without_circuit', 0)} ללא מספר מעגל; "
                f"{paths.get('physical_no_path_to_any_panel', 0)} ללא נתיב IFC ללוח."
            )
        else:
            lead = (
                f"Checked logical assignment and physical continuity separately for {logical.get('elements', 0)} components: "
                f"{logical.get('without_panel', 0)} lack a feeding panel and "
                f"{logical.get('without_circuit', 0)} lack a circuit number; "
                f"{paths.get('physical_no_path_to_any_panel', 0)} have no IFC path to a panel."
            )
    elif plan.calculation == "percentage":
        value = evidence.operation_value if evidence.operation_value is not None else 0
        numerator = evidence.metric_numerator or 0
        denominator = evidence.metric_denominator or 0
        lead = (
            f"התוצאה המאומתת היא {value:g}% ({numerator:g} מתוך {denominator:g})."
            if hebrew else f"The verified result is {value:g}% ({numerator:g} of {denominator:g})."
        )
    elif plan.calculation in {"sum", "average", "min", "max", "distinct_count"}:
        value = evidence.operation_value if evidence.operation_value is not None else 0
        unit = f" {evidence.unit}" if evidence.unit else ""
        labels = {
            "sum": ("הסכום", "sum"), "average": ("הממוצע", "average"),
            "min": ("המינימום", "minimum"), "max": ("המקסימום", "maximum"),
            "distinct_count": ("מספר הערכים הייחודיים", "distinct count"),
        }
        label = labels[plan.calculation][0 if hebrew else 1]
        lead = (
            f"{label} המאומת הוא {value:g}{unit}."
            if hebrew else f"The verified {label} is {value:g}{unit}."
        )
    elif evidence.selected_count == 0:
        lead = (
            "לא נמצאו מופעים פיזיים תואמים בקובצי המודל שסופקו."
            if hebrew
            else "No matching physical instances were found in the supplied model files."
        )
    elif plan.minimum_group_count and not evidence.groups:
        lead = (
            f"לא נמצאו ערכים כפולים בין {evidence.distinct_identity_count} המופעים שנבדקו."
            if hebrew else f"No duplicate values were found among {evidence.distinct_identity_count} inspected instances."
        )
    elif plan.intent == "list":
        lead = (
            f"נמצאו {len(evidence.groups)} קבוצות ו-{evidence.distinct_identity_count} מופעים פיזיים."
            if hebrew
            else f"Found {len(evidence.groups)} groups covering {evidence.distinct_identity_count} physical instances."
        )
    else:
        lead = (
            f"נמצאו {evidence.distinct_identity_count} מופעים פיזיים תואמים."
            if hebrew
            else f"Found {evidence.distinct_identity_count} matching physical instances."
        )

    lines = [lead]
    intended_height = next((
        summary for summary in evidence.property_summaries
        if normalize(summary.get("field", "")) == normalize("BIM.Intended Height From Description")
    ), None)
    if intended_height and intended_height.get("values"):
        values = ", ".join(
            f"{item['value']} ({item['count']})" for item in intended_height["values"]
        )
        lines.append(
            f"הגבהים המתוכננים שנגזרו מתיאורי הטיפוסים: {values}."
            if hebrew else f"Planned heights derived from type descriptions: {values}."
        )
    if evidence.category_counts and len(evidence.category_counts) > 1:
        lines.append("לפי קטגוריה:" if hebrew else "By category:")
        for item in evidence.category_counts:
            lines.append(f"- {item['category']}: {item['count']}")
    if evidence.grouped_measurements:
        lines.append("פילוח מדידה:" if hebrew else "Measurement breakdown:")
        for group in evidence.grouped_measurements[:60]:
            dimension_fields = list(dict.fromkeys([*plan.group_by, *plan.group_by_properties]))
            labels = [str(group.get(field)) for field in dimension_fields if group.get(field)]
            unit = f" {group.get('unit')}" if group.get("unit") else ""
            aggregate = group.get("value", group.get("sum", 0))
            aggregate = aggregate if isinstance(aggregate, (int, float)) else 0
            lines.append(
                f"- {' :: '.join(labels)}: {group.get('count', 0)} items, "
                f"{aggregate:g}{unit} ({plan.calculation})"
            )
    elif evidence.groups:
        lines.append("פירוט:" if hebrew else "Breakdown:")
        for group in evidence.groups[:30]:
            labels = [
                str(group.get(field, ""))
                for field in [*plan.group_by, *plan.group_by_properties]
                if group.get(field)
            ]
            label = " :: ".join(labels) or plan.target_label
            lines.append(f"- {label}: {group['count']}")
    for summary in evidence.property_summaries:
        field = summary["field"]
        if summary["present"] == 0:
            message = (
                f"אין נתון עבור {field} באף אחד מ-{summary['selected']} המופעים."
                if hebrew else
                f"No value for {field} exists on any of the {summary['selected']} selected instances."
            )
            if "material" in field.casefold() and summary.get("ifc_material_associations", 0) == 0:
                message += " גם ב-IFC לא נמצא שיוך חומר למופעים אלה." if hebrew else " The IFC also has no material association for them."
            lines.append(message)
        else:
            lines.append((f"ערכי {field}:" if hebrew else f"Values for {field}:") )
            for item in summary["values"][:25]:
                lines.append(f"- {item['value']}: {item['count']}")
            if summary["missing"]:
                lines.append(
                    f"- {'חסר' if hebrew else 'Missing'}: {summary['missing']}"
                )
    if evidence.connectivity:
        logical = evidence.connectivity.get("logical", {})
        graph = evidence.connectivity.get("ifc", {})
        if logical.get("panels"):
            lines.append("שיוכי לוחות:" if hebrew else "Panel assignments:")
            for item in logical["panels"]:
                lines.append(f"- {item['value']}: {item['count']}")
        if graph.get("available"):
            lines.append("בדיקת המשכיות פיזית ב-IFC:" if hebrew else "IFC physical continuity:")
            metrics = (
                ("ports", "פורטים" if hebrew else "ports"),
                ("port_connections", "חיבורי פורטים" if hebrew else "port connections"),
                ("elements_with_ports", "רכיבים עם פורטים" if hebrew else "elements with ports"),
                ("elements_with_ports_without_connections", "רכיבים מנותקים" if hebrew else "isolated elements"),
                ("network_components", "רכיבי רשת נפרדים" if hebrew else "network components"),
                ("largest_component", "גודל הרכיב המחובר הגדול ביותר" if hebrew else "largest connected component"),
                ("selected_without_ifc_system", "רכיבים נבחרים ללא IfcSystem" if hebrew else "selected elements without IfcSystem"),
            )
            for key, label in metrics:
                lines.append(f"- {label}: {graph.get(key, 0)}")
            paths = graph.get("panel_path_analysis", {})
            if paths.get("executed"):
                lines.append("נתיבים ללוחות:" if hebrew else "Paths to panels:")
                path_metrics = (
                    ("physical_path_to_any_panel", "רכיבים עם נתיב פיזי ללוח" if hebrew else "components with a physical path to a panel"),
                    ("physical_no_path_to_any_panel", "רכיבים ללא נתיב פיזי ללוח" if hebrew else "components without a physical path to a panel"),
                    ("path_to_assigned_panel", "רכיבים עם נתיב ללוח המשויך" if hebrew else "components with a path to their assigned panel"),
                    ("no_path_to_assigned_panel", "רכיבים ללא נתיב ללוח המשויך" if hebrew else "components without a path to their assigned panel"),
                    ("either_logical_or_physical_failure", "כשל לוגי או פיזי" if hebrew else "logical or physical failures"),
                    ("both_logical_and_physical_failure", "כשל לוגי וגם פיזי" if hebrew else "both logical and physical failures"),
                )
                for key, label in path_metrics:
                    lines.append(f"- {label}: {paths.get(key, 0)}")
            if graph.get("isolated_panels"):
                lines.append("לוחות עם פורטים ללא חיבור:" if hebrew else "Panels with ports but no connection:")
                for item in graph["isolated_panels"]:
                    lines.append(f"- {item['family']} :: {item['type']}: {item['count']}")
    if evidence.related_groups:
        lines.append("מועמדים קשורים מחוץ לגבול הראשי:" if hebrew else "Related candidates outside the primary boundary:")
        for group in evidence.related_groups[:12]:
            label = " :: ".join(
                value for value in (group.get("category"), group.get("family"), group.get("type")) if value
            )
            lines.append(f"- {label}: {group['count']}")
    if plan.interpretation:
        lines.append(("פירוש: " if hebrew else "Interpretation: ") + plan.interpretation)
    if limitations:
        lines.append(("מגבלה: " if hebrew else "Limitation: ") + limitations[0])
    return "\n".join(lines)


def report_json(report: AnswerReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
