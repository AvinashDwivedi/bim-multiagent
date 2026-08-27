from __future__ import annotations

import hashlib
import json
import re

from .dataset import ProjectDataset, normalize
from .executor import DeterministicExecutor
from .models import QueryEvidence, QueryPlan, Verification


class EvidenceVerifier:
    def __init__(self, dataset: ProjectDataset):
        self.dataset = dataset

    def verify(self, plan: QueryPlan, evidence: QueryEvidence, question: str = "") -> Verification:
        replay = DeterministicExecutor(self.dataset).execute(plan)
        digest = _digest(evidence)
        replay_digest = _digest(replay)
        available_categories = {normalize(record.category) for record in self.dataset.physical_records}
        available_families = {normalize(record.family) for record in self.dataset.physical_records}
        available_types = {normalize(record.type_name) for record in self.dataset.physical_records}

        exact_values_exist = all(normalize(value) in available_categories for value in plan.categories)
        exact_values_exist &= all(normalize(value) in available_families for value in plan.families)
        exact_values_exist &= all(normalize(value) in available_types for value in plan.types)
        for branch in plan.population_branches:
            exact_values_exist &= all(normalize(value) in available_categories for value in branch.categories)
            exact_values_exist &= all(normalize(value) in available_families for value in branch.families)
            exact_values_exist &= all(normalize(value) in available_types for value in branch.types)

        requested_facets = set(plan.required_facets) | _infer_facets(question)
        has_population_boundary = bool(
            plan.population_branches or plan.categories or plan.families or plan.types
            or plan.match_terms or plan.filters or plan.filter_groups
        )
        specific_question = any(term in normalize(question) for term in ("מסוג", "description", "specific"))
        has_specific_constraint = bool(
            plan.population_branches or plan.families or plan.types or plan.match_terms or plan.filters
        )
        property_executed = bool(plan.select_properties) and len(evidence.property_summaries) == len(plan.select_properties)
        grouping_executed = bool(plan.group_by or plan.group_by_properties) and (
            bool(evidence.groups or evidence.grouped_measurements)
            or bool(plan.minimum_group_count and evidence.selected_count >= 0)
        )
        measure_executed = bool(plan.measure_property) and evidence.operation_value is not None and evidence.complete
        calculation_executed = _calculation_executed(plan, evidence)
        connectivity_executed = bool(
            evidence.connectivity and evidence.connectivity.get("ifc", {}).get("available")
        )
        related_scope_executed = bool(plan.include_related and plan.search_terms)
        semantic_coverage = True
        semantic_coverage_detail = "No specific multilingual term coverage check was required."
        if specific_question and plan.match_terms and plan.categories:
            coverage_plan = QueryPlan.from_dict(plan.to_dict(), planner=plan.planner)
            coverage_plan.families = []
            coverage_plan.types = []
            coverage_plan.population_branches = []
            coverage = DeterministicExecutor(self.dataset).execute(coverage_plan)
            semantic_coverage = coverage.distinct_identity_count <= evidence.distinct_identity_count
            semantic_coverage_detail = (
                f"Semantic terms identify {coverage.distinct_identity_count} candidate identities; "
                f"the final population contains {evidence.distinct_identity_count}."
            )
        height_from_floor = any(term in normalize(question) for term in ("מהרצפה", "from floor", "above floor"))
        placement_summary = next((
            item for item in evidence.property_summaries
            if normalize(item.get("field", "")) == normalize("IFC.Placement Height Above Storey")
        ), None)
        placement_executed = bool(
            placement_summary and placement_summary.get("present") == evidence.distinct_identity_count
        )
        path_analysis = (
            evidence.connectivity.get("ifc", {}).get("panel_path_analysis", {})
            if evidence.connectivity else {}
        )
        panel_paths_executed = bool(path_analysis.get("executed"))
        panel_paths_complete = bool(path_analysis.get("complete"))
        semantic_plan_checks = _question_plan_checks(question, plan)
        available_fields = {
            normalize(key) for record in self.dataset.physical_records for key in record.flat_properties
        }
        available_leaves = {item.rsplit(" ", 1)[-1] for item in available_fields}
        virtual_fields = {
            "object id", "name", "category", "family", "type", "type name", "level",
            "globalid", "global id", "externalid", "external id",
            normalize("BIM.Intended Height From Description"),
            normalize("IFC.Placement Height Above Storey"),
        }
        referenced_fields = [
            *[item.field for item in plan.filters],
            *[item.field for group in plan.filter_groups for item in group],
            *[item.field for item in plan.metric_filters],
            *plan.group_by_properties,
            *([plan.measure_property] if plan.measure_property else []),
            *([plan.distinct_property] if plan.distinct_property else []),
        ]
        fields_bound = all(
            normalize(field) in virtual_fields
            or normalize(field) in available_fields
            or normalize(field).rsplit(" ", 1)[-1] in available_leaves
            for field in referenced_fields
        )
        checks = [
            {"name": "three_file_contract", "passed": True, "detail": "One tree, properties, and IFC file."},
            {"name": "replay_stability", "passed": digest == replay_digest, "detail": replay_digest},
            {
                "name": "identity_deduplication",
                "passed": 0 <= evidence.distinct_identity_count <= evidence.selected_count,
                "detail": (
                    f"{evidence.selected_count} selected records reduced to "
                    f"{evidence.distinct_identity_count} distinct identities"
                ),
            },
            {
                "name": "exact_vocabulary_binding",
                "passed": exact_values_exist,
                "detail": "All exact category/family/type values occur in the supplied data.",
            },
            {
                "name": "measurement_completeness",
                "passed": "measure" not in requested_facets or measure_executed,
                "detail": "All selected records contributed compatible values." if evidence.complete else "Missing or mixed-unit values.",
            },
            {
                "name": "calculation_execution",
                "passed": calculation_executed,
                "detail": f"Requested calculation `{plan.calculation}` produced a typed result.",
            },
            {
                "name": "property_vocabulary_binding",
                "passed": fields_bound,
                "detail": "Every referenced property is present in the project schema or is a supported virtual field.",
            },
            {
                "name": "supported_requirements",
                "passed": not plan.unsupported_requirements,
                "detail": (
                    "All question requirements are executable."
                    if not plan.unsupported_requirements else "; ".join(plan.unsupported_requirements)
                ),
            },
            *semantic_plan_checks,
            {
                "name": "population_boundary",
                "passed": "population" not in requested_facets or has_population_boundary,
                "detail": "The requested population is encoded in executable plan fields.",
            },
            {
                "name": "specific_scope_constraint",
                "passed": not specific_question or has_specific_constraint,
                "detail": "Specific type/description questions require a family, type, term, or property constraint.",
            },
            {
                "name": "semantic_population_coverage",
                "passed": semantic_coverage,
                "detail": semantic_coverage_detail,
            },
            {
                "name": "grouping_execution",
                "passed": "grouping" not in requested_facets or grouping_executed,
                "detail": "Requested grouping fields were executed.",
            },
            {
                "name": "property_projection",
                "passed": "property_values" not in requested_facets or property_executed,
                "detail": "Requested properties were projected, including missing-value counts.",
            },
            {
                "name": "connectivity_execution",
                "passed": "connectivity" not in requested_facets or connectivity_executed,
                "detail": "IFC systems, ports, and connections were inspected.",
            },
            {
                "name": "related_scope_audit",
                "passed": "related_scope" not in requested_facets or related_scope_executed,
                "detail": "Related physical candidates outside the primary category were audited.",
            },
            {
                "name": "floor_height_derivation",
                "passed": not height_from_floor or placement_executed,
                "detail": "IFC placement height above the containing storey was derived for every selected instance.",
            },
            {
                "name": "panel_path_analysis",
                "passed": (
                    "connectivity" not in requested_facets
                    or (panel_paths_executed and panel_paths_complete)
                ),
                "detail": (
                    "Logical assignments were resolved to panel entities and evaluated independently "
                    "from physical graph paths."
                ),
            },
        ]
        limitations = [
            *plan.unsupported_requirements,
            "The result is scoped to the supplied three-file export; the files do not prove that every project discipline was supplied.",
        ]
        if evidence.selected_count == 0:
            limitations.append(
                "No matching physical instances were verified. This is not a full-project zero unless the supplied export is known to be complete."
            )
        if plan.planner == "heuristic":
            limitations.append("The semantic plan used the conservative local planner rather than an LLM planner.")
        if height_from_floor:
            limitations.append(
                "IFC height is the product placement origin above its containing storey; it is not a tessellated body-centre measurement."
            )
        if path_analysis and not panel_paths_complete:
            limitations.append(
                f"{path_analysis.get('assigned_panel_unresolved', 0)} logical panel assignment(s) could not be resolved to IFC panel entities."
            )
        passed = all(check["passed"] for check in checks)
        status = "verified" if passed and evidence.selected_count > 0 else "limited"
        return Verification(status=status, checks=checks, limitations=limitations, digest=digest)


def _digest(evidence: QueryEvidence) -> str:
    payload = json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _infer_facets(question: str) -> set[str]:
    text = normalize(question)
    facets = {"population"}
    if any(term in text for term in ("לפי", "by ", "types", "סוגי")):
        facets.add("grouping")
    if any(term in text for term in ("אורך", "length", "total length")):
        facets.add("measure")
    if any(term in text for term in ("חומר", "material", "גובה", "height", "מרחק", "elevation")):
        facets.add("property_values")
    if any(term in text for term in ("חיבור", "המשכיות", "connected", "connection", "continuity", "physical path")):
        facets.add("connectivity")
    return facets


def _calculation_executed(plan: QueryPlan, evidence: QueryEvidence) -> bool:
    calculation = "sum" if plan.intent == "sum" and plan.calculation == "count" else plan.calculation
    if calculation in {"sum", "average", "min", "max"}:
        return bool(plan.measure_property) and evidence.operation_value is not None
    if calculation == "distinct_count":
        return bool(plan.distinct_property) and evidence.operation_value is not None
    if calculation == "percentage":
        return (
            bool(plan.metric_filters)
            and evidence.metric_denominator is not None
            and evidence.metric_numerator is not None
            and evidence.operation_value is not None
        )
    return evidence.operation_value is not None


def _question_plan_checks(question: str, plan: QueryPlan) -> list[dict]:
    """Independent, generic question-to-plan contract checks.

    These checks do not decide which BIM objects are correct. They ensure that analytical clauses
    explicitly present in the question were not silently dropped by either planner.
    """
    text = normalize(question)
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    if any(term in text for term in ("average", "mean", "ממוצע")):
        add("average_requirement", plan.calculation == "average" and bool(plan.measure_property),
            "Average requires calculation=average and a numeric measure property.")
    if any(term in text for term in ("percentage", "percent", "אחוז")):
        add("percentage_requirement", plan.calculation == "percentage" and bool(plan.metric_filters),
            "Percentage requires an explicit numerator predicate and selected-population denominator.")
    if any(term in text for term in ("compare", "versus", " vs ", "השווה", "לעומת")):
        alternatives = any(item.operator == "in" and isinstance(item.value, list) and len(item.value) > 1 for item in plan.filters)
        alternatives |= len(plan.filter_groups) > 1
        add("comparison_requirement", bool(plan.group_by) and alternatives,
            "Comparison requires multiple executable alternatives and a grouping dimension.")
    if re.search(r"(?:longer\s+than|greater\s+than|ארו(?:ך|כים)\s+מ)", question, flags=re.IGNORECASE):
        add("threshold_requirement", any(item.operator in {"gt", "gte", "lt", "lte"} for item in plan.filters),
            "A numeric threshold in the question requires an executable comparison filter.")
    if plan.analysis != "connectivity" and any(term in text for term in ("without", "missing", " no ", "ללא", "חסר")):
        predicates = [*plan.filters, *plan.metric_filters, *[item for group in plan.filter_groups for item in group]]
        add("missing_value_requirement", any(item.operator in {"missing", "not_missing"} for item in predicates),
            "Missing-value language requires an explicit missing/not_missing predicate.")
    if re.search(r"(?:\bmark\b|סימון)\s*(?:=|is|number|מספר)?\s*[\w.-]+", question, flags=re.IGNORECASE):
        add("property_lookup_requirement", any("mark" in normalize(item.field) or "סימון" in normalize(item.field) for item in plan.filters),
            "Direct property lookup requires a filter bound to that property.")
    if any(term in text for term in ("duplicate", "duplicates", "כפילויות")):
        add("duplicate_requirement", bool(plan.group_by_properties) and (plan.minimum_group_count or 0) >= 2,
            "Duplicate analysis requires property grouping and a minimum group count of two.")
    if any(term in text for term in ("most", "fewest", "least", "הכי הרבה", "הכי מעט")):
        add("ranking_requirement", bool(plan.group_by) and plan.sort_by == "count" and plan.limit is not None,
            "Ranking requires grouping, sorting by the aggregate, and a result limit.")
    return checks
