from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from .dataset import expand_terms, normalize
from .models import PropertyFilter, QueryBranch, QueryPlan


_FILTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "field": {"type": "string"},
        "operator": {"type": "string", "enum": [
            "equals", "not_equals", "contains", "not_contains", "in", "not_in",
            "missing", "not_missing", "gt", "gte", "lt", "lte",
        ]},
        "value": {
            "anyOf": [
                {"type": "string"}, {"type": "number"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
    },
    "required": ["field", "operator", "value"],
}


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "enum": ["count", "list", "group_count", "sum", "describe"]},
        "scope": {"type": "string", "enum": ["physical_instances", "all_records"]},
        "target_label": {"type": "string"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
        "families": {"type": "array", "items": {"type": "string"}},
        "types": {"type": "array", "items": {"type": "string"}},
        "exclude_terms": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": _FILTER_SCHEMA},
        "filter_groups": {"type": "array", "items": {"type": "array", "items": _FILTER_SCHEMA}},
        "population_branches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "families": {"type": "array", "items": {"type": "string"}},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "match_terms": {"type": "array", "items": {"type": "string"}},
                    "exclude_terms": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "array", "items": _FILTER_SCHEMA},
                },
                "required": ["label", "categories", "families", "types", "match_terms", "exclude_terms", "filters"],
            },
        },
        "match_terms": {"type": "array", "items": {"type": "string"}},
        "group_by": {
            "type": "array",
            "items": {"type": "string", "enum": ["category", "family", "type_name", "level", "name"]},
        },
        "group_by_properties": {"type": "array", "items": {"type": "string"}},
        "measure_property": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "calculation": {"type": "string", "enum": [
            "count", "sum", "average", "min", "max", "distinct_count", "percentage",
        ]},
        "distinct_property": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "metric_filters": {"type": "array", "items": _FILTER_SCHEMA},
        "sort_by": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "sort_direction": {"type": "string", "enum": ["asc", "desc"]},
        "limit": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
        "minimum_group_count": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
        "select_properties": {"type": "array", "items": {"type": "string"}},
        "output_unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "analysis": {"type": "string", "enum": ["standard", "connectivity"]},
        "required_facets": {
            "type": "array",
            "items": {"type": "string", "enum": ["population", "grouping", "measure", "property_values", "connectivity", "related_scope"]},
        },
        "include_related": {"type": "boolean"},
        "interpretation": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "unsupported_requirements": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "intent", "scope", "target_label", "search_terms", "categories", "families", "types",
        "exclude_terms", "filters", "filter_groups", "population_branches", "match_terms", "group_by",
        "group_by_properties", "measure_property", "calculation", "distinct_property",
        "metric_filters", "sort_by", "sort_direction", "limit", "minimum_group_count",
        "select_properties", "output_unit",
        "analysis", "required_facets", "include_related", "interpretation", "assumptions",
        "unsupported_requirements",
    ],
}


class SemanticPlanner:
    def __init__(self, *, model: str, reasoning_effort: str, use_llm: bool):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))

    def plan(self, question: str, profile: dict[str, Any]) -> QueryPlan:
        if self.use_llm:
            try:
                return apply_domain_guardrails(question, profile, self._llm_plan(question, profile))
            except Exception as exc:
                plan = heuristic_plan(question, profile)
                plan.assumptions.append(f"LLM planner unavailable; used local planner ({type(exc).__name__}).")
                return apply_domain_guardrails(question, profile, plan)
        return apply_domain_guardrails(question, profile, heuristic_plan(question, profile))

    def refine(
        self, question: str, profile: dict[str, Any], plan: QueryPlan,
        evidence: dict[str, Any], failed_checks: list[dict[str, Any]],
    ) -> QueryPlan:
        if not self.use_llm:
            return plan
        try:
            return apply_domain_guardrails(question, profile, self._llm_plan(
                question,
                profile,
                feedback={
                    "previous_plan": plan.to_dict(),
                    "evidence_summary": evidence,
                    "failed_verification_checks": failed_checks,
                },
            ))
        except Exception:
            return plan

    def _llm_plan(
        self, question: str, profile: dict[str, Any], feedback: dict[str, Any] | None = None,
    ) -> QueryPlan:
        from openai import OpenAI

        instructions = """You are the semantic planner for a read-only BIM analysis agent.
Return only a structured query plan. Use exact category/family/type/property values present in the supplied
profile. Interpret the user's real-world concept, not mere substrings. Select physical instances for ordinary
counts; never count category, family, type, template, style, legend, system, or analytical definition records.
A broad Revit category may contain unrelated families. Use match_terms or property filters inside that category,
or use population_branches for a union of differently constrained category populations. Prefer an inclusive
category plus explicit exclusions when an incomplete family allow-list would omit valid instances.
search_terms are only for secondary/counterexample auditing; they never constrain the primary population.
Use select_properties whenever the answer asks for a material, description, height, panel, circuit, or another
property value. Use group_by_properties for arbitrary property dimensions such as Width or Material.
Use calculation for count, sum, average, min, max, distinct count, and percentage. For numerical calculations,
bind measure_property. For percentage, metric_filters define the numerator and the selected population is the
denominator. filters are global AND constraints; filter_groups are OR alternatives whose members are ANDed.
Use missing/not_missing for absent values. Use grouping plus sort_by=count and limit for top/bottom questions.
For connection, continuity, panel, circuit, port, or system
questions, set analysis=connectivity and require the connectivity facet. Declare every requested answer facet
in required_facets. Never leave an executable requirement only in interpretation: encode it in typed fields.
Put any clause that cannot be represented faithfully in unsupported_requirements. Do not invent project values.
If evidence is ambiguous, record the assumption explicitly."""
        payload = {
            "question": question,
            "project_vocabulary_profile": profile,
        }
        if feedback is not None:
            payload["refinement_feedback"] = feedback
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "bim_query_plan",
                    "strict": True,
                    "schema": PLAN_SCHEMA,
                }
            },
            store=False,
            max_output_tokens=4000,
        )
        value = json.loads(response.output_text)
        return QueryPlan.from_dict(value, planner=f"openai:{self.model}")


STOPWORDS = {
    "how", "many", "much", "what", "which", "are", "is", "in", "on", "of", "the", "project",
    "show", "list", "type", "types", "kind", "kinds", "give", "me", "total", "all", "there",
    "and", "or", "to", "has", "have", "no", "a", "an", "כמה", "מה", "מהם", "אילו", "יש",
    "בפרויקט", "בפרוייקט", "של", "את", "כל", "סהכ", "סה", "כ",
    "compare", "versus", "number", "average", "mean", "percentage", "percent",
    "most", "least", "fewest", "longer", "than", "missing", "without", "data", "categories",
}


def heuristic_plan(question: str, profile: dict[str, Any]) -> QueryPlan:
    normalized_question = normalize(question)
    raw_tokens = re.findall(r"[\w\u0590-\u05ff]+", normalized_question, flags=re.UNICODE)
    terms = [token for token in raw_tokens if token not in STOPWORDS and len(token) > 1]
    if not terms:
        terms = raw_tokens
    expanded = expand_terms(terms)

    intent = "count"
    group_by = ["family", "type_name"]
    if any(token in normalized_question for token in ("which", "what type", "types", "list", "אילו", "סוגי")):
        intent = "list"
    if any(token in normalized_question for token in ("length", "area", "volume", "אורך", "שטח", "נפח")):
        intent = "sum"

    candidates: list[tuple[int, str, str, str, int]] = []
    category_scores: Counter[str] = Counter()
    exact_category_mentions: set[str] = set()
    for branch in profile.get("categories", []):
        category = str(branch.get("category", ""))
        category_text = normalize(category)
        category_match = sum(term in category_text for term in expanded if len(term) >= 2)
        if category_text and category_text in normalized_question:
            category_scores[category] += 1000
            exact_category_mentions.add(category)
        if category_match:
            category_scores[category] += 20 * category_match
        for item in branch.get("family_types", []):
            family = str(item.get("family", ""))
            type_name = str(item.get("type", ""))
            text = normalize(f"{family} {type_name}")
            score = sum(term in text for term in expanded if len(term) >= 2)
            if score:
                count = int(item.get("count", 0))
                candidates.append((score, category, family, type_name, count))
                category_scores[category] += score

    selected_categories: list[str] = []
    selected_families: list[str] = []
    selected_types: list[str] = []
    if category_scores:
        best_score = max(category_scores.values())
        # Keep ties, but otherwise use the strongest semantic branch as the primary population.
        selected_categories = [name for name, score in category_scores.items() if score == best_score]
        for _, category, family, type_name, _ in candidates:
            if category in selected_categories and category not in exact_category_mentions:
                if family:
                    selected_families.append(family)
                if type_name:
                    selected_types.append(type_name)

    filters: list[PropertyFilter] = []
    mentioned_levels = []
    for item in profile.get("levels", []):
        value = str(item.get("value", ""))
        if value and normalize(value) in normalized_question:
            mentioned_levels.append(value)
    if len(mentioned_levels) == 1:
        filters.append(PropertyFilter("level", "equals", mentioned_levels[0]))
    elif mentioned_levels:
        filters.append(PropertyFilter("level", "in", mentioned_levels))

    level_terms = {normalize(value) for value in mentioned_levels}
    semantic_terms = [term for term in terms if normalize(term) not in level_terms]

    measure_property = _measure_property(normalized_question, profile.get("property_keys", []))
    if intent == "sum" and not measure_property:
        intent = "describe"

    switch_like = any(term in expanded for term in ("switch", "switches", "מפס"))
    asks_for_board = any(term in normalized_question for term in ("switchboard", "panel", "לוח"))
    exclusions = []
    if switch_like and not asks_for_board:
        exclusions.extend(["switchboard", "panel schedule", "legend component", "switch system"])

    target = " ".join(terms) or "matching elements"
    use_semantic_boundary = bool(
        selected_categories and not any(category in exact_category_mentions for category in selected_categories)
    )
    if use_semantic_boundary:
        selected_families = []
        selected_types = []
    calculation = "sum" if intent == "sum" else "count"
    if any(term in normalized_question for term in ("average", "mean", "ממוצע")):
        calculation = "average"
        intent = "sum"
    elif any(term in normalized_question for term in ("percentage", "percent", "אחוז")):
        calculation = "percentage"
    elif any(term in normalized_question for term in ("distinct", "ייחוד")):
        calculation = "distinct_count"
    return QueryPlan(
        intent=intent,
        target_label=target,
        search_terms=terms,
        categories=list(dict.fromkeys(selected_categories)),
        families=list(dict.fromkeys(selected_families)),
        types=list(dict.fromkeys(selected_types)),
        exclude_terms=exclusions,
        filters=filters,
        match_terms=semantic_terms if use_semantic_boundary else [],
        group_by=group_by,
        measure_property=measure_property,
        calculation=calculation,
        include_related=True,
        interpretation="Locally inferred from category, family, type, level, and property vocabulary.",
        assumptions=["The strongest matching model branch is treated as the primary population."],
        planner="heuristic",
    )


def _apply_generic_requirements(
    question: str, profile: dict[str, Any], plan: QueryPlan,
) -> QueryPlan:
    """Map common analytical language to generic executable fields.

    This layer is deliberately project-agnostic: it resolves against the supplied schema instead
    of naming a Revit category, family, type, or project property in code.
    """
    text = normalize(question)

    if any(term in text for term in ("average", "mean", "ממוצע")):
        plan.calculation = "average"
        plan.intent = "sum"
    elif any(term in text for term in ("percentage", "percent", "אחוז", "כמה אחוז")):
        plan.calculation = "percentage"
        plan.intent = "count"
    elif any(term in text for term in ("distinct count", "count distinct", "מספר ערכים ייחוד")):
        plan.calculation = "distinct_count"
    elif any(term in text for term in ("longest", "maximum length", "אורך מקסימלי", "הארוך ביותר")):
        plan.calculation = "max"
        plan.intent = "sum"
    elif any(term in text for term in ("shortest", "minimum length", "אורך מינימלי", "הקצר ביותר")):
        plan.calculation = "min"
        plan.intent = "sum"
    elif plan.intent == "sum" or any(term in text for term in ("total length", "total load", "סך אורך", "סהכ אורך", "עומס כולל")):
        plan.calculation = "sum"
    elif plan.calculation not in {"min", "max", "distinct_count", "percentage"}:
        plan.calculation = "count"

    measure_concepts = []
    if any(term in text for term in ("longest", "shortest", "maximum length", "minimum length", "הארוך ביותר", "הקצר ביותר")):
        measure_concepts.extend(("length", "אורך"))
    for concepts in (
        ("length", "אורך"), ("area", "שטח"), ("volume", "נפח"),
        ("load", "עומס"), ("cost", "עלות"),
    ):
        if any(term in text for term in concepts):
            measure_concepts.extend(concepts)
    if "total" in text and measure_concepts and plan.calculation == "count":
        plan.calculation = "sum"
        plan.intent = "sum"
    if plan.calculation in {"sum", "average", "min", "max"} and not plan.measure_property:
        plan.measure_property = _resolve_property(profile, measure_concepts)
    if plan.measure_property and not _question_names_category(text, profile):
        coverage = profile.get("property_category_coverage", {}).get(plan.measure_property, [])
        measured_categories = [
            str(item.get("category")) for item in coverage if item.get("nonempty")
        ]
        if measured_categories:
            plan.categories = measured_categories
            plan.families = []
            plan.types = []
            plan.population_branches = []
            plan.match_terms = []

    requested_groups: list[str] = []
    if any(term in text for term in ("by level", "per level", "each level", "which level", "לפי קומה", "בכל קומה", "איזו קומה")):
        requested_groups.append("level")
    if any(term in text for term in ("by category", "per category", "which categories", "לפי קטגוריה", "אילו קטגוריות")):
        requested_groups.append("category")
    if any(term in text for term in ("by family", "per family", "לפי משפחה")):
        requested_groups.append("family")
    if any(term in text for term in ("by type", "per type", "types", "לפי סוג", "סוגי")):
        requested_groups.append("type_name")

    mentioned_levels = [
        str(item.get("value", "")) for item in profile.get("levels", [])
        if item.get("value") and normalize(str(item["value"])) in text
    ]
    if len(mentioned_levels) > 1 or any(term in text for term in ("compare", "versus", " vs ", "השווה", "לעומת")):
        if mentioned_levels:
            plan.filters = [condition for condition in plan.filters if normalize(condition.field) != "level"]
            plan.filters.append(PropertyFilter("level", "in", list(dict.fromkeys(mentioned_levels))))
            requested_groups.append("level")

    if requested_groups:
        plan.group_by = list(dict.fromkeys(requested_groups))
        if "grouping" not in plan.required_facets:
            plan.required_facets.append("grouping")
        if requested_groups == ["category"] and not _question_names_category(text, profile):
            plan.categories = []
            plan.families = []
            plan.types = []
            plan.population_branches = []
    if any(term in text for term in ("most", "highest number", "largest number", "הכי הרבה", "המרב")) and plan.group_by:
        plan.sort_by = "count"
        plan.sort_direction = "desc"
        plan.limit = 1
    elif any(term in text for term in ("least", "fewest", "הכי מעט")) and plan.group_by:
        plan.sort_by = "count"
        plan.sort_direction = "asc"
        plan.limit = 1

    mark_match = re.search(r"(?:\bmark\b|סימון)\s*(?:=|is|number|מספר)?\s*([\w.-]+)", question, flags=re.IGNORECASE)
    if mark_match:
        mark_field = _resolve_property(profile, ("mark", "סימון"))
        if mark_field:
            _replace_filter(plan.filters, PropertyFilter(mark_field, "equals", mark_match.group(1)))
            plan.intent = "list"
            plan.group_by = ["category", "family", "type_name", "level", "name"]
            if not _question_names_category(text, profile):
                plan.categories = []
                plan.families = []
                plan.types = []
                plan.population_branches = []
        else:
            plan.unsupported_requirements.append("Resolve the requested Mark property.")

    duplicate_match = any(term in text for term in ("duplicate", "duplicates", "כפול", "כפילויות"))
    if duplicate_match:
        field = _resolve_property(profile, ("mark", "name", "id", "סימון"))
        if field:
            plan.intent = "list"
            plan.calculation = "count"
            plan.group_by = []
            plan.group_by_properties = [field]
            plan.minimum_group_count = 2
            plan.categories = []
            plan.families = []
            plan.types = []
            plan.population_branches = []
            plan.filters = [PropertyFilter(field, "not_missing", "")]
        else:
            plan.unsupported_requirements.append("Resolve the property whose duplicate values were requested.")

    negative_elevation = "negative elevation" in text or any(term in text for term in ("גובה שלילי", "הגבהה שלילית"))
    if negative_elevation:
        field = _resolve_property(profile, ("elevation from level", "elevation", "גובה"))
        if field:
            _replace_filter(plan.filters, PropertyFilter(field, "lt", 0))
            plan.intent = "list"
            plan.group_by = ["category", "family", "type_name", "level", "name"]
            plan.select_properties = list(dict.fromkeys([*plan.select_properties, field]))
            plan.categories = []
            plan.families = []
            plan.types = []
            plan.population_branches = []
        else:
            plan.unsupported_requirements.append("Resolve an elevation property for the negative-value predicate.")

    longer = re.search(
        r"(?:longer\s+than|greater\s+than|ארו(?:ך|כים)\s+מ|מעל)\s*(-?\d+(?:[.,]\d+)?)\s*(mm|cm|m|ft|in)?",
        question, flags=re.IGNORECASE,
    )
    if longer and any(term in text for term in ("length", "longer", "אורך", "ארוך")):
        field = _resolve_property(profile, ("length", "אורך"))
        if field:
            value = f"{longer.group(1)} {longer.group(2)}".strip() if longer.group(2) else float(longer.group(1).replace(",", "."))
            _replace_filter(plan.filters, PropertyFilter(field, "gt", value))
        else:
            plan.unsupported_requirements.append("Resolve a length property for the requested threshold.")

    missing_concepts: tuple[str, ...] = ()
    if any(term in text for term in ("without panel", "no panel", "missing panel", "ללא לוח")):
        missing_concepts = ("panel", "לוח")
    elif any(term in text for term in ("without circuit", "no circuit", "missing circuit", "ללא מעגל")):
        missing_concepts = ("circuit number", "circuit", "מעגל")
    elif any(term in text for term in ("missing material", "without material", "חומר חסר", "ללא חומר")):
        missing_concepts = ("material", "חומר")
    if missing_concepts:
        field = _resolve_property(profile, missing_concepts)
        if field:
            condition = PropertyFilter(field, "missing", "")
            if plan.calculation == "percentage":
                plan.metric_filters = [condition]
            else:
                _replace_filter(plan.filters, condition)
            if any(term in text for term in ("electrical components", "electrical elements", "רכיבי חשמל")):
                coverage = profile.get("property_category_coverage", {}).get(field, [])
                covered = [str(item.get("category")) for item in coverage if item.get("present")]
                if covered:
                    plan.categories = covered
                    plan.families = []
                    plan.types = []
                    plan.population_branches = []
                if not asks_connectivity_language(text):
                    plan.unsupported_requirements.append(
                        "The broad electrical-component population is ambiguous; specify categories or define whether it means records exposing the requested electrical property."
                    )
        else:
            plan.unsupported_requirements.append("Resolve the property used by the missing-value condition.")

    if plan.calculation in {"sum", "average", "min", "max"} and not plan.measure_property:
        plan.unsupported_requirements.append(f"Resolve a numeric measure for {plan.calculation}.")
    if plan.calculation == "percentage" and not plan.metric_filters:
        plan.unsupported_requirements.append("Define the numerator condition for the requested percentage.")
    if any(term in text for term in ("compare", "versus", " vs ", "השווה", "לעומת")) and not (plan.group_by or plan.filter_groups):
        plan.unsupported_requirements.append("Represent both populations requested by the comparison.")
    unsupported_language = (
        (("why", "למה", "מדוע"), "Causal explanations require evidence beyond descriptive model data."),
        (("nearest", "closest", "הקרוב ביותר"), "Nearest-object analysis requires a spatial geometry operator."),
        (("distance between", "מרחק בין"), "Distance-between analysis requires transformed product geometry."),
        (("compliant", "code compliant", "תקן", "עומד בדרישות"), "Compliance requires an explicit external rule set."),
        (("what changed", "difference between versions", "מה השתנה"), "Change analysis requires two versioned project snapshots."),
    )
    for terms, message in unsupported_language:
        if any(term in text for term in terms):
            plan.unsupported_requirements.append(message)
    plan.unsupported_requirements = list(dict.fromkeys(plan.unsupported_requirements))
    return plan


def _resolve_property(profile: dict[str, Any], concepts: Any) -> str | None:
    raw_concepts = [str(item) for item in concepts if str(item).strip()]
    terms = [term for term in expand_terms(raw_concepts) if len(term) >= 2]
    candidate_keys = [
        str(item.get("field")) for item in profile.get("property_candidates", []) if item.get("field")
    ]
    candidate_keys.extend(str(item) for item in profile.get("property_keys", []))
    best: tuple[int, int, str] | None = None
    for position, key in enumerate(dict.fromkeys(candidate_keys)):
        normalized_key = normalize(key)
        leaf = normalize(key.rsplit(".", 1)[-1])
        score = sum(
            20 if term == leaf else 8 if term in leaf else 2 if term in normalized_key else 0
            for term in terms
        )
        if score and (best is None or (score, -position, key) > best):
            best = (score, -position, key)
    return best[2] if best else None


def _resolve_category_name(profile: dict[str, Any], concepts: Any) -> str | None:
    terms = [term for term in expand_terms(str(item) for item in concepts) if len(term) >= 2]
    best: tuple[int, int, str] | None = None
    for branch in profile.get("categories", []):
        category = str(branch.get("category", ""))
        normalized_category = normalize(category)
        category_tokens = normalized_category.split()
        direct = max((100 - max(0, len(category_tokens) - len(term.split())) for term in terms if term in normalized_category), default=0)
        family_hits = sum(
            any(term in normalize(f"{item.get('family', '')} {item.get('type', '')}") for term in terms)
            for item in branch.get("family_types", [])
        )
        score = direct + min(family_hits, 20)
        candidate = (score, -len(category_tokens), category)
        if score and (best is None or candidate > best):
            best = candidate
    return best[2] if best else None


def _replace_filter(filters: list[PropertyFilter], condition: PropertyFilter) -> None:
    target = normalize(condition.field)
    filters[:] = [item for item in filters if normalize(item.field) != target]
    filters.append(condition)


def _question_names_category(text: str, profile: dict[str, Any]) -> bool:
    return any(
        normalize(str(item.get("category", ""))) in text
        for item in profile.get("categories", []) if item.get("category")
    )


def asks_connectivity_language(text: str) -> bool:
    return any(term in text for term in (
        "connected", "connection", "continuity", "physical path", "path to", "חיבור", "המשכיות",
    ))


def apply_domain_guardrails(question: str, profile: dict[str, Any], plan: QueryPlan) -> QueryPlan:
    """Encode common BIM analytical requirements that must not remain free-text planner prose."""
    plan = _apply_generic_requirements(question, profile, plan)
    text = normalize(question)
    categories = {str(item.get("category", "")): item for item in profile.get("categories", [])}
    property_keys = [str(item) for item in profile.get("property_keys", [])]
    facets = set(plan.required_facets) | {"population"}
    if plan.calculation in {"sum", "average", "min", "max"}:
        facets.add("measure")
    if plan.group_by or plan.group_by_properties:
        facets.add("grouping")

    asks_types = any(term in text for term in ("types", "סוגי", "לפי סוג"))
    asks_length = any(term in text for term in ("length", "אורך"))
    asks_material = any(term in text for term in ("material", "חומר"))
    asks_height = any(term in text for term in ("from floor", "above floor", "מהרצפה", "מרצפה"))
    asks_connectivity = asks_connectivity_language(text)

    if asks_types:
        facets.add("grouping")
        if not plan.group_by:
            plan.group_by = ["type_name"]
    if asks_length:
        facets.add("measure")
        plan.measure_property = plan.measure_property or _best_property(property_keys, "length")
        plan.output_unit = plan.output_unit or "m"
    if asks_material:
        facets.add("property_values")
        material = _best_property(property_keys, "material")
        if material and material not in plan.select_properties:
            plan.select_properties.append(material)
    if asks_height:
        facets.add("property_values")
        for concept in ("description", "גובה מרצפה", "elevation from level", "default elevation"):
            field = _best_property(property_keys, concept)
            if field and field not in plan.select_properties:
                plan.select_properties.append(field)
        # These are deterministic projections, not values invented by the planner.  They let the
        # executor distinguish the documented installation height from the IFC placement height.
        plan.select_properties.extend([
            "BIM.Intended Height From Description",
            "IFC.Placement Height Above Storey",
        ])
    if asks_connectivity:
        facets.add("connectivity")
        plan.analysis = "connectivity"
        for concept in ("panel", "circuit number"):
            field = _best_property(property_keys, concept)
            if field and field not in plan.select_properties:
                plan.select_properties.append(field)

    if asks_length and "רוחב" in text or asks_length and "width" in text:
        facets.add("grouping")
        plan.group_by = ["type_name"]
        for concept in ("width", "height"):
            field = _best_property(property_keys, concept)
            if field and field not in plan.group_by_properties:
                plan.group_by_properties.append(field)

    asks_trays = any(term in text for term in ("מגש", "cable tray", "trays"))
    tray_category = _resolve_category_name(profile, ("cable tray", "tray", "מגש"))
    if asks_trays and tray_category:
        plan.categories = [tray_category]
        plan.families = []
        plan.types = []
        plan.match_terms = []
        plan.population_branches = []
    if (asks_material or asks_height) and not asks_length:
        plan.intent = "describe"

    specific_match = re.search(r"מסוג\s+(.+?)(?:,|מהרצפה|$)", question, flags=re.IGNORECASE)
    if specific_match:
        concept = specific_match.group(1).strip()
        plan.match_terms = [concept]
    if specific_match:
        # A planner may bind an English description to the English-named families it saw in the
        # profile.  That is unsafe in multilingual projects: an equivalent Hebrew description is
        # then intersected out.  The semantic term is the population predicate; exact identity and
        # description filters are removed while unrelated spatial/numeric constraints are retained.
        plan.families = []
        plan.types = []
        plan.population_branches = []
        plan.filters = [item for item in plan.filters if not _is_identity_filter(item)]
        plan.group_by = ["family", "type_name"]
        facets.add("grouping")

    asks_nonmetallic_conduit = (
        any(term in text for term in ("צינור", "conduit"))
        and any(term in text for term in ("לא מתכתי", "nonmetallic", "hdpe"))
    )
    conduit_category = _resolve_category_name(profile, ("conduit", "צינור חשמל"))
    if asks_nonmetallic_conduit and conduit_category:
        branch = categories[conduit_category]
        matching = [
            item for item in branch.get("family_types", [])
            if any(term in normalize(f"{item.get('family', '')} {item.get('type', '')}") for term in ("nonmetallic", "hdpe"))
        ]
        plan.categories = [conduit_category]
        plan.families = list(dict.fromkeys(str(item.get("family", "")) for item in matching if item.get("family")))
        plan.types = list(dict.fromkeys(str(item.get("type", "")) for item in matching if item.get("type")))
        plan.match_terms = []
        plan.population_branches = []

    asks_switches = any(term in text for term in ("מפסק", "switch"))
    if asks_switches:
        plan.include_related = True
        facets.add("related_scope")
        if asks_types:
            plan.group_by = ["family", "type_name"]
        related = []
        for category, branch in categories.items():
            if category in plan.categories:
                continue
            for item in branch.get("family_types", []):
                boundary = f"{item.get('family', '')} {item.get('type', '')}"
                if any(term in normalize(boundary) for term in expand_terms(("switch", "מפסק"))):
                    related.append(str(item.get("family") or item.get("type")))
        plan.search_terms = list(dict.fromkeys([
            *related, "switch", "מפסק", "switchboard", "לוח", *plan.search_terms,
        ]))

    endpoint_terms = (
        any(term in text for term in ("שקע", "socket", "outlet"))
        and any(term in text for term in ("מפסק", "switch"))
        and any(term in text for term in ("גופי תאורה", "lighting fixture"))
    )
    socket_category = _resolve_category_name(profile, ("socket", "outlet", "שקע"))
    switch_category = _resolve_category_name(profile, ("lighting switch", "switch", "מפסק"))
    light_category = _resolve_category_name(profile, ("lighting fixture", "luminaire", "גוף תאורה"))
    if endpoint_terms and all((socket_category, switch_category, light_category)):
        fixture_exclusions = _families_matching(
            categories[socket_category], ("opening", "penetration", "sleeve", "פתח למערכות", "הזנה")
        )
        plan.population_branches = [
            QueryBranch(label="Sockets and power points", categories=[socket_category], exclude_terms=fixture_exclusions),
            QueryBranch(label="Switches", categories=[switch_category]),
            QueryBranch(label="Lighting fixtures", categories=[light_category]),
        ]
        plan.categories = []
        plan.families = []
        plan.types = []
        plan.group_by = ["category", "level"]
        facets.add("grouping")

    broad_connectivity = any(term in text for term in (
        "electrical components", "electrical elements", "רכיבי חשמל", "רכיבים חשמל",
    ))
    if asks_connectivity and broad_connectivity:
        transport_terms = expand_terms(("cable tray", "conduit", "מגש"))
        transport_categories = {
            name for name in categories
            if any(term in normalize(name) for term in transport_terms)
        }
        component_categories = [
            name for name in categories
            if name not in transport_categories
        ]
        branches = []
        for category in component_categories:
            exclusions = _families_matching(
                categories[category], ("opening", "penetration", "sleeve", "פתח למערכות")
            )
            branches.append(QueryBranch(label=category, categories=[category], exclude_terms=exclusions))
        if branches:
            plan.population_branches = branches
            plan.categories = []
            plan.families = []
            plan.types = []
            plan.group_by = ["category", "level"]
        plan.interpretation = (
            "Logical panel/circuit assignment and physical IFC continuity are evaluated as "
            "separate checks; physical continuity requires a graph path to a panel."
        )

    # Normalize dimensions after the LLM and domain rules have both contributed fields.  Empty
    # requested properties remain projections (so their absence is reported), but do not create a
    # meaningless `(missing)` grouping column.
    plan.group_by = list(dict.fromkeys(plan.group_by))
    plan.select_properties = list(dict.fromkeys(plan.select_properties))
    empty_properties = {
        normalize(str(item.get("field", "")))
        for item in profile.get("property_stats", [])
        if int(item.get("nonempty", 0)) == 0
    }
    plan.group_by_properties = list(dict.fromkeys(
        field for field in plan.group_by_properties if normalize(field) not in empty_properties
    ))

    plan.required_facets = sorted(facets)
    return plan


def _best_property(keys: list[str], concept: str) -> str | None:
    target = normalize(concept)
    exact_leaf = [key for key in keys if normalize(key.rsplit(".", 1)[-1]) == target]
    if exact_leaf:
        return exact_leaf[0]
    contains = [key for key in keys if target in normalize(key)]
    return contains[0] if contains else None


def _is_identity_filter(item: PropertyFilter) -> bool:
    field = normalize(item.field)
    leaf = field.rsplit(" ", 1)[-1]
    return any(term in field for term in ("description", "תיאור")) or leaf in {
        "name", "family", "type", "typename",
    }


def _families_matching(branch: dict[str, Any], terms: tuple[str, ...]) -> list[str]:
    output = []
    for item in branch.get("family_types", []):
        family = str(item.get("family", ""))
        if any(normalize(term) in normalize(family) for term in terms):
            output.append(family)
    return list(dict.fromkeys(output))


def _measure_property(question: str, property_keys: list[str]) -> str | None:
    concepts = {
        "length": ("length", "אורך"),
        "area": ("area", "שטח"),
        "volume": ("volume", "נפח"),
        "cost": ("cost", "עלות"),
    }
    requested = None
    for concept, terms in concepts.items():
        if any(term in question for term in terms):
            requested = concept
            break
    if not requested:
        return None
    for key in property_keys:
        if requested in normalize(key.rsplit(".", 1)[-1]):
            return key
    return None
