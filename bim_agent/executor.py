from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .dataset import ProjectDataset, expand_terms, normalize
from .models import ElementRecord, PropertyFilter, QueryBranch, QueryEvidence, QueryPlan


class DeterministicExecutor:
    def __init__(self, dataset: ProjectDataset):
        self.dataset = dataset

    def execute(self, plan: QueryPlan) -> QueryEvidence:
        population = self.dataset.physical_records if plan.scope == "physical_instances" else self.dataset.records
        raw_selected = [record for record in population if self._matches_plan(record, plan)]
        raw_selected = [record for record in raw_selected if self._matches_filters(record, plan.filters)]
        if plan.filter_groups:
            raw_selected = [
                record for record in raw_selected
                if any(self._matches_filters(record, group) for group in plan.filter_groups)
            ]
        selected = _distinct(raw_selected)

        groups = self._groups(selected, plan)
        related = self._related(selected, plan) if plan.include_related else []
        operation_value: float | int | None = len(selected)
        metric_numerator: float | int | None = None
        metric_denominator: float | int | None = None
        unit = None
        complete = True
        calculation = "sum" if plan.intent == "sum" and plan.calculation == "count" else plan.calculation
        if calculation in {"sum", "average", "min", "max"} and plan.measure_property:
            values = []
            units = Counter()
            for record in selected:
                raw = _field(record, plan.measure_property)
                number, value_unit = _number_and_unit(raw)
                if number is not None:
                    number, value_unit = _convert_unit(number, value_unit, plan.output_unit)
                    values.append(number)
                    if value_unit:
                        units[value_unit] += 1
            if calculation == "sum":
                operation_value = sum(values)
            elif calculation == "average":
                operation_value = sum(values) / len(values) if values else None
            elif calculation == "min":
                operation_value = min(values) if values else None
            else:
                operation_value = max(values) if values else None
            if len(units) == 1:
                unit = next(iter(units))
            elif len(units) > 1:
                complete = False
            if len(values) != len(selected):
                complete = False
        elif calculation == "distinct_count" and plan.distinct_property:
            values = {_field(record, plan.distinct_property) for record in selected}
            operation_value = len({value for value in values if value.strip()})
        elif calculation == "percentage":
            metric_denominator = len(selected)
            metric_numerator = sum(self._matches_filters(record, plan.metric_filters) for record in selected)
            operation_value = (
                100.0 * float(metric_numerator) / float(metric_denominator)
                if metric_denominator else None
            )
            unit = "%"

        grouped_measurements = self._grouped_measurements(selected, plan)
        property_summaries = self._property_summaries(selected, plan.select_properties)
        connectivity = self._connectivity(selected) if plan.analysis == "connectivity" else None
        category_counts = [
            {"category": category, "count": count}
            for category, count in Counter(record.category for record in selected).most_common()
        ]

        samples = [
            {
                "object_id": record.object_id,
                "identity": record.identity,
                "name": record.name,
                "category": record.category,
                "family": record.family,
                "type": record.type_name,
                "level": record.level,
            }
            for record in selected[:20]
        ]
        return QueryEvidence(
            selected_count=len(raw_selected),
            distinct_identity_count=len({record.identity for record in selected}),
            groups=groups,
            samples=samples,
            related_groups=related,
            operation_value=operation_value,
            unit=unit,
            complete=complete,
            grouped_measurements=grouped_measurements,
            property_summaries=property_summaries,
            connectivity=connectivity,
            category_counts=category_counts,
            metric_numerator=metric_numerator,
            metric_denominator=metric_denominator,
        )

    def _matches_plan(self, record: ElementRecord, plan: QueryPlan) -> bool:
        if plan.population_branches:
            return any(self._matches_branch(record, branch) for branch in plan.population_branches)
        return self._matches_semantic(record, plan)

    def _matches_branch(self, record: ElementRecord, branch: QueryBranch) -> bool:
        if branch.categories and not _exact(record.category, branch.categories):
            return False
        if branch.families and not _exact(record.family, branch.families):
            return False
        if branch.types and not _exact(record.type_name, branch.types):
            return False
        text = normalize(record.searchable_text)
        if any(normalize(term) in text for term in branch.exclude_terms if normalize(term)):
            return False
        if branch.match_terms and not _matches_any_term(record, branch.match_terms):
            return False
        return self._matches_filters(record, branch.filters)

    def _matches_semantic(self, record: ElementRecord, plan: QueryPlan) -> bool:
        category_ok = not plan.categories or _exact(record.category, plan.categories)
        family_ok = not plan.families or _exact(record.family, plan.families)
        type_ok = not plan.types or _exact(record.type_name, plan.types)
        if not category_ok:
            return False
        branch_ok = family_ok and type_ok
        if not branch_ok:
            return False

        text = normalize(record.searchable_text)
        if any(normalize(term) in text for term in plan.exclude_terms if normalize(term)):
            return False
        has_explicit_boundary = bool(
            plan.categories or plan.families or plan.types
            or plan.filters or plan.filter_groups or plan.metric_filters
        )
        if plan.match_terms and not _matches_any_term(record, plan.match_terms):
            return False
        if not has_explicit_boundary:
            terms = [term for term in expand_terms(plan.match_terms or plan.search_terms) if len(term) >= 2]
            return bool(terms) and any(term in text for term in terms)
        return True

    @staticmethod
    def _matches_filters(record: ElementRecord, filters: Iterable[PropertyFilter]) -> bool:
        for condition in filters:
            actual = _field(record, condition.field)
            if not _compare(actual, condition.operator, condition.value):
                return False
        return True

    def _groups(self, records: list[ElementRecord], plan: QueryPlan) -> list[dict[str, Any]]:
        fields = [*plan.group_by, *plan.group_by_properties]
        if not fields:
            fields = ["category", "family", "type_name"]
        grouped: Counter[tuple[str, ...]] = Counter()
        for record in records:
            values = []
            for field in fields:
                if field in {"category", "family", "type_name", "level", "name"}:
                    values.append(str(getattr(record, field, "")))
                else:
                    values.append(self._value(record, field))
            grouped[tuple(values)] += 1
        output = []
        for values, count in grouped.most_common():
            row = {field: value for field, value in zip(fields, values)}
            row["count"] = count
            output.append(row)
        if plan.minimum_group_count:
            output = [item for item in output if item["count"] >= plan.minimum_group_count]
        if plan.sort_by:
            sort_field = "count" if plan.sort_by == "count" else plan.sort_by
            output.sort(
                key=lambda item: _sortable(item.get(sort_field)),
                reverse=plan.sort_direction == "desc",
            )
        if plan.limit:
            output = output[:plan.limit]
        return output

    def _grouped_measurements(self, records: list[ElementRecord], plan: QueryPlan) -> list[dict[str, Any]]:
        if not plan.measure_property or not (plan.group_by or plan.group_by_properties):
            return []
        grouped: dict[tuple[str, ...], dict[str, Any]] = {}
        for record in records:
            dimensions = [str(getattr(record, field, "")) for field in plan.group_by]
            dimensions.extend(_field(record, field) for field in plan.group_by_properties)
            key = tuple(dimensions)
            row = grouped.setdefault(key, {"count": 0, "measured_count": 0, "sum": 0.0, "unit": None})
            row["count"] += 1
            number, unit = _number_and_unit(_field(record, plan.measure_property))
            if number is not None:
                number, unit = _convert_unit(number, unit, plan.output_unit)
                row["measured_count"] += 1
                row["sum"] += number
                row["min"] = number if "min" not in row else min(row["min"], number)
                row["max"] = number if "max" not in row else max(row["max"], number)
                row["unit"] = unit or row["unit"]
        output = []
        fields = [*plan.group_by, *plan.group_by_properties]
        for values, row in grouped.items():
            item = {field: value for field, value in zip(fields, values)}
            row["average"] = row["sum"] / row["measured_count"] if row["measured_count"] else None
            row["value"] = row.get(plan.calculation, row["sum"])
            item.update(row)
            output.append(item)
        return sorted(output, key=lambda item: tuple(str(item.get(field, "")) for field in fields))

    def _property_summaries(self, records: list[ElementRecord], fields: list[str]) -> list[dict[str, Any]]:
        output = []
        for field in fields:
            projected = [(record, self._value(record, field)) for record in records]
            values = Counter(value for _, value in projected if value.strip())
            present = sum(bool(value.strip()) for _, value in projected)
            grouped: Counter[tuple[str, str, str]] = Counter(
                (record.family, record.type_name, value)
                for record, value in projected if value.strip()
            )
            item = {
                "field": field,
                "selected": len(records),
                "present": present,
                "missing": len(records) - present,
                "values": [{"value": value, "count": count} for value, count in values.most_common(50)],
                "by_family_type": [
                    {"family": family, "type": type_name, "value": value, "count": count}
                    for (family, type_name, value), count in grouped.most_common(80)
                ],
            }
            if "material" in normalize(field):
                item["ifc_material_associations"] = sum(self.dataset.has_ifc_material(record) for record in records)
            output.append(item)
        return output

    def _value(self, record: ElementRecord, field: str) -> str:
        normalized = normalize(field)
        if normalized == normalize("IFC.Placement Height Above Storey"):
            value, unit = self.dataset.ifc_placement_height(record)
            return f"{value:g} {unit}" if value is not None else ""
        if normalized == normalize("BIM.Intended Height From Description"):
            return _intended_height(record)
        return _field(record, field)

    def _connectivity(self, records: list[ElementRecord]) -> dict[str, Any]:
        panel_field = "Electrical - Loads.Panel"
        circuit_field = "Electrical - Loads.Circuit Number"
        panel_values = Counter(_field(record, panel_field) for record in records if _field(record, panel_field).strip())
        without_panel = sum(not _field(record, panel_field).strip() for record in records)
        without_circuit = sum(not _field(record, circuit_field).strip() for record in records)
        graph = self.dataset.connectivity_summary(records)
        isolated = [
            record for record in records
            if self.dataset.has_ifc_ports(record) and not self.dataset.has_connected_ifc_port(record)
        ]
        isolated_groups = Counter((record.category, record.family, record.type_name) for record in isolated)
        isolated_panels = [
            {"category": category, "family": family, "type": type_name, "count": count}
            for (category, family, type_name), count in isolated_groups.most_common()
            if any(term in normalize(f"{category} {family}") for term in ("switchboard", "panel", "לוח"))
        ]
        graph["isolated_selected_groups"] = [
            {"category": category, "family": family, "type": type_name, "count": count}
            for (category, family, type_name), count in isolated_groups.most_common(20)
        ]
        graph["isolated_panels"] = isolated_panels
        path_analysis = self._panel_path_analysis(records)
        graph["panel_path_analysis"] = path_analysis
        return {
            "logical": {
                "elements": len(records),
                "with_panel": len(records) - without_panel,
                "without_panel": without_panel,
                "without_circuit": without_circuit,
                "panels": [{"value": value, "count": count} for value, count in panel_values.most_common()],
            },
            "ifc": graph,
        }

    def _panel_path_analysis(self, records: list[ElementRecord]) -> dict[str, Any]:
        if self.dataset.ifc_graph is None:
            return {"executed": False, "reason": "No STEP IFC graph is available."}
        ifc = self.dataset.ifc_graph
        components = ifc.component_index()
        panel_records = [record for record in self.dataset.physical_records if _is_panel_record(record)]
        panel_by_name: dict[str, ElementRecord] = {}
        panel_entities: set[int] = set()
        for panel in panel_records:
            entity = ifc.entity_for_guid(panel.global_id)
            if entity is not None:
                panel_entities.add(entity)
            for label in (
                _field(panel, "General.Panel Name"), panel.name, panel.family, panel.type_name,
            ):
                if normalize(label):
                    panel_by_name.setdefault(normalize(label), panel)

        no_panel: set[str] = set()
        no_circuit: set[str] = set()
        no_path_any: set[str] = set()
        path_any: set[str] = set()
        path_assigned: set[str] = set()
        no_path_assigned: set[str] = set()
        unresolved_assignment: set[str] = set()
        selected_with_entity = 0
        resolved_assignments = 0
        for record in records:
            identity = record.identity
            panel_name = _field(record, "Electrical - Loads.Panel").strip()
            circuit = _field(record, "Electrical - Loads.Circuit Number").strip()
            if not panel_name:
                no_panel.add(identity)
            if not circuit:
                no_circuit.add(identity)

            entity = ifc.entity_for_guid(record.global_id)
            if entity is None:
                no_path_any.add(identity)
            else:
                selected_with_entity += 1
                component = components.get(entity)
                connected_panels = {
                    candidate for candidate in panel_entities
                    if candidate != entity and component is not None and components.get(candidate) == component
                }
                (path_any if connected_panels else no_path_any).add(identity)

            if not panel_name:
                continue
            target_record = panel_by_name.get(normalize(panel_name))
            target_entity = ifc.entity_for_guid(target_record.global_id) if target_record else None
            if target_entity is None:
                unresolved_assignment.add(identity)
                continue
            resolved_assignments += 1
            if (
                entity is not None
                and entity != target_entity
                and components.get(entity) is not None
                and components.get(entity) == components.get(target_entity)
            ):
                path_assigned.add(identity)
            else:
                no_path_assigned.add(identity)

        either_failure = no_panel | no_path_any
        both_failure = no_panel & no_path_any
        return {
            "executed": True,
            "selected": len(records),
            "selected_with_ifc_entity": selected_with_entity,
            "panel_entities": len(panel_entities),
            "logical_missing_panel": len(no_panel),
            "logical_missing_circuit": len(no_circuit),
            "physical_path_to_any_panel": len(path_any),
            "physical_no_path_to_any_panel": len(no_path_any),
            "assigned_panel_resolved": resolved_assignments,
            "assigned_panel_unresolved": len(unresolved_assignment),
            "path_to_assigned_panel": len(path_assigned),
            "no_path_to_assigned_panel": len(no_path_assigned),
            "either_logical_or_physical_failure": len(either_failure),
            "both_logical_and_physical_failure": len(both_failure),
            "complete": len(unresolved_assignment) == 0,
        }

    def _related(self, selected: list[ElementRecord], plan: QueryPlan) -> list[dict[str, Any]]:
        if not plan.search_terms:
            return []
        selected_ids = {record.identity for record in selected}
        scored = []
        phrases = [normalize(term) for term in plan.search_terms if normalize(term)]
        expanded = [term for term in expand_terms(plan.search_terms) if len(term) >= 3]
        for record in self.dataset.physical_records:
            if record.identity in selected_ids:
                continue
            boundary = normalize(f"{record.category} {record.family} {record.type_name} {record.name}")
            text = normalize(record.searchable_text)
            phrase_score = max((100 for phrase in phrases if phrase in boundary), default=0)
            value_score = max((50 for phrase in phrases if phrase in text), default=0)
            token_score = sum(term in boundary for term in expanded)
            score = phrase_score + value_score + token_score
            if score:
                scored.append((score, record))
        grouped: dict[tuple[str, str, str], dict[str, int]] = {}
        distinct_scored = _distinct_scored(scored)
        # Exact family/type/name boundary hits are much stronger counterexamples than incidental
        # property-value hits (for example every receptacle carrying a Panel property).  Once exact
        # project-vocabulary candidates exist, suppress those incidental matches.
        if any(score >= 100 for score, _ in distinct_scored):
            distinct_scored = [(score, record) for score, record in distinct_scored if score >= 100]
        for score, record in distinct_scored:
            key = (record.category, record.family, record.type_name)
            item = grouped.setdefault(key, {"count": 0, "score": 0})
            item["count"] += 1
            item["score"] = max(item["score"], score)
        return [
            {"category": category, "family": family, "type": type_name, **values}
            for (category, family, type_name), values in sorted(
                grouped.items(), key=lambda item: (-item[1]["score"], -item[1]["count"], item[0])
            )[:30]
        ]


def _distinct(records: Iterable[ElementRecord]) -> list[ElementRecord]:
    output = []
    seen = set()
    for record in records:
        if record.identity in seen:
            continue
        seen.add(record.identity)
        output.append(record)
    return output


def _distinct_scored(records: Iterable[tuple[int, ElementRecord]]) -> list[tuple[int, ElementRecord]]:
    output = []
    seen = set()
    for score, record in sorted(records, key=lambda item: -item[0]):
        if record.identity in seen:
            continue
        seen.add(record.identity)
        output.append((score, record))
    return output


def _matches_any_term(record: ElementRecord, terms: Iterable[str]) -> bool:
    texts = [
        normalize(value)
        for value in (
            record.name, record.category, record.family, record.type_name, record.level,
            *record.flat_properties.values(),
        )
        if value
    ]
    for text in texts:
        for raw in terms:
            phrase = normalize(raw)
            if not phrase:
                continue
            if phrase in text:
                return True
            concepts = []
            for token in phrase.split():
                if (len(token) < 3 and not token.isdigit()) or token in _MATCH_STOPWORDS:
                    continue
                alternatives = [
                    value for value in expand_terms([token])
                    if (len(value) >= 3 or value.isdigit()) and value not in _MATCH_STOPWORDS
                ]
                concepts.append(alternatives)
            if concepts and all(any(_term_in_text(alternative, text) for alternative in choices) for choices in concepts):
                return True
    return False


def _term_in_text(term: str, text: str) -> bool:
    if term.isdigit():
        return bool(re.search(rf"(?<!\d){re.escape(term)}(?!\d)", text))
    return term in text


def _exact(value: str, allowed: Iterable[str]) -> bool:
    normalized = normalize(value)
    return normalized in {normalize(item) for item in allowed}


def _field(record: ElementRecord, field: str) -> str:
    virtual = {
        "object_id": str(record.object_id),
        "name": record.name,
        "category": record.category,
        "family": record.family,
        "type": record.type_name,
        "type_name": record.type_name,
        "level": record.level,
        "globalid": record.global_id or "",
        "global_id": record.global_id or "",
        "externalid": record.external_id or "",
        "external_id": record.external_id or "",
    }
    target = normalize(field)
    if target in virtual:
        return virtual[target]
    for key, value in record.flat_properties.items():
        if normalize(key) == target or normalize(key.rsplit(".", 1)[-1]) == target:
            return value
    return ""


def _compare(actual: str, operator: str, expected: Any) -> bool:
    if operator == "missing":
        return not str(actual).strip()
    if operator == "not_missing":
        return bool(str(actual).strip())
    if operator == "equals":
        return normalize(actual) == normalize(str(expected))
    if operator == "not_equals":
        return normalize(actual) != normalize(str(expected))
    if operator == "contains":
        return normalize(str(expected)) in normalize(actual)
    if operator == "not_contains":
        return normalize(str(expected)) not in normalize(actual)
    if operator == "in":
        options = expected if isinstance(expected, list) else [expected]
        return normalize(actual) in {normalize(str(item)) for item in options}
    if operator == "not_in":
        options = expected if isinstance(expected, list) else [expected]
        return normalize(actual) not in {normalize(str(item)) for item in options}
    actual_number, actual_unit = _number_and_unit(actual)
    expected_number, expected_unit = _number_and_unit(str(expected))
    if expected_number is None:
        return False
    if actual_number is None:
        return False
    if expected_unit and actual_unit:
        actual_number, _ = _convert_unit(actual_number, actual_unit, expected_unit)
    return {
        "gt": actual_number > expected_number,
        "gte": actual_number >= expected_number,
        "lt": actual_number < expected_number,
        "lte": actual_number <= expected_number,
    }.get(operator, False)


def _sortable(value: Any) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return 1, float(value)
    number, _ = _number_and_unit(str(value))
    return (1, number) if number is not None else (0, normalize(str(value)))


def _number_and_unit(value: str) -> tuple[float | None, str | None]:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value).replace(",", ""))
    if not match:
        return None, None
    number = float(match.group(0))
    remainder = str(value)[match.end():].strip().casefold()
    unit_match = re.match(r"(mm|cm|m|ft|in|m²|m2|m³|m3)\b", remainder)
    return number, unit_match.group(1) if unit_match else None


def _convert_unit(number: float, source: str | None, target: str | None) -> tuple[float, str | None]:
    if not target or not source or normalize(source) == normalize(target):
        return number, target or source
    linear_to_m = {"mm": 0.001, "cm": 0.01, "m": 1.0, "ft": 0.3048, "in": 0.0254}
    source_key = source.casefold()
    target_key = target.casefold()
    if source_key in linear_to_m and target_key in linear_to_m:
        return number * linear_to_m[source_key] / linear_to_m[target_key], target
    return number, source


def _intended_height(record: ElementRecord) -> str:
    descriptions = " | ".join(
        value for key, value in record.flat_properties.items()
        if "description" in normalize(key) or "תיאור" in normalize(key)
    )
    patterns = (
        r"\bH\s*=\s*(-?\d+(?:[.,]\d+)?)",
        r"בגובה\s*(-?\d+(?:[.,]\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, descriptions, flags=re.IGNORECASE)
        if match:
            return f"{float(match.group(1).replace(',', '.')):g} cm"
    # Some families keep the intended installation height in a dedicated H type parameter.
    value = _field(record, "Identity Data.H")
    number, unit = _number_and_unit(value)
    return f"{number:g} {unit or 'cm'}" if number is not None else ""


def _is_panel_record(record: ElementRecord) -> bool:
    text = normalize(f"{record.category} {record.family} {record.type_name} {record.name}")
    has_panel_name = bool(_field(record, "General.Panel Name").strip())
    return has_panel_name or any(term in text for term in (
        "electrical equipment", "switchboard", "panel", "לוח", "ארון חשמל",
    ))


_MATCH_STOPWORDS = {
    "for", "from", "with", "without", "type", "types", "project", "there",
    "של", "לפי", "מסוג", "פרויקט", "בפרויקט", "בפרוייקט",
}
