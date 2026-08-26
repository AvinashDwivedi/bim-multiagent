from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from .computation import execute_governed_computation
from .models import Claim, PipelineReport


def _identifier(value: Any) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        raise ValueError(f"Unsafe or missing scoped geometry identifier: {text!r}")
    return text


def _quantity(payload: Any, set_name: str, field: str) -> float | None:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        value = (data or {}).get(set_name, {}).get(field)
        return float(value) if value is not None else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _report(
    statement: str,
    value: Any,
    unit: str,
    basis: str,
    *,
    details: list[str] | None = None,
    source_tags: list[str] | None = None,
    method: str = "governed_geometry",
) -> PipelineReport:
    measurement = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        measurement = {
            "source_value": value,
            "source_unit": unit,
            "canonical_unit": unit,
            "conversion_factor": 1.0,
            "conversion_basis": "identity conversion from governed project quantities",
            "source_property": ", ".join(source_tags or []),
        }
    claim = Claim(
        statement=statement,
        value=value,
        unit=unit,
        basis=basis,
        details=details or [],
        measurement=measurement,
        source_tags=source_tags or [],
        method=method,
    )
    return PipelineReport(
        answer=statement,
        claims=[claim],
        stages_used=["Revit Geometry Query", "Deterministic Geometry Derivation"],
        investigation_trace=[basis],
        verification_status="verified",
    )


def _governed_computation_report(
    calculation: str, bim, allowed_sources: list[str], config: dict[str, Any],
) -> PipelineReport:
    result = execute_governed_computation(
        calculation_key=calculation,
        governed_config=config,
        query=bim.query,
        allowed_sources=allowed_sources,
    )
    if result.recipe == "composite_grouped_count":
        group_unit = str(result.provenance.get("group_unit") or "group")
        statement = (
            f"The governed composite population contains {result.total_count} "
            f"{result.unit}. Counts by {group_unit}:"
        )
        details = [
            f"{row['group']}: {row['count']}"
            for row in result.rows
        ]
        total = int(result.total_count or 0)
        claim = Claim(
            statement=statement,
            value=total,
            unit=result.unit,
            basis=result.basis,
            details=details,
            total_count=total,
            displayed_count=min(len(details), total),
            coverage={
                "candidate_count": total,
                "evaluated_count": total,
                "matched_count": total,
                "missing_count": 0,
                "unknown_count": 0,
                "excluded_count": 0,
                "exhaustive": True,
            },
            source_tags=[str(item) for item in result.provenance.get("component_keys") or []],
            method=result.recipe,
            caveats=result.limitations,
        )
    elif result.recipe == "scope_comparison":
        baseline = str(result.provenance.get("baseline_scope") or "baseline")
        details = [
            f"{row['scope']}: {row['count']} {result.unit}"
            for row in result.rows
        ]
        statement = (
            f"The governed scope comparison reports {len(details)} populations "
            f"relative to {baseline.replace('_', ' ')}."
        )
        claim = Claim(
            statement=statement,
            value="; ".join(
                f"{row['scope']}={row['count']}" for row in result.rows
            ),
            unit=result.unit,
            basis=result.basis,
            details=details,
            source_tags=[baseline, *[str(item) for item in result.provenance.get("scope_keys") or []]],
            method=result.recipe,
            caveats=result.limitations,
        )
    elif result.recipe == "property_coverage":
        candidate_count = int(result.provenance.get("candidate_count") or 0)
        populated_count = int(result.provenance.get("populated_count") or 0)
        missing_count = int(result.provenance.get("missing_count") or 0)
        assignment_unit = str(
            result.provenance.get("assignment_unit") or "governed property assignment"
        )
        fill_rate = result.provenance.get("fill_rate_percent")
        rate_text = f" ({fill_rate:g}%)" if isinstance(fill_rate, (int, float)) else ""
        statement = (
            f"Of {candidate_count} governed {result.unit}, {populated_count} have an explicit "
            f"{assignment_unit} and {missing_count} are missing it{rate_text}."
        )
        claim = Claim(
            statement=statement,
            value=missing_count,
            unit=f"{result.unit} missing {assignment_unit}",
            basis=result.basis,
            details=[
                f"Populated: {populated_count}",
                f"Missing: {missing_count}",
                f"Candidate population: {candidate_count}",
            ],
            total_count=candidate_count,
            displayed_count=candidate_count,
            coverage={
                "candidate_count": candidate_count,
                "evaluated_count": candidate_count,
                "matched_count": populated_count,
                "missing_count": missing_count,
                "unknown_count": 0,
                "excluded_count": 0,
                "exhaustive": True,
            },
            source_tags=[
                str(result.provenance.get("property") or "governed property")
            ],
            method=result.recipe,
            caveats=result.limitations,
        )
    else:  # The registry is closed, but retain a defensive boundary here.
        raise ValueError(f"Unsupported governed computation result: {result.recipe!r}.")
    claim.source_tags.append(f"config:{result.config_digest[:12]}")
    return PipelineReport(
        answer=statement,
        claims=[claim],
        limitations=result.limitations,
        stages_used=["Governed Calculation Registry", "Deterministic Computation"],
        investigation_trace=[
            f"Executed {calculation} with recipe {result.recipe} v{result.recipe_version}; "
            f"result digest {result.result_digest[:12]}."
        ],
        verification_status="verified",
    )


def _section_heights(bim, allowed_sources: list[str], knowledge: dict[str, Any]) -> PipelineReport | None:
    config = knowledge.get("massing_sections") or {}
    if not config:
        return None
    slab_label = _identifier(config.get("slab_label"))
    type_property = _identifier(config.get("predefined_type_property"))
    rows = bim.query(
        f"MATCH (roof:`{slab_label}`)-[:SPATIALLY_CONTAINS]-(storey:IfcBuildingStorey) "
        "WHERE roof.source IN $allowed_sources AND storey.source IN $allowed_sources "
        f"AND roof.`{type_property}` = $roof_type "
        "RETURN roof.GlobalID AS roof_id, roof.name AS roof_name, roof.psets_json AS quantities, "
        "storey.canonical_level AS level, storey.placement_z AS elevation_m",
        {"allowed_sources": allowed_sources, "roof_type": config.get("predefined_type_value")},
    )
    plates = []
    for row in rows:
        area = _quantity(row.get("quantities"), str(config.get("quantity_set")), str(config.get("area_quantity")))
        elevation = row.get("elevation_m")
        if area is not None and elevation is not None:
            plates.append((float(area), float(elevation), str(row.get("level") or "")))
    if not plates:
        return None
    # Small terrace fragments are not independent building sections. Primary massing
    # tops are roof plates at least half the area of the largest roof plate.
    fraction = float(config.get("primary_plate_min_largest_fraction"))
    threshold = max(area for area, _, _ in plates) * fraction
    heights = sorted({elevation for area, elevation, _ in plates if area >= threshold})
    if not heights:
        return None
    rendered = ", ".join(f"{height:g} m" for height in heights)
    storeys = bim.query(
        "MATCH (storey:IfcBuildingStorey) WHERE storey.source IN $allowed_sources "
        "AND storey.placement_z IS NOT NULL "
        "RETURN max(toFloat(storey.placement_z)) AS highest_storey_elevation_m",
        {"allowed_sources": allowed_sources},
    )
    highest_storey = (
        storeys[0].get("highest_storey_elevation_m") if storeys else None
    )
    statement = (
        f"The Revit geometry has {len(heights)} primary massing sections with roof elevations "
        f"of {rendered} above the ground/project datum."
    )
    if highest_storey is not None:
        statement += (
            f" The highest defined storey reference elevation is "
            f"{float(highest_storey):g} m. These are geometry reference elevations, "
            "not a directly modelled total building-height quantity."
        )
    return _report(
        statement,
        rendered,
        "m",
        "Project-scoped roof gross areas joined to containing-storey elevations using scoped knowledge: "
        + str(config.get("semantics") or "primary massing-section interpretation"),
        details=[
            *(f"Primary massing-section roof elevation: {height:g} m." for height in heights),
            *(
                [f"Highest defined storey reference elevation: {float(highest_storey):g} m."]
                if highest_storey is not None else []
            ),
        ],
        source_tags=[
            f"{slab_label}.{config.get('quantity_set')}.{config.get('area_quantity')}",
            "IfcBuildingStorey.placement_z",
        ],
        method="primary_roof_plate_elevations",
    )


def _tower_max_floor_area(
    bim, allowed_sources: list[str], knowledge: dict[str, Any]
) -> PipelineReport | None:
    config = knowledge.get("tower_floor_area") or {}
    if not config:
        return None
    label = _identifier(config.get("space_label"))
    type_property = _identifier(config.get("type_property"))
    basis_property = _identifier(config.get("basis_property"))
    area_property = _identifier(config.get("area_property"))
    level_property = _identifier(config.get("level_property"))
    rows = bim.query(
        f"MATCH (space:`{label}`) WHERE space.source IN $allowed_sources "
        f"AND space.`{type_property}` = $space_type "
        f"AND space.`{basis_property}` = $area_basis "
        f"AND space.`{area_property}` IS NOT NULL "
        f"RETURN space.`{level_property}` AS level, "
        f"sum(toFloat(space.`{area_property}`)) AS area_m2, count(*) AS records "
        "ORDER BY area_m2 DESC, level",
        {
            "allowed_sources": allowed_sources,
            "space_type": config.get("floor_plate_type"),
            "area_basis": config.get("floor_plate_basis"),
        },
    )
    values = [row for row in rows if row.get("area_m2") is not None]
    if not values:
        return None
    precision = int(config.get("repeated_area_precision", 3))
    signatures = Counter(round(float(row["area_m2"]), precision) for row in values)
    repeated = [
        (count, area) for area, count in signatures.items()
        if count >= int(config.get("repeated_floor_min_count", 2))
    ]
    if not repeated:
        return None
    _, maximum = max(repeated, key=lambda item: (item[0], item[1]))
    levels = sorted(
        str(row.get("level") or "unassigned level")
        for row in values
        if round(float(row["area_m2"]), precision) == maximum
    )
    level_text = ", ".join(levels)
    basis = str(config.get("floor_plate_basis"))
    statement = (
        f"The maximum repeated tower floor plate is {maximum:.2f} m² {basis}, attained on "
        f"{level_text}. The tower scope is the project-governed repeated upper-storey "
        "residential-zone floor-plate interpretation."
    )
    return _report(
        statement,
        round(maximum, 3),
        "m²",
        "Project-scoped residential floor plates calculated using scoped knowledge: "
        + str(config.get("semantics") or "tower floor-plate interpretation"),
        details=[
            f"Attaining floor/storey: {level}." for level in levels
        ] + [f"Area measurement basis: {basis}."],
        source_tags=[
            f"{label}.{area_property}", f"{label}.{level_property}",
            f"{label}.{basis_property}",
        ],
        method="maximum_repeated_floor_plate",
    )


def _facade_opening_percentage(bim, allowed_sources: list[str], knowledge: dict[str, Any]) -> PipelineReport | None:
    config = knowledge.get("tower_facade") or {}
    if not config:
        return None
    wall_label = _identifier(config.get("wall_label"))
    wall_name_property = _identifier(config.get("wall_name_property"))
    orientation_property = _identifier(config.get("orientation_property"))
    external_property = _identifier(config.get("external_property"))
    opening_labels = [_identifier(value) for value in config.get("opening_labels") or []]
    if not opening_labels:
        return None
    opening_predicate = " OR ".join(f"opening:`{label}`" for label in opening_labels)
    wall_rows = bim.query(
        f"MATCH (wall:`{wall_label}`) WHERE wall.source IN $allowed_sources "
        f"AND wall.`{wall_name_property}` = $wall_name AND wall.`{orientation_property}` IS NOT NULL "
        "RETURN wall.GlobalID AS id, wall.canonical_level AS level, wall.psets_json AS quantities",
        {"allowed_sources": allowed_sources, "wall_name": config.get("wall_name_value")},
    )
    opening_rows = bim.query(
        "MATCH (opening) WHERE opening.source IN $allowed_sources "
        f"AND ({opening_predicate}) "
        f"AND coalesce(toBoolean(opening.`{external_property}`), false) "
        "RETURN opening.GlobalID AS id, opening.canonical_level AS level, "
        "opening.IFCtype AS ifc_class, opening.psets_json AS quantities",
        {"allowed_sources": allowed_sources},
    )
    opaque_by_level: dict[str, float] = defaultdict(float)
    opening_by_level: dict[str, float] = defaultdict(float)
    valid_walls: list[tuple[str, float]] = []
    valid_openings: list[tuple[str, float]] = []
    for row in wall_rows:
        area = _quantity(row.get("quantities"), str(config.get("wall_quantity_set")), str(config.get("opaque_area_quantity")))
        if area is not None and row.get("level"):
            level = str(row["level"])
            opaque_by_level[level] += area
            valid_walls.append((level, area))
    for row in opening_rows:
        quantity_set = config.get("window_quantity_set") if row.get("ifc_class") == "IfcWindow" else config.get("door_quantity_set")
        area = _quantity(row.get("quantities"), str(quantity_set), str(config.get("opening_area_quantity")))
        if area is not None and row.get("level"):
            level = str(row["level"])
            opening_by_level[level] += area
            valid_openings.append((level, area))
    common_levels = set(opaque_by_level) & set(opening_by_level)
    if not common_levels:
        return None
    # Repeated equal façade schedules identify the tower's typical floor independently
    # of localized podium geometry. Use millimetre-square precision for grouping noise.
    signatures = {
        level: (round(opaque_by_level[level], 3), round(opening_by_level[level], 3))
        for level in common_levels
    }
    signature, repetitions = Counter(signatures.values()).most_common(1)[0]
    tower_levels = sorted(level for level, item in signatures.items() if item == signature)
    if repetitions < int(config.get("repeated_floor_min_count")):
        return None
    opaque_area, opening_area = signature
    denominator_area = opaque_area + opening_area
    percentage = 100.0 * opening_area / denominator_area
    approximate_percentage = round(percentage / 5.0) * 5
    statement = (
        f"The typical tower façade opening percentage is {percentage:.1f}% "
        f"(approximately {approximate_percentage:g}%). "
        f"It is consistent across {len(tower_levels)} repeated tower floors."
    )
    candidate_count = len(wall_rows) + len(opening_rows)
    valid_count = len(valid_walls) + len(valid_openings)
    selected_count = sum(
        1 for level, _area in [*valid_walls, *valid_openings] if level in tower_levels
    )
    missing_count = max(candidate_count - valid_count, 0)
    excluded_count = max(valid_count - selected_count, 0)
    basis_text = (
        "Project-scoped Revit facade quantities calculated using scoped knowledge: "
        + str(config.get("semantics") or "typical tower-facade interpretation")
    )
    claim = Claim(
        statement=statement,
        value=round(percentage, 3),
        unit="%",
        basis=basis_text,
        details=[
            f"Qualifying opening-area numerator: {opening_area:.3f} m².",
            f"Tower-facade denominator area: {denominator_area:.3f} m².",
            f"Opaque facade component: {opaque_area:.3f} m².",
            "Calculation basis: opening area / (opening area + oriented opaque facade area) × 100.",
            "Repeated tower floors: " + ", ".join(tower_levels) + ".",
        ],
        coverage={
            "candidate_count": candidate_count,
            "evaluated_count": candidate_count,
            "matched_count": selected_count,
            "missing_count": missing_count,
            "unknown_count": 0,
            "excluded_count": excluded_count,
            "exhaustive": True,
        },
        measurement={
            "source_value": round(percentage, 3),
            "source_unit": "%",
            "canonical_unit": "%",
            "conversion_factor": 1.0,
            "conversion_basis": "ratio converted to percent exactly once",
            "source_property": (
                f"{wall_label}.{config.get('wall_quantity_set')}.{config.get('opaque_area_quantity')} + "
                f"{','.join(opening_labels)}.{config.get('opening_area_quantity')}"
            ),
        },
        source_tags=[
            f"{wall_label}.{config.get('opaque_area_quantity')}",
            *(f"{label}.{config.get('opening_area_quantity')}" for label in opening_labels),
        ],
        method="typical_facade_opening_ratio",
    )
    return PipelineReport(
        answer=statement,
        claims=[claim],
        stages_used=["Revit Geometry Query", "Deterministic Geometry Derivation"],
        investigation_trace=[basis_text],
        verification_status="verified",
    )


def calculate_project_geometry(
    calculation: str, bim, allowed_sources: list[str], knowledge: dict[str, Any]
) -> PipelineReport | None:
    """Execute one named allowlisted project calculation from scoped knowledge."""
    governed = [
        config for config in knowledge.values()
        if isinstance(config, dict) and config.get("calculation") == calculation
    ]
    if len(governed) > 1:
        raise ValueError(f"Calculation {calculation!r} is ambiguously configured.")
    if governed and governed[0].get("recipe"):
        return _governed_computation_report(calculation, bim, allowed_sources, governed[0])
    if calculation == "section_heights":
        return _section_heights(bim, allowed_sources, knowledge)
    if calculation == "facade_opening_percentage":
        return _facade_opening_percentage(bim, allowed_sources, knowledge)
    if calculation == "tower_max_floor_area":
        return _tower_max_floor_area(bim, allowed_sources, knowledge)
    raise ValueError(f"Unknown project geometry calculation: {calculation!r}")
