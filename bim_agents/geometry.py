from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

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


def _report(statement: str, value: Any, unit: str, basis: str) -> PipelineReport:
    claim = Claim(statement=statement, value=value, unit=unit, basis=basis)
    return PipelineReport(
        answer=statement,
        claims=[claim],
        stages_used=["Revit Geometry Query", "Deterministic Geometry Derivation"],
        investigation_trace=[basis],
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
    for row in wall_rows:
        area = _quantity(row.get("quantities"), str(config.get("wall_quantity_set")), str(config.get("opaque_area_quantity")))
        if area is not None and row.get("level"):
            opaque_by_level[str(row["level"])] += area
    for row in opening_rows:
        quantity_set = config.get("window_quantity_set") if row.get("ifc_class") == "IfcWindow" else config.get("door_quantity_set")
        area = _quantity(row.get("quantities"), str(quantity_set), str(config.get("opening_area_quantity")))
        if area is not None and row.get("level"):
            opening_by_level[str(row["level"])] += area
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
    percentage = 100.0 * opening_area / (opaque_area + opening_area)
    approximate_percentage = round(percentage / 5.0) * 5
    statement = (
        f"The typical tower façade opening percentage is {percentage:.1f}% "
        f"(approximately {approximate_percentage:g}%). "
        f"It is consistent across {len(tower_levels)} repeated tower floors."
    )
    return _report(
        statement,
        round(percentage, 3),
        "%",
        "Project-scoped Revit facade quantities calculated using scoped knowledge: "
        + str(config.get("semantics") or "typical tower-facade interpretation"),
    )


def calculate_project_geometry(
    calculation: str, bim, allowed_sources: list[str], knowledge: dict[str, Any]
) -> PipelineReport | None:
    """Execute one named geometry derivation selected by the agent from scoped knowledge."""
    if calculation == "section_heights":
        return _section_heights(bim, allowed_sources, knowledge)
    if calculation == "facade_opening_percentage":
        return _facade_opening_percentage(bim, allowed_sources, knowledge)
    if calculation == "tower_max_floor_area":
        return _tower_max_floor_area(bim, allowed_sources, knowledge)
    raise ValueError(f"Unknown project geometry calculation: {calculation!r}")
