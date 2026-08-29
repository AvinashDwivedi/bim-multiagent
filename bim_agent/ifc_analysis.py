from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any


class IfcAnalysisError(ValueError):
    pass


class IfcAnalysisEngine:
    """Lazy IfcOpenShell access for model-selected semantic and geometry analysis."""

    def __init__(self, ifc_path: Path, records: list[dict[str, Any]]):
        self.ifc_path = ifc_path
        self.records = records
        self._model: Any | None = None
        self._geometry_inventory: list[dict[str, Any]] | None = None
        self._lock = threading.RLock()
        self._record_by_id = {str(item["object_id"]): item for item in records}
        self._records_by_guid: dict[str, list[dict[str, Any]]] = {}
        self._records_by_external_id: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            guid = record["properties"].get("IFC Parameters.IfcGUID")
            if guid:
                self._records_by_guid.setdefault(str(guid).casefold(), []).append(record)
            external_id = record.get("external_id")
            if external_id:
                self._records_by_external_id.setdefault(str(external_id).casefold(), []).append(record)

    @property
    def available(self) -> bool:
        header = self._header_bytes()
        if header[:16] == b"SQLite format 3\x00" or b"FILE_SCHEMA" not in header.upper():
            return False
        try:
            import ifcopenshell  # noqa: F401
        except ImportError:
            return False
        return True

    def _header_bytes(self) -> bytes:
        with self.ifc_path.open("rb") as stream:
            return stream.read(256 * 1024)

    def capability(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "engine": "IfcOpenShell" if self.available else None,
            "supports": [
                "named IFC attributes",
                "record-to-IFC identity candidates",
                "relationship roles",
                "world placement",
                "mesh bounding boxes",
                "mesh solid volume",
                "surface and XY projected areas",
                "spatial containment",
            ] if self.available else [],
        }

    def semantic_index(self) -> dict[str, list[tuple[Any, ...]]]:
        if not self.available:
            return {"objects": [], "relationships": [], "candidates": []}
        with self._lock:
            model = self._load_model()
            objects: list[tuple[Any, ...]] = []
            candidates: list[tuple[Any, ...]] = []
            # Include both instances (IfcObject) and definitions/types so the
            # agent can inspect type-instance identity rather than silently
            # losing type records at ingestion.
            for entity in model.by_type("IfcObjectDefinition"):
                global_id = _ifc_text(entity, "GlobalId")
                container = _ifc_container(entity)
                placement = getattr(entity, "ObjectPlacement", None)
                representation = getattr(entity, "Representation", None)
                objects.append((
                    entity.id(),
                    global_id,
                    entity.is_a(),
                    _ifc_text(entity, "Name"),
                    _ifc_text(entity, "Description"),
                    _ifc_text(entity, "ObjectType"),
                    _ifc_text(entity, "Tag"),
                    _ifc_text(entity, "PredefinedType"),
                    container.id() if container is not None else None,
                    _ifc_text(container, "GlobalId"),
                    container.is_a() if container is not None else None,
                    _ifc_text(container, "Name"),
                    placement.id() if placement is not None else None,
                    representation.id() if representation is not None else None,
                ))
                if global_id:
                    seen: set[tuple[str, str]] = set()
                    for record in self._records_by_guid.get(global_id.casefold(), []):
                        key = (str(record["object_id"]), "ifc_guid")
                        if key not in seen:
                            candidates.append((record["object_id"], entity.id(), global_id, "IFC Parameters.IfcGUID", 1.0))
                            seen.add(key)
                    for record in self._records_by_external_id.get(global_id.casefold(), []):
                        key = (str(record["object_id"]), "external_id")
                        if key not in seen:
                            candidates.append((record["object_id"], entity.id(), global_id, "external_id", 0.95))
                            seen.add(key)

            relationships: list[tuple[Any, ...]] = []
            for relationship in model.by_type("IfcRelationship"):
                relating: list[tuple[str, Any]] = []
                related: list[tuple[str, Any]] = []
                for name, value in relationship.get_info(recursive=False).items():
                    if name.startswith("Relating"):
                        relating.extend((name, item) for item in _ifc_entities(value))
                    elif name.startswith("Related"):
                        related.extend((name, item) for item in _ifc_entities(value))
                for relating_role, left in relating or [("", None)]:
                    for related_role, right in related or [("", None)]:
                        relationships.append((
                            relationship.id(),
                            relationship.is_a(),
                            left.id() if left is not None else None,
                            right.id() if right is not None else None,
                            relating_role or None,
                            related_role or None,
                        ))
            return {"objects": objects, "relationships": relationships, "candidates": candidates}

    def analyze_geometry(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return {
                "available": False,
                "reason": "IfcOpenShell geometry is unavailable for this IFC source format or installation.",
                "source_file": self.ifc_path.name,
            }

        selectors = self._selectors(arguments)
        if not any(selectors.values()):
            raise IfcAnalysisError(
                "Geometry analysis requires at least one record ID, IFC ID, GlobalId, entity type, or name term."
            )
        max_results = min(200, max(1, int(arguments.get("max_results", 50))))
        metrics = {str(item) for item in arguments.get("metrics", [])} or {
            "placement", "bounding_box", "solid_volume", "surface_area", "projected_area_xy",
        }

        with self._lock:
            model = self._load_model()
            selected = self._select_entities(model, selectors)
            try:
                from ifcopenshell.util.unit import calculate_unit_scale

                unit_scale = float(calculate_unit_scale(model))
            except Exception:
                unit_scale = 1.0
            entities = list(selected.values())[:max_results]
            rows = [self._geometry_row(entity, metrics, unit_scale) for entity in entities]
            return {
                "available": True,
                "engine": "IfcOpenShell",
                "units": {"length": "m", "area": "m2", "volume": "m3"},
                "selected_entities": len(selected),
                "returned_entities": len(rows),
                "truncated": len(selected) > max_results,
                "rows": rows,
                "source_file": self.ifc_path.name,
            }

    def geometry_inventory(self) -> list[dict[str, Any]]:
        """Build a reusable all-product mesh inventory for model-authored analysis."""
        if not self.available:
            return []
        with self._lock:
            if self._geometry_inventory is not None:
                return self._geometry_inventory
            import ifcopenshell.geom
            import numpy as np

            model = self._load_model()
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            workers = max(1, min(8, os.cpu_count() or 1))
            iterator = ifcopenshell.geom.iterator(settings, model, workers)
            rows: list[dict[str, Any]] = []
            if iterator.initialize():
                while True:
                    shape = iterator.get()
                    try:
                        entity = model.by_id(int(shape.id))
                        vertices = np.asarray(shape.geometry.verts, dtype=float).reshape((-1, 3))
                        faces = np.asarray(shape.geometry.faces, dtype=int).reshape((-1, 3))
                        if len(vertices) and len(faces):
                            row = self._identity_row(entity)
                            row.update(_mesh_metrics(vertices, faces, projected_area=True))
                            rows.append(row)
                    except Exception as exc:
                        rows.append({
                            "ifc_step_id": int(getattr(shape, "id", 0) or 0),
                            "geometry_error": f"{type(exc).__name__}: {exc}",
                        })
                    if not iterator.next():
                        break
            self._geometry_inventory = rows
            return rows

    def rank_geometry(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Filter and rank the complete geometry inventory by a model-selected metric."""
        if not self.available:
            return {
                "available": False,
                "reason": "IfcOpenShell geometry is unavailable for this IFC source format or installation.",
                "source_file": self.ifc_path.name,
            }
        selectors = self._selectors(arguments)
        if not any(selectors.values()):
            raise IfcAnalysisError(
                "Ranked geometry requires at least one record ID, IFC ID, GlobalId, entity type, or name term."
            )
        metric = str(arguments.get("metric") or "solid_volume")
        metric_paths = {
            "solid_volume": ("solid_volume_m3",),
            "surface_area": ("surface_area_m2",),
            "projected_area_xy": ("projected_area_xy_m2",),
            "bounding_box_volume": ("bounding_box", "volume_m3"),
            "length_x": ("bounding_box", "size", "x"),
            "length_y": ("bounding_box", "size", "y"),
            "length_z": ("bounding_box", "size", "z"),
            "max_dimension": ("bounding_box", "size"),
        }
        if metric not in metric_paths:
            raise IfcAnalysisError(f"Unsupported geometry ranking metric: {metric}")
        descending = str(arguments.get("order") or "descending") == "descending"
        max_results = min(200, max(1, int(arguments.get("max_results", 20))))

        matching = [row for row in self.geometry_inventory() if self._inventory_matches(row, selectors)]
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in matching:
            value = self._metric_value(row, metric_paths[metric])
            if value is not None and math.isfinite(value):
                ranked.append((value, row))
        ranked.sort(key=lambda item: (item[0], int(item[1].get("ifc_step_id") or 0)), reverse=descending)
        rows = []
        for rank, (value, source) in enumerate(ranked[:max_results], start=1):
            row = dict(source)
            row.update({"rank": rank, "ranking_metric": metric, "ranking_value": value})
            rows.append(row)
        return {
            "available": True,
            "engine": "IfcOpenShell",
            "metric": metric,
            "order": "descending" if descending else "ascending",
            "matching_entities": len(matching),
            "rankable_entities": len(ranked),
            "returned_entities": len(rows),
            "truncated": len(ranked) > max_results,
            "rows": rows,
            "source_file": self.ifc_path.name,
        }

    @staticmethod
    def _selectors(arguments: dict[str, Any]) -> dict[str, list[Any]]:
        return {
            "record_object_ids": [str(item) for item in arguments.get("record_object_ids", [])],
            "ifc_step_ids": [int(item) for item in arguments.get("ifc_step_ids", [])],
            "global_ids": [str(item) for item in arguments.get("global_ids", [])],
            "entity_types": [str(item) for item in arguments.get("entity_types", [])],
            "name_terms": [str(item) for item in arguments.get("name_terms", []) if str(item).strip()],
        }

    def _select_entities(self, model: Any, selectors: dict[str, list[Any]]) -> dict[int, Any]:
        selected: dict[int, Any] = {}
        for object_id in selectors["record_object_ids"]:
            record = self._record_by_id.get(object_id)
            if not record:
                continue
            guid = record["properties"].get("IFC Parameters.IfcGUID")
            if guid:
                entity = _by_guid(model, str(guid))
                if entity is not None:
                    selected[entity.id()] = entity
        for step_id in selectors["ifc_step_ids"]:
            entity = model.by_id(step_id)
            if entity is not None:
                selected[entity.id()] = entity
        for global_id in selectors["global_ids"]:
            entity = _by_guid(model, global_id)
            if entity is not None:
                selected[entity.id()] = entity
        for entity_type in selectors["entity_types"]:
            try:
                for entity in model.by_type(entity_type):
                    selected[entity.id()] = entity
            except RuntimeError:
                continue
        if selectors["name_terms"]:
            normalized_terms = [term.casefold() for term in selectors["name_terms"]]
            for entity in model.by_type("IfcObject"):
                text = " | ".join(filter(None, [
                    _ifc_text(entity, "Name"), _ifc_text(entity, "Description"),
                    _ifc_text(entity, "ObjectType"), _ifc_text(entity, "Tag"), entity.is_a(),
                ])).casefold()
                if any(term in text for term in normalized_terms):
                    selected[entity.id()] = entity
        return selected

    def _inventory_matches(self, row: dict[str, Any], selectors: dict[str, list[Any]]) -> bool:
        if row.get("ifc_step_id") in selectors["ifc_step_ids"]:
            return True
        if str(row.get("global_id") or "") in selectors["global_ids"]:
            return True
        if str(row.get("entity_type") or "").casefold() in {
            item.casefold() for item in selectors["entity_types"]
        }:
            return True
        if set(str(item) for item in row.get("record_object_ids", [])).intersection(selectors["record_object_ids"]):
            return True
        searchable = " | ".join(str(row.get(key) or "") for key in ("name", "object_type", "tag", "entity_type")).casefold()
        return any(term.casefold() in searchable for term in selectors["name_terms"])

    @staticmethod
    def _metric_value(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        if isinstance(value, dict):
            numeric = [float(item) for item in value.values() if isinstance(item, (int, float))]
            return max(numeric) if numeric else None
        return float(value) if isinstance(value, (int, float)) else None

    def _load_model(self) -> Any:
        if self._model is None:
            import ifcopenshell

            self._model = ifcopenshell.open(str(self.ifc_path))
        return self._model

    def _geometry_row(self, entity: Any, metrics: set[str], unit_scale: float) -> dict[str, Any]:
        row = self._identity_row(entity)
        global_id = row["global_id"]
        row.update({
            "ifc_step_id": entity.id(),
            "record_object_ids": [
                item["object_id"] for item in self._records_by_guid.get(global_id.casefold(), [])
            ] if global_id else [],
        })
        if "placement" in metrics:
            try:
                from ifcopenshell.util.placement import get_local_placement

                matrix = get_local_placement(getattr(entity, "ObjectPlacement", None))
                row["world_origin"] = {
                    "x": float(matrix[0][3]) * unit_scale,
                    "y": float(matrix[1][3]) * unit_scale,
                    "z": float(matrix[2][3]) * unit_scale,
                }
            except Exception as exc:
                row["placement_error"] = f"{type(exc).__name__}: {exc}"

        geometry_metrics = metrics.intersection({
            "bounding_box", "solid_volume", "surface_area", "projected_area_xy",
        })
        if not geometry_metrics:
            return row
        try:
            import ifcopenshell.geom
            import numpy as np

            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            shape = ifcopenshell.geom.create_shape(settings, entity)
            vertices = np.asarray(shape.geometry.verts, dtype=float).reshape((-1, 3))
            faces = np.asarray(shape.geometry.faces, dtype=int).reshape((-1, 3))
            if not len(vertices) or not len(faces):
                raise ValueError("geometry contains no triangulated faces")
            available = _mesh_metrics(
                vertices,
                faces,
                projected_area="projected_area_xy" in metrics,
            )
            if "bounding_box" in metrics:
                row["bounding_box"] = available["bounding_box"]
            if "surface_area" in metrics:
                row["surface_area_m2"] = available["surface_area_m2"]
            if "solid_volume" in metrics:
                row["solid_volume_m3"] = available["solid_volume_m3"]
            if "projected_area_xy" in metrics:
                row["projected_area_xy_m2"] = available["projected_area_xy_m2"]
            row["geometry_vertices"] = available["geometry_vertices"]
            row["geometry_triangles"] = available["geometry_triangles"]
        except Exception as exc:
            row["geometry_error"] = f"{type(exc).__name__}: {exc}"
        return row

    def _identity_row(self, entity: Any) -> dict[str, Any]:
        global_id = _ifc_text(entity, "GlobalId")
        container = _ifc_container(entity)
        return {
            "ifc_step_id": entity.id(),
            "global_id": global_id,
            "entity_type": entity.is_a(),
            "name": _ifc_text(entity, "Name"),
            "object_type": _ifc_text(entity, "ObjectType"),
            "tag": _ifc_text(entity, "Tag"),
            "container_step_id": container.id() if container is not None else None,
            "container_type": container.is_a() if container is not None else None,
            "container_name": _ifc_text(container, "Name"),
            "record_object_ids": [
                item["object_id"] for item in self._records_by_guid.get(global_id.casefold(), [])
            ] if global_id else [],
        }


def _ifc_text(entity: Any | None, attribute: str) -> str | None:
    if entity is None:
        return None
    try:
        value = getattr(entity, attribute, None)
    except Exception:
        return None
    return str(value) if value not in (None, "") else None


def _ifc_entities(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "id") and callable(value.id):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if hasattr(item, "id") and callable(item.id)]
    return []


def _ifc_container(entity: Any) -> Any | None:
    try:
        from ifcopenshell.util.element import get_container

        return get_container(entity)
    except Exception:
        return None


def _by_guid(model: Any, global_id: str) -> Any | None:
    try:
        return model.by_guid(global_id)
    except Exception:
        return None


def _mesh_metrics(vertices: Any, faces: Any, *, projected_area: bool) -> dict[str, Any]:
    import numpy as np

    triangles = vertices[faces]
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    size = maximum - minimum
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    # Translation avoids cancellation for models stored in large survey coordinates.
    volume_triangles = triangles - vertices[0]
    signed = np.einsum(
        "ij,ij->i",
        volume_triangles[:, 0],
        np.cross(volume_triangles[:, 1], volume_triangles[:, 2]),
    )
    output = {
        "bounding_box": {
            "min": {"x": float(minimum[0]), "y": float(minimum[1]), "z": float(minimum[2])},
            "max": {"x": float(maximum[0]), "y": float(maximum[1]), "z": float(maximum[2])},
            "size": {"x": float(size[0]), "y": float(size[1]), "z": float(size[2])},
            "volume_m3": float(size[0] * size[1] * size[2]),
        },
        "solid_volume_m3": float(abs(signed.sum()) / 6.0),
        "surface_area_m2": float(np.linalg.norm(cross, axis=1).sum() / 2.0),
        "geometry_vertices": int(len(vertices)),
        "geometry_triangles": int(len(faces)),
    }
    if projected_area:
        output["projected_area_xy_m2"] = _projected_area_xy(triangles)
    return output


def _projected_area_xy(triangles: Any) -> float | None:
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polygons = []
        origin_x = float(triangles[0][0][0])
        origin_y = float(triangles[0][0][1])
        for triangle in triangles:
            polygon = Polygon([
                (float(point[0]) - origin_x, float(point[1]) - origin_y)
                for point in triangle
            ])
            if polygon.is_valid and polygon.area > 1e-12:
                polygons.append(polygon)
        if not polygons:
            return 0.0
        area = float(unary_union(polygons).area)
        return area if math.isfinite(area) else None
    except Exception:
        return None
