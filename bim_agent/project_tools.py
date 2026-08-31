from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import operator
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ifc_analysis import IfcAnalysisEngine


class ProjectError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectFiles:
    tree: Path
    properties: Path
    ifc: Path


class RawProjectTools:
    """Generic, read-only data tools. The model owns all semantic decisions."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).resolve()
        self.files = self._discover(self.data_dir)
        self.tree = self._read_json(self.files.tree)
        raw_properties = self._read_json(self.files.properties)
        if isinstance(raw_properties, dict) and isinstance(raw_properties.get("data"), dict):
            raw_properties = raw_properties["data"]
        if not isinstance(raw_properties, dict):
            raise ProjectError("Properties JSON must be an object keyed by object ID.")
        self.properties = {
            str(key): value for key, value in raw_properties.items() if isinstance(value, dict)
        }
        self.paths: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = {}
        self.names: dict[str, str] = {}
        self._index_tree()
        self.records = [self._record(key, value) for key, value in self.properties.items()]
        self.by_id = {str(item["object_id"]): item for item in self.records}
        self._ifc_lines: list[str] | None = None
        self._workspace_connection: sqlite3.Connection | None = None
        self._workspace_lock = threading.RLock()
        self._workspace_ifc_attached = False
        self.ifc_analysis = IfcAnalysisEngine(self.files.ifc, self.records)
        self._cursor_key = hashlib.sha256(
            "|".join(item["sha256"] for item in self.manifest()).encode("ascii")
        ).digest()

    @staticmethod
    def _discover(data_dir: Path) -> ProjectFiles:
        if not data_dir.is_dir():
            raise ProjectError(f"Data directory does not exist: {data_dir}")
        trees = sorted(data_dir.glob("*-tree.json"))
        properties = sorted(data_dir.glob("*-properties.json"))
        ifcs = sorted(data_dir.glob("*.ifc"))
        if len(trees) != 1 or len(properties) != 1 or len(ifcs) != 1:
            raise ProjectError(
                "Expected exactly one *-tree.json, one *-properties.json, and one *.ifc; "
                f"found {len(trees)}, {len(properties)}, and {len(ifcs)}."
            )
        if trees[0].name.removesuffix("-tree.json") != properties[0].name.removesuffix("-properties.json"):
            raise ProjectError("Tree and properties JSON files do not share a model identifier.")
        return ProjectFiles(trees[0], properties[0], ifcs[0])

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Cannot parse {path.name}: {exc}") from exc

    def _index_tree(self) -> None:
        body = self.tree.get("data", self.tree) if isinstance(self.tree, dict) else {}
        roots = body.get("objects", []) if isinstance(body, dict) else []

        def visit(node: dict[str, Any], parent_path: list[str]) -> None:
            object_id = str(node.get("objectid", ""))
            if not object_id:
                return
            name = str(node.get("name") or f"object-{object_id}")
            path = [*parent_path, name]
            self.paths[object_id] = path
            self.names[object_id] = name
            raw_children = node.get("objects") if isinstance(node.get("objects"), list) else []
            self.children[object_id] = [str(child.get("objectid")) for child in raw_children]
            for child in raw_children:
                if isinstance(child, dict):
                    visit(child, path)

        for root in roots if isinstance(roots, list) else []:
            if isinstance(root, dict):
                visit(root, [])

    def _record(self, key: str, raw: dict[str, Any]) -> dict[str, Any]:
        object_id = str(raw.get("objectid", key))
        return {
            "object_id": object_id,
            "name": str(raw.get("name") or self.names.get(object_id) or f"object-{object_id}"),
            "external_id": raw.get("externalId"),
            "path": self.paths.get(object_id, []),
            "properties": _flatten(raw.get("properties", {})),
        }

    def manifest(self) -> list[dict[str, Any]]:
        output = []
        for kind, path in (("tree", self.files.tree), ("properties", self.files.properties), ("ifc", self.files.ifc)):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            output.append({
                "kind": kind,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
        return output

    def definitions(self) -> list[dict[str, Any]]:
        return [
            _tool("inspect_project", "Inspect source files, tree roots, record count, and available property keys.", {}),
            _tool("list_tree_children", "List a page of direct child nodes for a tree object ID. Use null parent_object_id for roots and fetch_more when cursor is non-null.", {
                "parent_object_id": {"type": ["string", "null"]},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            }),
            _tool("search_records", "Search a page of raw record names, hierarchy paths, property keys, and property values. Use fetch_more when cursor is non-null.", {
                "terms": {"type": "array", "items": {"type": "string"}},
                "match": {"type": "string", "enum": ["all", "any"]},
                "path_terms": {"type": "array", "items": {"type": "string"}},
                "property_keys": {"type": "array", "items": {"type": "string"}},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            }, required=["terms"]),
            _tool("get_records", "Get raw hierarchy and properties for exact object IDs returned by another tool.", {
                "object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 200},
            }, required=["object_ids"]),
            _tool("aggregate_records", "Calculate count, distinct count, sum, average, minimum, or maximum for exact object IDs. Can group by a raw property key, name, external_id, or path_N.", {
                "object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2000},
                "operation": {"type": "string", "enum": ["count", "distinct_count", "sum", "average", "min", "max"]},
                "field": {"type": ["string", "null"]},
                "distinct_by": {"type": ["string", "null"]},
                "group_by": {"type": ["string", "null"]},
                "output_unit": {"type": ["string", "null"]},
            }, required=["object_ids", "operation"]),
            _tool("search_ifc", "Search a page of raw IFC text lines for entity names, GUIDs, relationships, placements, systems, or values. Use fetch_more when cursor is non-null.", {
                "terms": {"type": "array", "items": {"type": "string"}},
                "match": {"type": "string", "enum": ["all", "any"]},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            }, required=["terms"]),
            _tool("fetch_more", "Fetch the next page represented by an opaque cursor returned by list_tree_children, search_records, or search_ifc.", {
                "cursor": {"type": "string"},
            }, required=["cursor"]),
            _tool("calculate", "Evaluate arithmetic using numbers, parentheses, +, -, *, /, %, and powers. Use this instead of mental arithmetic.", {
                "expression": {"type": "string"},
            }, required=["expression"]),
            _tool(
                "describe_bim_workspace",
                "Return the schema and capabilities of the model-authored read-only BIM SQL workspace. Call once before query_bim_workspace when its schema is not already known.",
                {},
                strict=True,
            ),
            _tool(
                "query_bim_workspace",
                "Execute one model-authored, read-only SQLite SELECT or WITH query over records, properties, hierarchy, named IFC objects/relationships, identity candidates, and raw IFC entities. Use compact aggregates plus representative IDs. Returns columns, rows, truncation, elapsed time, and source files; errors explain invalid SQL or safety/time limits.",
                {
                    "sql": {"type": "string"},
                    "parameters": {
                        "type": "array",
                        "items": {"type": ["string", "number", "integer", "boolean", "null"]},
                        "maxItems": 100,
                    },
                    "row_limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=["sql", "parameters", "row_limit"],
                strict=True,
            ),
            _tool(
                "analyze_ifc_geometry",
                "Run model-selected IfcOpenShell analysis for exact record IDs, IFC STEP IDs, GlobalIds, IFC entity types, or name terms. Computes world placement, triangulated bounding box, solid volume, surface area, projected XY area, spatial container, and identity matches. Select the metrics needed and compare geometry-derived values with properties before concluding.",
                {
                    "record_object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 200},
                    "ifc_step_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 200},
                    "global_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 200},
                    "entity_types": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
                    "name_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["placement", "bounding_box", "solid_volume", "surface_area", "projected_area_xy"],
                        },
                        "maxItems": 5,
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                required=[
                    "record_object_ids", "ifc_step_ids", "global_ids", "entity_types",
                    "name_terms", "metrics", "max_results",
                ],
                strict=True,
            ),
            _tool(
                "rank_ifc_geometry",
                "Filter the complete IFC mesh inventory using model-chosen identity/type/name selectors, then rank it by one physical metric. Use this for largest/smallest/longest comparisons so selection order or an early row limit cannot decide the answer.",
                {
                    "record_object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2000},
                    "ifc_step_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 2000},
                    "global_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2000},
                    "entity_types": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
                    "name_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
                    "metric": {
                        "type": "string",
                        "enum": [
                            "solid_volume", "surface_area", "projected_area_xy", "bounding_box_volume",
                            "length_x", "length_y", "length_z", "max_dimension",
                        ],
                    },
                    "order": {"type": "string", "enum": ["ascending", "descending"]},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                strict=True,
            ),
            _tool(
                "reconcile_populations",
                "Reconcile model-selected object populations without deciding their semantics. Reports raw/unique/found record counts, identity coverage, union size, duplicates, missing IDs, and pairwise overlaps. Use before finalizing material counts or subgroup totals.",
                {
                    "populations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 30,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 5000},
                            },
                            "required": ["label", "object_ids"],
                            "additionalProperties": False,
                        },
                    },
                },
                strict=True,
            ),
            _tool(
                "analyze_ifc_graph",
                "Traverse named IFC relationships between model-selected STEP IDs. Relationship types and direction are explicit model choices; results retain relationship type, roles, and shortest evidence paths. Do not interpret mere reference proximity as physical connectivity.",
                {
                    "start_step_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 500},
                    "target_step_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 2000},
                    "relationship_types": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
                    "direction": {"type": "string", "enum": ["forward", "reverse", "undirected"]},
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": 20},
                    "max_paths": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                strict=True,
            ),
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_project":
            keys = Counter(key for record in self.records for key in record["properties"])
            roots = [
                {"object_id": object_id, "name": self.names.get(object_id, ""), "child_count": len(children)}
                for object_id, children in self.children.items()
                if len(self.paths.get(object_id, [])) == 1
            ]
            return {
                "source_files": self.manifest(),
                "record_count": len(self.records),
                "tree_node_count": len(self.paths),
                "tree_roots": roots,
                "property_keys": [{"key": key, "record_count": count} for key, count in keys.most_common(300)],
                "ifc_source": self._ifc_inventory(),
            }
        if name == "list_tree_children":
            parent = arguments.get("parent_object_id")
            if parent is None:
                ids = [key for key, path in self.paths.items() if len(path) == 1]
            else:
                ids = self.children.get(str(parent), [])
            results = [
                {
                    "object_id": item,
                    "name": self.names.get(item, ""),
                    "path": self.paths.get(item, []),
                    "child_count": len(self.children.get(item, [])),
                }
                for item in ids
            ]
            page = self._page("list_tree_children", arguments, results)
            return {**page, "children": page["results"]}
        if name == "search_records":
            return self.search_records(arguments)
        if name == "get_records":
            return {"records": [self.by_id[item] for item in map(str, arguments.get("object_ids", [])) if item in self.by_id]}
        if name == "aggregate_records":
            return self.aggregate(arguments)
        if name == "search_ifc":
            return self.search_ifc(arguments)
        if name == "fetch_more":
            return self.fetch_more(str(arguments.get("cursor", "")))
        if name == "calculate":
            return {"expression": str(arguments.get("expression", "")), "value": _safe_calculate(str(arguments.get("expression", "")))}
        if name == "describe_bim_workspace":
            return self.analysis_workspace({"action": "describe"})
        if name == "query_bim_workspace":
            return self.analysis_workspace({"action": "query", **arguments})
        if name == "analyze_ifc_geometry":
            return self.ifc_analysis.analyze_geometry(arguments)
        if name == "rank_ifc_geometry":
            return self.ifc_analysis.rank_geometry(arguments)
        if name == "reconcile_populations":
            return self.reconcile_populations(arguments)
        if name == "analyze_ifc_graph":
            return self.analyze_ifc_graph(arguments)
        if name == "bim_analysis_workspace":
            return self.analysis_workspace(arguments)
        raise ProjectError(f"Unknown tool: {name}")

    def fetch_more(self, cursor: str) -> dict[str, Any]:
        payload = self._decode_cursor(cursor)
        name = str(payload.get("tool", ""))
        if name not in {"list_tree_children", "search_records", "search_ifc"}:
            raise ProjectError("Cursor does not identify a pageable project tool.")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ProjectError("Cursor arguments are invalid.")
        return self.execute(name, arguments)

    def _page(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        results: list[dict[str, Any]],
        *,
        default_limit: int = 100,
    ) -> dict[str, Any]:
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(200, max(1, int(arguments.get("limit", default_limit))))
        page_results = results[offset:offset + limit]
        next_offset = offset + len(page_results)
        cursor = None
        if next_offset < len(results):
            next_arguments = {
                key: value for key, value in arguments.items() if key not in {"offset", "cursor"}
            }
            next_arguments.update({"offset": next_offset, "limit": limit})
            cursor = self._encode_cursor({"tool": tool_name, "arguments": next_arguments})
        return {
            "results": page_results,
            "total_count": len(results),
            "returned_count": len(page_results),
            "cursor": cursor,
            "offset": offset,
        }

    def _encode_cursor(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hashlib.sha256(self._cursor_key + raw).hexdigest()[:24].encode("ascii")
        return base64.urlsafe_b64encode(signature + b"." + raw).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor: str) -> dict[str, Any]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            signed = base64.urlsafe_b64decode(padded.encode("ascii"))
            signature, raw = signed.split(b".", 1)
            expected = hashlib.sha256(self._cursor_key + raw).hexdigest()[:24].encode("ascii")
            if signature != expected:
                raise ValueError("signature mismatch")
            payload = json.loads(raw)
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectError("Cursor is invalid or belongs to different project data.") from exc
        if not isinstance(payload, dict):
            raise ProjectError("Cursor payload is invalid.")
        return payload

    def reconcile_populations(self, arguments: dict[str, Any]) -> dict[str, Any]:
        populations = arguments.get("populations") or []
        connection = self._workspace_database()
        candidate_rows = connection.execute(
            "SELECT record_object_id, ifc_step_id, global_id FROM record_ifc_candidates"
        ).fetchall() if self.ifc_analysis.available else []
        candidates: dict[str, set[tuple[int, str | None]]] = defaultdict(set)
        for object_id, step_id, global_id in candidate_rows:
            candidates[str(object_id)].add((int(step_id), str(global_id) if global_id else None))

        output = []
        sets: list[tuple[str, set[str]]] = []
        for population in populations:
            label = str(population.get("label") or "population")
            raw_ids = [str(item) for item in population.get("object_ids", [])]
            unique_ids = set(raw_ids)
            found = unique_ids.intersection(self.by_id)
            missing = sorted(unique_ids.difference(self.by_id))
            external_ids = {
                str(self.by_id[item]["external_id"])
                for item in found if self.by_id[item].get("external_id")
            }
            guid_values = {
                str(self.by_id[item]["properties"].get("IFC Parameters.IfcGUID"))
                for item in found if self.by_id[item]["properties"].get("IFC Parameters.IfcGUID")
            }
            mapped_steps = {step for item in found for step, _ in candidates.get(item, set())}
            output.append({
                "label": label,
                "input_count": len(raw_ids),
                "unique_object_ids": len(unique_ids),
                "duplicate_occurrences": len(raw_ids) - len(unique_ids),
                "found_records": len(found),
                "missing_object_ids": missing[:100],
                "missing_truncated": len(missing) > 100,
                "distinct_external_ids": len(external_ids),
                "distinct_ifc_guids_in_properties": len(guid_values),
                "mapped_ifc_step_ids": len(mapped_steps),
            })
            sets.append((label, unique_ids))
        union = set().union(*(ids for _, ids in sets)) if sets else set()
        overlaps = []
        for index, (left_label, left_ids) in enumerate(sets):
            for right_label, right_ids in sets[index + 1:]:
                shared = left_ids.intersection(right_ids)
                if shared:
                    overlaps.append({
                        "left": left_label,
                        "right": right_label,
                        "count": len(shared),
                        "sample_object_ids": sorted(shared)[:20],
                    })
        return {
            "populations": output,
            "population_count": len(output),
            "union_unique_object_ids": len(union),
            "union_found_records": len(union.intersection(self.by_id)),
            "pairwise_overlaps": overlaps,
        }

    def analyze_ifc_graph(self, arguments: dict[str, Any]) -> dict[str, Any]:
        connection = self._workspace_database()
        try:
            raw_relationships = connection.execute(
                "SELECT relationship_step_id, relationship_type, relating_step_id, related_step_id, "
                "relating_role, related_role FROM ifc_relationships "
                "WHERE relating_step_id IS NOT NULL AND related_step_id IS NOT NULL"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ProjectError("Named IFC relationship graph is unavailable for this IFC source.") from exc
        allowed = {str(item).casefold() for item in arguments.get("relationship_types", [])}
        direction = str(arguments.get("direction") or "undirected")
        adjacency: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        included_relationships = 0
        for rel_id, rel_type, left, right, left_role, right_role in raw_relationships:
            if allowed and str(rel_type).casefold() not in allowed:
                continue
            included_relationships += 1
            edge = {
                "relationship_step_id": int(rel_id),
                "relationship_type": str(rel_type),
                "relating_role": left_role,
                "related_role": right_role,
            }
            if direction in {"forward", "undirected"}:
                adjacency[int(left)].append((int(right), {**edge, "traversal": "relating_to_related"}))
            if direction in {"reverse", "undirected"}:
                adjacency[int(right)].append((int(left), {**edge, "traversal": "related_to_relating"}))

        starts = [int(item) for item in arguments.get("start_step_ids", [])]
        targets = {int(item) for item in arguments.get("target_step_ids", [])}
        max_depth = min(20, max(1, int(arguments.get("max_depth", 6))))
        max_paths = min(500, max(1, int(arguments.get("max_paths", 100))))
        paths = []
        reached_nodes: set[int] = set()
        for start in starts:
            queue: list[tuple[int, list[dict[str, Any]]]] = [(start, [])]
            visited = {start}
            while queue and len(paths) < max_paths:
                node, path = queue.pop(0)
                if path and (not targets or node in targets):
                    paths.append({"start_step_id": start, "target_step_id": node, "depth": len(path), "edges": path})
                    if targets and targets.issubset({item["target_step_id"] for item in paths if item["start_step_id"] == start}):
                        break
                if len(path) >= max_depth:
                    continue
                for neighbor, edge in adjacency.get(node, []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    reached_nodes.add(neighbor)
                    queue.append((neighbor, [*path, {"from_step_id": node, "to_step_id": neighbor, **edge}]))
        evidence_ids = set(starts)
        for item in paths:
            evidence_ids.add(item["target_step_id"])
            for edge in item["edges"]:
                evidence_ids.update((edge["from_step_id"], edge["to_step_id"]))
        node_details = []
        try:
            for step_id, global_id, entity_type, name, object_type, container_name in connection.execute(
                "SELECT step_id, global_id, entity_type, name, object_type, container_name FROM ifc_objects"
            ):
                if int(step_id) in evidence_ids:
                    node_details.append({
                        "ifc_step_id": int(step_id), "global_id": global_id, "entity_type": entity_type,
                        "name": name, "object_type": object_type, "container_name": container_name,
                    })
        except sqlite3.Error:
            pass
        return {
            "direction": direction,
            "relationship_types": sorted(allowed),
            "included_relationships": included_relationships,
            "start_count": len(starts),
            "target_count": len(targets),
            "reached_nodes": len(reached_nodes),
            "paths": paths,
            "node_details": node_details,
            "returned_paths": len(paths),
            "truncated": len(paths) >= max_paths,
        }

    def analysis_workspace(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute model-written analysis without granting host filesystem or mutation access."""
        action = str(arguments.get("action", "describe")).strip().lower()
        if action not in {"describe", "query"}:
            raise ProjectError("Workspace action must be 'describe' or 'query'.")

        with self._workspace_lock:
            connection = self._workspace_database()
            if action == "describe":
                return self._describe_workspace(connection)

            sql = str(arguments.get("sql") or "").strip()
            if not sql:
                raise ProjectError("Workspace query action requires SQL.")
            if len(sql) > 20_000:
                raise ProjectError("Workspace SQL is too long (maximum 20,000 characters).")
            if not re.match(r"(?is)^\s*(SELECT\b|WITH\b|EXPLAIN\s+QUERY\s+PLAN\b)", sql):
                raise ProjectError("Workspace accepts only a single read-only SELECT, WITH, or EXPLAIN QUERY PLAN statement.")
            parameters = arguments.get("parameters", [])
            if not isinstance(parameters, list):
                raise ProjectError("Workspace parameters must be an array.")
            row_limit = min(500, max(1, int(arguments.get("row_limit", 100))))

            started = time.monotonic()
            deadline = started + 3.0
            connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2_000)
            try:
                cursor = connection.execute(sql, parameters)
                if cursor.description is None:
                    raise ProjectError("Workspace query did not return rows.")
                columns = [str(item[0]) for item in cursor.description]
                raw_rows = cursor.fetchmany(row_limit + 1)
            except sqlite3.Error as exc:
                message = "Query exceeded the 3 second workspace limit." if "interrupted" in str(exc).lower() else str(exc)
                raise ProjectError(f"Workspace query failed: {message}") from exc
            finally:
                connection.set_progress_handler(None, 0)

            truncated = len(raw_rows) > row_limit
            rows = [
                {column: _sql_json_value(value) for column, value in zip(columns, row)}
                for row in raw_rows[:row_limit]
            ]
            return {
                "columns": columns,
                "rows": rows,
                "returned_rows": len(rows),
                "truncated": truncated,
                "row_limit": row_limit,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                "source_files": self._workspace_sources(sql),
            }

    def _workspace_database(self) -> sqlite3.Connection:
        if self._workspace_connection is not None:
            return self._workspace_connection

        connection = sqlite3.connect(":memory:", check_same_thread=False, uri=True)
        try:
            connection.executescript(
                """
                CREATE TABLE source_files(
                    source_kind TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    sha256 TEXT NOT NULL
                );
                CREATE TABLE records(
                    object_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    external_id TEXT,
                    path_text TEXT NOT NULL,
                    path_json TEXT NOT NULL,
                    path_depth INTEGER NOT NULL,
                    properties_json TEXT NOT NULL
                );
                CREATE TABLE properties(
                    object_id TEXT NOT NULL,
                    property_key TEXT NOT NULL,
                    property_value TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    number_value REAL,
                    unit TEXT
                );
                CREATE TABLE tree_nodes(
                    object_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_object_id TEXT,
                    path_text TEXT NOT NULL,
                    path_depth INTEGER NOT NULL,
                    child_count INTEGER NOT NULL
                );
                CREATE TABLE tree_edges(
                    parent_object_id TEXT NOT NULL,
                    child_object_id TEXT NOT NULL,
                    PRIMARY KEY(parent_object_id, child_object_id)
                );
                """
            )
            connection.executemany(
                "INSERT INTO source_files VALUES (?, ?, ?, ?, ?)",
                [
                    (item["kind"], Path(item["path"]).name, item["path"], item["bytes"], item["sha256"])
                    for item in self.manifest()
                ],
            )
            connection.executemany(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record["object_id"],
                        record["name"],
                        record["external_id"],
                        " > ".join(record["path"]),
                        json.dumps(record["path"], ensure_ascii=False),
                        len(record["path"]),
                        json.dumps(record["properties"], ensure_ascii=False),
                    )
                    for record in self.records
                ],
            )
            property_rows = []
            for record in self.records:
                for key, value in record["properties"].items():
                    number, unit = _number_unit(value)
                    property_rows.append((
                        record["object_id"], key, str(value), _normalize(key), _normalize(value), number, unit,
                    ))
            connection.executemany("INSERT INTO properties VALUES (?, ?, ?, ?, ?, ?, ?)", property_rows)

            parents = {
                child_id: parent_id
                for parent_id, children in self.children.items()
                for child_id in children
            }
            connection.executemany(
                "INSERT INTO tree_nodes VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        object_id,
                        self.names.get(object_id, ""),
                        parents.get(object_id),
                        " > ".join(self.paths.get(object_id, [])),
                        len(self.paths.get(object_id, [])),
                        len(children),
                    )
                    for object_id, children in self.children.items()
                ],
            )
            connection.executemany(
                "INSERT INTO tree_edges VALUES (?, ?)",
                [
                    (parent_id, child_id)
                    for parent_id, children in self.children.items()
                    for child_id in children
                ],
            )
            connection.executescript(
                """
                CREATE INDEX properties_object_id_idx ON properties(object_id);
                CREATE INDEX properties_key_idx ON properties(normalized_key);
                CREATE INDEX properties_value_idx ON properties(normalized_value);
                CREATE INDEX tree_nodes_parent_idx ON tree_nodes(parent_object_id);
                """
            )
            if self.files.ifc.read_bytes()[:16] == b"SQLite format 3\x00":
                self._attach_ifc_sqlite(connection)
            else:
                self._index_step_ifc(connection)
            connection.create_function("normalize_text", 1, _normalize, deterministic=True)
            connection.create_function("parse_number", 1, lambda value: _number_unit(value)[0], deterministic=True)
            connection.create_function(
                "convert_length", 3,
                lambda number, source, target: _convert(float(number), str(source), str(target))[0]
                if number is not None and source is not None and target is not None else None,
                deterministic=True,
            )
            connection.commit()
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(_workspace_authorizer)
        except Exception:
            connection.close()
            raise
        self._workspace_connection = connection
        return connection

    def _attach_ifc_sqlite(self, connection: sqlite3.Connection) -> None:
        connection.execute("ATTACH DATABASE ? AS ifc_source", (self.files.ifc.as_uri() + "?mode=ro",))
        self._workspace_ifc_attached = True
        connection.execute(
            "CREATE TABLE ifc_sqlite_schema(table_name TEXT, column_name TEXT, declared_type TEXT, ordinal INTEGER)"
        )
        tables = [
            str(row[0]) for row in connection.execute(
                "SELECT name FROM ifc_source.sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        rows = []
        for table in tables:
            quoted = table.replace('"', '""')
            rows.extend(
                (table, str(row[1]), str(row[2] or ""), int(row[0]))
                for row in connection.execute(f'PRAGMA ifc_source.table_info("{quoted}")')
            )
        connection.executemany("INSERT INTO ifc_sqlite_schema VALUES (?, ?, ?, ?)", rows)

    def _index_step_ifc(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE ifc_entities(
                step_id INTEGER PRIMARY KEY,
                entity_type TEXT NOT NULL,
                arguments TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                source_line INTEGER NOT NULL
            );
            CREATE TABLE ifc_references(
                source_step_id INTEGER NOT NULL,
                target_step_id INTEGER NOT NULL
            );
            """
        )
        self._index_structured_ifc(connection)

    def _index_structured_ifc(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE ifc_objects(
                step_id INTEGER PRIMARY KEY,
                global_id TEXT,
                entity_type TEXT NOT NULL,
                name TEXT,
                description TEXT,
                object_type TEXT,
                tag TEXT,
                predefined_type TEXT,
                container_step_id INTEGER,
                container_global_id TEXT,
                container_type TEXT,
                container_name TEXT,
                placement_step_id INTEGER,
                representation_step_id INTEGER
            );
            CREATE TABLE ifc_relationships(
                relationship_step_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                relating_step_id INTEGER,
                related_step_id INTEGER,
                relating_role TEXT,
                related_role TEXT
            );
            CREATE TABLE record_ifc_candidates(
                record_object_id TEXT NOT NULL,
                ifc_step_id INTEGER NOT NULL,
                global_id TEXT,
                matching_signal TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            """
        )
        try:
            semantic = self.ifc_analysis.semantic_index()
        except Exception:
            # Raw STEP tables remain available even when a malformed/minimal fixture
            # cannot be interpreted as a schema-valid IFC model by IfcOpenShell.
            semantic = {"objects": [], "relationships": [], "candidates": []}
        connection.executemany("INSERT INTO ifc_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", semantic["objects"])
        connection.executemany("INSERT INTO ifc_relationships VALUES (?, ?, ?, ?, ?, ?)", semantic["relationships"])
        connection.executemany("INSERT INTO record_ifc_candidates VALUES (?, ?, ?, ?, ?)", semantic["candidates"])
        connection.executescript(
            """
            CREATE INDEX ifc_objects_global_id_idx ON ifc_objects(global_id);
            CREATE INDEX ifc_objects_type_idx ON ifc_objects(entity_type);
            CREATE INDEX ifc_objects_container_idx ON ifc_objects(container_step_id);
            CREATE INDEX ifc_relationships_relating_idx ON ifc_relationships(relating_step_id);
            CREATE INDEX ifc_relationships_related_idx ON ifc_relationships(related_step_id);
            CREATE INDEX record_ifc_candidates_record_idx ON record_ifc_candidates(record_object_id);
            CREATE INDEX record_ifc_candidates_ifc_idx ON record_ifc_candidates(ifc_step_id);
            """
        )
        entities: list[tuple[int, str, str, str, int]] = []
        references: list[tuple[int, int]] = []
        for step_id, entity_type, arguments, raw_text, line_number in _iter_ifc_entities(self.files.ifc):
            entities.append((step_id, entity_type, arguments, raw_text, line_number))
            references.extend((step_id, int(target)) for target in re.findall(r"#(\d+)", arguments))
            if len(entities) >= 2_000:
                connection.executemany("INSERT OR REPLACE INTO ifc_entities VALUES (?, ?, ?, ?, ?)", entities)
                connection.executemany("INSERT INTO ifc_references VALUES (?, ?)", references)
                entities.clear()
                references.clear()
        if entities:
            connection.executemany("INSERT OR REPLACE INTO ifc_entities VALUES (?, ?, ?, ?, ?)", entities)
            connection.executemany("INSERT INTO ifc_references VALUES (?, ?)", references)
        connection.executescript(
            """
            CREATE INDEX ifc_entities_type_idx ON ifc_entities(entity_type);
            CREATE INDEX ifc_references_source_idx ON ifc_references(source_step_id);
            CREATE INDEX ifc_references_target_idx ON ifc_references(target_step_id);
            """
        )

    def _describe_workspace(self, connection: sqlite3.Connection) -> dict[str, Any]:
        tables = [
            {"name": row[0], "definition": row[1]}
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        attached_ifc_tables = []
        if self._workspace_ifc_attached:
            attached_ifc_tables = [
                row[0] for row in connection.execute(
                    "SELECT name FROM ifc_source.sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
        return {
            "engine": "SQLite",
            "read_only": True,
            "tables": tables,
            "attached_ifc_schema": "ifc_source" if attached_ifc_tables else None,
            "attached_ifc_tables": attached_ifc_tables,
            "functions": [
                "normalize_text(value)", "parse_number(value)",
                "convert_length(number, source_unit, target_unit)",
            ],
            "guidance": [
                "Use records for identity/path and properties for one-row-per-property analysis.",
                "Use recursive CTEs over tree_nodes or tree_edges for hierarchy traversal.",
                "Do not assume a fixed path depth identifies instances; inspect tree_nodes.child_count, external_id, and alternative scopes.",
                "For STEP IFC, prefer named ifc_objects, ifc_relationships, and record_ifc_candidates; use ifc_entities plus ifc_references only for raw details.",
                "Use analyze_ifc_geometry when placement, area, volume, physical size, or spatial containment matters.",
                "Use parameter placeholders (?) for user-originated text and return aggregated or evidence-sized rows.",
                "In a compound UNION query, SQLite ORDER BY expressions must match output columns; wrap the UNION in an outer SELECT before ordering by a CASE expression.",
            ],
            "ifc_geometry": self.ifc_analysis.capability(),
        }

    def _workspace_sources(self, sql: str) -> list[str]:
        normalized = sql.casefold()
        kinds = []
        if any(token in normalized for token in ("records", "properties")):
            kinds.append(self.files.properties.name)
        if any(token in normalized for token in ("tree_nodes", "tree_edges")):
            kinds.append(self.files.tree.name)
        if "ifc_" in normalized:
            kinds.append(self.files.ifc.name)
        return kinds or [item.name for item in (self.files.tree, self.files.properties, self.files.ifc)]

    def scope_profile(
        self,
        max_nodes: int = 80,
        *,
        sample_limit: int = 5,
        terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compact raw hierarchy evidence for planning and evidence review."""
        max_nodes = max(0, int(max_nodes))
        sample_limit = max(0, min(10, int(sample_limit)))
        parents = {
            child_id: parent_id
            for parent_id, children in self.children.items()
            for child_id in children
        }

        def descendants(root_id: str) -> list[str]:
            pending = list(self.children.get(root_id, []))
            output = []
            while pending:
                item = pending.pop()
                output.append(item)
                pending.extend(self.children.get(item, []))
            return output

        nodes = []
        for object_id, children in self.children.items():
            if not children:
                continue
            nested = descendants(object_id)
            leaves = [item for item in nested if not self.children.get(item)]
            nodes.append({
                "object_id": object_id,
                "name": self.names.get(object_id, ""),
                "parent_object_id": parents.get(object_id),
                "path": self.paths.get(object_id, []),
                "depth": max(0, len(self.paths.get(object_id, [])) - 1),
                "direct_children": len(children),
                "descendants": len(nested),
                "descendant_count": len(nested),
                "leaf_descendants": len(leaves),
                "leaf_descendant_count": len(leaves),
                "leaf_samples": [
                    {"object_id": item, "name": self.names.get(item, "")}
                    for item in leaves[:sample_limit]
                ],
                "leaf_path_samples": [
                    " > ".join(self.paths.get(item, [])) for item in leaves[:sample_limit]
                ],
                "direct_child_samples": [
                    {
                        "object_id": item,
                        "name": self.names.get(item, ""),
                        "child_count": len(self.children.get(item, [])),
                    }
                    for item in children[:sample_limit]
                ],
            })
        normalized_terms = [
            str(term).strip().casefold() for term in (terms or []) if str(term).strip()
        ]
        if normalized_terms:
            matched = [
                item for item in nodes
                if any(
                    term in " ".join([
                        str(item["name"]),
                        *map(str, item["path"]),
                        *(str(sample["name"]) for sample in item["direct_child_samples"]),
                        *(str(sample["name"]) for sample in item["leaf_samples"]),
                    ]).casefold()
                    for term in normalized_terms
                )
            ]
            if matched:
                nodes = matched
        nodes.sort(key=lambda item: (len(item["path"]), item["path"]))
        return {
            "record_count": len(self.records),
            "tree_node_count": len(self.paths),
            "hierarchy_nodes": nodes[:max_nodes],
            "truncated": len(nodes) > max_nodes,
            "identity_signals": {
                "leaf_nodes": sum(1 for children in self.children.values() if not children),
                "records_with_external_id": sum(1 for item in self.records if item.get("external_id")),
                "records_with_ifc_guid": sum(
                    1 for item in self.records if item["properties"].get("IFC Parameters.IfcGUID")
                ),
            },
        }

    def export_python_workspace(self, directory: Path) -> list[Path]:
        """Build a query-friendly snapshot for the on-device Python sandbox."""
        directory.mkdir(parents=True, exist_ok=True)
        database_path = directory / "bim_workspace.sqlite"
        guide_path = directory / "bim_workspace_guide.json"
        if not database_path.is_file():
            temporary = directory / "bim_workspace.building.sqlite"
            if temporary.exists():
                temporary.unlink()
            source = self._workspace_database()
            destination = sqlite3.connect(temporary)
            try:
                source.backup(destination, name="main")
                destination.executescript(
                    """
                    CREATE TABLE ifc_geometry(
                        ifc_step_id INTEGER PRIMARY KEY,
                        global_id TEXT,
                        entity_type TEXT,
                        name TEXT,
                        object_type TEXT,
                        tag TEXT,
                        container_step_id INTEGER,
                        container_type TEXT,
                        container_name TEXT,
                        record_object_ids_json TEXT NOT NULL,
                        min_x_m REAL, min_y_m REAL, min_z_m REAL,
                        max_x_m REAL, max_y_m REAL, max_z_m REAL,
                        size_x_m REAL, size_y_m REAL, size_z_m REAL,
                        bounding_box_volume_m3 REAL,
                        solid_volume_m3 REAL,
                        surface_area_m2 REAL,
                        projected_area_xy_m2 REAL,
                        geometry_vertices INTEGER,
                        geometry_triangles INTEGER,
                        geometry_error TEXT
                    );
                    CREATE INDEX ifc_geometry_type_idx ON ifc_geometry(entity_type);
                    CREATE INDEX ifc_geometry_global_id_idx ON ifc_geometry(global_id);
                    CREATE INDEX ifc_geometry_container_idx ON ifc_geometry(container_step_id);
                    """
                )
                rows = []
                for item in self.ifc_analysis.geometry_inventory():
                    box = item.get("bounding_box") or {}
                    minimum = box.get("min") or {}
                    maximum = box.get("max") or {}
                    size = box.get("size") or {}
                    rows.append((
                        item.get("ifc_step_id"), item.get("global_id"), item.get("entity_type"),
                        item.get("name"), item.get("object_type"), item.get("tag"),
                        item.get("container_step_id"), item.get("container_type"), item.get("container_name"),
                        json.dumps(item.get("record_object_ids", []), ensure_ascii=False),
                        minimum.get("x"), minimum.get("y"), minimum.get("z"),
                        maximum.get("x"), maximum.get("y"), maximum.get("z"),
                        size.get("x"), size.get("y"), size.get("z"), box.get("volume_m3"),
                        item.get("solid_volume_m3"), item.get("surface_area_m2"),
                        item.get("projected_area_xy_m2"), item.get("geometry_vertices"),
                        item.get("geometry_triangles"), item.get("geometry_error"),
                    ))
                destination.executemany(
                    "INSERT OR REPLACE INTO ifc_geometry VALUES ("
                    + ",".join("?" for _ in range(26)) + ")",
                    rows,
                )
                destination.commit()
            finally:
                destination.close()
            temporary.replace(database_path)

        if not guide_path.is_file():
            description = self._describe_workspace(self._workspace_database())
            guide_path.write_text(json.dumps({
                "purpose": "Read-only local bundle for model-authored Python BIM analysis.",
                "raw_source_files": [path.name for path in (self.files.tree, self.files.properties, self.files.ifc)],
                "local_container_paths": {
                    "raw_sources": "/project",
                    "normalized_workspace": "/workspace",
                    "temporary_writes": "/tmp",
                },
                "sqlite_file": database_path.name,
                "sqlite_tables": [item["name"] for item in description["tables"]] + ["ifc_geometry"],
                "ifc_geometry_units": {"length": "m", "area": "m2", "volume": "m3"},
                "analysis_principles": [
                    "Inspect schemas and populations before calculating.",
                    "Treat record_object_id, external_id, IFC STEP id, and GlobalId as different identity signals.",
                    "Do not infer duplicates merely because records share a type GUID or another repeated property.",
                    "Reconcile totals against raw populations and preserve counterexamples.",
                    "Open /workspace/bim_workspace.sqlite with sqlite3 URI mode=ro and print compact JSON evidence.",
                    "The local container has no outbound network access; project and workspace mounts are read-only.",
                    "Only /tmp is writable and it is removed with the container after execution.",
                ],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        return [self.files.tree, self.files.properties, self.files.ifc, database_path, guide_path]

    def search_records(self, arguments: dict[str, Any]) -> dict[str, Any]:
        terms = [_normalize(item) for item in arguments.get("terms", []) if str(item).strip()]
        path_terms = [_normalize(item) for item in arguments.get("path_terms", []) if str(item).strip()]
        key_terms = [_normalize(item) for item in arguments.get("property_keys", []) if str(item).strip()]
        match_all = arguments.get("match", "all") != "any"
        matches = []
        for record in self.records:
            properties = record["properties"]
            searchable = _normalize(" | ".join([
                record["name"], *record["path"],
                *[f"{key}: {value}" for key, value in properties.items()],
            ]))
            term_hits = [term in searchable for term in terms]
            if terms and not (all(term_hits) if match_all else any(term_hits)):
                continue
            path_text = _normalize(" | ".join(record["path"]))
            if path_terms and not all(term in path_text for term in path_terms):
                continue
            if key_terms and not all(any(term in _normalize(key) for key in properties) for term in key_terms):
                continue
            matched = [
                {"key": key, "value": value}
                for key, value in properties.items()
                if any(term in _normalize(f"{key}: {value}") for term in terms)
            ][:20]
            matches.append({
                "object_id": record["object_id"], "name": record["name"],
                "external_id": record["external_id"], "path": record["path"],
                "matched_properties": matched,
            })
        page = self._page("search_records", arguments, matches)
        return {
            **page,
            "total_matches": page["total_count"],
            "returned": page["returned_count"],
            "records": page["results"],
        }

    def aggregate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        records = [self.by_id[item] for item in map(str, arguments.get("object_ids", [])) if item in self.by_id]
        operation = str(arguments.get("operation", "count"))
        field = arguments.get("field")
        distinct_by = arguments.get("distinct_by") or "object_id"
        group_by = arguments.get("group_by")
        output_unit = arguments.get("output_unit")

        def calculate(group: list[dict[str, Any]]) -> dict[str, Any]:
            if operation == "count":
                values = {_record_value(item, str(distinct_by)) for item in group}
                return {"value": len({value for value in values if value not in (None, "")}), "records": len(group)}
            if operation == "distinct_count":
                target = str(field or distinct_by)
                values = {_record_value(item, target) for item in group}
                return {"value": len({str(value) for value in values if value not in (None, "")}), "records": len(group)}
            if not field:
                raise ProjectError(f"{operation} requires field")
            values: list[tuple[float, str | None, str]] = []
            missing = 0
            for item in group:
                number, unit = _number_unit(_record_value(item, str(field)))
                if number is None:
                    missing += 1
                    continue
                number, unit = _convert(number, unit, str(output_unit) if output_unit else None)
                values.append((number, unit, item["object_id"]))
            if not values:
                return {"value": None, "records": len(group), "missing": missing}
            numbers = [item[0] for item in values]
            if operation == "sum":
                result = sum(numbers)
            elif operation == "average":
                result = sum(numbers) / len(numbers)
            elif operation == "min":
                result = min(numbers)
            elif operation == "max":
                result = max(numbers)
            else:
                raise ProjectError(f"Unsupported operation: {operation}")
            units = {item[1] for item in values if item[1]}
            extremum_ids = [item[2] for item in values if item[0] == result] if operation in {"min", "max"} else []
            return {
                "value": result, "unit": next(iter(units)) if len(units) == 1 else None,
                "mixed_units": sorted(units) if len(units) > 1 else [],
                "records": len(group), "values_used": len(values), "missing": missing,
                "matching_object_ids": extremum_ids,
            }

        if not group_by:
            return {"operation": operation, "field": field, **calculate(records)}
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(_record_value(record, str(group_by)) or "(missing)")].append(record)
        return {
            "operation": operation, "field": field, "group_by": group_by,
            "groups": [{"group": key, **calculate(group)} for key, group in groups.items()],
        }

    def search_ifc(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.files.ifc.read_bytes()[:16] == b"SQLite format 3\x00":
            return self._search_ifc_sqlite(arguments)
        if self._ifc_lines is None:
            try:
                self._ifc_lines = self.files.ifc.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                raise ProjectError(f"Cannot read IFC file: {exc}") from exc
        terms = [_normalize(item) for item in arguments.get("terms", []) if str(item).strip()]
        match_all = arguments.get("match", "all") != "any"
        matches = []
        for index, line in enumerate(self._ifc_lines, 1):
            normalized = _normalize(line)
            hits = [term in normalized for term in terms]
            if terms and (all(hits) if match_all else any(hits)):
                matches.append({"line": index, "text": line[:4000]})
        page = self._page("search_ifc", arguments, matches)
        return {**page, "total_matches": page["total_count"], "matches": page["results"]}

    def _ifc_inventory(self) -> dict[str, Any]:
        if self.files.ifc.read_bytes()[:16] != b"SQLite format 3\x00":
            return {"format": "STEP_or_text", "bytes": self.files.ifc.stat().st_size}
        with self._connect_ifc_sqlite() as connection:
            tables = [
                str(row[0]) for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            return {
                "format": "SQLite",
                "tables": [
                    {
                        "name": table,
                        "columns": [str(row[1]) for row in connection.execute(
                            f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")'
                        )],
                    }
                    for table in tables[:200]
                ],
            }

    def _search_ifc_sqlite(self, arguments: dict[str, Any]) -> dict[str, Any]:
        terms = [_normalize(item) for item in arguments.get("terms", []) if str(item).strip()]
        match_all = arguments.get("match", "all") != "any"
        matches = []
        with self._connect_ifc_sqlite() as connection:
            tables = [
                str(row[0]) for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            for table in tables:
                quoted = table.replace('"', '""')
                columns = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{quoted}")')]
                for row in connection.execute(f'SELECT * FROM "{quoted}"'):
                    values = ["" if value is None else str(value) for value in row]
                    text = _normalize(" | ".join(values))
                    hits = [term in text for term in terms]
                    if terms and (all(hits) if match_all else any(hits)):
                        matches.append({"table": table, "row": dict(zip(columns, values))})
        page = self._page("search_ifc", arguments, matches)
        return {
            **page,
            "format": "SQLite",
            "total_matches": page["total_count"],
            "matches": page["results"],
        }

    def _connect_ifc_sqlite(self) -> sqlite3.Connection:
        return sqlite3.connect(self.files.ifc.as_uri() + "?mode=ro", uri=True)


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    required_fields = list(properties) if strict else (required or [])
    return {
        "type": "function", "name": name, "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False,
        },
        "strict": strict,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    output: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.update(_flatten(item, f"{prefix}[{index}]"))
    elif value is not None:
        output[prefix] = str(value)
    return output


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^\w\u0590-\u05ff.+-]+", " ", text).split())


def _record_value(record: dict[str, Any], field: str) -> Any:
    normalized = _normalize(field)
    if normalized in {"object id", "object_id"}:
        return record["object_id"]
    if normalized == "name":
        return record["name"]
    if normalized in {"external id", "external_id"}:
        return record["external_id"]
    match = re.fullmatch(r"path[_ ](\d+)", normalized)
    if match:
        index = int(match.group(1))
        return record["path"][index] if index < len(record["path"]) else None
    properties = record["properties"]
    if field in properties:
        return properties[field]
    candidates = [value for key, value in properties.items() if _normalize(key) == normalized]
    if not candidates:
        candidates = [value for key, value in properties.items() if normalized in _normalize(key)]
    return candidates[0] if len(candidates) == 1 else None


def _number_unit(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value).replace(",", "."))
    if not match:
        return None, None
    unit_match = re.search(r"\b(mm|cm|m|ft|in)\b", str(value), re.IGNORECASE)
    return float(match.group()), unit_match.group(1).lower() if unit_match else None


def _convert(number: float, source: str | None, target: str | None) -> tuple[float, str | None]:
    if not target or not source or source == target:
        return number, target or source
    metres = {"mm": 0.001, "cm": 0.01, "m": 1.0, "ft": 0.3048, "in": 0.0254}
    if source not in metres or target not in metres:
        return number, source
    return number * metres[source] / metres[target], target


def _iter_ifc_entities(path: Path):
    """Yield common STEP entity declarations, including declarations split over lines."""
    pending = ""
    start_line = 0
    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ProjectError(f"Cannot read IFC file: {exc}") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            stripped = raw_line.strip()
            if not pending:
                if not stripped.startswith("#"):
                    continue
                pending = stripped
                start_line = line_number
            else:
                pending += "\n" + stripped
            if not _ifc_statement_complete(pending):
                continue
            match = re.match(
                r"^#(\d+)\s*=\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*;\s*$",
                pending,
                re.DOTALL,
            )
            if match:
                yield (
                    int(match.group(1)), match.group(2).upper(), match.group(3), pending, start_line,
                )
            pending = ""
            start_line = 0


def _ifc_statement_complete(value: str) -> bool:
    in_string = False
    index = 0
    while index < len(value):
        character = value[index]
        if character == "'":
            if in_string and index + 1 < len(value) and value[index + 1] == "'":
                index += 2
                continue
            in_string = not in_string
        elif character == ";" and not in_string:
            return not value[index + 1:].strip()
        index += 1
    return False


def _workspace_authorizer(
    action: int,
    argument_one: str | None,
    argument_two: str | None,
    database_name: str | None,
    trigger_name: str | None,
) -> int:
    del database_name, trigger_name
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", -1),
    }
    if action not in allowed:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = str(argument_two or argument_one or "").casefold()
        if function_name in {"load_extension", "readfile", "writefile"}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _sql_json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        preview = value[:256].hex()
        return {"type": "blob", "bytes": len(value), "hex_preview": preview}
    if isinstance(value, str) and len(value) > 8_000:
        return {
            "truncated": True,
            "original_characters": len(value),
            "preview": value[:8_000],
        }
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_calculate(expression: str) -> float:
    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](visit(node.left), visit(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](visit(node.operand)))
        raise ProjectError("Expression contains unsupported syntax.")

    value = visit(ast.parse(expression, mode="eval"))
    if not math.isfinite(value):
        raise ProjectError("Expression result is not finite.")
    return value
