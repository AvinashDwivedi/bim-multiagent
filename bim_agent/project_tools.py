from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
            _tool("list_tree_children", "List direct child nodes for a tree object ID. Omit parent_object_id for roots.", {
                "parent_object_id": {"type": ["string", "null"]},
            }),
            _tool("search_records", "Search raw record names, hierarchy paths, property keys, and property values. Semantics are chosen by you.", {
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
            _tool("search_ifc", "Search raw IFC text lines for entity names, GUIDs, relationships, placements, systems, or values.", {
                "terms": {"type": "array", "items": {"type": "string"}},
                "match": {"type": "string", "enum": ["all", "any"]},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            }, required=["terms"]),
            _tool("calculate", "Evaluate arithmetic using numbers, parentheses, +, -, *, /, %, and powers. Use this instead of mental arithmetic.", {
                "expression": {"type": "string"},
            }, required=["expression"]),
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
            return {"children": [
                {
                    "object_id": item,
                    "name": self.names.get(item, ""),
                    "path": self.paths.get(item, []),
                    "child_count": len(self.children.get(item, [])),
                }
                for item in ids[:500]
            ]}
        if name == "search_records":
            return self.search_records(arguments)
        if name == "get_records":
            return {"records": [self.by_id[item] for item in map(str, arguments.get("object_ids", [])) if item in self.by_id]}
        if name == "aggregate_records":
            return self.aggregate(arguments)
        if name == "search_ifc":
            return self.search_ifc(arguments)
        if name == "calculate":
            return {"expression": str(arguments.get("expression", "")), "value": _safe_calculate(str(arguments.get("expression", "")))}
        raise ProjectError(f"Unknown tool: {name}")

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
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(200, max(1, int(arguments.get("limit", 100))))
        return {"total_matches": len(matches), "offset": offset, "returned": len(matches[offset:offset + limit]), "records": matches[offset:offset + limit]}

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
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(200, max(1, int(arguments.get("limit", 100))))
        return {"total_matches": len(matches), "offset": offset, "matches": matches[offset:offset + limit]}

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
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(200, max(1, int(arguments.get("limit", 100))))
        return {"format": "SQLite", "total_matches": len(matches), "offset": offset, "matches": matches[offset:offset + limit]}

    def _connect_ifc_sqlite(self) -> sqlite3.Connection:
        return sqlite3.connect(self.files.ifc.as_uri() + "?mode=ro", uri=True)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function", "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False},
        "strict": False,
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
