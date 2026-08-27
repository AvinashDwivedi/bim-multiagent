from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .models import ElementRecord
from .ifc_graph import IfcGraph


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetFiles:
    tree: Path
    properties: Path
    sqlite_ifc: Path


class ProjectDataset:
    """Read-only view over the three Autodesk-style BIM export files."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).resolve()
        self.files = self._discover_files(self.data_dir)
        self._tree_json = self._read_json(self.files.tree)
        raw_properties = self._read_json(self.files.properties)
        if not isinstance(raw_properties, dict):
            raise DatasetError("Properties JSON must contain an object keyed by object ID.")
        if "data" in raw_properties and isinstance(raw_properties["data"], dict):
            raw_properties = raw_properties["data"]
        self._properties: dict[str, dict[str, Any]] = {
            str(key): value for key, value in raw_properties.items() if isinstance(value, dict)
        }
        self._paths: dict[int, tuple[str, ...]] = {}
        self._leaf_ids: set[int] = set()
        self._children: dict[int, tuple[int, ...]] = {}
        self._index_tree()
        self.ifc_graph = IfcGraph.load(self.files.sqlite_ifc)
        self.records = self._build_records()
        if self.ifc_graph is not None:
            self.records = [
                replace(record, level=self.ifc_graph.storey_for_guid(record.global_id))
                if not record.level and self.ifc_graph.storey_for_guid(record.global_id)
                else record
                for record in self.records
            ]
        self.by_id = {record.object_id: record for record in self.records}
        self.physical_records = [record for record in self.records if record.is_physical]

    @staticmethod
    def _discover_files(data_dir: Path) -> DatasetFiles:
        if not data_dir.is_dir():
            raise DatasetError(f"Data directory does not exist: {data_dir}")
        trees = sorted(data_dir.glob("*-tree.json"))
        properties = sorted(data_dir.glob("*-properties.json"))
        sqlite_ifcs = sorted(data_dir.glob("*.ifc"))
        if len(trees) != 1 or len(properties) != 1 or len(sqlite_ifcs) != 1:
            raise DatasetError(
                "Expected exactly one *-tree.json, one *-properties.json, and one *.ifc; "
                f"found {len(trees)}, {len(properties)}, and {len(sqlite_ifcs)}."
            )
        tree_stem = trees[0].name.removesuffix("-tree.json")
        property_stem = properties[0].name.removesuffix("-properties.json")
        if tree_stem != property_stem:
            raise DatasetError("Tree and properties JSON files do not share a model identifier.")
        return DatasetFiles(trees[0], properties[0], sqlite_ifcs[0])

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Cannot parse {path.name}: {exc}") from exc

    def _index_tree(self) -> None:
        data = self._tree_json.get("data", self._tree_json)
        roots = data.get("objects", []) if isinstance(data, dict) else []
        if not isinstance(roots, list) or not roots:
            raise DatasetError("Tree JSON does not contain data.objects roots.")

        def visit(node: dict[str, Any], parent_path: tuple[str, ...]) -> None:
            try:
                object_id = int(node["objectid"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetError("Every tree node must have an integer objectid.") from exc
            name = str(node.get("name") or f"object-{object_id}")
            path = parent_path + (name,)
            self._paths[object_id] = path
            child_nodes = node.get("objects") if isinstance(node.get("objects"), list) else []
            child_ids = tuple(int(child["objectid"]) for child in child_nodes)
            self._children[object_id] = child_ids
            if child_nodes:
                for child in child_nodes:
                    visit(child, path)
            else:
                self._leaf_ids.add(object_id)

        for root in roots:
            if isinstance(root, dict):
                visit(root, ())

    def _build_records(self) -> list[ElementRecord]:
        records: list[ElementRecord] = []
        for key, raw in self._properties.items():
            try:
                object_id = int(raw.get("objectid", key))
            except (TypeError, ValueError):
                continue
            name = str(raw.get("name") or f"object-{object_id}")
            path = self._paths.get(object_id, (name,))
            properties = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
            flat = flatten_properties(properties)
            category = path[1] if len(path) > 1 else _find_value(flat, "Category")
            family = _family(path, flat)
            type_name = _type_name(path, flat, name)
            level = (
                _find_value(flat, "Schedule Level")
                or _find_value(flat, "Reference Level")
                or _find_value(flat, "Level")
            )
            external_id = _optional_string(raw.get("externalId"))
            global_id = _find_value(flat, "GlobalID") or _find_value(flat, "IfcGUID") or external_id
            is_leaf = object_id in self._leaf_ids
            workset = _find_value(flat, "Workset")
            is_physical = _physical_role(
                is_leaf=is_leaf,
                path=path,
                category=category,
                name=name,
                workset=workset,
            )
            records.append(
                ElementRecord(
                    object_id=object_id,
                    name=name,
                    external_id=external_id,
                    global_id=global_id,
                    path=path,
                    category=category,
                    family=family,
                    type_name=type_name,
                    level=level,
                    properties=properties,
                    flat_properties=flat,
                    is_leaf=is_leaf,
                    is_physical=is_physical,
                )
            )
        return records

    def profile(self, max_values: int = 160, focus_terms: Iterable[str] = ()) -> dict[str, Any]:
        categories = Counter(record.category for record in self.physical_records if record.category)
        family_types: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        levels = Counter()
        property_keys = Counter()
        for record in self.physical_records:
            family_types[record.category][(record.family, record.type_name)] += 1
            if record.level:
                levels[record.level] += 1
            property_keys.update(record.flat_properties.keys())
        branches = []
        focused = [term for term in expand_terms(focus_terms) if len(term) >= 2]
        per_category = max(20, min(40, max_values // max(1, len(categories)) * 2))
        for category, count in categories.most_common():
            entries = []
            ranked = list(family_types[category].items())
            ranked.sort(
                key=lambda item: (
                    -sum(term in normalize(f"{item[0][0]} {item[0][1]}") for term in focused),
                    -item[1],
                    item[0],
                )
            )
            focused_count = sum(
                1
                for (family, type_name), _ in ranked
                if focused and any(term in normalize(f"{family} {type_name}") for term in focused)
            )
            branch_limit = min(60, max(per_category, focused_count))
            for (family, type_name), item_count in ranked[:branch_limit]:
                entries.append({"family": family, "type": type_name, "count": item_count})
            branches.append({"category": category, "count": count, "family_types": entries})
        focused_property_values: Counter[tuple[str, str, str, str, str]] = Counter()
        significant_focus = {
            term for term in focused if len(term) >= 3 and term not in _PROFILE_STOPWORDS
        }
        if significant_focus:
            for record in self.physical_records:
                for key, value in record.flat_properties.items():
                    text = normalize(value)
                    if any(term in text for term in significant_focus):
                        focused_property_values[(key, value, record.category, record.family, record.type_name)] += 1

        requested_keys = _requested_property_keys(" ".join(str(item) for item in focus_terms), property_keys)
        property_stats = []
        property_category_coverage: dict[str, list[dict[str, Any]]] = {}
        for key in requested_keys:
            values = Counter(
                record.flat_properties[key]
                for record in self.physical_records
                if key in record.flat_properties and record.flat_properties[key].strip()
            )
            present = sum(key in record.flat_properties for record in self.physical_records)
            property_stats.append({
                "field": key,
                "present": present,
                "nonempty": sum(values.values()),
                "distinct": len(values),
                "samples": [{"value": value, "count": count} for value, count in values.most_common(12)],
            })
            coverage = Counter(
                record.category for record in self.physical_records if key in record.flat_properties
            )
            nonempty_coverage = Counter(
                record.category for record in self.physical_records
                if record.flat_properties.get(key, "").strip()
            )
            property_category_coverage[key] = [
                {
                    "category": category,
                    "present": count,
                    "nonempty": nonempty_coverage.get(category, 0),
                }
                for category, count in coverage.most_common()
            ]

        question_text = normalize(" ".join(str(item) for item in focus_terms))
        question_tokens = {
            token for token in expand_terms(focus_terms)
            if len(token) >= 2 and token not in _PROFILE_STOPWORDS
        }
        property_candidates = []
        for key, count in property_keys.items():
            normalized_key = normalize(key)
            leaf = normalize(key.rsplit(".", 1)[-1])
            score = sum(
                4 if token == leaf else 2 if token in leaf else 1 if token in normalized_key else 0
                for token in question_tokens
            )
            if score or (leaf and leaf in question_text):
                property_candidates.append({"field": key, "score": score, "present": count})
        property_candidates.sort(key=lambda item: (-item["score"], -item["present"], item["field"]))

        return {
            "record_count": len(self.records),
            "physical_instance_count": len(self.physical_records),
            "categories": branches,
            "levels": [{"value": value, "count": count} for value, count in levels.most_common(40)],
            "property_keys": [key for key, _ in property_keys.most_common(240)],
            "property_stats": property_stats,
            "property_candidates": property_candidates[:40],
            "property_category_coverage": property_category_coverage,
            "focused_property_values": [
                {
                    "field": key, "value": value, "category": category,
                    "family": family, "type": type_name, "count": count,
                }
                for (key, value, category, family, type_name), count
                in focused_property_values.most_common(80)
            ],
            "ifc": self.ifc_graph.inventory() if self.ifc_graph is not None else {"is_step_ifc": False},
        }

    def source_manifest(self) -> list[dict[str, Any]]:
        output = []
        for kind, path in (
            ("tree", self.files.tree),
            ("properties", self.files.properties),
            ("ifc_sqlite", self.files.sqlite_ifc),
        ):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            output.append(
                {"kind": kind, "path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
            )
        return output

    def search(self, terms: Iterable[str], *, physical_only: bool = True) -> list[ElementRecord]:
        expanded = expand_terms(terms)
        records = self.physical_records if physical_only else self.records
        scored = []
        for record in records:
            text = normalize(record.searchable_text)
            score = sum(1 for term in expanded if term and term in text)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].category, item[1].family, item[1].type_name))
        return [record for _, record in scored]

    def sqlite_inventory(self) -> dict[str, Any]:
        if self.files.sqlite_ifc.read_bytes()[:16] != b"SQLite format 3\x00":
            return {
                "is_sqlite": False, "tables": {},
                "ifc_graph": self.ifc_graph.inventory() if self.ifc_graph is not None else None,
            }
        with self._connect_sqlite() as connection:
            names = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            tables = {}
            for name in names:
                if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                    continue
                tables[name] = connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        return {"is_sqlite": True, "tables": tables}

    def connectivity_summary(self, records: Iterable[ElementRecord]) -> dict[str, Any]:
        if self.ifc_graph is None:
            return {"available": False, "reason": "The IFC source is not a parsed STEP connectivity graph."}
        return self.ifc_graph.connectivity_summary(
            record.global_id for record in records if record.global_id
        )

    def systems_for(self, record: ElementRecord) -> list[str]:
        return self.ifc_graph.systems_for_guid(record.global_id) if self.ifc_graph is not None else []

    def has_ifc_material(self, record: ElementRecord) -> bool:
        return self.ifc_graph.has_material(record.global_id) if self.ifc_graph is not None else False

    def has_ifc_ports(self, record: ElementRecord) -> bool:
        return self.ifc_graph.has_ports(record.global_id) if self.ifc_graph is not None else False

    def has_connected_ifc_port(self, record: ElementRecord) -> bool:
        return self.ifc_graph.has_connected_port(record.global_id) if self.ifc_graph is not None else False

    def ifc_placement_height(self, record: ElementRecord) -> tuple[float | None, str]:
        if self.ifc_graph is None:
            return None, ""
        return self.ifc_graph.height_above_storey_for_guid(record.global_id), self.ifc_graph.length_unit

    def search_sqlite_metadata(self, terms: Iterable[str], limit: int = 100) -> list[dict[str, Any]]:
        """Search hidden EAV metadata to expose definitions/templates as audit counterexamples."""
        if self.files.sqlite_ifc.read_bytes()[:16] != b"SQLite format 3\x00":
            return []
        selected = [term for term in expand_terms(terms) if len(term) >= 2][:12]
        if not selected:
            return []
        clauses = " OR ".join("lower(CAST(v.value AS TEXT)) LIKE ?" for _ in selected)
        parameters = [f"%{term}%" for term in selected]
        query = f"""
            SELECT DISTINCT e.entity_id, a.category, a.name, CAST(v.value AS TEXT)
            FROM _objects_eav e
            JOIN _objects_attr a ON a.id = e.attribute_id
            JOIN _objects_val v ON v.id = e.value_id
            WHERE {clauses}
            ORDER BY e.entity_id
            LIMIT ?
        """
        try:
            with self._connect_sqlite() as connection:
                rows = connection.execute(query, [*parameters, limit]).fetchall()
        except sqlite3.DatabaseError:
            return []
        return [
            {
                "entity_id": row[0],
                "attribute_category": row[1],
                "attribute": row[2],
                "value": _clean_sqlite_text(row[3]),
                "in_tree": int(row[0]) in self._paths,
            }
            for row in rows
        ]

    def _connect_sqlite(self) -> sqlite3.Connection:
        uri = self.files.sqlite_ifc.resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)


def flatten_properties(value: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(item: Any, prefix: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")
        elif item is not None:
            result[prefix] = str(item)

    visit(value, "")
    return result


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[_:/\\|\[\](),.-]+", " ", text)
    return " ".join(text.split())


TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "מפסקים": ("מפס", "switch", "switches", "switch button"),
    "מפסק": ("מפס", "switch", "switches", "switch button"),
    "דירות": ("דירה", "apartment", "apartments", "flat"),
    "דירה": ("דירה", "apartment", "flat"),
    "צינורות": ("צינור", "pipe", "pipes"),
    "צינור": ("צינור", "pipe"),
    "ספרינקלרים": ("sprinkler", "sprinklers", "מתז"),
    "קומה": ("level", "floor", "storey", "קומה"),
    "קירות": ("wall", "walls", "קיר"),
    "דלתות": ("door", "doors", "דלת"),
    "חלונות": ("window", "windows", "חלון"),
    "שקעים": ("socket", "outlet", "receptacle", "שקע"),
    "socket": ("שקע", "שקעים"),
    "sockets": ("socket", "sock", "outlet", "receptacle", "שקע", "שקעים"),
    "box": ("קופס", "קופסה", "קופסאת"),
    "module": ("מודול", "מודולים"),
    "modules": ("מודול", "מודולים"),
    "לוחות": ("panel", "switchboard", "לוח"),
    "חומר": ("material",),
    "רוחב": ("width",),
    "גובה": ("height", "elevation", "offset"),
    "אורך": ("length",),
    "חיבור": ("connection", "connected", "panel", "circuit"),
    "המשכיות": ("continuity", "connection", "system"),
    "מגשים": ("cable tray", "cable trays", "tray"),
    "מגש": ("cable tray", "tray"),
    "עומס": ("load", "apparent load", "electrical load"),
    "סימון": ("mark",),
    "מספר": ("number", "mark"),
    "ממוצע": ("average", "mean"),
    "אחוז": ("percentage", "percent"),
    "חסר": ("missing", "without", "empty"),
    "ללא": ("missing", "without", "empty"),
}


_PROFILE_STOPWORDS = {
    "what", "which", "how", "many", "much", "from", "with", "project", "type", "types",
    "כמה", "מה", "אילו", "של", "לפי", "בפרויקט", "בפרוייקט", "מסוג", "מתוכננות",
}


def _requested_property_keys(question: str, property_keys: Counter[str]) -> list[str]:
    text = normalize(question)
    concepts = {
        "material": ("material", "חומר"),
        "width": ("width", "רוחב"),
        "height": ("height", "elevation", "offset", "גובה", "מרחק", "רצפה"),
        "length": ("length", "אורך"),
        "description": ("description", "תיאור", "מסוג"),
        "panel": ("panel", "circuit", "system", "לוח", "חיבור", "המשכיות"),
    }
    requested = {
        concept
        for concept, terms in concepts.items()
        if any(normalize(term) in text for term in terms)
    }
    leaf_terms = {
        "material": ("material",),
        "width": ("width",),
        "height": ("height", "elevation", "offset", "גובה מרצפה", "default elevation"),
        "length": ("length", "tray length"),
        "description": ("description",),
        "panel": ("panel", "circuit number", "system"),
    }
    output = []
    for key, _ in property_keys.most_common():
        normalized_key = normalize(key)
        if any(
            term in normalized_key
            for concept in requested
            for term in leaf_terms.get(concept, ())
        ):
            output.append(key)
    tokens = {
        token for token in expand_terms([question])
        if len(token) >= 2 and token not in _PROFILE_STOPWORDS
    }
    ranked = []
    for key, count in property_keys.items():
        normalized_key = normalize(key)
        leaf = normalize(key.rsplit(".", 1)[-1])
        score = sum(
            4 if token == leaf else 2 if token in leaf else 1 if token in normalized_key else 0
            for token in tokens
        )
        if score:
            ranked.append((score, count, key))
    ranked.sort(reverse=True)
    output.extend(key for _, _, key in ranked[:20])
    return list(dict.fromkeys(output))[:40]


def expand_terms(terms: Iterable[str]) -> list[str]:
    output: list[str] = []
    for raw in terms:
        normalized = normalize(raw)
        if not normalized:
            continue
        output.append(normalized)
        for token in normalized.split():
            output.append(token)
            output.extend(normalize(alias) for alias in TERM_ALIASES.get(token, ()))
            if token.endswith("ies") and len(token) > 4:
                output.append(token[:-3] + "y")
            elif token.endswith("ches") or token.endswith("shes") or token.endswith("xes") or token.endswith("zes"):
                output.append(token[:-2])
            elif token.endswith("s") and len(token) > 3:
                output.append(token[:-1])
    return list(dict.fromkeys(output))


def _find_value(flat: dict[str, str], leaf_name: str) -> str:
    target = normalize(leaf_name)
    for key, value in flat.items():
        if normalize(key.rsplit(".", 1)[-1]) == target and value.strip():
            return value.strip()
    return ""


def _family(path: tuple[str, ...], flat: dict[str, str]) -> str:
    workset = _find_value(flat, "Workset")
    if normalize(workset).startswith("family") and ":" in workset:
        return workset.split(":")[-1].strip()
    if len(path) >= 4:
        return path[2]
    return ""


def _type_name(path: tuple[str, ...], flat: dict[str, str], name: str) -> str:
    explicit = _find_value(flat, "Type Name")
    if explicit:
        return explicit
    if len(path) >= 5:
        return path[-2]
    clean = re.sub(r"\s*\[\d+\]\s*$", "", name)
    return clean if clean != name else ""


def _physical_role(
    *, is_leaf: bool, path: tuple[str, ...], category: str, name: str, workset: str
) -> bool:
    if not is_leaf or len(path) < 3:
        return False
    combined = normalize(f"{category} {name} {workset}")
    excluded = (
        "object styles",
        "loaded family information",
        "panel schedule template",
        "legend component",
        "project info",
        "analytical",
    )
    if any(term in combined for term in excluded):
        return False
    if ".dwg" in category.casefold() or " dwg" in normalize(category):
        return False
    return True


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_sqlite_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"[\x00-\x1f]", " ", str(value)).strip()
