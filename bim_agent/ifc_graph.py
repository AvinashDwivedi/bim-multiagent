from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_ENTITY = re.compile(r"(?ms)^#(\d+)\s*=\s*(IFC[A-Z0-9_]+)\s*\((.*?)\)\s*;\s*$")
_REFERENCE = re.compile(r"#(\d+)")
_STRING = re.compile(r"'((?:''|[^'])*)'")


@dataclass
class IfcGraph:
    path: Path
    entity_types: dict[int, str] = field(default_factory=dict)
    entity_args: dict[int, str] = field(default_factory=dict)
    guid_to_entity: dict[str, int] = field(default_factory=dict)
    entity_to_guid: dict[int, str] = field(default_factory=dict)
    storey_names: dict[int, str] = field(default_factory=dict)
    element_storeys: dict[int, str] = field(default_factory=dict)
    port_owner: dict[int, int] = field(default_factory=dict)
    port_edges: list[tuple[int, int]] = field(default_factory=list)
    element_systems: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    material_elements: set[int] = field(default_factory=set)
    element_placements: dict[int, int] = field(default_factory=dict)
    placement_parents: dict[int, int] = field(default_factory=dict)
    placement_z: dict[int, float] = field(default_factory=dict)
    storey_entities: dict[int, int] = field(default_factory=dict)
    length_unit: str = ""

    @classmethod
    def load(cls, path: Path) -> "IfcGraph | None":
        if path.read_bytes()[:16] == b"SQLite format 3\x00":
            return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if not text.lstrip().startswith("ISO-10303-21"):
            return None
        graph = cls(path=path)
        graph._parse(text)
        return graph

    def _parse(self, text: str) -> None:
        for match in _ENTITY.finditer(text):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            args = match.group(3)
            self.entity_types[entity_id] = entity_type
            self.entity_args[entity_id] = args
            strings = _strings(args)
            if strings and _looks_like_global_id(strings[0]):
                self.guid_to_entity[strings[0]] = entity_id
                self.entity_to_guid[entity_id] = strings[0]
            if entity_type == "IFCBUILDINGSTOREY" and len(strings) >= 2:
                self.storey_names[entity_id] = strings[1]
            if entity_type == "IFCSIUNIT" and ".LENGTHUNIT." in args and not self.length_unit:
                self.length_unit = _si_length_unit(args)

        for entity_id, entity_type in self.entity_types.items():
            parts = _split_args(self.entity_args[entity_id])
            if entity_type == "IFCLOCALPLACEMENT" and len(parts) >= 2:
                parent = _single_ref(parts[0])
                relative = _single_ref(parts[1])
                if parent is not None:
                    self.placement_parents[entity_id] = parent
                if relative is not None:
                    self.placement_z[entity_id] = self._axis_z(relative)
            elif len(parts) > 5:
                placement = _single_ref(parts[5])
                if placement is not None and self.entity_types.get(placement) == "IFCLOCALPLACEMENT":
                    self.element_placements[entity_id] = placement
                    if entity_type == "IFCBUILDINGSTOREY":
                        self.storey_entities[entity_id] = placement

        for entity_id, entity_type in self.entity_types.items():
            parts = _split_args(self.entity_args[entity_id])
            if entity_type == "IFCRELCONTAINEDINSPATIALSTRUCTURE" and len(parts) >= 6:
                storey = _single_ref(parts[5])
                name = self.storey_names.get(storey or -1, "")
                if name:
                    for related in _refs(parts[4]):
                        self.element_storeys[related] = name
            elif entity_type == "IFCRELCONNECTSPORTTOELEMENT" and len(parts) >= 6:
                port = _single_ref(parts[4])
                element = _single_ref(parts[5])
                if port is not None and element is not None:
                    self.port_owner[port] = element
            elif entity_type == "IFCRELCONNECTSPORTS" and len(parts) >= 6:
                first = _single_ref(parts[4])
                second = _single_ref(parts[5])
                if first is not None and second is not None:
                    self.port_edges.append((first, second))
            elif entity_type == "IFCRELASSIGNSTOGROUP" and len(parts) >= 6:
                related = _refs(parts[4])
                group = _single_ref(parts[-1])
                group_name = self._entity_name(group) if group is not None else ""
                if group_name:
                    for related_id in related:
                        self.element_systems[related_id].add(group_name)
            elif entity_type == "IFCRELASSOCIATESMATERIAL" and len(parts) >= 6:
                self.material_elements.update(_refs(parts[4]))

    def _entity_name(self, entity_id: int) -> str:
        strings = _strings(self.entity_args.get(entity_id, ""))
        return strings[1] if len(strings) >= 2 else (strings[0] if strings else "")

    def _axis_z(self, entity_id: int) -> float:
        if self.entity_types.get(entity_id) not in {"IFCAXIS2PLACEMENT3D", "IFCAXIS2PLACEMENT2D"}:
            return 0.0
        point = _single_ref(_split_args(self.entity_args.get(entity_id, ""))[0])
        if point is None or self.entity_types.get(point) != "IFCCARTESIANPOINT":
            return 0.0
        coordinates = _numbers(self.entity_args.get(point, ""))
        return coordinates[2] if len(coordinates) >= 3 else 0.0

    def entity_for_guid(self, guid: str | None) -> int | None:
        return self.guid_to_entity.get(guid or "")

    def storey_for_guid(self, guid: str | None) -> str:
        entity = self.entity_for_guid(guid)
        return self.element_storeys.get(entity or -1, "")

    def systems_for_guid(self, guid: str | None) -> list[str]:
        entity = self.entity_for_guid(guid)
        return sorted(self.element_systems.get(entity or -1, set()))

    def has_material(self, guid: str | None) -> bool:
        entity = self.entity_for_guid(guid)
        return entity in self.material_elements if entity is not None else False

    def has_ports(self, guid: str | None) -> bool:
        entity = self.entity_for_guid(guid)
        return entity in set(self.port_owner.values()) if entity is not None else False

    def has_connected_port(self, guid: str | None) -> bool:
        entity = self.entity_for_guid(guid)
        if entity is None:
            return False
        connected_ports = {port for edge in self.port_edges for port in edge}
        return any(owner == entity and port in connected_ports for port, owner in self.port_owner.items())

    def height_above_storey_for_guid(self, guid: str | None) -> float | None:
        """Return the product placement origin above its containing storey in IFC length units.

        This deliberately reports placement, not an inferred Revit property.  Full body geometry may
        have a different centre, so callers label this value explicitly as a placement height.
        """
        entity = self.entity_for_guid(guid)
        storey = self.element_storeys.get(entity or -1)
        if entity is None or not storey:
            return None
        storey_entity = next(
            (item for item, name in self.storey_names.items() if name == storey), None
        )
        stop = self.storey_entities.get(storey_entity or -1)
        placement = self.element_placements.get(entity)
        if placement is None:
            return None
        total = 0.0
        seen: set[int] = set()
        while placement is not None and placement not in seen and placement != stop:
            seen.add(placement)
            total += self.placement_z.get(placement, 0.0)
            placement = self.placement_parents.get(placement)
        return total if placement == stop else None

    def component_index(self) -> dict[int, int]:
        adjacency = self._element_adjacency()
        return {
            entity: index
            for index, component in enumerate(_components(adjacency))
            for entity in component
        }

    def _element_adjacency(self) -> dict[int, set[int]]:
        owners = set(self.port_owner.values())
        adjacency: dict[int, set[int]] = {owner: set() for owner in owners}
        for first_port, second_port in self.port_edges:
            first_owner = self.port_owner.get(first_port)
            second_owner = self.port_owner.get(second_port)
            if first_owner is None or second_owner is None or first_owner == second_owner:
                continue
            adjacency[first_owner].add(second_owner)
            adjacency[second_owner].add(first_owner)
        return adjacency

    def connectivity_summary(self, selected_guids: Iterable[str] = ()) -> dict:
        owners = set(self.port_owner.values())
        adjacency = self._element_adjacency()
        connected_port_pairs = 0
        for first_port, second_port in self.port_edges:
            first_owner = self.port_owner.get(first_port)
            second_owner = self.port_owner.get(second_port)
            if first_owner is None or second_owner is None:
                continue
            connected_port_pairs += 1
            if first_owner != second_owner:
                adjacency[first_owner].add(second_owner)
                adjacency[second_owner].add(first_owner)

        isolated = {element for element, neighbours in adjacency.items() if not neighbours}
        components = _components(adjacency)
        selected_entities = {
            entity for guid in selected_guids if (entity := self.entity_for_guid(guid)) is not None
        }
        selected_with_ports = selected_entities & owners
        selected_isolated = selected_with_ports & isolated
        selected_without_system = {
            entity for entity in selected_entities if not self.element_systems.get(entity)
        }
        return {
            "available": True,
            "ports": len(self.port_owner),
            "port_connections": connected_port_pairs,
            "elements_with_ports": len(owners),
            "elements_with_ports_without_connections": len(isolated),
            "elements_with_connections": len(owners - isolated),
            "network_components": len(components),
            "largest_component": max((len(component) for component in components), default=0),
            "selected_elements": len(selected_entities),
            "selected_with_ports": len(selected_with_ports),
            "selected_with_ports_without_connections": len(selected_isolated),
            "selected_without_ifc_system": len(selected_without_system),
        }

    def inventory(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for entity_type in self.entity_types.values():
            counts[entity_type] += 1
        return {
            "is_step_ifc": True,
            "entities": len(self.entity_types),
            "distribution_ports": counts.get("IFCDISTRIBUTIONPORT", 0),
            "port_connections": counts.get("IFCRELCONNECTSPORTS", 0),
            "systems": counts.get("IFCSYSTEM", 0) + counts.get("IFCDISTRIBUTIONSYSTEM", 0),
            "spatial_containment_relations": counts.get("IFCRELCONTAINEDINSPATIALSTRUCTURE", 0),
            "material_relations": counts.get("IFCRELASSOCIATESMATERIAL", 0),
        }


def _split_args(value: str) -> list[str]:
    output: list[str] = []
    start = 0
    depth = 0
    quoted = False
    index = 0
    while index < len(value):
        char = value[index]
        if char == "'":
            if quoted and index + 1 < len(value) and value[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif not quoted:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                output.append(value[start:index].strip())
                start = index + 1
        index += 1
    output.append(value[start:].strip())
    return output


def _refs(value: str) -> list[int]:
    return [int(item) for item in _REFERENCE.findall(value)]


def _single_ref(value: str) -> int | None:
    values = _refs(value)
    return values[0] if values else None


def _strings(value: str) -> list[str]:
    return [_decode_ifc_string(item.replace("''", "'")) for item in _STRING.findall(value)]


def _decode_ifc_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            return bytes.fromhex(match.group(1)).decode("utf-16-be")
        except (ValueError, UnicodeDecodeError):
            return match.group(0)

    return re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\", replace, value)


def _looks_like_global_id(value: str) -> bool:
    return len(value) == 22 and bool(re.fullmatch(r"[0-9A-Za-z_$]+", value))


def _numbers(value: str) -> list[float]:
    return [
        float(item)
        for item in re.findall(r"[-+]?\d+(?:\.\d*)?(?:E[-+]?\d+)?", value, flags=re.IGNORECASE)
    ]


def _si_length_unit(value: str) -> str:
    if ".CENTI." in value:
        return "cm"
    if ".MILLI." in value:
        return "mm"
    if ".DECI." in value:
        return "dm"
    return "m"


def _components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    unseen = set(adjacency)
    output: list[set[int]] = []
    while unseen:
        root = unseen.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, set()):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        output.append(component)
    return output
