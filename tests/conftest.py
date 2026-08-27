from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def sample_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "three-files"
    data_dir.mkdir()

    tree = {
        "data": {
            "type": "objects",
            "objects": [
                {
                    "objectid": 1,
                    "name": "Model",
                    "objects": [
                        {
                            "objectid": 10,
                            "name": "Lighting Devices",
                            "objects": [
                                {
                                    "objectid": 11,
                                    "name": "Alpha Switch",
                                    "objects": [
                                        {
                                            "objectid": 12,
                                            "name": "Single",
                                            "objects": [
                                                {"objectid": 13, "name": "Alpha Switch [1001]"},
                                                {"objectid": 14, "name": "Alpha Switch [1002]"},
                                            ],
                                        },
                                        {
                                            "objectid": 15,
                                            "name": "Double",
                                            "objects": [{"objectid": 16, "name": "Alpha Switch [1003]"}],
                                        },
                                    ],
                                }
                            ],
                        },
                        {
                            "objectid": 20,
                            "name": "Electrical Fixtures",
                            "objects": [
                                {
                                    "objectid": 21,
                                    "name": "Door Switch",
                                    "objects": [
                                        {
                                            "objectid": 22,
                                            "name": "Intercom",
                                            "objects": [{"objectid": 23, "name": "Door Switch [2001]"}],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "objectid": 30,
                            "name": "Electrical Equipment",
                            "objects": [
                                {
                                    "objectid": 31,
                                    "name": "Beta Switchboard",
                                    "objects": [
                                        {
                                            "objectid": 32,
                                            "name": "Panel",
                                            "objects": [
                                                {"objectid": 33, "name": "Beta Switchboard [3001]"},
                                                {"objectid": 34, "name": "Beta Switchboard [3002]"},
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "objectid": 40,
                            "name": "Pipes",
                            "objects": [
                                {
                                    "objectid": 41,
                                    "name": "Pipe Types",
                                    "objects": [
                                        {
                                            "objectid": 42,
                                            "name": "CW",
                                            "objects": [
                                                {"objectid": 43, "name": "Pipe [4001]"},
                                                {"objectid": 44, "name": "Pipe [4002]"},
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    }

    def record(object_id: int, name: str, *, type_name: str = "", level: str = "", length: str = "") -> dict:
        properties = {"Identity Data": {"Workset": "Workset1", "Type Name": type_name}}
        if level:
            properties["Constraints"] = {"Level": level}
        if length:
            properties["Dimensions"] = {"Length": length}
        # Deliberately reuse IfcGUID to verify externalId is the instance identity.
        properties["IFC Parameters"] = {"IfcGUID": "reused-type-guid"}
        return {
            "objectid": object_id,
            "name": name,
            "externalId": f"external-{object_id}",
            "properties": properties,
        }

    properties = {}

    def add_nodes(node: dict) -> None:
        object_id = node["objectid"]
        properties[str(object_id)] = record(object_id, node["name"])
        for child in node.get("objects", []):
            add_nodes(child)

    add_nodes(tree["data"]["objects"][0])
    for object_id, type_name, level in ((13, "Single", "GF"), (14, "Single", "GF"), (16, "Double", "B1")):
        properties[str(object_id)] = record(object_id, properties[str(object_id)]["name"], type_name=type_name, level=level)
    properties["23"] = record(23, "Door Switch [2001]", type_name="Intercom", level="GF")
    properties["33"] = record(33, "Beta Switchboard [3001]", type_name="Panel", level="GF")
    properties["34"] = record(34, "Beta Switchboard [3002]", type_name="Panel", level="B1")
    properties["43"] = record(43, "Pipe [4001]", type_name="CW", level="GF", length="2.5 m")
    properties["44"] = record(44, "Pipe [4002]", type_name="CW", level="GF", length="3.0 m")

    (data_dir / "model-tree.json").write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
    (data_dir / "model-properties.json").write_text(json.dumps(properties, ensure_ascii=False), encoding="utf-8")

    database_path = data_dir / "model.ifc"
    with sqlite3.connect(database_path) as database:
        database.executescript(
            """
            CREATE TABLE _objects_attr(id INTEGER PRIMARY KEY, name TEXT, category TEXT);
            CREATE TABLE _objects_val(id INTEGER PRIMARY KEY, value BLOB UNIQUE);
            CREATE TABLE _objects_id(id INTEGER PRIMARY KEY, external_id BLOB, viewable_id BLOB);
            CREATE TABLE _objects_eav(id INTEGER PRIMARY KEY, entity_id INTEGER, attribute_id INTEGER, value_id INTEGER);
            INSERT INTO _objects_attr VALUES (1, 'name', '__name__');
            INSERT INTO _objects_val VALUES (1, 'Water Flow Alarm Switch [9001]');
            INSERT INTO _objects_id VALUES (9001, 'style-9001', NULL);
            INSERT INTO _objects_eav VALUES (1, 9001, 1, 1);
            """
        )
    return data_dir

