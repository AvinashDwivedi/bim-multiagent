from __future__ import annotations

import json
import os

from bim_context import BimContext, Settings

from .graph_contract import load_graph_contract, validate_live_schema


def main() -> None:
    bim = BimContext(Settings.from_env())
    try:
        bim.connect()
        contract = load_graph_contract(os.getenv("BIM_GRAPH_SCHEMA_PATH") or None)
        report = validate_live_schema(bim, contract, use_cache=False)
        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    finally:
        bim.close()


if __name__ == "__main__":
    main()
