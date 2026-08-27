# Model-Directed BIM Agent

A single OpenAI Responses API agent for answering questions over Autodesk-style three-file BIM exports.
The model owns planning, tool selection, iteration, interpretation, checking, and final answer composition.

## Architecture

```text
while not finished:
    response = model(question, observations, tools)
    if response calls tools:
        observations += execute(tool calls)
    else:
        return response
```

There is no heuristic planner, typed query planner, fixed executor, replay verifier, answer template,
supervisor, subagent, fixed stage order, or offline fallback.

The application exposes only generic read-only capabilities:

- `inspect_project`: source metadata, raw tree roots, record count, and property keys.
- `list_tree_children`: arbitrary hierarchy traversal.
- `search_records`: raw name, path, property-key, and property-value search.
- `get_records`: exact raw records and properties by object ID.
- `aggregate_records`: count, distinct count, sum, average, minimum, maximum, grouping, and unit conversion.
- `search_ifc`: raw STEP IFC text search.
- `calculate`: safe arithmetic.

The model decides which tools are relevant, supplies their arguments, observes their output, changes direction
when needed, and decides when the answer is complete. Python performs only file parsing, requested search and
arithmetic operations, API transport, tracing, output-size limits, and the iteration safety bound.

## Input contract

Each project directory must contain exactly:

```text
project-data/
├── <model-id>-tree.json
├── <model-id>-properties.json
└── <database-id>.ifc
```

The tree and properties filenames must use the same model ID.

## Setup

```powershell
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

`OPENAI_API_KEY` is required. There is intentionally no deterministic or offline answer path.

```dotenv
OPENAI_API_KEY=
BIM_MODEL=gpt-5.4
BIM_REASONING_EFFORT=medium
BIM_DATA_DIR=test-project-data
BIM_PROJECTS_ROOT=
BIM_TRACE_DIR=logs/traces
BIM_MAX_AGENT_ITERATIONS=20
BIM_MAX_TOOL_OUTPUT_CHARS=80000
```

For multiple projects, `BIM_PROJECTS_ROOT/<project_id>/` must contain the same three-file contract.

## CLI

```powershell
python -m bim_agent --data-dir test-project-data inspect --json
python -m bim_agent --data-dir test-project-data ask "כמה מפסקים בפרוייקט?"
python -m bim_agent --data-dir test-project-data chat
```

## Server

```powershell
python -m bim_agent --data-dir test-project-data serve --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /api/health`
- `GET /api/inspect`
- `POST /api/ask` with `{"question":"..."}`
- Evaluator compatibility: `POST /api/chat` and `POST /api/chat/stream`

Every answer includes source hashes, model response IDs, tool-call telemetry, and a request-scoped JSONL trace.
It does not claim independent verification because no deterministic verifier remains.

## Tests

Tests mock model responses and verify the function-calling loop and generic tool mechanics without creating a
deterministic semantic fallback.

```powershell
pytest
```
