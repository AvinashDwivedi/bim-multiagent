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

The application exposes generic read-only project capabilities plus model-backed analysis helpers:

- `inspect_project`: source metadata, raw tree roots, record count, and property keys.
- `list_tree_children`: arbitrary hierarchy traversal.
- `search_records`: raw name, path, property-key, and property-value search.
- `get_records`: exact raw records and properties by object ID.
- `aggregate_records`: count, distinct count, sum, average, minimum, maximum, grouping, and unit conversion.
- `search_ifc`: raw STEP IFC text search.
- `calculate`: safe arithmetic.
- `describe_bim_workspace` / `query_bim_workspace`: model-authored, read-only SQL over normalized records,
  properties, arbitrary-depth hierarchy, named IFC objects and relationships, and record-to-IFC identity candidates.
- `analyze_ifc_geometry`: model-selected IfcOpenShell analysis for world placement, bounding boxes, solid volume,
  surface area, projected XY area, and spatial containers.
- `rank_ifc_geometry`: complete-population mesh ranking by volume, area, or an explicit X/Y/Z extent.
- `reconcile_populations`: identity/count reconciliation for model-selected populations, overlaps, duplicates, and missing IDs.
- `analyze_ifc_graph`: direction- and role-aware traversal of named IFC relationships with evidence paths.
- `explore_object_scope`: a model-backed hierarchy explorer that proposes competing scopes without fixed-depth rules.
- `review_scope_and_evidence`: a model-backed critic that challenges omissions, duplicates, identities, conflicting
  evidence, exclusions, and unreconciled totals without consulting expected answers.
- `research_standards`: model-directed web research for external codes and standards, kept separate from project facts.
- `code_interpreter`: model-written Python in an isolated hosted container with networking disabled. The container
  receives the three raw project files plus `bim_workspace.sqlite`, including normalized hierarchy, IFC identity,
  relationships, and a full-product geometry inventory for dataframe, graph, ranking, and reconciliation work.

On GPT-5.6 models, Programmatic Tool Calling is also enabled. The model may write a short hosted JavaScript program
to coordinate predictable project-tool calls for filtering, joining, deduplication, aggregation, and evidence
compression. Adaptive scope choices and the final answer remain direct model decisions.

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
BIM_MODEL=gpt-5.6-sol
BIM_REASONING_EFFORT=medium
BIM_DATA_DIR=test-project-data
BIM_PROJECTS_ROOT=
BIM_TRACE_DIR=logs/traces
BIM_MAX_AGENT_ITERATIONS=40
BIM_MAX_TOOL_OUTPUT_CHARS=80000
BIM_ENABLE_HOSTED_PYTHON=false
BIM_PYTHON_MEMORY_LIMIT=4g
BIM_PYTHON_EXPIRY_MINUTES=120
BIM_OPENAI_MAX_RETRIES=4
BIM_OPENAI_TIMEOUT_SECONDS=180
```

Hosted Python is opt-in because it uploads the selected project's three source files and a generated analysis snapshot to an OpenAI
Code Interpreter container. The container has outbound networking disabled, a configured memory cap, and expires
after inactivity. Set `BIM_ENABLE_HOSTED_PYTHON=true` only when project policy permits this upload; the local
read-only SQL and IFC geometry tools remain available when it is disabled.

The workspace is built lazily for the selected project. The agent asks for its schema and writes its own SQL for
the current question; there are no question-specific queries, expected answers, or semantic mappings. Only one
`SELECT`, `WITH`, or `EXPLAIN QUERY PLAN` statement is accepted, mutations and external database attachment are
denied, results are row-limited, and long-running queries are interrupted. STEP IFC files expose raw
`ifc_entities`/`ifc_references` plus named `ifc_objects`, role-aware `ifc_relationships`, and
`record_ifc_candidates`. SQLite IFC exports are attached read-only as `ifc_source`. Exact geometry is delegated
to IfcOpenShell rather than inferred from names or evaluator cases.

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
Retryable upstream failures are returned as HTTP 503 with a structured `retryable` flag. The health response
reports the loaded hosted-Python configuration; containers are still created lazily on a question that needs one.

## Tests

Tests mock model responses and verify strict tool schemas, caller transport, the model-controlled review path,
IFC/workspace mechanics, and the generic function-calling loop without creating a deterministic semantic fallback.

```powershell
pytest
```
