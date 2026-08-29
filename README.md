# Model-Directed BIM Agent

A single OpenAI Responses API agent for answering questions over Autodesk-style three-file BIM exports.
The model owns planning, tool selection, iteration, interpretation, checking, and final answer composition.

## Architecture

```text
while not finished:
    response = model(question, observations, tools)
    if response calls tools:
        observations += execute_or_reuse(tool calls)
        observations += automatic_population_reconciliation
    else:
        completeness_gate(response)
        citation_grounding_gate(response)
        return response
```

There is no heuristic planner, typed query planner, answer template, supervisor, subagent, fixed stage order,
or offline answer fallback. Termination is constrained by deterministic completeness and citation-grounding checks.

The application exposes generic read-only project capabilities plus model-backed analysis helpers:

- `inspect_project`: source metadata, raw tree roots, record count, and property keys.
- `list_tree_children`: cursor-paginated arbitrary hierarchy traversal.
- `search_records`: cursor-paginated raw name, path, property-key, and property-value search.
- `get_records`: exact raw records and properties by object ID.
- `search_ifc`: cursor-paginated raw STEP IFC text search.
- `fetch_more`: resumes any cursor returned by the pageable tree/record/IFC tools.
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
- `run_local_python`: model-written Python in an on-device Docker sandbox. Networking is disabled, project inputs
  and the normalized workspace are mounted read-only, the container root filesystem is read-only, and only an
  ephemeral `/tmp` is writable. No OpenAI Containers or Files API is used.

On GPT-5.6 models, Programmatic Tool Calling is also enabled. The model may write a short hosted JavaScript program
to coordinate predictable project-tool calls for filtering, joining, deduplication, aggregation, and evidence
compression. Adaptive scope choices and the final answer remain direct model decisions.

The model decides which tools are relevant, supplies their arguments, observes their output, and changes direction
when needed. Python additionally enforces per-run call deduplication, automatic population reconciliation,
one-shot completeness continuation, unresolved-page checks, and claim-level `[ref: call_id]` validation. Numeric
citations are accepted only when the cited observation contains the claimed value. Routing rules live in the
versioned `bim_agent/skills/bim-routing/SKILL.md`; ordinary project filters, joins, and aggregates are routed to SQL.
The older `aggregate_records` executor remains callable only through the internal Python API for compatibility; it
is not exposed to model runs.

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
BIM_MAX_ANSWER_COST_USD=0
BIM_ENABLE_LOCAL_PYTHON=true
BIM_LOCAL_PYTHON_IMAGE=python:3.12-slim
BIM_PYTHON_MEMORY_LIMIT=4g
BIM_LOCAL_PYTHON_CPUS=1.0
BIM_LOCAL_PYTHON_TIMEOUT_SECONDS=120
BIM_LOCAL_PYTHON_OUTPUT_CHARS=40000
BIM_OPENAI_MAX_RETRIES=4
BIM_OPENAI_TIMEOUT_SECONDS=180
BIM_MODEL_PRICING_JSON=
```

Python analysis runs only on the device through Docker. Install Docker Desktop (or another compatible local Docker
engine), start it, and install the configured image once:

```powershell
docker pull python:3.12-slim
```

The application never pulls an image automatically and never sends the raw BIM files through OpenAI's Containers
or Files APIs. At execution time it mounts the selected project at `/project` and the generated SQLite workspace at
`/workspace`, both read-only. The container uses `--network none`, a read-only root filesystem, dropped Linux
capabilities, `no-new-privileges`, a non-root user, CPU/memory/PID caps, an execution timeout, bounded output, and an
ephemeral `/tmp`. Configure a preinstalled custom image with `BIM_LOCAL_PYTHON_IMAGE` if analyses require packages
beyond the Python standard library.

The question and compact tool observations still go to the configured OpenAI model because this is a Responses API
agent. Raw BIM source files remain on the device and are never uploaded by the Python integration.

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

Every answer includes source hashes, model response IDs, tool-call telemetry, and a request-scoped JSONL audit trace.
The trace carries a session ID and the full model/tool transcript (with secret-key redaction), including arguments,
cached-call markers, automatic reconciliation, and untrimmed tool observations.

Every report also includes `cost`, which totals token usage across all Responses API calls needed for that answer,
including nested scope exploration, evidence review, and standards research. It reports USD list-price estimates,
cached/uncached input tokens, output tokens, per-request breakdowns, pricing date/source, and whether the estimate is
complete. GPT-5.4 and GPT-5.6 Sol defaults follow their official model pages. Unknown/private model prices can be
supplied with `BIM_MODEL_PRICING_JSON`; tool-specific fees such as web search are listed as excluded rather than
silently reported as zero. Local Docker execution has no OpenAI tool fee.
`BIM_MAX_ANSWER_COST_USD` is an opt-in guard. It defaults to `0` (disabled) while real question-cost distributions
are collected; set a positive amount only after choosing a threshold from observed workloads. When enabled, it stops
further investigation once the running list-price estimate reaches the configured amount. It then preserves an
already-produced answer or permits one bounded, tool-disabled best-effort finalization from accumulated evidence and
appends the canonical budget notice. That finalization is included in the cost report. The completed request may exceed
the threshold because usage is available only after a request returns and the terminal synthesis is intentionally
allowed after the trigger. Stable prompt-cache keys and compact model-facing observations reduce repeated input.
Retryable upstream failures are returned as HTTP 503 with a structured `retryable` flag. The health response
reports whether the local Docker runtime and configured image are ready. A local container is created only if the
model calls `run_local_python`, and it is removed immediately after the call.

## Tests

Tests mock model responses and verify strict tool schemas, caller transport, the model-controlled review path,
IFC/workspace mechanics, and the generic function-calling loop without creating a deterministic semantic fallback.

```powershell
pytest
```
