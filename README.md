# Local BIM Agent

An evidence-first question-answering service for Autodesk-style three-file BIM exports. It reads the files locally, discovers project vocabulary at runtime, turns a natural-language question into a typed query plan, executes that plan deterministically, audits nearby interpretations, and replay-verifies the result.

## Input contract

Place exactly these three files in one directory:

```text
project-data/
├── <model-id>-tree.json
├── <model-id>-properties.json
└── <database-id>.ifc
```

- The tree JSON supplies `Model → Category → Family → Type → Instance` hierarchy and object IDs.
- The properties JSON is an object-ID keyed record map.
- The `.ifc` may be either a SQLite export database or an ISO-10303-21 STEP IFC. SQLite sources are searched read-only for hidden metadata. STEP IFC sources are parsed read-only for spatial containment, systems, material associations, distribution ports, and port-to-port connectivity.

The tree and properties filenames must share the same model ID. The application refuses ambiguous or incomplete directories.

## How an answer is produced

```text
File Inspector
  → Vocabulary Profiler
  → Semantic Planner
  → Deterministic Investigator
  → Counterexample Auditor
  → Replay Verifier
  → Answer Composer
```

1. Recursively index the tree and join nodes to property records by `objectid`.
2. Identify physical instance leaves and exclude project metadata, styles, templates, legends, analytical records, and imported DWG records.
3. Profile categories, families, types, levels, properties, and their counts.
4. Ask the optional LLM planner for a strict JSON query plan containing executable population branches, property projections, grouping dimensions, measurements, and connectivity requirements.
5. Apply exact category/family/type/property boundaries locally and count stable element identities. In this format, `externalId` is preferred because `IfcGUID` may contain a reused type GUID.
6. Execute grouped, unit-normalized measurements and explicit missing-property analysis. STEP IFC containment fills missing levels, while IFC ports/systems support continuity audits.
7. Search outside the primary boundary for ranked related interpretations and audit SQLite EAV metadata when present.
8. Verify that every requested answer facet actually executed. If verification finds a missing facet, the LLM planner may refine the plan once and the deterministic calculation is replayed.
9. Return the answer, breakdown, related candidates, limitations, source hashes, and a request-scoped JSONL trace.

This design follows the OpenAI Responses API’s support for structured JSON output and custom-code workflows; see the [official Responses API documentation](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

## Setup

```powershell
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to enable the semantic LLM planner. Without a key, the conservative bilingual local planner remains available.

```dotenv
OPENAI_API_KEY=
BIM_MODEL=gpt-5.4
BIM_REASONING_EFFORT=medium
BIM_USE_LLM=true
BIM_DATA_DIR=test-project-data
BIM_PROJECTS_ROOT=
BIM_TRACE_DIR=logs/traces
```

For multiple projects, set `BIM_PROJECTS_ROOT` to a directory whose immediate child directories are
project IDs. Each project directory must contain the same three-file contract. HTTP requests then resolve
`project_id` to that directory, and file size/mtime fingerprints automatically invalidate the cached agent
when any source file changes. Without `BIM_PROJECTS_ROOT`, the server intentionally uses `BIM_DATA_DIR`.

## CLI

Inspect the discovered project vocabulary:

```powershell
python -m bim_agent --data-dir test-project-data inspect --json
```

Ask questions:

```powershell
python -m bim_agent --data-dir test-project-data ask "כמה מפסקים בפרויקט?"
python -m bim_agent --data-dir test-project-data ask "how many pipes are on B1?"
python -m bim_agent --data-dir test-project-data ask "what types of sprinklers are present?"
python -m bim_agent --data-dir test-project-data ask "what is the total pipe length?" --json
```

Open an interactive session that loads the model once and accepts repeated questions:

```powershell
python -m bim_agent --data-dir test-project-data chat
```

Force the offline planner:

```powershell
python -m bim_agent --no-llm --data-dir test-project-data ask "how many pipes?"
```

## HTTP API

```powershell
python -m bim_agent --data-dir test-project-data serve --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /api/health`
- `GET /api/inspect`
- `POST /api/ask` with `{"question": "..."}`

The evaluator compatibility facade also exposes `POST /api/chat`, `POST /api/chat/stream`, and the legacy public CLI:

```powershell
python -m bim_agents.cli "how many pipes?" --client-id <id> --project-id <id> --quiet
python -m bim_agents.webapp
```

Example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/ask `
  -ContentType 'application/json' `
  -Body '{"question":"how many sprinklers are in the project?"}'
```

## Supported question shapes

- Counts of physical modeled instances
- Lists and grouped counts by category, family, type, or level
- Exact property filters and level constraints
- Numeric sums such as length, area, volume, or cost when compatible populated units exist
- Composite populations expressed as independent category/family/property branches
- Arbitrary property projection, missing-value reporting, and property-based grouping
- Unit-normalized grouped measurements such as length by type, width, and height
- STEP IFC spatial containment, system membership, distribution-port, and continuity analysis
- Bilingual Hebrew/English terminology
- Related-category disclosure for ambiguous concepts

The LLM planner makes unfamiliar and project-specific terminology much more flexible. Execution remains constrained to the supplied data and typed operations.

## Safety and limitations

- All source files and SQLite connections are read-only.
- The LLM receives a compact vocabulary profile, not API credentials and not the full model database.
- No model-generated Python, SQL, or filesystem command is executed.
- A zero in one supplied model is not asserted as a full-project zero. Every response states that the three files cannot prove all disciplines were supplied.
- Mixed-unit or missing numeric measurements are returned as limited rather than silently summed.
- Each response records SHA-256 source hashes and a JSONL trace under `BIM_TRACE_DIR`.

## Tests

```powershell
pytest
```

The tests construct synthetic three-file and STEP IFC models and cover recursive traversal, physical/type separation, SQLite metadata auditing, bilingual semantic mapping, composite populations, property projection, grouped numeric aggregation, IFC containment/connectivity, semantic facet verification, API behavior, and trace creation.
