# Read-only BIM Investigation Agent

A coding-agent-style, read-only system for investigating one authorized BIM project.

## Flow

1. The Task Architect creates one typed contract and a small evidence-work-package DAG.
2. A deterministic scheduler launches ready packages concurrently (three by default).
3. Every package gets its own BIM connection, mutable state, budgets, and causal workstream ID.
4. Exact project-governed calculations are selected deterministically before exploration; otherwise a
   Schema Scout resolves a trusted contract, governed mapping, learned mapping, or live mapping.
5. A fresh Query Worker receives only the package and compact typed scout handoff, then submits the
   smallest declarative answer query. It has no discovery or mapping tools.
6. The runtime validates each staged delta, atomically commits non-conflicting evidence, replays scoped
   Cypher, checks completion, curates reusable mappings, and renders the answer.

This is a coding-agent-style architecture: models are isolated workers while runtime code owns scheduling,
authorization, budgets, commits, dependencies, and stopping. Independent packages run concurrently;
dependent packages start only after their prerequisites commit.

Every response includes an ordered `investigation_trace` built from trusted runtime artifacts. It records
the inspection, mapping, query, and verification milestones without exposing prompts, credentials, or
private model reasoning.

Before a report leaves the runtime, every user-visible field is recursively redacted against the
configured scope identifiers, database URI and username/password, and API key. The health endpoint also
returns operational counts only; it does not expose scope identifiers.

The agent has no shell, filesystem-write, or raw database-query tool. It never supplies executable
Cypher. The Cypher Query Handler owns compilation and rejects mutating or unscoped statements.

Exploration happens through bounded metadata, property-profile, and semantic-value tools. Queries that
contribute to the answer are retained and verified together; the agent cannot silently discard a
contradictory result. A database account with native read-only privileges remains the final security
boundary.

The model roles exchange strict typed contracts and compact handoffs; trusted runtime stages retain
`TaskContract`, `EvidenceWorkPackage`, `EvidenceHandoff`, `GraphDiscovery`, `SchemaMapping`, `QueryPlan`,
`CypherQuery`, `QueryResult`, and `VerificationReport` artifacts. Raw scout exploration never enters the
Query Worker conversation or another workstream.

## Multiagent kernel

The Task Architect partitions required outputs exactly once into one to four packages (six is the schema
limit). Constraints are owned by packages, so a ground-floor filter cannot leak into a sibling whole-building
total. Outputs sharing an entity, filters, grouping dimensions, and metric stay together. Modelled facts,
requirements, and unrelated geometry can be independent packages. `BIM_MAX_PARALLEL_WORKERS` controls the
runtime concurrency limit.

Within one package, discovery and execution are intentionally split across fresh conversations. The Schema
Scout can inspect catalogs and the live graph and may register a mapping, but cannot execute an answer query.
The Query Worker can execute trusted declarative plans and named geometry calculations, but cannot discover
or mutate mappings. The typed `EvidenceHandoff` is their only conversational bridge; registered mappings
remain available in the package's isolated artifact ledger.

Answer-producing queries must name a stable `answer_key`, declare the exact package outputs they satisfy,
and include every task constraint as a real query filter. The runtime rejects a ground-floor claim backed by
an unfiltered project-wide query. Diagnostic `is_missing` filters remain non-rendered supporting evidence;
the first-class `coverage` operation may answer an absence/completeness question only after scanning the full
scoped candidate population and reporting populated and missing denominators. Zero matches, missing metrics,
and changed values for one answer key are diagnostics.

The runtime independently checks scope, entity evidence, physical identity, classification, units,
required-output coverage, contradictions, and replay stability. Replay-verified facts lead the response even
when secondary completion gates remain open; unresolved checks follow as caveats. Task complexity adjusts
model/tool budgets without storing project-specific numeric answers.

At run start, deterministic code builds one compact scoped model profile containing label/property surfaces
and relationship patterns. Isolated scouts share this immutable routing profile instead of rediscovering the
same graph in separate conversations. It is explicitly a cache hint: exact bindings, current values, counts,
and absence findings are always executed and replayed live.

Numeric mappings retain `source_unit`, canonical `unit`, `conversion_factor`, conversion basis, and source
property. Aggregation and numeric filters convert exactly once at the compiler boundary. Claims expose typed
population coverage, measurement provenance, method, source tags, per-claim caveats and plausibility flags;
reports also expose a terminal status for every required output. Impossible invariant violations (non-finite
or negative physical measurements and percentages outside 0–100) block verification, while unusual but
possible values can remain visible as warnings.

## Guarded project learning

The runtime maintains a separate learned-knowledge database at
`storage/learned_knowledge.sqlite3` (override with `BIM_LEARNED_KNOWLEDGE_DB`). The BIM graph remains
read-only. Set `BIM_LEARNING_ENABLED=0` to disable learning.

Registered live schema mappings and reusable interpretation rules can be saved as project-scoped
candidates. Only schema mappings used by answer-producing evidence that passes deterministic replay are
promoted automatically. Measurement and calculation rules may reach `verified` but require an explicit
governance step before promotion. Promoted mappings
are revalidated against the live identity population and property surface before reuse. A fingerprint of
the authorized live schema and graph-contract version prevents mappings from crossing schema revisions;
incompatible promoted mappings are deprecated automatically.

Knowledge records may contain semantic field/property mappings, exact classification values, aliases,
identity rules, relationship paths, units, and calculation semantics. They cannot contain query rows,
claims, counts, measurements, expected answers, or other answer values. Client/global promotion is not
available to the autonomous runtime: learned knowledge stays inside its originating project.

Exact stored value bindings bypass linguistic canonicalization. Promoted mappings are activated only when
their labels, properties, identities, and exact bound values still exist. Timed-out, cancelled, failed,
insufficient-evidence, zero-match, or diagnostic-bearing runs cannot promote knowledge.

Governed project mappings are a faster bootstrap for known difficult schemas. A scout can select only a
server-configured knowledge key; deterministic code then verifies the configured label, authorization and
identity properties, exact category/family boundary, numeric fields and units, optional missing-data fields,
and live value population before registering it. The
configuration stores semantics, never counts or answer values, and fails closed when the live schema changes.

The active registry contains a Task Architect, BIM Schema Scout, and BIM Query Worker. Replay verification,
completion gates, guarded knowledge curation, DAG scheduling, and merging are deterministic runtime
responsibilities rather than model-agent hops.

Parallelism is runtime-managed rather than model-issued. Results merge in architect order with validate-then-
commit semantics; mapping-only workstreams, conflicting mappings/evidence, invalid ownership, and unexpected
mapped zeroes are rejected without partially mutating authoritative state. Model-level parallel tool calls
remain disabled.

Discovery uses distinct live family/signature populations rather than arbitrary first element IDs. The
complete observed property surface remains available to deterministic search, while the agent-facing node
inventory is compact. A bounded hierarchy profiler exposes exact category → family → type branches and
their record counts, allowing unrelated families inside broad Revit categories to be rejected before an
answer query is built. Multilingual ontology expansion plus hybrid lexical/embedding ranking maps
user-language concepts to model-language Revit/IFC vocabulary without asserting that a candidate is
correct; exact value binding, live mapping registration, and replay remain mandatory.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Configure the Neo4j connection, OpenAI API key, client, and project using the environment variables
consumed by `Settings.from_env()`.

The Schema Scout defaults to 18 turns and the fresh Query Worker to 6. Override them with
`BIM_SCOUT_MAX_TURNS` and `BIM_QUERY_WORKER_MAX_TURNS`. The split prevents a long discovery transcript from
consuming query-planning context and ensures a successful mapping gets a fresh chance to produce evidence.

Task complexity may raise the available work budget for complex investigations, but it never reduces the
configured model-call or tool-call limits. A simple question can still require unknown-schema discovery,
exact classification, planning, and replay.

The default public timeout is ten minutes. By default, the model-work phase stops 30 seconds earlier so
deterministic replay, completion gates, guarded curation, report serialization, and the HTTP response can
finish before the public deadline. Override the reserve with `BIM_FINALIZATION_RESERVE_SECONDS`, or other
limits with `BIM_RUN_TIMEOUT_SECONDS`, `BIM_MAX_LLM_CALLS`, and `BIM_MAX_TOOL_CALLS`.

## CLI

```powershell
python -m bim_agents.cli "How many apartments are on the ground floor?"
```

For request-scoped execution, pass both scope UUIDs explicitly. These values select authorization,
graph-schema overlays, and project knowledge for this request instead of using the `.env` scope:

```powershell
python -m bim_agents.cli "How many apartments are there?" `
  --client-id 653fbe80-e4c5-11ed-95e8-fdb8a484b2c4 `
  --project-id 858ef0f0-454a-11f1-8957-1fe1b101e373
```

## FastAPI backend

```powershell
python -m bim_agents.webapp
```

This repository serves only the system-under-test API. Interactive chat and evaluation pages live in
the independent `bim-evaluator` repository. Every chat request must send `question`, `client_id`, and
`project_id`; project scope is not inferred from the backend `.env`.

OpenAPI documentation is at `http://127.0.0.1:8000/docs`. Set `BIM_CORS_ORIGINS` when the evaluator UI
is served from an origin other than the default `http://127.0.0.1:8090`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Independent evaluation

Evaluation code, datasets, reports, chat UI, and the evaluation dashboard live in the separate sibling
`bim-evaluator` repository. Start its server with `python -m server`, then open
`http://127.0.0.1:8090/evaluation`. The examiner calls this backend only through its public HTTP API.
