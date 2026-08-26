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
   Cypher, runs deterministic semantic-adequacy checks, checks completion, curates reusable mappings,
   and renders the answer.

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

Each package now closes its own bounded evidence loop before commit: explore, execute, replay, check semantic
adequacy, and revise. A rejected query remains auditable, while its failed check names and explanations are fed
back to the isolated specialist. Partial governed routes are checkpoints rather than terminal answers: the
Schema Scout reopens targeted exploration for missing outputs. Dependent packages receive a compact immutable
bundle containing only replay-verified claims from their declared dependencies.

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

Replay alone is not proof that a query answered the intended question. Each output carries typed semantic
intent: entity grain, measurement basis, population boundary, planned-versus-actual origin, absence meaning,
and requested projection. A deterministic semantic challenger compares that contract with the registered
mapping, executed plan, population coverage, and returned claim. Entity/aggregate swaps, wrong measurement
bases, incomplete populations, planned values presented as actual, and related-but-wrong projections fail
verification even when the Cypher result is stable.

Absence is explicit. The system distinguishes a missing population, an existing population with an
unpopulated property, a replay-verified zero, and unsupported evidence. Distinct/grouped queries measure the
candidate population separately, so an empty property column cannot be reported as if no BIM entities exist.

At run start, deterministic code builds one compact scoped model profile containing label/property surfaces
and relationship patterns. Isolated scouts share this immutable routing profile instead of rediscovering the
same graph in separate conversations. Compatible profiles persist in the operational SQLite store and are
keyed by project, authorized-source set, graph-contract version, and a schema fingerprint that includes both
node properties and relationship topology. It is explicitly a cache hint: exact bindings, current values, counts,
and absence findings are always executed and replayed live.

Named project calculations run through a frozen code-owned recipe registry. Current primitives support
composite grouped populations, exact inclusion/exclusion boundaries, nullable group buckets, scoped
relationship fallbacks, stable-identity source joins, and baseline/subscope comparisons. Project YAML may
select a recipe and its identifiers/semantics, but cannot store answer values or executable expressions.

Registered mappings may expose fixed, live-validated relationship paths. Plans can measure exhaustive
`relationship_coverage` or apply compiler-owned `exists`/`is_missing` relationship predicates; models cannot
invent relationship names or variable-length paths. Every intermediate node remains inside authorized scope.
If ingestion omitted source provenance or a containment/connectivity edge, the system reports the unresolved
population instead of converting a missing property into a physical-disconnection claim.

Exact query results are cached only within their isolated run/workstream. The cache key covers the fully
defaulted plan, package constraint bindings, source scope, active mapping, schema fingerprint, and contract
version. Verification bypasses that cache and independently replays the plan. Equivalent learned mappings use
a stable executable-semantics digest, so repeated discovery merges aliases/evidence rather than growing
duplicate knowledge records.

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
insufficient-evidence, zero-match, diagnostic-bearing, semantically inadequate, or supporting-only runs cannot
promote knowledge. Promotion also requires a ready project-scoped completion, matching provenance,
package/output ownership, an active mapping, and answer-producing replay evidence.

Governed project mappings are a faster bootstrap for known difficult schemas. A scout can select only a
server-configured knowledge key; deterministic code then verifies the configured label, authorization and
identity properties, exact category/family boundary, numeric fields and units, optional missing-data fields,
and live value population before registering it. The
configuration stores semantics, never counts or answer values, and fails closed when the live schema changes.

The active registry uses coding-agent-style specialists with small contexts:

```text
User question
  -> Task Architect (typed package DAG only)
  -> governed deterministic route, when available (zero worker-model calls)
  -> otherwise Schema Mapping Specialist (one package, discovery tools only)
  -> exactly one terminal specialist:
       Quantity | Relationship | Geometry | Requirements
  -> deterministic replay, completion gates, ordered merge, and answer assembly
```

Each terminal specialist receives only its package objective, projected constraints, typed outputs, and compact
schema handoff. It does not receive the root question, sibling packages, discovery transcript, credentials, or
sibling evidence. Schema and execution phases also use distinct mutable run contexts. Replay verification,
guarded knowledge curation, scheduling, and merging remain deterministic runtime responsibilities rather than
additional model-agent hops.

Typed specialist handoffs have a bounded repair loop. If a model returns malformed completion JSON after it
already executed a valid answer query, the package-local evidence ledger acts as a durable checkpoint and
validated evidence is recovered instead of discarded. Partial packages retain only outputs they genuinely
satisfied and expose the remaining limitations. Reports include `workstream_diagnostics` with specialist,
status, attempts, typed-output failures, recovery strategy, and required/satisfied outputs, allowing evaluator
runs to identify weak agents without private model traces.

Parallelism is runtime-managed rather than model-issued. Results merge in architect order with validate-then-
commit semantics; mapping-only workstreams, conflicting mappings/evidence, invalid ownership, and unexpected
mapped zeroes are rejected without partially mutating authoritative state. Model-level parallel tool calls
remain disabled.

Discovery uses distinct live family/signature populations rather than arbitrary first element IDs. The
complete observed property surface remains available to deterministic search, while the agent-facing node
inventory is compact. A bounded hierarchy profiler exposes exact category → family → type branches and
their record counts, allowing unrelated families inside broad Revit categories to be rejected before an
answer query is built. Multilingual ontology expansion plus hybrid lexical/local-vector ranking maps
user-language concepts to model-language Revit/IFC vocabulary without asserting that a candidate is
correct; exact value binding, live mapping registration, and replay remain mandatory.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Configure Claude Code, the Neo4j connection, client, and project in `.env`:

```dotenv
ANTHROPIC_API_KEY=your-anthropic-api-key
NEO4J_URI=neo4j+s://your-instance
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
BIM_CLIENT_ID=your-client-uuid
BIM_PROJECT_ID=your-project-uuid
BIM_AGENT_MODEL=claude-sonnet-4-6
BIM_AGENT_WORKER_MODEL=claude-sonnet-4-6
```

The Python Claude Agent SDK includes the Claude Code runtime. Every specialist runs with only its
explicit in-process BIM tools; filesystem, shell, web, skills, and ambient MCP configuration are disabled.
Semantic candidate ranking is local and deterministic, so no second model-provider key is required.

The Task Architect defaults to two turns, the Schema Scout to 18, and the fresh Query Worker to 6. Override
them with `BIM_ARCHITECT_MAX_TURNS`, `BIM_SCOUT_MAX_TURNS`, and `BIM_QUERY_WORKER_MAX_TURNS`. Invalid architect
contracts and architect turn limits receive a bounded fresh retry. Query evidence rejected by replay or semantic
verification receives one bounded correction by default; configure this with
`BIM_VERIFICATION_REPAIR_ATTEMPTS`. The split prevents a long discovery transcript from consuming
query-planning context and ensures a successful mapping gets a fresh chance to produce evidence.

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

### Privacy-safe evaluator analysis

Use the built-in analyzer to inspect an evaluator JSON report without emitting
case content. It returns aggregate outcomes, pipeline statuses, failure
categories, semantic-check results, implicated stages/specialists, and (when a
baseline is supplied) anonymous regression deltas. Questions, expected or actual
answers, answer values, explanations, and scope identifiers are never included
in the output.

```powershell
python -m bim_agents.evaluation_analysis latest-report.json
python -m bim_agents.evaluation_analysis latest-report.json --baseline previous-report.json
```

The same functionality is available as `analyze_report(report)` and
`compare_reports(current, baseline)` from `bim_agents.evaluation_analysis`.
