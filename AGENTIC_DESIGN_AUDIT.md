# Current Agentic Design and Audit Analysis

Date: 2026-08-30  
Scope: the implementation in this repository after the structural hardening prompted by the audit in
`C:\Users\avina\Downloads\bim-evaluator`.

## Executive verdict

The system is now a **schema-aware, model-directed, read-only BIM investigation agent with deterministic
execution and evidence gates**. It is not a general-purpose autonomous multi-agent swarm, and it is not yet a
system that can truthfully answer *any question over any BIM project*.

Its strongest use case is a bounded question over a project that follows the current three-file input contract,
where the requested population, metric, unit, and source can be represented by the typed interpretation plan and
proved by the supported SQL, record, or IFC tools. In that zone, the design now has unusually strong safeguards
against silent scope changes, arbitrary row selection, lexical citation matching, incomplete pagination, and
unreported cross-source disagreement.

Its weakest use cases remain new input formats, questions needing a project-specific ontology, typed comparisons,
open-ended connectivity, normative compliance, certified solid volume, top-N rankings with ties, very large exhaustive
narratives, and workloads requiring adaptive decomposition or cost allocation. Those cases now tend to clarify,
withhold, or report a limitation rather than invent a result. That is a correctness improvement, but it is also a
capability boundary.

The old evaluation report is a useful failure baseline, not proof of the hardened system's current accuracy. No
new multi-project semantic evaluation has been run after these changes.

Local verification of the current working tree: `python -m pytest -q` completed successfully with all 257 collected
tests passing. Those tests validate contracts and regressions; they are not a substitute for a new project-level
semantic evaluation.

## What “agentic” means here

There is one primary execution loop. Several model calls perform specialized roles, but they do not maintain
independent goals or delegate recursively:

1. **Question planner** — classifies the route and produces a typed interpretation contract from the user question
   and a bounded description of the live project schema.
2. **Execution model** — selects tools, examines observations, changes direction, and drafts an answer inside that
   contract.
3. **Model-assisted evidence reviewer** — challenges omissions, duplicates, exclusions, conflicts, and incomplete
   reconciliation. Its output is also advice, not project evidence.
4. **Structured-claim producer** — translates each cited answer statement into a typed claim binding. It does not
   decide whether the claim is true.
5. **Deterministic verifier and renderer** — checks the claim against the exact tool observation and renders only
   verifier-observed values.

The planner, executor, reviewer, and claim producer may use the same configured model family. They are separated by
role and schema, not guaranteed to be statistically independent. Python code, rather than another model, owns the
critical authorization, completeness, provenance, reconciliation, and final-value checks.

## End-to-end flow

```text
Project files
    |
    v
strict discovery and ingestion --> normalized read-only SQLite workspace + IFC engine
    |                                      |
    +---------- bounded schema context ----+
                                           v
User question --> structured question planner
                       |
                       +-- material ambiguity / invalid contract --> clarification
                       |
                       v
              route + interpretation plan
                       |
                       v
              primary model tool loop
                       |
          +------------+-----------------------------+
          |            |             |               |
          v            v             v               v
      SQL/raw data   IFC geometry   IFC graph   model-assisted review
          |            |             |          (advice only)
          +------------+-------------+
                       |
                       v
        cumulative evidence ledger and provenance
                       |
         completeness + pagination + reconciliation gates
                       |
                       v
                 cited draft answer
                       |
         legacy citation placement/content precheck
                       |
                       v
             structured claim production
                       |
                       v
 deterministic row -> field -> value -> unit -> predicate verification
                       |
                       v
       deterministic rendering + independent status dimensions
```

The execution model is intentionally free to choose the investigative sequence. There is no fixed query plan,
expected-answer lookup, or offline semantic fallback. The deterministic layer constrains what can be accepted as a
final project claim; it does not hard-code what the answer should be.

## 1. Project discovery and data contract

Each selected project currently must contain exactly one matching set of:

- `<model-id>-tree.json`
- `<model-id>-properties.json`
- `<database-id>.ifc`

For multi-project service operation, `BIM_PROJECTS_ROOT/<project_id>/` selects a directory with the same contract.
Without that setting, request project identifiers are metadata and `BIM_DATA_DIR` is the actual data scope.

Ingestion is strict and fail-closed:

- duplicate JSON object keys are rejected;
- tree roots, nodes, and child arrays must have the expected shape;
- tree IDs must be non-null, non-empty, scalar, and unique;
- child edges must resolve consistently;
- property records must be objects with a canonical non-empty ID;
- an embedded record ID cannot conflict with its map key;
- flattening nested properties rejects display-path collisions such as a literal dotted key colliding with an
  actually nested path;
- numeric parsing accepts only a complete scalar with an unambiguous supported unit representation; compound
  dimensions and locale-ambiguous separators remain raw text;
- numeric parsing is shared across ingestion and verification, accepts only ASCII decimal syntax, rejects Unicode
  numeric confusables and non-finite values, and never guesses a unit scale;
- non-finite JSON numbers are rejected rather than entering SQL or evidence as `NaN`/infinity.

This protects evidence identity and numeric lineage. It does not make the loader format-neutral: COBie, BCF,
multiple IFC files, federated models, databases with different exports, or discipline-specific source packages need
new adapters.

## 2. Runtime schema discovery and normalized model

The planner does not receive a hard-coded discipline vocabulary. It receives a bounded runtime context containing:

- normalized SQLite table definitions;
- record and tree-node counts;
- up to 100 frequent property keys with occurrence counts;
- sampled hierarchy shapes and paths, without assuming a fixed depth;
- identity signals;
- attached IFC schema/table information when applicable;
- IFC geometry availability and declared capabilities;
- available read-only project tools.

All project-controlled schema strings, sampled paths, collections, and nested identity/capability metadata are
bounded before the planner request. The final serialized schema context has a hard 60,000-character ceiling and
records truncation metadata; oversized input fails closed instead of bypassing the later cost guard.

The workspace exposes, as applicable:

- `records` for canonical record identity, name, path, parent, depth, leaf state, and raw record JSON;
- `properties` as an EAV-style property table with exact keys, raw values, strict numeric parse status, parsed value,
  parsed unit, and derivation metadata;
- `tree_nodes` for arbitrary-depth hierarchy;
- raw STEP `ifc_entities` and `ifc_references`;
- named `ifc_objects` and role-aware `ifc_relationships`;
- `record_ifc_candidates` for explicit cross-source identity candidates;
- an attached read-only `ifc_source` when the input is an IFC SQLite export.

This makes querying substantially more schema-agnostic than a list of English/Hebrew keywords. It does not provide
a universal ontology. A property named `System Classification`, for example, is discoverable, but the runtime does
not inherently know what that field means in every authoring discipline. Exact record-to-IFC mapping still relies
on known GUID/external-ID signals (including the current Autodesk-style IFC GUID property); unfamiliar identity
conventions may remain unmapped.

## 3. Interpretation planning and ambiguity handling

Planning is enabled by default in environment-based application construction. One low-reasoning structured model
call produces two objects.

### Route

The route contains:

- answer shape: narrative, list, count, grouped total, measurement, ranking, connectivity, compliance, or comparison;
- required capabilities;
- required sources;
- preferred compute path;
- confidence and uncertainties.

At confidence `>= 0.90`, the runtime exposes a focused tool set derived from declared capabilities. With a valid but
uncertain route, it fails open to a safe read-only superset so an uncertain classifier cannot silently remove a
needed capability. When planning is disabled explicitly, the legacy lightweight classifier remains; that mode does
not provide the same generalization guarantee.

### Interpretation plan

The plan types:

- objective;
- population description and identity basis;
- population universe;
- source-owned filters using the currently certifiable operators `equals`, `contains`, and `starts_with`;
- inclusions, exclusions, and their rationale;
- every output metric's name, definition, aggregation, value field/geometric basis, unit, source basis, null policy,
  grouping keys, and result cardinality;
- relationship meaning, direction, and explicit relationship types;
- assumptions and ambiguities, each with materiality and basis;
- execution decision: execute, inspect then execute, report alternatives, or clarify.

Normalization then distrusts the planner's own labels and checks the contract structurally. Among other rules:

- every executable project answer needs a typed population and output metric;
- a filtered universe needs filters and an all-records/all-products universe cannot carry hidden filters;
- identity bases must be compatible with record or IFC universes;
- every filter must have a concrete owning source;
- every executable filter literal and semantic field name must occur verbatim in the question, with exact casing and
  code points; synonym, code, morphology, or case mappings need a future typed derivation or clarification;
- explicit equality, containment, prefix, and negative wording is checked against the selected operator. Negative
  filters are not yet in the typed operator set, so they clarify instead of being inverted into a positive query;
- an unfiltered all-record/all-product universe requires an explicit, unqualified exhaustive request. Phrases such as
  “all door elements” cannot be reinterpreted as every project record;
- explicit answer form, metric dimension, unit, and ranking cardinality are bound back to the question. A planner
  cannot turn “count” into a list, volume into projected area, metres into feet, or top three into top one;
- property metrics must name the actual property key rather than a generic EAV value column;
- measurement metrics need an explicit unit; dimensionless measurements must say `1`;
- answer shape and metric aggregation must agree: counts use count operations, lists use direct fields, grouped totals
  use grouped aggregates, measurements use direct/numeric aggregates, and rankings use rank operations;
- count units normalize to `count`;
- count metrics must count the planned identity basis, not an arbitrary nullable field;
- aggregate and ranking metrics need explicit null behavior;
- two full grouping paths may not collapse to the same terminal SQL column, and a group key must be read from its
  exact owning source; `records.floor` cannot satisfy `properties.floor`, while property-key casing stays exact;
- verified ranking currently allows only one unique extreme, not top-N;
- free prose inclusions/exclusions without executable filter bindings are rejected;
- material assumptions and unresolved material ambiguities force clarification;
- comparison is currently forced to clarification because the claim contract does not yet type two operands, their
  population bindings, a comparator, and exhaustive operand coverage;
- a direct, non-aggregated measurement requires an exact equality filter on the planned identity; otherwise one
  arbitrary row could not prove a broad population's measurement;
- malformed, missing, incomplete, or unavailable planning output also forces clarification while leaving the safe
  read-only capability set available for a future corrected turn.

“Biggest element” therefore cannot silently become volume, length, area, or bounding-box extent. The planner must
bind one explicit metric and unit without a material unresolved alternative, or the runtime asks a concise question.
The final clarification sentence is runtime-owned and fixed (English/Hebrew); planner-authored clarification prose is
never returned verbatim.

## 4. Tool routing and execution

The primary model receives only the route-approved tools. Ordinary filtering, joining, grouping, counting, and
numeric aggregation over project records are routed to read-only SQL. Geometry and graph work use specialized IFC
tools. Arithmetic over already observed values uses the calculator. Custom work may use the local sandbox when it
cannot reasonably be expressed in SQL.

The runtime sets `parallel_tool_calls=false`, validates JSON arguments against strict tool definitions, executes one
model decision at a time, and caches identical calls within the run. On GPT-5.6 models, Programmatic Tool Calling may
coordinate supported local project functions; model-assisted tools and local Python remain direct-only.

The principal tools are:

| Tool | Role | Evidence class |
| --- | --- | --- |
| `inspect_project` | File metadata, hashes, roots, counts, property-key inventory | Project |
| `list_tree_children` | Exact, cursor-paginated hierarchy traversal | Project |
| `search_records` | Cursor-paginated discovery over names, paths, keys, and values | Project discovery |
| `get_records` | Exact records/properties by canonical object ID | Project |
| `search_ifc` | Cursor-paginated raw IFC text discovery | Project discovery |
| `fetch_more` | Continue the exact signed cursor scope | Project |
| `describe_bim_workspace` | Discover the live normalized schema | Project metadata |
| `query_bim_workspace` | Model-authored read-only SQL with output lineage | Project |
| `calculate` | Safe arithmetic over observed values | Derived |
| `analyze_ifc_geometry` | Placement, bounds, extents, areas, containers, volume diagnostics | Project |
| `rank_ifc_geometry` | Complete-population ranking for supported non-solid metrics | Project |
| `reconcile_populations` | Compare independently sourced identity populations | Project |
| `analyze_ifc_graph` | Direction- and role-aware named IFC relationship paths | Project |
| `review_scope_and_evidence` | Critique a proposed answer/evidence set | Advice, not evidence |
| `research_standards` | Explore external standards | External, not project evidence |
| `run_local_python` | Custom on-device computation in a constrained container | Derived calculation only |

The legacy `aggregate_records` implementation remains for internal compatibility but is not exposed to model runs.

## 5. SQL trust boundary

`query_bim_workspace` accepts one `SELECT`, `WITH`, or `EXPLAIN QUERY PLAN` statement. SQLite authorization and
runtime checks reject writes, attachment, unsafe functions, excessive SQL/value sizes, multiple statements, and
long-running work. Queries are parameterized, time-limited, and display-row-limited (maximum 500).

Two different standards apply:

1. **Exploration:** a safe read-only query may help the model inspect the project.
2. **Certification:** a query supporting a final claim must also pass conservative scope and lineage checks.

For certification, the runtime checks the planned filters against parameterized SQL predicates, source ownership,
EAV property-key/value pairing, population universe, exact grouping columns, aggregate shape, result completeness,
and projected-column lineage. It rejects `LIMIT`/`OFFSET`, hidden set operations, ambiguous bare aggregate columns,
duplicate output aliases, query-added filters, count-of-the-wrong-column, group-path/source collisions, and property
keys with different casing. Text `contains` and `starts_with` have one definition—exact, case-sensitive Unicode
sequence matching—and only safely escaped SQLite `GLOB` predicates can certify them; collation-dependent `LIKE`
remains exploratory. Model-visible SQL evidence is also cumulatively capped at 8,192 cells and 512,000 serialized
characters. Truncation is explicit and blocks certification rather than hiding rows. A query may be safe to execute
yet too ambiguous to certify; in that case it remains exploratory evidence only.

## 6. Pagination and complete populations

Raw hierarchy, record, and IFC searches return at most 200 rows per page and an opaque signed cursor. There is no
longer a fixed five-cursor ceiling. `fetch_more` is accepted only for a cursor that is currently outstanding in the
same run. The runtime merges every page into one cumulative observation, preserves the original tool and arguments,
checks total/returned counts, and marks the chain complete only when the terminal page has no cursor.

An unresolved cursor blocks an exhaustive answer unless structured claim verification is enabled and a later
complete SQL result independently matches the validated plan's source lineage, population universe, filters, and
grouping, and exposes project-derived output lineage for every planned metric (plus identity for direct outputs). In
that case only the coarse cursor gate is released; the later verifier must still bind every answer atom to the
complete SQL evidence. With structured verification disabled, every opened cursor must be exhausted.

Complete SQL identity populations can be retained behind an opaque in-process handle even when only the first 500
rows are shown to the model. Handles are created only for one direct identity column and a fully exhausted result,
are capped at 100,001 rows and 512 characters per identity, and share a 200,000-value cache. Missing, evicted,
over-cap, or count-inconsistent handles fail closed. These are deliberate memory/safety limits, not universal-scale
exhaustiveness.

## 7. IFC semantics, geometry, and graph behavior

IfcOpenShell is used for model semantics and geometry. Geometry output is interpreted in SI mesh coordinates as
provided by IfcOpenShell; native-unit placements are converted using a verified `IfcUnitAssignment`. If unit scale
cannot be established safely, the physical result is unavailable rather than guessed with a fallback factor.
Exact geometry currently applies to STEP IFC input; an attached IFC SQLite export can be queried read-only but is
reported as unavailable to the IfcOpenShell geometry engine.

Supported geometry includes:

- world placement;
- axis-aligned bounding box and X/Y/Z extents;
- maximum dimension;
- bounding-box volume (clearly distinct from solid volume);
- mesh surface area;
- projected XY area;
- spatial container identity;
- complete-population ranking by a supported metric.

Certified mesh solid volume is currently withheld. Edge topology and consistent triangle orientation are not enough
to exclude non-adjacent self-intersections, so the implementation reports diagnostics but does not present the
computed signed volume as certified. Solid-volume ranking is not exposed.

The graph tool traverses named, role-aware IFC relationships and returns evidence paths. However, open-ended
connectivity answers are currently forced to clarification because the interpretation contract does not yet type
start and target populations, direct versus transitive semantics, and an exhaustive depth boundary well enough to
certify a negative or complete conclusion.

## 8. Reconciliation

Reconciliation is now an enforced contract, not a log entry. It is required when the plan genuinely depends on
multiple project sources, exhaustive cross-source scope, or a cross-source conclusion. Filter sources are promoted
into route requirements, so a tree-owned location filter combined with a property metric cannot quietly execute as
a properties-only question.

Each reconciled view must be independently owned by the declared source, complete, bound to the same typed
population predicate, and expressed in the planned identity basis. The runtime does not allow one joined query to
masquerade as two independent source populations. A narrow loader-defined homologous projection is allowed only
for exact tree/record identity fields whose ownership is known.

Outcomes are `matched`, `mismatched`, `incomplete`, or identity-only where applicable. A mismatch remains sticky;
an incomplete attempt cannot be treated as success. A later valid matched reconciliation may supersede an earlier
incomplete attempt for the same scope. Required mismatch or incompleteness prevents an ordinary `completed` result
and produces an explicit limited status/disclosure.

Reconciliation proves agreement between loaded views. It does not prove that any source is correct, current, or
synchronized with the authoring system.

## 9. Citation grounding and structured claim verification

The answer model must cite every factual project statement with `[ref: call_id]`. The runtime first performs a legacy
citation placement and obvious-content check. This lexical stage is only a prefilter now; it is not the ground-truth
decision.

For every cited project statement, the structured-claim producer emits one of the supported typed bindings,
including:

- evidence reference;
- row identity predicates;
- field path;
- expected value and expected unit;
- filter predicates;
- aggregate operation, source rows, completeness, and group identity; or
- ranking direction, metric path, selected identity, population, and uniqueness.

The deterministic verifier then resolves the exact evidence observation and checks:

```text
evidence reference
  -> independently identified row/population
  -> exact field path and source lineage
  -> value
  -> unit
  -> predicate/operator
  -> completeness / aggregation / ranking condition
```

This closes the old failure where “5 doors” could pass merely because the cited output contained an unrelated `5`.
Identity comparisons are typed and exact; the claimed value cannot identify its own row. Property claims outside SQL
must bind the full nested property path, not just a matching terminal name. SQL aggregates must preserve independent
population or group identity. Numeric parsing and unit association must come from the same supported source pair.
Aggregate and ranking filters must exactly implement the planned population; a claim cannot add `object_id=A` to
turn an all-record total into one selected row. Direct per-identity outputs require exhaustive identity/group coverage
regardless of whether the planner labels the answer `list` or `narrative`. Complete grouped SQL evidence binds each
group path to its owning source before terminal output aliases may be used.
External-standard claims are blocked from certified output until their jurisdiction, edition, section, criterion,
comparator, and unit have a typed verification contract.

The claim producer is allowed one correction cycle when its mapping is incomplete or invalid. If coverage or any
binding still fails, the project answer is withheld. For a verified answer, final factual values are rendered from
the verifier's observations rather than copied from model prose. Empty headings, tables, or “Summary” scaffolding do
not count as a substantive project answer. Cited evidence serialization has per-observation and cumulative hard
bounds, preserves a full compact observation when it fits (including more than 60 rows), and emits an explicit
incomplete marker rather than a misleading partial blob when it does not.

## 10. Review, disclosures, and snapshot semantics

The model-assisted reviewer no longer trusts executor-authored summaries of its inputs. The runtime replaces the
question with the original request, scope/exclusions with the validated interpretation contract, evidence with a
cumulatively bounded projection of the actual ledger, and reconciliation with actual run status and observations.
Only the proposed draft remains executor-authored. Any omitted runtime input or truncated/empty draft makes review
non-final, and a materially changed final factual draft requires a fresh review.

Its response has an exact schema with bounded arrays, booleans, and disclosure codes. Malformed reviewer output
cannot turn strings such as `"false"` into a truthy approval, and `can_finalize=true` is invalid when required
follow-up is non-empty. The runtime independently treats any follow-up as blocking even if a caller bypasses that
schema check.

Required follow-up blocks finalization. Required disclosures must be satisfied with valid evidence. Canonical runtime
disclosures cover cross-source mismatch, snapshot/revision uncertainty, interpretation assumptions, and related
limitations. The runtime removes model-authored variants and appends a controlled version so a disclaimer cannot
quietly change a factual value.

## 11. Cost, iteration, concurrency, and traces

- Environment construction enables planning and structured verification by default.
- The code default is 20 execution iterations; the supplied `.env.example` configures 40.
- Tool output sent to the model is bounded and compacted; full observations remain in the trace.
- SQL model evidence and structured-claim evidence have separate cumulative size/cell caps; overflow is visible and
  cannot be certified as complete.
- Identical in-run tool calls are reused.
- Stable prompt-cache keys reduce repeated model input.
- Every active planner, executor, reviewer, schema-resolver, and standards-research response contributes to the cost
  report.
- Cost includes cached/uncached input and output tokens and a per-request breakdown. Unknown model pricing can be
  configured. Tool fees that cannot be priced are disclosed as excluded rather than treated as zero.
- `BIM_MAX_ANSWER_COST_USD=0` disables the guard by default. A positive guard is reactive because usage is known only
  after each response; crossing it now uses deterministic checkpoint salvage and makes no finalization model call.
- The cached service agent serializes complete requests with a per-agent lock because evidence aliases, handles,
  review deltas, and usage counters are run-scoped mutable state.
- The web cache is invalidated by project file and relevant configuration fingerprints.
- Retryable upstream failures are mapped to HTTP 503.
- Each request writes a JSONL audit trace with session ID, response IDs, model/tool transcript, raw observations,
  cached-call markers, reconciliation state, gate decisions, and secret-key redaction.

The trace can contain sensitive project observations even though raw files are not uploaded as OpenAI Files. Access,
retention, and deletion policy for the trace directory are deployment responsibilities.

## 12. Local Python security boundary

`run_local_python` executes model-written code only in a local Docker container. Project inputs and the generated
workspace are mounted read-only. The container has no network, a read-only root filesystem, an ephemeral writable
`/tmp`, a non-root user, dropped capabilities, no-new-privileges, CPU/memory/PID caps, timeout, and bounded output.
The application does not automatically pull an image.

This is safer than host execution, but it is still optional high-complexity computation. Its output is classified as
derived calculation and cannot independently certify a project fact; underlying project inputs still need direct,
lineaged evidence and final project values must pass the normal claim contract. Extracted tool observations and the
question are still sent to the configured model; “local files” does not mean “no project data ever leaves the
machine.”

## 13. Output and status model

The top-level `status` and evaluator `verification_status` are retained for compatibility and represent request
lifecycle, not factual truth. The report now exposes independent dimensions:

| Dimension | Examples | Meaning |
| --- | --- | --- |
| `answer` | answered, answered_with_limitations, clarification_requested, partial, withheld | Whether a substantive answer was delivered |
| `evidence` | citation_grounding_passed, structured_claim_verification_rejected, incomplete | Evidence-gate outcome |
| `reconciliation` | not_required, matched, mismatched, incomplete | Cross-source population outcome |
| `budget` | not_configured, within_budget, exceeded, enforcement_unavailable | Spending-guard outcome |
| `ambiguity` | resolved, assumption_stated, clarification_required, not_assessed | Interpretation outcome |

This prevents a single word such as `completed` from implying that scope, evidence, source agreement, and cost all
passed. The current external evaluator's `SystemAnswer` model does not ingest `status_dimensions`, so that evaluator
still loses this distinction unless updated.

## Analysis of the existing evaluation report

The semantic report at `C:\Users\avina\Downloads\bim-evaluator\eval-report.json` records the pre-hardening baseline:

| Measure | Observed baseline |
| --- | ---: |
| Cases | 13 |
| Semantically correct | 4 (30.8%) |
| Incorrect | 9 (69.2%) |
| Execution errors | 0 |
| Legacy pipeline `completed` | 5 |
| Legacy pipeline `limited` | 8 |
| Mean elapsed time | 143.675 s/question |
| API requests | 258 (19.8/question) |
| Total tokens | 3,803,182 (~292,553/question) |
| Reported cost | $5.9019154, partial (~$0.454/question) |
| Cases hitting the configured $0.50 guard | 6/13 |

The most important observations are structural:

- Two incorrect cases were still labeled `completed` (cases 5 and 10). The old lifecycle status was therefore not a
  sound correctness signal.
- One correct case was labeled `limited` (case 4). Conservative failure and semantic correctness were not the same
  dimension either.
- Six cases reached the same flat cost guard, including a case the judge considered correct. Complexity was not
  allocated before execution.
- Cases 6 and 7 were withheld after citation-grounding retries, showing that a gate could prevent an unsupported
  answer but could not recover the missing answer.
- Case 10 is the clearest ambiguity example: the response mixed solid/bounding volume and run-length/bounding-extent
  measurement bases. A citation verifier cannot repair a silently wrong metric definition chosen upstream.
- Several failures were scope or population failures: omitted categories, wrong inclusions, incomplete floor/type
  coverage, or broad claims over a narrow observed subset.
- The semantic judge was `gpt-5.6-sol`, the same family configured for the system in the supplied environment. That
  is convenient but not an independent judge-family design.
- The external evaluator validates only `answer`, legacy `verification_status`, and selected metadata. It currently
  drops the new `status_dimensions` field.
- The older exact-match report (`eval-report-agentic-v2-exact.json`) scored 0/9 with mean elapsed time 3.417 seconds.
  Exact string matching is not a useful primary metric for open-ended, multilingual BIM answers; it can remain a
  regression signal for intentionally canonical outputs.
- The supplied audit reports that reference answers for cases 12 and 13 are themselves wrong. That claim has not
  been independently adjudicated in this repository, but it is enough to make the current aggregate score unsuitable
  as a release truth until references receive source-backed review.

The report does **not** show random model noise. It shows ambiguity chosen before execution, weak old grounding,
dataset-specific routing, population drift, incomplete reconciliation, pagination/budget pressure, and an evaluation
method that conflates lifecycle with correctness. Those are the same failure modes the hardening work targets.

## Audit-priority implementation matrix

| Priority from the audit | Current state | What changed | What remains |
| --- | --- | --- | --- |
| 1. Resolve ambiguity before execution | **Implemented for the typed contract** | Schema-aware interpretation plan; material ambiguity/assumption fails to clarification; explicit population, metric, unit, filters, universe, and relationships; verbatim field/literal and explicit shape/operator/dimension/unit/cardinality binding | Interactive follow-up is a new request; synonym/code mappings, negative filters, and other untyped concepts remain unavailable rather than magically resolved |
| 2. Replace lexical grounding | **Implemented for final project claims** | Structured row/field/value/unit/predicate verification, exact aggregate scope, source-owned grouping, exhaustive direct-output coverage, lineage, completeness, ranking and deterministic rendering | Claim production is model-assisted and can withhold valid prose if mapping fails; exhaustive non-SQL per-identity output and external standards remain blocked |
| 3. Schema-agnostic routing | **Implemented by default** | Model classification over runtime schema; capability-derived tools; uncertainty fails open to safe read-only superset | Legacy classifier remains when planning is disabled; schema discovery is not a domain ontology |
| 4. Generalize the data model | **Partial** | Runtime SQLite schema discovery, arbitrary tree depth, exact properties, IFC semantics/geometry, strict units and identity | Loader still requires one tree + one properties + one IFC file; no federated/multi-format adapter layer; only one real discipline/project evaluated |
| 5. Scale cost to complexity | **Partial / not solved** | Flat guard disabled by default; full multi-call cost telemetry; deterministic grounded-checkpoint salvage | No preflight cost estimate, adaptive tier, sub-question decomposition, or independent sub-budgets |
| 6. Enforce reconciliation | **Implemented** | Source-owned complete populations; mismatch/incomplete outcomes block ordinary completion and downgrade status | Agreement does not establish freshness or truth; more identity adapters may be needed on new exports |
| 7. Fix pagination dead-end | **Implemented within safety caps** | No five-page ceiling; signed cursor continuation; cumulative provenance-preserving results; complete SQL population handles | Overall iteration, 100,001-row handle, 200,000-value cache, and response-size caps remain |
| 8. Broaden evaluation | **Not yet done** | Added extensive deterministic regression tests around the structural failures | Still needs multiple disciplines/projects, adversarial questions, verified references, and an independent judge family |

## When the design is good

It is a good fit when:

- the project satisfies the input contract and ingests without identity/path conflicts;
- the question can be expressed as an exact population plus a supported field, count, group, aggregate, or unique
  extreme;
- filter literals, semantic field names, operators, requested units, and metric dimensions are stated explicitly
  enough to survive the conservative question-to-plan binding;
- the relevant property keys and units exist in the loaded snapshot;
- SQL can return a complete, independently identified result;
- a geometry request uses a supported area, extent, placement, or bounding-box metric;
- cross-source identities can be materialized and reconciled;
- the user prefers a clarification or withheld answer over an implicit semantic guess;
- read-only execution, detailed provenance, bounded tools, and auditability are requirements.

Examples include “count distinct records whose exact type property contains X,” “group the selected records by the
exact floor field,” “return this known object's exact property value and unit,” and “which uniquely has the largest
X extent among this explicitly defined complete mesh population?”

## When the design is bad or intentionally unavailable

It is a poor fit, or will deliberately stop, when:

- “biggest,” “near,” “critical,” “connected,” “material,” or “compliant” has not been defined sufficiently;
- the request relies on a synonym, translated schema value, implicit case conversion, negative filter, or project
  code mapping that is not represented by a typed derivation contract;
- the user expects discipline expertise that is absent from the data and no project ontology is supplied;
- the answer needs a normative code conclusion rather than exploratory standards context;
- connectivity requires a complete negative assertion or implicit endpoint/depth semantics;
- the requested metric is certified solid volume;
- the request asks for top-N ranking or tie-boundary completeness rather than one unique extreme;
- the request needs an exhaustive per-identity geometry narrative/list; current exhaustive direct-output coverage is
  certified only from one complete SQL row set;
- the request is a general A-versus-B or threshold comparison; operands and comparator semantics are not yet typed;
- the calculation needs a derived formula whose exact inputs, units, and lineage are not represented in the plan;
- a numeric property is compound or locale-ambiguous;
- an exhaustive population exceeds the handle/cache/output/iteration limits;
- different sources cannot be mapped to one identity basis;
- project freshness or source synchronization must be guaranteed;
- the input is a federated or nonconforming project package;
- latency or token cost comparable to a conventional indexed query service is required.

Failing closed is appropriate for engineering truthfulness, but a system that clarifies or withholds many questions
still needs capability work before it can be marketed as general.

## Residual risks and design trade-offs

1. **No post-change generalization evidence.** Regression tests establish code contracts, not semantic performance on
   unseen disciplines.
2. **Model-correlated roles.** Planner, executor, reviewer, claim producer, and evaluator may share a model family.
3. **Strict verifier, narrower recall.** Exact paths and lineage reduce false acceptance but can reject semantically
   correct answers that the claim producer cannot express. Verbatim question binding deliberately rejects reasonable
   synonym, inflection, translation, and case mappings until those mappings become typed evidence.
4. **Schema discovery is not ontology discovery.** Seeing a field name does not establish its engineering meaning.
5. **Reactive resource control.** Iteration and optional dollar caps stop work; they do not estimate or allocate work.
6. **Bounded exhaustiveness.** SQL display rows, cursors, handles, caches, output compaction, and model iterations all
   impose finite limits.
7. **Reconciliation is snapshot-local.** Matching IDs/counts cannot verify authoring-system revision state.
8. **Geometry certification boundary.** Areas and extents are supported; solid volume is intentionally unavailable.
9. **Standards are exploratory only.** Current typed claims cannot certify jurisdiction-specific compliance.
10. **Sensitive traces.** Full auditability increases local data-retention responsibility.
11. **Reviewer independence is limited.** Runtime-owned inputs prevent executor fabrication, but the reviewer can
    still share a model family with the planner/executor and remains advisory rather than an independent oracle.

## Recommended next work

1. Build a source-backed evaluation corpus across electrical, structural, HVAC, plumbing, and at least two projects
   per discipline. Include malformed exports, multilingual terms, ambiguous questions, adversarial scope changes,
   missing values, and cross-source conflicts.
2. Independently verify each reference from immutable source evidence, version the reference and interpretation, and
   use a judge from a different model family plus deterministic field-level checks.
3. Update the evaluator to ingest and score `status_dimensions` separately. Track answer correctness, refusal
   appropriateness, evidence precision/recall, reconciliation, latency, tokens, and cost rather than one boolean.
4. Add a planner-side complexity estimate and decompose multi-part questions into typed subplans with independent
   evidence and budgets. Treat the dollar limit as a final safety guard, not the scheduler.
5. Introduce a project-adapter interface so the normalized workspace can be populated from federations, multiple IFC
   files, COBie/BCF, databases, and client-specific exports without weakening provenance.
6. Extend the typed contract for connectivity endpoints/depth/directness and for standards jurisdiction/edition/
   section/comparator/unit before enabling those conclusions.
7. Add a trusted self-intersection-capable mesh validation path before re-enabling certified solid volume.
8. Add explicit tie and rank-position semantics before supporting top-N.

## Release criteria for a generalization claim

Do not claim “any project, any question” until all of the following are demonstrated:

- multiple unseen projects and disciplines pass a versioned, source-backed evaluation;
- ambiguous questions reliably clarify before tools are called;
- wrong-but-`completed` is zero on the release suite;
- structured evidence precision is measured independently of answer fluency;
- source mismatches always appear in status and answer disclosure;
- exhaustive workloads complete or return an explicit capacity status;
- budget/latency targets are met by question-complexity tier;
- evaluator references and judge decisions are independently audited;
- unsupported capabilities are advertised honestly.

The current system is best described as **a hardened, auditable BIM investigation agent for a defined project export
contract**, with strong correctness controls for a deliberately bounded set of answer forms. That is a credible and
useful design. It is not yet universal.

## Implementation map

- `bim_agent/question_planning.py` — schema discovery, route classification, interpretation-plan validation.
- `bim_agent/agent_loop.py` — model-directed loop, evidence ledger, gates, reconciliation enforcement, rendering,
  cost handling, and telemetry.
- `bim_agent/claim_verification.py` — structured claim schema, exact row/field/value/unit/predicate verification.
- `bim_agent/project_tools.py` — strict ingestion, read-only raw/SQL tools, pagination, handles, lineage, reconciliation,
  graph analysis.
- `bim_agent/ifc_analysis.py` — IFC semantic index, geometry, selectors, metric ranking, unit handling.
- `bim_agent/model_tools.py` — model-assisted evidence review, external research, strict review schema.
- `bim_agent/local_python.py` — local Docker execution boundary.
- `bim_agent/models.py` — answer report and independent status dimensions.
- `bim_agent/runtime.py` — configured agent assembly and per-agent run serialization.
- `bim_agents/webapp.py` / `bim_agents/adapter.py` — HTTP service, project/config cache, evaluator compatibility.
- `bim_agent/skills/bim-routing/SKILL.md` — versioned runtime routing and evidence rules.
- `tests/` — mocked model-loop, ingestion, SQL, pagination, reconciliation, claim-verification, status, API, geometry,
  and safety regressions.
