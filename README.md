# BIM Multi-Agent Answering

A hierarchical, read-only agent system for answering questions from one authorized BIM project.

## Agent hierarchy

Only three agents participate:

- BIM Supervisor — routes once, requests verification once, and writes the final answer.
- BIM Analyst — maps civil-engineering or plain-language questions to safe declarative query plans.
- Verification Agent — independently reruns and checks every query result.

The ontology and graph contract are deterministic resources, not additional agents. The analyst uses
one general query language for model elements, canonical measures, permit knowledge, and project-graph
scope. Schema discovery, Cypher writing/review, result analysis, and answer composition are not
separate agents.

## Graph Schema Contract

`graph_schema.yaml` is the versioned bridge between canonical ontology concepts and Neo4j storage.
It declares supported node labels, properties, relationships, the project-authorization path, and
the semantic entities and fields exposed to the general query language. The query engine reads this
contract; agents never invent labels, relationship paths, property names, or Cypher.

The same query plan supports `count`, `list`, `group_count`, `group_summary`, `distinct`, `sum`,
`average`, `minimum`, and `maximum`, with validated equality, text, existence, membership, and numeric
filters. `group_summary` returns a distinct record count and summed numeric metric per group. This covers
questions such as:

- How many existing load-bearing walls are modelled?
- Which IFC classes occur in the model, and how many of each are there?
- What is the total modelled wall area?
- Show the project height and setback measures.
- Which spaces are on the ground floor?
- What permit knowledge is available about balconies?

Filtered space and apartment answers are analytical by default. A count or list includes a bounded,
verified record breakdown with the source object ID, IFC class, name, modelled level, area, segment,
owner, and room-count programme when those values exist. A caller can request a scalar-only space count
with `include_details=false`; large result sets remain bounded by the query-plan limit.

For relationship questions or unfamiliar graph concepts, the Analyst can optionally call
`inspect_project_graph_structure`. It returns distinct project-scoped node-label signatures, bounded
sample names, relationship types, endpoint labels, counts, and optional property names. A focus term
keeps discovery compact. The result informs a contract-backed plan; it never enables raw Cypher.

The runtime validates the contract against live Neo4j metadata before starting any agent. A missing
label, relationship, or property fails closed. Validate it separately with:

```powershell
python -m bim_agents.schema_check
```

## Safety invariants

- Neo4j access is read-only at the application policy layer.
- Agents cannot generate or execute arbitrary Cypher.
- The query engine compiles only contract-approved plans to parameterized, read-only Cypher with
  `.source IN $allowed_sources`.
- The runtime supplies `allowed_sources`; the model cannot choose them.
- Verification independently reruns every saved plan and compares its complete result digest.

Use a Neo4j account with database-level read-only privileges as the final security boundary.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure `.env`:

```text
OPENAI_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
BIM_CLIENT_ID=...
BIM_PROJECT_ID=...
BIM_AGENT_MODEL=gpt-5.6-sol
BIM_AGENT_WORKER_MODEL=gpt-5.6-terra
BIM_RUN_TIMEOUT_SECONDS=180
BIM_SUPERVISOR_MAX_TURNS=6
BIM_MAX_LLM_CALLS=12
BIM_MAX_TOOL_CALLS=20
BIM_MAX_AGENT_STARTS=8
BIM_QUERY_TIMEOUT_SECONDS=20
# Optional override; defaults to graph_schema.yaml in the repository root.
BIM_GRAPH_SCHEMA_PATH=graph_schema.yaml
```

Run:

```powershell
python -m bim_agents.cli "How many apartments are on the seventh floor?"
```

Web chat interface:

```powershell
python -m bim_agents.webapp
```

Then open `http://127.0.0.1:8000`. The interface shows project connectivity, processing stages,
verified claims, model limitations, and evidence IDs returned by the same supervised runtime.

Progress logs are printed to stderr. To retain detailed logs and impose a shorter deadline:

```powershell
python -m bim_agents.cli "How many apartments are on the seventh floor?" --timeout 120 --log-file logs/bim-run.log
```

Every nested agent has its own turn limit. The complete run also has wall-clock, LLM-call,
tool-call, agent-start, and supervisor-turn guardrails. Exceeding one stops the run with a
progress summary instead of continuing indefinitely.

Offline tests:

```powershell
python -m unittest discover -s tests -v
```

## Current boundary

The general query layer can answer questions expressible from the graph's model elements, canonical
measures, permit knowledge, and project scope. It does not infer facts absent from the BIM, perform
structural-design calculations, or assert code compliance without corresponding trusted data and
rules. New graph fields or relationship families are added declaratively to `graph_schema.yaml`, not
by creating another agent or one tool per question. A project graph count includes source-scoped data
nodes plus its Client, Project, and BIMHub nodes; it never includes unrelated database nodes.
