# BIM Multi-Agent Answering

A hierarchical, read-only agent system for answering questions from one authorized BIM project.

## Agent hierarchy

Only three agents participate:

- BIM Supervisor — routes once, requests verification once, and writes the final answer.
- BIM Query Agent — uses ontology-aware, typed BIM query tools.
- Verification Agent — independently reruns and checks candidate counts.

The ontology is a deterministic tool, not another agent. Schema discovery, Cypher writing/review,
result analysis, and answer composition are not separate agents. Unsupported question types fail
closed until a corresponding typed BIM tool is implemented.

## Graph Schema Contract

`graph_schema.yaml` is the versioned bridge between canonical ontology concepts and Neo4j storage.
It declares supported node labels, properties, relationships, the project-authorization path, and
each query capability's exact graph mapping. Typed tools read this contract; agents never invent
labels, relationship paths, or property names.

The runtime validates the contract against live Neo4j metadata before starting any agent. A missing
label, relationship, or property fails closed. Validate it separately with:

```powershell
python -m bim_agents.schema_check
```

## Safety invariants

- Neo4j access is read-only at the application policy layer.
- Agents cannot generate or execute arbitrary Cypher.
- Typed query tools contain fixed read-only Cypher with `.source IN $allowed_sources`.
- The runtime supplies `allowed_sources`; the model cannot choose them.
- Verification independently reruns candidate counts before the supervisor answers.

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

The current typed tools support authorized project-wide graph-node counts and ontology-resolved
element counts by level. A project-wide node count includes source-scoped data nodes plus its Client,
Project, and BIMHub hierarchy nodes; it does not count unrelated nodes in the Neo4j database.
Additional BIM question types should be added as typed tools only when needed. Production rollout
should also add a Neo4j read-only user and representative BIM evaluation questions.
