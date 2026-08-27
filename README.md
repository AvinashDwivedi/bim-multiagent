# Agentic BIM Graph Assistant

A provider-neutral, evidence-first multi-agent system that answers construction and BIM questions
against an authorized Neo4j graph transformed from IFC/Revit data. It discovers each project's
vocabulary and measurement conventions at runtime; there are no project-specific query shortcuts.

## Architecture

Each request passes through independent stages:

1. **Graph Inspector** profiles labels, relationships, properties, identities, and sources.
2. **Ontology Analyst** identifies candidate real-world entity populations.
3. **Measurement Analyst** independently defines scope, identity, basis, aggregation, and units.
4. **Semantic Planner** reconciles those analyses into a typed, falsifiable query contract.
5. **BIM Investigator** gathers scoped graph evidence and emits atomic cited claims.
6. **Counterexample Auditor** receives no planner rationale and runs its own graph checks.
7. **Claim Verifier** applies model review and a deterministic evidence gate. Both auditor and verifier
   must support a claim, and every citation must exist, before it can reach the answer.
8. **Answer Composer** receives only accepted claims and leads with a concise direct result.

Generated Cypher artifacts carry semantic provenance. The system inventories documents before asserting
absence, caches live schema profiles and identical evidence requests, bounds calls/artifacts/rows/turns,
reserves independent evidence capacity per phase, permits one capacity-aware targeted repair, and
returns controlled limitations instead of inventing project facts.

## Provider selection

OpenAI GPT-5.6 Sol is the default. Switch providers without code changes:

```dotenv
BIM_LLM_PROVIDER=openai
BIM_OPENAI_AGENT_MODEL=gpt-5.6-sol
BIM_OPENAI_WORKER_MODEL=gpt-5.6-sol
OPENAI_API_KEY=...
```

or:

```dotenv
BIM_LLM_PROVIDER=anthropic
BIM_ANTHROPIC_AGENT_MODEL=claude-sonnet-4-6
BIM_ANTHROPIC_WORKER_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=...
```

Only the selected provider's key is required. Reasoning effort is routed by agent responsibility and
semantic complexity:

| Work | Effort |
|---|---|
| Ontology and measurement analysis | `BIM_REASONING_EFFORT` (default `medium`) |
| Semantic planning | `BIM_REASONING_EFFORT_HIGH` (default `high`) |
| Routine investigation | Default effort |
| Ambiguous, alternative-heavy, or repair investigation | High effort |
| Counterexample audit and claim verification | High effort |
| Final answer composition | `BIM_REASONING_EFFORT_LOW` (default `low`) |

The policy uses contract-level signals such as unresolved terms, competing interpretations, assumptions,
entity-role ambiguity, evidence requirements, and repair status. It does not inspect project-specific names.
Budgets use `BIM_AGENT_MAX_TURNS`, `BIM_AGENT_MAX_ARTIFACTS`, `BIM_AGENT_MAX_REPAIRS`,
`BIM_AGENT_MAX_MODEL_CALLS`, `BIM_AGENT_MAX_RESULT_ROWS`, and `BIM_QUERY_TIMEOUT_SECONDS`.

Evidence capacity is reserved rather than first-come-first-served:

```dotenv
BIM_AGENT_MAX_ARTIFACTS=18
BIM_INVESTIGATOR_ARTIFACT_BUDGET=8
BIM_AUDITOR_ARTIFACT_BUDGET=4
BIM_REPAIR_ARTIFACT_BUDGET=4
BIM_REPAIR_AUDITOR_ARTIFACT_BUDGET=2
BIM_AGENT_MAX_CYPHER_REPAIRS=2
BIM_INVESTIGATOR_MAX_TURNS=6
BIM_AUDITOR_MAX_TURNS=3
BIM_REPAIR_MAX_TURNS=3
BIM_REPAIR_AUDITOR_MAX_TURNS=2
```

The auditor must cite evidence generated in an audit phase before a direct claim can pass. A repair starts
only when both repair phases and at least seven model calls remain. Invalid Cypher receives at most the configured
number of model-visible correction opportunities.

## Agent traces

`BIM_TRACE_ENABLED=true` writes one request-scoped JSONL file to `BIM_TRACE_DIR` (default
`logs/agent-traces`). Each trace records agent/stage transitions, model-call numbers, tool arguments,
small evidence samples, phase budgets, structured semantic decisions, claim/citation summaries, provider response
IDs, latency and token usage, repair rounds, tool failures, and final status. Evaluation requests also carry run and
case IDs into the trace. API keys and full evidence rows are never logged;
long values are truncated. Trace I/O failures cannot fail a BIM request.

## Run

```powershell
python -m pip install -e ".[test]"
python -m bim_agents.webapp
```

```powershell
python -m bim_agents.cli "how many apartments are on the seventh floor?" `
  --client-id <client-uuid> --project-id <project-uuid> --quiet
```

Endpoints are `GET /api/health`, `POST /api/chat`, and `POST /api/chat/stream`. Health reports the
active provider and model.

## Safety

- Neo4j sessions are read-only; generated Cypher is guarded and run through `EXPLAIN`.
- Every BIM population must carry client/project scope, and arbitrary values are parameters.
- Mutations, comments, multiple statements, APOC, DBMS procedures, and unapproved procedures are rejected.
- Answers preserve population, identity, measurement basis, units, and missing-versus-zero semantics.
- Production still needs database-level read-only credentials; application checks are defense in depth.
