# Read-only BIM Investigation Agent

A coding-agent-style, read-only system for investigating one authorized BIM project.

## Flow

1. The Investigator creates an explicit task contract.
2. It inspects live labels, properties, relationships, identities, and stored values as needed.
3. It registers an evidence-backed semantic mapping and submits a minimal declarative query plan.
4. The deterministic handler compiles and executes parameterized, project-scoped Cypher.
5. The Investigator examines the result and can refine its inspection or plan.
6. It replays and verifies query evidence before returning an answer.

This is an iterative tool loop rather than a fixed one-pass pipeline. The model chooses the next safe
inspection action from observed results, while budgets bound turns, tool calls, and agent starts.

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

The agents exchange `TaskContract`, `GraphDiscovery`, `SchemaMapping`, `QueryPlan`, `CypherQuery`,
`QueryResult`, and `VerificationReport` artifacts rather than relying on prose handoffs.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Configure the Neo4j connection, OpenAI API key, client, and project using the environment variables
consumed by `Settings.from_env()`.

The iterative investigator defaults to 24 model turns. Override this bounded limit with
`BIM_SUPERVISOR_MAX_TURNS`; values below roughly 12 may stop ordinary investigations before querying
and verification finish.

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

## Web application

```powershell
python -m bim_agents.webapp
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Evaluation

Create a JSON dataset as a list of question/reference-answer records:

```json
[
  {
    "question": "How many apartments are on the ground floor?",
    "answer": "There are 5 apartments on the ground floor."
  }
]
```

A single `{question, answer}` object, `{ "cases": [...] }`, and a simple
`{ "question text": "answer text" }` mapping are also accepted.

Run semantic evaluation and save the full report:

```powershell
python -m bim_agents.evaluator eval-cases.json --output eval-report.json --timeout 360
```

For deterministic normalized exact matching without a grader model:

```powershell
python -m bim_agents.evaluator eval-cases.json --judge-mode exact
```

The report includes the expected and actual answer, pipeline status, latency, exact-match flag,
semantic correctness/completeness/groundedness scores, pass/fail result, errors, pass rate, and average
score. Evaluation is sequential by default to avoid overloading the BIM and model services.
