# Ground-floor apartment query trace

Question: `How many apartments are there on ground floor?`

## Constraints followed

- Did not import or execute repository application code.
- Read only `.env` variable names and used their values at runtime.
- Did not print or store credentials.
- Queried Neo4j directly with the official Python driver.
- Used the client-to-project authorization path before selecting BIM spaces.

## Environment inputs

The direct query used these `.env` variables:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `BIM_CLIENT_ID`
- `BIM_PROJECT_ID`

`OPENAI_API_KEY` was not needed. The question maps deterministically to graph fields, so no LLM call was made.

## Semantic mapping

- Entity: `IfcSpace`
- Apartment filter: `canonical_type = 'apartment'`
- Ground-floor filter: `canonical_level = '00 begane grond'`
- Unique apartment key: `object_id`

The stored ground-floor label is Dutch: `begane grond`.

## Read-only Cypher

```cypher
MATCH (c:Client {id: $client})
      -[:HAS_PROJECT]->(p:Project {id: $project})
      -[:HAS_BIM]->(:BIMHub)
      -[:CONTAINS]->(ip:IfcProject)
MATCH (s:IfcSpace)
WHERE s.source = ip.source
  AND s.canonical_type = 'apartment'
  AND s.canonical_level = '00 begane grond'
RETURN count(s) AS node_count,
       count(DISTINCT s.object_id) AS distinct_apartments,
       collect({object_id: s.object_id, name: s.name}) AS matches
```

## Verification result

- Matching nodes: `5`
- Distinct `object_id` values: `5`
- Object IDs: `28`, `29`, `30`, `31`, `32`
- Names: `Area:2654784`, `Area:2654785`, `Area:2654786`, `Area:2654787`, `Area:2654792`

## Answer

There are **5 apartments on the ground floor**.
