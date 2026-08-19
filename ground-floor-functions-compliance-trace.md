# Ground-floor functions and compliance investigation

Question: `what type of functions are in ground floor and does it comply with the requirements?`

## Execution trace

1. Started the read-only BIM agent through the CLI with a 300-second timeout.
2. The runtime resolved the authorized client/project and reached `pipeline_start`.
3. The model request failed with `APIConnectionError` before the first BIM tool call. The runtime log is `logs/ground-floor-functions-compliance.log`.
4. Continued with a direct, read-only Neo4j check using only `.env` connection and scope values. Credentials were neither printed nor stored.
5. Followed the authorized path `Client -> Project -> BIMHub -> IfcProject` and restricted all records to the resulting `IfcProject.source`.
6. Grouped distinct `IfcSpace.object_id` values at exact stored level `00 begane grond` by `canonical_type`.
7. Checked the same authorized sources for `PermitKnowledge` and `CanonicalMeasure` records.

## Scoped evidence

Ground-floor space functions:

- parking: 355
- technical_room: 6
- apartment: 5
- circulation: 3
- retail: 3
- amenity: 2
- bicycle_storage: 2
- storage: 2
- horeca: 1
- residential_zone: 1

Requirements evidence:

- `PermitKnowledge` records: 0
- `CanonicalMeasure` records: 0

## Conclusion

The ground floor contains 10 modelled function types. Compliance cannot be assessed from the authorized
BIM evidence because no project-scoped requirement records are available. This is not evidence of either
compliance or non-compliance.

## Solution changes prompted by this run

- Permit-knowledge absence can now be queried safely without a live mapping, including after a space
  mapping has been registered. The query still uses only contract-approved fields and authorized sources.
- Classification verification now accepts grouping by the classification field; it no longer incorrectly
  requires a classification filter for a function schedule.
- The investigator prompt now requires separate BIM-fact and requirement-evidence checks for compliance
  questions and prohibits treating missing requirements as compliant or non-compliant.

## Production rerun supplied by the user

The rerun successfully found 10 function groups covering 380 ground-floor records, then issued two
additional classification diagnostics. It ended as insufficient evidence after the verifier reviewed all
three query artifacts.

Root causes:

1. No `permit_knowledge` query was produced, so the compliance half of the task had no requirements evidence.
2. The explicit `regulatory usage function is missing` diagnostic was rejected by classification-purity
   verification because `is_missing` was not recognized as an exact classification boundary.
3. Verification had already run before a deterministic compliance evidence check could be added.

Additional remediation:

- Compliance/requirements wording now triggers a deterministic, authorized `permit_knowledge` availability
  query before the final verification pass.
- An explicit `is_missing` classification filter is accepted as a valid diagnostic boundary.
- Cached verification is reused only when it covers the complete current set of query evidence; adding a
  requirements query forces a fresh replay of every query.

## Updated live verification

After restarting the port-8000 service, the exact question completed with `verification_status=verified`.
All four query artifacts passed replay, authorization, identity, constraint, counting-unit, boundary,
deduplication, classification, and coverage checks. The requirements query again found zero scoped permit
knowledge records, so the correct compliance conclusion remains “cannot be assessed.”

The updated run took 144.4 seconds, used 12 model calls and 24 tool calls, and returned no scope IDs,
connection details, usernames, passwords, or API keys. Remaining optimization work is to reduce repeated
mapping attempts and supporting diagnostic queries; these no longer affect correctness.
