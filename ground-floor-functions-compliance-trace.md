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
