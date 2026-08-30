# Identity and Reconciliation Policy

Use source identities according to their declared roles. Do not assume that identifiers from different exports have identical cardinality or semantics.

- Distinguish hierarchy nodes, property records, external instance identifiers, IFC entity identifiers, IFC global identifiers, type identifiers, and geometry products.
- A repeated, missing, or many-to-one identifier is evidence about mapping quality; by itself it is not proof that physical instances are duplicates.
- Deduplicate only on the identity basis selected for the question and supported by the source. State the basis when it materially affects the result.
- Compare independently selected populations using explicit source-owned identity sets. Do not create both sides of a reconciliation from one already-filtered result.
- Investigate a mismatch for scope, type-versus-instance confusion, absent mappings, unit differences, export revision, or relationship direction before describing it as a project inconsistency.
- Do not let an unresolved secondary mapping invalidate a complete single-source quantity unless the question requires cross-source identity agreement. Report the supported quantity and the separate mapping limitation.
- Agreement establishes population consistency for the compared identities, not source freshness, geometric validity, or engineering compliance.
