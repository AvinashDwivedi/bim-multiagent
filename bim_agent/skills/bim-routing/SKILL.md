# BIM Tool Routing Rules

Version: 1.2.0

These rules are runtime instructions for the BIM answer agent. They are deliberately kept outside the loop code so routing changes can be reviewed and tested independently.

## Evidence and completion

- Ground every factual or numeric answer claim with one or more inline tool-output references using exactly `[ref: <call_id>]`.
- For a claim supported by multiple observations, repeat the tag: `[ref: call_3] [ref: call_7]`. Do not put multiple IDs inside one tag. Prefer the stable `_citation_reference` supplied in each direct function-call output; opaque direct IDs and automatic reconciliation IDs are also valid. Do not invent a programmatic sub-call ID that was not supplied as a citation reference.
- Put references on every factual table row. Markdown headings and separator rows do not require references.
- A citation must point to a tool observation that contains the stated value or fact. A tool call by itself is not evidence.
- Project claims require project-tool evidence. `research_standards` is external standards evidence and cannot establish a fact about the project.
- `explore_object_scope` and `review_scope_and_evidence` are model-assisted advice, not project observations. A category, exclusion, count, or other project fact must also cite a direct project-tool observation, regardless of answer language.
- If a claim cannot be grounded in a project-tool observation, do not state it as fact. Issue another tool call to verify it, or explicitly tell the engineer that the information is unavailable or unverified. Never guess or present an ungrounded claim as sourced.
- Reconcile the selected population across the available tree, properties, and IFC identities before an exhaustive “all”/“every” claim, a ranking, a cross-source conclusion, or a compliance conclusion. A routine count over one clearly defined SQL population does not require reconciliation by itself. Reuse a completed reconciliation for the same selected population.
- When a pageable result has `returned_count < total_count` and a non-null `cursor`, use `fetch_more` before making a claim that requires the complete population. Do not use raw pagination to compute totals over a large population: after at most five continuation calls, use `query_bim_workspace` aggregate functions or narrow the inspection query.
- If the same tool with the same arguments has already been called in this run, reuse its observation instead of issuing the call again.

## Enforced compute path

| Work | Tool path |
| --- | --- |
| Filter, join, group, count, sum, average, minimum, or maximum over JSON-backed project records | `describe_bim_workspace`, then `query_bim_workspace` |
| Arithmetic using values already present in tool evidence, including conversions | `calculate` |
| Geometry, graph, statistical, or custom logic that cannot reasonably be expressed in SQL | `run_local_python` when the on-device Docker sandbox is ready, or a specialized IFC tool |

Do not use `calculate` to derive an aggregate directly from project records. Do not use local Python for an ordinary SQL-expressible filter, join, or aggregate. Make compact SQL return every grouped value and final total that the answer will cite; do not retrieve raw leaves merely to add their counts later.

## BIM domain checks

- Never assume a fixed tree depth identifies physical instances.
- When an element category, location, or property name is ambiguous, use `explore_object_scope` before querying the selected project population. Before finalizing an answer that depends on that scope decision, use `review_scope_and_evidence` with the selected scope, alternatives, exclusions, evidence, reconciliation, and draft.
- If project tools cannot resolve a material ambiguity, ask the engineer one concise clarifying question instead of guessing the scope.
- Preserve distinct dimension axes and units. IFC geometry uses the project units declared by `IfcUnitAssignment`; confirm those units before comparing IFC geometry with JSON exports. Treat JSON units as source-specific unless an explicit unit accompanies the value, and convert only after both source units are known.
- When sources disagree about the same property for the same resolved element identity, do not silently choose one value. Report the conflicting values with separate references and source labels, check whether units, identity mapping, or provenance resolves the disagreement, and otherwise mark it unresolved.
- Treat the loaded tree, properties, and IFC files as one analysis snapshot, not as proof that the files are current or mutually synchronized. If revision, export time, or synchronization metadata is absent, say that freshness/version alignment is unverified and scope conclusions to the loaded snapshot.
- Use `analyze_ifc_geometry` or `rank_ifc_geometry` for physical geometry questions.
- Use named IFC relationships and `analyze_ifc_graph` for reachability or connectivity.
- Treat reconciliation mismatches as unresolved evidence that must be investigated or disclosed.

## Standards evidence

- Keep external requirements separate from project observations.
- Cite the `research_standards` call for standards claims and cite project tools separately for compliance comparisons.
- If the standards result carries the unverified disclaimer, reproduce it verbatim in the final answer.
