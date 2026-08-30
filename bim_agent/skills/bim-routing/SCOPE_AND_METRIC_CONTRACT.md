# Scope and Metric Contract

Apply this contract to the runtime schema and the user's wording. Do not rely on a fixed category taxonomy, language, discipline, project, benchmark, or expected result.

## Population

- Bind requested nouns and qualifiers to exact observed schema values before executing a certifying query.
- Treat a category, family, type, system, spatial container, and physical instance as different population boundaries.
- When a general user term has multiple plausible observed populations, inspect the alternatives. Include all materially matching populations, report material alternatives separately, or ask one clarification; do not silently select the first or largest match.
- Preserve explicit user restrictions. Do not broaden or narrow the population merely because another scope is easier to query or reconcile.
- For grouped or exhaustive answers, define the universe, inclusion rules, exclusions, identity basis, and null handling before calculating values.

## Metric

- Bind comparative words to a measurable property and source basis before ranking or aggregating.
- Keep nominal/property dimensions, placement-derived dimensions, bounding-box axes, maximum extent, path/run length, area, bounding-box volume, and validated solid volume distinct.
- Keep design values, default/type values, instance values, and geometry-derived actual values distinct. If the question can reasonably refer to more than one, label each or ask for clarification.
- Preserve requested group dimensions and units. Do not merge groups across a requested dimension merely to simplify the output.
- If a required denominator, conversion factor, unit, standard threshold, or relationship is absent, return the supported numerator or observations and identify the missing input; do not manufacture the requested derived metric.

## Completion

Before finalizing, verify that the evidence covers every requested population, grouping dimension, metric, and relationship. A correct value for a narrower scope is not a complete answer to a broader request.
