# Height, floor-area, façade, and dwelling-area regression trace

## Questions

1. What are the various section heights of the building?
2. What is the tower's maximum floor area?
3. What is the opening percentage of the tower facade?
4. Whats the ground floor space height?
5. How high is the model and is this allowed according to the documents?
6. Are there homes in the model that are smaller than 50m² gross floor area?

## Direct scoped evidence

- Storey reference elevations exist, but no section-height, clear-space-height, or model-height metric exists.
- Opening dimensions exist for some opening elements, but no compatible total façade-area denominator exists.
- No project-scoped permit knowledge or canonical requirement measures are available.
- Space areas have explicit bases: BVO, GO, NVO, and VVO.
- Apartment spaces use GO rather than BVO, so the gross-area threshold for individual homes is not assessable.
- The term `tower` is not explicitly represented in the model, so a tower-only floor maximum cannot be scoped safely.

## Improvements

- Added `area_basis` so gross/BVO cannot be conflated with GO, NVO, or VVO.
- Added a contract-backed `levels` entity exposing reference elevation while explicitly declaring it is not height.
- Added `maximum_group_sum` for general maximum-total-per-group questions such as maximum floor area.
- Added a compact capability catalog and direct use of trusted contract entities, reducing repeated discovery.
- Added evidence-backed insufficient-result limitations to the agent completion signal.
- Added a capability guard for missing height and façade-ratio operands.
- Added incomplete-measurement semantics: verified diagnostics remain visible, but the overall result is not labelled verified when the requested measurement cannot be assessed.

## Live outcomes

- Section heights: insufficient evidence; returned in about 6 seconds.
- Façade opening percentage: insufficient evidence; returned in about 2 seconds.
- Ground-floor space height: insufficient evidence; returned in about 2 seconds.
- Model height / document allowance: insufficient evidence; returned in about 2 seconds.
- Tower maximum floor area: insufficient evidence because tower scope is absent and area basis is ambiguous.
- Homes below 50 m² gross: the system verified that no apartment BVO records exist and correctly reports the gross-area question as not assessable; 105 apartment records use GO.

No user-visible result contains scope identifiers or credentials.
