# Repository Working Agreement

## Scope

Keep runtime behavior schema-driven and project-agnostic. Do not encode dataset labels, benchmark questions, expected quantities, language-specific aliases, or one project's hierarchy as universal routing logic.

## Changes

- Preserve the distinction between planning constraints, project evidence, and external standards evidence.
- Prefer a reusable contract or schema-derived rule over a question-specific branch.
- Treat existing uncommitted files as user-owned work and avoid unrelated rewrites.
- Keep runtime instruction files concise because they are added to repeated model context.

## Verification

- Add behavior tests for observable contracts, not exact generated prose.
- Run focused tests for the changed path, then the full available suite.
- For evaluation changes, report semantic accuracy, errors, latency, tool/API calls, tokens, cost, and termination reasons separately.
- Audit evaluation references independently; never tune production behavior solely to reproduce an unverified expected answer.
