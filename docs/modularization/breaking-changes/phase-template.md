# Phase `<N>` — `<short title>` breaking-change table

Copy this template into `docs/modularization/breaking-changes/phase-<N>-<slug>.md`
before opening the phase PR. Every row that cannot be filled with a
concrete path / identifier is a sign the phase is not ready.

## 1. Summary

- Owner: `<@mention>`
- Reviewer(s): `<@mention>`
- Linked task: `task #<N>` in `#模块化重构`
- Relies on: `<earlier phase>` / `<none>`
- Rollback strategy: `<revert this phase PR>` / `<destructive, no downgrade accepted>`

## 2. API changes

| Old path | New path | Verb(s) | OpenAPI component / schema | FE adapter caller | Hurl file updated | Notes |
| --- | --- | --- | --- | --- | --- | --- |

## 3. DB / SQLAlchemy changes

| Table / Model | Change | Migration revision | Downgrade | Owner domain | Notes |
| --- | --- | --- | --- | --- | --- |

## 4. Python import changes

| Old import path | New canonical path | Shim retained? | Shim deletion PR / phase | Notes |
| --- | --- | --- | --- | --- |

## 5. FE changes

| Old module / identifier | New module / identifier | Consumer files migrated | Allowlist delta (`tests/boundaries/*.txt`) | Notes |
| --- | --- | --- | --- | --- |

## 6. Tests / CI

- Unit tests added or updated: `<list>`
- Boundary tests touched: `test_modularization_boundaries.py` — deltas expected: `<describe>`
- Hurl updated: `<list of .hurl files>`
- GitHub workflow jobs required to pass: `Unit-Test`, `e2e-http-smoke`,
  `e2e-http-provider` (if applicable), `e2e-aperag-test` (if
  applicable).

## 7. Out of scope (explicit "not done in this phase")

- `<item 1>`
- `<item 2>`

## 8. Risk / rollback log

- Known risk: `<describe>`
- Recovery / rollback plan: `<describe>`
- Flaky Hurl / provider-dependent scoping decisions: `<describe>`
