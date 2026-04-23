# Per-phase gate checklist

Every Phase PR must satisfy the matching checklist before it leaves
`in_review`. Items listed as "hard gate" fail the merge; items listed
as "recorded in PR body" are not automatic blockers but must be present
so review can reason about them.

## Universal PR-body requirements

Every modularization PR body (any phase) must list:

- **Breaking changes** — every removed / renamed OpenAPI path, DB
  schema change, internal-API change, or FE public import change,
  explicitly enumerated. Absence of a breaking change is itself an
  explicit "no breaking changes in this PR" line.
- **Old paths removed / new canonical paths added** — both sides must
  appear even for intra-repo Python import renames.
- **Migrations** — Alembic revisions added or dropped, plus downgrade
  notes or "destructive, no downgrade accepted" when schema is
  destructive.
- **Hurl files updated** — list files under `tests/e2e_http/hurl/**`
  and which workflow job will run them (`e2e-http-smoke` /
  provider / `e2e-aperag-test`).
- **Tests run** — local `pytest` invocation, `yarn` steps if FE was
  touched, and the GitHub workflow jobs that must go green.
- **Temporary shims** — any short-lived re-export or wrapper introduced
  to keep the PR reviewable must name the phase / PR that deletes it.
- **Allowlist delta** — each destructive-first phase must call out the
  specific entries it removes from
  `tests/boundaries/web_*_allowlist.txt`. A PR that does not shrink any
  allowlist is treated as a blocker by review, not an effective
  modularization PR.

## Phase 0 — baseline (this PR)

Hard gates:

- `tests/boundaries/web_{legacy_api,raw_schema,route_data}_allowlist.txt`
  match the actual tree (no missing entries, no hidden offenders).
- `tests/unit_test/test_modularization_boundaries.py` green.
- Existing `tests/unit_test/test_web_typed_api_contract.py` still green.
- `make openapi-check` green — Phase 0 introduces no backend change;
  the exported spec must stay byte-identical.
- No change under `aperag/**` or `web/src/**` runtime code. Only docs,
  tests, and fixtures may be added / modified.

Recorded in PR body:

- Baseline counts (post-#1609, `main @ 526639f0`): 49 legacy `@/api`
  files, 9 canonical `@/api-v2/schema` consumers (exact allowlist,
  tracks new typed adapters in the same PR that adds them), 24
  `app/**` direct-client callers.
- Pointer to `docs/modularization/README.md` as the authoritative
  design baseline.

## Phase 1a — FE typed adapter skeleton + `api-key` sample

Hard gates:

- `web/src/features/api-key/{types,client-api,server-api}.ts` created
  and actually consumed by real route / page callers under
  `web/src/app/workspace/api-keys/**` (no adapter-only commit).
- `tests/unit_test/test_web_typed_api_contract.py` extended with an
  `api-key` positive + negative typed-contract test (match the
  `bot` / `collection` / `document` / `evaluation` pattern).
- `tests/boundaries/web_legacy_api_allowlist.txt` shrinks by exactly
  the `api-key` entries that the PR migrates.
- `yarn lint --quiet` clean; FE build / type behaves at least no worse
  than current `origin/main`.
- No OpenAPI path rename; no DB schema change; no other domain touched.

Recorded in PR body:

- Allowlist delta: `web/src/app/workspace/api-keys/api-key-actions.tsx`
  and `web/src/app/workspace/api-keys/api-key-table.tsx` removed
  (assuming Phase 0 baseline is in place before merge).
- Confirmation that `components/shared` and `services/cookies.ts` are
  out of scope.

## Phase 1b — per-domain caller migration batches

Hard gates (repeated per batch PR):

- `web_legacy_api_allowlist.txt` entries for the migrated domain are
  removed; the PR body lists the exact diff.
- No new `@/api` import in any file (`test_web_no_legacy_api_import_outside_allowlist`
  fails if it appears).
- No stretch into `components/shared` hard-move, no unrelated import
  churn, no route rename.
- `yarn lint --quiet` / `yarn typecheck` / `yarn test` (or current FE
  harness) as applicable.

## Phase 1c — legacy SDK deletion milestone

Hard gates:

- `web/src/api/` is deleted.
- `rg "from ['\"]@/api(?!-v2)['/]" web/src` returns zero.
- `tests/boundaries/web_legacy_api_allowlist.txt` is empty.
- `tests/boundaries/web_route_data_allowlist.txt` is empty.
- `web/src/lib/api/client.ts` / `server.ts` are either deleted or
  replaced by canonical typed-client entry points.
- GitHub `Unit-Test` + `e2e-http-smoke` green; no Hurl content change
  expected.

## Phase 2 — retrieval / knowledge_graph / web_access hard-cut

Hard gates:

- Breaking-change table under `docs/modularization/breaking-changes/`
  filled for each touched domain using
  `breaking-changes/phase-template.md`.
- `aperag/domains/{retrieval,knowledge_graph,web_access}/api/*` added
  with canonical routes; old `aperag/views/web.py` /
  `aperag/views/graph.py` / search routes removed (no shim).
- `tests/unit_test/test_modularization_boundaries.py::test_aperag_domains_never_import_legacy_aggregate_modules`
  still green (strict ban — new domain code never imports the legacy
  aggregates).
- OpenAPI regenerated; `features/search` / `features/graph` updated
  (`@/api-v2/schema` allowlist must stay exact); `web_access` FE stays
  `none / currently internal-only` unless a real caller appears.
- Hurl: `full/14_graph_http.hurl` updated to v2 paths; new
  `full/18_retrieval_http.hurl` added; deterministic
  contract Hurl for `/web/search` / `/web/read` added.
- GitHub `e2e-http-smoke` green; provider-dependent runs either green
  or explicitly scoped to provider/full with a justification.

## DB split phase (separate)

Hard gates:

- `aperag/platform/db/base.py` (or equivalent neutral location) houses
  the session + metadata registry.
- Pilot domain SQLAlchemy models moved under `aperag/domains/<d>/models/`.
- `Base.metadata.tables` diff reviewed and either equivalent or
  documented.
- Alembic `autogenerate` shows either no-op or a reviewed migration
  with downgrade notes.
- No runtime import cycle; relationship string references updated and
  verified at import time.

## Phase 4 — control-plane cleanup (identity / governance / model_platform / marketplace)

Hard gates:

- Each touched domain lands with breaking-change table + Hurl update.
- `tests/boundaries/web_legacy_api_allowlist.txt` reaches zero for the
  touched domain — no long-term allowlist residue allowed.
- Provider-dependent Hurl stays in provider/full job.

## Phase 5 — conversation / agent_runtime / evaluation cleanup

Hard gates:

- `agent_runtime` SSE / event / artifact shape unchanged unless
  explicitly listed in the breaking-change table.
- `evaluation` legacy `QuestionSet` / v1 evaluation references dropped
  only after tests / docs / FE references are updated.
- Full Hurl suite for the touched domain green on GitHub.

## Local gate commands (quick reference)

```
# Unit tests + typed contract + modularization boundaries
uv run --extra test pytest tests/unit_test -q

# OpenAPI freeze
make openapi-check

# FE (when touched)
cd web && yarn install && yarn lint --quiet && yarn typecheck
```
