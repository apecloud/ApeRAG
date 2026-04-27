# ApeRAG modularization — v2 destructive-first baseline

This directory captures the **`#模块化重构` v2** target baseline, the
deletion / migration plan, and the gates every Phase PR must satisfy.
It fixates what was agreed in the Slock `#模块化重构` thread
`81bc8be5` (Weston `94549083` + `dd802275` backend baseline, 符炫炜
`8c87e8cb` + `1a66bc6d` + `bffd2a02` FE evidence/mirror, @earayu2
`8850da66` destructive-first policy shift, architect PM rulings
`c5c00c20` / `e449c00b` / `8ea4414b` / `549bcac9`) so the Phase 1 / 2 / …
owners can execute against a single reviewable source of truth.

## Files in this directory

- [`target-domain-map.md`](./target-domain-map.md) — the 13 canonical
  backend domains and their FE mirrors.
- [`fe-legacy-sdk-inventory.md`](./fe-legacy-sdk-inventory.md) — the 49
  legacy `@/api` caller files grouped by domain, the 9 canonical
  `@/api-v2/schema` consumers, and the 24 `app/**` direct HTTP-client
  call sites (post-#1609 baseline; `main @ 526639f0`). Phase 1a/1b/1c
  sequencing input.
- [`hurl-coverage-matrix.md`](./hurl-coverage-matrix.md) — map of
  `tests/e2e_http/hurl/**` suites to domains and to GitHub workflow
  jobs (`e2e-http-smoke` / provider / EKS full).
- [`gate-checklist.md`](./gate-checklist.md) — per-phase PR-body
  requirements, local gates, and GitHub CI gates.
- [`roadmap.md`](./roadmap.md) — execution order, PR slicing,
  dependencies, and explicit non-goals for Phase 1b remaining through
  final cleanup.
- [`breaking-changes/phase-template.md`](./breaking-changes/phase-template.md) —
  the breaking-change table template every destructive phase must fill
  out before merge.

## Boundary tests

The boundary tests live in
[`tests/unit_test/test_modularization_boundaries.py`](../../tests/unit_test/test_modularization_boundaries.py)
and consume the allowlist fixtures in
[`tests/boundaries/`](../../tests/boundaries/):

| Test | Fixture | Target |
| --- | --- | --- |
| `test_web_no_legacy_api_import_outside_allowlist` | `web_legacy_api_allowlist.txt` | Shrinks to zero at Phase 1c; any new `@/api` caller fails. |
| `test_web_raw_schema_import_limited_to_typed_adapters` | `web_raw_schema_allowlist.txt` | Exact allowlist (currently 9 canonical typed-adapter files); new canonical typed adapters must update the allowlist in the same PR and explain in the PR body. |
| `test_web_app_routes_use_feature_adapters_only` | `web_route_data_allowlist.txt` | Shrinks to zero as each domain route is migrated to its `features/<d>/{server,client}-api`. |
| `test_aperag_domains_never_import_legacy_aggregate_modules` | — | Strict ban on `aperag/service.*`, `aperag/schema/view_models`, and `aperag/db/models` inside any `aperag/domains/<d>/**`. Passes trivially until the first domain is extracted. |

Phase PRs that claim to remove legacy surface but do not shrink an
allowlist are treated as `blocker` by review — every migration PR must
prove its delta with an allowlist delta.

## Non-goals for Phase 0

Phase 0 intentionally ships:

- documentation of the v2 target (this directory),
- baseline allowlists pinned to the exact state of `origin/main`,
- boundary tests that protect new canonical code,
- no runtime behaviour change.

Phase 0 does **not**:

- migrate any caller,
- delete any legacy module,
- change any OpenAPI path, DB schema, or Hurl payload,
- create empty `features/<d>/` directories just to mirror a backend
  domain (see `target-domain-map.md` for the `FE=none` entries).
