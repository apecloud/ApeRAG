# FE legacy SDK inventory (Phase 1a/1b/1c input)

Snapshot of `main @ 526639f0` (post-#1607 Phase 1a + #1609 Phase 1b
batch 1). Numbers are produced by `rg` and are the authoritative
baselines referenced by
`tests/boundaries/web_{legacy_api,raw_schema,route_data}_allowlist.txt`.
Update this inventory whenever a Phase 1b PR lands so the remaining
migration workload stays visible.

## Headline counts (post-#1609 Phase 1b batch 1 baseline)

| Pattern | Files | Notes |
| --- | --- | --- |
| `from '@/api'` (legacy generated SDK) | 49 | Was 52 before #1607 (api-key), 50 before #1609 (prompt). Must reach 0 at Phase 1c. Fixture: `web_legacy_api_allowlist.txt`. |
| `from '@/api-v2/schema'` | 9 | `features/api-key/types.ts` joined the lock at #1607; `features/prompt/types.ts` joined at #1609 (aliases `PromptsPayload` / `UpdateUserPromptsRequest`). Fixture: `web_raw_schema_allowlist.txt` — exact lock. |
| `\b(defaultApi|apiClient|browserApiClient|createServerApiClient)\b` in `web/src/app/**` | 24 | Was 28 before #1607, 26 before #1609. Must reach 0 as each domain route switches to its `features/<d>/{server,client}-api`. Fixture: `web_route_data_allowlist.txt`. |

## Phase 1a sample — `api-key` (pinned by architect msg=e27e8b52) ✅ done

**Status: merged in #1607 (commit `8990329f`).** The `identity / api-key`
domain has moved to the canonical typed adapter pattern:

- `web/src/features/api-key/{types,client-api,server-api}.ts` exists
  and wraps the current `/api/v1/apikeys*` surface. No API path rename
  was made at this phase.
- Real caller migration landed for the three user-scoped files:
  `api-key-actions.tsx`, `api-key-table.tsx`, `page.tsx`.
- `test_api_key_feature_uses_v2_typed_api_boundary` in
  `tests/unit_test/test_web_typed_api_contract.py` guards the positive
  / negative contract, scoped to `app/workspace/api-keys/**` +
  `features/api-key/**` so the provider credential field is not
  touched.

Kept warning: the word "api key" is overloaded. `ApiKey` in
`app/workspace/api-keys/**` is user-scoped and belongs to `identity`.
`api_key` in `app/workspace/providers/**` and
`features/providers/types.ts` is an LLM provider credential and belongs
to `model_platform`. Later phases must keep the two domains separate.

## Phase 1b sequencing (low → high risk)

The legacy `@/api` allowlist split by domain. Numbers are current
allowlist entries, not post-migration targets.

| Batch | Domain | Legacy `@/api` files in batch | Notes |
| --- | --- | --- | --- |
| 1 | `identity` (`api-key`) | ~~2~~ 0 | ✅ done in #1607 (Phase 1a sample above). |
| 2 | `model_platform` / `prompt` | ~~1~~ 0 | ✅ done in #1609 (Phase 1b batch 1); `features/prompt/*` added, `prompts/prompt-settings.tsx` + `prompts/page.tsx` migrated. Technical-debt note: `GET /prompts/user` response lacks a concrete component and the `prompt_type` path param lacks an enum — both recorded as Phase 2 `model_platform` breaking-table input. |
| 3 | `governance` / `quota` | 2 (`quotas/page.tsx`, `quotas/quota-radial-chart.tsx`) | Low caller surface. |
| 4 | `governance` / `audit` | 4 (`audit-logs/*`, `admin/audit-logs/page.tsx`) | Includes an admin route. |
| 5 | `model_platform` / `providers` | `components/providers/app-provider.tsx` + `components/providers/bot-provider.tsx` | Shared provider shell; do not touch LLM provider `api_key` field. |
| 6 | `retrieval` (`search`) | 4 (`search-delete`, `search-result-drawer`, `search-table`, `search-test`) | URL stays under `/collections/{id}/search/*`. |
| 7 | `knowledge_graph` (`graph`) | 4 (`collection-graph*`, `collection-graph-node-*`, `collection-graph-showcase`) | URL stays under `/collections/{id}/graph*`. |
| 8 | `marketplace` | 5 (`marketplace/page.tsx`, `collection-list`, `collection-header`, `documents-table`, `document-detail`, `document-index-status`) | Consumes public knowledge_base contract only. |
| 9 | `knowledge_base` / `document` | 5 (`documents/*`, `document-upload`, `url-import`, `document-rebuild-index`, `document-index-status`) | Upload flow stays in scope; follow #30 regression guards. |
| 10 | `knowledge_base` / `collection` | 6 (`collections/*`, `collection-form`, `collection-header`, `collection-list`, `tools.ts`, `feature-visibility.ts`) | Large surface; may split into multiple PRs. |
| 11 | `conversation` / cross-domain shell | `components/chat/*` (9 files), `components/user-avatar.tsx`, `components/collections/export-dialog.tsx` | Ownership split for `components/chat/*` must happen before the caller migration; no wholesale directory move. |
| 12 | `identity` / admin | `admin/users/*`, `admin/configuration/*` | Admin caller surface. |
| 13 | `platform` / lib | `lib/api/client.ts`, `lib/api/server.ts` | Low-level HTTP shim; removed as part of Phase 1c. |

## Phase 1c — final SDK deletion gate

- `web/src/api/*` directory deleted.
- `rg "from '@/api(?!-v2)'" web/src` returns zero hits.
- `web/src/lib/api/client.ts` / `server.ts` deleted or replaced by
  canonical typed clients.
- `tests/boundaries/web_legacy_api_allowlist.txt` is empty.
- `tests/boundaries/web_route_data_allowlist.txt` is empty.
- `tests/boundaries/web_raw_schema_allowlist.txt` remains an **exact
  lock** of canonical typed-adapter files (not a shrinking list).
  Adding or removing a `features/<d>/types.ts` or a typed client under
  `lib/api/typed/` requires updating the allowlist in the same PR
  and explaining the change in the PR body.

## Running the inventory commands locally

```
rg -l "from ['\"]@/api['/]" web/src
rg -l "from ['\"]@/api-v2/schema['\"]" web/src
rg -l "\b(defaultApi|apiClient|browserApiClient|createServerApiClient)\b" web/src/app
```

The Phase 1b PR that migrates a batch must update this inventory in
the same commit so the remaining `@/api` file list is always
authoritative.
