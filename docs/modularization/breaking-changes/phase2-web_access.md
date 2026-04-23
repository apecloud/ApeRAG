# Phase 2a — `web_access` backend hard-cut breaking-change table

## 1. Summary

- Owner: `@chenyexuan`
- Reviewer(s): `@apecloud-backend`
- Linked task: `task #5` in `#模块化重构`
- Relies on: Phase 0 boundary fixtures + Phase 1a/1b FE typed-adapter base.
- Rollback strategy: revert this phase PR. No DB change, no shim — a
  single `git revert` restores the legacy `aperag/views/web.py` +
  `aperag/websearch/**` tree and the `view_models` Web\* classes.

## 2. API changes

| Old path | New path | Verb(s) | OpenAPI component / schema | FE adapter caller | Hurl file updated | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `/api/v1/web/search` | `/api/v2/web/search` | `POST` | `WebSearchRequest` → `WebSearchResponse` (unchanged component name / shape) | none — no FE caller consumes this endpoint today | `tests/e2e_http/hurl/full/18_web_access_http.hurl` (new) | route decorator stays relative (`@router.post("/web/search")`); version prefix migrated via `app.include_router(..., prefix="/api/v2")`. |
| `/api/v1/web/read` | `/api/v2/web/read` | `POST` | `WebReadRequest` → `WebReadResponse` (unchanged component name / shape) | none — no FE caller consumes this endpoint today | `tests/e2e_http/hurl/full/18_web_access_http.hurl` (new) | `aperag/mcp/server.py` updated to hit the new v2 paths. |

## 3. DB / SQLAlchemy changes

| Table / Model | Change | Migration revision | Downgrade | Owner domain | Notes |
| --- | --- | --- | --- | --- | --- |
| _n/a_ | no DB changes | _n/a_ | _n/a_ | `web_access` | Phase 2a is API + code layout only. |

## 4. Python import changes

| Old import path | New canonical path | Shim retained? | Shim deletion PR / phase | Notes |
| --- | --- | --- | --- | --- |
| `aperag.views.web` | `aperag.domains.web_access.api.routes` | No | _n/a — destructive hard-cut_ | file deleted, router re-mounted in `aperag/app.py`. |
| `aperag.websearch` (package) | `aperag.domains.web_access` (package) | No | _n/a_ | legacy `ReaderService` / `SearchService` re-exports deleted — no consumer outside the domain needs them. |
| `aperag.websearch.search.search_service` | `aperag.domains.web_access.search.search_service` | No | _n/a_ | |
| `aperag.websearch.search.base_search` | `aperag.domains.web_access.search.base_search` | No | _n/a_ | |
| `aperag.websearch.search.providers.duckduckgo_search_provider` | `aperag.domains.web_access.search.providers.duckduckgo_search_provider` | No | _n/a_ | |
| `aperag.websearch.search.providers.jina_search_provider` | `aperag.domains.web_access.search.providers.jina_search_provider` | No | _n/a_ | |
| `aperag.websearch.reader.reader_service` | `aperag.domains.web_access.reader.reader_service` | No | _n/a_ | `aperag/service/document_service.py::fetch_url_documents` updated. |
| `aperag.websearch.reader.base_reader` | `aperag.domains.web_access.reader.base_reader` | No | _n/a_ | |
| `aperag.websearch.reader.providers.jina_read_provider` | `aperag.domains.web_access.reader.providers.jina_read_provider` | No | _n/a_ | |
| `aperag.websearch.reader.providers.trafilatura_read_provider` | `aperag.domains.web_access.reader.providers.trafilatura_read_provider` | No | _n/a_ | |
| `aperag.websearch.utils.*` | `aperag.domains.web_access.utils.*` | No | _n/a_ | `content_processor`, `url_validator`. |
| `aperag.schema.view_models.WebSearchRequest` (and 6 sibling Web\* classes) | `aperag.domains.web_access.schemas.WebSearchRequest` (etc.) | No | _n/a_ | 7 classes relocated verbatim; OpenAPI component names + shapes byte-identical. |

## 5. FE changes

| Old module / identifier | New module / identifier | Consumer files migrated | Allowlist delta (`tests/boundaries/*.txt`) | Notes |
| --- | --- | --- | --- | --- |
| `/api/v1/web/search` + `/api/v1/web/read` path literals in `web/src/api-v2/schema.d.ts` | `/api/v2/web/search` + `/api/v2/web/read` path literals | _generated file only_ | no delta | Regenerated via `yarn api:v2:types` after `make openapi-generate`. No non-generated `web/src/**` file imports or calls these endpoints today. |

## 6. Tests / CI

- Unit tests added or updated: existing `tests/unit_test/websearch/**` and
  `tests/unit_test/test_document_service_fetch_url.py` imports retargeted to
  `aperag.domains.web_access.*` (test assertions unchanged). The boundary
  fixture `tests/unit_test/test_modularization_boundaries.py::test_aperag_domains_never_import_legacy_aggregate_modules`
  now actually scans `aperag/domains/web_access/**` and passes zero
  violations.
- Boundary tests touched: none of the `tests/boundaries/*.txt` allowlists
  shrink (web_access never had FE callers); the backend strict-ban test is
  now exercised for the first time.
- Hurl updated: `tests/e2e_http/hurl/full/18_web_access_http.hurl` (new,
  deterministic contract only; login + 4 validation asserts + logout).
- GitHub workflow jobs required to pass: `Unit-Test`,
  `e2e-http-smoke`.

## 7. Out of scope (explicit "not done in this phase")

- `/api/v1/chats/{chat_id}/search` stays on v1 and in its current owner;
  splitting it into `retrieval` is a later phase.
- All other `aperag/service/*` modules stay put; this phase only touches
  the single `fetch_url_documents` import in `aperag/service/document_service.py`.
- Provider-dependent live-fetch Hurl coverage for JINA / DuckDuckGo — to
  remain in the provider/full job only once written, never in smoke.
- Any FE caller migration — there is no FE caller today to migrate.

## 8. Risk / rollback log

- Known risk: any out-of-tree MCP consumer hitting the literal URL
  `/api/v1/web/*` will stop working. `aperag/mcp/server.py` is updated in
  the same commit so the in-tree MCP wiring is safe.
- Recovery / rollback plan: `git revert <this PR>` restores the legacy
  layout wholesale. No migration needed.
- Flaky Hurl / provider-dependent scoping decisions: `18_web_access_http.hurl`
  deliberately avoids live provider calls; see the file header comment
  and `docs/modularization/hurl-coverage-matrix.md` for the scoping rule.
