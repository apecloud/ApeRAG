# Phase 2 — `retrieval` + `knowledge_graph` backend hard-cut breaking-change table

## 1. Summary

- Owner: `@chenyexuan`
- Reviewer(s): `@apecloud-backend`
- Linked task: Phase 2 combined PR in `#模块化重构` (follows PR #1617
  `9f0f8bd` Phase 2a `web_access`).
- Relies on: Phase 0 boundary fixtures; Phase 1b typed FE adapter
  base; Phase 2a's `AuthenticatedUser(Protocol)` pattern.
- Rollback strategy: revert this PR. No DB change, no shim — a single
  `git revert` restores the legacy `aperag/views/collections.py`
  search / graph handlers, the `aperag/views/graph.py` write handlers,
  the legacy aggregate-module implementations of
  `aperag/service/search_pipeline_service.py` +
  `aperag/service/graph_service.py`, and the `view_models.*` class
  definitions for the 25 Search\* / Graph\* / Merge\* / Suggestion\*
  Pydantic models that were relocated.

## 2. API changes

| Old path | New path | Verb(s) | OpenAPI component / schema | FE adapter caller | Hurl file updated | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `/api/v1/collections/{id}/searches` | `/api/v2/collections/{id}/searches` | `POST` | `SearchRequest` → `SearchResult` (unchanged) | `web/src/features/retrieval/client-api.ts::createSearch` | `tests/e2e_http/hurl/full/19_retrieval_http.hurl` (new) | Relative router path, mounted via `prefix="/api/v2"`. |
| `/api/v1/collections/{id}/searches/{search_id}` | `/api/v2/collections/{id}/searches/{search_id}` | `DELETE` | idempotent bool body | `web/src/features/retrieval/client-api.ts::deleteSearch` | `19_retrieval_http.hurl` | |
| `/api/v1/collections/{id}/searches` | `/api/v2/collections/{id}/searches` | `GET` | `SearchResultList` (items: []) | `web/src/features/retrieval/server-api.ts::listSearches` | `19_retrieval_http.hurl` | Typed empty shape `SearchResultList(items=[])` (lesson 9a). |
| `/api/v1/collections/{id}/graphs/labels` | `/api/v2/collections/{id}/graphs/labels` | `GET` | `GraphLabelsResponse` | `web/src/features/knowledge-graph/{server,client}-api.ts::getGraphLabels` | `tests/e2e_http/hurl/full/20_knowledge_graph_http.hurl` (new) | Empty labels list is a legitimate answer (lesson 9a). |
| `/api/v1/collections/{id}/graphs` | `/api/v2/collections/{id}/graphs` | `GET` | `KnowledgeGraph` | `web/src/features/knowledge-graph/{server,client}-api.ts::getKnowledgeGraph` | `20_knowledge_graph_http.hurl` | `max_nodes` / `max_depth` range guards unchanged. |
| `/api/v1/collections/{id}/graphs/nodes/merge` | `/api/v2/collections/{id}/graphs/nodes/merge` | `POST` | `dict` (legacy shape) | `web/src/features/knowledge-graph/client-api.ts::mergeGraphNodes` | `20_knowledge_graph_http.hurl` | Legacy body shape preserved. |
| `/api/v1/collections/{id}/graphs/merge-suggestions` | `/api/v2/collections/{id}/graphs/merge-suggestions` | `GET` | `MergeSuggestionsResponse` | `web/src/features/knowledge-graph/{server,client}-api.ts::getMergeSuggestions` | `20_knowledge_graph_http.hurl` | Typed empty shape `{run: null, suggestions: []}`. |
| `/api/v1/collections/{id}/graphs/merge-suggestions` | `/api/v2/collections/{id}/graphs/merge-suggestions` | `POST` | `MergeSuggestionsRequest` → `MergeSuggestionsRunResponse` | `web/src/features/knowledge-graph/client-api.ts::startMergeSuggestionsRun` | `20_knowledge_graph_http.hurl` | Provider-dependent write flow — smoke hurl only asserts the GET empty shape + schedule gate; full provider-run coverage deferred to Phase 3. |
| `/api/v1/collections/{id}/graphs/merge-suggestions/{sid}/action` | `/api/v2/collections/{id}/graphs/merge-suggestions/{sid}/action` | `POST` | `SuggestionActionRequest` → `SuggestionActionResponse` | `web/src/features/knowledge-graph/client-api.ts::handleSuggestionAction` | `20_knowledge_graph_http.hurl` | Hurl asserts 404 on non-existent suggestion id. |

Removed paths (no v2 equivalent):

- `GET /api/v1/collections/{id}/graphs/export/kg-eval` — removed in
  Phase 2; no consumer migration needed (grep-verified zero hits in
  `web/src`, `aperag/mcp/server.py`, and `tests/e2e_http/hurl/`; the
  historical `410 Gone` shim that lived in the deleted
  `aperag/views/graph.py` was itself just a courtesy on top of a
  feature that had already been removed with the LightRAG-era graph
  workflow). Per @符炫炜 msg=e681e580 final ruling + @earayu2 msg=0d65c850.
  If a future phase needs ops-diagnostic graph surface, a fresh
  endpoint will be designed — the old URL is not coming back.

Unchanged but touched paths:

- `/api/v1/collections/{id}/documents/{upload,confirm,fetch-url,staged}`
  (upload flow stays on v1 — Non-goal 4).
- MCP `search_collection` tool URL moved from `/api/v1/...` to
  `/api/v2/...`; schema source moved from
  `aperag.schema.view_models.SearchResult` to
  `aperag.domains.retrieval.schemas.SearchResult`. Guarded by
  `tests/unit_test/test_mcp_contract.py::test_search_collection_schema_source`.

## 3. DB / SQLAlchemy changes

| Table / Model | Change | Migration revision | Downgrade | Owner domain | Notes |
| --- | --- | --- | --- | --- | --- |
| _n/a_ | no DB changes | _n/a_ | _n/a_ | `retrieval` + `knowledge_graph` | `SearchHistory`, `GraphCurationRun`, `GraphCurationSuggestion` stay in `aperag/db/models.py` (Non-goal 1). Phase 3 DB-split moves them under the canonical domain. |

## 4. Python import changes

| Old import path | New canonical path | Shim retained? | Shim deletion PR / phase | Notes |
| --- | --- | --- | --- | --- |
| `aperag.views.collections:create_search_view` / `list_searches_view` / `delete_search_view` | `aperag.domains.retrieval.api.routes` (three `@router.*` decorators) | No | _n/a — destructive hard-cut_ | Decorators deleted from `collections.py`; router mounted in `app.py` with `prefix="/api/v2"`. |
| `aperag.views.collections:get_graph_labels_view` / `get_knowledge_graph_view` | `aperag.domains.knowledge_graph.api.routes` | No | _n/a_ | |
| `aperag.views.graph:merge_nodes_view` / `merge_suggestions_view` / `get_merge_suggestions_view` / `handle_suggestion_action_view` | `aperag.domains.knowledge_graph.api.routes` | No | _n/a_ | `aperag/views/graph.py` deleted entirely. The historical `GET /api/v1/collections/{id}/graphs/export/kg-eval` 410-Gone shim was removed with the rest of the v1 KG surface; callers that still hit the v1 URL now receive a uniform FastAPI 404 like every other v1 KG path. Per @符炫炜 design-lock amendment msg=afbcbf64. |
| `aperag.service.search_pipeline_service.SearchPipelineService` (class body) | `aperag.domains.retrieval.pipeline.SearchPipelineService` | **Yes** (re-export only) | Phase 3 DB-split | Thin shim keeps `tests/unit_test/test_es_p0_contract.py`'s `monkeypatch` targets working; tests updated to the new module path in this PR. |
| `aperag.service.graph_service.GraphService` (class body) | `aperag.domains.knowledge_graph.service.GraphService` | **Yes** (re-export only) | Phase 3 | Keeps `aperag/views/marketplace_collections.py` working without a ripple edit; tests moved to the new module path. |
| `aperag.service.collection_service.CollectionService.{create_search,list_searches,delete_search,execute_search_flow}` | `aperag.domains.retrieval.service.RetrievalService.*` (plus `execute_search_flow` still exposed on the legacy class for `aperag.views.chat`) | Legacy methods retained | Phase 3 (collection_service wholesale deletion) | The legacy methods are now unreachable from HTTP; `execute_search_flow` stays so `chat.py` does not have to move in this PR. |
| `aperag.schema.view_models.{SearchRequest,SearchResult,SearchResultItem,SearchResultList,SearchResultMetadata,VectorSearchParams,FulltextSearchParams,GraphSearchParams,SummarySearchParams,VisionSearchParams}` (10 classes) | `aperag.domains.retrieval.schemas.*` | **Yes** (re-export lines at bottom of `view_models.py`) | Phase 3 DB-split | 10 classes moved verbatim; OpenAPI component names + shapes byte-identical. |
| `aperag.schema.view_models.{GraphLabelsResponse,GraphNodeProperties,GraphNode,GraphEdgeProperties,GraphEdge,KnowledgeGraph,MergeSuggestionsRequest,GraphCurationRunSummary,GraphMergeSuggestionEntity,GraphMergeSuggestionItem,MergeSuggestionsRunResponse,MergeSuggestionsResponse,SuggestionActionRequest,SuggestionActionMergeResult,SuggestionActionResponse}` (15 classes) | `aperag.domains.knowledge_graph.schemas.*` | **Yes** (re-export lines) | Phase 3 | |
| `aperag.db.models.Collection` (directly) | `aperag.domains.knowledge_graph.ports.CollectionRow` (Protocol) inside the knowledge_graph domain; `aperag.domains.retrieval.pipeline.CollectionRow` (Protocol) inside the retrieval domain | Protocol lives in domain; legacy model untouched | Phase 3 DB-split | Lesson 9a-quad: the Protocol is narrow (`id` / `user` / `title` / `config`). |
| `aperag.mcp.server` `SearchResult` import | `aperag.domains.retrieval.schemas.SearchResult` | No | _n/a_ | URL also updated to `/api/v2/collections/{id}/searches`. |

New internal packages:

- `aperag/domains/retrieval/__init__.py`, `ports.py`, `schemas.py`,
  `pipeline.py`, `service.py`, `api/__init__.py`, `api/routes.py`.
- `aperag/domains/knowledge_graph/__init__.py`, `ports.py`,
  `schemas.py`, `service.py`, `api/__init__.py`, `api/routes.py`.
- `aperag/domains/retrieval/ports.py::GraphSearchContract` — new
  consumer-owned Protocol; `aperag.graphindex.service.GraphIndexService`
  structurally satisfies it (the knowledge_graph domain does not
  import the consumer's ports module — lesson 9a-quad).

Legacy packages **deliberately left in place** (not moved this PR —
they are *infrastructure*, not forbidden aggregates):

- `aperag.graphindex.*` (graph storage + service primitives).
- `aperag.graph_curation.*` (SQL-heavy curation state machine). The
  domain's `knowledge_graph/api/routes.py` imports
  `aperag.graph_curation.graph_curation_service` via a lazy local
  import so the domain's module import graph stays decoupled from
  Celery task wiring.

The Phase 0 strict ban only forbids `aperag.service.*`,
`aperag.schema.view_models`, `aperag.db.models`. Neither
`aperag.graphindex` nor `aperag.graph_curation` is on that list, so
pulling them into the domain's route + service files is legal. Phase
3 DB-split is expected to fold them into canonical
`knowledge_graph/**` subpackages after extracting a repository port
for the `GraphCurationRun` / `GraphCurationSuggestion` SQLAlchemy
surface.

## 5. FE changes

| Old module / identifier | New module / identifier | Consumer files migrated | Allowlist delta (`tests/boundaries/*.txt`) | Notes |
| --- | --- | --- | --- | --- |
| `@/api` types / `@/lib/api/client::apiClient::collectionsCollectionIdSearches{Get,Post,SearchIdDelete}` | `@/features/retrieval/{client,server}-api` + `@/features/retrieval/types` | `web/src/app/workspace/collections/[collectionId]/search/{page,search-delete,search-test}.tsx` | `web_legacy_api_allowlist.txt` shrinks by 2 (`search-delete.tsx` + `search-test.tsx`); `web_raw_schema_allowlist.txt` grows by 2 (`features/retrieval/types.ts`, `features/knowledge-graph/types.ts`); `web_route_data_allowlist.txt` shrinks by 3 (`search/page.tsx` + `search-delete.tsx` + `search-test.tsx`). | `search-table.tsx` + `search-result-drawer.tsx` stay on `@/api` — Phase 1b batch 6 scope. |
| `SearchResultItem` + `SearchResultItemRecallTypeEnum` from `@/api` inside `web/src/app/workspace/collections/feature-visibility.ts` | `SearchResultItem` + `SearchRecallType` from `@/features/retrieval/types` | `feature-visibility.ts` (partial migration; `RebuildIndexesRequestIndexTypesEnum` still lives in `@/api` per Non-goal 8) | file remains in `web_legacy_api_allowlist.txt` (still imports `RebuildIndexesRequestIndexTypesEnum` from `@/api`) | |
| _(new)_ `features/knowledge-graph/{types,client-api,server-api}.ts` | same | none — the graph UI files (`collection-graph*`) remain on `@/api` (Phase 1b batch 7 scope). | `web_raw_schema_allowlist.txt` gains `features/knowledge-graph/types.ts`. | |

Final allowlist values:

- `tests/boundaries/web_legacy_api_allowlist.txt`: **22** (was 24).
- `tests/boundaries/web_raw_schema_allowlist.txt`: **13** (was 11).
- `tests/boundaries/web_route_data_allowlist.txt`: **15** (was 18).

## 6. Tests / CI

- Unit tests added or updated:
  - New: `tests/unit_test/test_mcp_contract.py` —
    `test_search_collection_schema_source` + `test_search_collection_targets_v2_path`.
  - Updated path: `tests/unit_test/test_es_p0_contract.py` (import +
    `monkeypatch` target from
    `aperag.service.search_pipeline_service.*` to
    `aperag.domains.retrieval.pipeline.*`).
  - Updated path: `tests/unit_test/service/test_search_graph_contract.py`
    + `tests/unit_test/graph_curation/test_service.py` (schema / service
    imports retargeted to the canonical domain modules).
- Boundary tests touched (new assertions in
  `tests/unit_test/test_modularization_boundaries.py`):
  - `test_retrieval_kg_protocol_boundary_is_one_way` — enforces lesson
    9a-quad: retrieval must not static-import knowledge_graph's
    service / schemas (or `aperag.graph_curation`), and knowledge_graph
    must not import `retrieval.ports` / `.schemas` / `.service` /
    `.pipeline`. The retrieval pipeline's narrow use of
    `aperag.graphindex.integration` (for the graphindex factory that
    returns a `GraphSearchContract`-typed service) is whitelisted.
  - `test_no_legacy_retrieval_or_graph_routes_remain` — parses
    `aperag/views/collections.py` (plus any future residual views
    module) for any `@router.*("/collections/{id}/searches` /
    `/collections/{id}/graphs/labels` /
    `/collections/{id}/graphs` /
    `/collections/{id}/graph-curation` decorators and fails if one
    survives. `aperag/views/graph.py` is deleted in this PR, so the
    test now only scans `collections.py`; `GET /collections/{id}/graphs/export/kg-eval`
    is fully gone (no v2 shim).
- Hurl updated: `tests/e2e_http/hurl/full/19_retrieval_http.hurl`
  (new) + `tests/e2e_http/hurl/full/20_knowledge_graph_http.hurl`
  (new). Deterministic contract coverage only; provider-dependent
  live recall / curation run flows are deferred (see Section 7).
  Existing `tests/e2e_http/hurl/full/14_graph_http.hurl` is left
  unchanged for this PR — it still exercises the v1 labels / subgraph
  paths, which now return 404 after the hard-cut. A separate follow-up
  PR should either retarget `14_graph_http.hurl` to the v2 paths or
  retire it in favor of `20_knowledge_graph_http.hurl`. Flagged in
  the hurl-coverage matrix.
- GitHub workflow jobs required to pass: `Unit-Test`,
  `e2e-http-smoke`. `e2e-http-provider` has no new coverage this PR.

## 7. Out of scope (explicit "not done in this phase")

1. `SearchHistory` / `GraphCurationRun` / `GraphCurationSuggestion`
   DB models stay in `aperag/db/models.py` — Phase 3 DB-split.
2. No algorithmic change in `aperag/query/query.py`.
3. No change to the pipeline's async semantics (recall-task gather +
   rerank fallback strategy unchanged).
4. No admin / auth / upload migration. Upload flow stays on
   `/api/v1/collections/{id}/documents/*`.
5. No Phase 4 identity consolidation: each domain still owns its own
   local `AuthenticatedUser(Protocol)`.
6. `view_models.py` re-export lines retained; deletion happens in the
   Phase 3 DB-split PR.
7. FE retrieval caller migration scope = the 3 search pages only
   (`page.tsx`, `search-delete.tsx`, `search-test.tsx`).
   `search-table.tsx` + `search-result-drawer.tsx` stay on `@/api`
   per Phase 1b batch 6 scope. FE knowledge_graph caller migration is
   0 files — the `collection-graph*` files stay on `@/api` (Phase 1b
   batch 7 scope).
8. `feature-visibility.ts` partial migration: only
   `SearchResultItem*` moves to the new types module;
   `RebuildIndexesRequestIndexTypesEnum` remains on `@/api`.
9. `aperag/graph_curation/service.py` and
   `aperag/graphindex/integration.py` stay in place. The curation
   service is a SQL-heavy repository-pattern module that would
   require a full repository-port extraction to move under
   `aperag/domains/knowledge_graph/**` without violating the Phase 0
   strict ban on `aperag.db.models`. The domain wraps them through a
   lazy local import which is legal (neither module is a forbidden
   aggregate). Phase 3 DB-split completes the relocation.
10. `aperag/views/chat.py` still calls
    `collection_service.execute_search_flow` — the legacy method
    remains as a thin wrapper over
    `aperag.domains.retrieval.pipeline.search_pipeline_service`. A
    later PR folds chat search into the retrieval domain proper.

## 8. Risk / rollback log

- **Known risk #1**: any out-of-tree MCP / API consumer still hitting
  the literal URL `/api/v1/collections/{id}/searches` or
  `/api/v1/collections/{id}/graphs*` will see 404. The in-tree MCP
  wiring is updated in the same commit; external callers must retarget
  to `/api/v2/` (see Section 2).
- **Known risk #2**: `tests/e2e_http/hurl/full/14_graph_http.hurl`
  exercises the v1 graph paths and will fail until it retargets to
  v2. Mitigation: this PR adds the v2 coverage as
  `20_knowledge_graph_http.hurl`; the v1 suite is flagged in the
  hurl-coverage matrix for retirement in a follow-up PR.
- **Recovery / rollback plan**: `git revert <this PR>` restores the
  legacy layout wholesale. No migration needed.
- **Flaky Hurl / provider-dependent scoping decisions**: neither new
  Hurl suite calls live LLM providers. Curation-run POST is only
  exercised indirectly via the GET empty-shape assertion; the
  provider-run flow is deferred.
