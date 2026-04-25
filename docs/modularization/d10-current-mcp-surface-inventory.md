# D10.a — Current MCP / RAG Surface Inventory + Gap Matrix

> **Task**: #83 (Phase 9 D10.a) — read-only, no code changes
> **Author**: @明书
> **Ground truth**: `origin/main` HEAD `51137301` (`refactor(phase8 #73 D8.1): backend AI SDK v5 stream emitter (#1695)`)
> **Worktree path**: `/Users/earayu/.slock/agents/6587495c-1e01-4c55-b9e0-bfb7c4330ffd/aperag-main/` (detached, agent-private)
> **Deliverable shape (locked by PM @架构师 / 总架构师 @符炫炜)**:
> - **Body** — 6-interface inventory (Vector / Graph / Full-text / Web Search / Summary / Vision)
> - **Appendix A** — D9 base reuse matrix
> - **Appendix B** — earayu2 三条 open question 影响面 1-page 表
> - **Inventory expansion (PM msg=7a954e9b)**: also catalog HTTP-only / internal-only capabilities not yet exposed via MCP, marked as gaps
> - **Access taxonomy (architect msg=112d3aad)**: every surface row tagged `MCP-exposed` / `HTTP-only` / `internal-only` / `none`

---

## 0. Reading guide

This document is a **state-of-the-codebase snapshot** intended to feed D10 design pack (task #82). It does NOT propose new APIs — that is design pack's job. It records:

1. What surface ApeRAG exposes today (Body + §C).
2. What D9 land already provides as base for D10 (Appendix A).
3. Where earayu2's three open questions land in current code (Appendix B).

Citations use `path:line` relative to the worktree root above. Every claim is grounded in code or doc reads at HEAD `51137301`.

---

## A. MCP server (current state)

### A.1 Construction & transport
- **File**: `aperag/mcp/server.py` (848 lines).
- **Framework**: `FastMCP("ApeRAG")` constructed at `aperag/mcp/server.py:32`.
- **Transport**: HTTP+SSE via FastMCP default. Does not currently emit explicit stdio config — runs as web service alongside FastAPI.
- **Backend bridge**: every MCP tool calls FastAPI on `http://localhost:8000` (`aperag/mcp/server.py:35`). No direct service-layer Python imports — purely HTTP proxy, which keeps domain tenancy / auth in one canonical place but pays an in-process round-trip.
- **Auth**: `get_api_key()` at `aperag/mcp/server.py:803-844`. Order: HTTP `Authorization: Bearer ...` header (extracted via `get_http_headers(include={"authorization"})`) → fallback to `APERAG_API_KEY` env. Raises `ValueError` if neither.

### A.2 Tools (`@mcp_server.tool`)

| Tool | Signature (params + defaults) | Returns | File:Line | Access tier |
|------|------------------------------|---------|-----------|-------------|
| `list_collections` | `()` | `Dict[str,Any]` (CollectionViewList shape) | `aperag/mcp/server.py:38-91` | MCP-exposed |
| `search_collection` | `(collection_id: str, query: str, use_vector_index=True, use_fulltext_index=True, use_graph_index=True, use_summary_index=True, use_vision_index=True, rerank=True, topk=5, query_keywords: list[str] \| None = None)` | `Dict[str,Any]` (SearchResult shape) | `aperag/mcp/server.py:94-256` | MCP-exposed |
| `search_chat_files` | `(chat_id: str, query: str, use_vector_index=True, use_fulltext_index=True, rerank=True, topk=5)` | `Dict[str,Any]` (SearchResult shape, chat-scoped) | `aperag/mcp/server.py:259-367` | MCP-exposed |
| `web_search` | `(query: str = "", max_results=5, timeout=30, locale="en-US", source: str = "")` | `Dict[str,Any]` (WebSearchResponse shape) | `aperag/mcp/server.py:370-479` | MCP-exposed |
| `web_read` | `(url_list: list[str], timeout=30, locale="en-US", max_concurrent=5)` | `Dict[str,Any]` (WebReadResponse shape) | `aperag/mcp/server.py:482-577` | MCP-exposed |

Tool count: **5**. Note that the current MCP shape **bundles all retrieval modes inside `search_collection`** via boolean flags — i.e. there are no per-mode MCP tools (`vector_search`, `graph_search`, `fulltext_search`, etc.); the discrimination is via parameter, not via tool surface.

### A.3 Resources & prompts

| Kind | URI / name | File:Line |
|------|-----------|-----------|
| Resource | `aperag://usage-guide` | `aperag/mcp/server.py:580-757` |
| Prompt | `search_assistant` | `aperag/mcp/server.py:760-800` |

### A.4 Authentication path (MCP → FastAPI)

```
MCP client (Claude Code / Codex / Cursor / ...)
  --(Bearer <api-key>)-->  FastMCP HTTP+SSE
     ↓ get_api_key()                     # mcp/server.py:803
     ↓ httpx.AsyncClient.<get|post>(...)
        Authorization: Bearer <api-key>
     → FastAPI handler in aperag/domains/<domain>/api/routes.py
        ↓ Depends(required_user) → AuthenticatedUser  # identity-domain protocol
        ↓ domain service.create_xxx(...)
           ↓ db_ops.query_collection(user, collection_id)  # canonical tenancy gate
```

Canonical tenancy gate is reused for every search-class tool (see B.1 line citations below). All MCP tools share one auth surface; user resolution happens in the FastAPI dependency layer, not in MCP.

---

## B. The 6 retrieval / search interfaces

For every interface: MCP exposure, HTTP endpoint(s), request/response schema, service entry, implementation file(s), tenancy boundary.

### B.1 Vector search

| Field | Value |
|-------|-------|
| **Access tier** | MCP-exposed (via `search_collection use_vector_index=True`) |
| **MCP entry** | `aperag/mcp/server.py:94-256` (param `use_vector_index: bool = True`) |
| **HTTP endpoint** | `POST /api/v2/collections/{collection_id}/searches` |
| **HTTP handler** | `aperag/domains/retrieval/api/routes.py:create_search_view` (~line 64) |
| **Request schema** | `SearchRequest` in `aperag/domains/retrieval/schemas.py`; vector branch = `VectorSearchParams { topk: Optional[int]; similarity: Optional[confloat(ge=0.0, le=1.0)] }` |
| **Response schema** | `SearchResult { id, query, vector_search, items: list[SearchResultItem], created }`; per-item `recall_type="vector_search"` |
| **Service entry** | `RetrievalService.create_search()` at `aperag/domains/retrieval/service.py:80-149` |
| **Pipeline** | `SearchPipelineService.execute_search()` in `aperag/domains/retrieval/pipeline.py` (vector branch around `_vector_search()`) |
| **Embedding** | `aperag/llm/embed/base_embedding.py` — `get_collection_embedding_service_sync()` |
| **Store** | `aperag/vectorstore/` (Qdrant primary, Pinecone/Milvus adapters) |
| **Tenancy gate** | `aperag/domains/retrieval/service.py:96-106` — `db_ops.query_collection(user, collection_id)` raises `CollectionNotFoundException` on miss; marketplace fallback resolves owner for provider-key lookup but still scopes data to the original collection's owner |

### B.2 Graph search

| Field | Value |
|-------|-------|
| **Access tier** | MCP-exposed (via `search_collection use_graph_index=True`) |
| **MCP entry** | `aperag/mcp/server.py:94-256` (param `use_graph_index: bool = True`) |
| **HTTP endpoint** | `POST /api/v2/collections/{collection_id}/searches` (shared) |
| **Request schema** | `GraphSearchParams { topk: Optional[int] }` |
| **Response shape** | Same `SearchResult` envelope; per-item `recall_type="graph_search"` |
| **Service entry** | Same `RetrievalService` → pipeline `_graph_search()` |
| **Provider** | `aperag/domains/knowledge_graph/graphindex/integration.py: make_service_for_collection(collection)` (called from pipeline ~line 86) |
| **Implementation** | `aperag/domains/knowledge_graph/graphindex/service.py: query_context()` per `GraphSearchContract` (Protocol) in `aperag/domains/retrieval/ports.py` |
| **Models** | `aperag/domains/knowledge_graph/db/models.py`, `aperag/domains/knowledge_graph/graphindex/models.py` |
| **Tenancy gate** | Same `db_ops.query_collection(user, collection_id)` at `aperag/domains/retrieval/service.py:96-106` |
| **Adjacent read APIs (HTTP-only)** | `GET /api/v2/collections/{cid}/graphs` + `/graphs/labels` — see C.4 |
| **Adjacent write APIs (HTTP-only)** | `/graphs/nodes/merge`, `/graphs/suggestions`, `/graphs/suggestions/{id}/apply` — see C.4 |

### B.3 Full-text search

| Field | Value |
|-------|-------|
| **Access tier** | MCP-exposed (via `search_collection use_fulltext_index=True`) |
| **MCP entry** | `aperag/mcp/server.py:94-256` (param `use_fulltext_index: bool = True`) |
| **HTTP endpoint** | `POST /api/v2/collections/{collection_id}/searches` (shared) |
| **Request schema** | `FulltextSearchParams { topk: Optional[int]; keywords: Optional[list[str]] }` (custom keywords override auto-extraction) |
| **Response shape** | Same `SearchResult` envelope; per-item `recall_type="fulltext_search"` |
| **Service entry** | Same `RetrievalService` → pipeline FT branch |
| **Implementation** | `aperag/domains/indexing/fulltext_index.py` (~31KB; `build_fulltext_index`, `search_fulltext_index`, keyword extraction) |
| **Index name** | `aperag/utils/utils.py: generate_fulltext_index_name()` |
| **Backend** | Elasticsearch (and/or Postgres FTS via collection config) |
| **Tenancy gate** | Same as B.1 |

### B.4 Web search

| Field | Value |
|-------|-------|
| **Access tier** | MCP-exposed (separate `web_search` tool; not bundled into `search_collection`) |
| **MCP entry** | `aperag/mcp/server.py:370-479` |
| **HTTP endpoint** | `POST /api/v2/web/search` |
| **HTTP handler** | `aperag/domains/web_access/api/routes.py: web_search_endpoint()` (~line 87) |
| **Request schema** | `WebSearchRequest { query: Optional[str], max_results, timeout, locale, source }` (`aperag/domains/web_access/schemas.py`) |
| **Response schema** | `WebSearchResponse { query, results: List[WebSearchResult], total_results, search_time, meta: WebSearchMeta }`; per-result `{ rank, title, url, snippet, domain, publish_date? }`; meta carries `provider_used`, `backend_used`, `fallback_used`, `error_code` |
| **Provider chain** | `_search_with_jina_fallback()` at handler ~line 117 — JINA primary (per-user API key), DuckDuckGo fallback |
| **Per-user provider key** | `_get_user_jina_api_key(user)` (~line 231) → `aperag/db/ops.py: query_provider_api_key()` |
| **Tenancy gate** | `Depends(required_user)` at handler entry; per-user provider key isolates external API quotas / billing |

### B.5 Summary search

| Field | Value |
|-------|-------|
| **Access tier** | MCP-exposed (via `search_collection use_summary_index=True`) — **note**: front-end has hidden the toggle, but backend pathway is fully active |
| **MCP entry** | `aperag/mcp/server.py:94-256` (param `use_summary_index: bool = True`) |
| **HTTP endpoint** | `POST /api/v2/collections/{collection_id}/searches` (shared) |
| **Request schema** | `SummarySearchParams { topk: Optional[int]; similarity: Optional[confloat(ge=0.0, le=1.0)] }` |
| **Response shape** | Same `SearchResult` envelope; per-item `recall_type="summary_search"` |
| **Indexer** | `aperag/domains/indexing/summary_index.py` (~18KB) |
| **Generation trigger (HTTP-only, NOT MCP)** | `POST /api/v2/collections/{cid}/summary/generate` → `generate_collection_summary_view()` at `aperag/domains/knowledge_base/api/routes.py:168` |
| **Status model** | `CollectionSummary` at `aperag/domains/knowledge_base/db/models.py:127+` (PENDING / GENERATING / COMPLETE / FAILED) |
| **Tenancy gate** | Same as B.1 |
| **FE-hidden status** | Backend infra fully operational; FE has merely removed the user-facing toggle. Removing this from MCP would be code change, not just config. |

### B.6 Vision search

| Field | Value |
|-------|-------|
| **Access tier** | MCP-exposed (via `search_collection use_vision_index=True`) — same FE-hidden state as Summary |
| **MCP entry** | `aperag/mcp/server.py:94-256` (param `use_vision_index: bool = True`) |
| **HTTP endpoint** | `POST /api/v2/collections/{collection_id}/searches` (shared) |
| **Request schema** | `VisionSearchParams { topk: Optional[int]; similarity: Optional[confloat(ge=0.0, le=1.0)] }` |
| **Response shape** | Same `SearchResult` envelope. Vision-recalled items carry `metadata["indexer"] == "vision"` and an asset URL of the form `asset://{asset_id}?document_id=...&collection_id=...&mime_type=...` (constructed at `aperag/mcp/server.py:185-200`). If `item.content` is empty → multimodal-embedding-only recall; if non-empty → contains Vision LLM description. |
| **Indexer** | `aperag/domains/indexing/vision_index.py` (~15KB). Two indexing paths (multimodal embedding + vision-to-text), with dedup logic in pipeline (~lines 91-123). |
| **Image extraction** | `aperag/docparser/image_parser.py` |
| **Asset retrieval** | `GET /api/v2/collections/{cid}/documents/{did}/object?path=...` (HTTP-only; supports Range / 206) — see C.1 |
| **Tenancy gate** | Same as B.1 |
| **FE-hidden status** | Same as Summary — toggle hidden in FE, full backend infra still wired into search pipeline & object store. |

### B-summary

- **MCP exposure for retrieval**: 5 tools total. Vector / Graph / Full-text / Summary / Vision are bundled inside `search_collection`. Web is its own tool. There are no individual `vector_search`, `graph_search`, `fulltext_search` MCP tools — just one omnibus tool with boolean flags.
- **Single shared HTTP endpoint** (`POST /collections/{cid}/searches`) handles 5 of 6 retrieval modes. Web search is its own endpoint.
- **All 6 share one canonical tenancy gate**: `RetrievalService` (and equivalent for web_access) calling `query_collection(user, collection_id)`.
- **Summary + Vision are FE-hidden but backend-live**: omitting them in D10 needs an explicit deprecation decision (see Appendix B-1).

---

## C. Capabilities in HTTP API or service layer NOT exposed via MCP (gaps)

Per PM msg=7a954e9b + architect msg=112d3aad, every HTTP / internal capability that could plausibly be a future MCP tool is enumerated here, tagged.

### C.1 Document operations

| Operation | HTTP endpoint | Handler (`aperag/domains/knowledge_base/api/routes.py`) | Access tier | D10 candidate? |
|-----------|--------------|--------------------------------------------------------|-------------|---|
| List documents | `GET /api/v2/collections/{cid}/documents` | `list_documents_v2_view` (~line 262) | HTTP-only | **Yes** — primary `list` primitive |
| Get document | `GET /api/v2/collections/{cid}/documents/{did}` | `get_document_v2_view` (~line 295) | HTTP-only | **Yes** — metadata primitive |
| Get preview | `GET /api/v2/collections/{cid}/documents/{did}/preview` | `get_document_preview_v2_view` (~line 346) | HTTP-only | **Yes** — first-pass read (returns `DocumentPreview`: title, first N chars, metadata) |
| Download original | `GET /api/v2/collections/{cid}/documents/{did}/download` | `download_document_v2_view` (~line 309) | HTTP-only | Maybe — original bytes not LLM-friendly; decide per security model |
| Object / asset | `GET /api/v2/collections/{cid}/documents/{did}/object?path=...` | `get_document_object_v2_view` (~line 359) | HTTP-only | **Yes** — image / asset retrieval (Vision result trail) |
| List staged | `GET /api/v2/collections/{cid}/documents/staged` | `list_staged_documents_v2_view` (~line 286) | HTTP-only | Maybe — admin-ish |
| Confirm staged | `POST /api/v2/collections/{cid}/documents/confirm` | `confirm_documents_v2_view` (~line 411) | HTTP-only | No — write op, deferred to D9 consent lane |
| Create | `POST /api/v2/collections/{cid}/documents` | `create_documents_v2_view` (~line 252) | HTTP-only | No — write op |
| Upload | `POST /api/v2/collections/{cid}/documents/upload` | `upload_document_v2_view` (~line 402) | HTTP-only | No — write op |
| Delete one | `DELETE /api/v2/collections/{cid}/documents/{did}` | `delete_document_v2_view` (~line 320) | HTTP-only | No — write op |
| Delete many | `DELETE /api/v2/collections/{cid}/documents` | `delete_documents_v2_view` (~line 331) | HTTP-only | No — write op |
| Fetch URL | `POST /api/v2/collections/{cid}/documents/fetch-url` | `fetch_url_document_v2_view` (~line 422) | HTTP-only | Maybe — write op |
| Rebuild indexes | `POST /api/v2/collections/{cid}/documents/{did}/rebuild_indexes` | `rebuild_document_indexes_v2_view` (~line 375) | HTTP-only | Maybe — admin / control plane |

`Document` model: `aperag/domains/knowledge_base/db/models.py:160-221`. Object-store layout helper at `:209-211`: `f"user-{user.replace('|','-')}/{collection_id}/{id}"`.

### C.2 Collection operations

| Operation | HTTP endpoint | Handler | Access tier | D10 candidate? |
|-----------|--------------|---------|-------------|----------------|
| Create collection | `POST /api/v2/collections` | `create_collection_view` (~line 108) | HTTP-only | No — write op |
| Get collection | `GET /api/v2/collections/{cid}` | `get_collection_view` (~line 130) | HTTP-only | **Yes** — only `list_collections` is MCP, no `get_collection` tool |
| Update | `PUT /api/v2/collections/{cid}` | `update_collection_view` (~line 141) | HTTP-only | No — write op |
| Delete | `DELETE /api/v2/collections/{cid}` | `delete_collection_view` (~line 152) | HTTP-only | No — write op |
| Sharing status | `GET /api/v2/collections/{cid}/sharing` | `get_collection_sharing_status_view` (~line 198) | HTTP-only | Maybe |
| Publish | `POST /api/v2/collections/{cid}/sharing` | `publish_collection_sharing_view` (~line 214) | HTTP-only | No — write op |
| Unpublish | `DELETE /api/v2/collections/{cid}/sharing` | `unpublish_collection_sharing_view` (~line 231) | HTTP-only | No — write op |
| Rebuild failed indexes | `POST /api/v2/collections/{cid}/rebuild_failed_indexes` | `rebuild_failed_indexes_v2_view` (~line 390) | HTTP-only | Maybe — admin |
| Trigger summary | `POST /api/v2/collections/{cid}/summary/generate` | `generate_collection_summary_view` (~line 168) | HTTP-only | Maybe — write op (LLM cost) |
| Test MinerU token | `POST /api/v2/collections/test-mineru-token` | `test_mineru_token_view` (~line 93) | HTTP-only | No — admin / config |

`Collection` model: `aperag/domains/knowledge_base/db/models.py:112-125`.

### C.3 Chunk-level / outline / structure (largest gap — internal-only)

This is the most consequential gap: ApeRAG's docparser, indexers, and graph layer all maintain rich chunk + heading-tree data, but **none of it has a stable read surface** — neither HTTP nor MCP.

| Concept | Where it exists | Access tier | Notes |
|---------|----------------|-------------|-------|
| Chunk text + metadata | Vector DB record metadata + Elasticsearch `_source` + LightRAG `LightRAGDocChunksModel` (PG row) | internal-only | No public schema; chunk IDs are per-store private |
| Chunk → source-line map (`md_source_map`) | Computed in `aperag/docparser/parse_md.py` and propagated through chunk metadata | internal-only | Used internally for citation rendering; no read API |
| Heading tree / outline | Constructed during parse: `aperag/docparser/parse_md.py` (heading detection) → `aperag/docparser/chunking.py:_to_groups()` (`Group{title_level, title, items}`, ~line 43) | internal-only | Never persisted as a structure; recomputable from re-parse |
| Section / heading-anchor handle | Implicit only (heading text + level inside chunk metadata) | none (no stable handle) | D10 R1 lock requires this — needs to be made first-class |
| Chunk-by-id read | None | none | D10 read primitive will need a new chunk-id contract |
| Document outline read | None | none | Cheapest D10 primitive to add (compute on demand from existing parser) |

**Implication for D10**: of the four candidate read primitives in @符炫炜's sketch (`read_document(range)`, `read_document_section`, `read_document_chunk`, `read_document_outline`), only `read_document(full)` has any current backing (via `download` / `preview`). Section / chunk / outline all require new persistence or on-the-fly computation.

### C.4 Graph operations

| Operation | HTTP endpoint | Handler (`aperag/domains/knowledge_graph/api/routes.py`) | Access tier | D10 candidate? |
|-----------|--------------|---------------------------------------------------------|-------------|----------------|
| Get graph | `GET /api/v2/collections/{cid}/graphs?label=*&max_nodes=1000&max_depth=3` | `get_knowledge_graph_view` (~line 102) | HTTP-only | **Yes** — graph navigation primitive |
| Get labels | `GET /api/v2/collections/{cid}/graphs/labels` | `get_graph_labels_view` (~line 74) | HTTP-only | Maybe — discovery aid |
| Merge nodes | `POST /api/v2/collections/{cid}/graphs/nodes/merge` | `merge_nodes_view` (~line 131) | HTTP-only | No — write op (D9 consent gate required) |
| Suggest merge | `POST /api/v2/collections/{cid}/graphs/suggestions` | `suggest_merge_entities_view` (~line 172) | HTTP-only | Maybe — read-mostly suggestion |
| Apply suggestion | `POST /api/v2/collections/{cid}/graphs/suggestions/{id}/apply` | `apply_suggestion_view` (~line 227) | HTTP-only | No — write op |

### C.5 Bot / agent / chat (out of scope for D10 lower-layer, listed for completeness)

| Operation | HTTP endpoint | Handler | Access tier |
|-----------|--------------|---------|-------------|
| Bots CRUD | `/api/v2/bots[...]` | `aperag/domains/conversation/api/routes.py:221+` | HTTP-only |
| Chats CRUD + title gen | `/api/v2/chats[...]` | `aperag/domains/conversation/api/routes.py:278+` | HTTP-only |
| Agent turn create / get / cancel | `/api/v2/agent/chats/{cid}/turns[...]` | `aperag/domains/agent_runtime/api/routes.py:105+` | HTTP-only |
| Stream turn events (SSE) | `GET /api/v2/agent/chats/{cid}/turns/{tid}/events` | `aperag/domains/agent_runtime/api/routes.py:173` | HTTP-only |
| Get artifact | `GET /api/v2/agent/artifacts/{aid}` | `aperag/domains/agent_runtime/api/routes.py:145` | HTTP-only |

D10 scope per user / architect framing is the **lower (substrate) layer**; agent runtime + chat lifecycle stay in D9 / agent layer above the MCP surface.

### C.6 Marketplace operations

| Operation | HTTP endpoint | Handler (`aperag/domains/marketplace/api/routes.py`) | Access tier | D10 candidate? |
|-----------|--------------|------------------------------------------------------|-------------|----------------|
| List public collections | `GET /api/v2/marketplace/collections` | `list_marketplace_collections` (~line 62) | HTTP-only | Maybe — extends `list_collections` |
| List subscribed | `GET /api/v2/marketplace/subscriptions` | `list_user_subscribed_collections` (~line 79) | HTTP-only | Maybe |
| Subscribe | `POST /api/v2/marketplace/subscriptions/{cid}` | `subscribe_collection` (~line 94) | HTTP-only | No — write op |
| Unsubscribe | `DELETE /api/v2/marketplace/subscriptions/{cid}` | `unsubscribe_collection` (~line 114) | HTTP-only | No — write op |
| Get marketplace collection | `GET /api/v2/marketplace/collections/{cid}` | `get_marketplace_collection` (~line 131) | HTTP-only | Maybe |
| List marketplace docs | `GET /api/v2/marketplace/collections/{cid}/documents` | ` list_marketplace_collection_documents` (~line 148) | HTTP-only | Maybe |
| Marketplace doc preview | `GET /api/v2/marketplace/collections/{cid}/documents/{did}/preview` | (~line 202) | HTTP-only | Maybe |
| Marketplace doc object | `GET /api/v2/marketplace/collections/{cid}/documents/{did}/object` | (~line 232) | HTTP-only | Maybe |
| Marketplace graph | `GET /api/v2/marketplace/collections/{cid}/graphs` | `get_marketplace_collection_graph` (~line 261) | HTTP-only | Maybe |

Marketplace is the **cross-tenant boundary** — earayu2's open question on cross-collection ops (Appendix B-3) lives here.

### C.7 Evaluation / quality

`aperag/domains/evaluation/api/routes.py` — datasets, runs, items, retries (~14 endpoints). All HTTP-only. **Out of scope for D10** (meta-layer; not a document-substrate primitive).

### C.8 Governance / admin / model platform

| Category | Endpoints | Routes file | Access tier |
|----------|-----------|-------------|-------------|
| API keys | `/api/v2/apikeys/*` (create/list/rotate/revoke) | `aperag/domains/governance/api/apikeys_routes.py` | HTTP-only |
| Audit logs | `/api/v2/audit-logs[...]` | `aperag/domains/governance/api/audit_routes.py` | HTTP-only |
| Quota / system defaults | `/api/v2/quota[...]` | `aperag/domains/governance/api/quota_routes.py` | HTTP-only |
| Provider config | `/api/v1/providers[...]` | `aperag/domains/model_platform/api/providers_routes.py` | HTTP-only |

These are admin-plane; out of scope for D10's document-substrate redesign.

### C-summary (gap topology)

The gap matrix concentrates in three buckets, ordered by D10 value:

1. **Document discovery / read primitives** (C.1 list+get+preview+object, C.3 outline/chunks) — highest D10 value; @符炫炜 D10 sketch §A's "Direct content access" cluster lives here. C.3 is partially **internal-only**, requiring new schemas.
2. **Collection metadata** (C.2 get_collection, C.4 graph nav) — medium D10 value; small additions on top of existing HTTP.
3. **Marketplace cross-tenant** (C.6) — gates earayu2's Q3 (cross-collection / cross-tenant in Appendix B-3).

Buckets C.5 / C.7 / C.8 are explicitly out of D10 lower-layer scope.

---

## D. Document storage & parsing layer

### D.1 What is persisted?

| Artifact | Persistence | Where |
|---------|-------------|-------|
| Original file | object store | `user-<user>/<collection_id>/<document_id>/...` (`Document.object_store_base_path()` at `aperag/domains/knowledge_base/db/models.py:209-211`) |
| Parsed markdown | **NOT persisted** as standalone artifact | Re-derived from object store on demand by docparser |
| Vector chunks | vector DB (Qdrant / Milvus / Pinecone via `aperag/vectorstore/`) | embedding + metadata only |
| Full-text chunks | Elasticsearch / Postgres FTS | text + position metadata |
| Graph chunks / entities / relations | LightRAG: `LightRAGDocChunksModel`, `LightRAGVDBEntityModel`, `LightRAGVDBRelationModel` (Postgres) | per-collection KG store |
| Summary text | `CollectionSummary.summary` column at `aperag/domains/knowledge_base/db/models.py:127+` | one row per collection |
| Vision asset | object store (same path as original) | image binary + Vision-LLM description embeddings in vector DB |

**Key implication**: There is no canonical "parsed document" record that a `read_document(section)` primitive can read. To support D10 R1's stable handles (`section_path` / `chunk_id` / `heading_anchor`) without re-parsing on every call, D10 will need either:
- (a) **Persist the parsed document tree** (new domain table), or
- (b) **Cache parse output** in an LRU layer keyed by `(document_id, parse_version)`.

Re-parsing on each call is doable for small docs but expensive for PDFs / OCR'd scans.

### D.2 Outline / heading tree

- Markdown parser at `aperag/docparser/parse_md.py` (~18KB) extracts heading levels + text.
- Chunking at `aperag/docparser/chunking.py:29-100`: `Group(title_level, title, items)` (`_to_groups()` ~line 43) builds hierarchical groups from `Part` list.
- **No public schema or API** exposes this tree.

### D.3 Object-store layout

`f"user-{user.replace('|','-')}/{collection_id}/{document_id}"` (the `|` → `-` replacement normalises the auth-provider delimiter that appears in user IDs like `oauth|sub`). Backends abstracted by `aperag/objectstore/` (S3, OSS, ABS, local fs).

---

## Appendix A — D9 base reuse matrix

Goal: map D8 / D9 primitives that D10 either reuses directly or extends. Citations follow D8 / D9 design canon + on-disk code.

| Primitive | Design canon | Code location at `51137301` | D10 disposition |
|-----------|--------------|------------------------------|-----------------|
| **`SafeToolName` resolver** (D9 §A1 + §A6) | `docs/modularization/agent-message-protocol-design.md:98-129`, `docs/modularization/agent-runtime-mcp-design.md` (D9 SSoT) | `aperag/domains/agent_runtime/wire/translator.py:82` (`SafeToolNameResolver = Callable[[str], tuple[str, dict[str, Any]]]`); plumbing at lines 107, 111-120, 286, 341, 417 (parts.py also references it). **Resolver implementation pending — explicit `TODO(#75 chenyexuan)` at translator.py:120.** | **Reuse for MCP tool naming**. Every D10 MCP tool surfaces with a SafeName; metadata `{mcpServer, mcpToolName}` per AI SDK part. Need to extend with **collision registry table** (proposed schema `MCPToolCache.safe_name` per D9 design — not yet on disk). |
| **3-tier registry** (D9 §1.1 + §A5) | `docs/modularization/agent-runtime-mcp-design.md` §1.1 + §A5 | **Not yet on disk** — design canonical only. Proposed tables `MCPServer`, `MCPToolCache` (scope ∈ system/bot/user; reserved system namespace; collision rejection / quarantine). | **D10 needs registry table + resolution logic**. system tier is pre-populated by ApeRAG (`aperag-knowledge-base`, `aperag-web-search`); bot/user tiers later. **D10 lower-layer surface = system tier only**; bot/user tiers fall under D9 implementation lane. |
| **7-point contract** (D8.3 owner enforcement) | `docs/modularization/agent-message-protocol-design.md:159-166` — exact 7 lines:<br>① SafeToolName + MCP metadata (A1 + A6)<br>② AI SDK v5 + custom `data-tool-consent` (A2)<br>③ `argsPreview` + `argsHash` — raw args backend-private (A7)<br>④ Registry no silent system override (A5)<br>⑤ `data-elicitation` schema-validated input (§5)<br>⑥ Three-level authorization (§2)<br>⑦ PydanticAI as default runtime backbone (A3) | Wire parts in `aperag/domains/agent_runtime/wire/parts.py` already typed; translator at `aperag/domains/agent_runtime/wire/translator.py` plumbs name resolution. Stream emitter merged at `51137301` (#73 D8.1). Backend citations + tool lifecycle = task #75 (in_review at HEAD). | **D10 must enforce all 7 in design pack**. Items ②③⑤ are runtime concerns and largely out-of-scope for D10's read-only retrieval surface; ①④⑥⑦ are core MCP-layer concerns: D10 design pack §A must demonstrate compliance with each per added tool. |
| **Multi-tenant auth boundary** (D9 §2) | `docs/modularization/agent-runtime-mcp-design.md` §2 — three-level authorization (visibility / invocation / consent) | Canonical gate: `RetrievalService.create_search()` at `aperag/domains/retrieval/service.py:96-106` (`db_ops.query_collection(user, collection_id)` + marketplace owner fallback). Auth dependency: `Depends(required_user)` per route, narrow `AuthenticatedUser(Protocol)` defined at `aperag/domains/retrieval/api/routes.py:38-50` (only `id` field). | **Reuse directly**. Every D10 MCP tool reaches FastAPI on `localhost:8000` and inherits this gate. **Action**: promote the narrow `AuthenticatedUser` Protocol from `domains/retrieval/api/routes.py` to a canonical `domains/identity/` location (D9 cleanup task) so D10 docs can reference one place. |
| **Wire envelope** (D8) | `docs/modularization/agent-message-protocol-design.md` (UIMessage / UIMessagePart) | `aperag/domains/agent_runtime/wire/parts.py`, `aperag/domains/agent_runtime/wire/translator.py`. Stream emitter merged at `51137301`. | **D10 read primitives output shape** should align with UIMessagePart conventions when consumed by ApeRAG's own agent (e.g. citations as `data-citation`); external clients see plain MCP tool returns. |
| **AI SDK v5 stream emitter** (D8.1, #73 done) | `docs/modularization/agent-message-protocol-design.md` §A2 | Landed at `51137301` (#1695). | Not directly reused by D10 read primitives (those are stateless tool calls), but D10 design must not regress AI-SDK-v5 wire compatibility for `search_collection`. |

### A-summary

- **Already on disk and reusable**: tenancy gate (canonical), wire parts, AI-SDK-v5 emitter, `SafeToolNameResolver` plumbing point.
- **Designed but not yet on disk** (D10 will likely cause us to actually need them): full SafeToolName resolver impl + collision registry, 3-tier `MCPServer` / `MCPToolCache` tables, `data-tool-consent` part type, `data-elicitation` part type. Items 4-5 of the 7-point contract gate any D10 *write* tool.
- **D10 read-only surface lower bound** for compliance: items ①④⑥⑦ of the 7-point contract — these are non-negotiable even for read-only tools.

---

## Appendix B — earayu2 三条 open question 影响面 1-page 表

For each open question, this appendix shows: which interfaces + endpoints are affected today, what D10 surface each yes/no answer enables or precludes, and the rough cost asymmetry.

### B-1. Summary / Vision deprecate 程度

| Choice | Effect on Body §B.5/B.6 | Effect on §C / future surface | Cost direction |
|--------|-------------------------|-------------------------------|----------------|
| **Full deprecate (drop backend infra too)** | Remove `use_summary_index` / `use_vision_index` flags from `search_collection`; delete summary + vision pipeline branches; drop `summary_index.py` + `vision_index.py`; drop `CollectionSummary` table + summary trigger endpoint; drop image embedding path in docparser | Code + DB migration; loses an inferable Vision asset URL surface (asset:// scheme stops being meaningful for vision results) | **High** — irreversible, requires migration; loses signal that Agents could in principle re-enable later |
| **FE-hidden, backend-kept (status quo)** | Body §B.5/B.6 stays as-is; flags continue to default `True` | D10 design pack §A can keep summary/vision in the omnibus bundle or split into per-mode tools without code change | **Low** — zero-cost; confusion risk is that "hidden" is not communicated in MCP docs |
| **Backend-kept + MCP-explicit** (recommend Appendix B candidate) | Body §B.5/B.6 unchanged; D10 explicitly documents Summary / Vision as "internal-mode" MCP capabilities behind `requires: ["vision"]` annotation | External Agents that opt in can use them; default ApeRAG user agent doesn't | **Low** — purely doc + tool-annotation work |

**Code touch surface for "full deprecate" path** (for cost realism):
- `aperag/mcp/server.py:94-256` — remove 2 boolean flags + asset URL logic
- `aperag/domains/retrieval/schemas.py` — drop `SummarySearchParams`, `VisionSearchParams`
- `aperag/domains/retrieval/pipeline.py` — drop summary/vision branches + dedup logic
- `aperag/domains/indexing/summary_index.py` + `aperag/domains/indexing/vision_index.py` — delete
- `aperag/domains/knowledge_base/db/models.py:127+` — drop `CollectionSummary` model (DB migration)
- `aperag/domains/knowledge_base/api/routes.py:168` — drop `generate_collection_summary_view` endpoint
- `aperag/docparser/image_parser.py` — at minimum, drop the embedding-export path

### B-2. Write / mutation tools (`add_document` / `delete_document` / `tag_document` / merge / etc.)

| Choice | Effect on Body / §C | Effect on D10 design pack | Cost direction |
|--------|---------------------|---------------------------|----------------|
| **In scope (read + write in D10)** | C.1 / C.2 / C.4 mutation rows promote to D10 candidates; need `data-tool-consent` (D9 §3 / 7-point items ②③) to be implemented before any merge | Surface roughly doubles; design pack must include consent UX shape, audit-log spec, idempotency keys | **High** — gated on D9 consent + elicitation work; cannot ship D10 read primitives without also shipping consent infra |
| **Read-only first, write in D11** (recommend) | C.1 / C.2 / C.4 mutation rows stay HTTP-only for now; D10 surface is purely read | D10 design pack §A scope shrinks to `list_*` / `read_*` / `search_*` / discovery / outline | **Low** — D9 consent work decouples; D11 picks up writes once consent / elicitation lifecycle is in code |
| **Write but only via HTTP API stays** (no MCP write tools, ever) | Same as "read-only first" but with explicit non-goal | Stable, conservative | **Low** — but limits external Agents (Claude Code etc.) to read-only collaboration |

**Per-collection write endpoints already in code (HTTP-only) that "in-scope" would need to wrap**:
- C.1 rows for upload / fetch-url / delete / rebuild-index (~8 endpoints)
- C.2 rows for create / update / delete / publish / unpublish / rebuild_failed_indexes / summary trigger (~7 endpoints)
- C.4 graph mutation rows (`merge`, `suggestions`, `apply`)
- Plus marketplace subscribe / unsubscribe (C.6)

That is **~20 net-new MCP write tools**, every one of which needs a consent contract.

### B-3. Cross-collection operations

| Choice | Effect on Body / §C | Effect on D10 design pack | Cost direction |
|--------|---------------------|---------------------------|----------------|
| **Single-collection only (status quo)** | Body §B.* keep `collection_id` as required positional; `list_collections` as the only fan-out | Tool list stays simple; predictable tenancy; per-call clear cost / quota | **Low** — no work |
| **Cross-collection within one tenant** | Need new tools `search_collections([cid1, cid2, ...], query)`, `read_documents_across([{cid, did}, ...])`; OR `collection_id: Optional` with tenant-wide default | `RetrievalService` change at `aperag/domains/retrieval/service.py:96-106`: `query_collection` becomes batch + per-row check; pipeline changes for fan-out + result merging | **Medium** — code + DB query-plan work; risk of N+1 if not designed for batch |
| **Cross-tenant (marketplace)** | Marketplace endpoints in C.6 promote to D10 surface; need explicit "subscribed but not owned" semantics in tool docs | Auth gate becomes two-level: owner-or-subscriber; existing marketplace-owner fallback at `aperag/domains/retrieval/service.py:100-106` is the prototype but only resolves provider keys, not data scope | **High** — touches tenancy invariants; needs explicit cross-tenant tests; quota accounting non-trivial |

### B-summary

- **Cheapest decision combination**: B-1 = "FE-hidden, backend-kept" + B-2 = "read-only first, write in D11" + B-3 = "single-collection only". This is roughly the status quo and lets D10 design pack ship as a pure-additive read primitive.
- **Most ambitious combination**: B-1 = "Backend-kept + MCP-explicit annotation" + B-2 = "in scope" + B-3 = "cross-collection within one tenant". This is the "let Agents loose on the document substrate" version — but every "in scope" answer pulls D9 consent / elicitation work into D10's critical path.

---

## End-of-document checklist (for review)

- [x] Body §B covers all 6 retrieval interfaces with MCP / HTTP / schema / service / impl / tenancy.
- [x] §C catalogs HTTP-only / internal-only capabilities per PM expansion (msg=7a954e9b).
- [x] Every row tagged with access tier per architect taxonomy (msg=112d3aad).
- [x] Appendix A maps D9 base primitives — what is on-disk vs. design-only at `51137301`.
- [x] Appendix B 1-page impact table for earayu2's three open questions.
- [x] Citations are `path:line` against worktree at HEAD `51137301`; no claim is unsupported by code or design doc.
- [x] No code modifications. Read-only.

This document is intended as the input to the D10 design pack (task #82). Recommended next step: when @符炫炜 opens the D10 design lane, copy this file into `docs/modularization/d10-current-mcp-surface-inventory.md` (or rename per architect taste) and treat as the frozen current-state record.

---

## Delta from `51137301` → `e290488b` (#74 D8.2 merge)

> Trigger: PM @架构师 (msg=99e7b73a, msg=a96d7e6c) requested a delta pass when #74/#75 land main before this doc is filed. #74 merged at `e290488b` (PR #1694, msg=e75a748e). #75 still in_review at the time of this delta — second delta to follow when it lands.

### Diff scope (purely additive)

`git diff --stat 51137301..e290488b` reports 6 new files / 1115 insertions / 0 deletions:

| Path | Lines | Purpose |
|------|-------|---------|
| `aperag/domains/agent_runtime/uimessage.py` | +346 | UIMessagePart Pydantic schemas (D8 §2 + D9 §3.1/§5.1 canonical) |
| `aperag/domains/agent_runtime/uimessage_store.py` | +187 | DB+Redis at-rest store |
| `aperag/domains/agent_runtime/storage.py` | +25 | Storage adapter |
| `aperag/domains/agent_runtime/db/models.py` | +33 | `agent_message` ORM table |
| `aperag/migration/versions/...d8e2c4a17b91_add_agent_message_table.py` | +67 | Additive migration |
| `tests/unit_test/agent_runtime/test_uimessage_at_rest.py` | +457 | 11 contract tests pinning round-trip / transient exclusion / snapshot consistency / wrapped-shape / SafeToolName / camelCase / D9 §5.1 elicitation canonical |

No file in `aperag/mcp/`, `aperag/domains/retrieval/`, `aperag/domains/knowledge_base/`, `aperag/domains/web_access/`, `aperag/domains/knowledge_graph/`, `aperag/domains/indexing/`, or `aperag/docparser/` is touched.

### Effect on this document

| Section | Affected? | Notes |
|---------|-----------|-------|
| §A MCP server (current state) | **No** | `aperag/mcp/server.py` unchanged (still 848 lines, 5 tools). |
| §B 6 retrieval interfaces | **No** | Retrieval pipeline / schemas / tenancy gates unchanged. |
| §C HTTP-only / internal-only gaps | **No** | No new HTTP endpoints; no new public surface. |
| §D Document storage / parsing | **No** | docparser, indexers, object store layout unchanged. |
| **Appendix A (D9 base reuse matrix)** | **Yes — classification only** | See A-delta below. |
| Appendix B (open-question impact) | **No** | The earayu2 questions (Summary/Vision deprecate, write tools, cross-collection) are unaffected by D8.2 storage merge. |

### A-delta — D9 base reuse matrix updates

These rows in **Appendix A** shift classification:

| Primitive | Before (`51137301`) | After (`e290488b`) | Citation at HEAD `e290488b` |
|-----------|---------------------|---------------------|------------------------------|
| `data-tool-consent` part type (D9 §A7) | designed only, not on disk | **on-disk and typed** | `aperag/domains/agent_runtime/uimessage.py:229-244` (`type: Literal["data-tool-consent"]`); `argsPreview` + `argsHash` fields at `:214-215` |
| `data-elicitation` part type (D9 §5.1) | designed only, not on disk | **on-disk and typed** | `aperag/domains/agent_runtime/uimessage.py:255-263` (`type: Literal["data-elicitation"]`); `elicitationId` at `:242` |
| Tool-call part with `tool-<safeName>` discriminator (D8 §2.4) | mentioned in design + plumbing point in translator | **on-disk schema with `tool-<safeName>` discriminator** | `aperag/domains/agent_runtime/uimessage.py:122-131` (docstring confirms: "discriminator carries SafeToolName directly"); UIMessagePart union at `:264-285` |
| `argsHash` stability for consent audit (D9 §A7) | design only | **on-disk + tested** | `aperag/domains/agent_runtime/uimessage.py:325-345` (stable hash); test at `tests/unit_test/agent_runtime/test_uimessage_at_rest.py` (1 of 11 contract tests) |
| At-rest persistence of UIMessage | design only | **on-disk + tested** | `aperag/domains/agent_runtime/uimessage_store.py` + DB table `agent_message` at `aperag/domains/agent_runtime/db/models.py` (+33 lines) |
| Transient-vs-persistable filtering | design only | **on-disk + tested** | `aperag/domains/agent_runtime/uimessage.py:301-323` (`_is_transient` + `persistable_parts`) |

These rows in **Appendix A** are **unchanged**:

| Primitive | Status at `e290488b` | Citation |
|-----------|----------------------|----------|
| SafeToolName **resolver impl** + collision registry | Still pending — `TODO(#75 chenyexuan)` at `aperag/domains/agent_runtime/wire/translator.py:120` | `translator.py:82, 107, 111-120, 286, 341, 417` (type alias + plumbing only; no resolver body) |
| 3-tier `MCPServer` / `MCPToolCache` tables | Still design-only — no DB tables landed | `docs/modularization/agent-runtime-mcp-design.md` D9 SSoT |
| Multi-tenant tenancy gate | Unchanged on-disk reusable | `aperag/domains/retrieval/service.py:96-106` |
| Wire emitter (D8.1 stream) | Unchanged on-disk reusable | Landed at `51137301` (#1695) |

### Effect on conclusions (verified unchanged)

- **D10 read-only surface compliance lower bound = 7-point contract items ①④⑥⑦** — **unchanged**. Items ②③⑤ now have part-schema + at-rest backing on disk, but they remain runtime/write-side concerns; D10 read-only tools do not invoke them.
- **D10 will need new persistence for outline / chunk / section reads** (per §C.3 + §D) — **unchanged**. The D8.2 merge added agent-message persistence, not document-substrate persistence.
- **Body §B + §C inventories and tagging** — **unchanged**.

### Watch list for second delta (#75)

When task #75 (D8.3 backend citations + tool lifecycle + consent/elicitation contract) merges main, the following Appendix A rows are likely to flip from design-only / partial-impl to fully on-disk:
- 7-point contract items ②③ enforcement points in tool lifecycle (consent decision flow + audit trail)
- 7-point contract item ⑤ elicitation lifecycle (request/response handlers, not just part schemas)
- SafeToolName resolver impl + collision registry (closes `TODO(#75 chenyexuan)` at translator.py:120)

When that merge lands I'll re-pin to the new HEAD and write a second `## Delta from e290488b → <new HEAD>` block here.

---

## Delta from `e290488b` → `bd4052d5` (#75 D8.3 merge)

> Trigger: per the watch list above. #75 merged at `bd4052d5` (PR #1696, 2026-04-25 21:58). PM @架构师 explicitly triggered this second delta pass at msg=4b13bd46. This pass closes the watch-list items above; no third delta is currently scheduled.

### Diff scope

`git diff --stat e290488b..bd4052d5` reports 20 files / 4449 insertions / 18 deletions. Excluding the prior delta's own doc PR (which landed in between as `38616050`, the D10.a inventory itself), the actual code surface added by #75 is:

| Path | Lines | Purpose |
|------|-------|---------|
| `aperag/domains/agent_runtime/tools/__init__.py` | +83 | Subpackage exports |
| `aperag/domains/agent_runtime/tools/safe_name.py` | +225 | `sanitize_tool_name`, `SafeToolNameResult`, `SafeNameRegistry` — full D9 §A1+§A6 resolver impl with collision detection + sha256 hash suffix |
| `aperag/domains/agent_runtime/tools/registry.py` | +352 | `RegistryScope` (Enum: system/bot/user), `MCPServerEntry`, `MCPServerRegistry` — 3-tier registry impl with `(scope_ref, name)` composite key + `RegistryConflictError` for system-namespace protection |
| `aperag/domains/agent_runtime/tools/authorization.py` | +278 | `ToolRiskClassification`, `Principal`, `ToolAuthorizationPolicy`, `default_policy()` — D9 §2 three-level authorization (visibility / invocation / consent) with B4 default-deny on unknown risk |
| `aperag/domains/agent_runtime/tools/consent.py` | +378 | `ConsentService`, `ConsentBinding`, `ConsentRequestResult`, `ConsentDecisionResult`, `ConsentOwnershipError` — D9 §A7 consent lifecycle with chat-scoped ownership defense |
| `aperag/domains/agent_runtime/tools/elicitation.py` | +389 | `ElicitationService`, `ElicitationBinding`, `ElicitationSubmitResult`, `_required_fields_validator` — D9 §5 elicitation lifecycle |
| `aperag/domains/agent_runtime/tools/lifecycle.py` | +288 | `translate_lifecycle_envelope()`, `LifecycleEmitter`, `ConsentEnvelopeEmission`, `ElicitationEnvelopeEmission` — tool execution state machine |
| `aperag/domains/agent_runtime/tools/citations.py` | +162 | `build_citation()`, `transform_reference_bundle_items()` — D8.3 backend citations |
| `aperag/domains/agent_runtime/tools/args_cache.py` | +145 | `RawArgsCache` (Protocol), `InMemoryRawArgsCache` — A7 raw args backend-private store keyed by `argsHash` |
| `aperag/domains/agent_runtime/api/routes.py` | +214 / -... | New chat-scoped REST endpoints `POST /agent/chats/{chat_id}/turns/{turn_id}/consent/{tool_call_id}` + `/elicit/{eid}` with HTTP-layer ownership pre-check + service-layer defense-in-depth |
| `aperag/domains/agent_runtime/wire/parts.py` | +50 / -... | `DataToolConsentPart` / `DataElicitationPart` placeholder cleanup; wrapped typed data aligned with #74 `uimessage.py` SSoT |
| 8 test files under `tests/unit_test/agent_runtime/test_tools_*.py` | ~1419 | 95 contract tests pinning all 7 D9 §A4 points + B1/B2/B3/B4 fixes |

No file in `aperag/mcp/`, `aperag/domains/retrieval/`, `aperag/domains/knowledge_base/`, `aperag/domains/web_access/`, `aperag/domains/knowledge_graph/`, `aperag/domains/indexing/`, or `aperag/docparser/` is touched by #75.

### Effect on this document

| Section | Affected? | Notes |
|---------|-----------|-------|
| §A MCP server (current state) | **No** | `aperag/mcp/server.py` unchanged. |
| §B 6 retrieval interfaces | **No** | Retrieval pipeline / schemas / tenancy gates unchanged. |
| §C HTTP-only / internal-only gaps | **No new MCP/HTTP surface from #75** | The new chat-scoped consent / elicit REST endpoints are agent-runtime concerns (out of D10 lower-layer scope per Lock #6). |
| §D Document storage / parsing | **No** | docparser, indexers, object store layout unchanged. |
| **Appendix A (D9 base reuse matrix)** | **Yes — major flip** | See A-delta-2 below. |
| Appendix B (open-question impact) | **No** | Q1/Q2/Q3 unaffected by D8.3 tool-lifecycle merge. |

### A-delta-2 — D9 base reuse matrix updates

These rows in **Appendix A** flip from "design only" / partial-impl to fully on-disk:

| Primitive | Before (`e290488b`) | After (`bd4052d5`) | Citation at HEAD `bd4052d5` |
|-----------|---------------------|---------------------|------------------------------|
| **SafeToolName resolver impl + collision registry** (D9 §A1 + §A6) | type alias + plumbing only; `TODO(#75 chenyexuan)` at translator.py:120 | **on-disk + tested** | `aperag/domains/agent_runtime/tools/safe_name.py:68` (`sanitize_tool_name`), `:80` (`_hash_suffix`), `:93` (`SafeToolNameResult`), `:109` (`SafeNameRegistry`); 12 contract tests in `tests/unit_test/agent_runtime/test_tools_safe_name.py` |
| **3-tier `MCPServer` registry** (D9 §1.1 + §A5) | designed only — no DB tables, no in-memory impl | **on-disk + tested as in-memory data structure** (NOT yet a DB table — registry is in-process; persistence layer would be a separate concern) | `aperag/domains/agent_runtime/tools/registry.py:67` (`RegistryScope` Enum), `:76` (`MCPServerEntry`), `:103` (`RegistryConflictError`), `:119` (`_ScopeIndex`), `:133` (`MCPServerRegistry`), `:318` (`_tier_key`); 22 contract tests in `tests/unit_test/agent_runtime/test_tools_registry.py` |
| **`data-tool-consent` lifecycle** (7-point contract item ②) | part schema only (post-#74); no service / endpoint | **on-disk + tested with full lifecycle** | `aperag/domains/agent_runtime/tools/consent.py:78` (`ConsentRequestResult`), `:91` (`ConsentBinding`), `:106` (`ConsentOwnershipError`), `:111` (`ConsentDecisionResult`), `:119` (`ConsentService`); REST `POST /agent/chats/{chat_id}/turns/{turn_id}/consent/{tool_call_id}` in `aperag/domains/agent_runtime/api/routes.py`; 25 contract tests in `tests/unit_test/agent_runtime/test_tools_consent.py` |
| **`argsPreview` + `argsHash` raw-args-backend-private cache** (7-point contract item ③) | part field schema only (post-#74) | **on-disk + tested** | `aperag/domains/agent_runtime/tools/args_cache.py:58` (`RawArgsCache` Protocol), `:74` (`InMemoryRawArgsCache`); 12 contract tests in `tests/unit_test/agent_runtime/test_tools_args_cache.py` |
| **`data-elicitation` lifecycle** (7-point contract item ⑤) | part schema only (post-#74); no service / endpoint | **on-disk + tested with full lifecycle** | `aperag/domains/agent_runtime/tools/elicitation.py:68` (`ElicitationRequestResult`), `:74` (`ElicitationBinding`), `:83` (`ElicitationOwnershipError`), `:88` (`ElicitationSubmitResult`), `:103` (`_required_fields_validator`), `:120` (`ElicitationService`); REST `POST /agent/chats/{chat_id}/turns/{turn_id}/elicit/{eid}`; 25 contract tests in `tests/unit_test/agent_runtime/test_tools_elicitation.py` |
| **Three-level authorization** (7-point contract item ⑥) | tenancy gate at `RetrievalService` only | **on-disk + tested with formal policy + B4 default-deny** | `aperag/domains/agent_runtime/tools/authorization.py:42` (`ToolRiskClassification`), `:83` (`AuthorizationDecision`), `:114` (`Principal`), `:130` (`ToolAuthorizationPolicy`), `:259` (`default_policy`); 14 contract tests in `tests/unit_test/agent_runtime/test_tools_authorization.py` |
| **Tool lifecycle state machine + lifecycle envelope translator** | scattered (no canonical home) | **on-disk + tested** | `aperag/domains/agent_runtime/tools/lifecycle.py:90` (`translate_lifecycle_envelope`), `:123` (`ConsentEnvelopeEmission`), `:139` (`ElicitationEnvelopeEmission`), `:147` (`LifecycleEmitter`); 19 contract tests in `tests/unit_test/agent_runtime/test_tools_lifecycle.py` |
| **D8.3 backend citations** (Anthropic-shape `data-citation`) | designed only | **on-disk + tested** | `aperag/domains/agent_runtime/tools/citations.py:65` (`build_citation`), `:88` (`transform_reference_bundle_items`), `:117` (`_detect_location`); 11 contract tests in `tests/unit_test/agent_runtime/test_tools_citations.py` |

### Nuance — `translator.py:120` TODO not closed by #75

The TODO comment `TODO(#75 chenyexuan): plug SafeToolName resolver to populate` at `aperag/domains/agent_runtime/wire/translator.py:120` is **still present** at `bd4052d5` — translator.py was not modified by #75. The resolver impl now lives in `tools/safe_name.py` (`SafeNameRegistry`), and `tools/lifecycle.py:LifecycleEmitter` is the canonical wire-emission path that consumes it. The `safe_tool_name_resolver` parameter in `translator.py:107, 341` therefore remains a designed-but-unwired hook on the legacy translator path; the new tool lifecycle path does NOT route through it. D10 design pack §F can treat `SafeNameRegistry` as the canonical resolver and the translator hook as a separate (possibly retired) integration point — not a blocker for D10 read-only surface compliance.

### Effect on conclusions (verified unchanged)

- **D10 read-only surface compliance lower bound = 7-point contract items ①④⑥⑦** — **unchanged**. With #75 merged, all four anchor points now have canonical on-disk impls (safe_name + registry conflict-error + authorization + AI-SDK-v5 stream emitter), so D10's compliance burden becomes "use these primitives correctly" rather than "wait for design-to-code". Items ②③⑤ are now also on-disk but remain write/runtime concerns; D10 read-only tools still do not invoke them.
- **D10 will need new persistence for outline / chunk / section reads** (per §C.3 + §D) — **unchanged**. #75 added agent-runtime tool lifecycle, not document-substrate persistence.
- **Body §B + §C inventories and tagging** — **unchanged**.

### Watch list closed

All three watch-list items from the previous delta block (#75 7-point items ②③, item ⑤, SafeToolName resolver impl) are now on disk. No third delta is currently scheduled. Future deltas would be triggered if D10.f / D10.g / D10.c-h implementation lanes (per @符炫炜's design pack §G in task #84) materially change the D9 base reuse matrix.
