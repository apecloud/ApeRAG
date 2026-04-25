# D10 — Universal MCP API Redesign Design Pack

> **Task**: #84 (Phase 9 D10.b) — design pack only, no code changes
> **Author**: @符炫炜 总架构师
> **Status**: §A first-cut draft (proactive deliverable per earayu2 msg=c5812880)
> **Ground truth**: post-#90 merge expected (`60d85308` BE + post-#78 main FE Phase B closed)
> **Inventory base**: `docs/modularization/d10-current-mcp-surface-inventory.md` (PR #1698 / #1699 by @明书)

## Preface

### Scope

Lower-layer document substrate MCP redesign — make ApeRAG indexing/retrieval/document layer expose primitives that are universal (consumable by ApeRAG's own Agent + external Agents like Claude Code / Codex / Cursor).

### Non-goals (out of D10 scope)

- Bot/user MCP registry tier (D9 §1.1 territory; #75 D8.3 implementation)
- Write/mutation tools (add/delete/tag document, etc.) — deferred to D11 (gated on D9 consent + elicitation full land)
- Cross-collection / cross-tenant operations — deferred to D11+

### Decision delegation acknowledged

per earayu2 msg=8ed9ec5c: architect lock cheapest combo for D10:
- B-1 Summary/Vision: backend-kept + future MCP-explicit `requires:["vision"]` annotation
- B-2 Write tools: read-only first (D11 picks up writes)
- B-3 Cross-collection: single-collection only (status quo)

### Architect locks accumulated through D10.a inventory

| Lock | Source | Lock content |
|---|---|---|
| **R1** Read primitive stable handles | architect msg=c4ac5b80 | `section_path` / `chunk_id` / `heading_anchor` primary; byte-range optional |
| **R2** Pagination contract | architect msg=c4ac5b80 | Opaque base64 cursor + invariant key + explicit-error-on-invalid + cross-session reuse spec |
| **R3** Capability negotiation | architect msg=afc870c7 | Option A (per-tool annotation + client-side filter) default; Option B (server-side session filter) escape hatch only |
| **#5** Omnibus vs split | architect msg=fc5d8971 | Default split per AI SDK v5 idiom; keep `search_collection` as deprecated alias for migration |
| **#6** D10 lower-layer scope | architect msg=fc5d8971 | System tier MCP only; bot/user tier = D9 lane |
| **#7** Read primitive persistence | architect msg=fc5d8971 | Default Option β (LRU cache keyed by `(document_id, parse_version)`) |
| **#8** Read-only D10 compliance | architect msg=fc5d8971 | 4-point lower bound (D9 §A4 items ①④⑥⑦); items ②③⑤ NOT required for read-only |

---

## §A Read primitives surface

### A.0 Design philosophy

D10 read primitives 让 Agent 像 human-on-filesystem 那样自主探索文档：先 `list` 知道有什么，再 `read_document_outline` 看结构，按需 `read_document_section` 或 `read_document_chunk` 精读，必要时用 `search_*` 作为 fast-path。Search 不再是唯一入口，是 primitives 之一。

每个 primitive 都是 **stateless / idempotent** (per MCP spec)，便于 Agent retry / parallel；都是 **tenant-scoped via session auth**，外部 Agent 拿不到原生文件系统 visibility。

### A.1 `list_collections`

**Purpose**: 列出当前 user 可见 collections。

**Signature**:
```python
@mcp_server.tool
async def list_collections(
    *,
    cursor: str | None = None,
    limit: int = 50,
    sort_by: Literal["created_at", "updated_at", "title"] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    title_filter: str | None = None,
) -> CollectionList:
    """List collections accessible by current user (tenant-scoped)."""
```

**Returns**:
```python
class CollectionList:
    items: list[CollectionMetadata]
    next_cursor: str | None  # opaque base64; null when no more
    total_count: int | None  # optional, may be omitted for performance
```

**Tenancy gate**: existing `db_ops.query_collections(user)` (per inventory §A.2 D9 base reuse).

**Pagination**: per R2 — cursor encodes `(sort_by, last_position)` invariant; cursor invalidation returns explicit `cursor_invalid` error code.

### A.2 `list_documents`

**Purpose**: 列出 collection 内文档。

**Signature**:
```python
@mcp_server.tool
async def list_documents(
    collection_id: str,
    *,
    cursor: str | None = None,
    limit: int = 50,
    sort_by: Literal["created_at", "title", "size_bytes"] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    title_filter: str | None = None,
    type_filter: list[str] | None = None,  # mime types
    indexed_only: bool = False,  # exclude pending/failed indexing
) -> DocumentList:
    """List documents within a collection."""
```

**Returns**:
```python
class DocumentList:
    items: list[DocumentMetadata]
    next_cursor: str | None
    total_count: int | None
```

**Tenancy gate**: `db_ops.query_collection(user, collection_id)` raises `CollectionNotFoundException` on non-owner non-subscriber miss.

### A.3 `get_document_metadata`

**Purpose**: 单个文档的元数据 — title / type / size / indexed_chunks_count / created_at / outline_summary。

**Signature**:
```python
@mcp_server.tool
async def get_document_metadata(
    collection_id: str,
    document_id: str,
) -> DocumentMetadata:
    """Get metadata for a specific document."""
```

**Returns**:
```python
class DocumentMetadata:
    document_id: str
    collection_id: str
    title: str
    media_type: str
    size_bytes: int
    indexed_chunks_count: int
    indexing_status: Literal["pending", "indexing", "complete", "failed"]
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime
    outline_summary: list[OutlineHeading] | None  # top 2 levels for nav
```

### A.4 `get_collection_metadata`

**Purpose**: collection 配置 + 索引状态。

**Signature**:
```python
@mcp_server.tool
async def get_collection_metadata(
    collection_id: str,
) -> CollectionDetailMetadata:
    """Get metadata for a specific collection."""
```

**Returns**: schema config + doc count + index modes available + permission model.

### A.5 `read_document`

**Purpose**: 读完整 document 的 parsed markdown content（不是原始 binary）。

**Signature**:
```python
@mcp_server.tool
async def read_document(
    collection_id: str,
    document_id: str,
    *,
    range: ByteRange | None = None,  # optional, best-effort, NOT stable across re-parse
) -> DocumentContent:
    """Read parsed markdown content of a document."""
```

**Returns**:
```python
class DocumentContent:
    document_id: str
    collection_id: str
    parsed_markdown: str
    parse_version: str  # for cache invalidation; client doesn't usually consume
    truncated: bool  # if range applied or content > size limit
    truncation_reason: str | None
```

**Persistence note** (per Lock #7): server-side LRU cache keyed by `(document_id, parse_version)`; cold cache miss triggers re-parse.

### A.6 `read_document_outline`

**Purpose**: 读 heading tree — Agent 用 outline 决定下一步读哪段。**Highest-value primitive per inventory §C.3 gold mine.**

**Signature**:
```python
@mcp_server.tool
async def read_document_outline(
    collection_id: str,
    document_id: str,
    *,
    max_depth: int = 6,  # heading levels; clamp at 6
) -> DocumentOutline:
    """Read heading tree (table of contents) of a document."""
```

**Returns**:
```python
class DocumentOutline:
    document_id: str
    headings: list[OutlineHeading]
    parse_version: str

class OutlineHeading:
    level: int  # 1-6
    text: str
    section_path: str  # e.g., "1/2.3/4" — primary stable handle (per R1)
    heading_anchor: str  # e.g., "#chapter-2-implementation" — slug-style alternative
    chunk_id: str | None  # corresponding chunk if mapped
    children: list[OutlineHeading]
```

### A.7 `read_document_section`

**Purpose**: 读 outline 上某 heading 对应的 section content。

**Signature**:
```python
@mcp_server.tool
async def read_document_section(
    collection_id: str,
    document_id: str,
    *,
    section_path: str | None = None,
    heading_anchor: str | None = None,  # alternative to section_path
) -> DocumentSection:
    """Read content of a specific section by section_path or heading_anchor."""
```

**Constraints**: at least one of `section_path` / `heading_anchor` required; both provided → server prefers `section_path`.

**Returns**:
```python
class DocumentSection:
    document_id: str
    collection_id: str
    section_path: str
    heading_anchor: str
    heading_text: str
    parsed_markdown: str
    parse_version: str
    parent_section_path: str | None
    sibling_count: int  # context for Agent navigation
```

### A.8 `read_document_chunk`

**Purpose**: 读 chunk 级 content — 与现有 indexing chunk 对齐，granular access for citation drill-down。

**Signature**:
```python
@mcp_server.tool
async def read_document_chunk(
    collection_id: str,
    document_id: str,
    chunk_id: str,
) -> DocumentChunk:
    """Read content of a specific chunk by stable chunk_id."""
```

**Returns**:
```python
class DocumentChunk:
    chunk_id: str
    document_id: str
    collection_id: str
    parsed_markdown: str
    section_path: str | None  # if chunk maps to a section
    chunk_index: int  # ordering within document
    chunk_total: int  # total chunks in document
    parse_version: str
```

### A.9 Stable handle invariants (R1 Lock)

- `section_path` 是 primary stable handle — slash-separated heading position (e.g., "1/2.3/4")
- `chunk_id` 是 indexing-layer-issued stable id (与现有 vector/full-text/graph index 共用)
- `heading_anchor` 是 slug-style alternative (e.g., "#chapter-2-implementation")
- `byte-range` 只标 optional / best-effort，parse-version-bound，未来如果 parsed markdown re-derived 可能失效

所有 stable handles 在同一 `(document_id, parse_version)` 内 byte-stable；跨 parse_version 不保证（但 section_path 通常 robust，chunk_id 取决于 indexing chunker 是否 deterministic — D10.d implementation 决定）。

---

## §B Search primitives surface (split + omnibus deprecation)

### B.0 Design philosophy (Lock #5)

per architect msg=fc5d8971: 当前 MCP `search_collection` 是 omnibus tool with 5 boolean mode flags (per @明书 D10.a inventory msg=71a476e3 §A.2)。D10 lock = **split into discrete tools** + 保留 omnibus 为 deprecated alias for backward-compat migration window。

理由复习：
1. AI SDK v5 idiom — 每个 tool 一个 schema + return shape
2. R3 Option A per-tool annotation 自然落地 (split 后 `requires:` annotation per tool)
3. 解墙哲学 — primitives 优先，search 降级为 optional 的多种之一
4. D10 future scope (Claude Code / Codex / Cursor 接入) 期望 SDK-spec 标准 tool surface

### B.1 `vector_search`

**Purpose**: 向量相似度搜索 (与现有 `search_collection use_vector_index=True` 等价)。

**Signature**:
```python
@mcp_server.tool
async def vector_search(
    collection_id: str,
    query: str,
    *,
    top_k: int = 5,
    similarity_threshold: float | None = None,  # None = use collection default
    rerank: bool = True,
    cursor: str | None = None,  # for pagination beyond top_k
) -> SearchResult:
    """Vector similarity search within a collection."""
```

**Returns**:
```python
class SearchResult:
    items: list[SearchResultItem]
    next_cursor: str | None
    query: str
    search_mode: Literal["vector", "graph", "fulltext", "web"]
    elapsed_ms: int

class SearchResultItem:
    chunk_id: str
    document_id: str
    text: str
    score: float
    section_path: str | None  # for navigation back to outline
    heading_anchor: str | None
    metadata: dict[str, Any]  # source-specific (e.g., page_number for PDFs)
```

**Tenancy gate**: `db_ops.query_collection(user, collection_id)` (复用现有 D9 base canonical tenancy gate per inventory §A.4)。

### B.2 `graph_search`

**Purpose**: 知识图谱搜索 (与现有 `search_collection use_graph_index=True` 等价)。

**Signature**:
```python
@mcp_server.tool
async def graph_search(
    collection_id: str,
    query: str,
    *,
    top_k: int = 5,
    depth: int = 2,
    entity_types: list[str] | None = None,
    cursor: str | None = None,
) -> SearchResult:
    """Knowledge graph search with entity expansion."""
```

**Returns**: same `SearchResult` envelope；`search_mode = "graph"`；`metadata` 含 graph-specific 字段 (entity / relation / path)。

**Notes**: 
- D10 surface 不暴露 graph 内部 entity ID schema — 通过 `chunk_id` / `document_id` 提供 navigation back to read primitives
- `depth` 参数显式让 Agent 控制图扩展程度 (per "agent should drive its own search" 哲学)

### B.3 `fulltext_search`

**Purpose**: 全文搜索 (与现有 `search_collection use_fulltext_index=True` 等价)。

**Signature**:
```python
@mcp_server.tool
async def fulltext_search(
    collection_id: str,
    query: str,
    *,
    top_k: int = 5,
    keywords: list[str] | None = None,  # override auto-extraction
    rerank: bool = True,
    cursor: str | None = None,
) -> SearchResult:
    """Full-text search (Elasticsearch / PostgreSQL FTS)."""
```

**Returns**: same envelope；`search_mode = "fulltext"`；`metadata` 含 highlight snippet / matched terms (for FE rendering)。

### B.4 `web_search`

**Purpose**: Web 外部搜索 (与现有 `web_search` MCP tool 等价；保留独立 tool，与 collection 系搜索分开)。

**Signature**:
```python
@mcp_server.tool
async def web_search(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 30,  # seconds
    locale: str = "en-US",
    source: str | None = None,  # optional provider hint (e.g., "jina" / "duckduckgo")
) -> WebSearchResult:
    """External web search (provider chain: Jina → DuckDuckGo fallback)."""
```

**Returns**:
```python
class WebSearchResult:
    items: list[WebSearchResultItem]
    query: str
    elapsed_ms: int
    provider_used: str
    fallback_used: bool

class WebSearchResultItem:
    rank: int
    title: str
    url: str
    snippet: str
    domain: str
    publish_date: str | None
```

**Notes**:
- 与 collection-scoped tools 分开，因为 web 不需要 `collection_id` + tenancy gate works on user-level provider key (per inventory §B.4)
- Provider key resolution per-user via `query_provider_api_key(user)` (复用现有 path)

### B.5 `search_collection` — deprecated alias for omnibus

**Purpose**: backward compatibility migration window，保留旧 omnibus tool 但 deprecate。

**Signature**: 与现有完全一致：
```python
@mcp_server.tool
@deprecated("Use vector_search / graph_search / fulltext_search separately. To be removed in D11.")
async def search_collection(
    collection_id: str,
    query: str,
    use_vector_index: bool = True,
    use_fulltext_index: bool = True,
    use_graph_index: bool = True,
    use_summary_index: bool = True,  # backend-kept per cheapest combo B-1
    use_vision_index: bool = True,    # backend-kept per cheapest combo B-1
    rerank: bool = True,
    topk: int = 5,
    query_keywords: list[str] | None = None,
) -> Dict[str, Any]:
    """[DEPRECATED] Use vector_search/graph_search/fulltext_search separately."""
```

**Notes**:
- `@deprecated` annotation per Python 3.13+ + MCP tool metadata
- backend implementation 不变；只标 deprecated
- D11 (post 1-quarter usage analytics) 决定 hard removal timing
- summary/vision modes 继续 backend-kept (per B-1 cheapest combo lock)

### B.6 Search primitives 与 read primitives 联动

per "解墙" 哲学，search 不是终点而是 navigation 入口：

```
Agent flow:
  list_collections → list_documents → 
    [optional] read_document_outline (overview)
    OR
    [optional] vector_search → (chunk hit) → read_document_chunk → read_document_section
    OR  
    [optional] graph_search → (related entity) → read_document_chunk
```

`SearchResultItem.chunk_id` + `section_path` + `heading_anchor` 是连接 search → read 的桥梁。Agent 不是 "search-and-summarize"，是 "search-then-explore-with-read primitives"。

### B.7 SDK type guards (R3 Option A application)

每个 search tool 自带 capability annotations:

```python
@mcp_server.tool(
    requires=["collection_access"],  # require user to have collection access
    annotations={
        "search_mode": "vector",  # for client-side tool grouping
        "supports_pagination": True,
        "deprecated": False,
    },
)
async def vector_search(...): ...
```

External Agent (Claude Code / Codex / Cursor) 可以按 `requires` / `annotations` 做 client-side filtering (per R3 Option A canonical)。

### B-summary

| Tool | Replaces | Default top_k | Pagination |
|---|---|---|---|
| `vector_search` | `search_collection use_vector_index=True` | 5 | cursor (per R2) |
| `graph_search` | `search_collection use_graph_index=True` | 5 | cursor |
| `fulltext_search` | `search_collection use_fulltext_index=True` | 5 | cursor |
| `web_search` | unchanged (already separate) | 5 | none (provider-bounded) |
| `search_collection` | self (deprecated alias) | 5 | none |

post-#90 + D8 §2 byte-equal canonical 已就位，所有 search tools 返回 shape 与 ApeRAG agent runtime UIMessage parts 互通 (`SearchResultItem.text` 可直接 wrap 成 `data-citation` part if needed)。

---

## §C Pagination + cursor contract (R2 lock detailed)

### C.0 Goals

- Stateless / idempotent (per MCP spec)
- Stable across server restarts (within TTL)
- Explicit error on invalidation, never silent reset
- Cross-session reusable within TTL window
- **Cursor stability** (per Weston msg=95b07155): same query/scope 下 cursor 不能因后台重排 (e.g., index reindex / score reorder) 随机漂移

### C.1 Cursor format

**Type**: opaque base64-encoded server-issued token。

**Internal structure** (server-only, never expose to client):

```python
@dataclass
class CursorPayload:
    schema_version: int  # 1 — bump on incompat changes
    sort_key: str  # e.g., "created_at" / "score" / "title"
    last_position: dict[str, Any]  # last item's sort_key value(s) for tie-breaking
    invariant_hash: str  # sha256 of (sort_key + filters + collection_id + tenant_id)
    issued_at: int  # Unix timestamp
    ttl_seconds: int = 3600  # default 1h
    server_id: str  # for cross-instance debugging
```

**Wire format**:
```
cursor = base64url(json.dumps(CursorPayload).encode())
```

Client treats cursor as opaque string — never parse / decode / mutate.

### C.2 Cursor invariants

per R2 lock + Weston msg=95b07155 stability requirement:

| Invariant | Required behavior |
|---|---|
| **Same query → stable order** | server MUST use stable secondary sort by primary key (e.g., `created_at DESC, id DESC`) so cursor pagination doesn't skip/duplicate items even if primary sort key has ties |
| **Reindex tolerance** | when index is rebuilt mid-pagination, cursor MUST either continue (if index ID stable) OR fail with `cursor_invalid` — never silently change semantic |
| **Score reorder tolerance** | for vector_search/fulltext_search where score may shift on rerank, cursor MUST encode the score boundary, not just position; if score boundary becomes ambiguous, fail explicit |
| **Filter immutability** | cursor encodes filter set hash; if client passes different filters with same cursor, fail `cursor_filter_mismatch` |
| **Tenancy boundary** | cursor encodes tenant_id; cross-tenant cursor reuse = security violation, fail with `cursor_tenant_mismatch` |

### C.3 Error semantics (explicit-never-silent)

```python
class CursorError(Exception):
    code: Literal[
        "cursor_invalid",          # malformed / can't decode
        "cursor_expired",          # past TTL
        "cursor_filter_mismatch",  # filter set changed
        "cursor_tenant_mismatch",  # tenant changed
        "cursor_index_changed",    # underlying index incompatible
        "cursor_schema_unsupported", # schema_version not supported
    ]
    message: str
    details: dict[str, Any]  # server diagnostic info (only when safe to expose)
```

每个 error code 对应明确的 client recovery path:
- `cursor_invalid` / `cursor_schema_unsupported` → restart pagination from null cursor
- `cursor_expired` → restart pagination
- `cursor_filter_mismatch` / `cursor_tenant_mismatch` → client bug, surface to user
- `cursor_index_changed` → backend operations issue, retry from null cursor

**Anti-pattern to forbid**: 服务端 silently 重置 cursor 到 first page。这违反 explicit-not-silent 原则 + 让 client 难以 detect 重复 item。

### C.4 Cross-session reusability

per R2 cross-session reuse spec:
- Cursor TTL默认 1h，可在 design lane per-tool override (e.g., search 短 TTL，list 长 TTL)
- TTL 内 cursor 可由不同 session / different agent instance 复用 (assumes tenancy unchanged)
- TTL 过期后 cursor → `cursor_expired` 错误；client 重新发 null cursor 起新 pagination

### C.5 SDK type guards

```python
class PaginationParams:
    cursor: str | None = None
    limit: int = 50  # max enforced server-side per tool
    
class PaginationResult[T]:
    items: list[T]
    next_cursor: str | None  # null means no more
    total_count: int | None  # optional, may be omitted for performance
```

每个 paginated tool 的 return type extends `PaginationResult[ToolItemType]`。Client (Claude Code / Codex / Cursor) 可以 generic 实现 cursor loop。

---

## §D Capability negotiation (R3 Option A canonical)

### D.0 Goals

- 不依赖 vanilla MCP `initialize` capabilities (which are protocol-level, not per-tool)
- Per-tool annotation 在 spec-idiomatic 的 tool metadata 中
- Client (external Agent) 自行 filter，不依赖 server-side session-state filtering
- **Capability degradation explicit-not-silent** (per Weston msg=95b07155): 某 backend 不支持的 capability，必须 explicit `unsupported_capability` error 或显式 declared fallback，不能静默换语义

### D.1 Per-tool annotation schema

每个 D10 tool 在 MCP server registration 时附 metadata:

```python
@mcp_server.tool(
    annotations={
        "requires": ["collection_access"],  # client / server should ensure
        "capabilities": {                    # for client-side filtering
            "vision": False,                 # this tool needs no vision
            "long_context": False,           # this tool needs no long context
            "graph_index": True,             # ONLY for graph_search
            "fulltext_index": True,          # ONLY for fulltext_search
            "web_access": True,              # ONLY for web_search
        },
        "deprecated": False,
        "deprecated_until": None,            # ISO date
        "fallback_to": None,                 # tool name to fall back to
    },
)
async def tool_function(...): ...
```

### D.2 Client-side filter pattern (Option A)

External Agent (Claude Code / Codex / Cursor) 启动时:

```python
# Pseudocode
all_tools = mcp.list_tools()
client_capabilities = self.get_capabilities()  # vision-capable? long-context?

usable_tools = [
    t for t in all_tools
    if all(client_capabilities.get(req, False) 
           for req, needed in t.annotations.capabilities.items()
           if needed)
]

# Expose only usable_tools to LLM
llm.set_available_tools(usable_tools)
```

Server 不知道 client 选了哪些 — server 仍 expose 全 tool surface，client 自行 filter。

### D.3 Capability degradation explicit-not-silent

per Weston msg=95b07155:

| Scenario | Required behavior |
|---|---|
| **Backend service down** (e.g., graph index 不可用) | `graph_search(...)` 调用返回 `ServiceUnavailableError` with `code="graph_index_unavailable"`，不静默 fall back to vector |
| **Collection 没启用某 mode** (e.g., 没 vision index) | `search_collection(use_vision_index=True)` 在 deprecated alias 中返回 partial result，但**显式** `unavailable_modes: ["vision"]` 在 result envelope；新 split tool `vision_search` 直接拒绝 |
| **Tool fallback declared** (e.g., `web_search` 主 provider 失败 fallback secondary) | 在 result envelope 中显式 `provider_used` + `fallback_used: True`，不在 metadata 中隐藏 (现有 behavior，保留) |
| **Capability missing** (per D.1 annotation) | client 不应 surface 给 LLM，但如果 LLM 仍尝试调用 (illegitimate)，server 返回 `capability_required: ["..."]` 错误 |

### D.4 Option B (server-side session filter) escape hatch

per architect msg=afc870c7: **Option B 不作为 default**，仅在以下 narrow 例外保留：

- 某 tool 因法律/合规/安全原因必须 server-side 隐藏 (不只是 client 不能用) — 当前 D10 read-only surface 没此场景
- 某 tier (system / bot / user 三层 registry) 的 admin alias 需要 audit-logged + 不可在 client filter

D10 implementation 不预留 Option B 实现；如未来真有 narrow 例外，单独开新 task design + lock。

### D.5 Annotation registry (D10.g implementation lane scope)

D10.g (per §G implementation decomposition) 负责把所有 D10 tools 的 annotation 集中注册:

```python
# aperag/domains/mcp/registry.py
TOOL_REGISTRY: dict[str, ToolMetadata] = {
    "vector_search": ToolMetadata(
        capabilities={"collection_access": True},
        annotations={"search_mode": "vector"},
    ),
    "graph_search": ToolMetadata(
        capabilities={"collection_access": True, "graph_index": True},
        annotations={"search_mode": "graph"},
    ),
    "read_document_outline": ToolMetadata(
        capabilities={"collection_access": True, "long_context": False},
    ),
    # ... all D10 tools
}
```

External Agent 通过 `mcp.list_tools()` 获取 + filter；ApeRAG 自己 Agent 也通过同一 registry，guarantees consistent filtering logic。

### D-summary

- ✅ Per-tool annotation = Option A canonical (default)
- ✅ Client-side filter = idiomatic per MCP spec
- ✅ Server-side session filter (Option B) = narrow escape hatch only, NOT in D10 implementation
- ✅ Capability degradation = explicit-not-silent (per Weston msg=95b07155)
- ✅ Tool fallback (web_search provider chain) = displayed in result envelope, not hidden

---

## §E Read primitive persistence strategy (Lock #7 detailed)

### E.0 Architect canonical default

per architect msg=fc5d8971 Lock #7：**Option β (LRU cache + parse_version) as default**。

Three options revisit:
| Option | Pros | Cons | D10 verdict |
|---|---|---|---|
| α: persistent table (`document_outline`) | guaranteed-fast read; survives restart | DB schema sprawl; needs migration on every parse_version bump; storage bloat | Reject — overkill for read-mostly workload |
| **β: LRU cache + parse_version**  | no schema change; auto-invalidate on reparse; bounded memory | cold cache miss = re-parse cost | **DEFAULT** |
| γ: re-parse on demand | cheapest, no cache state | PDF/OCR re-parse cost on every call | Reject — UX degradation for large docs |

### E.1 Cache architecture

**Layer**: Python in-memory LRU + Redis tier-2 for cross-instance shared

```python
# L1: in-process (per agent worker)
@functools.lru_cache(maxsize=256)
def parse_outline_l1(document_id: str, parse_version: str) -> DocumentOutline: ...

# L2: Redis (cross-instance + survival across restarts)
async def parse_outline_l2(document_id: str, parse_version: str) -> DocumentOutline:
    key = f"d10:outline:{document_id}:{parse_version}"
    cached = await redis.get(key)
    if cached:
        return DocumentOutline.parse_raw(cached)
    outline = await reparse_outline(document_id)
    await redis.setex(key, 3600, outline.json())  # 1h TTL
    return outline
```

**Sequencing**: read primitive → L1 hit → L2 hit → re-parse → write L2 + L1 → return

### E.2 parse_version 选择

`parse_version` is `(parser_pipeline_hash, document_md5)` 复合 hash。Bumping any of:
- Parser pipeline (e.g., MarkItDown version upgrade)
- Document content (re-upload)
- Chunking config (chunk size / strategy)

→ Auto-invalidates cache per Option β goals。Cache 不需要手动 invalidate logic。

### E.3 Cache miss budget

per-document parse latency budgets:

| Document type | Cold parse latency (P95) | L2 hit latency (P95) | L1 hit latency (P95) |
|---|---|---|---|
| Markdown / TXT < 100KB | 50ms | 5ms | <1ms |
| PDF / DOCX 1-10MB | 1-3s | 5ms | <1ms |
| OCR'd scan PDF | 5-30s | 5ms | <1ms |

Cold parse for OCR is the worst case — but L2 hit covers it cleanly. D10 implementation MUST set L2 TTL appropriate per document type (proposal: 1h universal default; per-document type override via metadata column if needed in future).

### E.4 Cache eviction + memory bounds

- L1 LRU `maxsize=256` per worker (configurable)
- L2 Redis namespace `d10:outline:*` + `d10:section:*` + `d10:chunk:*` 三 prefix；按需 set memory cap via Redis `maxmemory-policy=allkeys-lru`
- Eviction trigger: memory pressure → drop oldest by access time

### E.5 Cache invalidation 显式 trigger

Most invalidation is implicit (parse_version change → new key → old key untouched until LRU evicts)。但 explicit invalidation 在以下场景必要：

- `delete_document` → invalidate `d10:*:{document_id}:*` (D11+ scope when write tools land)
- `rebuild_indexes` → bump parse_version → automatic
- Manual cache flush (admin endpoint, optional) → for emergency consistency

### E.6 read_document_chunk 特殊性

chunk-level read 不需要 parse_version 加权 cache，因为 chunk_id 已经是 indexing-layer immutable key (per #74 D8.2 + #75 D8.3 indexing pipeline)。直接走 vector/full-text store metadata fetch + Redis single-key cache。

### E.7 Cache 只加速，不改语义 (Weston msg=52616723 hard lock)

per Weston refinement: **cache layer 仅是 performance optimization**，绝不能改变以下 read primitive 的语义 invariants：

- **Visibility**: cache hit 不能让 user 看到他不该看到的 collection/document (tenancy gate 必须每次 invocation 都过，不依赖 cache)
- **Permission**: cache hit 不能跳过 D9 §2 three-level authorization (即使内容已 cache，授权检查仍 invoke)
- **Version semantics**: cache hit 不能返回旧 parse_version 的内容当作 current state (parse_version mismatch 强制 invalidate)
- **Cache miss → authoritative storage**: 不能 fall back to stale data 或 alternative shape；miss 必须 hit 真实 source-of-truth (object store + parser pipeline)

**Implementation invariant**:
```python
async def read_document_section(user, collection_id, document_id, *, section_path):
    # ALWAYS: tenancy gate first (no cache shortcut)
    await db_ops.query_collection(user, collection_id)
    
    # ALWAYS: 3-level auth check (no cache shortcut)
    await tools.authorization.check(user, collection_id, "read")
    
    # THEN: cache lookup or compute
    parse_version = await get_parse_version(document_id)
    section = await get_or_compute_section(document_id, parse_version, section_path)
    return section
```

Tenancy + auth checks are **always-invoked**，cache is downstream of these gates。任何 cache implementation 必须遵守此顺序。

### E-summary

- ✅ Option β LRU + parse_version is default canonical
- ✅ L1 (in-process) + L2 (Redis) two-tier cache
- ✅ parse_version composite hash auto-invalidates on parser/doc/config change
- ✅ Cold parse budget: PDF 1-3s, OCR 5-30s — covered by L2 hit on subsequent reads
- ✅ chunk-level read uses chunk_id direct fetch, no parse cache layer

---

## §F D9 base reuse boundary (Lock #6 + Lock #8 detailed)

### F.0 Lock revisit

per architect msg=fc5d8971:
- **Lock #6**: D10 lower-layer scope = system tier MCP only
- **Lock #8**: Read-only D10 compliance lower bound = D9 §A4 7-point items ①④⑥⑦ (NOT items ②③⑤)

### F.1 D9 base inventory (per #1698 + #1699)

reference @明书 D10.a inventory + delta：

| D9 base primitive | Pre-D8.x state | Post-D8.x state (after #75 + #90) | D10 复用 disposition |
|---|---|---|---|
| `SafeToolName` resolver (D9 §A1+§A6) | TODO at translator.py:120 | ✅ on-disk `tools/safe_name.py:SafeNameRegistry` + collision sha256 hash suffix + reverse lookup + 12 tests | **Direct reuse**: D10 MCP tools 都通过 `SafeNameRegistry` 注册命名 |
| **3-tier MCP registry** (D9 §1.1 + §A5) | designed only | ✅ on-disk `tools/registry.py` `_ScopeIndex` `(scope_ref, name)` composite key + 22 tests | **Reuse for system tier only** (per Lock #6); bot/user tier 留 D9 implementation |
| **Multi-tenant auth boundary** (D9 §2 three-level) | tenancy gate only | ✅ on-disk `tools/authorization.py` 3-level (visibility / invocation / consent) + 14 tests + chat-scoped endpoints with HTTP+service-layer defense-in-depth | **Direct reuse**: D10 read tools 通过 `tools/authorization.py` 调用 |
| **`data-tool-consent` part** (D9 §A2+§A7) | designed only | ✅ on-disk `tools/consent.py` + REST endpoint + 25 tests | **NOT used in read-only D10** (per Lock #8 item ②③ excluded) |
| **`data-elicitation` part** (D9 §5) | designed only | ✅ on-disk `tools/elicitation.py` + REST endpoint + 25 tests | **NOT used in read-only D10** (per Lock #8 item ⑤ excluded) |
| **Tool lifecycle** (translator chain) | scattered | ✅ on-disk `tools/lifecycle.py` `LifecycleEmitter` + 19 tests | **Read-only D10 不直接 emit**; static read tool calls skip lifecycle |
| **AI SDK v5 wire emitter** (D8.1) | designed only | ✅ on-disk `wire/translator.py` (#73 merged 51137301) | **Reuse**: D10 search tools 返回 `SearchResultItem.text` 可投影成 `data-citation` part for embedding into agent stream |
| **UIMessage at-rest storage** (D8.2) | designed only | ✅ on-disk `agent_message` table + `UIMessageStore` (#74 merged e290488b) | **NOT used directly**; D10 read primitives 是 stateless tool calls，不写 message store |
| **D8 §2 wire/at-rest byte-equal canonical** | spec only | ✅ enforced via 3-shape canonical (text/tool/source/data-* + camelCase outer + Anthropic snake_case inner) | **Direct reuse**: D10 search/read result envelope 可直接 wrap 成 UIMessage parts |

### F.2 D10 read-only 4-point compliance lower bound (Lock #8 detailed)

per D9 §A4 7-point:

| # | Point | Read-only D10 status |
|---|---|---|
| **①** | SafeToolName + MCP metadata (A1+A6) | **REQUIRED** — D10 tool naming via SafeNameRegistry |
| ② | AI SDK v5 + `data-tool-consent` | NOT REQUIRED (no write actions in read-only D10) |
| ③ | argsPreview/argsHash backend-private (A7) | NOT REQUIRED (no consent gate) |
| **④** | Registry no silent system override (A5) | **REQUIRED** — D10 tools 注册到 system tier registry，no silent override |
| ⑤ | data-elicitation schema-validated input | NOT REQUIRED (no interactive prompts in read tools) |
| **⑥** | Three-level authorization (visibility/invocation/consent) | **REQUIRED** — visibility (per-tool annotation) + invocation (capability check + tenancy gate) used; consent gate skipped (no write) |
| **⑦** | PydanticAI as default candidate (A3) | **REQUIRED** — recommended runtime backbone (still default candidate, not mandate per architect C2 lock) |

D10.c-h implementation lanes 必须显式在 PR description verify 4 points (①④⑥⑦) covered。

### F.3 D9 base 不复用面 (out-of-scope explicit)

per Lock #6 (system tier only) + Lock #8 (read-only):
- ❌ Bot/user tier registry — D9 main lane (chenyexuan #75 implementation 已 land system tier)
- ❌ data-tool-consent / data-elicitation enforcement — write-tools (D11)
- ❌ argsPreview/argsHash redaction — read tools 没 raw args concern
- ❌ Tool consent state machine REST endpoints — read tools 不暴露

### F.4 Policy/backend-owned tenancy + consent boundary (Weston msg=52616723 hard lock)

per Weston refinement: D9 base reuse boundary **不能让 external agent 绕过 ApeRAG 自身的 tenancy / consent / authorization policy**。所有 policy/policy-enforcement 是 ApeRAG backend-owned，external Agent 仅 consume tool surface。

**Hard invariants**:

1. **Tenancy enforcement = backend-only**:
   - `tools/authorization.py` + `db_ops.query_collection(user, collection_id)` 是唯一 source-of-truth tenancy gate
   - External Agent 拿到 D10 MCP tool 后，每次 tool invocation 都通过 ApeRAG `required_user` dependency + service-layer ownership check 进行 tenancy 验证
   - **External Agent 不能 self-attest tenancy**：无论 client 声明什么 user_id / tenant_id，server 一律 ignore，仅信任 authenticated session

2. **Consent policy = backend-owned**:
   - D10 read tools 因 read-only 无 consent gate，但写工具 (D11+) 必须走 D9 §A2 `data-tool-consent` 后端流程
   - **External Agent 不能 self-grant consent**：consent 由 user (人类) 通过 ApeRAG UI / API 显式 approve，不能 client-side 绕过
   - System tier registered tools (per D10 §F.1 system-only) 没有 user-level consent override

3. **Capability degradation = explicit error path** (per §D.3):
   - External Agent 缺 capability 时 server 返回 `capability_required` error，不能 silent fallback 让用户拿到他不该拿的能力
   - Provider fallback (e.g., web_search Jina → DDG) 在 result envelope 显式标注，不隐藏

4. **Audit trail enforcement**:
   - 所有 tool invocation 通过 `tools/registry.py` 的 audit log
   - External Agent 调用产生 audit entry with `agent_kind: external` + `client_metadata` (Claude Code 版本号等)
   - Audit 不可被 client mute / filter

### F.6 Bridge 投影点

D10 surface 与 D9/D8 base 的 wire/at-rest 投影对接：

```
D10 SearchResultItem.text + chunk_id + section_path + score
   ↓ (when consumed by ApeRAG own Agent during agent_runtime turn)
   ↓ via wire/translator.py emit
data-citation UIMessagePart {data: {cited_text: text, location: {chunk_id, section_path}}}
```

`SearchResultItem` 不需要直接 wrap UIMessage shape — agent_runtime layer 在调用 search tool 后做投影。external Agent (Claude Code 等) 拿到 raw `SearchResultItem` 不必经 UIMessage 中间层。

### F-summary

- ✅ D10 system tier registry 复用现有 #75 implementation
- ✅ D10 4-point compliance lower bound (①④⑥⑦)，items ②③⑤ 不要求 (read-only scope)
- ✅ Multi-tenant auth boundary direct reuse via `tools/authorization.py`
- ✅ `SearchResultItem` 在 ApeRAG own Agent 内通过 wire emitter 投影 `data-citation`；external Agent 直接 raw shape

---

## §H Migration & backward compatibility plan

### H.0 Goals

- 现有 5 MCP tools 迁移路径 clear 且 explicit
- search_collection omnibus deprecation timeline + telemetry
- D9 base reuse + 不破现有 #75 D8.3 lifecycle / consent / elicitation contracts
- 外部 Agent (Claude Code / Codex / Cursor) 迁移负担最小

### H.1 search_collection deprecation timeline

per Lock #5 + earayu2 cheapest combo (Summary/Vision backend-kept):

| Phase | Timing | Action |
|---|---|---|
| **D10.b merge** | T0 | Deprecation announcement in tool annotations + docs；`@deprecated("Use vector_search/graph_search/fulltext_search separately. To be removed in D11.")` |
| **Soft deprecate** | T0 + 0 | New split tools available；`search_collection` continues to work |
| **Telemetry collect** | T0 → T0+1 quarter | Track per-tool usage analytics (FE local + external Agent calls) |
| **Hard removal eligibility** | T0+1 quarter | If usage trends below 5% of new tools, advance to T1; else extend |
| **T1: Hard removal** | T0+1-2 quarters | Drop `search_collection` MCP tool registration；keep BE pipeline (other internal callers may still use) |
| **Pipeline cleanup** | T2 (D11+) | Remove omnibus pipeline if no internal callers |

### H.2 Existing 5 MCP tools migration matrix

| Existing MCP tool | D10 disposition |
|---|---|
| `list_collections` | refine (add cursor/limit/sort_by/title_filter) per §A.1 — backward-compatible additive params |
| `search_collection` | **deprecated alias** per H.1 timeline |
| `search_chat_files` | **deprecated alias** (chat-scoped variant) — same timeline as search_collection |
| `web_search` | unchanged (already separate) — refine type annotation per R3 Option A |
| `web_read` | unchanged |

### H.3 New D10 MCP tools added

per §A + §B:
- `list_documents` + `get_document_metadata` + `get_collection_metadata`
- `read_document` + `read_document_outline` + `read_document_section` + `read_document_chunk`
- `vector_search` + `graph_search` + `fulltext_search`

Total new MCP tools = 10 (post-D10 surface)。

### H.4 External client migration guides

D10.h implementation lane (per §G decomposition) 负责 packaging + docs:

- README.mdx + example MCP server config for Claude Code / Codex / Cursor
- Migration guide: 旧 search_collection user → new split tools
- D10 capability negotiation (Option A) docs explaining client-side filter pattern
- Data structures reference: SearchResult / DocumentMetadata / DocumentOutline / DocumentSection / DocumentChunk

### H.5 Backward compatibility for ApeRAG own Agent (#75 D8.3)

D10 不能破 #75 chenyexuan implemented contracts:

| #75 contract | D10 backward compat |
|---|---|
| 7-point D9 §A4 enforcement on agent_runtime tool calls | ✅ D10 tools 注册到 system tier，agent_runtime 调用 D10 tool 时 7-point 自动 enforce (per architect Lock #8) |
| `data-tool-consent` lifecycle | ✅ Read-only D10 tools 不触发 consent (low risk by definition); write tools (D11) 复用 #75 lifecycle |
| 3-tier registry boundary | ✅ D10 system tier 注册不影响 bot/user tier |
| Tool lifecycle state machine | ✅ Read-only D10 tools 是 stateless invocation, no state machine traversal |

### H.6 fresh DB / migration considerations

per earayu2 destructive-aggressive philosophy + #79 D8.5 hard-cut precedent:

- D10 implementation 不引入 destructive DB schema change
- Search index reindex 仅在 implementation lanes 显式 declare 时进行
- D10 read primitives 不影响现有 chat/agent_runtime data

### H-summary

- ✅ search_collection deprecation timeline locked (T0 deprecate → T0+1q telemetry → T1 hard removal)
- ✅ Existing 5 MCP tools backward-compatible (additive refines + deprecated aliases)
- ✅ 10 new MCP tools added per D10 surface
- ✅ External client packaging via D10.h lane
- ✅ Backward compat with #75 D8.3 contracts preserved
- ✅ No destructive DB schema change (read-only nature)

---

## End-of-document checklist (for review)

- [x] Preface (scope / cheapest combo / locks)
- [x] §A Read primitives surface (8 primitives + R1)
- [x] §B Search primitives surface (split + omnibus deprecation)
- [x] §C Pagination + cursor contract (R2 + Weston cursor stability)
- [x] §D Capability negotiation (R3 Option A + Weston explicit-degradation)
- [x] §E Read primitive persistence strategy (Lock #7 LRU + parse_version)
- [x] §F D9 base reuse boundary (Lock #6 + Lock #8 + #1698/#1699 inventory)
- [x] §G Implementation guidelines (5 hard gates accumulated)
- [x] §H Migration & backward compatibility plan
- [x] D10.c-h implementation decomposition included in §G

This document is the input to D10.c-h implementation lanes. Recommended next step: push as pure-doc PR per #69/#1692/#1698 same pattern; merge after PM/Weston scoped review (architect already pre-locked content). Once merged, PM creates implementation tasks #94-#99 per §G decomposition.

---

## §G Implementation guidelines (lessons accumulated)

These guidelines are accumulated from D8.x Phase A/B implementation experience and apply to all D10 implementation lanes (D10.c-h) + future modularization work.

### Hard gate: contract shape change → comprehensive grep sweep

per Bryce msg=5ca10d26 + Weston msg=5d69ee54 + architect msg=5407fdd0 + Bryce msg=d6ef3742:

> Any contract shape change MUST grep the entire test+script tree (with patterns like `$.turn\.` / `.turn.` for return-shape changes), covering at minimum:
> - `tests/e2e_http/hurl/` (hurl assertion files)
> - `tests/unit_test/` (Python unit + contract tests)
> - `tests/e2e_http/scripts/` (bash scripts that invoke endpoints — also callers!)
>
> Not just files the inventory called out. Inventory-listed files are necessary but never sufficient.

**Rationale**: Inventory by Explorer agents may use file-name-based filtering (e.g., `agent_runtime_v3` matches `15_agent_runtime_v3.hurl` but misses `17_chat_collection_flow.hurl` despite same-shape assertion). Comprehensive tree-wide grep catches assertions in tests not named after the changed module. Bash scripts (`run_*.sh`) are ALSO callers — they reach the endpoint via `curl` + jq, and assert legacy shape just like hurl tests do (Bryce caught this on round 3 fix-forward when bash script was reading `.turn.status` post-#90 shape change).

**Application**: PR description must include grep command + result count for ALL three roots. PM checklist must verify grep was performed before LGTM stage.

**Audit pattern (Bryce msg=12f389a8 textbook 应用)**: grep 不止是 raw match list — 必须 **categorize each match against expected post-change shape**：
1. Legitimate top-level access (post-change new shape)
2. Unchanged sibling responses (e.g., CreateTurnResponse envelope still uses old field path)
3. ORM/internal-field access (NOT API shape)
4. Code comments referring to deprecated shape (acceptable as long as marker present)
5. Test mock internal fields (NOT API shape)

每个 category 数量 + sample 应在 PR description 列表，让 reviewer 一眼判定 grep sweep 完整性。

### Hard gate: CI red canonical decision requires actual root cause

per architect msg=5407fdd0 self-correction + Weston msg=5d69ee54:

> Architect canonical decisions on CI failures MUST be based on actual root cause analysis (read full log + dig actual assertion failure with file/line/assert text), not just upstream PM-summarized log excerpts. PM log summaries may miss real failures behind cleanup/teardown noise. CI red without identified root cause = `pending investigation`, NOT immediately `infra-flake` or `mainline-baseline`.

**Application**:
1. Identify application-layer assertion failure FIRST (check actual hurl/pytest assertion output)
2. ONLY THEN consider infra/network flake hypothesis
3. PM-summarized excerpts are clues, not evidence
4. PR owner has primary responsibility for root-cause; architect quick-check on CI is secondary

### Hard gate: caller migration → preserve original assertion semantics

per Bryce msg=7d7c90f0 (round 4 fix-forward)：

> When migrating a caller from legacy shape to new shape, the migrated caller MUST preserve the original assertion semantics — including which fields are required vs optional, which conditions are pass vs fail. Unconscious tightening (e.g., changing "reference_count is optional" to "reference_count must be > 0") is a regression even if the new-shape syntax is correct.

**Rationale**: Bryce's round 3 fix on `run_chat_collection_flow.sh` correctly migrated reading `.parts` instead of `.turn.artifacts`, but inadvertently tightened "reference_bundle artifact optional" to "reference_count > 0 required". CI passed scenario where agent's answer included citations; CI failed scenario where agent gave clarification reply without citations (legitimate behavior, was passing pre-#90).

**Application**: When migrating callers, write tests for BOTH branches of the original semantics (e.g., "with references" + "without references"), not just the assumed-default branch. PR description must enumerate caller's pre-change assertion contract + post-change contract; reviewer verifies they're equivalent.

### Hard gate: bridge/adapter deletion → ALL caller path validation

per architect msg=711f8c2f + Bryce msg=72ac5713:

> Any change to a service method return type or removal of a bridge/adapter layer MUST verify ALL caller paths in BOTH unit AND e2e test layers. Inventory-listed callers are necessary but not sufficient — service-of-service indirection often misses Explorer agent inventory.

**Application**: Bryce's #90 fix-forward on `evaluation/worker.py` + `chat_completion_service.py` is the canonical example — both were missed in #90 first-cut inventory because they reach snapshot via service-of-service indirection, not direct API call.

### Single-Opus-CR routing (RR2)

per earayu2 msg=0642be5b + PM msg=c65f75df:

> All tasks → Opus tier only. Single Opus reviewer no-blocker = sufficient merge condition. GPT/Codex tier on Codex out-of-budget. Weston offline (does not block but may be @-mentioned for second-line architecture).

**Active Opus pool**: @符炫炜 (architect+reviewer) / @cuiwenbo / @chenyexuan / @Bryce / @huangheng / @明书.

### Architect proactive engagement (post earayu2 msg=c5812880)

> Architect (符炫炜) does not stay in reactive standby — proactively reads code (not just PR description), drives D10 design pack delivery, surfaces issues before reviewer second-pass, and takes deeper code-level reviewer responsibility. Quick-check + structural-only mode is insufficient and has been flagged for replacement.

**Application**: each PR architect quick-check should include explicit verification of one or more of {spec line-by-line diff, contract shape grep sweep, caller path enumeration, hurl/pytest assertion verification}. "Structural only" quick-checks must explicitly say so and defer deep verification (in case of complex PRs that warrant it).

### D10.c-h implementation lane decomposition

Per Weston msg=71d8d605 + PM msg=db923645 — implementation phase decomposed into 6 parallel-friendly lanes. Each lane lists name, deliverable scope, owner candidate (final claim happens via `slock task claim` on the corresponding task #), write-set boundary (which directories/files each lane is allowed to touch), and dependency graph.

Lane-to-§ mapping is one-to-one with the design pack so reviewers can locate spec text by lane name.

#### D10.c — Read primitives BE implementation (§A)

- **Deliverable**: 8 read primitive MCP tools (`list_collections` / `list_documents` / `get_document_metadata` / `get_collection_metadata` / `read_document` / `read_document_outline` / `read_document_section` / `read_document_chunk`) backed by stable-handle invariants (§A.9).
- **Owner candidate**: @cuiwenbo (BE depth, prior #1697 / #88 ownership) — fallback @明书.
- **Write-set boundary**:
  - `aperag/mcp/tools/read_*.py` (new files; one file per primitive or grouped by collection/document axis)
  - `aperag/service/document_service.py` (extension methods only — additive; no shape change to existing callers)
  - `aperag/service/collection_service.py` (extension methods only — additive)
  - `tests/unit_test/mcp/test_read_primitives.py` + `tests/e2e_http/hurl/<NN>_d10_read_primitives.hurl`
  - **Forbidden**: any change to `aperag/mcp/tools/search_*.py` (that's D10.d), any change to `aperag/cache/` (that's D10.g), any change to existing `search_collection` tool (that's D10.h).
- **Depends-on**: D9 base (`SafeNameRegistry`, `ToolRegistry`) — ✅ already merged; F.2 4-point compliance lower bound.
- **Blocks**: D10.d (search primitives need stable handles from read primitives), D10.e (pagination cursors are produced by list_* read primitives), D10.g (cache layer wraps read primitives).

#### D10.d — Search primitives split + omnibus deprecation (§B)

- **Deliverable**: 4 split search MCP tools (`vector_search` / `graph_search` / `fulltext_search` / `web_search`) + `search_collection` omnibus marked deprecated with banner + R3 SDK type guards (B.7).
- **Owner candidate**: @chenyexuan (search/retrieval domain familiarity from prior chunk-handle work) — fallback @huangheng.
- **Write-set boundary**:
  - `aperag/mcp/tools/search_vector.py` / `search_graph.py` / `search_fulltext.py` / `search_web.py` (new files)
  - `aperag/mcp/tools/search_collection.py` (deprecation banner only — implementation untouched until D10.h)
  - `aperag/service/search_service.py` refactor split (preserve existing callers via thin compatibility layer; do NOT delete the layer in this lane — see D10.h)
  - `tests/unit_test/mcp/test_search_split.py` + `tests/e2e_http/hurl/<NN>_d10_search_split.hurl`
  - **Forbidden**: deleting `search_collection`'s implementation (that's D10.h), changing read primitive tool surface (that's D10.c).
- **Depends-on**: D10.c stable handles (search results contain collection_handle / document_handle for follow-up read).
- **Blocks**: D10.h (cutover requires both split tools and deprecation banner present).

#### D10.e — Pagination + cursor contract (§C)

- **Deliverable**: opaque base64 cursor + `invariant_hash` field + 6 explicit error codes (`CURSOR_EXPIRED` / `CURSOR_INVARIANT_MISMATCH` / `CURSOR_MALFORMED` / `CURSOR_FOREIGN` / `CURSOR_PAGE_OUT_OF_RANGE` / `CURSOR_VERSION_MISMATCH`); cursor is **never silently reset** — explicit error always.
- **Owner candidate**: @Bryce (canonical caller-migration discipline from #90 round-4 fix-forward; pagination semantics are caller-sensitive) — fallback @cuiwenbo.
- **Write-set boundary**:
  - `aperag/mcp/cursor/codec.py` + `cursor/invariants.py` + `cursor/errors.py` (new package)
  - `aperag/service/pagination.py` (new helper; do NOT modify existing `aperag/db/pagination.py` ORM layer — design pack uses ORM `id`-based seek pagination internally but exposes only opaque cursor externally)
  - Integration call sites: only the read primitives produced in D10.c (`list_collections` / `list_documents` / `read_document_chunk`) — no other caller in this lane.
  - `tests/unit_test/mcp/test_cursor_contract.py` + `tests/e2e_http/hurl/<NN>_d10_pagination.hurl`
  - **Forbidden**: changing search primitive pagination (search uses score-rank cursor with different invariants — covered in D10.d's own cursor type, NOT shared with this lane).
- **Depends-on**: D10.c (pagination is plumbed into list_* read primitives at integration time).
- **Blocks**: D10.h migration (legacy callers using offset/limit need cursor migration).

#### D10.f — Capability negotiation (§D Option A canonical)

- **Deliverable**: per-tool annotation schema (D.1) + client-side filter pattern (D.2 Option A) + degradation explicit-not-silent (D.3) + annotation registry (D.5). Option B server-side session filter is escape hatch only — NOT implemented in D10.f.
- **Owner candidate**: @huangheng (FE/SDK + capability metadata familiarity from D9 §A4 7-point contract review work) — fallback @符炫炜 if capacity slot opens.
- **Write-set boundary**:
  - `aperag/mcp/capabilities.py` (new annotation schema)
  - `aperag/mcp/tools/_annotations.py` (new registry — populated by each tool's own decorator)
  - Tool annotation decorators applied to all D10.c + D10.d tool files (additive only; no logic change in those files)
  - SDK type guards: `aperag/sdk/capability_filter.py` (new — client-side filter helper)
  - `tests/unit_test/mcp/test_capability_negotiation.py` + `tests/e2e_http/hurl/<NN>_d10_capabilities.hurl`
  - **Forbidden**: implementing Option B server-side session filter (escape hatch only — out of scope until needed).
- **Depends-on**: D10.c + D10.d both merged (need full tool surface to annotate).
- **Blocks**: D10.h migration (external client migration guides reference annotation schema).

#### D10.g — Read primitive persistence (§E Lock #7 LRU + parse_version L1+L2)

- **Deliverable**: L1 in-process LRU cache + L2 parse_version-keyed Redis cache; cache invalidation explicit triggers (E.5); `read_document_chunk` special-case handling (E.6); cache only accelerates, never changes semantics (E.7 Weston hard lock).
- **Owner candidate**: @明书 (caching domain familiarity + #1698 / #1699 inventory ownership) — fallback @cuiwenbo.
- **Write-set boundary**:
  - `aperag/cache/read_primitive_cache.py` (new — L1 LRU implementation)
  - `aperag/cache/parse_version_cache.py` (new — L2 Redis adapter, keyed on `parse_version` watermark)
  - `aperag/cache/invalidation.py` (new — explicit trigger helpers)
  - Integration call sites: only D10.c read primitive files (wrap their service-layer calls)
  - `tests/unit_test/cache/test_read_primitive_cache.py` + cache-miss-budget regression test
  - **Forbidden**: changing read primitive return shape (cache must be transparent — E.7 hard lock); applying caching to search primitives (search has its own cache path — out of scope).
- **Depends-on**: D10.c (cache wraps read primitives; primitives must be stable first).
- **Blocks**: D10.h (cache invalidation triggers are part of the cutover hard-cut sequence).

#### D10.h — Migration & cutover (§H hard-cut, per earayu2 msg=f20d5034)

- **Deliverable**: execute §H.1 `search_collection` deprecation timeline → removal; §H.2 existing 5 MCP tools migration matrix execution; §H.3 new D10 tools enabled; §H.4 external client migration guide published; §H.5 ApeRAG own-Agent (#75 D8.3) backward-compat path; §H.6 fresh-DB / migration considerations.
- **Owner candidate**: @符炫炜 (architect — cutover is destructive + cross-stack, requires architect-level scope/gate/risk judgment; matches "总架构师 canonical 定型 + scope/gate/risk" role per memory feedback_role_architecture_only.md). Code-execution co-owner: @Bryce (caller-migration depth).
- **Write-set boundary**:
  - **Destructive deletions**: `aperag/mcp/tools/search_collection.py` body (after deprecation window ends); legacy compatibility layer in `aperag/service/search_service.py` (introduced by D10.d)
  - Caller migration sweep: all callers of legacy 5 MCP tools across `aperag/`, `tests/`, `scripts/`, `web/` — comprehensive grep per §G hard gate #1 (3-root sweep + 5-category classification)
  - External client migration guide: `docs/modularization/d10-migration-guide.md` (new doc)
  - `tests/e2e_http/hurl/<NN>_d10_cutover.hurl` (validates legacy tools gone + new tools live)
  - **Forbidden**: introducing new feature surface in this lane (D10.c-g must already cover all surface; D10.h is destruction + cutover only).
- **Depends-on**: D10.c + D10.d + D10.e + D10.f + D10.g **all merged**; soak window per #80-style 4 hard prerequisites; comprehensive grep sweep complete.
- **Blocks**: nothing (D10 program closure).

#### Dependency graph summary

```
D9 base (merged) ──┐
                   ▼
                D10.c ─┬─→ D10.d ──┐
                       ├─→ D10.e ──┤
                       ├─→ D10.g ──┤
                       │           │
                D10.c + D10.d ─→ D10.f
                       │           │
                       ▼           ▼
            (all 5 above merged) → D10.h (cutover)
```

Parallel-friendly windows:
- **Window 1** (after D10.c merge): D10.d / D10.e / D10.g can run concurrently
- **Window 2** (after D10.c + D10.d merge): D10.f can join the parallel set
- **Window 3** (after all 5 merged + soak): D10.h cutover (single-lane, architect-led)

Tasks #95-#99 (or whatever PM assigns post-merge of #1708) map 1:1 to D10.c-h. Owner-candidate is a suggestion only — final claim happens via `slock task claim`; if claim conflicts with capacity, pool fallback applies (RR2 routing).

---

End of design pack. §G remains an open ledger — new lessons that emerge during D10.c-h implementation should be appended here as additional "Hard gate" subsections, not in a separate doc.

