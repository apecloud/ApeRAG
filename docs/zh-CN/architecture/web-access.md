# Architecture: `web_access` Domain

> **读者定位**：架构师、后端开发者，以及需要理解"为什么 ApeRAG 能从互联网抓取内容"的 SRE / 集成方。
>
> **范围**：`web_access` canonical domain 的职责、目录结构、provider 抽象、跨 domain 消费路径。本文只写 **current state**；模块化演进历史见 `docs/modularization/architecture.md`。
>
> **Baseline**：`origin/main @ 28a9f531` 代码面 + `docs/modularization/architecture.md` Section 2.12（canonical SSoT）。

## 概述

`web_access` 是 ApeRAG 12 个 canonical domain 之一，提供**从互联网抓取信息**的能力：

- **web search**：关键词 / 站内搜索，返回 `WebSearchResultItem[]`
- **web read**：把若干 URL 的正文提取出来，返回 `WebReadResponse`

这两个能力既通过 HTTP 路由对外暴露（`/api/v2/web/search` / `/api/v2/web/read`），也被其他 domain 作为内部依赖消费（知识库 URL 导入、MCP server 工具）。

这个 domain **没有 ORM 实体**（无 `db/models.py`）：它只做"功能聚合"，数据由其他 domain（identity 域的 provider API key 配置、knowledge_base 域的 document）持有。这一点在 SSoT Section 2.12 被明确标注。

## 目录结构

```
aperag/domains/web_access/
├── __init__.py
├── api/
│   ├── __init__.py
│   └── routes.py                  # FastAPI APIRouter，两个端点
├── schemas.py                     # Pydantic view models（Phase 2a hard-cut 从 view_models 迁入）
├── reader/
│   ├── base_reader.py             # Provider 抽象基类
│   ├── reader_service.py          # ReaderService 编排
│   └── providers/
│       ├── jina_read_provider.py
│       └── trafilatura_read_provider.py
├── search/
│   ├── base_search.py
│   ├── search_service.py          # SearchService 编排
│   └── providers/
│       ├── duckduckgo_search_provider.py
│       └── jina_search_provider.py
└── utils/
    ├── content_processor.py
    └── url_validator.py
```

与其他 canonical domain 不同的地方：

- **没有 `db/models.py`**：无 ORM 实体
- **没有 `ports.py`（consumer-owned Protocol）**：本 domain 是被消费方，不主动消费任何 User-related ORM；只在 `api/routes.py` 顶部声明了一个本地 `AuthenticatedUser(Protocol)` 用于 `Depends(required_user)` 类型收窄
- **多 provider 子包**：`search/providers/` + `reader/providers/` 属于"可插拔 provider"模式的典型落地

## 路由注册

在 `aperag/app.py`：

```python
from aperag.domains.web_access.api.routes import router as web_access_router
...
app.include_router(web_access_router, prefix="/api/v2", tags=["web_access"])
```

注意 **prefix 是 `/api/v2`**，不是 `/api/v1`。web_access 和 retrieval 两个 domain 目前在 v2 命名空间下；其他 domain 大多数在 v1。这是 Phase 2a 引入新 domain 时刻意选择的分区，避免把新 endpoint 混进 v1 的 legacy aggregate URL 表里。

## Search 能力

### 端点

```http
POST /api/v2/web/search
Authorization: Bearer sk-...
Content-Type: application/json

{
  "query": "ApeRAG 最新技术路线",
  "max_results": 5,
  "timeout": 30,
  "locale": "zh-CN",
  "source": "example.com"
}
```

- `query`：关键词（可空，但与 `source` 至少要有一个）
- `source`：限制到某个域名（`site:example.com query` 语义）。可以只提供 `source` 做"站内浏览"
- `max_results` / `timeout` / `locale`：常规可选参数

### Provider 选择策略

`/api/v2/web/search` handler 的策略（见 `api/routes.py`）：

1. 查当前用户是否配置了 JINA API key（`model_platform_service.get_user_provider_api_key(user_id=..., provider_type="jina", fallback_to_public=True)`）
2. 若有 → 尝试 JINA 搜索
3. 若 JINA 失败或无 key → 回落到 DuckDuckGo

DuckDuckGo 本身在 provider 内部还有一次 backend 降级（`auto` → `html` → `lite`），提升 zero-config 可用性。

### Soft-fail

Provider 出错时，handler **不返回 500**：它把搜索失败翻译成 `WebSearchResponse{results: [], meta: {search_status: "unavailable" | "empty" | "disabled", error_code: ...}}`，由调用方自行判断是"搜索无结果"、"provider 挂了"还是"未启用"。

### 结果合并

当两条 provider path 都有结果（比如 JINA 主搜 + fallback 的 DuckDuckGo 补齐）时，`_merge_and_rank_results` 按 URL 去重，按 `rank` 排序，限制到 `max_results`。

## Read 能力

### 端点

```http
POST /api/v2/web/read
Authorization: Bearer sk-...
Content-Type: application/json

{
  "url_list": ["https://example.com/a", "https://example.com/b"],
  "timeout": 30,
  "locale": "en-US"
}
```

- `url_list`：必填，至少一个 URL，长度限制在消费方各自强制（例如 knowledge_base 上游限制为 10 个以内）
- 单个 URL 就是 `url_list: ["..."]`，不提供单独的 `url` 字段

### Provider 选择策略

1. 查当前用户的 JINA API key
2. 若有 → JINA 优先，失败回退到 Trafilatura
3. 若无 → 仅用 Trafilatura（本地 HTML 解析，无外部依赖）

Trafilatura 是纯本地库，适合内网部署 / 不想把 URL 外泄给第三方的场景。

### 失败处理

Read 不做 soft-fail：某个 URL 抓取失败会体现在结果 item 的 `success=false`；但整个 endpoint 失败会抛 500。

## Provider 抽象

`search/base_search.py` 和 `reader/base_reader.py` 各自定义基类：

- `BaseSearchProvider` / `BaseReaderProvider`：定义 `async search(...)` / `async read(...)` 接口
- Concrete provider 继承基类，`config` 作为初始化参数传入 provider-specific 选项

`SearchService` / `ReaderService` 的工厂方法按 `provider_name` 字符串从 registry 里挑：

```python
provider_registry = {
    "duckduckgo": DuckDuckGoProvider,
    "ddg": DuckDuckGoProvider,
    "jina": JinaSearchProvider,
    "jina_search": JinaSearchProvider,
}
```

添加新 provider 的路径：

1. 在 `providers/` 下新建文件，继承 `BaseSearchProvider` / `BaseReaderProvider`
2. 在 service 的 `provider_registry` 里注册
3. 不需要改 HTTP 路由层，也不需要改 schema

## 跨 domain 消费

### knowledge_base

`knowledge_base/service/document_service.py` 通过 URL 导入功能把网页内容做成 document：

```python
from aperag.domains.web_access.reader.reader_service import read_with_jina_fallback
from aperag.domains.web_access.schemas import WebReadRequest
```

**直接 import**（不走 Protocol+DI），因为 `web_access` 是已 domain-moved 的 provider（canonical rule：domain-moved provider → direct import；详见 `docs/modularization/architecture.md` Section 3）。

该消费路径有自己的业务约束：

- URL 数量上限 10（在 knowledge_base 消费方强制，不是 web_access 的约束）
- 抓到内容后会走 knowledge_base 的 ingestion pipeline：embedder + chunker + vector store

### MCP server

`aperag/mcp/server.py` 把 `web_access` 的 search / read 暴露成 MCP tool：

```python
@mcp_server.tool
async def web_search(query: str = "", max_results: int = 5, ...):
    ...
```

这样接 Claude Desktop / Cursor 等 MCP host 时，它们能直接调 ApeRAG 的 web search 能力。

### Agent Runtime（间接消费）

Agent Runtime 通过 bot 配置里的 `web_search` / `web_read` tool 在 turn 里按需调用同一批 provider。这条路径经过 tool 封装层，不直接 import `web_access`；详见 `architecture/conversation-agent-evaluation.md` 里的 tool 章节。

## Identity 耦合（仅读 `id`）

`web_access/api/routes.py` 顶部有一个本地 `AuthenticatedUser(Protocol)` 声明：

```python
class AuthenticatedUser(Protocol):
    """Minimal auth-context contract the `web_access` domain depends on.
    web_access only reads the authenticated user's id ..."""
    id: object
```

原因：

1. **G16 规则**：非 identity 域不能 import `aperag.db.models.User` ORM
2. 这里只需要用户 `id` 做 JINA API key 的 per-user lookup，不需要 `role` / `email` 等其他字段
3. 声明本地 Protocol 比"collapse 到 Any"更诚实地表达依赖契约

SQLAlchemy `User` 类会 structurally 满足这个 Protocol（鸭子类型），所以 `Depends(required_user)` 不用改就能工作。

> 相关 canonical 细节（G15 literal-compare admin、UserView Protocol、14 份 AuthenticatedUser 有意重复未合并）见 `docs/modularization/architecture.md` Section 4。

## Schema 归属（Phase 2a hard-cut）

早期 `WebSearchRequest` / `WebReadResponse` 等 Pydantic 类定义在 `aperag/schema/view_models.py` 大聚合里。Phase 2a 把它们搬到 `web_access/schemas.py`，同时保证 **OpenAPI 组件名 + shape byte-for-byte 一致**，这样前端 SDK 生成文件不会 diff。

聚合 `view_models.py` 里保留一层 re-export shim（给 Phase 2a 之前的 import 继续工作），属于 Phase 7+ 的 cleanup 候选。

## 失败模式与运维

### JINA API key 泄露

API key 按 user 粒度配置，单个用户 key 泄露只影响该用户的搜索配额。Admin 通过 model provider 配置页面撤换即可。

### DuckDuckGo 限流

`duckduckgo-search` 包可能被 DDG 反爬限流。observe 到大量 `RatelimitException` 时：

- 检查是否有异常调用源（单 IP 大量请求）
- 临时给用户配 JINA API key，跳过 DDG

### Trafilatura 提取失败

部分 JavaScript-heavy 页面 Trafilatura 提取不出正文。配 JINA API key 后会走 JINA（能执行 JS 渲染），通常可解决。

## 相关文档

- `docs/modularization/architecture.md` Section 2.12 — canonical SSoT 的 `web_access` domain 定义
- [`architecture/domains.md`](./domains.md) — 12 domain 通览
- [`architecture/indexing-retrieval-kg.md`](./indexing-retrieval-kg.md) — knowledge_base URL 导入消费链路
- [`architecture/conversation-agent-evaluation.md`](./conversation-agent-evaluation.md) — Agent Runtime tool 层
- [`integration/openai-compat.md`](../integration/openai-compat.md) — 相关集成接口
- [`user-guide/content-import.md`](../user-guide/content-import.md) — URL 导入的用户面流程
