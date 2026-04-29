---
title: 知识库 / 索引 / 检索 / 知识图谱四域架构
position: 2
---

# Knowledge Base / Indexing / Retrieval / Knowledge Graph 四域架构

> 本文是 Post-Phase-6 current-state 架构文档，覆盖 `knowledge_base` · `indexing` · `retrieval` · `knowledge_graph` 四个后端 domain。**跨 domain 的 canonical rules 与 invariants**（direct import vs Protocol + DI、G1–G19 gates、dual-hook Scenario A、permanent seams、shim lifecycle）集中记录在 [`docs/modularization/architecture.md`](../../modularization/architecture.md)，本文 cross-reference 引用，不重复定义。

> **Baseline**: `origin/main @ 003073be`（Phase 6 / PR #1635 + architecture doc / PR #1636 + blueprint / PR #1637 + task #8 / PR #1640 + task #13 / PR #1638 + task #16 / PR #1639 + task #9+14 / PR #1641 全部 merged）。

---

## 1. 四域边界总览

这四个 domain 合并在同一篇，是因为它们共同构成 ApeRAG 的 **知识摄入 → 索引构建 → 检索召回 → 图谱抽取** 闭环：

```
用户创建 Collection (knowledge_base)
   │
   ├─ 用户导入文档 / URL / 文本 (knowledge_base.document_service)
   │    │
   │    └─→ indexing.manager 派发 5 类索引任务 (vector / fulltext / graph / summary / vision)
   │                 │
   │                 └─→ 产出实体与关系 → knowledge_graph.graphindex reconciler
   │
   └─ 用户 / Agent 发起检索 (retrieval.pipeline)
           │
           └─→ 聚合向量 + 全文 + 图谱召回 (retrieval 消费 GraphSearchContract → knowledge_graph)
```

四域分工：

| Domain | 职责 | 关键对象 |
| --- | --- | --- |
| `knowledge_base` | Collection / Document / CollectionSummary 的主体生命周期 | `Collection`、`Document`、`CollectionSummary`、`collection_service` / `document_service` / `collection_summary_service` |
| `indexing` | 索引 reconciler + 每种索引类型的 worker | `DocumentIndex`、`DocumentIndexManager`、vector / fulltext / graph / summary / vision 5 类 worker |
| `retrieval` | 检索 pipeline 编排 + chunk 聚合 + reranking | `SearchHistory`、`pipeline.py` + `service.py` |
| `knowledge_graph` | 实体 / 关系 ORM + Nebula + `graphindex` reconciler | `GraphCurationRun` / `Suggestion`、`service.py` + `graphindex/` 11 子模块 |

每个 domain 各自遵循 `db/` · `schemas.py` · `ports.py` · `service/` / `service.py` · `api/routes.py` 的 per-domain layout，详见 [`architecture.md §2 Domain map`](../../modularization/architecture.md#2-domain-map)。

### 1.1 四域之间的依赖形态

```
knowledge_base ──→ (IndexingTrigger Protocol)          → indexing (域内 provider)
knowledge_base ──→ (Collection / Document 直接 import)  ← 被 retrieval / agent_runtime / web_access 消费

indexing ──→ (CollectionIndexingView Protocol)         ← 结构性满足 by knowledge_base.Collection

retrieval ──→ (GraphSearchContract Protocol，单向)       → knowledge_graph 结构性满足
knowledge_graph 禁反向 import retrieval（G10 / G3 enforced）

knowledge_graph ──→ (CollectionRow Protocol)           ← 结构性满足 by knowledge_base.Collection

knowledge_base ──→ (SearchPipelineOps Protocol)         ← 结构性满足 by legacy aperag.service.search_pipeline_service
knowledge_base ──→ (QuotaOps Protocol)                  ← 结构性满足 by legacy aperag.service.quota_service（standalone-infra 永久 seam，SSoT §5.1）
```

以上 Protocol 全部是 **consumer-owned**（lesson 9a-quad）：consumer 自己在 `ports.py` 声明，provider 结构性满足而不反向 import consumer。详见 [`architecture.md §3.1`](../../modularization/architecture.md#31-direct-import-vs-protocol--di)。

---

## 2. Knowledge Base domain

canonical 位置：`aperag/domains/knowledge_base/`。

> **本节由 @cuiwenbo 作为 contributor/reviewer 负责供稿补充**（per blueprint Section 5.1 + PM msg=9b712260 point 3）。下面写的是 huangheng 的 scaffold；KB 内部细节（Collection / CollectionSummary / Document 三实体的完整字段、`collection_service` 作为 consumer 的 5 个 Protocol wire、CollectionConfig 的 JSON 配置 hydration、`document_service` 的两阶段状态机、SHA-256 去重、ingest 事务边界）由 cuiwenbo 的 markdown block 补齐。

### 2.1 数据模型

`aperag/domains/knowledge_base/db/models.py` 拥有 **3 个 ORM 实体** + **4 个 Enum**：

- `Collection` + `CollectionType` + `CollectionStatus` — 知识库；持 `user` owner、`title`、`description`、以及 `config`（JSON 存储的 `CollectionConfig` 序列化结果）。
- `CollectionSummary` + `CollectionSummaryStatus` — 知识库级别摘要（AI 自动生成，独立于 Document summary）。
- `Document` + `DocumentStatus` — 知识库里的单个文档；持 `collection_id` + `name` + `status` + `size` + `doc_metadata` 等字段。

24 个 Pydantic schema 定义在 `aperag/domains/knowledge_base/schemas.py`（包括 `Collection`、`CollectionView`、`CollectionCreate`、`CollectionUpdate`、`Document`、`DocumentList`、`DocumentView`、`MineruTokenTestResponse` 等），通过 **dual-hook Scenario A**（SSoT §3.3）绑回 `aperag.schema.view_models`，兼容 pre-migration 的 `from aperag.schema.view_models import Collection` import。

### 2.2 Service 层拓扑

3 个 service 位于 `aperag/domains/knowledge_base/service/`：

- `collection_service` — Collection CRUD + 5 个 **consumer-owned Protocol wire**：`MarketplaceOps` / `MarketplaceCollectionOps` / `SearchPipelineOps` / `QuotaOps` / `AuthenticatedUser`。前 4 个都是 legacy-provider（`aperag.service.*` 仍 active），通过 `aperag/app.py` 启动时注入到 `collection_service._marketplace_ops` / `_marketplace_collection_ops` / `_search_pipeline_ops` / `_quota_ops` 槽。
- `collection_summary_service` — 知识库摘要的生成 / 刷新 / 失败重试；独立于 Document summary。
- `document_service` — Document 两阶段提交（upload → confirm）+ SHA-256 去重 + 与 indexing domain 的 `IndexingTrigger` Protocol 接口；`_quota_ops` 同样通过 DI wire 检查与消费 `max_document_count_per_collection` 限额。

### 2.3 Consumer-owned Protocols（共 5 条）

`aperag/domains/knowledge_base/ports.py` 声明：

| Protocol | 作用 | Provider | 分类（SSoT §3.2） |
| --- | --- | --- | --- |
| `AuthenticatedUser` | KB 路由 handler 的 auth 依赖类型（lesson 9a-ter，避免 import `User` ORM） | fastapi-users `required_user` 返回的 `User` 行 | 本地 Protocol，不走 DI 槽 |
| `MarketplaceOps` | Collection 的 marketplace 发布 / 订阅 / 访问 gate | `marketplace.marketplace_service`（已搬 domain） | (A) legacy-not-moved-yet → Phase 4 后可回直接 import；Phase 6 未回收 |
| `MarketplaceCollectionOps` | marketplace subscriber search fallback | `marketplace.marketplace_collection_service`（已搬 domain） | (A) 同上 |
| `SearchPipelineOps` | 实际搜索执行（`execute_search`）；KB 作为 consumer 把检索请求代理到 retrieval pipeline | `aperag.service.search_pipeline_service`（尚未搬 domain；SSoT §8 F15 记录永久性分类 pending） | (A) 待定 |
| `QuotaOps` | 配额 check / release / query（针对 `max_collection_count` / `max_document_count` 等） | `aperag.service.quota_service`（standalone-infra 永久 seam，SSoT §5.1） | (B) standalone-infra permanent |

5 条 Protocol 全部 **consumer-owned**：G17 / KB consumer-owned boundary test 守住 provider 永远不能 `import aperag.domains.knowledge_base.ports`（详见 SSoT §4.1）。

### 2.4 跨 domain 消费者

- `indexing.manager.DocumentIndexManager` — 读取 `CollectionIndexingView` 判断需要物化哪几类索引。
- `retrieval.pipeline` — 执行检索时直接 `from aperag.domains.knowledge_base.schemas import Collection as KBCollectionSchema` 获取 Collection 上下文（provider-in-domain 的直接 import）。
- `conversation.chat_collection_service` / `chat_document_service` — 跨 domain 直接 import `collection_service` / `document_service` 做 chat 关联。
- `agent_runtime.runtime` — late-import `knowledge_base` 的 `Collection` schema 作 turn 上下文。
- `web_access.api.routes` — 跨 domain 直接 import KB `Collection` 做带网页增强的查询。

### 2.5 KB 作为全域 consumer-owned Protocol 的首个落地

KB 是整个后端 consumer-owned Protocol 模式（lesson 9a-quad）的 **首个完整落地 domain**：

- `collection_service` 持 4 个 module-level DI 槽（`_marketplace_ops` / `_marketplace_collection_ops` / `_search_pipeline_ops` / `_quota_ops`），全部由 `aperag/app.py` 启动时注入。G17 `test_phase4_di_critical_wirings_at_app_startup` 守住 runtime smoke：`import aperag.app` 后任一槽为 `None` 则 CI 红。
- `test_knowledge_base_protocol_boundary_is_consumer_owned` 进一步守住 **Protocol 方向** —— AST 扫描 `marketplace_service.py` / `marketplace_collection_service.py` / `search_pipeline_service.py` / `quota_service.py` 4 个 legacy provider file，禁止任一反向 `import aperag.domains.knowledge_base.ports`；provider 只能结构性满足，不能反向依赖 consumer 契约。
- KB 的 `QuotaOps` 与 `conversation.bot_service` 的 `QuotaOps` 是两份**独立**的 consumer-owned Protocol，声明文件不同但 provider 都是 `aperag.service.quota_service`（standalone-infra permanent）— 这是 lesson 9a-quad 的典型示范，消费方彼此互不引用，各自声明最小契约。

完整 G17 / KB consumer-owned boundary test 的 runtime smoke 与 AST 扫描语义见 [`architecture.md §4.1 Backend gate catalog`](../../modularization/architecture.md#41-backend-gate-catalog)。

### 2.6 Collection / CollectionSummary / Document 状态机速览

供稿 by @cuiwenbo (thread msg=f58f4310 精简后)：

- **`CollectionStatus`**: `ACTIVE` / `INACTIVE` / `DELETED` —— 软删除语义，由 `collection_service.delete_collection` 推进。
- **`CollectionSummaryStatus`**: `PENDING` / `RUNNING` / `COMPLETE` / `FAILED` —— 异步 LLM 摘要任务状态；由 `collection_summary_service` 触发与重试。
- **`DocumentStatus`**: `UPLOADED` / `PENDING` / `RUNNING` / `COMPLETE` / `FAILED` / `DELETED` —— 两阶段提交的用户可见状态：`upload` API 落 `UPLOADED` 暂存；`confirm` API → `PENDING` → indexing worker 驱动至 `COMPLETE` / `FAILED`。
- **User write cross-ref**: `Collection.id` 在用户首次使用时通过 `identity.service.identity_user_ops.set_chat_collection(session, user_id, collection_id)` facade 写回 `User.chat_collection_id` —— 非 identity domain 对 User 写操作的**唯一合法路径**（lesson 9a-sexdec hierarchy-1，详见 [`architecture.md §3.4`](../../modularization/architecture.md#34-user-write-hierarchy-lesson-9a-sexdec)）。

---

## 3. Indexing domain

canonical 位置：`aperag/domains/indexing/`。

### 3.1 数据模型

`aperag/domains/indexing/db/models.py` 有 1 实体 + 2 枚举：

- `DocumentIndex` — 单个文档的单个索引 spec；`(document_id, index_type)` 为复合主键；track `status` / `task_id` / `config_hash` / `error_message` 等字段。
- `DocumentIndexType` — 5 类：`VECTOR` / `FULLTEXT` / `GRAPH` / `SUMMARY` / `VISION`。
- `DocumentIndexStatus` — `PENDING` / `RUNNING` / `COMPLETED` / `FAILED` / `DELETING` 等。

Pydantic schema 不在此 domain 内声明（KB consumer 直接消费 DB model 即可；indexing 不对外暴露 API）。

### 3.2 Reconciler 与 worker

`aperag/domains/indexing/manager.py::DocumentIndexManager` 是 **核心 reconciler**，它只负责 `DocumentIndex` 行的 CRUD 与状态推进：

```python
class DocumentIndexManager:
    async def create_or_update_document_indexes(session, document_id, index_types=None):
        # 若 index_types=None → 从 collection.config 推导 live set
        # 对每个 index_type，upsert DocumentIndex 行；status 置 PENDING
        ...
    async def delete_document_indexes(session, document_id, index_types):
        # 显式指定 index_types；status 置 DELETING
        # 实际删除由 indexing-worker cleanup loop 按状态驱动
        ...
```

真正的索引构建逻辑分布在各 modality worker 里，由独立 `indexing-worker` 进程消费 Redis-backed queue：

| 模块 | 对应 `DocumentIndexType` | 职责 |
| --- | --- | --- |
| `vector_index.py` | `VECTOR` | 生成 embedding + 写向量库（Qdrant / pgvector / Milvus 等，由 Collection 配置选择） |
| `fulltext_index.py` | `FULLTEXT` | 抽取关键词 + 写 Elasticsearch 全文索引 |
| `graph_index.py` | `GRAPH` | 抽取实体 / 关系 → 调用 `knowledge_graph.graphindex` 入库 |
| `summary_index.py` | `SUMMARY` | 对文档生成 LLM 摘要 → 存 Document summary 字段 |
| `vision_index.py` | `VISION` | 图像识别 / OCR（若 Collection 启用 `enable_vision`） |

此外 `document_parser.py` 是共享的文档解析入口：把上传的原始文件转换成标准化的 chunk 序列（`base.py::BaseDocumentParser`），所有索引 worker 都从这里取 chunk 而不是各自重新解析。

### 3.3 Consumer-owned Protocols（2 条）

`aperag/domains/indexing/ports.py`：

| Protocol | 作用 | 结构性 provider |
| --- | --- | --- |
| `CollectionIndexingView` | indexing 读 Collection 时的最小字段集：`id` + `user` + `config`（`Any`，由 `parseCollectionConfig` 在调用点 hydrate） | `knowledge_base.Collection` ORM |
| `IndexingTrigger` | 写侧 entry：consumer（KB）调 `create_or_update_document_indexes` / `delete_document_indexes` 把文档塞进或移出索引 pipeline | `DocumentIndexManager` 单例（通过 factory 绑定） |

两条 Protocol 都是 consumer-owned（lesson 9a-quad 的 **provider-side variant**：`IndexingTrigger` 的 consumer 是 KB，但 Protocol 声明在 indexing 侧，因为 indexing 才知道写侧语义 `DocumentIndexType` 和 idempotent upsert 策略）。KB `document_service` 把依赖绑到 `IndexingTrigger`，运行时由 factory 注入 `DocumentIndexManager`。

### 3.4 无 API 路由

indexing domain **不对外暴露 HTTP 路由** — 它是 KB 驱动的内部重建任务后端。所有 HTTP 交互都以 KB 侧的 "重建索引" / "删除索引" 路由为入口，KB 把请求翻译成 `IndexingTrigger` 调用。

### 3.5 indexing worker 协作

索引任务的异步派发由 KB service 写入 `DocumentIndex` intent 并 enqueue 到 Redis-backed queue；`python -m aperag.cli.indexing_worker` 启动 parse、vector、fulltext、graph、graph_facts、graph_vectors、summary、vision、reconciler、cleanup 等 lane。`DocumentIndex` 行是业务状态真源，Redis 只作为可丢 transport。

---

## 4. Retrieval domain

canonical 位置：`aperag/domains/retrieval/`。

### 4.1 数据模型

- `SearchHistory`（`aperag/domains/retrieval/db/models.py`） — 持久化用户历史检索记录（query + 结果快照 + collection_id + user），用于 UI 回溯与调试。

`retrieval/schemas.py` 声明 10 个 Pydantic schema：`SearchRequest` / `SearchResult` / `SearchResultItem` / `SearchResultMetadata` / `SearchType` 等，走 dual-hook 绑回 `view_models`。

### 4.2 Service 层

- `service.py` — `retrieval_service` 单例；负责 SearchHistory CRUD + HTTP 路由背后的轻逻辑（历史查询 / 分页 / 权限过滤）。
- `pipeline.py` — `SearchPipeline` 核心：一次检索请求的完整编排，合并向量 / 全文 / 图谱 / 知识图谱补充召回、reranker 打分、chunk 聚合，byte-for-byte 保留 Phase 2 hard-cut 前的 `aperag.service.search_pipeline_service` 算法语义。

### 4.3 Consumer-owned Protocol — `GraphSearchContract`（单向 Protocol bridge）

`aperag/domains/retrieval/ports.py` 声明：

```python
class GraphSearchContract(Protocol):
    async def query_context(self, ...) -> HasTextPayload: ...
```

- **retrieval 是 consumer**：`pipeline.py::_graph_search` 的依赖类型绑到 `GraphSearchContract`。
- **provider 是 knowledge_graph**：`knowledge_graph.graphindex.service` 构造的图索引 service 实例结构性满足 `GraphSearchContract` 的 `query_context` 方法。
- **单向边界**：retrieval 可以读 `aperag.domains.knowledge_graph.graphindex.*`（infrastructure，不受 G1 ban）；但**不能** import `aperag.domains.knowledge_graph.ports` / `.service` / `.schemas` —— 否则会在 AST 层面制造循环依赖。G10 / G3 同步禁止 knowledge_graph 反向 import retrieval（`retrieval.ports` / `retrieval.service` / `retrieval.schemas` / `retrieval.pipeline`）。

详见 [`architecture.md §4.1 G10/G3 test function`](../../modularization/architecture.md#41-backend-gate-catalog) + lesson 9a-quad。

### 4.4 API

- `POST /api/v2/collections/{id}/search` — 同步检索（pipeline 执行完返回全部结果）
- `GET /api/v2/collections/{id}/search_history` — 历史记录分页
- `DELETE /api/v2/collections/{id}/search_history/{sid}` — 历史删除

所有 handler 使用 local-decl `AuthenticatedUser(Protocol)`（G4 + G16 要求）。

### 4.5 与 indexing 的关系

检索流程本身不触碰 `indexing.manager`：retrieval 只从向量库 / ES / 图库 **读**，不管索引如何构建。但 retrieval 的 chunk 聚合依赖 `indexing.fulltext_index::extract_keywords` 做关键词抽取（常见的小工具函数直接 import），这是已搬 domain 的 provider 到 provider 的直接 import。

---

## 5. Knowledge Graph domain

canonical 位置：`aperag/domains/knowledge_graph/`。

### 5.1 数据模型

`aperag/domains/knowledge_graph/db/models.py`：

- `GraphCurationRun` + `GraphCurationRunStatus` — 一次「图谱归并候选发现」运行（batch 级），持 `collection_id` + `status` + `generator_config_hash` + 时间戳。
- `GraphCurationSuggestion` + `GraphCurationSuggestionStatus` — 单条合并建议（「某两个实体可能重复」），关联 `run_id`；状态覆盖 `PENDING / APPROVED / REJECTED / APPLIED`。

15 个 Pydantic schema 在 `schemas.py` 里（`GraphLabelsResponse`、`KnowledgeGraph`、`MergeEntityRequest`、`GraphCurationRunView` 等），走 dual-hook。

### 5.2 Service

- `service.py::knowledge_graph_service` — HTTP-facing 业务：
  - `get_graph_labels(collection)` — 返回实体类型清单，给 UI 下拉框用。
  - `get_knowledge_graph(collection, ...)` — 抓取子图，给前端可视化用。
  - `merge_entities(collection, entity_ids, ...)` — 把多个实体合并成一个，LLM 生成聚合描述。
- `graphindex/` 11 子模块 —— 实际存图 / 查图 / 抽取实体关系的核心，下一节详述。

### 5.3 `graphindex/` 子包 — 图引擎

```
graphindex/
├── __init__.py            # 统一 public API export
├── config.py              # Nebula / Neo4j / PostgreSQL 三种后端配置
├── dto.py                 # Entity / Relation / Chunk 等内部 DTO
├── engine/
│   ├── chunking.py        # 文档切片
│   ├── extraction.py      # LLM entity / relation extraction
│   └── indexer.py         # 索引组合执行器
├── integration.py         # 与 indexing.graph_index worker 的接口
├── models.py              # 用于存图的中间 model
├── prompts.py             # LLM extraction / merge prompt templates
├── service.py             # graphindex 运行时 service（是 retrieval GraphSearchContract 的 structural provider）
└── storage/
    ├── base.py            # 统一图存储 Port
    ├── connector.py       # 连接 factory
    ├── nebula.py          # Nebula Graph backend
    ├── neo4j.py           # Neo4j backend
    └── postgres.py        # PostgreSQL age backend
```

`graphindex/` 是「基础设施」而非 domain 下的业务层——G1 不禁止其他 domain 从这里直接 import（retrieval 就这么用），但它仍属于 knowledge_graph 的物理包。Phase 2 hard-cut 把它从 top-level `aperag/graphindex/` 搬到 `aperag/domains/knowledge_graph/graphindex/`，保留全部原算法。

### 5.4 Consumer-owned Protocol — `CollectionRow`

`aperag/domains/knowledge_graph/ports.py` 声明：

```python
class CollectionRow(Protocol):
    id: str
    user: str
    title: str
    config: Any
```

knowledge_graph 的 `service.py` + `graphindex/integration.py` 只读 Collection 的 4 个字段（id / user / title / config）；`knowledge_base.Collection` ORM 结构性满足，但 knowledge_graph 不 import KB 的 `Collection` class，而是把函数参数类型绑到 `CollectionRow` Protocol。这样避免了 KG → KB 的静态依赖。

### 5.5 与 retrieval 的单向 Protocol

本节在 §4.3 已讲：retrieval 声明 `GraphSearchContract`，`graphindex.service` 的运行时实例结构性满足。G10 / G3 boundary test 禁止 knowledge_graph 反向 import retrieval。

### 5.6 `graph_curation` 外部模块

归并建议发现逻辑仍住在 `aperag/graph_curation/`（top-level，非 domain），G1 不禁止从 knowledge_graph domain 调用它。`service.py::merge_entities` 内部实际调用的就是 `aperag.graph_curation.service` 的候选生成器。

### 5.7 路由

`aperag/domains/knowledge_graph/api/routes.py` 提供图谱查询 + 合并 endpoint。外部还保留 `aperag/views/graph.py` 里的 **一条 410-Gone legacy shim**（`/collections/{id}/graphs/export/kg-eval`），是 Phase 2 hard-cut 的 tombstone；G14 boundary test 守住除这一条外再无 KG route 留在 `aperag/views/graph.py`。

---

## 6. 共享基础设施（跨四域）

这是本文最跨 domain 的一节。本节描述 **`aperag/schema/common.py` 里的共享原语** 与 **dual-hook Scenario A** 两条 cross-cutting 机制——它们跨过 KB / indexing / retrieval / KG 四个 domain 而存在，但都不属于任何单一 domain。

详细不变式（准入准则、原理证明、AST 绕过技巧）全部在 [`architecture.md §2.3`](../../modularization/architecture.md#23-shared-primitive-module--aperagschemacommonpy) + [`§3.3`](../../modularization/architecture.md#33-dual-hook-scenario-a--view_models-re-export-without-triggering-g1) 写死，本节仅做 zh-CN 导读。

### 6.1 `aperag/schema/common.py` 共享原语 [@cuiwenbo 供稿位]

当前 8 个 entry：

- `ModelSpec` — LLM 模型 spec（`embedding` / `completion` 配置）；消费方：`knowledge_base` / `model_platform` / `conversation`。
- `KnowledgeGraphConfig` — 图谱配置子树；消费方：`knowledge_base`（Collection.config 一部分） + `indexing`（graph worker 的 prompt 配置）。
- `IndexPrompts` — 索引 prompt 覆盖（`graph` / `summary` / `vision`）；消费方：`knowledge_base` + `indexing`。
- `CollectionConfig` — KB 公共配置 shape；消费方：`knowledge_base` / source ingestion / `conversation.bot` / `retrieval`。
- `PageResult` + `PaginatedResponse` — 分页包络；消费方：≥6 个 domain 的 list endpoint。
- `Chunk` + `VisionChunk` — 检索 / 切片原语；消费方：`knowledge_base` + `retrieval` + `indexing` + vision。

**严格准入准则**（SSoT §2.3）：一个类型能进 `common.py` 当且仅当：

1. 它是**纯 Pydantic 原语**（值对象，无 domain 特定语义，无 ORM 依赖），且
2. 被 **≥2 个 domain 消费**。

任何只被 1 个 domain 用的类型必须留在该 domain 的 `schemas.py`，禁止进 `common.py`。本 gate 由 CR 人审，**没有 AST 自动检查** —— 这是刻意设计，避免 `common.py` 退化回 `view_models.py`-style 的 catch-all。

反例（绝**不**能进 `common.py`）：

- `Collection` / `CollectionView` — 仅 KB 语义，留 `aperag/domains/knowledge_base/schemas.py`。
- `ChatMessage` / `TurnFeedback` — 仅 conversation 语义。
- `AgentMessage` — 仅 agent_runtime 语义（Phase 5 5-S5a 单独 carve）。
- `User` / `Role` ORM — identity 所有权，由 G15 / G16 硬禁。

与其他 3 类 legacy 聚合层（`view_models.py` / `db/models.py` / `service/*.py`）的关键区别：后三者是"大杂烩"容器，domain 若 import 就会把一切未搬 symbol 暴露出来；`common.py` 靠严格准入防止 catch-all 漂移，G1 因此**刻意不**把它列入禁令清单，`aperag/domains/<d>/schemas.py` 可以直接 import。

### 6.2 Dual-hook Scenario A — view_models 兼容层 [@cuiwenbo 供稿位]

**问题**：pre-migration 的代码用 `from aperag.schema.view_models import <X>`；P0 重构把 schema 搬进 `aperag/domains/<d>/schemas.py`；G1 禁止 `aperag/domains/**` import `aperag.schema.view_models`。如何既保留 pre-migration import 又不违反 G1？

**解法**（"Scenario A"，Phase 3 Step 4b 首次 in KB，之后 Phase 4 / 5 所有迁移 schema 的 domain 都重用）：

```python
# aperag/schema/view_models.py 末尾
try:
    from aperag.domains.knowledge_base.schemas import (  # noqa: E402, F401
        Collection, CollectionView, ..., MineruTokenTestResponse,
    )
except ImportError:
    pass
```

```python
# aperag/domains/knowledge_base/schemas.py 末尾
def _bind_view_models_reexports() -> None:
    import sys
    vm = sys.modules.get("aperag.schema.view_models")
    if vm is None:
        return
    for name in __all__:
        setattr(vm, name, globals()[name])

_bind_view_models_reexports()
```

**关键点**：`sys.modules.get("aperag.schema.view_models")` 是**运行时的字符串查找**，不是 `import` 语句，因此 G1 AST 扫描不会报警。无论哪个模块先加载，最终都收敛到 `view_models.X is aperag.domains.<d>.schemas.X`（**单一 class 对象 identity**）。

`aperag/schema/view_models.py` 当前尾部有 **6 个 dual-hook `try:` block**（knowledge_base / identity / governance / marketplace / model_platform / conversation）+ 1 个 agent_runtime `AgentMessage` 单独 block，共 7 条。

**两种加载顺序都 converge**（cuiwenbo thread msg=179d4a3e 精简）：

- 若 `view_models` 先加载 → 末尾 try-block 触发 `import aperag.domains.<d>.schemas` → domain 模块运行 `_bind_view_models_reexports()`，`sys.modules.get("aperag.schema.view_models")` 命中 in-progress module → `setattr` 把 class 对象写回。
- 若 domain schemas 先加载 → hook 里 `sys.modules.get(...)` 返回 `None`，early-return noop → 稍后 `view_models` 加载，末尾 try-block `import aperag.domains.<d>.schemas` → domain 已在 `sys.modules`，Python import 机制直接复用，try-block 拿到同一 class 对象。

两条路径的终态都是 `aperag.schema.view_models.X is aperag.domains.<d>.schemas.X`（单一 class identity），与加载顺序无关。

---

## 7. 跨 domain runtime flow

以下三条 flow 串讲四域的运行时协作。

### 7.1 Document ingestion flow（KB → indexing → KG）

```
1. 用户 POST /api/v2/collections/{id}/documents (knowledge_base.api.routes)
2. document_service.create_document:
   - _quota_ops.check_and_consume_quota(max_document_count)
   - Document ORM insert (status=UPLOADED)
   - SHA-256 dedup
   - response 返回；用户侧看到文档进入「待确认」
3. 用户 POST .../documents/{id}/confirm:
   - Document.status → PENDING
   - 调用 IndexingTrigger.create_or_update_document_indexes(collection, document_id, None)
     ↓
   - indexing.manager 为每种 index_type upsert DocumentIndex 行 (status=PENDING)
4. indexing worker 启动 (`python -m aperag.cli.indexing_worker`):
   - 按 DocumentIndex 行 PENDING → RUNNING
   - document_parser 解析原始文件 → 标准化 chunk 序列
   - 对每种 index_type 调对应 worker：
     · vector_index → embed + 向量库写
     · fulltext_index → ES 写
     · graph_index → graphindex/engine/extraction.py 抽实体关系 → graphindex/storage/ 写图
     · summary_index → LLM 生成摘要 → Document summary 字段
     · vision_index → 图像 OCR + 视觉 chunk
   - 每个 worker 完成后 DocumentIndex.status=COMPLETED
5. 全部 COMPLETED 后 Document.status → COMPLETED
```

**边界语义**：
- KB 永远不直接操作 `DocumentIndex` 行（只通过 `IndexingTrigger` Protocol 触发）。
- indexing worker 永远不直接写 `Document` 表（只读 `Document` + 写 `DocumentIndex`）。
- knowledge_graph 的 graphindex storage 不被 KB / retrieval 绕过，所有写入都必须经 `graph_index.py` worker。

### 7.2 Retrieval flow（retrieval 聚合四路召回）

```
1. 用户 / Agent POST /api/v2/collections/{id}/search
2. retrieval.api.routes → SearchPipeline.run:
   - 构造 embedding (llm/embed)
   - 并发执行：
     · 向量召回（vector store SDK，直接读）
     · 全文召回（Elasticsearch，直接读）
     · 图谱召回：依赖 GraphSearchContract Protocol
                 → knowledge_graph.graphindex.service.query_context
   - 聚合 + reranker 打分（llm/rerank）
   - SearchHistory 写入一条历史记录
3. response 返回 Hit list + metadata
```

**边界语义**：
- retrieval 通过 `GraphSearchContract` Protocol 读 KG，**不** import KG 的 service / schemas / ports。
- retrieval 调用 `indexing.fulltext_index.extract_keywords` 是 provider-in-domain 的直接 import（已搬 domain，不需要 Protocol 中转）。

### 7.3 Knowledge graph curation flow（KG 实体合并）

```
1. 异步后台 task 扫描 collection 实体 → 生成合并建议
   (graph_curation/candidate_generation.py，属于 top-level 基础设施，非 domain)
   ↓
   写入 GraphCurationRun + GraphCurationSuggestion 行
2. 用户在 UI 看到合并建议 → POST /api/v1/knowledge_graph/curation/suggestions/{id}/approve
3. knowledge_graph.service.merge_entities:
   - 读取两个实体的属性 + 关联关系
   - 调用 LLM 生成聚合描述
   - 通过 graphindex.storage 把子图合并写回
   - GraphCurationSuggestion.status → APPLIED
```

**边界语义**：
- `graph_curation/` 是基础设施（不在 `aperag/domains/` 下），domain 可以直接调用它（不受 G1 ban）。
- KG 的所有实体 / 关系写入都走 `graphindex.storage.base::BaseGraphStorage`，屏蔽 Nebula / Neo4j / PostgreSQL age 三种后端的差异。

---

## 8. 边界注意事项

写新代码时常见陷阱：

- **G1**：任何 `aperag/domains/**` 文件都禁止 import `aperag.service.*` / `aperag.schema.view_models` / `aperag.db.models`。要用老路径的代码，先确认它是否已搬进 `aperag/domains/**`；若已搬则直接 import domain 版本，若没搬则走 consumer-owned Protocol + DI 槽。
- **G10 / G3**：retrieval ↔ knowledge_graph 是**单向** Protocol bridge。knowledge_graph 永远不 import `retrieval.*`；retrieval 只允许 import `aperag.domains.knowledge_graph.graphindex.*`（infrastructure），不能 import `knowledge_graph.ports` / `service` / `schemas`。
- **G14**：`aperag/views/collections.py` / `aperag/views/graph.py` 除了既有的 1 条 410-Gone tombstone，不能再出现 retrieval / KG 路由装饰器。新 route 直接加到 `aperag/domains/<d>/api/routes.py`。
- **G19**：`aperag/domains/**/api/routes.py` 文件头部**禁用** `from __future__ import annotations`。PEP 563 与 FastAPI 204 响应模型处理的相互作用会产生静默 bug（lesson 9a-quatuordec）。
- **Schema 落位**：跨 ≥2 domain 的纯值对象才能进 `aperag/schema/common.py`（准入严格）；只有一个 domain 用的 schema 进该 domain 的 `schemas.py`，走 dual-hook 绑回 `view_models` 即可。
- **indexing domain 永不暴露 HTTP**：所有「重建索引」/「删除索引」路由必须从 KB 侧入口，把写请求翻译成 `IndexingTrigger` 调用。
- **KB 的 5 条 Protocol**：新加 consumer 时按现有 pattern 声明 —— consumer 自持 Protocol + 模块级 `_ops` 槽 + `aperag/app.py` 启动时注入。不要把 legacy provider（如 `search_pipeline_service`）的 class 直接 import。
- **graphindex 是基础设施**：`aperag.domains.knowledge_graph.graphindex.*` 可以被 retrieval / indexing.graph_index 等多域 import，这是刻意的，G1 不禁。但具体存图 / 查图 API 仍通过 `GraphSearchContract` 等 Protocol 约束，避免业务层跨 domain 直连。

---

## 9. 相关文档

- [`docs/modularization/architecture.md`](../../modularization/architecture.md) — 后端整体 canonical current-state；本文多处 cross-ref 这里的 §2 domain map / §2.3 common.py / §2.4 indexing-retrieval-kg high-level / §3 canonical rules / §4.1 gate catalog / §5 runtime seams / §6 shim lifecycle。
- [`docs/modularization/breaking-changes/phase3-knowledge_base.md`](../../modularization/breaking-changes/phase3-knowledge_base.md) — Phase 3 knowledge_base + indexing 硬切分记录，含原始 G1 / G13 / G14 gate 设计讨论与 dual-hook Scenario A 首次落地。
- [`architecture/overview.md`](./overview.md) — 整个 `architecture/` 目录的入口导航。
- [`architecture/domains.md`](./domains.md) — 12 domain 通览。
- [`user-guide/document-upload.md`](../user-guide/document-upload.md) — 文档上传的用户视角流程（cuiwenbo 主笔）。
- [`user-guide/content-import.md`](../user-guide/content-import.md) — URL / 文本导入的用户视角流程。
- [`user-guide/knowledge-export.md`](../user-guide/knowledge-export.md) — 知识库 ZIP 导出流程。
- [`admin-guide/prompt-customization.md`](../admin-guide/prompt-customization.md) — `IndexPrompts` / `index_graph` / `index_summary` / `index_vision` 用户层覆盖机制。

---

*Document baseline: `origin/main @ 003073be`（post-PR #1641 merge）. 基础设施变动（Nebula / Neo4j / Postgres age backend）、新增 graph worker、新增 common.py entry，都应先更新 `docs/modularization/architecture.md` 的 canonical section，再回头更新本文 cross-ref。*
