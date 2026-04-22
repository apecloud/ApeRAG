# 图数据库抽象层设计与计划（ApeRAG）

> Status: **设计与计划文档（仅文档，无代码改动）**。落地分 M1 / M2 / M3
> 多个小 PR，详见 §8 路线图。
>
> **先读这个**：[`lightrag_refactor.md`](./lightrag_refactor.md)。本文
> 的 **Layer B** == 那份文档的 **Phase 1 facade**。实际推荐的落地顺序
> 是"先做 LightRAG facade（覆盖了本文的 M1 + M2），再按需清理内部存储
> 抽象（本文的 M3）"。两份文档**刻意拆成两份**是为了分别从两个角度回答
> 同一组问题，读顺序见 `lightrag_refactor.md` §7。
>
> 关联文档：[`vector_db_abstraction.md`](./vector_db_abstraction.md)。
> 向量抽象层的设计思路与本文保持一致，优先"最小可行、不留尾巴、不过
> 度设计"。

---

## 1. 背景与目标

### 1.1 为什么现在做这件事

ApeRAG 的知识图谱（KG）路径目前**完全由内嵌的 LightRAG 实现**承担：
三种图后端（`PGOpsSyncGraphStorage` / `Neo4JSyncStorage` /
`NebulaSyncStorage`）并存，通过 `GRAPH_INDEX_GRAPH_STORAGE` env 切换。
"支持多后端" 这件事**技术上已经可用**，问题出在：

- 业务代码（`graph_service.py` / `search_pipeline_service._graph_search`
  / `tasks/collection.py::_delete_lightrag`）直接调用 `create_lightrag_instance(...)`
  + `rag.X(...)`——LightRAG 的 API 细节泄漏到业务层。
- `LightRAG` 实例是 **per-request 构造 + `try/finally finalize_storages()`**
  的，每次调用 API 走一遍完整的 storage 初始化/拆除流程。路径稠密后，
  一个"查询图"API 就是 5 个子 storage 的冷启动。
- 已经踩到过第一个真实 bug：`search_pipeline_service._graph_search`
  **创建了 LightRAG 实例但从来没 `finalize_storages()`**——见 §3.R1。
- 未来计划把 lightrag **从 "内嵌 Python 对象" 改成 "web service"**；
  到时候 `create_lightrag_instance` 要变成 "连接一个服务并发 RPC"。如果
  现在业务层直接依赖 LightRAG，迁移时整个 ApeRAG 侧代码都要跟着改。

### 1.2 本文档的目标

**指导下一阶段的图数据库抽象改造**。具体来说：

1. 以**事实**清单的形式把现状固定下来，后面做任何改动都有对照。
2. **勘定抽象层的边界**：哪些抽象该做在 ApeRAG 侧，哪些该做在（未来的）
   lightrag 服务内部——两个边界不能混。
3. 列出**顺手能修的 code review 问题**，避免抽象落地时被它们污染。
4. 给出一个**分阶段的路线图**（M1/M2/M3），每个阶段都能独立上线、独立
   回滚，不留代码债。
5. 明确**反过度设计**：哪些"看起来该抽象"的地方，实际上**不应该**动。

### 1.3 非目标

- 本文**不提议**重写 LightRAG 的 `BaseGraphStorage` 或其 25 个跨后端
  一致性测试。现有实现是可用的，重写成本远高于收益。
- 本文**不提议**把 entity / relation 向量合并进
  [`aperag/vectorstore`](./vector_db_abstraction.md) 的 shard。那件事属
  于 lightrag 内部重构，且 lightrag 要搬家，详见 §7.4。
- 本文**不提议**替换 Neo4j / Nebula / pg-emulated 三种后端中任意一种。
  这三个已经各有各的使用场景（见 §6 能力矩阵）。

---

## 2. 现状事实清单

所有引用基于 2026-04-22 的代码快照。

### 2.1 入口与生命周期

- **唯一工厂**：`aperag/graph/lightrag_manager.py::create_lightrag_instance(collection)`
  （约 59–124 行）。`LightRAG` 实例**不缓存**，每次调用重新构造 +
  `await rag.initialize_storages()`。
- **workspace = collection.id**：LightRAG 的 "workspace" 绑定 ApeRAG
  collection id，实现按 collection 的逻辑隔离。
- **`finalize_storages()` 的触发**：
  - `_process_document_async` / `_delete_document_async` 的 `finally`
    （约 215、227 行）；
  - `graph_service.py` 中 5 处 handler 的 `try/finally`（约 43-48、85-118、
    287-296、413-425、449-454 行）；
  - `tasks/collection.py::_delete_lightrag`（约 193-194 行）。
- **漏写 `finalize_storages` 的地方**：`search_pipeline_service._graph_search`
  （约 265-273 行）——见 §3.R1。

### 2.2 三种图后端

`aperag/graph/lightrag/kg/__init__.py::STORAGES`（42–48 行）注册 5 个类：

| 类型 | 类名 | 后端 | 文件 |
|---|---|---|---|
| GRAPH | `PGOpsSyncGraphStorage` | PostgreSQL（模拟图，表：`lightrag_graph_nodes` / `lightrag_graph_edges`） | `kg/pg_ops_sync_graph_storage.py` |
| GRAPH | `Neo4JSyncStorage` | Neo4j 原生图，Cypher | `kg/neo4j_sync_impl.py` |
| GRAPH | `NebulaSyncStorage` | NebulaGraph 原生图，nGQL | `kg/nebula_sync_impl.py` |
| KV | `PGOpsSyncKVStorage` | PostgreSQL（分 namespace：text_chunks / llm_cache / doc_status / ...） | `kg/pg_ops_sync_kv_storage.py` |
| VECTOR | `PGOpsSyncVectorStorage` | PostgreSQL + pgvector（分 namespace：entities / relationships / chunks） | `kg/pg_ops_sync_vector_storage.py` |

可通过 env 任意组合，但实际部署常见的是 `(PGOpsSyncKVStorage,
PGOpsSyncVectorStorage, X)`，其中 `X` 根据规模选 PG / Neo4j / Nebula。

### 2.3 已经存在的抽象基类

- `aperag/graph/lightrag/base.py::StorageNameSpace`（约 128-172 行）：
  `initialize / finalize / drop` 三方法抽象。
- `BaseVectorStorage`（约 175-251 行）：8 个抽象方法。
- `BaseKVStorage`（约 254-292 行）：5 个抽象方法。
- `BaseGraphStorage`（约 295-606 行）：**13 个必须实现 + 11 个带默认实现
  的 batch/扩展方法**，共 24 个接口点。

### 2.4 跨后端一致性测试

`tests/integration/graphstorage/test_graph_storage.py::GraphStorageTestSuite`
定义 **25 个测试方法**，覆盖 `has_node / get_node / node_degree /
upsert_node / delete_node / has_edge / get_edge / get_nodes_batch /
edge_degrees_batch / data_integrity / large_batch_operations / ...`。

- 每个后端（`test_postgres_graph_storage.py` / `test_neo4j_storage.py`
  / `test_nebula_storage.py`）把 `Oracle` 实例化后喂给同一个 suite。
- 这是"**事实上的契约**"——任何后端都要过同一套 25 个测试才算实现正确。
- 本文档不提议替换这个模式；它就是跨后端等价性的最佳形式。

### 2.5 连接管理

- **Neo4j**：`aperag/db/neo4j_sync_manager.py::Neo4jSyncConnectionManager`
  （约 28-44 行）——class-level lazy singleton + `threading.Lock`，driver
  进程级复用。
- **Nebula**：`aperag/db/nebula_sync_manager.py` 同结构。
- **PG**：不需要独立管理器，直接走 `aperag/config.py` 的 `sync_engine`。

**结论**：连接池本身**不是**当前的瓶颈。瓶颈在 LightRAG 级别的 "per-request
构造 storage 对象 + initialize + finalize" 开销。

### 2.6 业务层对 LightRAG 的调用面

按功能归类：

| 业务动作 | LightRAG 方法 | 调用点 |
|---|---|---|
| 索引文档 | `rag.ainsert_and_chunk_document` + `rag.aprocess_graph_indexing` | `lightrag_manager._process_document_async` |
| 删除文档图数据 | `rag.adelete_by_doc_id` | 同上 + `tasks/collection.py` |
| 图检索（for RAG） | `rag.aquery_context` | `search_pipeline_service._graph_search` |
| 查标签列表（for UI） | `rag.get_graph_labels` | `graph_service.get_graph_labels` |
| 查子图（for UI） | `rag.get_knowledge_graph` | `graph_service.get_knowledge_graph` |
| 生成合并建议 | `rag.agenerate_merge_suggestions` | `graph_service.generate_merge_suggestions` |
| 合并节点 | `rag.amerge_nodes` | `graph_service._execute_merge_operation` |
| 导出 KG 评测数据 | `rag.export_for_kg_eval` | `graph_service.export_for_kg_eval` |

**8 个稳定的业务动作**。这就是"图索引服务"的天然外表面（见 §5.2）。

---

## 3. Code review：顺手发现的问题

按优先级排序。R1/R2/R3 建议纳入 M1 小 PR；其余分到 M2/M3 做。

### R1. `_graph_search` 缺 `finalize_storages()` — 资源泄漏 🔴

`aperag/service/search_pipeline_service.py` 约 265-273 行：

```python
rag = await lightrag_manager.create_lightrag_instance(collection)
param = QueryParam(mode="hybrid", only_need_context=True, top_k=top_k)
context = await rag.aquery_context(query=query, param=param)
if not context:
    return []
return [DocumentWithScore(text=context, metadata={"recall_type": "graph_search"})]
```

对照 `graph_service.py` 同类 handler 的 `try/finally` 模式，这里漏了
`finalize_storages()`。每次图检索查询都会：

- 构造 5 个子 storage 对象（`text_chunks_kv` / `llm_cache_kv` /
  `entities_vdb` / `relationships_vdb` / `chunks_vdb` / `chunk_entity_relation_graph`），
- 调用它们的 `initialize()`（对 PG 实现是打日志；对 Neo4j / Nebula 会
  触发 `prepare_database` / `prepare_space`），
- **不调用** `finalize()` 直接 GC。

短期影响有限（storage 对象 drop 时会被 GC），但这是**唯一的不对称调用
点**，纳入 M1 修正。

### R2. `LightRAG` dataclass 默认 graph_storage 与 env 不一致 🟡

`aperag/graph/lightrag/lightrag.py` 约 112 行：

```python
graph_storage: str = field(default="Neo4JSyncStorage")
```

而 `envs/env.template` 的默认是 `PGOpsSyncGraphStorage`。`create_lightrag_instance`
显式从 env 读并传入，所以生产路径没问题，但**任何绕开 manager 直接
`LightRAG(...)` 的代码**（少数测试 / 工具）会默认连 Neo4j，出错信息
不直观。

建议：M1 把 dataclass 默认值改成 `PGOpsSyncGraphStorage`，并在 docstring
注明"推荐总是通过 `create_lightrag_instance` 构造"。

### R3. `_configure_storage_backends` 引用已废弃的类名 🟡

`aperag/graph/lightrag_manager.py` 约 329-335 行：

```python
using_pg = any([
    kv_storage in ["PGKVStorage", "PGSyncKVStorage", "PGOpsSyncKVStorage"],
    vector_storage in ["PGVectorStorage", "PGSyncVectorStorage", "PGOpsSyncVectorStorage"],
    graph_storage == "PGGraphStorage",  # <- 此类已不在 STORAGES 注册表里
])
```

`STORAGES` 里只有 `PGOpsSyncGraphStorage` / `PGOpsSyncKVStorage` /
`PGOpsSyncVectorStorage` 三个 PG 实现。`PGKVStorage` / `PGSyncKVStorage`
/ `PGGraphStorage` 等是**历史遗留**，现在的任何配置都不会用到它们，但
这段代码会迷惑未来读代码的人（"是不是还有别的类我没看到？"）。

建议：M1 把分支收窄到 `"PGOpsSyncKVStorage"` / `"PGOpsSyncVectorStorage"`，
graph 分支整个删掉——因为 graph 即便是 PG 实现也不需要额外 env 检查
（KV/Vector 已经触发了）。

### R4. 接口面过大：`BaseGraphStorage` 24 个方法 🟡

24 个方法可以自然分成三层：

| 层 | 方法数 | 举例 | 必要性 |
|---|---|---|---|
| **核心** | 13 | `has_node` / `get_node` / `upsert_node` / `upsert_edge` / `delete_node` / `get_knowledge_graph` / ... | 每个后端**必须**实现 |
| **批量** | 8 | `get_nodes_batch` / `node_degrees_batch` / `edge_degrees_batch` / ... | 有**默认实现**（N 次串行调用），后端为了性能**应该**覆盖 |
| **UI 扩展** | 5 | `get_top_degree_nodes` / `get_node_ids` / `search_node_ids_by_label` / `get_nodes_by_source_ids` / `get_edges_by_source_ids` | 默认返回 `None`，调用方需要自己兜底；PG 实现了、Neo4j/Nebula 没有 |

第三层是隐患：调用方（一般是 export / UI）在某些后端会拿到 `None`
而在另一些会拿到数据，**体验不一致且难以发现**。

建议：M3 做层次划分，把第三层改为**显式**的 `NotImplementedError` 且在
`GraphIndexService` 里做"能力探测"（`supports_top_degree_nodes()`）。

### R5. LightRAG 的 `BaseVectorStorage` 与 ApeRAG 的 `aperag/vectorstore` 未打通 🔵

LightRAG 内部 entity / relation / chunk 向量走它自己的
`BaseVectorStorage`（目前只有 `PGOpsSyncVectorStorage` 实现）。ApeRAG
主向量存储（存文档 chunk）走 [`aperag/vectorstore`](./vector_db_abstraction.md)
抽象层（Qdrant / pgvector）。

**两套向量系统并存**，各自有自己的 tenant 约束、分片策略、升级节奏。
功能上互不影响，但运维视角不太好——"我这个部署里的向量到底在哪几个地方？"
需要翻文档回答。

**不建议本次修**：lightrag 要搬家，改造成本会被浪费。纳入 §7.4 的未来规划。

### R6. PGOps* 实现普遍用 `asyncio.to_thread` 包同步 SQLAlchemy 🔵

每次 `upsert_node` / `get_node` / `has_edge` 等单次调用都通过
`asyncio.to_thread(...)` 切到线程池。N 次这样的调用等于 N 次 thread pool
调度开销。

已有 `node_degrees_batch` 等 batch 方法走的是**应用层组装**，底下还是
串行单次 `to_thread`。

改法：把 `GraphRepositoryMixin` 的核心方法改成 `async` + `AsyncSession`，
去掉 `asyncio.to_thread` 包装。好处明显但工作量不小，且 lightrag 搬家
时这段代码会搬走——所以**不建议本次修**。

### R7. `search_pipeline_service._graph_search` 与 `graph_service` 有重复初始化 🔵

一次"用户发起查询"包含多条子路径（vector / graph / fulltext /
summary），每条路径如果启用 graph 都可能独立 `create_lightrag_instance`。
如果同一请求内多次触发 graph 相关动作，就是 N 次完整的 LightRAG 初始化。

改法：**request-scoped cache**——在 FastAPI 的 per-request context 里
缓存 `rag` 实例，请求结束时统一 finalize。这是干净的并发模式，可以
减少大量 cold-start 开销。

纳入 M2（引入 `GraphIndexService` 时自然一并做掉）。

---

## 4. 未来约束：lightrag 改 web service 形态

用户明确说过未来会做："把 lightrag 改成更贴近 web service 的形态，而不
是现在的内置一个 lightrag 对象"。本节把这件事的约束写清楚，让本次抽象
层不被提前废掉。

### 4.1 未来的部署形态（假设）

```
┌──────────┐   HTTP/gRPC   ┌──────────────────┐
│  ApeRAG  │ ────────────► │  LightRAG svc    │
│          │               │                  │
└──────────┘               │  ┌────────────┐  │
                           │  │ BaseGraphS │──┼──► PG / Neo4j / Nebula
                           │  └────────────┘  │
                           │  ┌────────────┐  │
                           │  │ BaseKVStor │──┼──► PostgreSQL
                           │  └────────────┘  │
                           │  ┌────────────┐  │
                           │  │ BaseVectSt │──┼──► pgvector
                           │  └────────────┘  │
                           └──────────────────┘
```

### 4.2 这意味着什么

- **图存储抽象（`BaseGraphStorage`）的归宿是 lightrag 服务内部**。搬家
  那天，`aperag/graph/lightrag/kg/*` 整个目录会搬走。ApeRAG 侧不再
  直接接触 Neo4j / Nebula / pg-emulated 的 SDK。
- **ApeRAG 侧需要的是"对 lightrag 服务的抽象"**：一个可以切换 embedded
  与 remote 实现的接口。这是 §5.2 说的 Layer B。
- **生命周期简化**：`initialize_storages` / `finalize_storages` 变成
  lightrag 服务内部事——客户端只需要管理 HTTP 连接（这是标准事务）。
- **延迟增加**：原本 in-process 的 `rag.get_knowledge_graph(...)` 变成
  RPC。对频繁调用路径要有客户端缓存（batch / dedup / memoize）。

### 4.3 对本次抽象的启示

**Layer A（图存储后端层）本次不动；Layer B（图索引服务层）本次做。**

- Layer A：`BaseGraphStorage` + 三种实现 + 25 测试套件——让它原样跟 lightrag
  一起走。本次只做 §3 的 R1-R3 清洁工作。
- Layer B：`GraphIndexService` 是 ApeRAG 对 "图索引能力" 的**抽象**。
  今天的实现是 `EmbeddedGraphIndexService`（调 `create_lightrag_instance`
  + `rag.X`）；搬家那天改成 `RemoteGraphIndexService`（HTTP 客户端），
  业务代码一行不用改。

---

## 5. 建议的抽象（两层）

### 5.1 分层思想

| 层 | 接口代号 | 所在位置 | 归宿 |
|---|---|---|---|
| Layer A | `BaseGraphStorage` / `BaseKVStorage` / `BaseVectorStorage` | `aperag/graph/lightrag/base.py`（现有） | **lightrag 搬家时一起走** |
| Layer B | `GraphIndexService`（新） | `aperag/graph/service.py`（新增） | **ApeRAG 自己的资产** |

### 5.2 Layer B 接口草案（核心增量）

```python
# aperag/graph/service.py

from typing import Protocol, Sequence
from aperag.db.models import Collection
from aperag.graph.dto import (
    KnowledgeGraph, GraphLabels, MergeSuggestion, MergedNode,
    IndexDocumentResult, DeleteDocumentResult, GraphContext,
    KGEvalExport,
)


class GraphIndexService(Protocol):
    """ApeRAG's business-facing contract for knowledge-graph operations.

    Any 'graph engine' (today: embedded LightRAG; tomorrow: remote
    LightRAG service; day-after: something else entirely) implements this.
    Business code (`graph_service.py`, search pipeline, collection tasks)
    depends ONLY on this Protocol.
    """

    # ---- write ----
    async def index_document(
        self, collection: Collection, doc_id: str,
        content: str, file_path: str,
    ) -> IndexDocumentResult: ...

    async def delete_document(
        self, collection: Collection, doc_id: str,
    ) -> DeleteDocumentResult: ...

    # ---- read ----
    async def query_context(
        self, collection: Collection, query: str, top_k: int,
    ) -> GraphContext: ...

    async def get_labels(
        self, collection: Collection,
    ) -> GraphLabels: ...

    async def get_knowledge_graph(
        self, collection: Collection,
        label: str | None, max_depth: int, max_nodes: int,
    ) -> KnowledgeGraph: ...

    # ---- curation ----
    async def generate_merge_suggestions(
        self, collection: Collection, top_k: int,
    ) -> Sequence[MergeSuggestion]: ...

    async def merge_nodes(
        self, collection: Collection, source_ids: Sequence[str], target_id: str,
    ) -> MergedNode: ...

    # ---- export ----
    async def export_for_kg_eval(
        self, collection: Collection,
    ) -> KGEvalExport: ...
```

**9 个方法，全 DTO 化**。对应 §2.6 的 8 个稳定业务动作（`index_document`
拆了 insert+process_graph 两步为一步）。

### 5.3 两个实现

**`EmbeddedGraphIndexService`**（今天的默认实现）：

- 内部还是 `await create_lightrag_instance(collection)` + `try/finally finalize_storages()`；
- 但业务代码**不再 import `lightrag_manager`**，只依赖 `GraphIndexService`
  Protocol；
- request-scoped `rag` cache（见 §3.R7）可以封在这里。

**`RemoteGraphIndexService`**（lightrag 搬家时启用）：

- HTTP / gRPC 客户端，单例进程级持有连接池；
- 对接"lightrag 服务"的 OpenAPI（那是 lightrag 搬家那次 PR 的产出，和
  本文档无关）；
- 本地实现不需要 `create_lightrag_instance`，lightrag 服务自己管存储。

**切换方式**：`GRAPH_INDEX_SERVICE=embedded|remote` env。默认 embedded。

### 5.4 Layer A 的保留与清理

**保留**（不动）：

- `BaseGraphStorage` / `BaseKVStorage` / `BaseVectorStorage` 的接口面；
- `STORAGES` 注册表；
- 25 个 `GraphStorageTestSuite` 跨后端等价性测试；
- 三个具体实现（PG / Neo4j / Nebula）及其 connection manager。

**清理**（纳入 M1）：

- R1：`_graph_search` 补 `finalize_storages`。
- R2：`LightRAG` dataclass 默认 `graph_storage` 改成 `PGOpsSyncGraphStorage`。
- R3：`_configure_storage_backends` 删除废弃类名分支。

**延迟**（纳入 M3，按需）：

- R4：`BaseGraphStorage` 接口面分层（核心 / 批量 / UI 扩展）。
- R5：LightRAG vector store 与 `aperag/vectorstore` 的合并。
- R6：`PGOps*` 的 `asyncio.to_thread` 包装去掉。

---

## 6. 后端能力矩阵

| 能力 | pg-emulated (PGOpsSync) | Neo4j (sync driver) | Nebula (nebula3) |
|---|---|---|---|
| **原生图引擎** | ❌ 用 `(src, dst)` 表模拟 | ✅ Cypher | ✅ nGQL |
| **多跳 BFS / 路径查询** | SQL 递归 CTE；深度 3 以上代价陡增 | 原生高效 | 原生高效 |
| **批量 upsert** | ✅（SQL `INSERT ... ON CONFLICT`） | ✅（UNWIND） | ✅（UNWIND + `INSERT VERTEX ... VALUES ...`） |
| **事务** | ✅（PG 标准） | ✅（显式 tx） | 有限（Nebula 仅支持 session-level，无跨语句 tx） |
| **多标签节点** | JSONB array 存 | ✅ | 有限（Nebula tag 机制，多 tag 需多次写） |
| **分区 / 水平扩展** | PG 分区或分库 | 企业版集群 / Neo4j Fabric | 原生分布式（meta+graph+storage 分层） |
| **索引管理** | B-tree / GIN | Label + property index | Tag / edge index |
| **运维复杂度** | ★☆☆☆☆（与主 DB 共享） | ★★★☆☆（单独组件） | ★★★★☆（三组件分布式） |
| **小规模成本** | ~0（复用主 PG） | 中（Neo4j Community 免费但需独立机器） | 中-高 |
| **百万节点性能** | ⚠️ 深度遍历会慢 | ✅ | ✅ |
| **亿节点性能** | ❌（不建议） | 集群模式 ✅ | ✅（原生分布式） |
| **当前使用场景** | ApeRAG-Lite / 私有化默认 | 中等规模生产 | 大规模生产 |
| **备份/恢复** | pg_dump | 原生备份 | 原生备份但流程复杂 |

**部署选型的默认建议**：

- 文档 < 10 万 / collection < 千：pg-emulated。零运维，一个 PG 搞定一切。
- 文档 10 万 ~ 百万：Neo4j Community。单机能扛，Cypher 生态好。
- 文档 百万+：Nebula 或 Neo4j 企业版。上分布式。

以上数字是**数量级级别的 rule of thumb**，不是 benchmark 结论。正式
切换前应跑对应数据量的 pilot。

---

## 7. 路线图

四个里程碑。每个都是独立 PR，独立上线，独立回滚。

### 7.1 M1：清洁工作（小 PR）

- [ ] R1：`_graph_search` 补 `try/finally finalize_storages()`。
- [ ] R2：`LightRAG` dataclass 默认 `graph_storage` 对齐 env（`PGOpsSyncGraphStorage`）。
- [ ] R3：`_configure_storage_backends` 删废弃类名分支。
- [ ] 本文档新增的测试：至少一个**回归测试**覆盖 R1 的 finalize 调用
  （mock `rag.finalize_storages`，断言 `_graph_search` 走 finally 分支）。

**预估**：半天。

### 7.2 M2：引入 `GraphIndexService`（中等 PR）

- [ ] 新增 `aperag/graph/service.py`：`GraphIndexService` Protocol +
  `EmbeddedGraphIndexService` 实现。
- [ ] DTO 一套：`aperag/graph/dto.py`，9 个业务动作对应的请求/响应类型。
  原则对齐 `aperag/vectorstore/dto.py`——frozen dataclass，零后端依赖。
- [ ] 迁移业务层：
  - `graph_service.py` 全部 handler 改为依赖 `GraphIndexService`；
  - `search_pipeline_service._graph_search` 同样；
  - `tasks/collection.py::_delete_lightrag` 同样；
  - `lightrag_manager.process_document_for_celery` / `delete_document_for_celery`
    内部也改用 `GraphIndexService`。
- [ ] Request-scoped `rag` cache（见 §3.R7）—— FastAPI dependency，
  同一请求内多次调 graph 方法时复用 LightRAG 实例。
- [ ] 单元测试：`GraphIndexService` 的契约测试（mock LightRAG），独立
  于真实后端。
- [ ] 集成测试：`EmbeddedGraphIndexService` 配 pg-emulated backend 的
  端到端测试（复用 `tests/integration/graphstorage/` 的测试数据）。

**预估**：3~5 天。

### 7.3 M3：Layer A 清理（小-中 PR，可选）

按需做，只在有明确痛点时启动：

- [ ] R4：`BaseGraphStorage` 接口分层。第三层（UI 扩展，默认返回
  `None`）改为显式 `NotImplementedError` + `GraphIndexService.capabilities()`
  能力探测。
- [ ] R6：`PGOpsSync*` 的 `asyncio.to_thread` 包装拆掉，直接走 `AsyncSession`。
  需配套 benchmark 证明收益。
- [ ] 备选：为 `GraphStorageTestSuite` 增加 "NetworkX baseline oracle"
  的对照模式（已有 `networkx_baseline_storage.py`），让跨后端等价性
  测试的**语义正确性**有参考实现验证。

**预估**：1~2 周（不全做）。

### 7.4 M4：lightrag 改 web service（大 PR，独立项目）

不在本文档范围内的独立工程。本文档给它提供两件礼物：

1. **Layer B 存在**：ApeRAG 侧不用改业务代码，只实现新的
   `RemoteGraphIndexService` + 切 env。
2. **Layer A 规整**：接口 + 25 测试 + 三个实现搬去 lightrag 服务时，
   可以直接抬走不用返工。

lightrag 服务本身的 OpenAPI / 部署方案 / 数据迁移 / 并发模型留给那个
PR 定。

---

## 8. Open questions

需要更多信息或实测才能拍板的事：

### Q1. Entity / relation 向量要不要合并到 `aperag/vectorstore`？

当前：LightRAG 用 `PGOpsSyncVectorStorage` 存 entity / relation / chunk
向量，与 `aperag/vectorstore` 的 Qdrant / pgvector 分片**物理隔离**。

- **合并**：减少一套向量系统。但打破 lightrag 的独立性，搬家时要重新
  设计。
- **不合并**：两套向量系统共存，运维多一个维度。lightrag 搬家更干净。

倾向：**不合并**。lightrag 服务搬出去之后，两套向量分别属于不同服务，
是清晰的。

### Q2. `RemoteGraphIndexService` 的缓存策略？

lightrag 变 service 后，`rag.get_knowledge_graph(label="*", max_nodes=1000)`
这种"UI 展示用"的查询如果每次都走 RPC，体验会卡。

候选方案：

- 客户端侧 LRU（by `(collection_id, label, max_depth, max_nodes)` 键）；
- lightrag 服务端 etag + 304；
- 业务层节流（UI 隔 N 秒才允许重新请求）。

三者互斥的程度不高，但实现位置差别大。M4 定。

### Q3. 合并建议（merge suggestions）的所有权归属？

当前：
- 建议**生成**逻辑在 LightRAG（`rag.agenerate_merge_suggestions`）；
- 建议**存储**在 ApeRAG 主 DB（`graph_merge_suggestion` 表，由
  `async_db_ops` 管理）；
- 建议**审核 UI** 在 ApeRAG 前端。

lightrag 搬家后，"生成"会变成 RPC，"存储"和"审核"仍在 ApeRAG。可能需要
把建议存储也搬到 lightrag 服务，让生成和存储同侧；或保留现状，"生成 →
RPC 返回 → ApeRAG 落库" 的扇出。M4 定。

### Q4. pg-emulated 在多大规模下开始不够用？

目前没有正式的 benchmark。M3 的一个子任务是：用
`GraphStorageTestSuite` 里的 `test_large_batch_operations`（697 行起）
改造成可配置规模的压测脚本，在 10 万 / 100 万 / 1000 万 节点规模下对
三种后端做 p50 / p99 查询延迟对比，作为未来运维手册的选型依据。

---

## 9. 反过度设计：什么时候**不**做这个抽象

与 `vector_db_abstraction.md` §10 一脉相承的安全阀：

- **只用一种图后端且没打算切换** → M1 清洁做完就行，M2/M3 都是过度设计。
- **lightrag 一直内嵌、不打算拆服务** → M2（`GraphIndexService`）只节省
  了业务层的 import 深度；值还是有，但不是紧迫需求。
- **团队<3 人，且图功能不是产品核心** → 维护抽象层的成本超过直接调
  LightRAG 的成本。

触发**做 M2** 的信号（任一即可）：

1. lightrag 拆服务提上日程（即便没开始动手）。
2. 新增一个本文档没讨论的图后端（如 ArangoDB、TigerGraph）的需求出现。
3. 单次查询里 graph 路径的性能成为瓶颈，request-scoped cache 成为必要。

触发**做 M3** 的信号：

1. Layer A 接口面的 "默认返回 None" 陷阱真的在生产环境坑过人。
2. `asyncio.to_thread` 的线程池调度被 profiler 点名成为热点。

在触发信号出现前，**本文档的价值在于把思路写清楚**——不是承诺要做。

---

## 10. 附：与向量抽象层的对照

本文档在结构、命名、设计原则上有意与 [`vector_db_abstraction.md`](./vector_db_abstraction.md)
对齐，便于后来人类比阅读：

| 维度 | 向量抽象（已落地） | 图抽象（本文档） |
|---|---|---|
| 业务动作数 | 5 个主要方法 | 9 个主要方法 |
| DSL 层 | `VectorFilter`（Eq/In/IsEmpty/And/Or/Not） | 暂时不需要（图查询语义更业务化） |
| DTO 层 | `VectorPoint`、`QueryRequest`、`SearchHit`、`TenantRef`、`VectorShape` | `KnowledgeGraph`、`GraphContext`、`MergeSuggestion`、... |
| 后端数 | 2（Qdrant、pgvector） | 3（pg-emulated、Neo4j、Nebula） |
| 抽象层位置 | ApeRAG 本身持有 | **Layer B** 在 ApeRAG；**Layer A** 会随 lightrag 搬家 |
| 一次性做完 | ✅ | ❌（M1 先做清洁，M2 做 Layer B，M3 按需） |

向量抽象能一次到位，是因为"向量后端"是纯基础设施；图抽象分阶段，是
因为 LightRAG 这一大块**未来要搬家**，现在深度重构会白做。这是两个
文档最核心的区别。
