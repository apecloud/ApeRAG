# Graph Index 模块重写（v2）

> Status: **v2 已落地**。本文档与 [`lightrag_refactor.md`](./lightrag_refactor.md)
> 的 Phase 1 共同交付。v2 模块在 `aperag/graphindex/`；旧的 `aperag/graph/`
> 目录仍保留在树里（作为 fallback 路径 + 少量 curation 功能的临时 host），
> 预计下一个 PR 完整删除。
>
> 相关文档：[`lightrag_refactor.md`](./lightrag_refactor.md)、
> [`graph_db_abstraction.md`](./graph_db_abstraction.md)、
> [`vector_db_abstraction.md`](./vector_db_abstraction.md)。

---

## 1. 本次重写的边界

用户口径：
> 我认为只要能做到根据文档生成图结构，然后按照一定的 schema 存储到图数据库
> 就好了，没必要特别强调复用 lightrag 的一些数据结构。你可以认为这是一次
> 垂直重写+切换。代码需要高内聚低耦合，写的好一点简单一些，我希望未来是
> 0 维护免答疑的。

按这个口径，v2 **只保留 5 个核心能力**，其它特性按重要度分类处置：

| 功能 | v2 状态 | 理由 |
|---|---|---|
| `index_document`（文档→图） | ✅ **原生实现** | 核心写路径 |
| `delete_document`（删文档对应的实体/关系/chunks） | ✅ **原生实现** | 清理路径 |
| `query_context`（RAG 查询→图上下文） | ✅ **原生实现** | 被 search pipeline 依赖 |
| `get_labels`（列实体类型） | ✅ **原生实现** | UI 图浏览器依赖 |
| `get_knowledge_graph`（拉一张子图） | ✅ **原生实现** | UI 图浏览器依赖 |
| `generate_merge_suggestions`（LLM 生成合并建议） | ⏸ **v1 继续跑** | 重策展特性，下次 PR 再搬 |
| `merge_nodes`（执行合并） | ⏸ **v1 继续跑** | 同上 |
| `export_for_kg_eval`（导出评测数据） | ⏸ **v1 继续跑** | 管理工具，低频 |

**三个 curation 功能本 PR 不搬**，因为：

1. 它们是 LLM 密集型的策展特性（merge suggestions 用 LLM 比较两个实体；
   export 用 LLM 做摘要），搬过来会让本 PR 的 LLM 部分翻倍，增加 review 风险。
2. 它们目前在生产用得不多，慢一轮替换不会影响核心链路。
3. 本次口径是"0 维护免答疑"——大就是反 0 维护，所以先砍。

下一个 PR 会完整搬掉这三个，届时 `aperag/graph/` 整个目录删除。

---

## 2. 存储选型决策：只实现 PostgreSQL

LightRAG v1 原本支持三个图后端：
- `PGOpsSyncGraphStorage`（PG 模拟）
- `Neo4JSyncStorage`
- `NebulaSyncStorage`

**v2 只实现 PostgreSQL**，Neo4j/Nebula 留接口不实现（抛
`NotImplementedError` 带清晰消息）。理由：

1. **用户要"简单 + 0 维护"**。一个后端活得最干净。
2. PG 已经在 ApeRAG 的主链路里——新增不增加部署组件。
3. Neo4j/Nebula 的接口形状在 v1 已经被 `BaseGraphStorage` 定义得差不多了，
   未来要加只是按 v2 的 `GraphStore` Protocol 实现一遍，不需要额外设计。
4. 真的有客户需要 Neo4j 时再加；当下没有生产用户使用这两个后端。

---

## 3. LLM 提取：JSON 而不是 tuple-delimited

LightRAG v1 用自定义的 tuple-delimited 格式：

```
("entity"<|>Alex<|>person<|>description)##
("relationship"<|>Alex<|>Bob<|>reason<|>keywords<|>5)##
<|COMPLETE|>
```

好处是**轻量**；坏处是**LLM 经常输出格式错位**（少个分隔符、缺一段等），
解析代码要容错大量 case。

**v2 用 JSON 输出**：

```json
{
  "entities": [
    {"name": "Alex", "type": "person", "description": "..."}
  ],
  "relations": [
    {"source": "Alex", "target": "Bob", "description": "...", "weight": 5}
  ]
}
```

理由：

1. **现代 LLM API 原生支持 `response_format={"type":"json_object"}`**，结构
   保证由 provider 端做（OpenAI、DeepSeek、通义、文心等主流都支持）；
2. JSON 解析一行代码 `json.loads(...)`，不需要自定义解析器；
3. 格式错误的恢复代码从 "50 行正则 + 状态机" 降到 "try/except 抛掉坏 chunk"；
4. 实体/关系的字段**明确有类型**（weight 是 int，不是字符串），不需要 post-parse 强转。

代价是输出 token 数略多（每个实体多几个 `"name":` 这类 JSON key），但
在写路径不是热点，LLM 调用成本也没显著变化。

---

## 4. 模块结构

```
aperag/graphindex/
├── __init__.py          # 仅 re-export 公共符号
├── service.py           # GraphIndexService：5 个 async 方法 + Celery 同步包装
├── dto.py               # 所有 DTO 集中定义（Entity/Relation/Chunk/...）
├── config.py            # GraphIndexConfig：注入式，不读 env
├── prompts.py           # 独立 LLM prompt；JSON 输出；一套即可
├── engine/
│   ├── __init__.py
│   ├── chunking.py      # 文档切分：简单按 token 窗口+重叠
│   ├── extraction.py    # 单次 LLM 调用：chunk → {entities, relations}
│   └── indexer.py       # 协调：chunks → extractions → persist
├── storage/
│   ├── __init__.py
│   ├── base.py          # GraphStore Protocol（~10 方法，全 DTO 化）
│   └── postgres.py      # PostgresGraphStore：唯一可用实现
└── models.py            # SQLAlchemy 模型：graphindex_nodes / _edges / _chunks
```

**文件数：~11**。每个文件职责单一；`service.py` 是唯一外部 import 入口。

### 4.1 `service.py` 的契约

```python
from aperag.graphindex import GraphIndexService

svc = GraphIndexService.from_config(config)  # 单例也行、每次 new 也行

# 写
await svc.index_document(collection_id, doc_id, content, file_path)
# 删
await svc.delete_document(collection_id, doc_id)
# 查（RAG 用）
ctx = await svc.query_context(collection_id, query, top_k=10)
# UI
labels = await svc.get_labels(collection_id)
kg = await svc.get_knowledge_graph(collection_id, label="*", max_depth=2, max_nodes=500)
```

**没有第 6 个方法**。策展功能暂留 v1。

### 4.2 `GraphStore` Protocol（`storage/base.py`）

10 个方法，覆盖 v2 全部需要：

```python
class GraphStore(Protocol):
    # collection lifecycle
    async def ensure_schema(self) -> None: ...           # DDL 幂等
    async def drop_collection(self, collection_id: str) -> None: ...

    # write
    async def upsert_chunks(self, collection_id: str, chunks: Sequence[Chunk]) -> None: ...
    async def upsert_entities(self, collection_id: str, entities: Sequence[Entity]) -> None: ...
    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None: ...

    # delete
    async def delete_document_rows(self, collection_id: str, doc_id: str) -> DeleteDocumentResult: ...

    # read
    async def find_entities_by_names(self, collection_id: str, names: Sequence[str]) -> list[Entity]: ...
    async def find_entities_near(
        self, collection_id: str, anchor_ids: Sequence[str], max_hop: int, limit: int
    ) -> tuple[list[Entity], list[Relation]]: ...
    async def list_labels(self, collection_id: str) -> list[str]: ...
    async def list_subgraph(
        self, collection_id: str, label: str | None, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph: ...
```

**比 LightRAG 的 `BaseGraphStorage`（24 方法）小一半**，因为 v2 不需要：

- `has_node` / `has_edge`：`upsert_*` 是幂等的，调用方不需要先 exist-check；
- `node_degree` / `edge_degree` + 所有 `*_batch` 变体：读路径不用；
- `get_nodes_by_source_ids` / `get_top_degree_nodes`：UI 查 label 用
  `list_subgraph` 即可；
- `remove_nodes` / `remove_edges`：用 `delete_document_rows` 按 doc 删，
  不按 node 删。

**设计原则**：每个抽象方法都有实际调用方；没有调用方 == 不该存在。

---

## 5. 数据模型

### 5.1 新表（不碰 `lightrag_graph_*`）

```sql
CREATE TABLE graphindex_nodes (
    id              BIGSERIAL PRIMARY KEY,
    collection_id   TEXT NOT NULL,
    entity_id       TEXT NOT NULL,  -- hash(collection_id + normalized_name)
    name            TEXT NOT NULL,
    type            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    source_chunks   TEXT[] NOT NULL DEFAULT '{}',  -- chunk_id list
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (collection_id, entity_id)
);
CREATE INDEX graphindex_nodes_cid_type ON graphindex_nodes (collection_id, type);
CREATE INDEX graphindex_nodes_cid_name ON graphindex_nodes (collection_id, name);
-- GIN on source_chunks for "find nodes touching this chunk" queries:
CREATE INDEX graphindex_nodes_source_chunks ON graphindex_nodes USING GIN (source_chunks);

CREATE TABLE graphindex_edges (
    id              BIGSERIAL PRIMARY KEY,
    collection_id   TEXT NOT NULL,
    source_id       TEXT NOT NULL,   -- entity_id
    target_id       TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    weight          NUMERIC(6,3) NOT NULL DEFAULT 0,
    source_chunks   TEXT[] NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (collection_id, source_id, target_id)
);
CREATE INDEX graphindex_edges_cid_src ON graphindex_edges (collection_id, source_id);
CREATE INDEX graphindex_edges_cid_tgt ON graphindex_edges (collection_id, target_id);

CREATE TABLE graphindex_chunks (
    id              BIGSERIAL PRIMARY KEY,
    collection_id   TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,       -- stable UUID
    doc_id          TEXT NOT NULL,
    order_in_doc    INTEGER NOT NULL,
    text            TEXT NOT NULL,
    file_path       TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (collection_id, chunk_id)
);
CREATE INDEX graphindex_chunks_cid_doc ON graphindex_chunks (collection_id, doc_id);
```

**三张表**（vs v1 的 `lightrag_graph_nodes` / `lightrag_graph_edges` + KV
+ vector_storage namespace 分表 = 约 8 张表 / namespace）。

### 5.2 实体向量：复用 `aperag/vectorstore`

v2 **不在 graphindex 内部管理实体向量**。实体的向量落在
`aperag/vectorstore/pgvector` 的 `aperag_vectors_<dim>_cosine` 里，通过
`VectorPoint.payload = {"kind": "entity", "collection_id": ..., "entity_id": ...}`
打标。检索时用 `VectorStoreConnector.search(QueryRequest(..., flt=...))` 加
DSL 过滤。

这样**向量抽象只有一套**，运维看到的就是统一的
`aperag_vectors_*` 集合。

---

## 6. 切换策略（数据层面）

- **旧表 `lightrag_graph_*` 保留不动**。业务层切换到 v2 之后，新写入都走
  `graphindex_*`；旧数据"冷冻"在 lightrag 表里。
- **用户手动触发 re-index** 才能把旧数据搬到新表。不做自动迁移——自动迁移
  意味着要跑一次全量 LLM 抽取，成本不可忽视，应由用户明确触发。
- **Curation 特性暂时仍访问旧表**（LightRAG 依然在用 `lightrag_graph_*`）。
  等下次 PR 把 curation 也搬到 v2，再一次性 drop 旧表。

**如果用户在本 PR 合并后想立刻清理旧数据**：执行一次 `re-index` 全部
collection 并手动 `DROP TABLE lightrag_graph_nodes, lightrag_graph_edges
CASCADE`。不属于本 PR 范畴。

---

## 7. 业务层切换点

v2 改 5 处 import，原样保留 3 处（curation）：

| 位置 | 本 PR 改动 |
|---|---|
| `aperag/service/graph_service.py::get_graph_labels` | → `graph_index_service.get_labels` |
| `aperag/service/graph_service.py::get_knowledge_graph` | → `graph_index_service.get_knowledge_graph` |
| `aperag/service/graph_service.py::generate_merge_suggestions` | **不动**（仍用 LightRAG） |
| `aperag/service/graph_service.py::_execute_merge_operation` | **不动**（仍用 LightRAG） |
| `aperag/service/graph_service.py::export_for_kg_eval` | **不动**（仍用 LightRAG） |
| `aperag/service/search_pipeline_service.py::_graph_search` | → `graph_index_service.query_context`；顺手修掉 finalize 泄漏 bug |
| `aperag/tasks/collection.py::_delete_lightrag` | → `graph_index_service.delete_collection_data`（遍历删 doc） |
| `aperag/tasks/document.py` 的 3 处 celery 调用 | → `graph_index_service.index_document_sync` / `delete_document_sync` |

---

## 8. 代码质量约束

全文档贴一条（避免未来再问为什么这样写）：

- **禁止 from aperag.graphindex.engine import ...**：业务层只准 `from
  aperag.graphindex import GraphIndexService`（和 DTO）。engine 是内部。
- **禁止 models.py 被外部导入**：SQLAlchemy 模型只给 `storage/postgres.py`
  和 alembic 用。
- **每个 public 方法有 docstring**，解释：做什么、幂等性、线程安全性、
  生命周期。
- **每个 module 顶部有 module docstring**，解释本文件的边界和不负责什么。
- **测试覆盖：**每个 public 方法 **至少 1 个** unit test + （gated by env
  的）1 个 integration test。

---

## 9. 未来工作（不在本 PR）

- [ ] Phase 2 PR：搬 curation（merge_suggestions / merge_nodes / KG export），
      删 `aperag/graph/lightrag/` 目录 + `lightrag_manager.py`。
- [ ] Phase 3 PR：`GraphStore.neo4j` / `GraphStore.nebula` 实现，如果有
      真实客户需求。
- [ ] Phase 4 PR：数据迁移工具（老 lightrag 表 → 新 graphindex 表）。当前
      切换用 re-index，未来可以做零停机迁移。

---

## 10. 不做什么

- 不做 `BaseGraphStorage` 那种 24 方法的胖接口；
- 不做 "gleaning"（LightRAG 的多轮提取）—— 一次 LLM 调用足够，多轮复杂度
  不值这么点召回率；
- 不做 incremental merge（LLM 判断两个 entity 该不该合并）—— 留在 curation；
- 不做 workspace 嵌套—— 直接用 `collection_id` 作为分区键；
- 不做 KV storage / doc status storage —— 这些是 LightRAG 内部实现细节，
  v2 的流程不需要它们。
