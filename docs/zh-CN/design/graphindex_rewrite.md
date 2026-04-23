# Graph Index 模块重写（v2）

> Status: **v2 已落地 + 归一化/合并能力已回填 + merge suggestion 已迁入独立
> `graph_curation` 模块**。
>
> 历史记录：
>
> 1. 第一次落地：完成 extraction、storage、query 核心路径，删除 LightRAG + 三个 curation 功能。
> 2. 第二次修订：把"简单截断 + 不走 LLM 合并"的激进裁剪**撤回**。
>    用户反馈（原话）："我不同意'upsert_entities 改成累积描述 …
>    纯确定性，不走 LLM'。我认为得走 LLM 做总结，你这样是丢失信息！
>    合并 API 同理，原本的功能至少是能运行的 …"
>    本次修订在 §4 新增 "归一化 + 合并" 章节，基于 LLM 摘要重新实现
>    这两条路径；merge 端点从 410 改回 200。
>
> 相关文档：`[vector_db_abstraction.md](./vector_db_abstraction.md)`。

---

## 1. 本次重写的边界

用户口径：

> 我认为只要能做到根据文档生成图结构，然后按照一定的 schema 存储到图数据库
> 就好了，没必要特别强调复用 lightrag 的一些数据结构。你可以认为这是一次
> 垂直重写+切换。代码需要高内聚低耦合，写的好一点简单一些，我希望未来是
> 0 维护免答疑的。
>
> 我希望能彻底删掉 lightrag 的代码。你可以自己判断哪些值得做、哪些不值得做，
> 也没有必要把 LightRAG 的功能全部搬过去。因为我是想要做一个 Graph Index 层，
> 而不是重写一遍 LightRAG。

落地后的功能集：


| 功能                                       | 状态               | 说明                                     |
| ---------------------------------------- | ---------------- | -------------------------------------- |
| `index_document`（文档→图）                   | ✅ **原生实现**       | 核心写路径                                  |
| `delete_document`（删文档对应的实体/关系/chunks）    | ✅ **原生实现**       | 清理路径                                   |
| `query_context`（RAG 查询→图上下文）             | ✅ **原生实现**       | 被 search pipeline 依赖                   |
| `get_labels`（列实体类型）                      | ✅ **原生实现**       | UI 图浏览器依赖                              |
| `get_knowledge_graph`（拉一张子图）             | ✅ **原生实现**       | UI 图浏览器依赖                              |
| **description 归一化（LLM 摘要）**               | ✅ **原生实现（v2.2）** | 累积多片段 → LLM 总结，详见 §4                   |
| **`merge_entities`（多个 entity 合并成 1 个）**   | ✅ **原生实现（v2.2）** | 走 SQL 结构合并 + LLM 总结 description，详见 §4  |
| `generate_merge_suggestions`（LLM 挖合并候选）  | ✅ **迁出 Graph Index** | 现在归属独立 `graph_curation` workflow，详见 `graph_curation.md` |
| `export_for_kg_eval`（导出评测数据）             | ❌ **删除**         | 管理工具；需要时可以基于 graphindex 表直接 dump        |


**被保留下来的两件事**（归一化 + 合并）是被第二轮评审明确要回的：
没有 LLM 摘要，单纯拼接描述会让高频实体的 description 越长越乱；
单纯在字符上限处截断又会丢信息。v2.2 的实现用 LLM 总结在写后和合并
后各做一次压缩，保证语义不丢。详细设计见 §4。

**边界保持不变：Graph Index 仍然不负责 merge suggestion discovery。**
变化只在于：这条能力已经按第一性原理迁到了独立的
`graph_curation` 模块，而不是继续停留在 `410 Gone`。`export_for_kg_eval`
仍然不回；它是单独的管理工具，不属于 Graph Index 主链。

---

## 2. 存储选型决策：只实现 PostgreSQL

LightRAG v1 原本支持三个图后端：

- `PGOpsSyncGraphStorage`（PG 模拟）
- `Neo4JSyncStorage`
- `NebulaSyncStorage`

**v2 只实现 PostgreSQL**。Neo4j/Nebula 整个代码路径随 LightRAG 一起
被删除，包括 `aperag/db/neo4j_sync_manager.py`、`aperag/db/nebula_sync_manager.py`、
对应的驱动依赖（`neo4j`、`nebula3-python`、`nano-vectordb`）。理由：

1. **用户要"简单 + 0 维护 + 免答疑"**。一个后端活得最干净。
2. PG 已经在 ApeRAG 的主链路里——新增不增加部署组件。
3. 真的有客户需要 Neo4j 时再加；代码路径可以按 `GraphStore`
   Protocol 从零实现一遍，不需要背上历史包袱。

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

## 4. 归一化（description 摘要）+ 合并（merge_entities）

### 4.1 问题陈述

用户反馈：

> upsert_entities 不能简单 concat 再截断 —— 那是在丢信息。merge API
> 同理，不走 LLM 的"合并"会丢语义。我想要的是**改善**代码，不是
> **破坏功能**。

触发这段反馈的上一版方案是纯确定性：
- `upsert_entities`：`description = current || "\n\n" || new_fragment`，超过 4000 字符就截断到词边界 + `"…"`。
- `merge_entities`：SQL 合并，结束；**不调 LLM**。

问题：
- 在高频实体（例如 "张三" 被 30 个 chunk 提到）上，accumulate 会很快撞上 4000 字符的硬上限，随后 **70% 的描述被截断**。
- 合并后的 description 是 N 个碎片堆在一起，既难读又让下游的 retrieval prompt 膨胀。LightRAG 原版在 `force_llm_summary_on_merge` 分支会走 LLM 汇总；我们直接扔掉了这条路径。

本次修订把 LLM 摘要加回来，但**关键分层没动**：存储层依旧不碰 LLM，
压缩决策完全由 service 层负责。

### 4.2 分层职责

```
┌───────────────────────────────────────────────────────┐
│ aperag/graphindex/service.py   GraphIndexService       │
│  • index_document 末尾 sweep oversized → LLM 摘要      │
│  • merge_entities: store.merge_entities 之后按需 LLM 摘要 │
│  • _should_summarize / _summarize / _fallback_truncate  │
└───────────────────────────────────────────────────────┘
              │  rewrite_entity_description / rewrite_relation_description
              ▼
┌───────────────────────────────────────────────────────┐
│ aperag/graphindex/storage/base.py   GraphStore Protocol│
│  • upsert_entities / upsert_relations  — 纯 concat      │
│  • merge_entities       — 纯 SQL 结构合并              │
│  • find_oversized_entities / find_oversized_relations  │
│  • rewrite_entity_description / rewrite_relation_description │
└───────────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────┐
│ aperag/graphindex/storage/postgres.py                  │
│  • ON CONFLICT DO UPDATE: concat + substring-dedup     │
│  • 没有字符 cap、没有 LLM                              │
└───────────────────────────────────────────────────────┘
```

**好处**：存储层可以在没有 LLM stub 的情况下做集成测试（`test_postgres_store.py`）。
Service 层可以用纯 Python stub 测 LLM 决策分支（`test_service.py`）。

### 4.3 累积规则（upsert 路径）

**SQL 语句**：

```sql
description = CASE
  WHEN existing IS NULL OR existing = ''         THEN incoming
  WHEN incoming IS NULL OR incoming = ''         THEN existing
  WHEN position(incoming IN existing) > 0        THEN existing
  ELSE existing || :sep || incoming
END
```

- **`:sep` = `"\n\n"`**，定义在 `aperag.domains.knowledge_graph.graphindex.dto.DESCRIPTION_SEPARATOR`。
- **substring dedup**：同一份 boilerplate 出现在多个 chunk 时不会被写两次。这是第一版最常见的抱怨（"我的 description 在重复自己"）。
- **没有 cap**：上层决定是否摘要、何时摘要。SQL 层的 contract 是"写多少存多少，只做 dedup"。

### 4.4 摘要触发条件（`_should_summarize`）

```python
fragments = description.split("\n\n")
if len(fragments) >= summarize_at_fragments:  # 默认 6
    return True
if len(description) >= max_description_chars:  # 默认 4000
    return True
return False
```

**两个阈值的职责不同**：
- `summarize_at_fragments=6`（默认）：正常触发点。LightRAG 默认 10；
  我们取 6，因为**当 description 已经是 6 段拼凑出来的时候，人眼
  读起来就已经像"N 条流水账"而不是一段连贯描述了**，早一点摘要能
  让 RAG prompt 更紧凑。
- `max_description_chars=4000`（默认）：安全兜底。正常路径下不会
  命中（6 片段 × 平均 400 字符 ≈ 2400）。只有当 `llm=None`（开发模式
  禁用了 LLM）或者一个 chunk 里塞满了同一个实体时会兜到。

### 4.5 摘要实现（`_summarize_description`）

```python
prompt = render_summarization_prompt(
    subject_kind="entity" | "relation",
    subject_label=entity.name | f"{src}→{tgt}",
    fragments=description.split("\n\n"),
    language=config.extraction_language,
    target_chars=config.summary_target_chars,  # 默认 800
)
raw = await self._llm(prompt)
```

prompt（`aperag/graphindex/prompts.py::DESCRIPTION_SUMMARIZATION`）
的核心约束：

1. "每个 fragment 的事实都必须保留" —— 明确反对选择性丢弃；
2. 矛盾点**两份都留**，下游 pipeline 再做冲突标注（不让 LLM 自作主张挑一边）；
3. 不添加原文外的信息；
4. 只输出纯文本，不要 JSON 外壳 —— 避免单独写一个解析分支。

**失败处理**：LLM 抛异常 / 返回空 → `_fallback_truncate`（词边界截断 +
`" … [truncated]"` 标记）。`[truncated]` marker 是可 grep 的，运维需要
的时候可以批量审计哪些行是"降级"写入的。

### 4.6 merge_entities（合并 API）

**两步走**，故意不放在一个 SQL transaction 里：

**第 1 步** — `PostgresGraphStore.merge_entities`（纯 SQL）：

1. `SELECT ... FOR UPDATE` 锁 target + 每个 source；
2. 在 Python 里 dedup 拼接 target.description + 每个 source.description；
3. `SELECT` 出所有涉及 source 的 edge；`DELETE` 掉它们；
4. 在 Python 里做 endpoint rewrite（source_id/target_id → target_id）；自环 drop；**同 key 的 redirected edge 在 Python 里合并（union chunks / max weight / concat description）**，避免 `INSERT ... ON CONFLICT` 在一条语句里撞到两行同 key 触发 PG 的 `CardinalityViolationError`；
5. `UPDATE` target 行（new description + union chunk_ids）；
6. `DELETE` source 行；
7. 返回 `MergeEntitiesResult`（包括 **pre-summary** 的 description）。

**第 2 步** — `GraphIndexService.merge_entities`（LLM 决策）：

```python
result = await store.merge_entities(...)
if self._should_summarize(result.description):
    summary = await self._summarize_description(...)
    await store.rewrite_entity_description(..., summary)
    result = dataclasses.replace(result, description=summary)
return result
```

为什么**分两步、不把 LLM 放进同一个 transaction**：
- LLM 调用延迟高（P95 几秒），不能占着 PG 的行锁；
- 摘要失败时结构合并不应回滚 —— 结构合并的价值独立于描述美化；
- 单元测试可以用 `_StubStore` 直接测 service 决策，不需要真 PG。

### 4.7 配置

`GraphIndexConfig`：

| 字段                                | 默认  | 含义                                       |
| --------------------------------- | --- | ---------------------------------------- |
| `summarize_at_fragments`          | 6   | 达到多少个 `\n\n` 片段就触发 LLM 摘要                |
| `max_description_chars`           | 4000| 硬上限 / 兜底阈值                               |
| `summary_target_chars`            | 800 | 给摘要 prompt 的目标字数（实际会浮动）                  |

约束（`__post_init__`）：
- `summarize_at_fragments >= 2`；
- `summary_target_chars < max_description_chars`（否则刚摘要完又会触发截断）。

### 4.8 测试覆盖

- `test_dto.py`：`MergeEntitiesResult` 字段锁定、`DESCRIPTION_SEPARATOR == "\n\n"`。
- `test_postgres_store.py`（gated on `APERAG_TEST_GRAPHINDEX_PG_URL`）：
  - `test_upsert_entity_accumulates_descriptions`：多次 upsert 拼接 N 段；
  - `test_upsert_entity_dedupes_identical_fragments`：相同片段不重复写入；
  - `test_find_oversized_entities_returns_rows_past_threshold`：按 char / fragment 阈值查询；
  - `test_rewrite_entity_description_replaces_in_place`：整段替换；
  - `test_merge_entities_redirects_edges_and_unions_chunks`：结构合并 + edge redirect + 自环 drop + 重复 edge collapse；
  - `test_merge_entities_missing_target_raises`。
- `test_service.py`：
  - `test_index_document_summarizes_oversized_entities_via_llm`：**保证走 LLM，不是截断**；
  - `test_summarization_falls_back_to_truncation_only_when_llm_fails`：LLM 故障降级；
  - `test_index_document_skips_summary_when_no_oversized_rows`：happy path 不付出 LLM 成本；
  - `test_merge_entities_summarizes_merged_description`：合并后必定调 LLM + 持久化；
  - `test_merge_entities_skips_summary_on_short_description`：小合并不浪费 LLM call。

---

## 5. 模块结构

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
from aperag.domains.knowledge_graph.graphindex import GraphIndexService

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

- `has_node` / `has_edge`：`upsert_`* 是幂等的，调用方不需要先 exist-check；
- `node_degree` / `edge_degree` + 所有 `*_batch` 变体：读路径不用；
- `get_nodes_by_source_ids` / `get_top_degree_nodes`：UI 查 label 用
`list_subgraph` 即可；
- `remove_nodes` / `remove_edges`：用 `delete_document_rows` 按 doc 删，
不按 node 删。

**设计原则**：每个抽象方法都有实际调用方；没有调用方 == 不该存在。

---

## 6. 数据模型

### 5.1 新表（不碰 `lightrag_graph_`*）

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

**三张表**（vs v1 的 `lightrag_graph_nodes` / `lightrag_graph_edges` +
按 namespace 拆分的 KV / vector 约 8 张表）。旧表在这个 PR 里被一次性
drop（见 §6）。

### 5.2 实体向量：复用 `aperag/vectorstore`

v2 **不在 graphindex 内部管理实体向量**。实体的向量落在
`aperag/vectorstore/pgvector` 的 `aperag_vectors_<dim>_cosine` 里，通过
`VectorPoint.payload = {"kind": "entity", "collection_id": ..., "entity_id": ...}`
打标。检索时用 `VectorStoreConnector.search(QueryRequest(..., flt=...))` 加
DSL 过滤。

这样**向量抽象只有一套**，运维看到的就是统一的 `aperag_vectors_*` 集合。

---

## 7. 数据切换策略

- **旧表彻底删除**。新 Alembic 迁移 `f1e2d3c4b5a6` 一次性 drop：
  - `lightrag_graph_nodes` / `lightrag_graph_edges`
  - `lightrag_doc_chunks` / `lightrag_vdb_entity` / `lightrag_vdb_relation`
  - `graph_index_merge_suggestions` / `graph_index_merge_suggestions_history`
- **不做数据迁移**。用户口径是"切换"——每个 collection 按需 re-index 到
  `graphindex_*`，旧 LightRAG 表里的内容直接丢弃。
- **没有 fallback、没有 cutover 标记表**。上一版文档描述的 "双后端回退 +
  `graphindex_collection_state`" 方案被删除：无 LightRAG 可回退的情况下，
  这套机制是纯维护负担。一个新建 collection 在首次 `index_document`
  之前 `get_labels` 就会返回空 list，这是正确行为，不是 bug。

---

## 8. 业务层切换点

所有 5 处 graph truth 调用都改指 `aperag/graphindex`；merge suggestion
改由独立的 `graph_curation` 模块接管；只剩 `kg-eval` 导出路由保留 `410`。


| 位置                                                         | 本 PR 改动                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| `aperag/service/graph_service.py::get_graph_labels`        | → `graphindex.get_labels`                                                |
| `aperag/service/graph_service.py::get_knowledge_graph`     | → `graphindex.get_knowledge_graph`                                       |
| `aperag/service/search_pipeline_service.py::_graph_search` | → `graphindex.query_context`                                             |
| `aperag/tasks/collection.py::_delete_knowledge_graph_data` | → `run_drop_collection_sync`                                             |
| `aperag/tasks/document.py`（3 处 celery 调用）                   | → `run_index_document_sync` / `run_delete_document_sync`                 |
| `aperag/service/prompt_template_service.py::hardcoded[graph]` | → `graphindex.prompts.ENTITY_RELATION_EXTRACTION`                        |
| `aperag/views/graph.py::merge_nodes_view`                    | **200**，委托 `graph_service.merge_entities` → `GraphIndexService.merge_entities` |
| `aperag/views/graph.py::merge_suggestions*`                 | → `graph_curation_service.start_run / get_latest / handle_action`                    |
| `aperag/views/graph.py::export_kg_eval_view`               | **410 Gone**（管理工具，本次范围外）                                                  |


---

## 9. 代码质量约束

全文档贴一条（避免未来再问为什么这样写）：

- **禁止 from aperag.domains.knowledge_graph.graphindex.engine import ...**：业务层只准 `from
  aperag.domains.knowledge_graph.graphindex import GraphIndexService`（和 DTO）。engine 是内部。
- **禁止 models.py 被外部导入**：SQLAlchemy 模型只给 `storage/postgres.py`
  和 alembic 用。
- **每个 public 方法有 docstring**，解释：做什么、幂等性、线程安全性、
  生命周期。
- **每个 module 顶部有 module docstring**，解释本文件的边界和不负责什么。
- **测试覆盖**：每个 public 方法 **至少 1 个** unit test +（gated by env
  的）1 个 integration test。

---

## 10. 不做什么

- 不做 `BaseGraphStorage` 那种 24 方法的胖接口；
- 不做 "gleaning"（LightRAG 的多轮提取）—— 一次 LLM 调用足够，多轮复杂度
  不值这么点召回率；
- 不做 incremental merge（LLM 判断两个 entity 该不该合并）—— 不属于
  Graph Index 层的责任；
- 不做 workspace 嵌套—— 直接用 `collection_id` 作为分区键；
- 不做 KV storage / doc status storage —— 这些是 LightRAG 内部实现细节，
  v2 的流程不需要它们；
- 不做 v1 ↔ v2 数据迁移工具 —— 切换是硬切换，用户按需 re-index。
