---
title: task #32 — MCP 接口审计 spec v1
description: ApeRAG 对外 MCP 检索接口现状 inventory + 缺口识别 + 接口改造方向 + rerank 删除 fold-in
---

# task #32 — MCP 接口审计 spec v1

> earayu2 directive (`#indexing优化` msg=c9c7cf31 / msg=42b74170 / msg=70cb0f6b)：Graph 形态变化（task #17 hard cut 拆分部署 + Wave 5 graph_facts/graph_vectors 三层 + description 不再生成）后，重审所有 MCP 接口的组合 / 互补关系；ApeRAG 没有大一统 search 接口，每个索引各自暴露 MCP tool，需确认链路完整 + 参数 / 行为 / 输出合理。

## 1. 现状 inventory（grep 实证）

### 1.1 MCP tool 全集（15 个，分 5 类）

| 分类 | Tool | 入参 | 输出 | 后端 |
| --- | --- | --- | --- | --- |
| **chunk-level 检索** | `search_vector` | `collection_id`, `query`, `top_k=5`, `similarity_threshold=None`, **`rerank=True`** | `SearchResult` (items, recall_type=`vector_search`, `chunk_id` 暴露) | `POST /api/v2/collections/{id}/searches` |
| | `search_fulltext` | `collection_id`, `query`, `top_k=5`, `keywords=None`, **`rerank=True`** | `SearchResult` (recall_type=`fulltext_search`, `chunk_id` 暴露) | 同上 |
| | `search_graph` | `collection_id`, `query`, `top_k=5` | `SearchResult` (recall_type=`graph_search`, `chunk_id` 暴露) | 同上 |
| **graph 元素操作** | `query_graph_entities` | `collection_id`, `query`, `top_k=10` | `{entities: [{name, entity_type, description, source_chunk_count}]}` | `GET /api/v2/collections/{id}/graphs/entities/search` |
| | `expand_graph_subgraph` | `collection_id`, `entity_names`, `hops=1` | `{entities: [...], relations: [{source, target, description}]}` | `POST /api/v2/collections/{id}/graphs/subgraph/expand` |
| | `get_entity_detail` | `collection_id`, `name` | `{name, entity_type, description, source_chunk_count}` 或 `{error}` | `GET /api/v2/collections/{id}/graphs/entities/{name}` |
| **文档读取** | `read_document_chunk` | `collection_id`, `document_id`, `chunk_id` | `DocumentChunk` (parsed_markdown / section_path / chunk_index / parse_version) | DB direct |
| | `read_document` | `collection_id`, `document_id`, `range=None` | `DocumentContent` (parsed_markdown / parse_version / truncated) | DB direct |
| | `read_document_outline` | `collection_id`, `document_id`, `max_depth=6` | `DocumentOutline` (headings tree / section_path / heading_anchor) | DB direct |
| | `read_document_section` | `collection_id`, `document_id`, `section_path \| heading_anchor` | `DocumentSection` | DB direct |
| **元数据** | `get_document_metadata` | `collection_id`, `document_id` | `DocumentMetadata` (indexed_chunks_count / indexing_status / title) | DB direct |
| | `list_documents` | `collection_id`, `cursor`, `limit=50`, `sort_by`, `sort_order`, `title_filter`, `type_filter`, `indexed_only` | `DocumentList` (items / next_cursor / total_count) | DB direct |
| | `get_collection_metadata` | `collection_id` | `CollectionDetailMetadata` (index_modes_available / document_count) | DB direct |
| | `list_collections` | `cursor`, `limit=50`, `sort_by`, `sort_order`, `title_filter` | `CollectionList` | DB direct |
| **网络** | `web_search` | `query`, `top_k=5`, `timeout=30`, `locale="en-US"`, `source=None` | `WebSearchResponse` | `POST /api/v2/web/search` |

### 1.2 entity → chunk → doc 链路 verify

正向链路（**chunk-level search → 文档读取**）✓：
- `search_vector` / `search_fulltext` / `search_graph` 输出 `SearchResultMetadata.chunk_id` (`schemas.py:101-112`) → `read_document_chunk(collection_id, document_id, chunk_id)` ✓
- `recall_type` 字段区分召回路径（`vector_search` / `fulltext_search` / `graph_search`）

反向链路（**graph 元素 → 原始 chunk**）✗ — 这是当前最严重的缺口：
- `query_graph_entities()` 输出 `{name, entity_type, description, source_chunk_count}` (`graph_tools.py:97-98`)，**不返回 chunk_id / document_id**
- `expand_graph_subgraph()` 输出 `{entities, relations}`(`graph_tools.py:153-154`)，同样**不返回 evidence chunk_id**
- agent 拿到 entity name 后，必须再走一次 `search_vector` / `search_graph` 才能拿到 chunk_id 链路 — 重复召回 + cost double

### 1.3 互补 / 重叠 matrix

| Tool 对 | 关系 | 说明 |
| --- | --- | --- |
| `search_vector` ↔ `search_fulltext` | **完全重叠**（功能正交） | 同 chunk 级召回，模式不同（语义 vs 关键词） |
| `search_vector` ↔ `search_graph` | **互补**（chain-friendly） | vector 给 content evidence，graph 给 entity 关系，并列召回 + agent 融合 |
| `search_graph` ↔ `query_graph_entities` | **粒度重叠** | search_graph 是 chunk-level（含 graph evidence 的 chunks），query_graph_entities 是 entity-level（语义搜索 entity 名） |
| `query_graph_entities` → `expand_graph_subgraph` | **自然 chain** | query 拿 entity names → expand 喂 entity_names 列表（`graph_tools.py:125-169`） |
| `expand_graph_subgraph` → `read_document_chunk` | **断链** | expand 输出无 chunk_id → agent 需另调 search 才能跳到 chunk 层 |
| `list_documents` → `get_document_metadata` | **替代**（粗 vs 细） | list 已含 metadata 摘要；get_document_metadata 提供 indexed_chunks_count / indexing_status 详情 |
| `read_document` ↔ `read_document_section` ↔ `read_document_chunk` | **分层完整** | 全文 / section / chunk 三粒度，`read_document_outline` 是导航入口 |

## 2. 缺口识别（按 severity 排序）

### 2.1 BLOCKER：graph 元素 → chunk_id 断链

`query_graph_entities` 和 `expand_graph_subgraph` 输出无 `chunk_id` / `document_id` 列表，导致 agent 无法直接跳到 chunk 读取。

**影响**：earayu2 directive msg=c9c7cf31 明确「entity → chunk_id → doc_id 完整链路」是 MCP 接口必须 cover 的核心场景。当前断链强制 agent 走「graph 拿 entity → 再 vector_search 拿 chunk → 再 read_document_chunk」三跳，每跳都重复 LLM / vector 调用。

**根因**：当前 `aperag/mcp/tools/graph_tools.py` 只 forward backend graph entity / subgraph endpoint，backend response schema 本身不返回 evidence chunk_ids（Wave 7 / task #17 阶段未 explicit 设计）。

### 2.2 P1：rerank 残留（fold-in task #35 决策）

`search_vector.py:64` + `search_fulltext.py:62` 的 `rerank: bool = True` 默认参数 + `SearchRequest.rerank` schema (`retrieval/schemas.py:287`)，跟 task #35 「彻底删除 rerank」directive (earayu2 msg=a81bc213) 直接冲突。task #35 PR 链已开（#1898 ziang MCP + #1899 Bryce BE core + #1897 dongdong UI），**task #32 spec 落地时这两个参数应已删除**，但 spec 仍需把「MCP 不暴露 rerank」作为 invariant 明文锁住，防 future re-introduction。

### 2.3 P1：search_graph 与 query_graph_entities 粒度边界混淆

- `search_graph` 是 **chunk-level**（返回含 graph evidence 的 chunks）
- `query_graph_entities` 是 **entity-level**（返回 entity 名 + metadata）
- 两个 tool 名都含 "graph" 但语义粒度不同；agent / docs 容易混用

**影响**：agent 调用选择成本高 + MCP 文档需要明确两者职责差异。

### 2.4 P2：输出 schema 异构（SearchResult vs dict）

- `search_*` 三类返回 `SearchResult`（统一 items + metadata 容器）
- `query_graph_entities` / `expand_graph_subgraph` / `get_entity_detail` 返回 raw dict（容器结构异构）
- `list_documents` / `list_collections` 返回 List 容器（含 next_cursor 分页）

**影响**：agent 对每个 tool 需要定制 parse 逻辑 — 增加 prompt 工作量 + LLM 解析失败率。

### 2.5 P2：参数 / 默认值不一致

- `search_vector` 暴露 `similarity_threshold`，`search_fulltext` / `search_graph` 不暴露
- `expand_graph_subgraph` 文档写「max 3 hops」但参数无 schema validation
- `top_k` 在 search 类是 `int=5`，在 `query_graph_entities` 是 `int=10`，无统一基线

### 2.6 P3：后端 fine-grained 能力未暴露

- backend `aperag/domains/retrieval/pipeline.py` 有 `_apply_ranking_strategy` / `_apply_fallback_strategy` 等内部机制
- backend `graph_search_service` 支持 keyword / vector / traversal 多模式
- 当前 MCP 只暴露粗粒度 unified search，agent 无法精确控制召回策略

**判断**：P3 不优先解决 — 当前 agent 主要诉求是「完整链路 + 互补组合」，细粒度能力是 future feature（task #33 测试审计后再决定）。

## 3. 接口改造方向（task #32 主线）

### 3.1 必须做（Hard scope）

#### 3.1.1 补 graph element → chunk → doc 链路（解 §2.1 BLOCKER）

**关键 fix**（per Weston msg=7500e57d）：`read_document_chunk(collection_id, document_id, chunk_id)` 是 document-scoped，chunk_id **不保证全局唯一**，所以只暴露 chunk_id 仍然断链 — agent 必须额外查 document_id 才能读 chunk。spec 早期版本写「`evidence_chunk_ids: list[str]`」不够，修订为暴露 **轻量 evidence ref** 含 `(document_id, chunk_id, parse_version)`：

```json
// 改造后 query_graph_entities 输出
{
  "entities": [
    {
      "name": "...",
      "entity_type": "...",
      "description": "...",
      "source_chunk_count": 12,
      "evidence_refs": [
        {"document_id": "doc_abc", "chunk_id": "chunk_abc", "parse_version": "v1"},
        {"document_id": "doc_def", "chunk_id": "chunk_def", "parse_version": "v1"}
      ]
    }
  ]
}
```

`parse_version` 推荐带（防 chunk 跨 parse_version 歧义）；`evidence_refs` 默认上限 10（payload 控制 + agent reasoning 优先级），仍保留 `source_chunk_count` / `total_source_chunks` 字段告知总数。

**A1 scope 覆盖三处**（per Weston msg=7500e57d，避免漏 endpoint）：
- `query_graph_entities`：entity refs
- `expand_graph_subgraph`：entity refs + relation refs（双侧都暴露）
- `get_entity_detail`：entity refs（或 spec 明确说明为何 defer）

backend 实施口径：lineage / source_chunk 数据已存在（Wave 5 GraphFactsWorker 写入 `source_chunk_ids` list per entity / relation），endpoint / service 需要 **collect / project 到 response**（不是已经 join 好），新增 lightweight DTO `GraphEvidenceRef`（已经有 `GraphEvidenceChunk` 类似模型可参考）。

#### 3.1.2 锁 MCP 不暴露 rerank（fold-in task #35）

task #35 PR 链 merge 后：
- `search_vector` / `search_fulltext` 函数签名 **不再有 `rerank` 参数**
- `SearchRequest.rerank` schema 字段已删（PR #1899 已 cover）
- spec 增列 invariant：MCP 接口面**不允许暴露 rerank 参数**（boundary test 钉死）— 防 future PR 误加回

#### 3.1.3 文档明确 search_graph vs query_graph_entities 粒度差异（解 §2.3）

不改代码 / API 名 — 只在 MCP tool docstring + 用户文档明确：
- `search_graph`：**chunk-level** 检索，返回含 graph evidence 的 chunks，跟 `search_vector` / `search_fulltext` 对齐
- `query_graph_entities`：**entity-level** 检索，返回 entity 名 + metadata，给 agent 做 graph reasoning 用
- 两者**联合使用**：agent 先 `query_graph_entities` 拿 entity，再用 entity name 喂 `search_graph` 或 `search_vector` 拿 chunk evidence

### 3.2 建议做（Soft scope，benchmark 后定）

#### 3.2.1 输出 schema 统一容器（解 §2.4）

graph 类 tool 输出加 unified `Result<T>` 容器：

```json
// 统一容器
{
  "items": [...],
  "next_cursor": null,
  "total_count": 12
}
```

但因 graph 类 tool 数据语义跟 search 类不同（entity / relation 不是 ranked list），强制统一可能反而失语义。建议 task #33 测试审计 + agent 实测对比后定。

#### 3.2.2 参数默认值对齐（解 §2.5）

- 所有 search 类 `top_k` 默认 5
- `query_graph_entities` 默认 10 调到 5（跟 search 类对齐）— 但需要 verify 这不破坏 caller 期望
- `similarity_threshold` 在 fulltext / graph 是否可暴露？需要 backend 支持

### 3.3 不做（YAGNI）

- 不暴露后端 fine-grained pipeline 能力（§2.6）— P3，等 task #33 测试审计 + 真实用例驱动再决定
- 不引入 cross-index fusion / rerank layer（rerank 已删除，新 fusion 层需要明确 use case 才考虑）
- 不重命名 tool（API 兼容 hard 要求）

## 4. rerank 删除 fold-in（产品决策落地）

### 4.1 product 决定（earayu2 msg=a81bc213 + 团队 6 lane 共识）

**直接删除**所有 rerank 相关代码 / 配置 / 文档 — 不留 escape hatch、不走 6 周 deprecation 观察。

**理由**（合并 architect msg=62894d7f + earayu2 directive）：
- 无大一统 search 聚合接口 → rerank 没合理执行位置
- 每个单索引（vector / fulltext / graph）有自己的 ranking 语义（向量相似度 / BM25 / graph evidence），不该共用 rerank
- backend `pipeline._apply_fallback_strategy` (graph results 优先 + 其他按 score 倒序) 已经是稳定 ranking — 删 rerank 不破用户体验
- 6 处维护面（4 runner + RerankService 类 + invocation_service.rerank + model_use scenario + UI/quickstart docs + provider 故障点）跟「私有化部署免维护」directive 反向

### 4.2 删除范围（task #35 PR 链已 cover）

- ✅ `aperag/llm/rerank/` 整 dir（PR #1899）
- ✅ 4 个 runner：jina_rerank / dashscope_rerank / litellm.rerank / openai_compatible.rerank（PR #1899）
- ✅ `RerankService` 类 + test（PR #1899）
- ✅ `pipeline._rerank()` + `_resolve_default_rerank_model_id()`（PR #1899）
- ✅ `model_invocation_service.rerank()` dispatch（PR #1899）
- ✅ `/api/v1/rerank` endpoint（PR #1899）
- ✅ `RerankRequest/Response/Document/Usage` schemas + `ModelCapability.RERANK`（runtime types）（PR #1899）
- ✅ `SearchRequest.rerank` field + `NAMESPACE_RERANK` + `cache_rerank_ttl_seconds`（PR #1899）
- ✅ MCP tool `rerank` 参数（PR #1898 ziang）
- ✅ `model_uses` 表 `retrieval_rerank` scenario（PR #1898 ziang）
- ✅ 前端 SearchTest / 模型配置页 / quickstart docs rerank 入口（PR #1897 dongdong）
- ⏳ 验收：grep gate（无 active runtime path）+ smoke（无 rerank model 配置时 vector / fulltext / graph / MCP search 正常）+ e2e provider matrix 不再要求 rerank model（task #40 huangzhangshu + Planetegg）

### 4.3 invariant lock（task #32 spec 必带）

加进 `tests/boundaries/`（建议命名 `test_no_rerank_in_mcp.py`）：
- `aperag/mcp/tools/` 下任何 tool 函数签名**不允许**含 `rerank` 参数
- `aperag/llm/` 下不允许 import `rerank` 模块（已删但 grep gate 防 reintroduction）
- `aperag/schema/` / `retrieval/schemas.py` 下不允许 `RerankRequest` / `RerankResponse` / `SearchRequest.rerank` 等字段
- AST level grep + module-level import gate，方便 future PR 误加回时立刻 CI 红

## 5. 实施 sub-task 拆分（parallel-friendly）

按依赖关系排序，每个子任务独立可 claim：

### Phase A（必须做，并行）

- **#32-A1**：补 graph endpoint + MCP wrapper 输出 `evidence_refs: list[{document_id, chunk_id, parse_version?}]`（per Weston msg=7500e57d 修订 — 不再是裸 `chunk_id` list，因 chunk_id 不全局唯一）
  - **Scope 三处全覆盖**（避免漏 endpoint）：`query_graph_entities` (entity refs) + `expand_graph_subgraph` (entity refs **+ relation refs**) + `get_entity_detail` (entity refs，或显式说明为何 defer)
  - backend：`aperag/domains/knowledge_graph/api/graph_routes.py` + `graph_service.py` 新增 lightweight DTO `GraphEvidenceRef` (`document_id` + `chunk_id` + 可选 `parse_version`) + endpoint response 投影
  - schema：`aperag/domains/knowledge_graph/schemas.py` + MCP tool wrapper 同步
  - payload 控制：`evidence_refs` 默认上限 10 + 保留 `source_chunk_count` / `total_source_chunks` 告知总数
  - 测试：unit + e2e hurl + boundary test 钉 `evidence_refs` 字段含 `document_id + chunk_id` 必须在响应中出现
  - 推荐 owner：@ziang（熟 indexing/search/graph）
- **#32-A2**：MCP tool docstring + 用户面文档全套统一更新
  - 不改代码 — 只改 docstring + 用户面文档
  - 文档 grep 范围（per dongdong msg=3072630b NIT）：`docs/zh-CN/integration/mcp.md` + `docs/zh-CN/integration/mcp-api.md` + `docs/zh-CN/integration/dify.md` 三个文件全 update
  - 验收：grep 旧「5 个工具」口径全 codebase 零命中（spec inventory 已是 15 个 tool）
  - 内容 scope：明确 `search_graph`（chunk-level）vs `query_graph_entities`（entity-level）粒度差异
  - 推荐 owner：@dongdong（前端/文档 lane）
- **#32-A3**：boundary test 钉 「MCP 不暴露 rerank」invariant
  - `tests/boundaries/test_no_rerank_in_mcp.py` 新建
  - AST grep 钉 4 类反 pattern
  - 推荐 owner：@huangheng（boundary test lane）

### Phase B（建议做，等 Phase A 数据 + agent 实测）

- **#32-B1**：输出 schema 统一容器（agent 实测 cost / parse failure 数据后定）
- **#32-B2**：参数默认值对齐（top_k / similarity_threshold / hops validation）
- **#32-B3**：MCP 工具命名 / 分类 review（不改 API 名，只在 docs 层重组）

### Phase C（YAGNI，待真实用例驱动）

- backend fine-grained pipeline 能力暴露（细分 ranking / fallback / graph search 模式）
- cross-index fusion 层（如真有需求）

## 6. 验收口径

### 6.1 Phase A 完成标准

- entity → chunk → doc 完整链路：agent 一次 `query_graph_entities` 调用即可拿到 `evidence_refs`（含 `document_id + chunk_id + parse_version?`），再走 `read_document_chunk(collection_id, document_id, chunk_id)` 闭环（不再需要二次 `search_*` 召回，不再需要额外查 document_id）
- MCP 接口面 grep `rerank` 字面**零命中**（test 数据迁移注释允许 allowlist）
- search_graph vs query_graph_entities docstring 明确区分粒度

### 6.2 boundary test gate（CI must pass）

- `tests/boundaries/test_no_rerank_in_mcp.py` 钉 MCP tool 不能再加 rerank 参数
- **rerank grep gate allowlist 明确白名单**（per Planetegg msg=ebbe468a + task #35 验收 lane 共用）：
  - 允许命中：迁移说明 / `docs/zh-CN/architecture/task-17-cr-review-checklist.md` 历史 changelog / `docs/modularization/breaking-changes/` 等历史文档
  - 必 0 命中：active MCP tool schema / runtime pipeline / provider config / `aperag/llm/` 当前代码 / quickstart 当前文档
- 现有 G1-G19 + `test_modularization_boundaries.py` + `test_worker_di_parity.py` 不破坏

### 6.3 e2e smoke + integration

- 无 rerank model 配置时 vector / fulltext / graph / MCP search 正常返回 candidates
- entity → chunk → doc 链路完整验证（per Planetegg msg=ebbe468a + chenyexuan msg=8a931200 + 冬柏 msg=6fb022d5 + Weston msg=7500e57d）：
  - **integration / hurl test** 必证：`query_graph_entities` 或 `expand_graph_subgraph` 返回的 `evidence_refs` (`document_id + chunk_id + parse_version?`) 能被 `read_document_chunk(collection_id, document_id, chunk_id)` 真实消费（不只暴露 schema 字段）
  - 避免「字段暴露但链路仍断」的伪修复 — agent 必须能用一次完整 chain 拿到 chunk content
  - 反向验证：spec 早期版本 `evidence_chunk_ids: list[str]` 不够，因为 `chunk_id` 不全局唯一，agent 仍需查 document_id — 必须 `evidence_refs` 含 `document_id` 才闭环

## 7. 关联文档

- task #35 rerank 删除 PR 链：#1898 (ziang) + #1899 (Bryce) + #1897 (dongdong) + task #40 验收 (huangzhangshu / Planetegg)
- task #17 任务系统不变式：[`task-system-invariants.md`](./task-system-invariants.md)
- 模块化重构 canonical SSoT：[`docs/modularization/architecture.md`](../../modularization/architecture.md)
- earayu2 directive：`#indexing优化` msg=c9c7cf31 (MCP 审计) + msg=42b74170 (无大一统接口) + msg=70cb0f6b (代码研究) + msg=a81bc213 (rerank 全删)

## 8. CR mandatory checklist（spec 落地时必走）

按 `docs/zh-CN/architecture/task-17-cr-review-checklist.md` 既有 5 cross-check + 4-pattern matrix + 6 hard gate，task #32 实施 PR 必带：

- Lesson #11 v5（entry-point migration sub-check — 不适用，无 process split）
- Lesson #12 v4（PR `lint-and-unit` CI 全绿是 mandatory ratify gate）
- Lesson #12 v5（CI status 解读 trust framing 反模式）
- Lesson #12 v6（grep line number ≠ 执行顺序，必 walk function / API endpoint / data type scope）
- **Lesson #12 v7**（caller signature → backend schema → runtime fallback 三层 grep — task #34 grep 漏 schema-layer fallback 教训）
- Lesson #13 v2.1 + v2.2（删 source 必删 obsolete test 文件 + 同步 update test data assertion — task #36 fix-forward¹ + fix-forward² 实证）
- 简单稳定 + 私有化部署免维护 4 guardrail（不无限扩 scope / 尽快上线 / 简单稳定优于复杂 / 免维护）

---

**起草**：@符炫炜（总架构师）
**日期**：2026-04-30
**版本**：v1（task #32 spec lock 候选；team review + earayu2 ratify 后由 PM @不穷 派单实施）
