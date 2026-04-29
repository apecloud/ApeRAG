# 图谱索引状态语义拆分设计

任务来源: #indexing优化 task #4 (拆分「图谱事实层可用」「图谱检索向量可用」+ 名字驱动检索三层降级).

设计敲定后, task #5 (同步主流程改造) / task #6 (事实层并发改造) / task #7 (向量异步任务) 都按这个文档实施.

## 1. 现状回顾

每条文档每个模态当前用一个 `DocumentIndex` 行表示状态.

```
DocumentIndex(
  document_id, parse_version, modality,   -- 三元组
  status,                                  -- PENDING / RUNNING / ACTIVE / FAILED
  is_serving, derived_artifact_path, ...
)
```

`Modality` 当前 5 种: `vector` / `fulltext` / `graph` / `summary` / `vision`.

图谱模态 (`graph`) 的同步流程在 `aperag/indexing/graph.py:GraphModalityWorker`:

- Phase 1: 清理旧 lineage (实体, 关系)
- Phase 2: 写新 lineage (实体名+类型 / 关系起点+终点+类型 / 实体-关系对应的 chunk 编号)
- Phase 3: compactor (生成压缩描述) + embedder + 向量库 upsert + merge_detector

整个 Phase 1+2+3 全部跑完, `DocumentIndex(modality=graph).status` 才置 `ACTIVE`.

## 2. 现在的问题

新加坡现场反映 graph 同步极慢, 根因之一就是 Phase 3 (描述压缩 + 向量) 慢且不稳, 还会调失败的 LLM/embedding 接口. 但因为整条同步在一个 modality 行下, 任何 Phase 3 失败都阻塞 graph index 进入 ACTIVE.

实际语义上, agent / 检索能依赖的最小集合只是 Phase 1+2 写出来的事实层. 描述和向量都是派生数据, 失败应该可降级.

#1866 暴露的 compactor 调用错误 (subject_kind/subject_label 漏传) 也是这层问题的一个表现 — 它让 Phase 3 实际跳过了, 但日志一直在喊错, 而 graph index 仍能进入 ACTIVE 这件事也说明描述压缩本来就不该在主流程关键路径上.

## 3. 拆分目标

把图谱模态状态从一个 `graph` 拆成两个独立的子模态:

| 新模态 | 原数据范围 | 状态语义 |
|---|---|---|
| `graph_facts` (事实层) | 实体名+类型 / 关系起点+终点+类型 / 实体-关系对应的 chunk 编号 | `ACTIVE` 表示能从图谱回到原文 chunk; agent / 内容驱动检索可用 |
| `graph_vectors` (检索向量) | 实体向量, 关系向量 | `ACTIVE` 表示名字驱动检索的向量层可用; 不可用时降级到精确匹配 + 别名 + 模糊匹配 |

描述这次不再生成, 也不需要新的模态 — 只在 schema 字段保留兼容老数据. 描述向量永远不再生成.

文档的图谱索引完成 = `graph_facts.status = ACTIVE`. 不再依赖 `graph_vectors`.

## 4. 落地方案

### 4.1 模态枚举改动

`aperag/indexing/models.py:Modality` 加两个值, 保留 `graph` 当作只读的兼容值用于读取老数据:

```python
class Modality(str, Enum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    GRAPH = "graph"               # 历史值, 仅用于读取老数据, 新写入不再使用
    GRAPH_FACTS = "graph_facts"   # 新: 实体, 关系, chunk 关联
    GRAPH_VECTORS = "graph_vectors"  # 新: 实体/关系向量
    SUMMARY = "summary"
    VISION = "vision"
```

`Modality` 是字符串枚举, 列宽够 (`String(32)`), 不需要 alembic schema 改动. 加值即可.

### 4.2 数据迁移策略

老数据保留兼容, 不强行迁移. 具体规则:

- 现有 `Modality.GRAPH` 行不动. 它代表「老语义下事实层 + 描述 + 向量都已写完」. 读路径需要兼容这个值.
- 新写入永远写 `GRAPH_FACTS` 和 `GRAPH_VECTORS` 两行, 不再写 `GRAPH`.
- 兼容期内, 读路径同时认 `GRAPH` 和 `GRAPH_FACTS` 当作事实层可用; 同时认 `GRAPH` 和 `GRAPH_VECTORS` 当作检索向量可用.

为什么不迁移老行? 因为:
- 老行已经表示「全部完成」, 拆成两行需要写老数据生成时间, 等价信息可以从老行直接推断
- 不迁移避免一次性 DDL 操作影响线上库
- 两种值并存的兼容逻辑代价小 (查询时 `WHERE modality IN ('graph', 'graph_facts')`)

老 `GRAPH` 行的清理留到后续单独的清理任务, 不在本次范围.

### 4.3 同步主流程改造 (task #5)

`GraphModalityWorker.sync()` 拆成两个独立 worker:

#### `GraphFactsWorker.sync()` (新, 替换原 `GraphModalityWorker` 的 Phase 1+2)

- Phase 1: 清理旧 lineage
- Phase 2: 写新 lineage (实体, 关系, chunk 关联)
- 完成 → `DocumentIndex(modality=graph_facts).status = ACTIVE`
- **不调** compactor, **不写** 描述, **不调** embedder, **不写** 向量
- merge_detector 仍然在 Phase 2 末尾跑 (它写 `GraphCurationSuggestion` 表, 跟描述/向量无关)

#### `GraphVectorsWorker.sync()` (新, 处理原 Phase 3 的向量部分)

- 输入: `kg.jsonl` derived artifact (跟 `graph_facts` 共用同一份 derived 数据)
- 调 embedder 给每个实体生成向量, 输入文本 = 实体名 + 实体类型
- 调 embedder 给每个关系生成向量, 输入文本 = 起点实体 + 关系类型 + 终点实体
- 写向量库 (Qdrant 等)
- 完成 → `DocumentIndex(modality=graph_vectors).status = ACTIVE`
- **不调** compactor, **不生成** 描述向量

#### compactor 不再在主流程调用

`_maybe_compact` / `_maybe_compact_relation` 在 `GraphFactsWorker.sync()` 不调用. 描述列写入跳过 (新数据 description 字段为 NULL).

老 description 数据保留, schema 不改.

`GraphIndexCompactor` 类本身不删 (老数据如果需要重新压缩, 可以独立调用), 但不放在主同步路径上.

### 4.4 调度改造

orchestrator 在 `graph_index` 任务派发时, 同时插入 `GRAPH_FACTS` 和 `GRAPH_VECTORS` 两行 PENDING `DocumentIndex`:

- `GRAPH_FACTS` 派给 `GraphFactsWorker`
- `GRAPH_VECTORS` 派给 `GraphVectorsWorker`
- 两个 worker 互不依赖, 可以并行跑

`GRAPH_VECTORS` 失败不影响 `GRAPH_FACTS` 的 ACTIVE 状态. 反过来 `GRAPH_FACTS` 失败时 `GRAPH_VECTORS` 也可以独立跑 (前提是 `kg.jsonl` 已生成).

实际依赖关系: 两者都依赖 `parser` 阶段产出的 `chunks.jsonl` 和 `kg.jsonl` 派生工件. `kg.jsonl` 是 `graph.derive` 产出的, 抽取阶段不在本次改动范围.

### 4.5 文档级 graph 状态判定

agent / 内容驱动检索 / UI 展示 看到的「图谱可用」就是事实层可用:

```
ACTIVE if EXISTS (
  SELECT 1 FROM document_index
  WHERE document_id = ?
    AND modality IN ('graph', 'graph_facts')
    AND status = 'ACTIVE'
    AND is_serving = TRUE
)
```

向量是否可用单独判定:

```
ACTIVE if EXISTS (
  SELECT 1 FROM document_index
  WHERE document_id = ?
    AND modality IN ('graph', 'graph_vectors')
    AND status = 'ACTIVE'
    AND is_serving = TRUE
)
```

文档详情接口或 UI 如果要展示「图谱完整可用 = 事实层 + 向量都 ACTIVE」, 上层做 AND 拼接.

## 5. 名字驱动检索的三层降级

「按实体名查图谱」检索路径在向量层不可用时仍要工作. 顺序:

1. **精确匹配** — 用户输入的实体名跟 `aperag_lineage_entity.name` 精确相等
2. **别名 / 规范化匹配** — 实体的 alias 列表 (如有) + 大小写 / 繁简 / 空格规范化
3. **实体向量匹配** — 用查询文本的 embedding 在图谱检索向量库中召回相似实体

前两层不依赖向量库, 在 `GRAPH_VECTORS` 还没 ACTIVE 时也能服务.

第三层按 `GRAPH_VECTORS.is_serving` 判定是否可用. 不可用时检索接口跳过这一层, 不报错.

实现位置: `GraphSearchService` 或类似的图谱检索接口. 本次设计文档定下契约, 实施细节在 task #5/#6 的 PR 里推进.

## 6. 抽取阶段不动

`graph.derive` (LLM 调 chunk 抽取实体/关系) 不在本次范围. 抽取阶段并发改动风险 (成本, 限流, 质量) 大于收益.

## 7. 改动清单

下面给 task #5 和 task #6 的实施列出具体改动点.

### task #5 (同步主流程改造)

1. `aperag/indexing/models.py:Modality` 加 `GRAPH_FACTS` / `GRAPH_VECTORS`.
2. `aperag/indexing/graph.py`:
   - 拆 `GraphModalityWorker` 成 `GraphFactsWorker` (modality = `graph_facts`) 和 `GraphVectorsWorker` (modality = `graph_vectors`).
   - `GraphFactsWorker.sync()` 只跑 Phase 1+2 + merge_detector.
   - `GraphVectorsWorker.sync()` 跑 embedder + 向量 upsert. 输入文本按本设计 §4.3 规则.
   - 老 `GraphModalityWorker` 保留 (兼容跑完未迁移的老任务) 或直接删除 (新调度只走两个新 worker). 这次倾向保留, 标 deprecated, Phase 6 清理一起删.
3. `aperag/indexing/orchestrator.py` 派发逻辑: `graph` 模态作业改成同时派 `GRAPH_FACTS` + `GRAPH_VECTORS` 两行.
4. 文档级 graph 状态判定改成本设计 §4.5 SQL.
5. `_maybe_compact` / `_maybe_compact_relation` 不再调用. compactor 类本身保留, 但 `GraphFactsWorker` / `GraphVectorsWorker` 都不持有它.
6. 描述字段写 NULL: `LineageGraphStore.upsert_entity_with_lineage` / `upsert_relation_with_lineage` 调用方传 `description_parts=[]`.
7. 单元测试: 钉两个 worker 各自 ACTIVE 不依赖另一个; `kg.jsonl` 缺失时降级.

### task #6 (事实层并发改造)

1. `GraphFactsWorker.sync()` 把 Phase 2 的循环写入改成批量: 收集所有 entity / relation 后调一次 `LineageGraphStore.upsert_entities_bulk` / `upsert_relations_bulk` (新接口, store 层加).
2. `LineageGraphStore` 抽象层加 bulk 接口. Postgres 实现走 `INSERT ... ON CONFLICT (collection_id, name) DO UPDATE` 批量执行 (~100 条一批).
3. 受限并发: `entity_lock` 仍按实体名串行, 但跨实体可以并发. 使用 `asyncio.Semaphore` 限制并发度 (默认 4, mirror PR #1809 graph extractor).
4. 单元测试: 钉 bulk 写入幂等 + 受限并发不互相覆盖.

## 8. 风险和兼容

- **读取老数据**: 兼容期内 `WHERE modality IN ('graph', 'graph_facts')` 同时支持新老格式. 兼容期长度待 task #3 调用路径检查结果出来后再定.
- **现有 graph_search 接口**: 调用路径检查 (task #3) 结果出来后, 决定是否需要改接口实现. 本设计不强制改接口.
- **向量库 ID 冲突**: `GraphVectorsWorker` 使用跟原 `GraphModalityWorker` 相同的 `_entity_vector_id` / `_relation_vector_id` 算法, 老向量数据可以被覆盖更新, 不重复.
- **抽取阶段未动**: 如果抽取本身仍然慢 (没并发), 同步并发改造的提升有限. 后续单独评估抽取阶段并发.

## 9. 评审请求

@符炫炜 / @Weston / @Planetegg / @huangzhangshu / @明书 / @不穷 评审重点:

1. modality 枚举加值方案 vs 单独表方案, 取舍是否合理?
2. 老数据不迁移, 仅在读路径兼容 — 风险可接受吗?
3. `GraphVectorsWorker` 的输入文本规则 (实体名+类型 / 起点+关系+终点) 是否需要扩展?
4. 三层降级 (精确 / 别名+模糊 / 向量) 是否要在本次同步落地, 还是等 task #3 调用路径检查后再实施?
5. compactor 类保留还是直接删? 兼容期内有没有读取老 description 的接口需要调用 compactor?

评审通过后启动 task #5 实施.
