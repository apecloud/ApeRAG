# 图谱索引状态语义拆分设计 (v3)

任务来源: #indexing优化 task #4 (拆分「图谱事实层可用」「图谱检索向量可用」+ 名字驱动检索三层降级).

设计敲定后, task #5 (同步主流程改造) / task #6 (事实层并发改造) / task #7 (向量层 reconciler 入队) 都按这个文档实施.

v3 吸收第二轮评审反馈 (符炫炜 / Weston / Planetegg / huangzhangshu).

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

#1866 暴露的 compactor 调用错误也是这层问题的一个表现 — Phase 3 实际跳过了, 但日志一直在喊错, 而 graph index 仍能进入 ACTIVE 这件事也说明描述压缩本来就不该在主流程关键路径上.

## 3. 拆分目标

把图谱模态状态从一个 `graph` 拆成两个独立的子模态:

| 新模态 | 原数据范围 | 状态语义 |
|---|---|---|
| `graph_facts` (事实层) | 实体名+类型 / 关系起点+终点+类型 / 实体-关系对应的 chunk 编号 | `ACTIVE` 表示能从图谱回到原文 chunk; agent / 内容驱动检索可用 |
| `graph_vectors` (检索向量) | 实体向量, 关系向量, 候选合并检测结果 | `ACTIVE` 表示名字驱动检索的向量层可用; 不可用时降级到精确匹配 + 别名 + 模糊匹配 |

描述这次不再生成, 也不需要新的模态 — 只在 schema 字段保留兼容老数据. 描述向量永远不再生成.

文档的图谱索引完成 = `graph_facts.status = ACTIVE`. 不再依赖 `graph_vectors`.

## 4. 落地方案

### 4.1 模态枚举改动

`aperag/indexing/models.py:Modality` 加两个值, 老 `GRAPH` 保留并标 deprecated:

```python
class Modality(str, Enum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    # Deprecated: 老语义下事实+描述+向量都已写完才进入 ACTIVE.
    # 读路径兼容期保留. 新代码不再使用. 后续清理任务一起删.
    GRAPH = "graph"
    GRAPH_FACTS = "graph_facts"   # 新: 实体, 关系, chunk 关联
    GRAPH_VECTORS = "graph_vectors"  # 新: 实体/关系向量 + 候选合并检测
    SUMMARY = "summary"
    VISION = "vision"
```

`Modality` 的 DB 列宽 `String(32)`, 不需要 alembic schema 改动. 加值即可.

### 4.2 数据迁移策略

老数据保留兼容, 不强行迁移. 具体规则:

- 现有 `Modality.GRAPH` 行不动. 它代表「老语义下事实层 + 描述 + 向量都已写完」. 读路径需要兼容这个值.
- 新写入永远写 `GRAPH_FACTS` 和 `GRAPH_VECTORS` 两行, 不再写 `GRAPH`.
- 兼容期内, 读路径按 §4.5 的双场景规则区分「只有老数据」和「已有新事实层」, 避免老 `graph` 行误判向量层可用.

为什么不迁移老行?

- 老行已经表示「全部完成」, 拆成两行需要写老数据生成时间, 等价信息可以从老行直接推断
- 不迁移避免一次性 DDL 操作影响线上库
- 双场景兼容代价小, 等本次改动稳定后单独清理

老 `GRAPH` 行的清理留到后续单独的清理任务, 不在本次范围.

### 4.3 同步主流程改造 (task #5)

`GraphModalityWorker.sync()` 拆成两个独立 worker, 都消费**同一份** `kg.jsonl` derived 工件 (`graph.derive` 在抽取阶段产出, 不在两个 worker 范围内).

#### `GraphFactsWorker.sync()` (新, 处理 Phase 1+2 的 lineage 写入)

- 输入: `kg.jsonl` (从 `graph.derive` 派生工件读取)
- Phase 1: 清理旧 lineage
- Phase 2: 写新 lineage (实体, 关系, chunk 关联)
- 完成 → `DocumentIndex(modality=graph_facts).status = ACTIVE`
- **不调** compactor, **不调** embedder, **不写** 向量, **不调** merge_detector
- **不追加新描述片段**, 同时**清掉同 `(document_id, parse_version)` 的旧描述片段** (per Weston msg=d2324ea3 第 1 点). `description_parts` 数组列处理: lineage member 写入时传空数组. `compacted_description` 字段写空.

#### `GraphVectorsWorker.derive()` (新, 复用事实层的 kg.jsonl)

不重新调 LLM 抽取实体/关系. 实现 (per huangzhangshu msg=66a9127b + Planetegg msg=753f5cf5):

```python
async def derive(self, *, document_id, parse_version, source_path):
    # 查同一 (document_id, parse_version) 的 graph_facts 当前服务行
    facts_row = ops.query_document_index_serving(
        document_id=document_id,
        parse_version=parse_version,
        modality=Modality.GRAPH_FACTS,
    )
    if facts_row is None or facts_row.status != ACTIVE:
        # graph_facts 还没就绪 → 返空路径, orchestrator 重新排队
        return DeriveResult(derived_artifact_path=None)
    # 复用 graph_facts 的 derived_artifact_path (= kg.jsonl)
    return DeriveResult(derived_artifact_path=facts_row.derived_artifact_path)
```

#### `GraphVectorsWorker.sync()` (新, 处理原 Phase 3 的向量 + merge_detector)

- 输入: `kg.jsonl` (复用 `graph_facts` 服务行的 `derived_artifact_path`)
- 调 embedder 给每个实体生成向量, 输入文本 = 实体名 + 实体类型 + 如果有 alias 列表 concat 进去 (例如「张三 / Zhang San / 张医生 / Person」). alias 字段没有就跳过, 不强求 schema 改.
- 调 embedder 给每个关系生成向量, 输入文本 = 起点实体 + 关系类型 + 终点实体
- 写向量库 (Qdrant 等)
- merge_detector 移到这里跑 (best-effort, 失败仅 log warning, 不阻塞 `GRAPH_VECTORS` ACTIVE). 它依赖 embedder 和向量库做相似度查询, 必须在向量层之后才能跑, 不能留在事实层主链路.
- 完成 → `DocumentIndex(modality=graph_vectors).status = ACTIVE`
- **不调** compactor, **不生成** 描述向量

#### compactor 不再在主流程调用

`_maybe_compact` / `_maybe_compact_relation` 在两个 worker 都**不调用**.

老 description 数据保留, schema 不改.

`GraphIndexCompactor` 类本身**保留并标 deprecated**, 不删. 兼容期内不调用. 保留它的两个用途:

- 老 description 数据如果运维想批量重新压缩 (极小概率), 可以独立调用
- 后续兼容期清理任务 (清理老 `graph` 行 + 老 description) 一起删除

### 4.4 调度改造

#### 派发顺序: 事实层先, 向量层后由 reconciler 入队 (per 符炫炜 msg=d9ae9a00 第 1 点)

两个 worker 都消费同一份 `kg.jsonl`, 但状态机上**不并行**, 走保守序列:

1. orchestrator 在 `graph_index` 任务派发时, 只插入 `GRAPH_FACTS` 一行 PENDING.
2. `GraphFactsWorker` 消费 `kg.jsonl` 写事实层.
3. **reconciler 周期**检测 `(document_id, parse_version)` 同时满足以下两个条件:
   - `graph_facts.status = ACTIVE`
   - 不存在同 `(document_id, parse_version)` 的 `graph_vectors` 行 (说明事实层 ACTIVE 后还没首次入队)
4. 满足条件 → reconciler insert `graph_vectors` PENDING 行.
5. `GraphVectorsWorker` 消费同一份 `kg.jsonl` 写向量层 + merge_detector.
6. 事实层失败 → reconciler 不会满足条件 1, 向量层永远不入队. 避免「向量库写了半成品 + 事实层失败」的脏状态.

注意: 向量层入队**只走 reconciler**, 不让 orchestrator 在 worker 完成时同步触发. 理由:

- reconciler 已经有失败重试机制, 复用即可
- 30 秒延迟无所谓 (向量层不是文档完成关键路径)
- 少一个触发路径降低复杂度

#### 失败 / 重试语义

- 事实层 PENDING / RUNNING / FAILED: 走现有 orchestrator + reconciler 的 retry pattern.
- 事实层 ACTIVE 后, reconciler 周期入队 `GRAPH_VECTORS` PENDING.
- 向量层 PENDING / RUNNING / FAILED: reconciler 走独立 retry stage (这是 task #7 的范围, 不重新写 worker, 只在 reconciler 周期检测「事实层 ACTIVE + 向量层 FAILED 或缺失」的行 → enqueue 向量重试或首次入队).
- 向量层 ACTIVE 后, 整个图谱完整可用. 向量层 FAILED 永久 (超过 max attempts) → 标 FAILED, UI 标「向量层不可用」, 检索降级到精确 + 别名+模糊匹配.

#### 双 PENDING 行原子事务 (废弃)

之前版本提过「两行 PENDING 同一事务原子插入」, 这版改成串行后**不需要**: orchestrator 只插一行 (`GRAPH_FACTS`), 事实层 ACTIVE 后 reconciler insert 第二行. 没有同时插两行的原子性顾虑.

### 4.5 文档级 graph 状态判定 (双场景兼容)

#### 场景 A: 只有老 `graph` 数据, 没有新 `graph_facts`

老 `graph` 行兼容为事实层和向量层**都可用**:

```
EXISTS (
  SELECT 1 FROM document_index
  WHERE document_id = ?
    AND modality = 'graph'
    AND status = 'ACTIVE'
    AND is_serving = TRUE
)
```

#### 场景 B: 已有新 `graph_facts` 服务行

事实层只看 `graph_facts`, 向量层只看 `graph_vectors`. **不再用老 `graph` 行推断向量层** (per Planetegg msg=406dba9f).

事实层可用:

```
EXISTS (
  SELECT 1 FROM document_index
  WHERE document_id = ?
    AND modality = 'graph_facts'
    AND status = 'ACTIVE'
    AND is_serving = TRUE
)
```

向量层可用:

```
EXISTS (
  SELECT 1 FROM document_index
  WHERE document_id = ?
    AND modality = 'graph_vectors'
    AND status = 'ACTIVE'
    AND is_serving = TRUE
)
```

#### 场景判定 SQL 整合

```python
def graph_facts_available(document_id):
    has_new = exists(modality='graph_facts', is_serving=True, status='ACTIVE')
    if has_new:
        return True
    # 场景 A: 老数据兼容
    return exists(modality='graph', is_serving=True, status='ACTIVE')

def graph_vectors_available(document_id):
    has_new_facts = exists(modality='graph_facts', is_serving=True)
    if has_new_facts:
        # 场景 B: 已升级到新事实层, 向量层只看 graph_vectors
        return exists(modality='graph_vectors', is_serving=True, status='ACTIVE')
    # 场景 A: 老数据兼容, graph 行同时承担向量层语义
    return exists(modality='graph', is_serving=True, status='ACTIVE')
```

`is_serving` 字段在两个新模态上**独立判断**: 当 `graph_facts` ACTIVE 但 `graph_vectors` FAILED 时, 前者 `is_serving=TRUE`, 后者 `is_serving=FALSE`. 文档详情查询时两个字段独立解读.

#### 文档整体完成状态不聚合 `graph_vectors`

即使 `graph_vectors` 失败, 文档整体也仍然是可完成的状态 (因为图谱可用只看 `graph_facts`). `graph_vectors` 是补充功能, 不应该让文档卡在「向量没生成完」状态.

跟向量库 / 全文索引的关键区别: 那两个是文档级必备能力, 失败要算文档失败; `graph_vectors` 是「图谱检索的额外能力」, 失败不算文档失败.

文档详情接口或 UI 如果要展示「图谱完整可用 = 事实层 + 向量都 ACTIVE」, 上层做 AND 拼接, 仅作展示用途, 不影响文档整体完成判定.

### 4.6 兼容清单 (按读写路径)

下面把每个涉及 graph 状态的读写路径都列出来, 明确老 `graph` 行 / 新 `graph_facts` / 新 `graph_vectors` 各自怎么处理. task #3 调用路径检查会按这个清单一一确认.

| 路径 | 老 `graph` (兼容期保留) | 新 `graph_facts` | 新 `graph_vectors` |
|---|---|---|---|
| 文档详情 `graph_index_status` | 走 §4.5 双场景判定 | 走 §4.5 双场景判定 | 不参与 (展示在独立字段 `graph_vectors_status`) |
| 失败索引查询 (`/rebuild_failed_indexes`) | 老 `graph` FAILED 行重建变成两行新 modality (老行**直接删除**) | 直接查 `graph_facts` FAILED | 独立查 `graph_vectors` FAILED |
| 重建接口 (`POST /rebuild_index`) | 老 `graph` FAILED 行**直接删除**, 重建写 `graph_facts` + (reconciler 后续) `graph_vectors` 两行 | 重建只走 `graph_facts` | 重建只走 `graph_vectors` |
| 删除清理 (`DELETE /collection`) | cleanup pipeline 删除老 `graph` 行 + 向量库点 + lineage 表行 | cleanup 删除 `graph_facts` 行 + lineage 表行 | cleanup 删除 `graph_vectors` 行 + 向量库点 |
| 文档级整体完成判定 | 走 §4.5 双场景判定 | 走 §4.5 双场景判定 | **不参与** 整体完成判定 |
| 图谱检索接口读 lineage | 读 `aperag_lineage_entity` / `aperag_lineage_relation` (跟 modality 无关) | 同上 | 同上 |
| 图谱检索接口读向量库 | 走 §4.5 双场景判定 | 没向量, 跳过 | 走 §4.5 双场景判定 |
| 图谱可视化入口 | 由 #indexing优化 task #9 (郭子昂) 单独 PR 处理, **不阻塞事实层/向量层状态拆分**. 评估结论见 thread `#indexing优化:77809cb9` | 同 task #9 | 同 task #9 |

注意:

- 老 `graph` FAILED 重建时**直接删除**老行 (不留 superseded 状态机), 避免兼容逻辑变得复杂. (per 符炫炜 msg=d9ae9a00 第 2 点)
- 删除 collection 时**双 modality 同 cleanup pipeline 一起删**, 避免向量层孤儿数据.

## 5. 名字驱动检索的三层降级 (本次落地前两层, 第三层等向量层 ACTIVE 自然启用)

「按实体名查图谱」检索路径在向量层不可用时仍要工作. 顺序:

1. **精确匹配** — 用户输入的实体名跟 `aperag_lineage_entity.name` 精确相等
2. **别名 / 规范化匹配** — 实体的 alias 列表 (如有) + 大小写 / 繁简 / 空格规范化, 用 PostgreSQL `ILIKE` 或 `pg_trgm` 模糊匹配
3. **实体向量匹配** — 用查询文本的 embedding 在图谱检索向量库中召回相似实体

前两层不依赖向量库, 在 `GRAPH_VECTORS` 还没 ACTIVE 时也能服务.

第三层按 `GRAPH_VECTORS.is_serving` 判定是否可用. 不可用时检索接口跳过这一层, 不报错.

#### 实施位置归属 (per 符炫炜 msg=d9ae9a00 第 3 点)

**前两层 (精确 + 别名/模糊) 在 task #5 范围内实施**, 跟 worker 拆分一起 ship. 实施位置: `GraphSearchService` 或类似图谱检索接口加 fallback 逻辑.

第三层在向量层 ACTIVE 后自然可用, 不需要额外编码 (检索接口已经接入向量库, 任 `GRAPH_VECTORS.is_serving=TRUE` 后自动启用).

不分到 task #6 (因为 #6 是事实层并发改造, scope 不同).

## 6. 抽取阶段不动

`graph.derive` (LLM 调 chunk 抽取实体/关系) 不在本次范围. 抽取阶段并发改动风险 (成本, 限流, 质量) 大于收益.

## 7. 改动清单

下面给 task #5 / task #6 / task #7 的实施列出具体改动点.

### task #5 (同步主流程改造) — Bryce 接

1. `aperag/indexing/models.py:Modality` 加 `GRAPH_FACTS` / `GRAPH_VECTORS`. `GRAPH` 加 deprecated 注释.
2. `aperag/indexing/graph.py`:
   - 拆 `GraphModalityWorker` 成 `GraphFactsWorker` (modality = `graph_facts`) 和 `GraphVectorsWorker` (modality = `graph_vectors`).
   - `GraphFactsWorker.sync()` 只跑 Phase 1+2 lineage 写入, 描述片段传空数组, **不调** compactor / embedder / merge_detector.
   - `GraphVectorsWorker.derive()` 复用同 `(document_id, parse_version)` 的 `graph_facts` 服务行 `derived_artifact_path`; 找不到返空路径让 orchestrator 重排队.
   - `GraphVectorsWorker.sync()` 跑 embedder + 向量 upsert + merge_detector (best-effort). 输入文本按本设计 §4.3 规则.
   - 老 `GraphModalityWorker` 保留, 标 deprecated, Phase 6 清理一起删.
3. `aperag/indexing/orchestrator.py` 派发逻辑: `graph` 模态作业改成只派 `GRAPH_FACTS` 一行 PENDING.
4. 文档级 graph 状态判定改成本设计 §4.5 双场景 SQL.
5. `_maybe_compact` / `_maybe_compact_relation` 不再调用. compactor 类保留并标 deprecated, 但两个新 worker 都不持有它.
6. `LineageGraphStore.upsert_entity_with_lineage` / `upsert_relation_with_lineage` 调用方 (`GraphFactsWorker`) 传 `description_parts=[]` + 清掉同 `(document_id, parse_version)` 的旧描述片段.
7. 兼容期处理 §4.6 兼容清单的读写路径: 文档详情字段 `graph_index_status` 加双场景映射逻辑; `rebuild_failed_indexes` 老 `graph` FAILED 行直接删 + 重建写两行新 modality; 删除清理双 modality 同 pipeline.
8. `GraphSearchService` 加名字驱动检索前两层降级 (精确 + 别名/模糊).
9. 单元测试钉以下契约:
   - 两个 worker 各自 ACTIVE 不依赖对方
   - `kg.jsonl` 缺失时 `GraphFactsWorker.derive()` 降级返空路径
   - `GraphVectorsWorker.derive()` 在 `graph_facts` 未 ACTIVE 时返空路径让 orchestrator 重排
   - 实体向量输入文本: (a) 有 alias 时 concat 进去 (b) 没 alias 时跳过. 这两个 case 必须独立钉死.
   - 老 `graph` ACTIVE 行的 §4.5 场景 A 兼容路径
   - 已升级到新 `graph_facts` 后, §4.5 场景 B 不再用老 `graph` 推断向量层

### task #6 (事实层并发改造) — Bryce 接

1. `GraphFactsWorker.sync()` 把 Phase 2 的循环写入改成批量: 收集所有 entity / relation 后调一次 `LineageGraphStore.upsert_entities_bulk` / `upsert_relations_bulk` (新接口, store 层加).
2. `LineageGraphStore` 抽象层加 bulk 接口. Postgres 实现走 `INSERT ... ON CONFLICT (collection_id, name) DO UPDATE` 批量执行 (~100 条一批).
3. 受限并发: `entity_lock` 仍按实体名串行, 但跨实体可以并发. 使用 `asyncio.Semaphore` 限制并发度 (默认 4, mirror PR #1809 graph extractor).
4. 单元测试钉 bulk 写入幂等 + 受限并发不互相覆盖.

### task #7 (向量层 reconciler 入队) — 明书接

task #7 **不是** 重新实现 `GraphVectorsWorker` (那是 task #5 的范围). task #7 是 **reconciler 周期任务**, 负责两件事:

1. **首次入队** (per Weston msg=d2324ea3 第 3 点): 扫 `(document_id, parse_version)` 满足「`graph_facts` ACTIVE + 不存在 `graph_vectors` 行」, insert `graph_vectors` PENDING.
2. **失败重试**: 扫 `(document_id, parse_version)` 满足「`graph_facts` ACTIVE + `graph_vectors` FAILED + retry_count < max_attempts + retry_after 已过」, 重新 enqueue.

实施位置: `aperag/indexing/reconciler.py` 加一个新的 stage. 单元测试钉以下契约:

- 「事实层 ACTIVE + 向量层缺失」会被 reconciler 首次入队
- 「事实层 ACTIVE + 向量层 FAILED + retry_count < max_attempts」会被 reconciler 重新 enqueue
- 「事实层 FAILED」时不会入队 `graph_vectors` (避免半成品)
- max attempts 后停止重试, 状态标 FAILED 不再触发

## 8. 风险和兼容

- **读取老数据**: 兼容期内按 §4.5 双场景 SQL 处理. 兼容期长度待 task #3 调用路径检查结果出来后再定.
- **现有 graph_search 接口**: 调用路径检查 (task #3) 结果出来后, 决定是否需要改接口实现. 本设计强制要求三层降级前两层在 task #5 落地.
- **向量库 ID 冲突**: `GraphVectorsWorker` 使用跟原 `GraphModalityWorker` 相同的 `_entity_vector_id` / `_relation_vector_id` 算法, 老向量数据可以被覆盖更新, 不重复.
- **抽取阶段未动**: 如果抽取本身仍然慢 (没并发), 同步并发改造的提升有限. 后续单独评估抽取阶段并发.
- **图谱可视化**: 评估在 task #8 (郭子昂) 完成 (thread `#indexing优化:77809cb9`), 实施在 task #9 单独 PR 处理 (前端 facts 兜底布局 / 边 relation_type 透传 / evidence 懒加载 / FE 搜索三层降级). 不阻塞本设计.

## 9. v2 → v3 变更日志

吸收的反馈来源:
- 符炫炜 (msg=d9ae9a00): 5 点细节
- Weston (PR #1870 review): 3 个边界 + 名字驱动检索前两层落地
- huangzhangshu (msg=66a9127b): GraphVectorsWorker.derive 复用 graph_facts 的 derived_artifact_path
- Planetegg (msg=406dba9f): 老 graph 双场景判定规则细化
- Planetegg (PR #1870 review 753f5cf5): 同 huangzhangshu 的复用建议

主要变化:

1. §4.1 `GRAPH` 加 deprecated 注释
2. §4.3 描述写空真实做法明确 (不追加新片段 + 清旧片段, `compacted_description` 写空); GraphVectorsWorker.derive 复用 graph_facts 的 derived_artifact_path
3. §4.4 向量层入队**统一走 reconciler**, 不让 orchestrator 同步触发; task #7 范围明确包含「首次入队 + 失败重试」
4. §4.5 双场景兼容规则明确: 只有老数据 / 已有新事实层
5. §4.6 老 `graph` FAILED 行重建时**直接删除**, 不留 superseded
6. §5 名字驱动检索前两层 (精确 + 别名/模糊) 归 task #5 实施, 第三层向量层 ACTIVE 后自然启用
7. §7 task #5 单测加 alias 有/无两个 case 钉死; task #7 范围加首次入队
