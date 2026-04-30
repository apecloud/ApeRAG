---
title: task #31 — Graph 节点合并扫描 + 后台建议任务 spec v1
description: ApeRAG graph 重复/近似实体检测 + 异步 suggestion 任务 + 适配 task #17 hard cut 部署 + Wave 5 description 不再生成 invariant
---

# task #31 — Graph 节点合并扫描 + 后台建议任务 spec v1

> earayu2 directive (`#indexing优化` msg=a6186741 + msg=2d47e107 全员协作)：graph 节点合并扫描和后台建议任务，需要随着新的架构（分离部署 indexing）和（不再生成 description）而做对应的设计和修改。

## 1. 现状 inventory（grep 实证）

### 1.1 现有 entity dedup / merge 路径

`aperag/graph_curation/`:
- `service.py` 已有 `merge_entities()` API（用户面 / agent 触发 manual merge）
- `alias_map.py` `resolve_canonical()` per-query alias resolution（**Planetegg msg=db7fb085** + Singapore 实证 P2-S1：per-node `asyncio.gather` 在 PG 连接池吃紧时放大压力）
- 无 **后台异步 dedup scan / suggestion** 路径 — 当前 dedup 全靠 (a) 提取阶段 prompt v2 dedup（Bryce A3 hard requirement #4）+ (b) merge_entities manual API

### 1.2 现有 background task 路径

`aperag/indexing/`：
- `cleanup.py` cleanup task（`Document.status=DELETED + gmt_deleted` SoT，task #17 hard cut 后 worker 独立 deployment）
- `reconciler.py` 5-stage reconciler (PENDING dispatch / FAILED retry / RUNNING reclaim / Stuck parse / Graph vectors enqueue+stale)
- 无 **graph node merge 后台 task / queue lane** — 当前 graph 异步任务只有 `graph_facts` / `graph_vectors` extraction

### 1.3 task #17 hard cut deployment 状态

`aperag/cli/indexing_worker.py` 启动 10 lane: parse / vector / fulltext / graph / graph_facts / graph_vectors / summary / vision / reconciler / cleanup（per `task-system-invariants.md` § 1.1）— 任何 graph node merge background task 必须 **加进 worker lane list**，不能跑在 API 进程（per task-system-invariants § 2.3 「API 不拥有重型执行面」hard gate）。

### 1.4 Wave 5 description 不再生成 invariant

`aperag/indexing/graph_extractor.py`（task #30 A3 prompt v2）已移除 description regeneration（Wave 5 task #5 facts/vectors split 后老 graph 失败 FAILED 行直接归一化删除，新 entity record 不再写 description — per `task-30-graph-chunk-window-spec-v1.md` § 4.5 invariant）— 节点合并 suggestion 必须 **基于 entity name / type / source_chunk_ids 跨 doc 共享性，不依赖 description 文本相似度**（description 字段 NULL）。

### 1.5 graph store contract（task #61 close 锁定）

`aperag/indexing/graph_storage/{neo4j,nebula,postgres}.py` 跨 3 backend `LineageGraphStore` Protocol contract 已 task #61 P0/P1 锁定:
- `bulk_upsert_entity_with_lineage_parts` cross-backend boundary test 38 cases (PR #1927 `9c94cbc1`)
- `merge_entities` 行为契约 task #64 ziang audit slice 输出 (任何 merge 必须 atomic + replay idempotent + zero-side-effect on raise)
- alias / source_chunk_ids list-typed schema (task #61 spec § 1.2 + Wave 5 multi-chunk provenance)

task #31 不引入新 graph store method — 复用 task #61 locked contract。

## 2. 缺口识别（按 severity）

按 Weston msg=85e527e3 + msg=d0e00405 三层框架 + earayu2 directive 三 sub-task 边界:

### 2.1 P0（必须做 — 影响上层正确性）

- **P0-31-A** background suggestion task lane 缺失：当前 `aperag/cli/indexing_worker.py` 10 lane 不含 `graph_node_merge_suggestion` lane — 必须新加 worker lane（per task #17 hard cut + task-system-invariants § 2.3）；suggestion task 不能跑 API 进程
- **P0-31-B** suggestion 必须 **可审阅 + 不静默 destructive merge**（per Weston msg=d0e00405 边界 3）：suggestion 写入独立 store（not auto-applied），用户面 (UI) 显式 review + accept/reject，不允许 background task 直接调 `merge_entities()` 静默改 graph
- **P0-31-C** Wave 5 description-NULL 兼容（per § 1.4 invariant）：dedup 算法 input **不能依赖 entity description 文本相似度** — 必须基于 entity name / type / source_chunk_ids overlap / vector embedding 跨 doc 共享性 surface 候选

### 2.2 P1（允许差异但显式 declaration）

- **P1-31-A** dedup detection 算法 capability matrix：name exact match (case-insensitive normalize) / name fuzzy (Levenshtein / Jaro-Winkler) / type compatibility / vector embedding similarity / source_chunk overlap — 各算法 trigger 条件 + threshold 显式 declare collection-level config
- **P1-31-B** suggestion store schema：`MergeSuggestion(id, collection_id, candidate_entities, score, reason, status (pending/accepted/rejected/dismissed), created_at, reviewer_user_id, reviewed_at)` — 用 PG 通用 store（不引入新 backend dependency）
- **P1-31-C** scan trigger 策略：(a) 周期 cron（reconciler 30s poll 已存在，可 piggyback）+ (b) 显式 manual trigger API（用户面 /admin "扫描重复实体" 按钮）+ (c) 文档全部入库后自动触发（new entity batch surface 时 enqueue scan）— 三策略并存 collection-level config 选启用

### 2.3 P2（性能优化）

- **P2-31-A** dedup scan complexity：N entity 全配对 O(N²) 不可行（10k entity 1 亿 pair）— 必须 candidate pre-filter（如 name first-letter / type hash bucket / embedding ANN top-K）— 算法层后续优化
- **P2-31-B** Planetegg P2-S1 alias resolution `asyncio.gather` 放大 PG 连接 — task #61 spec § 1.3 已 surface，task #31 实施时可 fold-in batch resolve primitive

### 2.4 P3 / YAGNI（不做）

- 不实施 silent destructive merge（per Weston 边界 3 + earayu2 directive 「保留人工审核 / 可回滚」）
- 不引入跨 collection / 跨 tenant 节点合并（boundary 太大 + 安全风险）
- 不实施 description regeneration（per Wave 5 invariant，description 永远 NULL）
- 不引入新 graph store backend / 新 vector backend（复用 task #61 locked LineageGraphStore + vectorstore Protocol）
- 不实施 background auto-accept 高 score suggestion（even score 0.99 也必须人工 review，per earayu2 「保留人工审核」directive）

## 3. 设计方向（task #31 主线）

### 3.1 必须做（Hard scope per Weston msg=d0e00405 4 边界 + earayu2 directive）

#### 3.1.1 后台 worker lane 加入 task #17 hard cut deployment

新增 worker lane:
- `graph_node_merge_suggestion`（11th lane，加进 `aperag/cli/indexing_worker.py` startup）
- 跑独立队列 `redis_queue.graph_node_merge_suggestion`（reuse RedisWorkQueue + RedisQuotaBackend infrastructure per task #17）
- API 进程 **不直接调用 dedup scan**（per `task-system-invariants` § 2.3 hard gate）— API 仅入队 + 读 suggestion store
- boundary test：`tests/boundaries/test_indexing_worker_lanes.py` 加钉「11th lane `graph_node_merge_suggestion` 必启动」

#### 3.1.2 suggestion store + reviewable workflow（per Weston msg=d0e00405 边界 3）

- 新 PG table `merge_suggestion`（schema § 2.2 P1-31-B）
- background task **只写 suggestion**（pending status）— 不调 `merge_entities()` apply
- 用户面 review API：`POST /api/v2/collections/{id}/merge-suggestions/{suggestion_id}/accept|reject|dismiss` — accept 时调 `merge_entities()` apply（async via worker），reject/dismiss 仅改 status
- review audit trail：accept 调用记录 `reviewer_user_id` + `reviewed_at`，可回滚（rollback API 仅 admin role）
- AlembicMigration 加 `merge_suggestion` table（chain 在 latest head 后）

#### 3.1.3 Wave 5 description-NULL 兼容 dedup 算法

dedup detection 输入仅:
- `EntityRecord.name`（normalize lowercase + trim + Unicode NFC）
- `EntityRecord.entity_type`
- `EntityRecord.source_chunk_ids` (list, task #30 A3 schema)
- entity vector embedding（如已有 — task #30 A3 vector 路径）

**不依赖** `EntityRecord.description`（Wave 5 后永远 NULL）。如算法层 fallback 到 description 文本对比 → boundary test 钉死「dedup detection 不读 description 字段」防 silent regression。

#### 3.1.4 graph store contract 复用（per Weston msg=d0e00405 边界 4）

`merge_entities()` apply 路径复用 task #61 locked invariants:
- `bulk_upsert_entity_with_lineage_parts` atomicity（PR #1927 38 cases pin）
- `LineageGraphStore.merge_entities` cross-backend contract（ziang task #64 close）
- replay idempotent（前一次 accept fail 后 retry 不会重复合并）

不引入新 graph store method — 全 reuse。

### 3.2 P1 实施（capability matrix declaration）

per § 2.2 三 strategy + 算法 capability matrix collection-level config:
- `kg.merge_suggestion_strategy: ['cron', 'manual', 'auto_post_ingest']`（list）
- `kg.merge_suggestion_algorithms: ['name_exact', 'name_fuzzy', 'type_compatible', 'embedding_topk', 'source_chunk_overlap']`（list，可启用任意子集）
- `kg.merge_suggestion_score_threshold: float`（默认 0.85，suggestion 入 store 阈值）
- `kg.merge_suggestion_max_per_scan: int`（默认 100，单次 scan 上限防 PG 连接放大 per § 2.3 P2-31-A）

### 3.3 不做（YAGNI per § 2.4）

## 4. 实施 sub-task 拆分（parallel-friendly）

### Phase A（必须做，并行）

- **#31-A1**：worker lane + queue infrastructure
  - `aperag/cli/indexing_worker.py` 加 11th lane `graph_node_merge_suggestion`
  - `RedisWorkQueue.graph_node_merge_suggestion` queue declaration
  - boundary test 钉「lane 必启动」
  - 推荐 owner：@Bryce / @ziang（熟 task #17 worker CLI + RedisWorkQueue）
- **#31-A2**：suggestion store + Alembic migration
  - 新 `MergeSuggestion` Pydantic schema (`aperag/schema/common.py` 或 `domains/knowledge_graph/schemas.py`)
  - 新 PG table + Alembic migration（chain head）
  - CRUD service layer + boundary test (round-trip + status transition)
  - 推荐 owner：@ziang（熟 graph_curation domain）
- **#31-A3**：dedup detection 算法 + Wave 5 description-NULL 兼容
  - 算法实现 (name_exact / name_fuzzy / type_compatible / embedding_topk / source_chunk_overlap)
  - Pre-filter (per § 2.3 P2-31-A) candidate bucket
  - boundary test 钉「不读 description 字段」+ 各算法 invariant
  - 推荐 owner：@Bryce / @ziang
- **#31-A4**：用户面 review API + UI
  - `POST /api/v2/collections/{id}/merge-suggestions/{id}/accept|reject|dismiss`
  - typed schema 暴露 `MergeSuggestion` shape (per task #71 dongdong typed schema lane)
  - UI（前端 graph viz 页面加 review queue panel）
  - 推荐 owner：@dongdong + @cuiwenbo

### Phase B（依赖 Phase A 实施 PR merged）

- **#31-B1**：CR + boundary test gate（@huangheng）
- **#31-B2**：integration test e2e（@huangzhangshu / @冬柏）
- **#31-B3**：deploy verify（@Planetegg SRE — Helm worker 11th lane env / queue 配置）

### Phase C（数据驱动 follow-up）

- **#31-C1**：算法 ROC / suggestion accuracy benchmark（基于 task #30 PR #1863 graph_extraction benchmark framework 扩展 — 跑 dedup detection 算法 vs 人工 ground truth）
- **#31-C2**：default `kg.merge_suggestion_score_threshold` lock（per task #30 B3 「benchmark 数据决定 default」pattern）

## 5. 验收口径

### 5.1 Phase A 完成标准

- 11th worker lane `graph_node_merge_suggestion` 启动 + boundary test 钉
- `merge_suggestion` table + Alembic migration head chain ✅ +`alembic upgrade head` 跨 backend (PG) 跑过
- dedup detection 5 算法全实施 + capability matrix collection-level config 暴露 typed schema + UI 显示 capability flag (per task #71 dongdong recommend)
- review API accept/reject/dismiss 全实施 + typed schema 暴露 + UI review queue panel
- description-NULL 兼容 boundary test 钉「dedup detection 不读 description 字段」

### 5.2 boundary test gate（CI must pass）

- 现有 G1-G19 + `test_modularization_boundaries.py` + `test_worker_di_parity.py` + `test_no_rerank_in_mcp.py` + `test_graph_window_caps_co_scale.py` + `test_score_normalization_in_vector.py` + 新加 `test_graph_node_merge_suggestion_boundaries.py` 不破坏
- 钉死 invariants:
  - 11th lane `graph_node_merge_suggestion` 必启动（lifespan boundary）
  - API 进程不直接调 `_dedup_scan` (G1+ extension `aperag/api/` 不能 import `_dedup_scan`)
  - background task 只写 `MergeSuggestion.status='pending'`，不直接调 `merge_entities()`
  - dedup detection 不读 `description` 字段（Wave 5 invariant）
  - accept 调用记录 `reviewer_user_id` + `reviewed_at` audit trail

### 5.3 e2e smoke

- 创建 collection + ingest 多 doc（含同名实体跨 doc）→ scan trigger → suggestion 入 store status='pending'
- review API accept → graph 实体 merged + audit trail 记录 + idempotent replay
- review API reject → suggestion status='rejected' + graph 不动
- 多 backend (Neo4j / Nebula / PG) 跨 shape e2e verify (per task #61 ci-flake-policy.md § 2.1 Lite 必绿)

### 5.4 Phase C 数据驱动验收

per task #30 B3 pattern:
- 算法 ROC curve（precision vs recall）跨 entity 数据集 (3+ sample 跨 doc 类型)
- default score threshold lock 由 PM + architect + earayu2 三方 confirm benchmark 数据后定
- 「中等偏保守的最小有效阈值」选择规则（不锁拍脑袋数字）

## 6. CR mandatory checklist

按 `task-17-cr-review-checklist.md` 既有 framework + huangheng PR #1916 + #1924 + #1922 sediment family + 即将 fold 的 task #61 sediment 8 项（Lesson #16 / #13 v3.1 / #12 v9 / #12 v7 ext / #17 / #14 / #13 v3 / 3 deploy capability）应用：

- **Lesson #11 v5**（entry-point migration cross-process parity）— 11th worker lane 加进 `cli/indexing_worker.py` startup + boundary test 钉死，不在 API 进程跑（per task #17 hard cut + task-system-invariants § 2.3）
- **Lesson #12 v4-v9**（CI gate / trust framing / scope walk / 三层 grep / composite key / fake guardrail / first-principles verify）— 全 family 应用
- **Lesson #13 v2.1 + v2.2 + v3 + v3.1**（dual-side rewrite / 不重复事实保证 / deploy manifest dual-side）— 任意 PR 改 typed schema 跟 Pydantic Field 必双侧同步
- **Lesson #14**（架构 invariant 删除多轮迭代收尾）— Wave 5 description-NULL invariant 应用，dedup 算法不能 fallback 到 description
- **Lesson #16**（workflow paths filter dead reference）— 新 worker lane 加 `compat-test.yml` paths filter 同步
- **Lesson #17**（backend 收敛 contract 而非 FE 加 branch / simple-stable family）— suggestion store schema + capability matrix backend 收敛，FE typed schema 仅消费不分支
- **Migration chain 时序**（如 task #31 涉及 Alembic migration — `merge_suggestion` table）
- **简单稳定 + 私有化部署免维护 4 guardrail**

## 7. 关联文档

- earayu2 directives: `#indexing优化` msg=a6186741 (task #31 创建) + msg=2d47e107 (huangzhangshu 重启 catch-up — 团队全员协作)
- task #17 任务系统不变式: [`task-system-invariants.md`](./task-system-invariants.md)（worker lane / API 不拥有执行面 hard gate）
- task #30 graph chunk window spec v1: [`task-30-graph-chunk-window-spec-v1.md`](./task-30-graph-chunk-window-spec-v1.md)（chunks → entity extraction 上游 + B3 default=2 sweet spot pattern）
- task #61 DB adapter compat spec v1: [`task-61-db-adapter-compat-spec-v1.md`](./task-61-db-adapter-compat-spec-v1.md)（graph store contract + capability/degradation declaration pattern）
- task #32 MCP audit spec v1: [`task-32-mcp-audit-spec-v1.md`](./task-32-mcp-audit-spec-v1.md)（typed schema 暴露 capability flag pattern）
- cr-checklist accumulated sediment: [`task-17-cr-review-checklist.md`](./task-17-cr-review-checklist.md)
- Wave 5 description 不再生成 invariant: graph_extractor.py § 4.5 + task #30 A3 prompt v2 schema

## 8. 不阻塞主线

本 spec **不阻塞**:
- task #61 P1 / P2 follow-up implementation 队列（vector adapter capability matrix / graph store deprecation / Helm Nebula first-class / typed schema vector backend exposure）
- huangheng follow-up 子 PR（task #61 sediment 8 项 fold-in cr-checklist § 四）
- task #33 Layer 2 P3 workflow gate
- task #11 GC orphan vector follow-up

---

**起草**：@符炫炜（总架构师）
**日期**：2026-04-30
**版本**：v1（task #31 spec lock 候选；@Weston 架构 CR + earayu2 ratify 后 PM @不穷 按 Phase A / Phase B / Phase C 调度实施 PR）
