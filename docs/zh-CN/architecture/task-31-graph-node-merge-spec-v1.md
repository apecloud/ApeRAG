---
title: task #31 — Graph 节点合并扫描 + 后台建议任务 spec v1
description: ApeRAG graph 重复/近似实体检测 + 异步 suggestion 任务 + 适配 task #17 hard cut 部署 + Wave 5 description 不再生成 invariant
---

# task #31 — Graph 节点合并扫描 + 后台建议任务 spec v1

> earayu2 directive (`#indexing优化` msg=a6186741 + msg=2d47e107 全员协作)：graph 节点合并扫描和后台建议任务，需要随着新的架构（分离部署 indexing）和（不再生成 description）而做对应的设计和修改。

## 1. 现状 inventory（grep 实证）

### 1.1 现有 entity dedup / merge 路径（per Bryce msg=74f33e19 BLOCKER reframe — Wave 7 §K.12.4 全栈已存在）

⚠️ **关键 reframe**：spec 不是「build new」，是「extract sync-inline + fix Wave 5 description-NULL violation + add 三 trigger strategy」。Wave 7 §K.12.4 task #4 **detector + suggestion store + scoring 全栈已存在**:

- `aperag/indexing/merge_candidate_detector.py` — Wave 7 §K.12.4 task #4 detector (Wave 7 task #3 已 wired 进 `GraphModalityWorker.sync` end-of-sync)
- `aperag/domains/knowledge_graph/db/models.py:107` — `GraphCurationSuggestion` table + `GraphCurationSuggestionStatus` enum (`PENDING/ACCEPTED/REJECTED/DISMISSED` 已就位)
- `aperag/graph_curation/candidate_generation.py` — `build_candidate_pairs()` 算法 (vector ANN + name/type/description signal scoring)
- `aperag/graph_curation/dto.py::CurationEntity` — entity wrapper
- `MergeCandidateDetector.AUTO_DETECT_SOURCE = "auto_detect"` discriminator on `GraphCurationRun.config_json["source"]` — admin UI split 已设计完成 (per K.12.4 amend2 ratify msg=6ab89fbb)
- `DEFAULT_VECTOR_TOP_K = 4` + `DEFAULT_VECTOR_SCORE_THRESHOLD = 0.72` (Bryce PR #1930 commit `873a7541` 已 docstring annotated 是 cosine-tuned `[0, 1]` normalized scale)
- `aperag/graph_curation/service.py` `merge_entities()` API（用户面 / agent 触发 manual merge）
- `aperag/graph_curation/alias_map.py` `resolve_canonical()` per-query alias resolution（**Planetegg msg=db7fb085** + Singapore 实证 P2-S1：per-node `asyncio.gather` PG 连接池吃紧放大压力）

**当前 gap**（task #31 真 scope）:
1. 跑在 `GraphModalityWorker.sync` end-of-sync（**API 进程或 sync graph_facts/graph_vectors lane 内联**）— 违反 task #17 hard cut + task-system-invariants § 2.3 hard gate
2. `candidate_generation.py:43/179-181/196-197` 三处读 / scoring weight `description` — 违反 Wave 5 description-NULL invariant
3. `merge_entities` apply 路径 (`lineage_merge.py`) 调 LLM unified description / 写 vector — 违反 Wave 5 invariant (per huangzhangshu msg=c9f81309 BLOCKER 1)
4. 仅 1 trigger strategy（post-sync 自动）— 缺 cron / manual / auto_post_ingest 三策略 collection-level 配置
5. status enum 缺 apply/applying/failed 语义（per huangzhangshu msg=c9f81309 BLOCKER 2）— 测试无法区分用户决策 vs worker 应用 vs 失败

### 1.2 现有 background task 路径

`aperag/indexing/`：
- `cleanup.py` cleanup task（`Document.status=DELETED + gmt_deleted` SoT，task #17 hard cut 后 worker 独立 deployment）
- `reconciler.py` 5-stage reconciler (PENDING dispatch / FAILED retry / RUNNING reclaim / Stuck parse / Graph vectors enqueue+stale)
- 无 **graph node merge 后台 task / queue lane** — 当前 graph 异步任务只有 `graph_facts` / `graph_vectors` extraction

### 1.3 task #17 hard cut deployment 状态

`aperag/cli/indexing_worker.py` 启动 10 lane: parse / vector / fulltext / graph / graph_facts / graph_vectors / summary / vision / reconciler / cleanup（per `task-system-invariants.md` § 1.1）— 任何 graph node merge background task 必须 **加进 worker lane list**，不能跑在 API 进程（per task-system-invariants § 2.3 「API 不拥有重型执行面」hard gate）。

### 1.4 Wave 5 description 不再生成 invariant

`aperag/indexing/graph_extractor.py`（task #30 A3 prompt v2）已移除 description regeneration（Wave 5 task #5 facts/vectors split 后老 graph 失败 FAILED 行直接归一化删除，新 entity record 不再写 description — per `task-30-graph-chunk-window-spec-v1.md` § 4.5 invariant）— 节点合并 suggestion 必须 **基于 entity name / type / source_chunk_ids 跨 doc 共享性，不依赖 description 文本相似度**（description 字段 NULL）。

### 1.5 graph store contract（task #61 close 锁定，per Weston msg=2b441dc2 NIT 修订）

`aperag/indexing/graph_storage/{neo4j,nebula,postgres}.py` 跨 3 backend `LineageGraphStore` Protocol contract 已 task #61 P0/P1 锁定:
- `bulk_upsert_entity_with_lineage_parts` cross-backend boundary test 38 cases (PR #1927 `9c94cbc1`)
- alias / source_chunk_ids list-typed schema (task #61 spec § 1.2 + Wave 5 multi-chunk provenance)

⚠️ **修正引用**（per Weston msg=2b441dc2 NIT）：`LineageGraphStore` Protocol **不含** `merge_entities` 方法 — 现有 `aperag/graph_curation/lineage_merge.py::LineageEntityMerger` (合并 entity + 写 lineage) + `aperag/graph_curation/service.py::GraphCurationService` (orchestrator) 使用 `LineageGraphStore` primitives 实现 merge。task #31 复用这两层但需要 description-free variant (per § 3.1.5)。

task #31 不引入新 graph store method — 复用 task #61 locked contract + LineageEntityMerger / GraphCurationService primitive 复用。

## 2. 缺口识别（按 severity）

按 Weston msg=85e527e3 + msg=d0e00405 三层框架 + earayu2 directive 三 sub-task 边界:

### 2.1 P0（必须做 — 影响上层正确性）

- **P0-31-A** background suggestion task lane 缺失：当前 `aperag/cli/indexing_worker.py` 10 lane 不含 `graph_curation_run` lane — 必须新加 worker lane（独立 queue family 不污染 `Modality`，per § 3.1.1 + task #17 hard cut + task-system-invariants § 2.3）；suggestion task 不能跑 API 进程
- **P0-31-B** suggestion 必须 **可审阅 + 不静默 destructive merge**（per Weston msg=d0e00405 边界 3）：suggestion 写入独立 store（not auto-applied），用户面 (UI) 显式 review + accept/reject，不允许 background task 直接调 `merge_entities()` 静默改 graph
- **P0-31-C** Wave 5 description-NULL 兼容（per § 1.4 invariant）：dedup 算法 input **不能依赖 entity description 文本相似度** — 必须基于 entity name / type / source_chunk_ids overlap / vector embedding 跨 doc 共享性 surface 候选

### 2.2 P1（允许差异但显式 declaration）

- **P1-31-A** dedup detection 算法 capability matrix：name exact match (case-insensitive normalize) / name fuzzy (Levenshtein / Jaro-Winkler) / type compatibility / vector embedding similarity / source_chunk overlap — 各算法 trigger 条件 + threshold 显式 declare collection-level config
- **P1-31-B** suggestion store schema：**复用现有** `GraphCurationSuggestion` table (`aperag/domains/knowledge_graph/db/models.py:107`) — 仅 extend status enum 4 新 value（`APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED`，per § 3.1.6 7-state machine）+ 加 `evidence_refs` field（per task #61 evidence_refs 模式）。**不引入新 `merge_suggestion` table**（per Bryce + Weston BLOCKER 1，避免 schema 漂移 + Lesson #14 multi-iteration cleanup family）。
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
- **v1 不实施 `entity_type` 归一化合并**（per § 3.1.3 entity_type scope lock — `entity_type_alias` suggestion kind 移 Phase B / P1 follow-up，独立设计 store/API/migration/UI 后再启动；v1 仅作为 dedup score compatibility signal，per ziang/dongdong/Weston 三方 converge）

## 3. 设计方向（task #31 主线）

### 3.1 必须做（Hard scope per Weston msg=d0e00405 4 边界 + earayu2 directive）

#### 3.1.1 后台 worker lane 加入 task #17 hard cut deployment（per ziang msg=92321bcc + Bryce msg=4c23f87e BLOCKER 1 reframe — **独立 queue family 不污染 Modality/DocumentIndex**）

⚠️ **关键 invariant**：现有 `WorkQueue.push/pop` keyed by `Modality`，Redis key `q:indexing:<modality>` (`orchestrator.py:113-136/220-256`)，`_entrypoint(Modality, concurrency)` 跟 `ModalityWorkerFactory` + `DocumentIndex` payload 绑定 (`orchestrator.py:749-787`)。`Modality` 是 **per-document state machine** (`models.py:47+`)，`graph_curation_run` 是 **per-collection / per-run** job — 强行复用 `Modality` 会污染 `DocumentIndex` / reconciler / index_state。

新增 **独立 queue family** (per ziang BLOCKER 1):
- 独立 lane name `graph_curation_run`（lane symbolic name appears in indexing-worker task list — 不 hardcode 计数 per cuiwenbo + Bryce + 冬柏 NIT）
- 独立 Redis key `q:graph_curation_run`（不在 `q:indexing:<modality>` family 内）
- 独立 push/pop API: `push_graph_curation_run(run_id, collection_id)` / `pop_graph_curation_run()`
- worker CLI 新增 **独立 loop task**（不走 `_entrypoint(Modality, ...)`），不引入 `Modality.GRAPH_NODE_MERGE_SUGGESTION` enum value
- 仍 reuse RedisWorkQueue / RedisQuotaBackend infrastructure 底层（Redis client / Lua atomic / quota partitioning），但 **state 隔离 / payload 隔离 / lane state 隔离**
- API 进程 **不直接调用 dedup scan**（per `task-system-invariants` § 2.3 hard gate）— API 仅 enqueue run_id + 读 suggestion store
- boundary test：`tests/boundaries/test_indexing_worker_lanes.py` 加钉「`graph_curation_run` lane symbolic name appears in indexing-worker task list **+ API deployment has no executor/import path**」（双侧 lane assertion，per ziang NIT + cuiwenbo NIT 2）

#### 3.1.1.b trigger 策略 reconcile 现有 sync detector 与新 worker（per ziang msg=92321bcc + Bryce msg=4c23f87e BLOCKER 2 — 三策略统一 queue path）

现有两条生成路径必须 reconcile，不允许 Phase A 实施成三套独立执行路径:
- 现有 `GraphModalityWorker.sync` 末尾 `MergeCandidateDetector.detect_for_sync()` 写 PENDING (`graph.py:1687-1698`)
- 现有 `GraphCurationService.start_run()` 走 `asyncio.to_thread(generate_graph_curation_run_task)` API fire-and-forget (`service.py:107-123`)

三 trigger 策略 lock：
- **manual / full sweep**: 复用现有 `GraphCurationRun`，API `POST /graphs/merge-suggestions` 创 run + **enqueue run_id 到新独立 queue family** (`q:graph_curation_run`)。worker pop 后调 `generate_graph_curation_run_task` integration path（必须 description-free per § 3.1.5）。
- **auto_post_ingest**: 现有 `GraphModalityWorker.sync` 末尾 `detect_for_sync()` 保留 write-only quick path（仅写 PENDING），但**必须同步做 description-free 修复**（per § 3.1.5 + § 3.1.7）— 不能绕过 Wave 5 invariant。auto_post_ingest 不走 worker queue family（避免双路径），但 detect_for_sync 必须 fix Wave 5 violation。
- **cron**: scheduler 创 `GraphCurationRun`（如同 manual） + enqueue run_id 到 `q:graph_curation_run` queue。复用同 worker pop loop。

不允许 Phase A 实施成三套独立 path — manual + cron 共享 enqueue → worker pop → integration path；auto_post_ingest 是 sync inline write-only 但必修 description-free invariant。

#### 3.1.2 suggestion store + reviewable workflow（per Weston msg=d0e00405 边界 3 + dongdong msg=11813333 BLOCKER read contract fold + Bryce msg=74f33e19 + Weston msg=2b441dc2 BLOCKER 1 reuse existing table）

- **复用现有** PG table `GraphCurationSuggestion` (`aperag/domains/knowledge_graph/db/models.py:107`) — **不引入新 `merge_suggestion` table**（per Bryce + Weston BLOCKER 1，避免 schema 漂移 + Lesson #14 multi-iteration cleanup family）。Phase A2 仅 extend status enum 4 新 value（`APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED`）+ 加 `evidence_refs` field（per task #61 evidence_refs 模式）。
- background task **只写 suggestion**（pending status）— 不调 `LineageEntityMerger` apply
- **read API**（per dongdong msg=11813333 BLOCKER + cuiwenbo msg=61800dd6 NIT 1 + dongdong msg=c4d3ae32 收窄 — **复用 FE 现有 endpoint 不另起 path**）：
  - **复用** `GET /api/v2/collections/{id}/graphs/merge-suggestions?status=pending&limit=&cursor=` — list 分页 + status filter，跟 FE `client-api.ts:302` 现有 caller align（不破 typed schema）
  - **复用** `GET /api/v2/collections/{id}/graphs/merge-suggestions/{suggestion_id}` — detail full record
  - typed schema 扩展现有 `MergeSuggestionItem` (`web/src/features/knowledge-graph/types.ts:80`) 为 `MergeSuggestionView` display-ready shape：`id` + `collection_id` + `suggestion_batch_id` + `candidate_entity_ids` + `confidence_score` + `merge_reason` + `suggested_target_entity` + `evidence_refs[].document_id/chunk_id/parse_version`（per task #61 evidence_refs 模式新加）+ `affected_doc_count` + `created_at` + `status` + `reviewer_user_id` + `reviewed_at`（FE 现有 `operated_at` rename）
- **action API**（**复用** FE 现有 endpoint per cuiwenbo msg=61800dd6 NIT 1 + dongdong msg=c4d3ae32）：
  - **复用** `POST /api/v2/collections/{id}/graphs/merge-suggestions/{suggestion_id}/action` body `{action: 'accept'|'reject'|'dismiss'}` — 不引入新 path 避免 typed schema 破坏 + FE 三处 caller (`client-api.ts:329`) 重写
  - accept 时 enqueue worker → status `pending → apply_pending → applying → applied | apply_failed`（per § 3.1.6 状态机；新 async path 写 `apply_pending`，**不写 `accepted`** — `accepted` 是 legacy sync handle_action terminal value，保留 backward-compat read-only）
  - reject/dismiss 仅改 status `pending → rejected | dismissed`，不 enqueue worker
  - 返回 updated `MergeSuggestionView`（不要 204 + refetch）
  - `SUGGESTION_ACTIONS` const FE 同步扩展 `dismiss` (FE 现有仅 `accept` | `reject`)
- UI 状态机 contract：pending / apply_pending / applying / applied / apply_failed / rejected / dismissed 7 新态 + legacy `accepted` read-only display + 空态 + 错误态 typed schema 显式（per task #61 backend 收敛 contract pattern）— FE 仅消费不基于 backend 类型分支；legacy `accepted` UI 显示等价于「已应用」终态（历史 sync 路径）
- review audit trail：accept 调用记录 `reviewer_user_id` + `reviewed_at`，可回滚（rollback API 仅 admin role）
- AlembicMigration **不建新 table** — 仅 extend `graph_curation_suggestions` table（status enum 加 4 新 value `APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED`，**现有 `ACCEPTED` 保留作 legacy terminal/back-compat read-only value**，新 async path 不再写 — 历史 `ACCEPTED` 在 sync `handle_action()` 末尾代表「merge 已执行完成」terminal semantic（per `aperag/graph_curation/service.py:534` 实证），新 async path 用 `apply_pending` 表示「用户已批准但未 apply」避免同名不同义；per dongdong msg=e7d7600a + Weston msg=013fdc47/14859580 + ziang msg=378455ad/c2228ba1 + dongdong msg=ceca6063 集体 converge）+ 加 `evidence_refs` field per task #61 evidence_refs 模式，chain 在 latest head 后

#### 3.1.3 Wave 5 description-NULL 兼容 dedup 算法 + entity_type scope lock（per ziang msg=d6d9dc3c + dongdong msg=83783bc6 + Weston msg=78ab2267 三方 converge）

dedup detection 输入仅:
- `EntityRecord.name`（normalize lowercase + trim + Unicode NFC）— **merge target 主体**
- `EntityRecord.entity_type` — **仅 compatibility / penalty signal，不是 merge target**（per ziang/dongdong/Weston 三方 converge — 详见下方 entity_type scope lock）
- `EntityRecord.source_chunk_ids` (list, task #30 A3 schema)
- entity vector embedding（如已有 — task #30 A3 vector 路径）

**不依赖** `EntityRecord.description`（Wave 5 后永远 NULL）。如算法层 fallback 到 description 文本对比 → boundary test 钉死「dedup detection 不读 description 字段」防 silent regression。

⚠️ **entity_type scope lock**（PM msg=05be0b52 question + ziang/dongdong/Weston 三方 converge）:

`entity_type` 是 LLM 自动生成的 label/属性，可能产生大量近似/重复 type（如「人物 / 人员 / 人」）。task #31 v1 **不把 `entity_type` 本身作为 merge 对象**，但 spec 显式覆盖三层边界:

1. **v1 仍以 entity name 为主 merge target**：`entity_type` 在 dedup detection 仅作为 compatibility / conflict signal — type 近似加分（如 cosine sim ≥ threshold）/ type 冲突降分；**不允许仅因为 type 近似就把同类型不同 entity 拉到同一候选池**（false positive 太高 per Weston msg=78ab2267 layer 1）
2. **merge suggestion 必须容忍 type 近似**：同名 / 强证据候选不应因 `人物 / 人员 / 人` 不完全相等被过滤掉；suggestion payload `MergeSuggestionView` 显式展示 `observed_types: list[str]` + `type_conflict: bool` + 可选 `suggested_entity_type: Optional[str]`，accept 时 canonical type 选择需 audit trail（per Weston layer 2）
3. **`entity_type` 归一化作独立 suggestion kind — Phase B / P1 follow-up**（不在 v1 Phase A 范围）：
   - 引入 `GraphCurationSuggestion.suggestion_kind: Literal['entity_merge', 'entity_type_alias']`（v1 仅写 `entity_merge`，Phase B 加 `entity_type_alias`）
   - `entity_type_alias` accept 仅更新 type 别名 / 规范化映射或批量 type rewrite，**不触发实体节点合并**
   - `entity_type_alias` 同样禁止 auto-apply — 必须 reviewable（per Weston layer 3 + earayu2 「保留人工审核」directive）
   - `entity_type_alias` UI / typed schema / migration / blast radius 较大（涉及 graph filter / legend / 颜色分组 / 历史数据 — per dongdong msg=83783bc6），等 task #31 v1 name-level suggestion 跑稳 + type 频次数据后再设计独立 store/API/migration（per ziang msg=d6d9dc3c）

**v1 Phase A 验收**: dedup 算法读 `entity_type` 仅用于 score signal，不作为 merge action 主语；boundary test 钉「v1 Phase A 不写 `suggestion_kind='entity_type_alias'` 入 store」防 scope creep。

#### 3.1.4 graph store contract 复用（per Weston msg=d0e00405 边界 4 + msg=2b441dc2 BLOCKER 5 — `LineageGraphStore.merge_entities` 不存在）

⚠️ **修正引用**：`LineageGraphStore` Protocol **不含** `merge_entities` 方法（per § 1.5）。merge apply 路径由 `aperag/graph_curation/lineage_merge.py::LineageEntityMerger` + `aperag/graph_curation/service.py::GraphCurationService` 在 application layer 基于 `LineageGraphStore` primitives 实现。

apply 路径复用 task #61 locked invariants:
- `bulk_upsert_entity_with_lineage_parts` atomicity（PR #1927 38 cases pin）— graph store layer primitive
- `LineageEntityMerger` description-free apply path（per § 3.1.5 抽 `merge_entities_apply_description_free()` variant）使用上述 primitives 实现合并 + 写 lineage — application layer 行为
- cross-backend boundary test 钉 `LineageEntityMerger` 行为契约（不钉 `LineageGraphStore.merge_entities`）— 加进 `tests/integration/compat/test_lineage_graph_compat.py` 跨 Neo4j/Nebula/PG 三 backend description-free merge 行为
- replay idempotent（前一次 accept fail 后 retry 不会重复合并）

不引入新 graph store method — 全 reuse task #61 locked Protocol primitives。

#### 3.1.5 description-free refactor — 6 call sites P0 enumeration（per huangzhangshu msg=c9f81309 BLOCKER 1 + Bryce msg=74f33e19 P1 SCOPE + ziang msg=92321bcc P1 + Weston msg=2b441dc2 BLOCKER 2 多源累计）

⚠️ **Wave 5 description-NULL hard invariant 修复**：现有 graph_curation 全栈多处依赖 `description` 字段（detection 输入 / scoring weight / DTO snapshot / accept LLM unified / vector embedding），`graph_extractor.py` Wave 5 后 `description` 永远 NULL — 必须 P0 refactor **6 call sites** 全 fix:

| # | 路径 | 修法 |
| --- | --- | --- |
| 1 | `aperag/graph_curation/candidate_generation.py:43` | `entity_snapshot()` 删 `entity.description` read |
| 2 | `aperag/graph_curation/candidate_generation.py:179-181` | 删 `description_overlap = _jaccard(_tokens(left.description), _tokens(right.description))` + `if description_overlap >= 0.2` |
| 3 | `aperag/graph_curation/candidate_generation.py:196-197` | 删 `score += min(float(signals["description_token_overlap"]) * 0.20, 0.15)` scoring weight |
| 4 | `aperag/graph_curation/dto.py:59-65` + `:101-105` | `CurationEntity.description` 字段 input 改成不依赖（基于 source_chunk_ids 兼容路径） |
| 5 | `aperag/indexing/merge_candidate_detector.py:257-284` | `_description_text_for_scoring()` 填 legacy DTO description — 改成 entity name + type embedding query |
| 6 | `aperag/indexing/merge_candidate_detector.py:322-328` | snapshot 写 description — 改成 不写 description 字段 |

外加 accept apply 路径 description-free variant:
- `aperag/graph_curation/lineage_merge.py:246-317` 抽 `merge_entities_apply_description_free()` variant — 不调 LLM unified description / compactor / `__curation_merge__` description part / vector embedding
- accept worker 仅调 description-free variant — 老 `lineage_merge.py` 路径保留作 manual API 兼容（标 deprecation Lesson #14 multi-iteration cleanup follow-up）

boundary test grep gate（per § 5.2）:
- 6 call sites 全部 grep zero match `description` read（type=`type=python` glob `aperag/graph_curation/**` + `aperag/indexing/merge_candidate_detector.py`）
- worker accept lane 模块 import allowlist 不含 `lineage_merge.unified_description` / `compactor` / `LLM helper` 类

#### 3.1.6 apply 状态机（per huangzhangshu msg=c9f81309 BLOCKER 2）

`GraphCurationSuggestion.status` enum 扩展为完整状态机（区分用户决策 vs worker 应用 vs 失败）:

| status | 转移条件 |
| --- | --- |
| `pending` | scan 写入新 suggestion |
| `dismissed` | 用户 dismiss action（不 enqueue worker） |
| `rejected` | 用户 reject action（不 enqueue worker） |
| `apply_pending` | 用户 accept action — enqueue worker（新 async path 决策态，不复用 `accepted` 避免历史 semantic 漂移） |
| `applying` | worker 拿到 task 开始 apply |
| `applied` | worker 成功 apply graph + audit trail 写入 |
| `apply_failed` | worker apply 失败（保留 retry，retry 次数 cap） |

⚠️ **legacy `ACCEPTED` semantic note**（per Weston msg=013fdc47/14859580 + ziang msg=378455ad/c2228ba1 + dongdong msg=ceca6063 集体 converge + architect 拍板 Option B）：

现有 `aperag/graph_curation/service.py:534` sync `handle_action()` 末尾 `suggestion.status = GraphCurationSuggestionStatus.ACCEPTED`（merge 已 sync 执行完成后写入），ACCEPTED 在历史代码中是 **terminal status = 「merge 已执行完成」**。新 async path 如果复用 `ACCEPTED` 表示「已批准但未 apply」会让旧数据和新数据同名不同义 → 引入新 enum value `apply_pending` 表示新 async 决策态，**`ACCEPTED` 保留作 legacy terminal/back-compat read-only value，新 async path 不再写**（FE typed schema 显示兼容 + DB 存在但新代码 zero-write）。

跟 cuiwenbo msg=61800dd6 NIT 2 FE 现有 enum (`PENDING/ACCEPTED/REJECTED/EXPIRED`) align 选择：
- spec lock **lowercase** + 新加 `dismissed/apply_pending/applying/applied/apply_failed` 5 enum value
- FE typed schema 同步扩展 `MergeSuggestionStatus` (Lesson #13 v3 dual-side rewrite + Lesson #14 multi-iteration cleanup — `EXPIRED` 老值保留作 backward compat 历史 placeholder，新代码不再写入；`ACCEPTED` 同样保留作 legacy terminal read-only)
- Migration chain 时序：PG enum 加 5 新 value `DISMISSED/APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED`（`alembic upgrade head` 跨 backend 跑过）— `ACCEPTED` 保留 legacy semantic

测试可区分「用户已批准（apply_pending）」「worker 应用中（applying）」「worker 已应用（applied）」「应用失败待重试（apply_failed）」四状态 + 历史 ACCEPTED legacy read。

### 3.2 P1 实施（capability matrix declaration）

per § 2.2 三 strategy + 算法 capability matrix collection-level config（**配置 schema path lock**，per dongdong msg=11813333 NIT 1 + 跟 task #30 / task #61 既有 schema path 对齐：`collection.config.knowledge_graph_config.merge_suggestion_*` — `kg.*` 仅 spec 内文 shorthand）:
- `merge_suggestion_strategy: list[Literal['cron', 'manual', 'auto_post_ingest']]`
- `merge_suggestion_algorithms: list[Literal['name_exact', 'name_fuzzy', 'type_compatible', 'embedding_topk', 'source_chunk_overlap']]`（可启用任意子集）
- `merge_suggestion_score_threshold: Optional[float]`（默认 0.85，suggestion 入 store 阈值，**初始 default = 0.85**, Phase C benchmark 后 lock per task #30 B3 sweet spot pattern）
- `merge_suggestion_max_per_scan: Optional[int]`（默认 100，单次 scan 上限防 PG 连接放大 per § 2.3 P2-31-A）

### 3.3 不做（YAGNI per § 2.4）

## 4. 实施 sub-task 拆分（parallel-friendly）

### Phase A（必须做，并行 — extract / fix / extend，**不 build new**）

- **#31-A1 (extract)**：worker lane + 独立 queue family 从 sync-inline 抽出（per § 3.1.1 + § 3.1.1.b）
  - `aperag/cli/indexing_worker.py` 加 **独立 lane `graph_curation_run`**（独立 loop task，不走 `_entrypoint(Modality, ...)`，不引入 `Modality.GRAPH_NODE_MERGE_SUGGESTION` enum value — per ziang BLOCKER 1）
  - 独立 Redis key `q:graph_curation_run`（不在 `q:indexing:<modality>` family 内）+ 独立 push/pop API: `push_graph_curation_run(run_id, collection_id)` / `pop_graph_curation_run()`
  - 现有 `aperag/graph_curation/service.py:114-123` `asyncio.create_task(asyncio.to_thread(generate_graph_curation_run_task, ...))` API 进程同步路径 → 改成 `push_graph_curation_run(run_id)` API 进程仅 enqueue
  - worker pop loop **manual / cron full sweep path** 调用现有 `generate_graph_curation_run_task` integration path（per § 3.1.1.b — 不调 `MergeCandidateDetector.detect_for_sync()`，那是 auto_post_ingest sync inline quick path）
  - **auto_post_ingest path** 保留 `GraphModalityWorker.sync` 末尾 `MergeCandidateDetector.detect_for_sync()` 写 PENDING（write-only，不入 worker queue），但**必须同步做 description-free 修复**（per § 3.1.5）— 不绕过 Wave 5 invariant
  - boundary test 钉 **lane name `graph_curation_run` symbolic appearance** in indexing-worker task list **+ API deployment has no executor/import path**（双侧 lane assertion，`tests/boundaries/test_app_lifespan_no_workers.py` extend per 冬柏 msg=e92af542 推荐 (a)）
  - 推荐 owner：@Bryce / @ziang
- **#31-A2 (extend)**：复用 `GraphCurationSuggestion` 现有 table + extend status enum
  - **不引入新 table** (per Bryce + Weston BLOCKER) — 复用 `aperag/domains/knowledge_graph/db/models.py:107` `GraphCurationSuggestion`
  - status enum extend：现有 `PENDING/ACCEPTED/REJECTED/DISMISSED/EXPIRED/SUPERSEDED` + 新加 `APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED`（per § 3.1.6 状态机；新 async path 写 `APPLY_PENDING` 不写 `ACCEPTED`，**`ACCEPTED` 保留作 legacy terminal/back-compat read-only value** — 历史 sync `handle_action()` 末尾 set ACCEPTED 代表「merge 已执行完成」terminal semantic per `service.py:534` 实证）
  - PG enum migration + FE typed schema sync (Lesson #13 v3 dual-side rewrite + Lesson #14 multi-iteration cleanup — `ACCEPTED` / `EXPIRED` / `SUPERSEDED` 老值保留作 backward compat read-only)
  - boundary test (status transition + apply state machine + `ACCEPTED` legacy read-only — 新代码 zero-write assertion)
  - 推荐 owner：@ziang（熟 graph_curation domain + Wave 7 §K.12.4）
- **#31-A3 (fix)**：description-free refactor — Wave 5 invariant 修复 **6 个 detector/snapshot call site + 1 个 apply path**（per § 3.1.5 enumeration，不能写 4 处 — Lesson #14 multi-iteration cleanup 自身案例）
  - **detector / snapshot 6 call sites**：
    1. `aperag/graph_curation/candidate_generation.py:43` `entity_snapshot()` 删 `entity.description` read（per Bryce P1 SCOPE）
    2. `aperag/graph_curation/candidate_generation.py:179-181` 删 `description_overlap = _jaccard(_tokens(left.description), _tokens(right.description))` + `if description_overlap >= 0.2`
    3. `aperag/graph_curation/candidate_generation.py:196-197` 删 `score += min(float(signals["description_token_overlap"]) * 0.20, 0.15)` scoring weight
    4. `aperag/graph_curation/dto.py:59-65` + `:101-105` `CurationEntity.description` 字段 input 改成不依赖（基于 source_chunk_ids 兼容路径）
    5. `aperag/indexing/merge_candidate_detector.py:257-284` `_description_text_for_scoring()` 填 legacy DTO description — 改成 entity name + type embedding query
    6. `aperag/indexing/merge_candidate_detector.py:322-328` snapshot 写 description — 改成 不写 description 字段
  - **apply path description-free variant**：`aperag/graph_curation/lineage_merge.py:246-317` 抽 `merge_entities_apply_description_free()` variant — 不调 LLM unified description / compactor / `__curation_merge__` description part / vector embedding（per § 3.1.5）；老 `lineage_merge.py` 路径保留作 manual API 兼容（标 deprecation Lesson #14 multi-iteration cleanup follow-up）
  - boundary test grep gate：worker accept lane allowlist 不含 `description`/`compactor`/`unified_description` 等 LLM helper module；6 detector/snapshot call sites 全部 grep zero match `description` read（type=`type=python` glob `aperag/graph_curation/**` + `aperag/indexing/merge_candidate_detector.py`）
  - 推荐 owner：@Bryce / @ziang（熟 graph_curation + indexing 双侧）
- **#31-A4 (reuse)**：复用现有 `/graphs/merge-suggestions` endpoint + extend
  - **不引入新 path** (per cuiwenbo + dongdong + Weston BLOCKER) — 复用 `aperag/domains/knowledge_graph/api/routes.py:187/213/242` 现有 `POST` (run) + `GET` (list/read) + `POST .../action` endpoints
  - extend list/detail typed schema 加 `evidence_refs`（per task #61 evidence_refs 模式）+ `affected_doc_count` + 新 status values
  - extend `SUGGESTION_ACTIONS` 加 `dismiss`（FE 现仅 `accept/reject`）
  - cron + manual + auto_post_ingest 三 trigger strategy（manual 复用 `POST` run）
  - 推荐 owner：@dongdong + @cuiwenbo

### Phase B（依赖 Phase A 实施 PR merged）

- **#31-B1**：CR + boundary test gate（@huangheng — 含 description-free grep gate + Lesson framework cross-link）
- **#31-B2**：integration test e2e（@huangzhangshu testing primary / @冬柏 peer）+ **`LineageEntityMerger` cross-backend boundary test 加进 `tests/integration/compat/test_lineage_graph_compat.py`**（per Bryce P1 GAP — 钉跨 Neo4j/Nebula/PG 三 backend description-free merge 行为契约）
- **#31-B3**：deploy verify（@Planetegg SRE — `helm template --set neo4j.enabled=true` 验证 indexing-worker 包含 **`graph_curation_run` lane / `q:graph_curation_run` queue 配置 symbolic appearance**；API deployment 不新增 graph curation executor / import path（symbolic lane assertion，不 hardcode 11th 计数）per Planetegg msg=305d7843 + msg=6b63b7e9 Helm render gate）

### Phase C（数据驱动 follow-up）

- **#31-C1**：算法 ROC / suggestion accuracy benchmark（基于 task #30 PR #1863 graph_extraction benchmark framework 扩展 — 跑 dedup detection 算法 vs 人工 ground truth）
- **#31-C2**：default `kg.merge_suggestion_score_threshold` lock（per task #30 B3 「benchmark 数据决定 default」pattern）
- **#31-C3**：`entity_type_alias` suggestion kind follow-up（per § 3.1.3 entity_type scope lock + § 2.4 YAGNI — Phase B / P1 启动条件：task #31 v1 name-level suggestion 跑稳 + 收集 type 频次数据 + 单独 spec design store/API/migration/UI 后再启动，per ziang msg=d6d9dc3c + dongdong msg=83783bc6 + Weston msg=78ab2267 converge）

## 5. 验收口径

### 5.1 Phase A 完成标准

- worker lane `graph_curation_run` 启动 + boundary test 钉**lane name symbolic appearance**（不 hardcode 计数，per Bryce + cuiwenbo + 冬柏 NIT）+ 独立 queue family `q:graph_curation_run`（不污染 `Modality`，per § 3.1.1）
- 复用 `GraphCurationSuggestion` table + Alembic migration extend status enum 4 新 value（`APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED`），跨 backend (PG) `alembic upgrade head` 跑过 — **不建新 `merge_suggestion` table**
- dedup detection 复用现有 `MergeCandidateDetector` + extend 5 算法 capability matrix + collection-level config 暴露 typed schema
- 复用现有 `/graphs/merge-suggestions` endpoint + extend `SUGGESTION_ACTIONS` 加 `dismiss` + UI review queue panel
- description-free refactor 完整：**6 个 detector/snapshot call site + 1 个 apply path** (candidate_generation.py:43/179-181/196-197 + dto.py:59-65/101-105 + merge_candidate_detector.py:257-284 + merge_candidate_detector.py:322-328 + lineage_merge.py:246-317 apply variant) 全 fix + boundary test grep gate（per § 3.1.5）

### 5.2 boundary test gate（CI must pass）

⚠️ **scan-generation invariants vs async accept-apply state machine invariants 分开钉死**（per huangzhangshu msg=f8213410 + ziang BLOCKER 清单 — 两套 surface 必须分别 boundary test，不能合在一处）:

#### 5.2.a scan-generation 阶段 invariants（worker pop → detect → 写 PENDING）

- 现有 G1-G19 + `test_modularization_boundaries.py` + `test_worker_di_parity.py` + `test_no_rerank_in_mcp.py` + `test_graph_window_caps_co_scale.py` + `test_score_normalization_in_vector.py` + 新加 `test_graph_curation_run_boundaries.py` 不破坏
- **lane symbolic dual-side**：lane name `graph_curation_run` appears in indexing-worker task list（`tests/boundaries/test_app_lifespan_no_workers.py` extend 正向）+ API deployment **没有** graph curation executor / import path（负向，G1+ extension `aperag/api/` 不能 import graph curation worker entry / `MergeCandidateDetector` 等执行类）
- **independent queue family**：`q:graph_curation_run` 独立 Redis key，不在 `q:indexing:<modality>` family 内（不污染 `Modality` enum / `DocumentIndex` payload）
- **description-free 6 call sites**：scan/snapshot 路径 grep zero match `description` read（type=`type=python` glob `aperag/graph_curation/**` + `aperag/indexing/merge_candidate_detector.py`），per § 3.1.5 + Wave 5 invariant
- **trigger split**：manual / cron 走 `push_graph_curation_run` → worker pop → `generate_graph_curation_run_task`；auto_post_ingest 走 `GraphModalityWorker.sync` 末尾 `detect_for_sync()` write-only quick path（同 description-free invariant）
- **safe-only write**：scan-generation worker 只写 `GraphCurationSuggestion.status='pending'`，**不直接调 `LineageEntityMerger.merge_entities_apply_*()`**（不允许 silent destructive merge per Weston 边界 3）

#### 5.2.b async accept-apply 状态机 invariants（accept → enqueue worker → apply）

- **7-state machine 完整覆盖**：`pending → dismissed | rejected | apply_pending → applying → applied | apply_failed`（per § 3.1.6；新 async path 写 `apply_pending`，**不写 `accepted`** — `accepted` 是 legacy sync `handle_action()` terminal value），boundary test 钉每条 transition 边 + 新代码 zero-write `accepted` assertion
- **enum lowercase + dual-side**：PG enum 4 新 value `APPLY_PENDING/APPLYING/APPLIED/APPLY_FAILED` + FE typed schema `MergeSuggestionStatus` 同步扩展（Lesson #13 v3 dual-side rewrite + Lesson #14 multi-iteration cleanup — `ACCEPTED/EXPIRED/SUPERSEDED` 老值保留作 backward compat read-only）
- **legacy `ACCEPTED` zero-write gate**：grep gate 钉 `aperag/graph_curation/` 新 async path 模块（`worker.py` / new merge worker）import allowlist 不含 `GraphCurationSuggestionStatus.ACCEPTED` 写入 — 仅 sync legacy `handle_action()` 路径允许（per Weston msg=013fdc47 + ziang msg=378455ad 集体 converge）
- **description-free apply variant**：accept worker 仅调 `merge_entities_apply_description_free()` variant — allowlist 不含 `description`/`compactor`/`unified_description` 等 LLM helper module
- **cross-backend apply contract**：`LineageEntityMerger` description-free 行为加进 `tests/integration/compat/test_lineage_graph_compat.py`（跨 Neo4j/Nebula/PG 三 backend，per § 3.1.4）
- **audit trail**：accept 调用记录 `GraphCurationSuggestion.reviewer_user_id` + `reviewed_at`；apply 完成写 `applied_at`；apply_failed 保留 retry 次数 cap
- **idempotent replay**：前一次 accept fail 后 retry 不会重复合并（per task #61 P0 atomicity invariant）
- **Pydantic Field validator on `MergeSuggestionView.confidence_score`**：锁 `0 <= score <= 1`（per cuiwenbo msg=49beb855 NIT 3 + Lesson #17 backend 收敛 contract pattern + PR #1930 SearchHit.score 同 pattern）

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

按 `task-17-cr-review-checklist.md` 既有 framework + huangheng PR #1916 + #1924 + #1922 sediment family + **已 fold per PR #1932 commit `dc79aad6`** task #61 sediment 8 项（Lesson #12 v7.4 / #12 v8 / #12 v9 / #13 v2.3 / #13 v3 demo 2 / #14 demo / #16 / #17）应用：

- **Lesson #11 v5**（entry-point migration cross-process parity）— `graph_curation_run` worker lane 加进 `cli/indexing_worker.py` startup + boundary test 钉死，不在 API 进程跑（per task #17 hard cut + task-system-invariants § 2.3）
- **Lesson #12 v4-v9**（CI gate / trust framing / scope walk / 三层 grep / composite key / fake guardrail / first-principles verify）— 全 family 应用
- **Lesson #13 v2.1 + v2.2 + v2.3 + v3**（dual-side rewrite / 不重复事实保证 / deploy manifest dual-side / cross-source default value alignment）— 任意 PR 改 typed schema 跟 Pydantic Field 必双侧同步；status enum 扩展 PG/FE typed schema 双侧 rewrite
- **Lesson #14**（架构 invariant 删除多轮迭代收尾）— Wave 5 description-NULL invariant 应用，dedup 算法不能 fallback 到 description；`EXPIRED` 老 enum 值保留作 backward compat 历史 placeholder
- **Lesson #16**（workflow paths filter dead reference）— 新 worker lane 加 `compat-test.yml` paths filter 同步
- **Lesson #17**（backend 收敛 contract 而非 FE 加 branch / simple-stable family）— suggestion store schema + capability matrix backend 收敛，FE typed schema 仅消费不分支
- **Lesson #18 候选**（lesson sediment + mechanical gate 双 layer codification — 一记一 enforce，per huangheng msg=b18d26ee + chenyexuan PR #1933 first-application demo）— task #31 P1 实施 `kg.merge_suggestion_score_threshold` default 走「lesson 文字 (#17 backend 收敛 contract) + mechanical gate (`tests/unit_test/contracts/test_merge_suggestion_score_threshold_default_consistency.py`) 双 layer codification」
- **Migration chain 时序**（task #31 仅 extend `graph_curation_suggestions` table status enum 4 新 value + 加 `evidence_refs` field — **不建新 `merge_suggestion` table**，per Bryce + Weston + ziang BLOCKER 1）
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
