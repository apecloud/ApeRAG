---
title: task #31 Phase A 实施方案 — 自然中文版（面向非技术读者）
description: ApeRAG 图谱节点合并扫描 + 后台建议任务 v1 落地的 4 个并行子任务，给项目经理派单 + 协作方对照阅读用
---

# task #31 Phase A 实施方案（自然中文版）

> 这份文档是 task #31 spec v1 ([`task-31-graph-node-merge-spec-v1.md`](./task-31-graph-node-merge-spec-v1.md)) 的人话版，方便 PM 派单 + 协作方对照阅读。
> 原 spec 是 architect lock 的技术口径，这份是落地步骤说明。

## 0. 这个 task 在做什么

简单说：**让产品自动发现「同一个东西在图谱里被记成了多个节点」，列出建议给人看，让人决定要不要合并。**

举个例子：
- 文档 A 里有一个实体叫 "Apple Inc."
- 文档 B 里有一个实体叫 "苹果公司"
- 文档 C 里有一个实体叫 "苹果"

现在系统不会自动把它们合并 —— 我们要做的，是 **让一个后台任务定期扫描图谱，发现这种"看起来像同一个东西"的实体，记下来给人看**。人点"接受"，再合并；人点"拒绝"，就不动；人点"忽略"，下次也不再提示。

**关键原则**：
- 不允许后台任务直接合并 —— 必须人工审核（avoid silent destructive 改图）
- 不允许扫描跑在 API 进程（API 应该轻量，扫描很重）
- 不能依赖 entity description 字段（Wave 5 之后这个字段永远是 NULL）

## 1. 现状是什么（不是从零开始）

⚠️ **重要 reframe**：这个 task **不是从零做**。Wave 7 §K.12.4 已经把 detector + suggestion 存储 + 算法评分 **全栈做完了**，但有 5 个问题需要修复或扩展：

| # | 已有 | 问题 |
|---|------|------|
| 1 | `MergeCandidateDetector` 检测器（`aperag/indexing/merge_candidate_detector.py`） | 跑在 API 同步路径（违反 task #17 hard cut，应该跑后台 worker） |
| 2 | `GraphCurationSuggestion` 表（`aperag/domains/knowledge_graph/db/models.py:107`） | 状态枚举只有 `pending/accepted/rejected/dismissed`，缺 worker apply 中间态 |
| 3 | `candidate_generation.py` 算法 | 6 处读 / 用 entity description 字段，但 Wave 5 之后这个字段永远是 NULL |
| 4 | `lineage_merge.py` 应用合并 | 调 LLM 生成统一 description + 写 vector，违反 Wave 5 |
| 5 | trigger 策略 | 只有"文档同步完后自动跑"，缺手动触发 + 定时巡检 |

所以 Phase A 是 4 个并行子任务：把这些坑各自填掉。

## 2. Phase A 的 4 个并行子任务

### #31-A1 抽 worker lane（推荐：Bryce / ziang）

**做什么**：把现在 API 进程同步跑的 detector 抽到独立的后台 worker 上去跑。

**具体步骤**：
1. 在 `aperag/cli/indexing_worker.py` 加一条独立的 lane 叫 `graph_curation_run`（不要加进 `Modality` 枚举，因为 `Modality` 是按文档维度的状态机，扫描是按 collection 维度跑的）
2. 加一个独立的 Redis 队列 `q:graph_curation_run`（不要塞进现有的 `q:indexing:<modality>`）
3. 加独立的 push/pop API：`push_graph_curation_run(run_id, collection_id)` / `pop_graph_curation_run()`
4. 现在 `graph_curation/service.py:114-123` 那段 `asyncio.create_task(asyncio.to_thread(generate_graph_curation_run_task, ...))` —— API 直接 fork 后台任务的代码 —— 改成 `push_graph_curation_run(run_id)`，API 只负责入队，不再自己起后台任务
5. worker 拿到 run_id 后调现有的 `generate_graph_curation_run_task` 完成扫描 + 写 suggestion

**触发策略要分清楚**：
- **手动触发 / 定时巡检**：API 创 run → 入队 → worker pop → 调 `generate_graph_curation_run_task`（重型 full sweep）
- **文档同步完后自动跑**：保留 `GraphModalityWorker.sync` 末尾原有的 `detect_for_sync()` 短路径（轻量 quick path），但同样要修描述字段读取（见 #31-A3）

**验收**：
- worker 启动时 lane 名 `graph_curation_run` 出现在任务列表里（boundary test 钉死）
- API 进程不能 import 任何 graph curation 执行类（boundary test 反向钉死）

---

### #31-A2 扩展状态枚举（推荐：ziang，他熟 Wave 7 §K.12.4）

**做什么**：现在的状态枚举不够区分"用户已批准"和"系统正在执行"和"执行失败"，要扩。

**现有状态**（不动）：
- `PENDING` —— 扫到的候选，等待人审
- `ACCEPTED` —— ⚠️ 这是 legacy 同步路径用的，**老代码里 `ACCEPTED` 已经表示"merge 已经执行完了"**（见 `aperag/graph_curation/service.py:534`）。我们 **不复用它**，免得新老代码同名不同义。
- `REJECTED` —— 用户拒绝
- `DISMISSED` —— 用户忽略
- `EXPIRED / SUPERSEDED` —— 老路径用的过期/被覆盖标记

**新增 4 个状态**：
- `APPLY_PENDING` —— 用户点了接受，已入队等 worker
- `APPLYING` —— worker 拿到任务，正在合并图谱
- `APPLIED` —— 合并成功，audit trail 已写
- `APPLY_FAILED` —— 合并失败（保留重试，重试次数有上限）

**最终状态机**：
```
pending ──→ dismissed
        ──→ rejected
        ──→ apply_pending ──→ applying ──→ applied
                                       ──→ apply_failed
```

**具体步骤**：
1. PG enum migration 加 4 个新值：`APPLY_PENDING / APPLYING / APPLIED / APPLY_FAILED`
2. 前端 typed schema `MergeSuggestionStatus` 同步加这 4 个新值
3. 前端 UI 显示规则：legacy `ACCEPTED` 读到时按"已应用"终态显示（兼容老数据）
4. 加 grep gate 防止新代码写 `ACCEPTED`：新 async path 模块（worker.py / merge worker）import allowlist 里不能出现 `GraphCurationSuggestionStatus.ACCEPTED` 写入
5. boundary test 钉每条状态转移边

**验收**：
- `alembic upgrade head` 跨 backend 跑过
- 测试能区分「用户已批准」「worker 应用中」「worker 已应用」「应用失败待重试」四态

---

### #31-A3 修 description-free（推荐：Bryce / ziang）

**做什么**：现在算法 6 处读 entity description 字段，但 Wave 5 之后这个字段永远是 NULL，必须改成不依赖 description 的实现。

**6 处 detector / snapshot 必修 call site**（不能少一处）：

| # | 文件 + 行号 | 怎么改 |
|---|-------------|--------|
| 1 | `candidate_generation.py:43` | `entity_snapshot()` 删 `entity.description` 读取 |
| 2 | `candidate_generation.py:179-181` | 删 `description_overlap = _jaccard(...)` + `if description_overlap >= 0.2` |
| 3 | `candidate_generation.py:196-197` | 删 `score += min(...description_token_overlap... * 0.20, 0.15)` |
| 4 | `dto.py:59-65` + `:101-105` | `CurationEntity.description` 字段 input 改成不依赖（基于 source_chunk_ids 兼容路径） |
| 5 | `merge_candidate_detector.py:257-284` | `_description_text_for_scoring()` 改成基于 entity name + type embedding query |
| 6 | `merge_candidate_detector.py:322-328` | snapshot 写 description → 改成不写 description 字段 |

**还有 1 处 apply 路径**：
- `lineage_merge.py:246-317` 抽 `merge_entities_apply_description_free()` 变体 —— 不调 LLM 生成统一 description / 不调 compactor / 不写 `__curation_merge__` description part / 不写 vector embedding
- 老的 `lineage_merge.py` 路径 **保留**作 manual API 兼容（标 deprecation，下一轮 sweep）

**验收**：
- 6 处 grep `description` 字段读取 → 零匹配
- worker accept lane 模块 import allowlist 不含 `description` / `compactor` / `unified_description` 等 LLM helper module

---

### #31-A4 复用 endpoint + 前端扩展（推荐：dongdong + cuiwenbo）

**做什么**：不要新建 endpoint，复用现有的 `/graphs/merge-suggestions` 路径。

**复用 3 个现有 endpoint**（不新建）：
- `POST /api/v2/collections/{id}/graphs/merge-suggestions` —— 创 run（手动触发用这个）
- `GET /api/v2/collections/{id}/graphs/merge-suggestions?status=pending&limit=&cursor=` —— 列分页
- `GET /api/v2/collections/{id}/graphs/merge-suggestions/{suggestion_id}` —— 详情
- `POST /api/v2/collections/{id}/graphs/merge-suggestions/{suggestion_id}/action` body `{action: 'accept'|'reject'|'dismiss'}` —— 决策

**typed schema extend**（不破现有）：
- 现有 `MergeSuggestionItem` (`web/src/features/knowledge-graph/types.ts:80`) 扩展为 `MergeSuggestionView`
- 加字段：`evidence_refs[].document_id/chunk_id/parse_version` + `affected_doc_count` + 新 status values + `observed_types: list[str]` + `type_conflict: bool` + `suggested_entity_type: Optional[str]`（per § 3.1.3 entity_type scope lock layer 2）
- `confidence_score` 加 Pydantic Field validator 锁 `[0, 1]`

**前端 UI 扩展**：
- `SUGGESTION_ACTIONS` const 加 `dismiss`（FE 现仅 `accept/reject`）
- review queue panel 显示 7 个 active 状态 + legacy `accepted` 兼容显示
- accept 后端返回 updated `MergeSuggestionView`（不要 204 + refetch）

**配置 schema lock**（path 跟 task #30 / task #61 既有 schema 对齐）：
- `collection.config.knowledge_graph_config.merge_suggestion_strategy: list[Literal['cron', 'manual', 'auto_post_ingest']]`
- `collection.config.knowledge_graph_config.merge_suggestion_algorithms: list[Literal['name_exact', 'name_fuzzy', 'type_compatible', 'embedding_topk', 'source_chunk_overlap']]`
- `collection.config.knowledge_graph_config.merge_suggestion_score_threshold: Optional[float]`（默认 0.85，Phase C benchmark 后 lock）
- `collection.config.knowledge_graph_config.merge_suggestion_max_per_scan: Optional[int]`（默认 100）

---

## 3. entity_type 边界锁

PM 问过：要不要把"人物 / 人员 / 人"这种 entity_type 也合并？

**架构师 + ziang + dongdong + Weston 三方答案**：
- v1 **不合并 entity_type 本身**，只把它作为 entity name 合并的"辅助信号"（type 近似加分，type 冲突降分）
- 同名候选不能因为 type 不完全相等就被过滤（要容忍 `人物 / 人员 / 人`）
- suggestion 显示时展示 `observed_types` + `type_conflict` + `suggested_entity_type`，accept 时人选最终 type
- entity_type 归一化作为 **独立的 `entity_type_alias` suggestion kind**，留给 Phase B / P1 follow-up（task #31-C3），独立设计 store/API/migration/UI

不在 v1 范围里。

---

## 4. Phase A 派单建议

| 子任务 | 推荐 owner | 理由 |
|--------|-----------|------|
| #31-A1 worker lane | @Bryce 或 @ziang | Bryce 熟 cli/indexing_worker.py 部署侧（task #17 hard cut 经验）；ziang 熟 graph_curation 域 |
| #31-A2 状态枚举扩展 | @ziang | Wave 7 §K.12.4 origin author，最熟 enum 历史 semantic |
| #31-A3 description-free 修 6+1 处 | @Bryce 或 @ziang | 跨 graph_curation + indexing 双侧 |
| #31-A4 endpoint reuse + FE | @dongdong + @cuiwenbo | dongdong 主前端 + cuiwenbo 配合 typed schema sync |

**4 个子任务可以并行**（除了 A2 migration 要先于 A4 FE typed schema sync）。

## 5. Phase B / Phase C 是什么

**Phase B**（依赖 Phase A merged）：
- B1 CR + boundary test gate（@huangheng）
- B2 integration test e2e（@huangzhangshu primary / @冬柏 peer）+ `LineageEntityMerger` cross-backend 行为契约测试
- B3 Helm 部署验证（@Planetegg SRE）

**Phase C**（数据驱动）：
- C1 算法 ROC / 准确率 benchmark（基于 task #30 PR #1863 framework）
- C2 default `merge_suggestion_score_threshold` lock
- C3 `entity_type_alias` suggestion kind follow-up

## 6. 不阻塞的事

这个 task 不阻塞：
- task #61 P1/P2 实施队列
- task #11 GC orphan vector follow-up
- huangheng follow-up 子 PR（task #61 sediment 已 fold per PR #1932）
- task #33 Layer 2 已 done（PR #1933 chenyexuan merged）

---

**起草**：@符炫炜（总架构师）
**日期**：2026-04-30
**版本**：v1（task #31 Phase A 派单实施版，spec lock 后立即可派）
**对应 spec**：[`task-31-graph-node-merge-spec-v1.md`](./task-31-graph-node-merge-spec-v1.md)
