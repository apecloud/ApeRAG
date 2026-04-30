---
title: task #61 — DB Adapter 兼容层审计 spec v1
description: ApeRAG vector + graph adapter 跨实现行为兼容性审计 + capability/degradation contract + sub-task 拆分
---

# task #61 — DB Adapter 兼容层审计 spec v1

> earayu2 directive (`#indexing优化` msg=8b989470 / msg=2bad8e75 / msg=f26b703e)：所有人一起看数据库层对于同类型数据库不同实现的兼容性 — vector (PGVector / Qdrant) + graph (Neo4j / Nebula / PG)。统一兼容层切换时行为必须一致；明显行为差异（影响上层 / 报错 / 数据错误）必修；测试覆盖必补；性能必检（代码级优化 + 接口语义 + batch primitive）。

## 1. 现状 inventory（多 lane streaming 输入实证）

### 1.1 Vector adapter (PGVector / Qdrant) — huangheng msg=ed2f2973 + Bryce msg=8e895471 grep 实证

11 finding dedupe 后（Bryce + huangheng 同时 surface 部分项已合并）：4 P0 + 3 P1 + 4 P2 (file:line)：

| # | 路径 | 现象 |
| --- | --- | --- |
| ~~P0-V1~~ → **P1-V4** | `aperag/vectorstore/qdrant_connector.py:442-446` collection_name binding | **重新定性 (per Bryce msg=23a2f514 first-principles verify)**：legacy mode `collection_name = tenant_id` 是 **per-tenant physical collection 隔离**，不靠 `retrieve()` 内部 filter — connector 已绑定 tenant-specific collection，无 cross-tenant leak；下沉 P1 defense-in-depth 不对称（legacy 路径少一层 belt-and-braces filter，未来 legacy mode 删除候选）+ Planetegg msg=41665d7e Singapore `QDRANT_MULTITENANT=True` 无生产暴露 |
| **P0-V2** | `aperag/vectorstore/qdrant_connector.py:293-298` | filter translation 未知 type **silent return None** + log warning → 静默不带 filter 返全集（PGVector `pgvector_connector.py:254-258` 同 case **fail-loud TypeError**） |
| **P0-V3** | `aperag/vectorstore/pgvector_connector.py:545-547` vs `qdrant_connector.py:626` | Score threshold 跨 distance metric 解释方向不一致：PGVector L2/dot 取负保「higher=better」，Qdrant 直接传 raw distance — 同 score_threshold 在 cosine OK，L2/dot 上 cutoff 范围发散 |
| **P0-V4** | `aperag/vectorstore/base.py:130-137` docstring | "higher = better" 说明，但 Qdrant native distance 方向无 guard — 未来若改 score 方向 → silent 排序倒过来 |
| **P1-V1** | `pgvector_connector.py:434-437/463` vs `qdrant_connector.py:511-514` | Collection init failure 行为分化：PGVector DDL fail 不缓存 + 重试 spam (fail-silent); Qdrant generic `Exception` 重抛 (fail-loud) |
| **P1-V2** | PGVector `engine.begin()` 包 INSERT ON CONFLICT vs Qdrant `client.upsert(points, wait=True)` | Batch upsert atomicity：PGVector all-or-nothing，Qdrant best-effort per-point — caller 不能假定原子 |
| **P1-V3** | PGVector `(part1 OR part2 OR ...)` SQL vs Qdrant `Filter(should=subs)` `min_should_match=0` | Filter Or 语义：Qdrant should-only query 可能 match 全集 — 数据正确性 risk |
| P2-V1 | `retrieve()` 返回顺序无保证 | caller 不能依赖输入顺序 |
| P2-V2 | Error type 分化 (SQLAlchemyError vs UnexpectedResponse) | 无 backend-neutral wrapper |
| P2-V3 | Hint 名称分化 (`ef_search` vs `hnsw_ef`) | 不能跨 adapter 透传 |
| P2-V4 | Embedding dimension 写时不验证 | PGVector SQL cast error, Qdrant 可能 silently truncate |

**FE 侧 cuiwenbo msg=dfebf706 surface**:

| # | 路径 | 现象 |
| --- | --- | --- |
| P1-V2 | `web/src/...search-result-drawer.tsx:88` + 2 处 score 显示 | PGVector 返 cosine_distance（0=match）vs Qdrant 返 cosine_similarity（1=match）— FE 显示同 query 在不同 backend 显示语义反向 |

### 1.2 Graph adapter (Neo4j / Nebula / PG LineageGraphStore) — ziang task #64 + 冬柏 task #67 streaming

`aperag/indexing/graph_storage/{neo4j,nebula,postgres}.py` (Wave 7 `aperag/domains/knowledge_graph/graphindex/` 已删 — 真实路径迁移到 `aperag/indexing/graph_storage/`)。

**冬柏 msg=3e93bb64 surface**:

| # | Protocol method | 现象 |
| --- | --- | --- |
| **P0-G1** | `LineageGraphStore.bulk_upsert_entity_with_lineage_parts` | 无 cross-backend test 覆盖 — bulk write 是后端差异最大点（batch 大小限制 / atomicity / error handling / concurrency）；indexing worker 走这条 path 大批量写实体 silent drop 风险 |
| P1-G1 | `LineageGraphStore.remove_relation_lineage_member` | counterpart `remove_entity_lineage_member` 已 test (line 355) 但 relation 路径完全没测 — Lesson #13 v3「dual-side rewrite」反 pattern |
| P1-G2 | `LineageGraphStore.list_entities` | pagination/sort stability 跨后端漂（Neo4j internal id / Nebula vid / PG ctid），上层取分页用 |

**ziang task #64 in_progress** — graph store contract diff list 即将出。

### 1.3 SRE / live env — Planetegg task #65 in_review

| # | 路径 | 现象 |
| --- | --- | --- |
| P2-S1 | `aperag/graph_curation/alias_map.py:resolve_canonical` + `alias_redirect_store.py:expand_neighbors_n_hops` | per-node `asyncio.gather` 在 PG 连接池吃紧时放大连接压力（Singapore 实证 stack `TooManyConnectionsError` per msg=db7fb085）— P2 性能/接口项候选 batch resolve 改造。**量化 (per Planetegg msg=eb9de4b0)**：`/graphs` overview seed 上限 `max_nodes * 2`（default 1000 → 2000 connections）；`/graphs/hybrid` default 1000 / max 5000 — P2 batch resolve 优先级跟这条 caller chain 量化对应 |
| P0-Env | Singapore api 2 副本 + 没有独立 indexing-worker deployment | task #17 hard cut 没 deploy → API + worker 同进程导致连接池放大；**deployment fix 不在本 spec scope**（task #17 deploy runbook 已 ready，huangzhangshu lane 跟进） |

### 1.4 Workflow gate — chenyexuan PR #1926 in flight

| # | 路径 | 现象 |
| --- | --- | --- |
| **P0-W1** | `.github/workflows/compat-test.yml:7-9` paths filter | 指向 `aperag/domains/knowledge_graph/graphindex/**`（**Wave 7 已删除目录** — dead reference）；真实路径在 `aperag/indexing/graph_storage/`。任何 graph adapter PR 都不 trigger compat-test workflow — 30 case cross-backend test 形同虚设 — **PR #1926 fix in flight** |

### 1.5 FE / deploy — cuiwenbo task #70 + dongdong task #71 streaming

cuiwenbo P1+P3 候选见 § 1.1。dongdong task #71 deploy/typed schema lane in_progress（Helm/compose env + typed schema cross-backend exposure）— 输出后 fold 进 spec amend。

## 2. 缺口识别（按 severity）

按 Weston msg=85e527e3 + msg=65cf3b8b 三层框架 + earayu2 directive 「明显行为差异（影响上层 / 报错 / 数据错误）必修」:

### 2.1 P0 CRITICAL（数据正确性 risk → 重新定性后无 hot-fix）

- ~~P0-V1 cross-tenant leak~~ → **下沉 P1-V4 defense-in-depth 不对称**（per Bryce msg=23a2f514 first-principles verify：legacy mode physical collection 隔离已 cover，无 leak）

### 2.2 P0（必须一致 — 影响上层正确性，per Weston msg=13dd5e91 BLOCKER 修订 score normalization 升回 P0）

- **P0-V2 / P0-A** (Bryce fix PR scope): filter translation silent divergence: Qdrant fail-silent `return None` 退化 unfiltered 全集 vs PGVector fail-loud — Lesson #12 v8 fake guardrail anti-pattern 应用
- **P0-V3+V4 / P0-B** (Bryce fix PR scope): score normalization 跨 distance metric 解释方向不一致（P0 必修 per Weston msg=13dd5e91 — score 方向是 caller 语义硬契约，FE/API/MCP 不能在 PGVector/Qdrant 间看到反向含义）：
  - PGVector L2/dot 取负保「higher=better」+ Qdrant 直传 raw distance 行为分化
  - cuiwenbo msg=dfebf706 surface FE 显示语义反向（score 0.05 vs 0.95）= 同根因
  - 修法：base contract 强制声明 0-1 + higher=better similarity；Qdrant L2/dot 加 sigmoid normalize；boundary test 跨 metric 全 enum coverage
- **P0-G1** `bulk_upsert_entity_with_lineage_parts` 跨 backend 行为差异 + 0 test coverage — bulk write atomicity / batch limit / error handling 必须 contract 一致（**冬柏 PR #1927 boundary test 已 deliver in flight**）
- **P0-W1** `compat-test.yml` paths filter dead reference — workflow gate 形同虚设 — 解锁所有其他 P0 验证能力前提（**chenyexuan PR #1926 in flight**）
- **P0-D1** Helm `indexing-worker-deployment.yaml` 缺 Neo4j env/secret 注入 vs API deployment（per dongdong msg=4201465a + cuiwenbo msg=bcec38ad root cause）— Singapore graph viz 故障真正 root cause 之一（worker 写入侧凭据漂移 → graph 写入静默失败 → 0 entity / 0 relation + read 失败 toast 混淆）。**dongdong PR #1929 in flight**

### 2.3 P1（允许差异但显式 declaration）

- **P1-V1** collection init failure 行为分化（fail-silent vs fail-loud）— 统一 fail-loud + retry helper
- **P1-V2** Batch upsert atomicity (atomic vs best-effort) — explicit capability declaration
- **P1-V3** Filter Or 语义 (Qdrant should-only match 全集 risk) — 拒绝 empty Or parts + boundary test 跨 adapter 命中相同集合
- **P1-V4 / 原 P0-V1 下沉**：Qdrant legacy mode defense-in-depth 不对称（physical collection 隔离已 cover，但 query filter 不对称 + legacy mode 删除候选 follow-up — Lesson #14 多轮迭代收尾）
- **P1-G1** `remove_relation_lineage_member` test gap — boundary test 钉 dual-side rewrite invariant
- **P1-G2** `list_entities` pagination/sort stability — explicit capability declaration（同分排序允许差异，order key 必稳定）
- **P1-D1** e2e shape matrix 缺 3 组合 (`qdrant+postgres` / `pgvector+neo4j` / `pgvector+nebula`) (per dongdong msg=4201465a) — 建议 nightly/manual 或 DB-compat change targeted matrix
- **P1-D2** Helm Nebula 缺 first-class dependency/secret（只有 `api.env.NEBULA_*`，无等价 dependency/secret values）— explicit deploy capability/degradation declaration 或补 first-class Nebula deploy path
- **P1-D3** web typed schema 只暴露 `graph_backend_type`，缺 vector backend / capability / degradation 结构化暴露 — backend contract 先补字段或 endpoint 后 FE 才能显示「允许差异但显式」

### 2.4 P2（性能优化 / 接口语义）

- **P2-S1** alias resolution `asyncio.gather` per-node 放大 PG 连接 — batch resolve primitive 接口改造（fold-in task #61 Phase 2，不阻塞 P0/P1 修复）

### 2.5 P3 / YAGNI (defer)

- cuiwenbo P3 候选 `confidence_score` range 跨 graph backend — task #31 graph node merge spec 启动时 fold-in，不在本 spec scope
- 不追求 100% 一致（per earayu2 directive）— 同分排序顺序、近似召回差异等天然差异允许，不强行收敛

## 3. 设计方向

### 3.1 必须做（Hard scope）

#### 3.1.1 ~~P0 hot-fix path~~（重新定性后无 hot-fix 必修）

Bryce msg=23a2f514 first-principles verify 后 P0-V1 下沉 P1-V4 defense-in-depth — Qdrant legacy mode physical collection 隔离已 cover，无 cross-tenant leak，无 hot-fix 紧迫性。Planetegg msg=41665d7e 实证 Singapore `QDRANT_MULTITENANT=True` 无生产 legacy mode 启用。

P1-V4 处理路径：跟其他 P1 一起 explicit declaration / 或随 legacy mode deprecation follow-up（Lesson #14 多轮迭代收尾）一并删除 legacy code path。

#### 3.1.2 P0 必须一致 contract list（task #61 主线）

每条 P0 必修项产出三个交付物:
1. **adapter 实现修复** PR (Bryce/ziang lane)
2. **boundary test 钉 invariant** (huangheng + 冬柏 lane)
3. **capability declaration**（如不可强行一致）— 写进 `aperag/vectorstore/base.py` / `aperag/indexing/graph.py` adapter Protocol docstring + spec 文档化

#### 3.1.3 P1 允许差异显式 declaration

- 各 adapter 实现暴露 `capabilities()` 或类似 contract surface — 列「不支持 X / 行为是 Y」（不允许 silent fallback）
- FE / agent / MCP caller 通过 typed schema (cuiwenbo + dongdong lane) 看到 capability flag
- spec 文档化 allowed differences list（如 score 单调性允许 distance 或 similarity 选一，不允许某 backend 静默切换）

### 3.2 P2 性能优化（contract 明确后做）

- **alias resolution batch primitive**（P2-S1）：`alias_redirect_store.py:expand_neighbors_n_hops` 改 batch resolve，不再 per-node `asyncio.gather`
- 其他 N+1 query / 错误映射 / 分页 stability — 等各 lane audit slice 输出后 fold

### 3.3 不做（YAGNI）

- 100% 跨 adapter 字节一致（per earayu2 directive）— 同分排序、近似召回差异允许
- 跨 vector + graph 混合 query 接口收敛（task #32 MCP 审计已处理 multi-tool 组合）
- 反向 verify pass / 全量 retroactive backfill — 旧数据保持，invariant 仅 lock new write path

## 4. 实施 sub-task 拆分（已 PM 派单）

### Phase A（已并行启动）

| sub-task | owner | 状态 |
| --- | --- | --- |
| #64 graph store audit slice (Neo4j/Nebula/PG `LineageGraphStore`) | @ziang | in_progress |
| #65 SRE live env + connection budget + perf gap scan | @Planetegg | in_review |
| #66 architecture contract matrix + adapter API audit | @Weston | claimed |
| #67 testing-lane compat coverage scan + Protocol method gap | @冬柏 | claimed |
| #69 backend vector adapter audit (PGVector/Qdrant filter/score/upsert/delete/dimension) | @Bryce | claimed |
| #70 FE impact audit (graph endpoint consumer behavior assumption) | @cuiwenbo | claimed |
| #71 FE/deploy impact audit (typed schema + Helm/compose env) | @dongdong | claimed |
| #72 spec v1 起草 + sub-task 收口 (本 spec) | @符炫炜 | in_progress |

### Phase B（每条 P0 拆三 PR）

按 § 3.1.2 三交付物模式，每条 P0 触发三 PR sequential:
1. **adapter 实现修复 PR** — 实施 owner (Bryce vector / ziang graph) 主推
2. **boundary test PR** — huangheng / 冬柏 lane 配合（不重复事实保证 per Lesson #13 v3）
3. **capability declaration + docstring + spec amend PR** — 我（架构师）整合

### Phase C（P2 性能 + 接口语义）

P0/P1 contract 锁定后启动:
- alias resolution batch primitive (Planetegg/ziang)
- 接口层语义收紧 (huangheng + Weston review 三分类框架)

### Phase D（PR #1926 unblocks）

`compat-test.yml` paths filter fix 后 → 30 case cross-backend test 真触发 → 所有 Phase B `boundary test PR` 跑 CI verify。

## 5. 验收口径

### 5.1 P0 完成标准

- 每条 P0 必修项: adapter 实现 fix + boundary test 钉死 + capability declaration 入仓
- `tests/integration/compat/test_vector_compat.py` + `test_lineage_graph_compat.py` cross-backend 跑 30+ case (含 § 1.2 冬柏 surface 3 missing methods 加进去) **跑过非 skip**
- `compat-test.yml` workflow 真触发（PR #1926 merged）
- P0-V1 Qdrant legacy cross-tenant filter 永久 boundary test 钉死

### 5.2 P1 完成标准

- 每条 P1 差异: 行为统一 OR explicit capability declaration in adapter Protocol docstring + `typed schema` 暴露 capability flag
- FE 消费侧（cuiwenbo / dongdong lane）按 capability flag 显示对应 UI

### 5.3 boundary test gate (per Weston msg=13dd5e91 BLOCKER 修订 — score normalization test 显式)

- 现有 G1-G19 + `test_modularization_boundaries.py` + `test_worker_di_parity.py` + `test_no_rerank_in_mcp.py` + `test_graph_window_caps_co_scale.py` 不破坏
- 新加 `test_no_silent_filter_fallback_in_vector.py` 钉死 P0-V2 invariant
- 新加 `test_score_normalization_in_vector.py` 钉死 P0-V3+V4 invariant：跨 (PGVector × cosine/L2/dot) × (Qdrant × cosine/L2/dot) 全 6 cell parametrize，同 embedding × 同 query 验证 score ∈ [0,1] + 排序一致
- 新加 `test_cross_tenant_isolation_in_vector.py` 钉死 P1-V4 defense-in-depth invariant（即使 collection-level 隔离成立，也加 query-level filter 钉跨 mode 一致）
- `test_lineage_graph_compat.py` 加 `bulk_upsert_entity_with_lineage_parts`（**冬柏 PR #1927 已 deliver**：38 cases incl. zero-side-effect + replay idempotency per `b2234aee`）+ `remove_relation_lineage_member` + `list_entities` 3 method 跨 backend test

### 5.4 e2e smoke

- e2e shape matrix (Lite / Qdrant+Nebula / Qdrant+Neo4j) 跨 backend 跑 ci-flake-policy.md § 2.1 Lite 必绿 + § 2.2 单 shape signature 放行规则
- 无 silent fallback / silent data loss 跨 adapter

### 5.5 sample 限制免责

本 spec evidence 来自 multi-lane streaming surface（huangheng / cuiwenbo / 冬柏 / Planetegg / ziang），不是 huangzhangshu 完整 collect 的 gap list。**huangzhangshu task #61 audit 收尾后**可能 surface 额外 P0/P1 候选 → spec amend fix-forward。

## 6. CR mandatory checklist

按 `task-17-cr-review-checklist.md` 既有 framework + huangheng PR #1916 + #1924 + chenyexuan PR #1922 sediment family 应用：

- **Lesson #11 v5**（entry-point migration cross-process parity）— P1-V1 collection init 行为分化跨 adapter 应用
- **Lesson #12 v4**（PR `lint-and-unit` CI 全绿是 mandatory ratify gate）
- **Lesson #12 v5**（CI status trust framing 反模式 — 跨 PR forensics）
- **Lesson #12 v6 / v6.1 / v6.2 / v6.3 / v6.4**（scope walk: function / endpoint / data type / aggregation chain）— P0-V1 caller chain cross-tenant verify 应用
- **Lesson #12 v7 / v7.1 / v7.2 / v7.3**（caller signature → backend schema → runtime fallback / composite key invariant / Pydantic schema layer / cross-PR default value alignment）— capability declaration Pydantic schema 暴露应用
- **Lesson #12 v8**（fake guardrail anti-pattern）— P0-V2 silent return None 应用
- **Lesson #13 v2.1 + v2.2 + v3**（dual-side rewrite + boundary 不重复事实保证）— P1-G1 remove_relation_lineage 应用
- **Migration chain 时序 invariant**（如本 task 涉及 DB schema 改动）
- **Lesson #14**（架构 invariant 删除多轮迭代收尾）— 跨 adapter 修复多轮 fix-forward 容忍
- **Lesson #15**（file-move 3-step verify）— 不适用
- **Lesson #16 候选**（workflow paths filter dead reference 反 pattern）— P0-W1 实证 demo，sediment fold 进 cr-checklist follow-up
- **简单稳定 + 私有化部署免维护 4 guardrail**

## 7. 关联文档

- earayu2 directives: `#indexing优化` msg=8b989470 (DB 兼容审计) + msg=2bad8e75 (全员协作) + msg=f26b703e (主动参与)
- huangheng grep 实证: msg=ed2f2973 (3 vector P0)
- Bryce vector audit: msg=8e895471 (11 finding) + msg=23a2f514 (P0-V1 first-principles 重新定性)
- 冬柏 testing scan: msg=3e93bb64 (compat-test paths + 3 method gap) + PR #1927 / commit `b2234aee`
- chenyexuan workflow gap: msg=f298011e + PR #1926
- cuiwenbo FE audit: msg=dfebf706 (3 FE 候选) + msg=bcec38ad (deploy root-cause connection)
- dongdong deploy: msg=4201465a (P0-D1 + 3 P1 audit findings) + PR #1929
- Planetegg SRE: msg=db7fb085 (Singapore alias gather connection pool) + msg=41665d7e (Singapore multitenant verify) + msg=eb9de4b0 (P2-S1 quantification)
- Weston 三层框架: msg=85e527e3 + msg=65cf3b8b + msg=13dd5e91 (BLOCKER score normalization P0 confirm)
- task #30 spec v1: [`task-30-graph-chunk-window-spec-v1.md`](./task-30-graph-chunk-window-spec-v1.md)
- task #32 MCP 审计 spec v1: [`task-32-mcp-audit-spec-v1.md`](./task-32-mcp-audit-spec-v1.md)
- task #17 任务系统不变式: [`task-system-invariants.md`](./task-system-invariants.md)
- cr-checklist accumulated sediment: [`task-17-cr-review-checklist.md`](./task-17-cr-review-checklist.md)

## 8. 不阻塞主线

本 spec **不阻塞**:
- task #30 Phase B PR #1925 default=2 lock (B3 in flight)
- PR #1926 compat-test paths filter fix
- Singapore 2pm release（per earayu2 directive 不止血代码，env 问题独立 deploy fix lane）
- task #31 graph node merge / task #33 P3 workflow gate

---

**起草**：@符炫炜（总架构师）
**日期**：2026-04-30
**版本**：v1（task #61 spec lock 候选；team review + earayu2 ratify 后 PM @不穷 按 Phase A / Phase B / Phase C 调度实施 PR）
