---
title: ApeRAG 可观测性需求对齐文档
description: 全路径埋点 + LLM 账本 + token/cost 字段 - 跨域指标 capability/scope/字段/维度/口径 alignment (NO 实现)
---

# ApeRAG 可观测性需求对齐文档

> earayu2 directive (飞书 + `#indexing优化` msg=1331d5e7): 汇总近期可观测性相关需求, 整理详细文档, 跟其他人讨论并对齐 — 先不考虑如何实现, 讨论完成发 PR 合并。
>
> 本文档仅 alignment **指标 capability + scope + 字段 + 维度 + 口径**, NOT 实现细节。具体实施 spec 见 [`task-89-telemetry-spec-v1.md`](./task-89-telemetry-spec-v1.md) (P0 已 spec lock 候选)。

## 0. 为什么现在做（驱动场景）

### 0.1 真实用户问题

- **太钢 token 消耗追问** (谭怀远 飞书群): 客户问"这次烧了多少 token", 当前线上 PG **无 usage 表** + LiteLLM callback 禁用 + worker 调用不记 token → 完全答不上来。
- **私有大模型 cost 担忧** (`ou_1d75b5...` 飞书): 客户都是私有大模型部署, 如果消耗太大客户使用会有问题 — token/cost 是**对外可见的运维必需品** (NOT internal-only debug)。
- **财务 + 容量预测**: 没有 per-collection / per-doc / per-LLM-call 维度的成本归因, 财务无法分摊, 容量规划只能拍脑袋。
- **Singapore 大文档调试** (Planetegg msg=1314ac59): 413+1316 chunks 大 PDF graph extraction RUNNING 现场, **无 window 级耗时 / entity/relation 计数** → 只能 `tail -f` 日志反推, 平均 / p50 / p95 / 失败率 SQL 全无。

### 0.2 缺口本质

ApeRAG 现有可见性 surface = `logger.info` / `logger.warning` 文本日志, **无结构化埋点 + 无可查询数据 + 无 SaaS telemetry (私有化部署不能用 Datadog/NewRelic)**。

跨用户问题 → 同一个根因: **没有 machine-readable, structured, queryable, privacy-safe 的 telemetry foundation**。

## 1. 文档定位 (per earayu2 directive)

### 1.1 本文档做什么

✅ **alignment 出指标 capability 范围 + 字段 + 维度 + 口径** — 跨域协作方在统一语言里讨论需求
✅ **优先级 + trigger condition + privacy boundary 锁定** — 决策者拍 P0/P1/P2 时机
✅ **驱动场景显式 link**: 每条 capability 关联具体业务问题 / 用户反馈 / 决策依据

### 1.2 本文档不做什么

❌ **不写实现** — 接入点 / SQL schema / Pydantic / Helm / boundary test 等已在 [`task-89-telemetry-spec-v1.md`](./task-89-telemetry-spec-v1.md) (P0) 或 future P1/P2 spec
❌ **不立 dashboard/UI** — 验收路径 = SQL/query + tests (per Weston msg=22e6df03 P0 lock)
❌ **不绑定具体 backend infra** — 复用现有 PG (per 4 guardrail 简单稳定 + 私有化部署免维护)

## 2. 全路径可观测性 capability map

### 2.1 全路径分层

ApeRAG 端到端有 4 大执行层, 每层有自己的 metric surface:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer A: API 层 (FastAPI)                                    │
│  - HTTP request latency / status code / endpoint            │
│  - 用户/租户/collection 维度                                 │
├─────────────────────────────────────────────────────────────┤
│ Layer B: 索引 worker 层 (indexing-worker)                    │
│  - lane lifecycle (claim/release/heartbeat)                 │
│  - 队列深度 (queue depth) / concurrent_tasks                │
│  - per-Modality task 耗时 / 失败率                          │
├─────────────────────────────────────────────────────────────┤
│ Layer C: 业务执行层 (graph extraction / vector / fulltext)  │
│  - per-window / per-chunk 耗时 + node/edge count            │
│  - per-document 汇总 (windows_total/success/failed/...)     │
│  - vector / fulltext indexing 耗时 + dimension count        │
│  - retrieval search top-K / hit rate / latency              │
├─────────────────────────────────────────────────────────────┤
│ Layer D: LLM 调用层 (cross-cutting concern)                  │
│  - per LLM call: model / provider / token / cost / latency  │
│  - call_purpose 维度: chunking / extraction / embedding / answer / rerank │
│  - 关联 Layer A/B/C: collection_id / doc_id / window_id / run_id │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各层 capability 范围 + 优先级

| Layer | Capability | 字段 | 关联维度 | 驱动场景 | 优先级 |
|-------|-----------|------|---------|---------|--------|
| C | **graph extraction window** | `chunk_ids` / `duration_ms` / `entity_count` / `relation_count` / `llm_call_count` / `llm_token_count` / `model_id` / `provider` / `timeout_seconds` / `chunks_truncated` / `status` / `error_type` | collection_id / document_id / parse_version / document_index_id | Singapore 大 PDF + per-chunk 时间 (Planetegg) + per-chunk node/edge (earayu2) | **P0** |
| C | **graph extraction document** | `chunks_total` / `windows_total` / `windows_success` / `windows_failed` / `windows_timeout` / `entities_total` / `relations_total` / `wall_time_ms` | collection_id / document_id / parse_version / document_index_id | 太钢 token 追问 (per-doc 汇总) + 大文档容量诊断 | **P0** |
| B | **worker lane lifecycle** | `lane: str` / `event: claim\|release\|heartbeat` / `queue_depth` / `concurrent_tasks` / `duration_ms` | (不绑 collection 因 worker 是 process-level) | indexing-worker 健康度 + 队列积压可见性 | **P1** |
| C | **vector indexing** | `chunk_count` / `embedding_count` / `dimension` / `model_id` / `provider` / `duration_ms` / `status` | collection_id / document_id / parse_version | 索引检索全路径覆盖 (earayu2 范围) | **P1** |
| C | **fulltext indexing** | `chunk_count` / `index_size_bytes` / `duration_ms` / `status` | collection_id / document_id / parse_version | 同上 | **P1** |
| C | **retrieval search** | `top_k` / `embedding_count` / `candidate_count` / `hit_count` / `latency_ms` / `search_mode: vector\|fulltext\|hybrid` / `status` | collection_id / query_hash (NOT query text) | 检索性能可见性 + hit rate + 慢查询定位 | **P2** |
| D | **LLM call ledger** (独立 event family per Weston msg=94df61b3) | `model_name` / `provider` / `prompt_tokens` / `completion_tokens` / `total_tokens` / `cost_usd` (or local currency) / `latency_ms` / `status` / `error_code` / `call_purpose: Literal['chunking','extraction','embedding','answer','rerank','summarization']` | collection_id / document_id / window_id / run_id / call_id | **太钢 token 追问 + 私有大模型 cost 担忧 + 财务分摊** | **P2** (优先级最高 P2 — 跟 P1 worker lane 几乎同时启动条件) |
| A | **HTTP request** | `method` / `endpoint` / `status_code` / `latency_ms` / `tenant_id` | user_id / collection_id (when applicable) | 索引检索 worker 全路径覆盖 (earayu2 范围) — API 入口侧 | **P3** (defer, 现有 access log 已有, 暂不优先) |

### 2.3 优先级判定逻辑 (per 4 guardrail + 真实需求驱动)

- **P0 (graph extraction 2 metric)**: earayu2 directive 直接 surface + Singapore 大 PDF 现场实证 + 太钢 token 追问 (per-window/document) — production blocking gap, 不能再等
- **P1 (worker lane + vector/fulltext indexing)**: 全路径覆盖前 50% — P0 production data 收集 ≥ 1 周后启动条件 (per 不无限扩范围 4 guardrail)
- **P2 (retrieval + LLM ledger)**: LLM ledger **优先级最高 P2** (太钢直接问 + 私有大模型 cost 必需), 跟 P1 几乎同时启动条件 — 不是 traditional last
- **P3 (API HTTP)**: 现有 access log 部分覆盖, 暂不优先 — earayu2 决策时机

## 3. LLM 调用账本 (Layer D) — 单独 spec out (per Weston msg=94df61b3)

### 3.1 为什么 LLM ledger 单独 spec out

- **业务 priority 高于其他 P2** (太钢 + 私有大模型 cost 担忧)
- **scope 跨 4 layer**: API answer LLM call + indexing-worker chunking/extraction/embedding LLM call + 跨 collection/doc/run 维度归因
- **独立 event family / ledger table** (per Weston): `llm_call_event` 独立, NOT 跟 `telemetry_event` 通用 schema 共表 — 避免 attrs JSON 通用化让 cost 计算 / token 累加慢

### 3.2 字段对齐 (跨域协作方对齐版)

```
event_type = 'llm.call'
ts = call timestamp
collection_id (optional, applicable when LLM call 关联具体 collection)
document_id (optional, applicable when 关联具体 doc, e.g. graph extraction)
window_id (optional, graph extraction context)
run_id (optional, GraphCurationRun / IndexingRun 等 run-level 关联)
call_id = unique LLM call ID (UUID)
call_purpose: Literal['chunking', 'extraction', 'embedding', 'answer', 'rerank', 'summarization']
model_name: str  (e.g. 'gpt-4o-mini', 'qwen2.5-72b', 'text-embedding-3-small')
provider: str  (e.g. 'openai', 'qwen', 'anthropic', 'private_llm')
prompt_tokens: int
completion_tokens: int
total_tokens: int  (= prompt + completion, redundant 但 SQL aggregation 方便)
cost_usd: Optional[Decimal]  (有 price 配置时计算, 无配置时 NULL)
latency_ms: int
status: Literal['success', 'failed', 'timeout', 'rate_limited']
error_code: Optional[str]  (whitelist enum classify_error(exc), e.g. 'context_overflow' / 'rate_limit' / 'timeout' / 'invalid_response')
```

### 3.3 双入口覆盖 (per Weston msg=94df61b3)

- **API 进程 LLM call**: answer (chat) / rerank — 不能只依赖 LiteLLM callback (per Planetegg 现状: callback 禁用)
- **indexing-worker LLM call**: chunking (LLM-aware splitter) / extraction (graph entity/relation) / embedding (vector indexing) / summarization
- 实施时**两侧都必须显式 emit** — 不能假设 LiteLLM callback 总是可用

### 3.4 Privacy hard gate (沿用 P0 boundary)

ledger event `attrs` payload **不含**:
- `prompt_text` / `completion_text` / `messages` / `chunk_text` / `query_text`
- `error_message` 原文 / `traceback` / `repr(exc)` (per huangzhangshu msg=171acb55) — 仅 `error_code` whitelist enum
- `model_response_metadata` 原始 dict (仅提取 token count / cost 字段)

仅允许: structured 元数据 (model name / provider / token count / cost / latency / status / call_purpose / 关联 ID)

### 3.5 Cost 计算 (declarative, NOT runtime probe)

- `cost_usd` 来自 deployment-time **price config table** (per-model / per-token rate)
- 私有大模型 → cost 可填 0 或 NULL (客户自定义)
- 公有模型 (OpenAI / Anthropic / Azure) → 跟 model card 一致价格
- **NOT runtime probe** — 静态 declaration, 跟 task #61 P1-D3 capability declaration pattern 一致 (Lesson #17 backend 收敛 contract)

## 4. 跨层关联维度 (跨 capability 一致 schema)

无论哪 layer, telemetry event 都共享下列关联维度 (跨 capability join 友好):

| 维度 | 定义 | 适用层 | 隐私敏感性 |
|------|------|--------|----------|
| `collection_id` | 业务集合 ID | A/B/C/D | 低 (内部 ID) |
| `document_id` | 业务 doc ID | A/C/D | 低 |
| `parse_version` | doc parse 版本 (retry 串联) | C/D | 低 |
| `document_index_id` | indexing 内部 retry 串联 | B/C | 低 |
| `run_id` | GraphCurationRun / IndexingRun ID | B/C/D | 低 |
| `window_id` | graph extraction window context | C/D | 低 |
| `call_id` | LLM call UUID | D | 低 |
| `tenant_id` / `user_id` | 用户 / 租户身份 | A | **高** — hash or pseudo |
| `query_hash` | 用户查询 hash (NOT raw text) | A/C | **高** — hash only |

**Privacy hard gate (跨 layer 沿用 P0 boundary 4 guardrail)**:
- ID-only, NO raw text (chunk / query / prompt / completion / error message / entity description)
- High-sensitivity 字段 (tenant_id / query_hash) hash 或 pseudonymize

## 5. 验收路径 (跨 layer 一致)

per Weston msg=22e6df03 P0 lock + earayu2 4 guardrail:

- **P0 验收 = SQL/query + tests**, NO admin UI / metrics endpoint
- **P1 验收 = SQL/query + admin metrics endpoint** (typed schema 三层区分: `RawTelemetryEvent` debug-only / `TelemetryAggregateBucket` dashboard primary / `TelemetryTimeRangeSummary` dashboard ready, per dongdong msg=076bfaec)
- **P2 验收 = SQL/query + admin metrics endpoint extend + Grafana datasource docs** (per Planetegg P1-T3)

跨 layer SQL aggregation 示例 (P0 + P2 LLM ledger 数据齐后):
```sql
-- per-doc graph extraction 总耗时 + token cost
SELECT
  te.collection_id,
  te.document_id,
  te.attrs->>'wall_time_ms' AS extraction_wall_time_ms,
  te.attrs->>'entities_total' AS entities_total,
  COUNT(llc.call_id) AS llm_calls,
  SUM((llc.attrs->>'total_tokens')::int) AS total_tokens,
  SUM((llc.attrs->>'cost_usd')::decimal) AS total_cost_usd
FROM telemetry_event te
LEFT JOIN llm_call_event llc
  ON llc.collection_id = te.collection_id AND llc.document_id = te.document_id
  AND llc.attrs->>'call_purpose' = 'extraction'
WHERE te.event_type = 'graph_extraction.document'
  AND te.ts > now() - interval '7 days'
GROUP BY te.collection_id, te.document_id, te.attrs->>'wall_time_ms', te.attrs->>'entities_total'
ORDER BY total_cost_usd DESC LIMIT 50;

-- per-collection / per-call_purpose token 消耗 + 成本归因
SELECT
  collection_id,
  attrs->>'call_purpose' AS purpose,
  attrs->>'model_name' AS model,
  COUNT(*) AS call_count,
  SUM((attrs->>'prompt_tokens')::int) AS prompt_tokens_total,
  SUM((attrs->>'completion_tokens')::int) AS completion_tokens_total,
  SUM((attrs->>'cost_usd')::decimal) AS cost_usd_total
FROM llm_call_event
WHERE ts > now() - interval '30 days'
GROUP BY collection_id, attrs->>'call_purpose', attrs->>'model_name'
ORDER BY cost_usd_total DESC LIMIT 100;
```

## 6. 实施 phase 拆分

| Phase | 包含 | 启动条件 | 实施 spec |
|-------|------|---------|----------|
| **P0** | Layer C graph extraction 2 metric (window + document) + Layer 0 data model + Layer 2 ingestion (DISABLE_TELEMETRY + fail-safe) + Layer 4 boundary | earayu2 已 confirm (msg=5dbbe60a "可以，做吧") — 立即启动 | [`task-89-telemetry-spec-v1.md`](./task-89-telemetry-spec-v1.md) PR #1951 |
| **P1** | Layer B worker lane lifecycle + Layer C vector/fulltext indexing producer + Layer 3 admin metrics endpoint (三层 typed schema) + Helm Grafana docs | P0 production data 收集 ≥ 1 周后, earayu2 + PM 决策 | spec v2 (待启动) |
| **P2** | Layer C retrieval search producer + **Layer D LLM call ledger 独立 event family / ledger table** + cost price config table | 太钢 / 私有大模型 cost 客户问题持续 + LLM ledger 优先级跟 P1 几乎同时 | LLM ledger 单独 spec (待启动) |
| **P3** | Layer A HTTP request producer | earayu2 决策 (现有 access log 部分覆盖) | defer |

## 7. Privacy hard gate (跨 phase 共同基线)

per huangzhangshu msg=171acb55 + Weston msg=94df61b3 + P0 spec § 3.1.5/§5.2 lock:

- **永远不进 attrs payload**:
  - chunk_text / chunk_content / query_text / user_query / prompt_text / completion_text / messages
  - entity_description / description_text
  - error_message 原文 / traceback / repr(exc)
- **AST data-flow gate** (NOT 全文 grep) 防误伤合法 extraction 路径
- **error 类信息**: 仅 `error_code` / `error_type` whitelist enum (classify_error(exc) 输出), 不带 raw exception
- **高敏感 ID**: tenant_id / query 等 hash or pseudonymize
- **disable switch**: `DISABLE_TELEMETRY=true` 部署级 opt-out (per Planetegg msg=db130d5e P0 directive)

## 8. 不做 (跨 phase 共同 YAGNI)

- 不引入 OpenTelemetry / Jaeger / Datadog SDK (跨 process span tracing, evidence-driven 才 trigger)
- 不做 real-time alert / SLO 计算 (admin metrics endpoint 可手动 query)
- 不做 chunk text / query text 抽样存档 (privacy hard gate 永远 NO)
- 不做 dashboard builder UI (P1 仅 fixed indicator dimensions)
- 不引入新存储 (复用 PG, 私有化部署免维护)

## 9. 跨域协作方对齐确认表

(本文档发 PR + merge 后, 跨域协作方在 PR review 留 ack 即视为对齐)

| 协作方 | scope | ack 形式 |
|--------|-------|---------|
| @符炫炜 (architect) | overall capability map + phase 优先级 + privacy gate 设计 | spec author |
| @不穷 (PM) | 实施时机 + task 拆分粒度 + dispatch | PR review LGTM |
| @earayu2 (decision maker) | P0/P1/P2 trigger condition + cost config table 启动时机 | PR ratify |
| @Planetegg (SRE) | 部署可视化 + retention/cardinality 容量估算 + Helm 接入 | PR review LGTM |
| @Weston (架构师 cross-CR) | LLM ledger 独立 event family 设计 + cross-layer field 一致性 | PR review LGTM |
| @ziang (index/worker) | indexing-worker producer 接入点 + worker lane lifecycle 命名 | PR review LGTM |
| @huangzhangshu (testing) | privacy boundary scope + fail-safe 两类 + 验收口径 | PR review LGTM |
| @dongdong (FE) | dashboard typed schema 三层区分 + raw event 不进 dashboard | PR review LGTM |
| @cuiwenbo (FE typed schema consumer) | typed schema sync 一致性 + capability declaration mirror | PR review LGTM |

## 10. 关联文档

- earayu2 directive: `#indexing优化` msg=1331d5e7 + 飞书 DM (msg=d3042add)
- 太钢 token 追问驱动: 谭怀远 飞书群
- 私有大模型 cost 担忧: `ou_1d75b5...` 飞书
- Singapore 大 PDF 现场: `#indexing优化` Planetegg msg=1314ac59
- P0 实施 spec: [`task-89-telemetry-spec-v1.md`](./task-89-telemetry-spec-v1.md) (PR #1951)
- task #17 任务系统不变式: [`task-system-invariants.md`](./task-system-invariants.md)
- task #61 capability declaration pattern (Lesson #17 backend 收敛 contract): [`task-61-db-adapter-compat-spec-v1.md`](./task-61-db-adapter-compat-spec-v1.md)
- ci-flake-policy: [`ci-flake-policy.md`](./ci-flake-policy.md)

---

**起草**: @符炫炜 (总架构师)
**日期**: 2026-04-30
**版本**: v1 (跨域需求对齐草稿; PR review 收齐协作方 ack 后 earayu2 ratify → merge)
**模式**: per earayu2 飞书 directive — "讨论完成发 PR 合并", 不写实现, 仅指标 capability/字段/维度/口径 alignment
