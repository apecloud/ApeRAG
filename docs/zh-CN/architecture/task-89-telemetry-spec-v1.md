---
title: task #89 — ApeRAG 全路径埋点 spec v1
description: ApeRAG telemetry foundation - per-window/per-document graph extraction metrics + privacy-safe attrs + best-effort fail-safe + DISABLE_TELEMETRY guardrail
---

# task #89 — ApeRAG 全路径埋点 spec v1

> earayu2 directive (`#indexing优化` msg=1331d5e7): 全路径埋点规划 & 落地，先解决线上两个无埋点关键指标 (per-chunk/window 成功耗时 + 平均 per-chunk node/edge 数)。

## 1. 现状 inventory（grep 实证）

### 1.1 现有可见性 surface

- `aperag/indexing/graph_extractor.py::_extract_one_window` — graph extraction 主 hot path, window-level 内部 LLM call + JSON parse + entity/relation aggregation, 当前 **仅 logger.info / logger.warning 输出**, 无结构化埋点
- `aperag/indexing/graph_facts_worker.py` (per ziang msg=785625f5: **不存在独立文件**, GraphModalityWorker 在 worker_factory 层组装) — document-level success/failed wall_time 仅日志, 无累计 counter emit
- `aperag/indexing/orchestrator.py` worker lane lifecycle (claim/release/heartbeat) — 仅日志, 无 queue depth / concurrent_tasks 结构化输出
- `aperag/retrieval/` + `aperag/llm/` — top-K / token count / latency 仅日志, 无埋点

### 1.2 当前 gap (task #89 scope, per earayu2 directive)

1. **window 级耗时 + entity/relation count** 无可查询数据 — Planetegg msg=1314ac59 surface case: 413+1316 chunks × window=2 ≈ 865 window events / doc, 无法 SQL aggregate
2. **document 级汇总** 无累计 counter — windows_total/success/failed/wall_time 只能 `tail -f` log 反推
3. **可见性下游路径** 无 typed schema, FE / admin / SRE 无 machine-readable consumer
4. **私有化部署** 无 SaaS telemetry (Datadog/NewRelic 不可用), 现有 Prometheus/Grafana infra 缺埋点 source

## 2. 缺口识别（按 severity）

### 2.1 P0（必须做 — earayu2 directive 直接覆盖 + 4 guardrail driven）

- **P0-T1** Layer 0 data model: PG `telemetry_event` table + Pydantic schema + Alembic migration + composite index (event_type, ts) + retention/cleanup task (default 30 days)
- **P0-T2** Layer 1 Phase 1 producer: graph extraction window-level + document-level emit (per ziang msg=785625f5 接入点 lock: `_extract_one_window` + `GraphModalityWorker.sync` outer)
- **P0-T3** Layer 2 ingestion: async batched flush (per-process buffer, 100 event / 5 sec) + best-effort fail-safe (DB missing → drop + log) + telemetry fail not propagate to indexing worker state + `DISABLE_TELEMETRY` env switch
- **P0-T4** Layer 4 boundary: privacy AST gate (producer + schema + serializer + attrs builder helper 全 scan) + DISABLE_TELEMETRY gate + fail-safe gate + retention cleanup test
- **P0-T5** Helm `DISABLE_TELEMETRY` env wiring (per Planetegg msg=db130d5e — 私有化部署快速 opt-out, 不留 P1)
- **P0-T6** Producer stats return contract: `_extract_one_window` 显式 return `WindowExtractionStats(duration_ms, entity_count, relation_count, llm_call_count, llm_token_count, status, error_type)` (per Planetegg + Weston: 不靠日志反推); `GraphModalityWorker.sync` 在 run 内累加 counters 为 document summary (per Weston msg=22e6df03 BLOCKER 2: 不实时 aggregate from window events)

### 2.2 P1（数据驱动 follow-up，P0 production data 收集 ≥ 1 周后启动）

- **P1-T1** Layer 1 Phase 2 worker lane lifecycle producer (per ziang msg=785625f5 命名 lock: NOT "Celery" — 现 indexing 是独立 worker 不 Celery hot path; legacy Celery audit 另启 task)
- **P1-T2** Layer 3 admin metrics endpoint (`TelemetryAggregateBucket` / `TelemetryTimeRangeSummary` typed schema 三层区分 per dongdong msg=076bfaec)
- **P1-T3** Grafana datasource docs (per Planetegg P1 — 私有化部署 Grafana 直接 query PG)

### 2.3 P2（性能优化 / 接口语义 — P1 production data 后启动）

- **P2-T1** Layer 1 Phase 3 retrieval + LLM call producer (top_k / embedding_count / token_count / model_id / latency_ms — 不含 prompt/completion text)
- **P2-T2** Aggregation cache table (P1 query 慢时启动)

### 2.4 P3 / YAGNI (defer)

- 不引入 OpenTelemetry / Jaeger / Datadog SDK (跨 process span tracing 是后续 evidence-driven 触发)
- 不做 real-time alert / SLO 计算 (P1 admin metrics endpoint 可手动 query)
- 不做 chunk text / query text 抽样存档 (privacy hard gate 永远 NO)
- 不做 dashboard builder UI (P1 仅 fixed indicator dimensions per dongdong)

## 3. 设计方向（task #89 主线）

### 3.1 必须做（Hard scope per earayu2 directive + 4 guardrail）

#### 3.1.1 Layer 0 — Data model

`aperag/telemetry/db/models.py` (新模块):

```python
class TelemetryEvent(Base):
    __tablename__ = "telemetry_event"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False)  # e.g. 'graph_extraction.window'
    ts = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    collection_id = Column(String(24), nullable=True)  # per Weston msg=a95b2546 NIT: nullable to accommodate P1 worker lane event 没 collection scope (e.g. global queue depth) — P0 graph extraction event 强制 set, P1 worker lane event 可空
    document_id = Column(String(24), nullable=True)  # window/document event 必填; future P1 worker lane event 可空
    document_index_id = Column(String(32), nullable=True)  # per ziang msg=785625f5: 串联同 doc 多次 retry
    parse_version = Column(String(32), nullable=True)
    duration_ms = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False)  # Literal['success', 'failed', 'timeout']
    error_type = Column(String(64), nullable=True)  # whitelist enum, 失败时填 (per huangzhangshu msg=171acb55)
    attrs = Column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("idx_telemetry_event_type_ts", "event_type", "ts"),
        Index("idx_telemetry_collection_event_ts", "collection_id", "event_type", "ts"),
        Index("idx_telemetry_document_event_ts", "document_id", "event_type", "ts"),
    )
```

`aperag/telemetry/schemas.py`:

```python
class WindowExtractionAttrs(BaseModel):
    """Pydantic schema for graph_extraction.window attrs payload."""
    chunk_ids: list[str] = Field(..., max_length=128, description="Window 内 chunk IDs (count capped, ID-only NO text)")
    chunk_count: int
    entity_count: int
    relation_count: int
    llm_call_count: int
    llm_token_count: Optional[int]  # nullable per Planetegg msg=db130d5e (LLM response 不稳定提供时)
    model_id: Optional[str]  # e.g. 'gpt-4o-mini' / 'qwen2.5-72b'
    provider: Optional[str]  # e.g. 'openai' / 'qwen'
    timeout_seconds: Optional[int]
    chunks_truncated: bool = False  # set True when len(input chunk_ids) > 128 (per Weston msg=a95b2546 BLOCKER 2)
    # 注: NOT 含 prompt_text / completion_text / chunk_text / entity_description / error_message — privacy gate

class DocumentExtractionAttrs(BaseModel):
    """Pydantic schema for graph_extraction.document attrs payload."""
    chunks_total: int
    windows_total: int
    windows_success: int
    windows_failed: int
    windows_timeout: int
    entities_total: int
    relations_total: int
    wall_time_ms: int
    # 注: NOT 含 error_message_list / failed_window_details (privacy gate)
```

**Privacy invariant (Layer 4 boundary AST gate enforce, per Weston msg=a95b2546 BLOCKER 1 修订 — 数据流 NOT 全文 grep)**: gate 必须钉「forbidden field 不能 flow INTO `TelemetryEvent.attrs` / `WindowExtractionAttrs` / `DocumentExtractionAttrs` / `telemetry_emit(attrs=...)` 参数」，**不是** indexing 全文 zero match (`aperag/indexing/fulltext.py` / `vision.py` / `summary.py` / `parser.py` + graph_extractor 本身必须读 chunk text 抽取实体, 全文 zero match 会误伤合法路径)。

`attrs` payload **不含** (data-flow constraint):
- `chunk_text` / `chunk_content` / `chunk.text` / `chunk.content` — 仅允许 `chunk_ids` (ID list)
- `query_text` / `user_query` — 不允许进 attrs
- `entity_description` / `description_text` — 仅允许 `entity_count` (count)
- `prompt_text` / `completion_text` / `llm_response` — 仅允许 `llm_token_count` / `model_id` / `provider`
- `error_message` / `traceback` / `repr(exc)` (per huangzhangshu msg=171acb55) — 仅允许 `error_type: str` whitelist enum

仅允许: ID list / count / duration / status enum / error_type whitelist enum / model_id / provider / Pydantic Field-typed primitives

**Boundary AST gate 实施**: 扫描范围 = `aperag/telemetry/**` (全文, telemetry module 自身) + producer call sites (`aperag/indexing/graph_extractor.py::_extract_one_window` 函数 body + `aperag/indexing/worker_factory.py::_build_graph_facts_worker` + `GraphModalityWorker.sync` 函数 body)，**仅 scan 进入 `telemetry_emit(attrs=...)` / `WindowExtractionAttrs(...)` / `DocumentExtractionAttrs(...)` 调用 keyword 参数 + `attrs.update(...)` / `attrs[k]=v`赋值** 的 expression — AST data-flow analysis 钉 forbidden read attribute access (e.g. `chunk.text`, `entity.description`) 不在这些 expression boundary 内。allowlist = Pydantic Field-typed schema (typed payload only, no untyped dict.update)。

**chunk_ids cardinality cap (per Weston msg=22e6df03 BLOCKER 1)**: `chunk_ids` list `max_length=128` (Pydantic validator), 超过截断 + 加 `chunks_truncated: bool` flag — 防 window 含百级 chunk 时 attrs payload 无界膨胀。

**Retention / cleanup (per Weston BLOCKER 1 + Planetegg P0 retention)**:
- `aperag/telemetry/cleanup.py` cleanup task: 删除 `ts < now() - retention_days days` 的 row, 每日跑一次 (cron 或 worker lane integrated)
- `retention_days` env config: 默认 30 (Planetegg 估算 865 event/doc * doc 量级 ≤ 千万 row in 30 days, 单表可承载)
- Helm values 暴露 `telemetryRetentionDays: 30`
- backwards-compat: retention_days = -1 → 不 cleanup (debug 用), 默认 30

#### 3.1.2 Layer 1 — Producer Phase 1

接入点 (per ziang msg=785625f5 lock):

**Window event** (`aperag/indexing/graph_extractor.py::_extract_one_window`):
```python
@dataclass(frozen=True)
class WindowExtractionStats:
    """Producer-returned stats from _extract_one_window (per Planetegg + Weston BLOCKER 2:
    NOT log-derived, NOT realtime-aggregate from telemetry events — extractor 显式 return)."""
    duration_ms: int
    entity_count: int
    relation_count: int
    llm_call_count: int
    llm_token_count: Optional[int]
    status: Literal['success', 'failed', 'timeout']
    error_type: Optional[str]  # whitelist enum classify_error(exc) 输出

async def _extract_one_window(self, chunk_ids, ...) -> tuple[WindowExtractResult, WindowExtractionStats]:
    start = time.monotonic()
    try:
        result = ... # existing extraction logic
        stats = WindowExtractionStats(
            duration_ms=int((time.monotonic() - start) * 1000),
            entity_count=len(result.entities),
            relation_count=len(result.relations),
            llm_call_count=...,
            llm_token_count=...,
            status='success',
            error_type=None,
        )
    except (LLMTimeoutError, ExtractionInvalidJsonError, ...) as exc:
        stats = WindowExtractionStats(
            duration_ms=int((time.monotonic() - start) * 1000),
            entity_count=0,
            relation_count=0,
            llm_call_count=...,
            llm_token_count=None,
            status='failed' if not isinstance(exc, LLMTimeoutError) else 'timeout',
            error_type=classify_error(exc),  # whitelist classifier
        )
        # per huangzhangshu msg=a563d88d implementation detail:
        # 用 structured exception 把 WindowExtractionStats 带给 caller,
        # 保 caller 的 outer finally 能可靠累加 windows_failed/windows_timeout
        # + entity_count/relation_count (即使失败也可能有 partial entities)
        raise WindowExtractionFailed(stats=stats) from exc
    finally:
        # emit telemetry (best-effort, fail-safe per § 3.1.3 Layer 2)
        try:
            telemetry_emit(
                event_type='graph_extraction.window',
                collection_id=self.collection_id,
                document_id=self.document_id,
                document_index_id=self.document_index_id,
                parse_version=self.parse_version,
                duration_ms=stats.duration_ms,
                status=stats.status,
                error_type=stats.error_type,
                attrs=WindowExtractionAttrs(
                    chunk_ids=chunk_ids[:128],  # cardinality cap
                    chunk_count=len(chunk_ids),
                    chunks_truncated=len(chunk_ids) > 128,  # per Weston msg=a95b2546 BLOCKER 2
                    entity_count=stats.entity_count,
                    relation_count=stats.relation_count,
                    llm_call_count=stats.llm_call_count,
                    llm_token_count=stats.llm_token_count,
                    model_id=self.model_id,
                    provider=self.provider,
                    timeout_seconds=self.timeout_seconds,
                ).model_dump(),
            )
        except Exception:
            logger.exception("telemetry emit failed (best-effort drop)")
    return result, stats
```

**Document event** (`aperag/indexing/worker_factory._build_graph_facts_worker` 外层 / `GraphModalityWorker.sync` outer try/finally, per ziang msg=785625f5 接入点修正 — 不存在独立 graph_facts_worker.py 文件 + per Weston msg=a95b2546 BLOCKER 3 outer try/finally guarantee emit exactly once 不论 success/failed/timeout):

```python
class GraphModalityWorker:
    async def sync(self, ...):
        # 在 run 内累加 counters (per Weston msg=22e6df03 BLOCKER 2: 不实时 aggregate from telemetry events)
        doc_stats = DocumentExtractionRunCounters(
            chunks_total=0, windows_total=0,
            windows_success=0, windows_failed=0, windows_timeout=0,
            entities_total=0, relations_total=0,
            wall_time_start_ms=int(time.monotonic() * 1000),
        )
        doc_status: Literal['success', 'failed'] = 'success'

        # OUTER try/finally guarantees document summary emit exactly once
        # regardless of success / window failure re-raise / timeout / crash
        # (per Weston msg=a95b2546 BLOCKER 3 — sample 把 emit 放 end of sync
        # 在 first window failure re-raise 时永远到不了, 必须 outer try/finally
        # cover, telemetry failure 仍不污染 indexing task state per § 3.1.3 Class 2 fail-safe)
        try:
            # ... existing extraction loop, 每 window 累加 doc_stats counters from WindowExtractionStats
            for window_chunks in ...:
                try:
                    _, win_stats = await graph_extractor._extract_one_window(window_chunks, ...)
                    doc_stats.windows_total += 1
                    if win_stats.status == 'success':
                        doc_stats.windows_success += 1
                    elif win_stats.status == 'timeout':
                        doc_stats.windows_timeout += 1
                    else:
                        doc_stats.windows_failed += 1
                    doc_stats.entities_total += win_stats.entity_count
                    doc_stats.relations_total += win_stats.relation_count
                except WindowExtractionFailed as exc:
                    # per huangzhangshu msg=a563d88d: structured exception
                    # carries WindowExtractionStats — caller 累加 actual stats
                    # 不依赖窗口外推 (failed window 的 duration_ms / partial entity_count
                    # 仍 captured + summary windows_failed/timeout 准确)
                    doc_stats.windows_total += 1
                    if exc.stats.status == 'timeout':
                        doc_stats.windows_timeout += 1
                    else:
                        doc_stats.windows_failed += 1
                    doc_stats.entities_total += exc.stats.entity_count
                    doc_stats.relations_total += exc.stats.relation_count
                    doc_status = 'failed'
                    # re-raise per existing logic — outer try/finally 仍 emit summary
                    raise exc.__cause__ if exc.__cause__ else exc
        except Exception:
            doc_status = 'failed'
            raise  # re-raise to preserve existing indexing worker task semantics
        finally:
            # emit document summary best-effort, ALWAYS once regardless of
            # success / partial failure / outer raise (per Weston BLOCKER 3)
            try:
            telemetry_emit(
                event_type='graph_extraction.document',
                collection_id=self.collection_id,
                document_id=self.document_id,
                document_index_id=self.document_index_id,
                parse_version=self.parse_version,
                duration_ms=int(time.monotonic() * 1000) - doc_stats.wall_time_start_ms,
                status='success' if doc_stats.windows_failed == 0 and doc_stats.windows_timeout == 0 else 'failed',
                error_type=None,  # document level 不 classify error_type (具体 error 在 window event)
                attrs=DocumentExtractionAttrs(
                    chunks_total=doc_stats.chunks_total,
                    windows_total=doc_stats.windows_total,
                    windows_success=doc_stats.windows_success,
                    windows_failed=doc_stats.windows_failed,
                    windows_timeout=doc_stats.windows_timeout,
                    entities_total=doc_stats.entities_total,
                    relations_total=doc_stats.relations_total,
                    wall_time_ms=int(time.monotonic() * 1000) - doc_stats.wall_time_start_ms,
                ).model_dump(),
            )
        except Exception:
            logger.exception("telemetry emit failed (best-effort drop)")
```

#### 3.1.3 Layer 2 — Ingestion (async batched flush + fail-safe)

`aperag/telemetry/emitter.py`:
- Per-process in-memory buffer (max 100 event / 5 sec window flush, whichever first)
- async background task flush via `asyncio.create_task` at process startup (per app.py + cli/indexing_worker.py 双 lifespan integrate)
- `DISABLE_TELEMETRY=true` env → producer no-op + buffer never starts (跟 PR #1938 worker fail-safe pattern 一致)

**Best-effort fail-safe (per huangzhangshu msg=171acb55 两类区分)**:
- **Class 1 (DB/table missing)**: `INSERT ... ON CONFLICT DO NOTHING` failure → log + drop event (不阻 hot path)
- **Class 2 (buffer flush exception)**: telemetry emit / flush 异常 NOT propagate 进 indexing worker state — telemetry failure ≠ document/graph task failure (跟 PR #1938 `_mark_run_failed_best_effort` swallow 同 pattern)

```python
async def telemetry_emit(...):
    if _disable_telemetry:
        return
    try:
        _buffer.append(...)
        if len(_buffer) >= _flush_threshold:
            asyncio.create_task(_flush_buffer())
    except Exception:
        logger.exception("telemetry emit failed (drop)")
        # NOT raise — telemetry never blocks hot path

async def _flush_buffer():
    try:
        rows = list(_buffer)
        _buffer.clear()
        async with _engine.begin() as conn:
            await conn.execute(insert(TelemetryEvent), rows)
    except Exception:
        logger.exception("telemetry flush failed (drop batch)")
        # NOT raise — flush failure is silent, indexing path continues
```

#### 3.1.4 Layer 3 — Consumer (P1, 不在 P0 scope)

P1 三层 typed schema 区分 (per dongdong msg=076bfaec):

```python
# RawTelemetryEvent: per-event raw shape (debug-only, optional admin endpoint)
# Used for ad-hoc debugging, NOT consumed by FE dashboard

# TelemetryAggregateBucket: per-time-window bucket (e.g. 1h/1d) with p50/p95/success_rate/count
# Consumed by FE dashboard (NOT raw event)

# TelemetryTimeRangeSummary: dashboard-ready summary
# - per-doc/window aggregates
# - failed_reason distribution (error_type counter)
# - Consumed by FE dashboard (primary)
```

P0 验收 = SQL/query + tests (per Weston msg=22e6df03), NO admin UI / metrics endpoint。FE 在 P1-T2 启动时 consume 这三层 typed schema, 不绑 raw event。

#### 3.1.5 Layer 4 — Boundary tests (Lesson #18 mechanical gate codification)

`tests/boundaries/`:
- `test_telemetry_attrs_privacy_ast_gate.py`: AST scan `aperag/telemetry/**` + `aperag/indexing/**` 全文 (NOT only producer files, per huangzhangshu msg=171acb55) 不含 forbidden read 进 attrs payload — `chunk_text` / `chunk_content` / `query_text` / `entity_description` / `prompt_text` / `completion_text` / `error_message` / `traceback` / `repr(exc)` zero match
- `test_telemetry_failed_event_redaction.py`: failed event `attrs` 不含 `error_message` / `traceback` field (per huangzhangshu msg=171acb55), 仅 `error_type: str` whitelist enum
- `test_telemetry_disable_env_gate.py`: `DISABLE_TELEMETRY=true` 时 `telemetry_emit` no-op + buffer never starts
- `test_telemetry_fail_safe_does_not_propagate_to_indexing_state.py`: telemetry flush exception not raise into indexing worker task state (Class 2 fail-safe)
- `test_telemetry_db_missing_does_not_block_hot_path.py`: PG missing → producer no-op + log (Class 1 fail-safe)
- `test_telemetry_chunk_ids_cardinality_cap.py`: window event attrs `chunk_ids` truncated at 128 + `chunks_truncated: bool` flag set (per Weston BLOCKER 1)
- `test_telemetry_retention_cleanup.py`: cleanup task delete events older than `retention_days`

`tests/unit_test/contracts/`:
- `test_telemetry_event_type_naming_invariant.py`: P0 event_type strings (`graph_extraction.window` / `graph_extraction.document`) lock as immutable identifiers — future P1/P2 加新 event_type 不改老 string (Lesson #14 enum invariant 删除多轮迭代收尾 family)

### 3.2 P1 实施 (数据驱动 follow-up，P0 production data 收集 ≥ 1 周后启动)

per § 2.2 P1-T1+T2+T3 — worker lane lifecycle producer (NOT Celery per ziang msg=785625f5 命名 lock) + admin metrics endpoint with 三层 typed schema + Grafana docs

### 3.3 不做（YAGNI per § 2.4）

## 4. 实施 sub-task 拆分（per Weston msg=22e6df03 BLOCKER 3 task 依赖顺序）

### Phase 1 (foundation PR — Weston BLOCKER 3 优先级)

**X1 + X3 foundation (单 PR, 必须先于 X2)**:
- task #X1 Layer 0 data model: `aperag/telemetry/db/models.py` + Alembic migration + Pydantic schemas (`WindowExtractionAttrs` / `DocumentExtractionAttrs`) + 3 composite index
- task #X3 Layer 2 ingestion: `aperag/telemetry/emitter.py` async batched flush + DISABLE_TELEMETRY env switch + fail-safe (Class 1 + Class 2) + buffer lifecycle integrate `app.py` + `cli/indexing_worker.py`
- 推荐 owner: @Bryce 或 @ziang (熟 worker fail-safe pattern + indexing infra)

### Phase 2 (producer + boundary, 依赖 Phase 1 merged)

**X2 producer (单 PR)**:
- `aperag/indexing/graph_extractor.py::_extract_one_window` window event emit + `WindowExtractionStats` return
- `GraphModalityWorker.sync` document summary emit + in-run counters (per ziang msg=785625f5 接入点 + Weston BLOCKER 2 in-run counters NOT realtime aggregate)
- 推荐 owner: @ziang (熟 graph extraction + GraphModalityWorker domain)

**X4 boundary (单 PR)**:
- 7 boundary test files (privacy AST gate + DISABLE_TELEMETRY gate + 2 fail-safe gate + chunk_ids cardinality cap + retention cleanup + event_type naming invariant)
- 推荐 owner: @huangzhangshu (testing primary + 冬柏 peer review)

### Phase 3 (deploy verify)

**X5 Helm + retention SRE review (单 PR)**:
- Helm `DISABLE_TELEMETRY` env wiring (default false, opt-out via values)
- `telemetryRetentionDays` Helm values (default 30)
- 推荐 owner: @Planetegg (per Planetegg msg=db130d5e)

### Phase 4 (P1, 数据驱动 follow-up)

P1 task X6/X7/X8 等 P0 production data ≥ 1 周后 PM + earayu2 决策启动时机:
- X6 Layer 1 Phase 2 worker lane lifecycle producer (推荐 @ziang)
- X7 Layer 3 admin metrics endpoint + 三层 typed schema (推荐 @cuiwenbo + @dongdong)
- X8 Grafana datasource docs (推荐 @Planetegg)

## 5. 验收口径

### 5.1 P0 完成标准 (per Weston msg=22e6df03: SQL/query + tests, NO UI)

- `telemetry_event` table land + Alembic migration + 3 composite index
- `aperag/telemetry/emitter.py` async batched flush + DISABLE_TELEMETRY env + fail-safe Class 1+2
- `_extract_one_window` window event emit + `GraphModalityWorker.sync` document summary in-run counters emit
- 7 boundary test 全过
- Helm `DISABLE_TELEMETRY` + `telemetryRetentionDays` values 暴露
- SQL query 可 aggregate (示例):
  ```sql
  -- per-doc window count + entity/relation total + wall time
  SELECT collection_id, document_id, parse_version, attrs
  FROM telemetry_event
  WHERE event_type = 'graph_extraction.document'
  ORDER BY ts DESC LIMIT 10;

  -- per-collection window p50/p95 duration + failed rate
  SELECT collection_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms,
    SUM(CASE WHEN status = 'failed' OR status = 'timeout' THEN 1 ELSE 0 END)::float / COUNT(*) AS fail_rate
  FROM telemetry_event
  WHERE event_type = 'graph_extraction.window' AND ts > now() - interval '24 hours'
  GROUP BY collection_id;
  ```

### 5.2 boundary test gate（CI must pass）

- 现有 G1-G19 + telemetry boundary 7 test 不破坏
- privacy AST gate scan 范围: `aperag/telemetry/**` (全文) + `aperag/indexing/**` (producer call sites) 全文 grep zero match forbidden read
- DISABLE_TELEMETRY env 默认 false, set true 时 producer no-op + buffer never starts assertion
- fail-safe Class 1+2 separately tested
- chunk_ids cardinality cap 128 truncate + flag set
- retention cleanup task delete `ts < now() - retention_days` row
- event_type naming invariant lock (P0 strings 不可改)

### 5.3 e2e smoke

- Planetegg msg=db130d5e 验收 case: 413+1316 chunks 大文档 RUNNING — 能 SQL query 看到 window 级失败 / document 级累计 / 最终 kg 落库摘要
- DISABLE_TELEMETRY=true 时 producer 0 emit verify
- PG missing 时 indexing 主路径 continue 不阻

## 6. CR mandatory checklist

按 `task-17-cr-review-checklist.md` 既有 framework + huangheng PR #1932/#1943 sediment family 应用:

- **Lesson #11 v5**（entry-point migration cross-process parity）— `aperag/telemetry/emitter.py` buffer lifecycle 加进 `app.py` + `cli/indexing_worker.py` 双 lifespan + boundary test 钉死
- **Lesson #12 v9**（first-principles verify catch surface signal mistakes）— attrs payload 必 grep main verify 不含 forbidden read (privacy AST gate first-application demo)
- **Lesson #13 v3**（dual-side rewrite + cross-source default value alignment）— Helm values + spec + boundary test + Pydantic Field 跨 source 同步
- **Lesson #14**（架构 invariant 删除多轮迭代收尾）— P0 event_type strings 锁定 immutable, P1/P2 加新不改老 (类比 Wave 5 description-NULL invariant family)
- **Lesson #16**（workflow paths filter dead reference）— 新 telemetry module 加 `compat-test.yml` paths filter 同步
- **Lesson #17**（backend 收敛 contract / simple-stable family）— Layer 0 schema lock + Layer 3 P1 三层 typed schema 区分, FE 仅消费不分支
- **Lesson #18 candidate**（lesson sediment + mechanical gate 双 layer codification）— 7 boundary test = mechanical gate first-application demo for telemetry privacy + DISABLE_TELEMETRY + fail-safe
- **mini-pattern 19**（spec lock pre-check grep main 实证）— 接入点 `_extract_one_window` + `GraphModalityWorker` per ziang msg=785625f5 grep main 实证 (NOT 假设独立 graph_facts_worker.py 文件存在)
- **mini-pattern 20 候选**（response_model wire-up boundary gate）— P1 admin metrics endpoint 启动时 fold (P0 不做 endpoint)
- **简单稳定 + 私有化部署免维护 4 guardrail**

## 7. 关联文档

- earayu2 directive: `#indexing优化` msg=1331d5e7 (task #89 创建)
- Planetegg P2-HIGH 起点: `#indexing优化` msg=1314ac59 (window 级 + document 级 minimum 埋点建议)
- task #17 任务系统不变式: [`task-system-invariants.md`](./task-system-invariants.md)（worker lane / API 不拥有执行面 hard gate — telemetry 走 indexing-worker process 不进 API path）
- task #31 spec v1.1: [`task-31-graph-node-merge-spec-v1.md`](./task-31-graph-node-merge-spec-v1.md)（worker fail-safe invariant pattern 复用）
- task #61 spec v1: [`task-61-db-adapter-compat-spec-v1.md`](./task-61-db-adapter-compat-spec-v1.md)（capability declaration pattern 类比）
- cr-checklist accumulated sediment: [`task-17-cr-review-checklist.md`](./task-17-cr-review-checklist.md)
- ci-flake-policy: [`ci-flake-policy.md`](./ci-flake-policy.md)（telemetry boundary test 加进 § 2.1 Lite 必绿 list）

## 8. 不阻塞主线

本 spec **不阻塞**:
- task #87 PR #1949 cuiwenbo lint fix-forward + CI 绿 → squash merge
- task #88 P2-S1+S2 batch alias resolution (Bryce in flight)
- task #61 P1 5/5 闭环 (#83/#84/#85/#86 done + #87 收尾)
- task #11 GC orphan vector follow-up
- huangheng follow-up sediment future 子 PR

---

**起草**：@符炫炜（总架构师）
**日期**：2026-04-30
**版本**：v1（task #89 spec lock 候选；@Weston 架构 CR + earayu2 ratify 后 PM @不穷 按 Phase 1 → 2 → 3 → 4 调度实施 PR）

input fold-in trail:
- @Planetegg msg=db130d5e (DISABLE_TELEMETRY P0 + retention P0 + producer stats return)
- @Weston msg=22e6df03 (retention/cardinality budget + in-run counters NOT realtime aggregate + foundation PR 依赖顺序 X1+X3 → X2+X4 + P0 验收 SQL+tests NOT UI)
- @huangzhangshu msg=171acb55 (privacy AST gate scan 范围扩展 + fail-safe 两类区分 + failed event error_type whitelist NO error_message)
- @ziang msg=785625f5 (接入点修正 `_extract_one_window` + `GraphModalityWorker` NOT graph_facts_worker.py + worker lane 命名 NOT Celery + document_index_id retry 串联)
- @dongdong msg=076bfaec + msg=414b7554 (P1 三层 typed schema 区分 + FE consume aggregated only NOT raw event)
