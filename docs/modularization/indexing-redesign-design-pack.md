# ApeRAG Indexing Redesign — Design Pack v2

**Owner**: 符炫炜 (chief architect, sole design owner per `#celery msg=d8080c08`)
**Status**: v2 — incorporates earayu2 拍板 (`msg=cc0a00d7`) + PM consolidation (`msg=32463d64`)
**Date**: 2026-04-26
**Trigger**: earayu2 directives `#celery msg=56812dd6` + `msg=d8080c08` + `msg=cc0a00d7`

## v2 changelog

This v2 supersedes v1. Changes driven by earayu2 拍板:

| 决策 | v1 提法 | v2 落点 |
|---|---|---|
| Celery 去留 | 列三选项让 earayu 选 | **接受 hard cut → Redis + asyncio**（§E 简化为已选方案，删 decision matrix 主体） |
| 原子 flip | 双列 `active/pending` + all-modalities-ACTIVE 才翻 | **删除原子 flip**；每个 modality 独立可见；接受短暂不一致（§F 大改） |
| derived/ 内容 | 概念性描述 | **§C.6 显式列出**：每模态什么是 canonical artifact、什么是可重建缓存 |
| 对象存储能否承载 | 未答 | **§C.7 明确答**：MinIO / S3-compatible / 本地 FS 都可承载本设计读写模型 |
| 私有化交付 | 未提 | **新增 §L**：deploy-and-forget / 弱运维 / SQLite-friendly 路径 |
| 多租户演进 | 仅 collection_id | **§H** 加 future `organization` forward-compat note |
| 配额管理 | 资源类 admission control 详尽设计 | **§I.4 / §H.5** 简化为 Redis token bucket，留接口给未来 tenant fairness |
| PR 拆分 | 7 个 PR | **§K 改为 3 个 wave**：少 PR、可并行、少上下文切换 |

不变项: §A（现状分析）/ §B（第一性原理）/ §C.1-§C.5（三层模型）/ §D（幂等 contract）/ §G（modality 统一 pipeline）/ §J（observability）。这些是 earayu2 已默认接受 / 没拍板要改的部分。

## Preface

ApeRAG is **pre-launch** (no users, no production data, no migration), targeting **私有化交付 + deploy-and-forget**（earayu2 `msg=cc0a00d7`）— 客户拿到包能跑、跑起来不用我们管。当前 Celery-based 索引方案是存量包袱，earayu2 已授权 hard-cut。

The architect's recommendation, in one sentence: **drop Celery; treat object-store-as-source-of-truth with per-modality derived artifacts (`chunks.jsonl` / `kg.jsonl` / `summary.json` / `vision/manifest.jsonl`); use a thin Redis-backed asyncio worker pool; let the database hold the state machine on `(document_id, parse_version, modality)` triples; flip per-modality independently and accept short eventual-consistency windows.**

Sections:
- §A — Current system analysis (with file:line evidence) — unchanged
- §B — First principles — unchanged
- §C — Three-layer document model + **§C.6 derived/ contents per modality** + **§C.7 object-store suitability**
- §D — Idempotency contract per modality — unchanged
- §E — Concurrency model: Redis + asyncio (decision locked)
- §F — State machine + **per-modality independent visibility** (atomic flip removed)
- §G — Multi-modal unified pipeline
- §H — Multi-tenant isolation (simple now, organization-ready later)
- §I — Failure recovery (with simple Redis token-bucket quota)
- §J — Observability
- §K — Migration plan: **3 waves, parallel-friendly**
- §L — Private / on-premise deployment ("deploy-and-forget")

---

## §A. Current system analysis

Repository state: `main @ 704b3cf3` (2026-04-26, post-D10.h cutover + Phase 8 D8.6 chunk-3 cleanup).

### A.1. Architecture overview

The current system has three coordinating layers:

```
┌─ User upload ────────────────────────────────────┐
│  POST /api/v2/collections/<id>/documents         │
└─────────────────┬────────────────────────────────┘
                  │ creates 5 DocumentIndex rows (PENDING)
                  ▼
┌─ DocumentIndexManager (synchronous DB write) ────┐
│  aperag/domains/indexing/manager.py              │
└─────────────────┬────────────────────────────────┘
                  │ (no direct task dispatch — relies on reconciler)
                  ▼
┌─ DocumentIndexReconciler (Celery beat schedule) ─┐
│  aperag/tasks/reconciler.py:54-81                │
│  - Polls DB every 30s for PENDING/stale rows     │
│  - Atomic claim: PENDING → CREATING + token      │
│  - Generates processing_token, sets lease        │
│  - Schedules Celery workflow                     │
└─────────────────┬────────────────────────────────┘
                  ▼
┌─ Celery workflow chain ──────────────────────────┐
│  parse_document_task                             │
│   └─→ trigger_create_indexes_workflow            │
│        └─→ chord(group([                         │
│              create_index_task(VECTOR),          │
│              create_index_task(FULLTEXT),        │
│              create_index_task(GRAPH),           │
│              create_index_task(SUMMARY),         │
│              create_index_task(VISION),          │
│            ]), notify_workflow_complete)         │
└─────────────────┬────────────────────────────────┘
                  ▼
┌─ Per-modality indexers (5 separate code paths) ──┐
│  aperag/domains/indexing/{vector,fulltext,graph, │
│    summary,vision}_index.py                      │
│  - Each reads parsed parts from worker memory    │
│  - Each writes to its own backend                │
│  - Each tags collection_id in metadata           │
└─────────────────┬────────────────────────────────┘
                  ▼
┌─ Backends ───────────────────────────────────────┐
│  Qdrant (vector + vision + summary)              │
│  Elasticsearch (fulltext)                        │
│  Nebula or Neo4j (graph)                         │
└─────────────────┬────────────────────────────────┘
                  │ on each task complete
                  ▼
┌─ IndexTaskCallbacks ─────────────────────────────┐
│  aperag/domains/indexing/manager.py callbacks    │
│  - on_index_created → DocumentIndex.status=ACTIVE│
│  - on_index_failed  → status=FAILED              │
│  - on_index_deleted → row deleted                │
└──────────────────────────────────────────────────┘
```

### A.2. State machine (DocumentIndex)

`aperag/domains/indexing/db/models.py:71-80`:

```python
class DocumentIndexStatus(str, Enum):
    PENDING                = "pending"
    CREATING               = "creating"           # claimed + lease active
    ACTIVE                 = "active"
    DELETING               = "deleting"
    DELETION_IN_PROGRESS   = "deletion_in_progress"
    FAILED                 = "failed"
```

Key fields on `DocumentIndex`:
- `(document_id, modality)` — composite unique key (no `parse_version` dimension)
- `status` — the enum above
- `version` — bumped on document re-parse (separate from `parse_version` in cache layer)
- `processing_token` — opaque string identifying the current task instance
- `lease_expires_at` — when the lease must be renewed; otherwise reconciler reclaims

### A.3. Ownership model — three layers, none authoritative

This is the structural defect Bryce correctly identified (msg=791082a4 §1):

| Layer | Role | Source of truth? |
|---|---|---|
| Celery task queue | "task is enqueued / running" | No — Celery has no native lease |
| `ProcessingLeaseRenewer` (Python thread) | "I still own this work" | No — dies with worker |
| `DocumentIndex.status` | "this document/modality is in state X" | Logical SoT, but updated as side-effect of task callbacks |

The reconciler (`aperag/tasks/reconciler.py`) exists because none of these three layers is reliable on its own. It periodically scans for inconsistencies — stale CREATING (lease expired) → reclaim; orphan tasks → ignore via token mismatch; etc. PR #1486 fixed one specific 3-layer skew (the "fake in-progress window") but the root cause remains.

### A.4. Lease renewal — Python thread tied to worker process

`aperag/tasks/processing_lease.py:38-84`:

```python
class ProcessingLeaseRenewer:
    def __init__(self, ..., renew_interval_seconds: int = 60, ...):
        self._thread = threading.Thread(target=self._renew_loop, daemon=True)
    def start(self):
        self._thread.start()
    def _renew_loop(self):
        while not self._stop.is_set():
            time.sleep(self._renew_interval_seconds)
            self._renew_in_db()
```

Problems:
- Daemon thread dies when Celery worker process dies → no graceful lease release
- 60s renewal interval × default 900s (15 min) TTL → if worker crashes mid-task, the document stays stuck CREATING for up to 15 minutes
- Uses Python `threading` inside async-friendly Celery — earayu2 explicitly flagged this as "Celery 用得不太对" / "通过 Thread 的方法去让它跑"

### A.5. Graph index complexity (earayu2's specific complaint)

`aperag/domains/indexing/graph_index.py` and `aperag/domains/indexing/tasks.py:464-756`:

The flow earayu2 describes as "在 Celery 的 Task 里绕来绕去，最后又绕回了 Graph Index 的代码文件" is concretely:

1. `create_index_task(GRAPH, ...)` (in `tasks.py`) — handles lease, ownership check, Celery retry
2. → calls `graph_index.create_index(parsed_parts, collection)` (in `graph_index.py`)
3. → calls `aperag/graphindex/v2.run_index_document_sync(content, file_path, ...)` (in a different package)
4. → which loads its own LLM service, extracts entities, calls `nebula.upsert_entities(...)` (in `aperag/graphindex/storage/nebula.py:354`)
5. → returns to `graph_index.py`
6. → returns to `create_index_task`
7. → fires callback `on_index_created` (in `manager.py`)

Seven hops across three packages for one indexing operation. Compare to `vector_index.py` which is a single `chunk → embed → store` flow in one file.

### A.6. Graph index NOT replace-idempotent (Bryce msg=38fbf962 hard data)

`aperag/graphindex/storage/nebula.py:354 upsert_entities`:

```python
# Fetches existing description / source_chunk_ids from Nebula
# Appends new content
# Writes back
```

Behavior: every retry **appends** to the existing entity, doubling description text and source_chunk_id arrays. A failed-and-retried index produces a corrupted knowledge graph that is silently wrong (no error, but every search retrieves duplicate-looking entities).

`aperag/graphindex/storage/neo4j.py` uses Cypher `MERGE` (replace semantics — better) but still has no `parse_version` dimension, so re-indexing the same document under a new parse never purges old entries.

This is the single biggest functional bug in the current system.

### A.7. Reconciler — polling, no priority

`aperag/tasks/reconciler.py:54-81`:

- Runs on Celery beat schedule (default 30s)
- Single SELECT scans for `status IN (PENDING, DELETING) OR (status IN (CREATING, DELETION_IN_PROGRESS) AND lease_expires_at < now)`
- No tenant fairness, no priority queue
- Worst case: 10K documents uploaded, all wait until next reconcile cycle, then all fan out simultaneously (possibly overwhelming downstream)

### A.8. Multi-tenant isolation — implicit

Tenancy enforcement lives in:
- `Collection.user` (DB-level FK) — checked at upload time, not at index time
- `DocumentIndex.collection_id` and `document_id` — passed through to indexers, tagged in chunk metadata
- Backend storage:
  - Qdrant: per-tenant collection name (`generate_vector_db_collection_name(collection_id)`)
  - Elasticsearch: shared index, `routing=str(collection_id)` (advisory only — direct queries can bypass)
  - Nebula: implicit (graphindex v2 internal partitioning, opaque)

Issues:
- A worker that mishandles `collection_id` could write into wrong tenant's Qdrant collection (no defense in depth)
- ES routing is non-enforced
- Processing token has no tenant context — if a token leaks/is replayed, no tenant validation prevents cross-tenant action
- No per-tenant concurrency cap → noisy neighbor

### A.9. Concurrency control — application-level only

`aperag/concurrent_control/redis_lock.py` provides Redis distributed locks but is **not used** in the indexing flow. Concurrency control is via `processing_token` + `version` checks in `aperag/domains/indexing/tasks.py:139-198` (`_validate_task_relevance`). This works as an application-level pessimistic lock but:
- Token validation happens only at task entry (no checkpoint inside long tasks)
- Cancellation (e.g. document deletion) is "soft" — the in-flight task continues until next checkpoint, which only exists at task entry
- A 30-second LLM call cannot be interrupted mid-flight

### A.10. Code complexity scorecard

| File | Lines | Complexity signal |
|---|---|---|
| `aperag/domains/indexing/tasks.py` | 995 | Mixes infra (lease, version check, retry) with business (chunking, embedding, store). Too large. |
| `aperag/tasks/reconciler.py` | ~600 | 3 dispatch branches × 5 modalities × claim/reclaim/dispatch — high cyclomatic complexity |
| `aperag/domains/indexing/orchestration.py` | ~300 | Celery chord composition + workflow status aggregation |
| `aperag/domains/indexing/graph_index.py` + `aperag/graphindex/v2/*` | 1500+ | Cross-package indirection per A.5 |
| `aperag/domains/indexing/vision_index.py` | ~250 | Multimodal detection logic spread across `is_enabled` and `create_index` |

The indexing layer is doing **too many jobs at once**:
1. Document parsing (parse PDF → markdown + image parts)
2. Chunking + embedding (CPU/API)
3. Backend writing (per-modality)
4. State machine maintenance (DB writes via callbacks)
5. Lease management (Python thread)
6. Retry and backoff (Celery retry decorator)
7. Cross-modality coordination (Celery chord)

A clean redesign separates these into independent layers with well-defined contracts.

### A.11. What works and shouldn't be thrown away

earayu2 explicitly listed these as good:
1. Reconciliation with auto-retry — keep, but simplify
2. Lifecycle state machine for indexes — keep, but add `parse_version` dimension
3. Fault tolerance for LLM rate-limits via reconciliation retry — keep, but with backpressure rather than blind retry

These are correct and stay. The redesign keeps the **observable behavior** of these three properties while changing their implementation.

---

## §B. First principles

The system's job is to take a `(document_id, parse_version, modality)` triple from input state to a stable, retrievable state. **Tasks, queues, leases, locks are means, not ends.** State is the goal; everything else exists to converge state.

Five first-principles propositions follow:

### B.1. Single source of truth

There is exactly **one** authoritative store for state: a relational database row. Everything else (queue messages, in-memory caches, derived artifacts on disk) is a projection or notification, not a source of truth. The reconciler exists because the database is, by definition, the only place that survives all worker crashes, all queue restarts, and all network partitions.

Implication: tasks never decide state. Tasks compute, then they tell the database what they computed; the database decides whether to accept. If the database says "you are stale" the task discards its work silently.

### B.2. Convergence is idempotent

Re-running the same `(document_id, parse_version, modality)` operation must produce the same final state at the backend, regardless of how many times it ran or how it was interrupted. This is **non-negotiable** — without it, retry is unsafe, concurrent reconciliation is unsafe, and the system cannot be reasoned about.

Implication: every backend write is `delete-then-insert by (document_id, parse_version)` or an upsert that fully replaces (not appends). The graph index's current append behavior (§A.6) is the canonical violation.

### B.3. Source separation

The original document file (PDF, DOCX, etc.) is one thing. The parsed/extracted/embedded artifacts derived from it are a separate thing. The backend index entries are a third thing. These three should live in three separate places, with clear data-flow direction:

```
source file  →  derived artifacts  →  backend index
   (immutable for one upload)   (regenerable from source)   (regenerable from artifacts)
```

This is the pattern earayu2 sketched: original files in one directory, derived files (jsonl, markdown, etc.) in a parallel directory, then sync into the graph database. The architecture formalizes this as the **three-layer document model** in §C.

Implication: any modality's "index" is the result of:
1. Read source
2. Compute derived artifact (e.g., `kg.jsonl` for graph)
3. Sync derived artifact into backend (e.g., upsert Nebula entries from `kg.jsonl`)

Each step is independently restartable. Failure at step 3 doesn't lose work from step 2.

### B.4. Concurrency is bounded by external capacity, not internal architecture

The indexing service itself does almost no CPU work. The bottlenecks are:
- LLM API rate limits (graph extraction, summary)
- Embedding API rate limits (vector, vision)
- GPU memory (vision local inference, if applicable)
- Database write throughput (relatively cheap)

This means: 100 concurrent documents is a **scheduling and rate-limiting problem**, not a parallelism problem. A single Python process with asyncio can supervise 500+ concurrent in-flight operations as long as those operations are mostly waiting on external APIs.

Implication: complex worker-pool architectures (Celery's prefork pool with worker concurrency) are overkill. A simpler model — small number of asyncio workers + per-resource token bucket — handles 100 concurrent docs with less code.

### B.5. Simple > complex

When two designs achieve the same correctness, the simpler one wins. "Simpler" means:
- Fewer files
- Fewer abstractions
- Fewer cross-package indirections
- Code that a new contributor can read top-to-bottom in one sitting and understand

This is earayu2's explicit preference. The redesign is graded by simplicity at every decision.

---

## §C. Three-layer document model

The redesign formalizes earayu2's "原始文件 + 派生 jsonl + 图数据库" sketch.

### C.1. Storage layout

Per document, the object store (or filesystem in dev) holds a directory tree:

```
collections/<collection_id>/documents/<document_id>/
├── source/
│   └── original.<ext>           ← the user's uploaded file (immutable for one upload)
├── derived/
│   └── parse_<parse_version>/
│       ├── markdown.md          ← parsed full-text markdown (from docparser)
│       ├── outline.json         ← heading tree (D10.c §A.6 outline)
│       ├── chunks.jsonl         ← text chunks for vector + fulltext (one chunk per line)
│       ├── kg.jsonl             ← knowledge-graph entities + relations (one record per line)
│       ├── summary.json         ← document-level summary
│       └── vision/
│           ├── manifest.jsonl   ← list of image parts + their derived embeddings
│           └── images/          ← extracted image blobs (for re-encoding)
└── meta.json                    ← collection_id, mime_type, parse_version history
```

`parse_version` is a 16-char hex hash (per D10.g spec §E.2) of `(parser_pipeline, document_md5, chunking_config)`. Re-uploading the same content produces the same `parse_version` and reuses the same derived directory.

### C.2. Layer responsibilities

| Layer | What it holds | Who writes | Who reads |
|---|---|---|---|
| **Source** (`source/`) | the original uploaded file | upload handler (write-once) | parser only |
| **Derived** (`derived/parse_<v>/`) | per-modality artifacts derived from source | parser + per-modality "deriver" workers | per-modality "syncer" workers |
| **Index** (Qdrant / ES / Nebula / Redis) | retrievable indexed state | per-modality "syncer" workers | search service |

### C.3. Why this layout matters

1. **Idempotent retry is trivial.** If syncing `kg.jsonl` into Nebula fails halfway, the next attempt re-reads the same `kg.jsonl` and re-syncs. No need to re-call the LLM (which produced `kg.jsonl`). The expensive step (LLM extraction) and the cheap step (DB upsert) are decoupled.

2. **Updates are clean.** A document content change → new `parse_version` → new derived directory. Old directory remains untouched. Atomic flip happens by updating one DB row pointer (active parse_version). Old directory garbage-collected after flip.

3. **Debugging is concrete.** Operators can `cat` the derived files to see exactly what was extracted. No need to dump Celery state or query backend stores.

4. **Per-modality reasoning is local.** Each modality has one input file (its derived artifact) and one output (its backend store). No cross-modality interaction during sync.

5. **Backend swap is cheap.** Switch Nebula → Neo4j → PostgreSQL graph without touching the deriver. Re-run "sync" against the new backend, reading the same `kg.jsonl`.

#### C.3.1. Per-collection parser override（Wave 4 T3 chunk 2 amendment, architect msg=9a6de002）

`collection.config.parser_config` is an optional dict forwarded verbatim to `DocParser` dispatch (`parse_document(parser_config=...)`). Operators / private deployments can opt into MinerU / per-collection OCR engines without touching code. The async parse worker reads it via `_resolve_parser_config_for_collection(collection_id)` (Wave 4 T3 chunk 2) and pins it onto the `ParseDispatchPayload` so a re-parse N minutes later still uses the collection's chosen parser config.

Shape: any JSON-serialisable dict; the chosen parser (`docparser` family) decides what keys it understands. Ill-formed values are graceful — the parse worker logs a warning and falls back to no-op `parser_config=None` rather than failing the parse.

### C.4. Object store choice

The object store is whatever ApeRAG already uses (S3 / MinIO / local filesystem in dev). This design pack does not impose a new dependency — the directory layout overlays the existing object store.

### C.5. What does not live in this layout

- Cache: `aperag/cache/` (D10.g) keeps its own L1/L2 read-primitive cache. It reads from object store but does not own state.
- Index state machine: lives in the database (`DocumentIndex` table), not on disk.
- Tenancy: `collection_id` is part of the path, but the database is the authority on who owns the collection.

### C.6. Derived/ 里到底产哪些东西（answer to earayu2 `msg=cc0a00d7` Q1）

> 原问: "derived/parse_<version>/中生成哪些东西？graph 肯定要，chunk 要吗？vector 要吗？fulltext 呢？"

直接答：**所有 5 个 modality 都把"输入到 backend 之前的最后形态"落到 derived/，作为可重建、可 diff、可复用的中间产物。** 这样每次重试只读 derived/，不重跑 LLM / embedding。

每个文件的角色分两类：
- **CANONICAL artifact（必须落盘）**：包含 LLM 调用 / embedding 调用 / GPU 推理产出的"贵"结果。丢了要重花钱。
- **CACHE artifact（可重建，建议落盘）**：解析出来的中间产物，不贵但每次重算浪费。

```
collections/<cid>/documents/<did>/derived/parse_<v>/
├── markdown.md             ← CACHE: 解析器产出的全文 markdown（docparser 输出）
├── outline.json            ← CACHE: 标题树 / section_path 索引（D10.c §A.6）
├── chunks.jsonl            ← CANONICAL: 每行一个 chunk，{chunk_id, text, embedding[], section_path, heading_anchor, page_idx}
│                              vector + fulltext 共用此文件。embedding 是贵的，必须落盘。
├── kg.jsonl                ← CANONICAL: 每行一个 entity 或 relation，{type, document_id, parse_version, name, description, source_chunk_ids[]}
│                              graph LLM 抽取产出，最贵的产物。
├── summary.json            ← CANONICAL: {summary_text, embedding[]}，单文档级 summary
├── vision/
│   ├── manifest.jsonl      ← CANONICAL: 每行一张图，{image_id, image_path, embedding[], alt_text, page_idx, bbox}
│   └── images/             ← CANONICAL: 抽取出的图片 blob（PDF 内嵌图片），重抽要解析 PDF
└── _meta.json              ← parse_version, parser_pipeline, chunking_config, derived_at
```

**逐模态明确**:

| Modality | derived/ 文件 | 类型 | 贵在哪 | 重试时是否要重产 |
|---|---|---|---|---|
| **Vector** | `chunks.jsonl`（embedding 字段） | CANONICAL | embedding API 调用费用 | NO — 只重做 sync |
| **Fulltext** | `chunks.jsonl`（text 字段，与 vector 共用） | CANONICAL | 与 vector 共享，无独立成本 | NO |
| **Graph** | `kg.jsonl` | CANONICAL | LLM entity/relation 抽取（最贵） | NO — 只重做 nebula sync |
| **Summary** | `summary.json` | CANONICAL | LLM summary + embedding | NO |
| **Vision** | `vision/manifest.jsonl` + `vision/images/` | CANONICAL | 图片抽取 + vision-LLM/embedding | NO |
| **Parser 共享** | `markdown.md` + `outline.json` | CACHE | docparser CPU/IO 时间 | 可重产，但落盘省时 |

**关键设计 invariant**: 任何 backend sync 失败重试时，worker 只 `read derived/parse_<v>/<file>` + 重写 backend，**绝不重新调用 LLM / embedding API**。这把"贵的不可重做"和"便宜的可重做"在物理上分开。

**`chunks.jsonl` 给 vector + fulltext 共用一份 — conscious trade-off（Bryce v2 review msg=7ccb176f #1 surface）**

最优 chunking 策略两个 modality **不同**：
- vector 偏大 chunk（800-1500 tokens）以保 semantic 完整性 + embedding context 利用率
- fulltext 偏小 chunk（200-400 tokens）以保 keyword precision + short-query recall 不被稀释

强制共用 = 两者都用妥协 chunk_size，理论 vector recall 和 fulltext precision 都会比独立优化时略低。**v2 选择共用是 conscious simplification**，理由：
- pre-launch 阶段，hybrid search dedup 按 `chunk_id` 对齐是硬要求；两份独立 chunks 边界对齐是另一个复杂度源（chunk_id 跨两份文件 collisions / mappings）
- 上线后真观测到 fulltext precision 显著低再 split，不在没数据时预先优化

**未来 split 扩展点（不锁死）**:
- `chunks.jsonl` 永远是 base canonical，含 vector 用的 token 范围
- 若未来要 split：新加 `chunks.fulltext.jsonl` 作 shadow 文件（fine-grained subdivision），fulltext 模态切到读 shadow；vector 继续读 `chunks.jsonl`
- shadow 文件的 chunk_id 用 `<base_chunk_id>:<sub_idx>` 命名空间，向上兼容 hybrid dedup
- 整改是 fulltext modality 单文件改动 + alembic 加 `fulltext_artifact_path` 列；不动 vector / graph / summary / vision

这条扩展路径明示，避免 1 年后想优化时被锁死。

**两个 deletion 行为**:
- 用户删除文档 → 整个 `collections/<cid>/documents/<did>/` 目录走 cleanup worker GC。
- 文档内容更新 → 新 `parse_version` 新建 `derived/parse_<v_new>/`；旧 `derived/parse_<v_old>/` 被 cleanup worker 在 1 小时后 GC（§F.5）。

### C.7. 对象存储能否承载（answer to earayu2 `msg=cc0a00d7` Q2）

> 原问: "我目前部署在线上貌似是 minio 以及对象存储？能承载读写吗？"

直接答：**能。MinIO / S3-compatible / 本地文件系统都可以承载本设计的读写模型。** 不需要换存储后端。

**读写量级估算（100 concurrent docs）**:

每文档生命周期 derived/ 写入 = `markdown.md` + `outline.json` + `chunks.jsonl` + `kg.jsonl` + `summary.json` + `vision/*`，假设平均文档 20 chunks × 1536 维 embedding × 4 bytes ≈ 130KB chunks.jsonl，加 kg.jsonl ~50KB、summary.json ~10KB、vision 假设 5 图 × ~200KB embedding+meta = 1MB，markdown ~100KB，outline ~20KB ⇒ 总计 **~1.5 MB / 文档**。

100 文档 burst 一次 = ~150 MB 写入，分布在 ~25 分钟（graph LLM 是瓶颈，§E.4）= **~100 KB/s 平均写**。MinIO 单实例轻松承载 GB/s 级吞吐，这个量级是噪声。

**读取模式**:
- Sync 阶段：每个 modality worker 读自己那个文件一次（流式读 jsonl），~每文档 ~100KB sequential read
- Search 阶段：read primitive 走 cache（`aperag/cache/` D10.g），cache miss 才回对象存储

**对象存储相对文件系统的两个不利点**:
1. **List 性能**：cleanup worker 要按前缀列出旧 parse_version 目录。MinIO/S3 的 LIST 是按字典序的分页 API，列大目录慢。
   - **缓解**: cleanup worker 不靠 LIST 发现垃圾；改为从 DB 查 "active/pending 之外的 parse_version"，精确按路径删（§F.5 已是这个模型）。LIST 只在异常恢复时用。
2. **小文件开销**：`outline.json` ~20KB 是小文件，对象存储有元数据 overhead。
   - **缓解**: 这个量级（每文档 6-8 个文件）在 MinIO 可忽略；如果要极致优化，可把小文件合并到 `_bundle.tar`。当前规模不必。

**为什么 SQLite-friendly（earayu2 `msg=cc0a00d7` 提到未来用 SQLite 的潜在路径）**:
- 本设计的 SoT 是 PostgreSQL `document_index` 表（小，~5 行/文档），可平移到 SQLite 而无 schema 改动
- derived/ 在本地文件系统下就是文件夹；`object_store.py` 抽象层在私有化最小部署时可用 LocalFS adapter
- 整个系统能在单机+SQLite+LocalFS 跑通（§L）

**读写 contract（worker 必须遵守）**:
- 写 derived 文件时**先写临时名 `<file>.tmp`，fsync，rename**（POSIX）；MinIO 上用 multipart upload + complete（原子可见）。避免 partial write 被下一个 worker 误读。
- 读 derived 文件时若不存在或为空 → 视作 "derive 还没完成" → reschedule，不报错。
- 删除 derived 目录是 cleanup worker 的事；其他 worker 永远不删 derived/。

---

## §D. Idempotency contract per modality

Every modality MUST be **replace-idempotent** for `(document_id, parse_version)`. Re-running the sync of any derived artifact produces a backend state byte-equivalent to a fresh sync.

### D.1. The contract

```
sync_<modality>(document_id, parse_version, derived_artifact_path) →
    1. delete all existing entries WHERE document_id = X AND parse_version = Y
    2. insert all entries from derived_artifact_path
    3. mark (X, Y, modality) ACTIVE in DB
```

This is a two-phase operation per (doc, version, modality) triple, with the **delete** preceding the **insert**. Because each (doc, version) tuple is unique and the artifact is immutable, this is naturally idempotent.

### D.2. Per-modality concretization

| Modality | Backend | DELETE clause | INSERT source |
|---|---|---|---|
| Vector | Qdrant | `WHERE document_id=X AND parse_version=Y` | `chunks.jsonl` (text + embedding) |
| Fulltext | Elasticsearch | `delete_by_query: document_id=X AND parse_version=Y` | `chunks.jsonl` |
| Graph (Nebula) | Nebula | `DELETE VERTEX WHERE document_id=X AND parse_version=Y; DELETE EDGE ... ;` | `kg.jsonl` |
| Graph (Neo4j) | Neo4j | `MATCH (n {document_id:X, parse_version:Y}) DETACH DELETE n;` | `kg.jsonl` |
| Summary | Qdrant | `WHERE document_id=X AND parse_version=Y` | `summary.json` |
| Vision | Qdrant | `WHERE document_id=X AND parse_version=Y` | `vision/manifest.jsonl` |

### D.3. Graph indexer fix — entity lineage model (cross-doc shared entities)

**Bryce v2 review msg=7ccb176f #3 catch**: simple "DELETE-by-(doc, version) + re-INSERT" 在 graph 模态会**丢掉其他文档对共享 entity 的贡献**。例: "Linus" 在 100 个文档里都被提到，简单 DELETE-by-doc 会从 entity 行抹掉其他 99 个 doc 的 source_chunk_ids。

正确模型是 **per-(document, parse_version) lineage tracking on shared entities**。这与其他 4 个模态（vector / fulltext / summary / vision）不同，因为它们的 backend 行不跨文档共享。

#### D.3.1. Schema 形态（Nebula / Neo4j 共同）

每个 entity vertex 持有 lineage **集合**字段（不是被一个 doc 整体替换的字段）：

```
Entity vertex:
  name              : "Linus Torvalds"
  type              : "Person"
  source_lineage    : SET<{document_id, parse_version, chunk_ids[]}>
  description_parts : SET<{document_id, parse_version, text}>   -- 每个 doc 一条
```

每个 relation edge 同理：
```
Relation edge:
  source            : "Linus Torvalds"
  target            : "Linux Kernel"
  type              : "created"
  evidence_lineage  : SET<{document_id, parse_version, chunk_ids[]}>
```

`source_lineage` / `evidence_lineage` 用 SET 语义，按 `(document_id, parse_version)` 唯一。这把"哪个 doc 的哪个 parse_version 贡献了什么"显式追踪。

#### D.3.2. sync(kg.jsonl, doc, parse_version) 算法

> **Amendment 2026-04-26 (PR #1726 Wave 1 implementation lock, Bryce msg=464d5b70 catch + architect ruling msg=80c5dc06)**: lineage cleanup filter 是**按 `document_id` only**（不限 `parse_version`），**不是**按 `(document_id, parse_version)` exact-match。这样 sync(doc_A, v_new) 单调用自包含完成"覆盖 doc_A 旧 lineage" supersede 语义，与 §D.3.6 narrative 一致。原 v3 pseudocode 的 `(doc, parse_version)` exact-match 与 §D.3.6 期望状态矛盾（会留下 lineage[A,v_old] + lineage[A,v_new] 同时存在），是 spec bug。下面 pseudocode 是正确版本。

```python
def sync_graph_from_jsonl(jsonl_path, document_id, parse_version, graph):
    # ── Step 1: 移除 doc 所有历史 parse_version 的 lineage 贡献 ──
    # 注意：不直接 DELETE entity；只移除该 document_id 在 SET 里的所有成员，
    # 不限 parse_version。这让 sync 单调用自包含完成 supersede 语义。

    # 1a. 拉所有受影响的 entity（曾被该 document_id 任意 parse_version 写过）
    affected_entities = graph.execute(
        "MATCH (n) WHERE EXISTS (s IN n.source_lineage "
        "WHERE s.document_id == $doc) RETURN n.name",
        doc=document_id,
    )

    # 1b. 从每个 entity 的 source_lineage / description_parts 里移除该 document_id 的所有 lineage member
    graph.execute(
        "MATCH (n) WHERE EXISTS (s IN n.source_lineage "
        "WHERE s.document_id == $doc) "
        "SET n.source_lineage = [s IN n.source_lineage WHERE NOT "
        "(s.document_id == $doc)], "
        "    n.description_parts = [d IN n.description_parts WHERE NOT "
        "(d.document_id == $doc)]",
        doc=document_id,
    )

    # 1c. 同样清 relation edges 的 evidence_lineage（按 document_id only）
    graph.execute(
        "MATCH ()-[e]->() WHERE EXISTS (s IN e.evidence_lineage "
        "WHERE s.document_id == $doc) "
        "SET e.evidence_lineage = [s IN e.evidence_lineage WHERE NOT "
        "(s.document_id == $doc)]",
        doc=document_id,
    )

    # 1d. GC：source_lineage 为空的 entity → DELETE 整行
    #         evidence_lineage 为空的 relation → DELETE edge
    graph.execute("MATCH (n) WHERE size(n.source_lineage) == 0 DETACH DELETE n")
    graph.execute("MATCH ()-[e]->() WHERE size(e.evidence_lineage) == 0 DELETE e")

    # ── Step 2: 从 kg.jsonl 重建当前 (doc, parse_version) 的 lineage 贡献 ──
    for line in open(jsonl_path):
        record = json.loads(line)
        if record.type == "entity":
            graph.upsert_entity_lineage(
                name=record.name,
                type=record.type,
                add_lineage={
                    "document_id": document_id,
                    "parse_version": parse_version,
                    "chunk_ids": record.source_chunk_ids,
                },
                add_description_part={
                    "document_id": document_id,
                    "parse_version": parse_version,
                    "text": record.description,
                },
            )
        else:  # relation
            graph.upsert_relation_lineage(
                source=record.source,
                target=record.target,
                type=record.relation_type,
                add_evidence={
                    "document_id": document_id,
                    "parse_version": parse_version,
                    "chunk_ids": record.source_chunk_ids,
                },
            )
```

**`upsert_entity_lineage` semantics**: if entity exists, append to lineage SET (deduplicated by `(document_id, parse_version)`); else create with single lineage member.

#### D.3.3. Description 暴露策略（read path）

retrieve 时 entity 的 description 不能只取一个 doc 的片段（会让其他 doc 的贡献消失），也不应每次拼所有 doc 的片段（可能很长）。读路径有两个选项，v2 选项 A：

- **Option A (v2 default)**: read primitive 返回 `description_parts` 全集（按 doc count 排序），上层 LLM 自行去重 / 摘要。简单。
- **Option B (future split path)**: 在 derive 阶段或 read 阶段调一次 LLM 聚合 `description_parts` → 单一 `description_aggregated`，写回 entity。增加一次 LLM 调用，但 read 层简单。

Option A 在 100 文档量级足够；如未来 entity 被 1000+ docs 共享导致 description_parts 太长，再切 Option B。Schema 已支持（`description_parts` 是 SET，加一个 `description_aggregated` 字段不破坏现状）。

#### D.3.4. 与 D.1 contract 的关系

D.1 "DELETE-by-(doc, parse_version) THEN INSERT" 对 graph 模态需要 reinterpret：

> Graph 的 sync 是 **lineage-level DELETE + lineage-level INSERT**，不是 entity-level。Entity 行的生命周期由 lineage SET 是否为空驱动（empty → DELETE）。

幂等性还是成立的：再跑一次 sync(jsonl, doc, parse_version) → 1b 会移除上轮加的 lineage member，1d 不会误删（因为 step 2 又把 lineage 加回去了），最终状态 byte-equivalent。

#### D.3.5. Neo4j vs Nebula 实现

- **Neo4j**: Cypher 原生支持 list filtering（`[s IN n.source_lineage WHERE ...]`），上面伪 Cypher 直接可执行。
- **Nebula**: nGQL 支持 LIST 类型 (Nebula 3.x)，但 list 操作语法不如 Cypher 直接；需要在应用层 read-modify-write（拉 lineage list → Python filter → 写回）。这增加 race condition 风险。**因此**：sync_graph 必须对单 entity 的更新串行化（per-entity lock，按 entity_name hash 路由到固定 worker，或者 Nebula transaction 范围 lock）。Wave 2 的 graph_worker 实现要含这条 invariant。
- 两个后端都需要 `(name, type)` 复合主键做 entity dedup（已有）。

##### D.3.5.1. Lineage SET 元素 dedup key — 三 backend 必 align（Wave 4 T8 chunk 4 amendment, architect msg=baf6618e）

`source_lineage` / `evidence_lineage` SET 元素 dedup key **MUST** 是 `(document_id, parse_version)` 复合，**三 backend (Postgres / Neo4j / Nebula) 必须 align**。`upsert_*_with_lineage` 操作的语义是：

1. 如果 SET 中已有元素满足 `elem.document_id == lineage.document_id AND elem.parse_version == lineage.parse_version`，**replace** 该元素。
2. 否则 **append** 新元素。

**关键 invariant**：同一 `(document_id, parse_version)` 不能在 SET 内共存两份。这覆盖三类场景：
- doc_A v1 同 entity 多 chunk 抽取重复触发 upsert（同 key → replace，幂等）。
- 同一 worker 因 retry 重新调用 upsert（同 key → replace，幂等）。
- doc_A v2 取代 doc_A v1（先 `remove_*_lineage_member(document_id="doc_A")` 全 strip 含 v1，再 upsert(v2)），不依赖 dedup-by-doc only。

**为什么这条必须显式声明**：§D.3.6 step 3 描述 "doc_A v2 写入（覆盖 doc_A 旧 lineage）" 的 narrative 是 orchestrator-level 行为（remove + upsert two-step），不是 single-upsert 的 dedup-by-doc 语义。不显式声明 composite key 容易让 backend implementer 误把 `document_id` 单独当 dedup key（导致同 doc 不同 parse_version 互相覆盖，丢 lineage 历史；或者 LightRAG-style 多 chunk 抽取被错误 collapse）。Wave 4 T8 三 backend 的 chunks 1-3 实施都按此 composite key 实现：

| Backend | Composite key 实现 |
|---|---|
| Postgres | `WHERE NOT (elem->>'document_id' = $d AND elem->>'parse_version' = $v)` 在 `jsonb_agg` 内 strip 后 append |
| Neo4j | parallel-list 三 list（lineage / doc_ids / parse_versions）+ `[i IN range \| WHERE NOT (doc_ids[i] = $d AND parse_versions[i] = $v)]` keep-index |
| Nebula | JSON STRING property + Python `[m for m in members if m.key() != lineage.key()] + [lineage]` （`LineageMember.key()` 返 `(document_id, parse_version)` 二元组） |

跨 backend contract test (`tests/integration/test_lineage_graph_store_contract.py` case 3 `test_doc_re_parse_replaces_old_parse_version_member`) 锁此 invariant — 任何 backend drift 即 fail。

#### D.3.6. 自测扩展

D.4 idempotency 自测对 graph 加 1 个 case：
- doc_A v1 写入 → 包含 entity "Linus" lineage[A,v1]
- doc_B v1 写入 → entity "Linus" lineage[A,v1] + lineage[B,v1]
- doc_A v2 写入（覆盖 doc_A 旧 lineage）→ entity "Linus" lineage[A,v2] + lineage[B,v1]
- doc_A 删除 → entity "Linus" lineage[B,v1]（仍存在）
- doc_B 删除 → entity "Linus" 整行 DELETE（lineage 空）

**这是 graph 模态相对其他 4 个模态最重的特殊性，必须 wave 1 graph implementer（建议 Bryce，他最熟）严格按此契约写。**

### D.4. Self-test

Each modality ships a unit test that:
1. Calls `sync_<modality>(doc, v1, artifact_v1)` twice
2. Asserts backend state byte-equivalent after both calls

This test is part of CI's idempotency guarantee. It runs on every PR.

### D.5. Why DELETE-before-INSERT instead of UPSERT-only

Three reasons:
- A previously-synced record might no longer be in the new artifact (e.g., a chunk merged with another). UPSERT alone leaves the orphan in the backend.
- DELETE establishes a clean baseline; INSERT fills it. Failure between steps still leaves the system in a known-empty state for `(doc, version)`, which the next retry re-fills.
- DELETE-by-`(document_id, parse_version)` is a single indexed query in every backend; cost is constant.

---

## §E. Concurrency model — Redis + asyncio (decision locked)

**earayu2 拍板（`msg=cc0a00d7`）**: "放弃 Celery → Redis + asyncio，我接受。"

The earlier v1 decision matrix (HTTP-only / lightweight task / refactor Celery) is removed; the locked direction is:

> **Drop Celery entirely. Use a small set of asyncio worker processes that BLPOP from per-modality Redis lists; let the database (`document_index` table) hold the state of truth on `(document_id, parse_version, modality)` triples; have one tiny reconciler loop convert DB intent → queue jobs.**

This collapses the three-layer ownership skew (§A.3 Celery / lease thread / DB) into one (DB).

### E.1. Why this beats the two non-chosen options (kept brief, for the record)

- **HTTP-only** loses out because graph LLM extraction takes 30s+ per doc; tying that to an HTTP request forces clients to poll, and rate-limit retries over minutes don't fit the HTTP cycle.
- **Refactor-Celery** has bounded simplification ceiling — chord/lease/processing_token aren't going away even with refactor; earayu2 said they preferred the cleaner cut.

The chosen design hits 100-concurrent on a single machine with no architectural change (§E.4) and degrades gracefully to single-process synchronous mode for very small private deployments (§L).

### E.2. Architecture (locked)

```
┌─ HTTP API (FastAPI) ─────────────────────────────────────────────────────┐
│  POST /collections/<id>/documents (upload)                               │
│  POST /documents/<id>/reindex                                            │
│  GET  /documents/<id>/status                                             │
└──┬───────────────────────────────────────────────────────────────────────┘
   │ Synchronous: write source file to object store; insert DB row
   │ Async kick: RPUSH a "parse" job onto Redis queue
   ▼
┌─ Redis queue (5 lists, one per modality) ───────────────────────────────┐
│  q:parse                                                                 │
│  q:vector                                                                │
│  q:fulltext                                                              │
│  q:graph                                                                 │
│  q:summary                                                               │
│  q:vision                                                                │
└──┬───────────────────────────────────────────────────────────────────────┘
   │ BLPOP (long-poll)
   ▼
┌─ Worker processes (one per modality, asyncio inside) ───────────────────┐
│  parse_worker          (1 process, asyncio concurrency = 8)              │
│  vector_worker         (1 process, asyncio concurrency = 16)             │
│  fulltext_worker       (1 process, asyncio concurrency = 32)             │
│  graph_worker          (1 process, asyncio concurrency = 4)              │
│  summary_worker        (1 process, asyncio concurrency = 4)              │
│  vision_worker         (1 process, asyncio concurrency = 4)              │
│                                                                          │
│  Each worker:                                                            │
│  - BLPOP from its queue                                                  │
│  - Validate task against DB (still relevant?)                            │
│  - Acquire resource token (LLM / embedding bucket) if applicable         │
│  - Do work (read derived artifact OR write derived artifact)             │
│  - Update DB row status                                                  │
│  - On failure: write failure reason; reconciler will retry               │
└──┬───────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ Reconciler (single process, simple loop) ──────────────────────────────┐
│  Every 30s: SELECT documents needing dispatch → RPUSH onto queue        │
│  Conditions:                                                             │
│  - status = PENDING (new)                                                │
│  - status = FAILED with retry_after < now (retry)                        │
│  - status = CREATING with last_heartbeat < now - 60s (stale)             │
│                                                                          │
│  ~80 lines of Python total                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

### E.3. How concurrency reaches 100 docs simultaneously

100 documents × 5 modalities = 500 in-flight operations at peak. Workers handle them as follows:

- `parse_worker` (concurrency 8): parses 8 docs at a time. With ~5s parse time, ~96 docs/minute throughput. 100 docs parsed in ~63 seconds.
- `vector_worker` (concurrency 16): embeds chunks. Constrained by embedding API rate limit (assumed 100 req/sec).
- `fulltext_worker` (concurrency 32): writes ES. ES handles thousands of writes/sec; 32 is generous.
- `graph_worker` (concurrency 4): LLM extraction is expensive (~30s each). 4 concurrent × 100 docs = 25 minutes total. **This is the bottleneck.**
- `summary_worker` (concurrency 4): LLM call, similar to graph but shorter.
- `vision_worker` (concurrency 4): GPU/embedding-bound.

End-to-end: a 100-doc burst takes ~25 min because of graph-modality LLM extraction. **All non-graph modalities complete in under 2 minutes.** Per the per-modality independent visibility model (§F), vector + fulltext become searchable first; graph trails as it completes; `index_state` discriminator (§G.5) lets clients know which modalities are live for any given doc/parse_version.

If this is too slow, scale horizontally: add a second graph_worker process (concurrency 4 × 2 = 8 graph in-flight). Same pattern as Celery worker scaling but with 1/10 the code.

### E.4. No lease thread, no chord, no token games

The current system has:
- Python lease thread that renews `lease_expires_at` every 60s (§A.4)
- Celery chord callback that aggregates 5 parallel tasks
- Processing token validated at task entry to prevent ghost callbacks

The new system has:
- Worker writes `last_heartbeat = now()` at task start (one DB UPDATE)
- No chord; each modality reports its own status independently
- Reconciler reclaims any task with `last_heartbeat < now - 60s` AND status=RUNNING

This is simpler and survives worker crashes naturally (heartbeat stops being updated → reconciler reclaims).

### E.5. Single-process synchronous mode (private deploy escape hatch)

For very small private deployments (single-customer dev, <10 docs/hour), the same code runs synchronously by setting `INDEXING_MODE=inline`: the HTTP handler calls `derive` + `sync` directly instead of RPUSH. No Redis required (LocalFS object store + SQLite + inline workers — the minimal-deploy stack, see §L). 50-line addition; same modality classes, just different entry point.

The default `INDEXING_MODE=async` ships the Redis worker-pool architecture; private "deploy-and-forget" stacks can either use it (more throughput) or downgrade to inline (zero ops).

---

## §F. State machine + per-modality independent visibility

**earayu2 拍板（`msg=cc0a00d7`）**: "原子切换策略，不太需要做原子性，因为我们的 index 层本身就是给上层 agent 提供查询信息，不需要非常准确，并且原子性可能引入额外的复杂性和性能问题，我能接受一点数据不一致等缺点。"

v2 删除 v1 的 `active_parse_version` / `pending_parse_version` 双列原子翻转设计。每个 `(document_id, parse_version, modality)` 三元组**独立可见、独立翻新**。短暂不一致由上层 agent 自行处理。

### F.1. `document_index` schema (simplified)

```sql
CREATE TABLE document_index (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR NOT NULL,
    parse_version VARCHAR(16) NOT NULL,
    modality VARCHAR NOT NULL,            -- 'vector' | 'fulltext' | 'graph' | 'summary' | 'vision'

    status VARCHAR NOT NULL,              -- PENDING | RUNNING | ACTIVE | FAILED
    error_message TEXT,
    retry_count INT DEFAULT 0,
    retry_after TIMESTAMPTZ,

    last_heartbeat TIMESTAMPTZ,           -- worker progress
    derived_artifact_path TEXT,           -- e.g. 'collections/A/documents/D/derived/parse_v123/kg.jsonl'

    -- Dispatch denormalization (added 2026-04-27 per huangheng Wave 2 CR
    -- finding msg=c94b57fe + architect ruling msg=498b12f0). orchestrator
    -- + cleanup workers query these directly without joining `document`.
    -- nullable during Wave 1+2 for fixture back-compat; Wave 3 hard-cut
    -- (task #14) flips both to NOT NULL after legacy table is dropped.
    collection_id VARCHAR(64) NOT NULL,   -- denormalized from document.collection_id
    source_path TEXT NOT NULL,            -- pointer to source/ artifact, worker derive reads directly

    is_serving BOOLEAN NOT NULL DEFAULT FALSE,  -- this triple is what search currently reads

    -- Forward-compat hook for future organization concept (§H.2). v2
    -- default fill: f"user:{user_id}". Future: f"org:{org_id}". Required
    -- key for Redis token bucket per (resource_class, tenant_scope_key)
    -- in §H.5 quota.
    tenant_scope_key VARCHAR(64) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (document_id, parse_version, modality)
);

-- Partial unique index — locks the per-(document, modality) "at most one
-- serving row" invariant at the DB layer (Bryce v2 review msg=7ccb176f #2).
-- Postgres native; SQLite (Tier 1, §L) supports the same syntax since 3.8.
-- Without this, an orchestrator bug could leave two rows is_serving=TRUE
-- for the same (doc, modality), which would JOIN out duplicate search
-- results — a class of bug the schema must make impossible.
CREATE UNIQUE INDEX uniq_document_index_serving
    ON document_index (document_id, modality)
    WHERE is_serving = TRUE;

-- Cleanup / quota scan support (added 2026-04-27 alongside collection_id).
CREATE INDEX idx_document_index_collection
    ON document_index (collection_id);

-- Quota bucket scans by (resource_class, tenant_scope_key) — see §H.5.
CREATE INDEX idx_document_index_tenant_scope
    ON document_index (tenant_scope_key);

CREATE TABLE document (
    id VARCHAR PRIMARY KEY,
    collection_id VARCHAR NOT NULL,
    latest_parse_version VARCHAR(16),     -- newest parse_version we've started indexing
    -- ... existing fields ...
);
```

`is_serving=TRUE` is the per-(document, modality) "serving pointer". The partial unique index above guarantees **at most one** row per `(document_id, modality)` has `is_serving=TRUE` — DB-enforced, not application-enforced. Search reads only `is_serving=TRUE` rows.

`latest_parse_version` on `document` is purely informational (for UI / admin); it has no flip semantics.

**Cutover transaction interaction with the partial unique index** (§F.3): the swap is `UPDATE old_row SET is_serving=FALSE` then `UPDATE new_row SET is_serving=TRUE`, in this order, in one transaction. Postgres evaluates the partial unique constraint at statement boundaries (deferred when DEFERRABLE; immediate by default) — within one transaction, the FALSE update lands first so the TRUE update doesn't conflict. If a worker tries to bypass the cutover order and TRUE-flip directly, the DB rejects.

### F.2. Status enum (4 states)

```
PENDING   — row created, work not yet started
RUNNING   — a worker is processing this triple; last_heartbeat updated periodically
ACTIVE    — backend reflects this triple's derived artifact (write succeeded)
FAILED    — terminal failure for this attempt; retry_after gives next try time
```

`is_serving` is orthogonal to status — a row can be ACTIVE but not yet serving (e.g., it just finished indexing and the cutover swap hasn't run yet). See §F.3.

### F.3. Per-modality cutover (no orchestration)

When a worker successfully completes a modality:

```sql
BEGIN;
UPDATE document_index
   SET status = 'ACTIVE', updated_at = NOW()
 WHERE id = $row_id;

UPDATE document_index
   SET is_serving = FALSE
 WHERE document_id = $doc AND modality = $mod AND is_serving = TRUE;

UPDATE document_index
   SET is_serving = TRUE
 WHERE id = $row_id;
COMMIT;
```

Three statements in one transaction, scoped to a single `(document_id, modality)`. **Per-modality**, not document-wide. Vector swap doesn't wait for graph; graph swap doesn't wait for vector.

Result: a fresh upload of the same document might briefly show "vector results from new parse_version, graph results from old parse_version" until graph also finishes. earayu2 has explicitly accepted this.

### F.4. Inconsistency window — what the upper layer sees

The window between "new parse_version starts" and "all 5 modalities cut over" can be ~25 minutes (§E.3, graph LLM extraction is the slowest). During this window:

| Reader sees | Behavior |
|---|---|
| New vector hits + old graph hits in same query | Mixed `parse_version`. Per `index_state` in search metadata (§G.5), client can see which modality served which `parse_version`. |
| Same chunk re-ranked across modalities | Possible duplicate; rerank dedup by `chunk_id` fixes this (existing dedup logic). |
| Old chunks linger after content deletion | Until cleanup worker (§F.5) runs (~5 min cycle). |

**Why this is acceptable** (from earayu2): the index layer feeds an upper agent that already reasons over heterogeneous evidence. A 5-25 minute "mixed parse_version" window does not cause user-visible incorrectness for agentic search — the agent re-ranks and the inconsistency dissolves on the next iteration. Atomic flip would buy strict consistency at the cost of: an extra coordination layer + atomicity-breaks-on-modality-stuck failure mode + write-blocking complexity.

**What we explicitly do NOT need**:
- Multi-modality coordination on cut-over.
- "All-or-nothing" upgrades.
- Cross-row distributed transactions.

### F.5. Cleanup worker (replaces v1's DELETING / DELETION_IN_PROGRESS states)

A separate cleanup worker runs every 5 minutes with **three independent paths**, each idempotent and resumable. Failure to complete any path mid-run does not corrupt state — the next 5-minute cycle picks up where the previous left off.

**Path A — orphan parse_version GC** (regular operating state):

Deletes backend entries for any `(document_id, parse_version, modality)` that is:
- `is_serving = FALSE` AND
- Not the latest `parse_version` per `(document_id, modality)` AND
- `updated_at < NOW() - INTERVAL '1 hour'`

After backend deletion, the `document_index` row itself is deleted. derived/ directory for the orphan `parse_version` is also removed if no remaining row references it.

For graph modality: per §D.3 amended canonical (lineage cleanup is by `document_id` only, not parse_version), Path A is **a backend no-op for graph** — every fresh `sync(doc, v_new)` already supersedes the doc's lineage SET. The DB row for the orphan parse_version is still deleted; only the backend graph mutation is skipped. A `graph_noop` counter tracks how often this fast-path fires for telemetry.

**Path B — single-document deletion cascade** (user deletes one document):

Trigger: caller (HTTP delete endpoint) writes `document.deleted_at = NOW()` synchronously and posts the document_id to the cleanup-worker job queue.

For each modality:
- Non-graph (vector / fulltext / summary / vision): per `(doc, parse_version)` `delete_by_filter(document_id, parse_version)` for every parse_version in that document's `document_index` rows.
- Graph: invoke `remove_entity_lineage_member(document_id=doc)` (per §D.3 — by `document_id` only, removes the doc's lineage SET member from every shared entity; entity-row GC happens when its lineage SET becomes empty). Per-doc dedup: only one call per document regardless of how many parse_versions existed.

After all modalities cleaned: delete the `document_index` rows + delete derived/ tree for the document.

**Path C — collection deletion cascade** (added 2026-04-27 per architect msg=3890c9d7):

Trigger: caller (HTTP delete-collection endpoint) writes `Collection.deleted_at = NOW()` synchronously, then returns 200 to the user. **The HTTP handler does not block on cascade**; the cleanup worker drives the rest.

Worker scan WHERE `Collection.deleted_at IS NOT NULL`. For each:
- Iterate every `Document` belonging to the collection and invoke Path B (`cleanup_for_deleted_documents`) per document. (Path B is idempotent so partial completion + reprocess on next 5-min cycle is safe.)
- After all documents cleaned: DELETE `Collection` row + cascade delete `collections/<collection_id>/` source / derived storage tree.
- If any per-document path B fails: log + skip that document; next cleanup cycle retries.

Path C provides durability for collection deletion that simple `asyncio.create_task()` cannot — if the HTTP server restarts mid-cascade, the cleanup worker resumes via state-driven recovery (the same pattern as document deletion + parse_version GC). This is the canonical replacement for the legacy Celery `collection_delete` task.

This GC pattern is independent of the main write path. A delayed cleanup wastes storage but never causes incorrectness.

### F.6. Why this is simpler than v1's atomic flip

v1 design had: `document.active_parse_version` + `document.pending_parse_version` columns + "all 5 modalities ACTIVE" trigger + transactional flip + handling of optional modalities + handling of stuck-FAILED modality.

v2 has: one boolean `is_serving` per `document_index` row, swapped within a 3-statement transaction scoped to one `(document_id, modality)`. No coordination across modalities. No "stuck blocks flip" failure mode (each modality flips independently when it's done). One concept.

Net: **~150 fewer lines of orchestration code** + an entire failure mode class eliminated.

---

## §G. Multi-modal unified pipeline

### G.1. The unified pattern

Every modality follows the same 4-step pipeline:

```
DERIVE step                          SYNC step
───────────                          ────────
read source file       ─────────►   read derived artifact
compute artifact                     write to backend
write to derived/path                update DB to ACTIVE
update DB to "derived"
```

Splitting into derive + sync is the key simplification:
- DERIVE is the expensive step (LLM, embedding API, GPU)
- SYNC is the cheap step (DB upsert)
- Decoupling them means SYNC failures don't waste DERIVE work

For most modalities derive and sync can be combined (especially when "derive" is just a pass-through). For graph and summary the split is meaningful.

### G.2. Per-modality implementation

#### G.2.1. Vector modality

```
file: aperag/indexing/vector.py  (~150 lines)

derive(source_path, parse_version) → chunks.jsonl
    1. Read markdown.md from derived/parse_<v>/
    2. Chunk text (existing chunker)
    3. For each chunk: call embedding API
    4. Write chunks.jsonl: one JSON line per chunk with {chunk_id, text, embedding, section_path, ...}

sync(chunks.jsonl, document_id, parse_version, qdrant) →
    1. qdrant.delete(filter={document_id: X, parse_version: Y})
    2. for each chunk in jsonl: qdrant.upsert(point_id=chunk_id, vector=..., payload=...)
```

#### G.2.2. Fulltext modality

```
file: aperag/indexing/fulltext.py  (~120 lines)

derive: same chunks.jsonl as vector (text portions only — no embedding needed for ES)
        OR re-use vector's chunks.jsonl directly (ES doesn't need embedding)

sync(chunks.jsonl, document_id, parse_version, es) →
    1. es.delete_by_query({document_id: X, parse_version: Y})
    2. es.bulk_index(chunks)
```

Note: vector and fulltext can **share `chunks.jsonl`**. The difference is just which fields ES consumes vs Qdrant consumes. This collapses duplicate chunking logic.

##### G.2.2.1. Fulltext backend dispatch（Wave 4 T9 amendment, architect msg=eba26fc2）

`collection.config.fulltext_backend_type` is a `Literal["elasticsearch", "opensearch"]` field (default `"elasticsearch"`) wired through `worker_factory._build_fulltext_backend(backend_type, index_name)`. The two backends share the same `_ElasticsearchFulltextBackend` adapter because OpenSearch (forked from Elasticsearch 7.10) preserves wire-compatible HTTP API for `index` / `bulk` / `delete_by_query`. Only the client-construction step differs:

| Backend | Driver | Client kwargs |
|---|---|---|
| `elasticsearch` | `elasticsearch` (always available) | `Elasticsearch(host, basic_auth=(user, pass), request_timeout=N)` |
| `opensearch` | `opensearchpy` (lazy import — `WorkerFactoryError` if missing, points at `fulltext-opensearch` extra) | `OpenSearch(hosts=[host], http_auth=(user, pass), timeout=N)` |

**Single `ES_HOST` env var serves both backends** — operators run one fulltext cluster per deployment. When choosing OpenSearch, configure `ES_HOST` to the OpenSearch endpoint (e.g. `https://opensearch.internal:9200`) — the adapter does not introspect server-side identity, only wire protocol. New backends (Solr / Typesense / MeiliSearch) plug in via the same dispatch by extending `_VALID_FULLTEXT_BACKENDS` + adding a `_build_*_client` helper.

#### G.2.3. Graph modality

```
file: aperag/indexing/graph.py  (~200 lines)

derive(source_path, parse_version) → kg.jsonl
    1. Read markdown.md from derived/parse_<v>/
    2. Call LLM to extract entities + relations (the expensive step)
    3. Write kg.jsonl: one JSON line per entity OR relation
    4. Each record tagged with document_id, parse_version

sync(kg.jsonl, document_id, parse_version, graph_backend) →
    1. graph_backend.delete_by(document_id, parse_version)
    2. for each record in jsonl:
         if record.type == 'entity': graph_backend.upsert_vertex(...)
         else: graph_backend.upsert_edge(...)
```

This is the major simplification of earayu2's complaint. Compare to current §A.5 (7 hops across 3 packages); the new design is 2 functions in 1 file, plus a `graph_backend` interface for Nebula vs Neo4j.

#### G.2.4. Summary modality

```
file: aperag/indexing/summary.py  (~80 lines)

derive(source_path, parse_version) → summary.json
    1. Read markdown.md
    2. LLM call: generate summary
    3. Embed summary
    4. Write summary.json: {summary_text, embedding}

sync(summary.json, document_id, parse_version, qdrant) →
    1. qdrant.delete(filter={document_id: X, parse_version: Y, modality: 'summary'})
    2. qdrant.upsert(point_id=document_id, vector=embedding, payload={summary_text, ...})
```

#### G.2.5. Vision modality

```
file: aperag/indexing/vision.py  (~150 lines)

derive(source_path, parse_version) → vision/manifest.jsonl + vision/images/
    1. Extract image parts from source PDF
    2. Save image blobs to vision/images/
    3. For each image: encode via embedding service or vision-LLM
    4. Write manifest.jsonl: {image_id, image_path, embedding, alt_text, ...}

sync(manifest.jsonl, document_id, parse_version, qdrant) →
    1. qdrant.delete(filter={document_id: X, parse_version: Y, modality: 'vision'})
    2. for each entry: qdrant.upsert(...)
```

##### G.2.5.1. Vision modality T7 wiring scope (Wave 4 backlog T7 + Wave 5 follow-up)

`worker_factory._build_vision_worker` currently raises
:class:`WorkerFactoryError` when the collection's embedding service is
not multimodal (`embedding_service.is_multimodal()` returns False —
the operator opted into vision but the configured embedder is text-
only). The Wave 3 lesson #10 explicit-gate pattern still applies; T7
**self-disables** the gate when `is_multimodal()` flips True.

T7 wiring requires three coordinated pieces beyond what chunk 4 ships:

1. **Multimodal embedding API surface** — extend
   :class:`aperag.llm.embed.embedding_service.EmbeddingService` with
   an `embed_image(image_bytes: bytes, alt_text: str = "") -> list[float]`
   method that the multimodal provider (CLIP / LLaVA / GPT-4V / etc.)
   resolves to a real visual embedding. Without this, `_embed` is
   forced into the existing `embed_query(str)` path which only
   handles text — the Wave 1+2 implementation papered over this with
   a string-concat ``f"{image_id}|{alt_text}"`` (Wave 3 lesson #10
   broken pattern; chunk 4 gate prevents production opt-in).

2. **Real PDF / image-source extraction in the parser pipeline** —
   the T1 simulator's `<source>.images.json` companion (a JSON list
   of image_id + alt_text records) does not include actual image
   bytes; chenyexuan T3 chunk 1 wired DocParser for markdown / PDF /
   Word / image inputs, but the per-image bytes path
   (``derived/parse_<v>/vision/images/<image_id>.<ext>``) is not yet
   produced by any parser branch. T7 needs the parser to land
   per-page image extraction (or single-image-input passthrough)
   before the vision worker has any image bytes to embed.

3. **Provider-specific multimodal model registration** — the model
   platform v3 router (``aperag/domains/model_platform/api/providers_v3_routes.py``)
   needs to surface the multimodal capability flag so operators can
   set `is_multimodal=True` on the collection's embedder spec. The
   capability flag is the source of truth for the chunk 4b gate.

Per architect msg=87e2b187 chunk 4d Option C ruling + Bryce
honest-scope clarification msg=bad51f10: T7 in PR #1731 is **doc-
only** — this §G.2.5.1 spec amendment locks the wiring contract
across all three pieces, but **none of items 1 / 2 / 3 ship code
in Wave 4**. The three pieces are tightly coupled (multimodal API
surface needs a provider flag to be useful + a parser branch to
have any image bytes to embed); splitting them across PRs risks the
Wave 3 lesson #10 broken-pattern (e.g., ship API surface with no
caller, ship provider flag with no API surface, etc.). The whole
T7 implementation defers to a single Wave 5 PR that bundles all
three.

Wave 4 close-out invariant: the chunk 4b vision gate stays
effective. Operators that opt into vision without a multimodal
embedder configured see a clean ``WorkerFactoryError`` with the
"Wave 4 wiring" message — which after T7 doc-only ships still
points at the same actionable status: configure a multimodal
embedder OR set ``enable_vision=false``.

### G.3. Deriver / syncer interfaces (Python)

```python
# aperag/indexing/base.py — the common contract
from abc import ABC, abstractmethod

class Modality(ABC):
    name: str  # 'vector' / 'fulltext' / etc.

    @abstractmethod
    async def derive(self, document_id: str, parse_version: str) -> str:
        """Read source/parsed inputs, write derived artifact. Returns artifact path."""

    @abstractmethod
    async def sync(self, artifact_path: str, document_id: str, parse_version: str) -> None:
        """Read derived artifact, write to backend. Idempotent: DELETE-then-INSERT."""
```

That is the entire interface. Each modality is a class that implements these two methods, plus a `name` constant. The orchestrator calls `derive` then `sync`, updating DB state between.

### G.4. Where shared code lives

```
aperag/indexing/
├── base.py            ← Modality ABC, common helpers
├── vector.py          ← VectorModality
├── fulltext.py        ← FulltextModality
├── graph.py           ← GraphModality
├── summary.py         ← SummaryModality
├── vision.py          ← VisionModality
├── parser.py          ← parse source → markdown.md, outline.json, chunks.jsonl, ...
├── orchestrator.py    ← receives jobs from queue, calls derive/sync, updates DB
├── reconciler.py      ← polling loop (PENDING/FAILED retry, stale RUNNING reclaim)
├── cleanup.py         ← garbage-collect old parse_versions
└── object_store.py    ← write derived/ paths, read artifacts
```

11 files total. Compare to current ~15 files in `aperag/domains/indexing/` + `aperag/tasks/` + `aperag/graphindex/v2/*`. And the new files are individually shorter.

### G.5. `SearchResultItem.metadata.index_state` discriminator (more important under per-modality independent visibility)

With v2's per-modality independent flip (§F.3), it becomes **structurally necessary** for search results to advertise which modality served the hit and at which `parse_version`. Otherwise upper-layer agents cannot reason about the short inconsistency window (§F.4).

Each `SearchResultItem.metadata` carries:

```python
class SearchResultMetadata(BaseModel):
    # ... existing fields (chunk_id / section_path / heading_anchor etc) ...
    parse_version: Optional[str] = None       # which parse_version served this hit
    index_modality: Optional[Literal["vector","fulltext","graph","summary","vision"]] = None
    index_state_per_modality: Optional[Dict[str, Literal["ACTIVE","FAILED","NOT_ENABLED","INDEXING"]]] = None
```

> **Naming amendment 2026-04-27 (Bryce T3.2 implementation, msg=1a02cbcb)**: the field for "which retrieval modality produced this hit" is **`index_modality`**, not `modality`. D10.h already locked `SearchResultMetadata.modality: Optional[Literal["text", "image"]]` for **content modality** (whether the hit's content is text or an image), and the two concepts are orthogonal — a hit can be `index_modality="vector"` + `modality="text"`, or `index_modality="vision"` + `modality="image"`. The `index_` prefix disambiguates and preserves the D10.h-locked field.

Clients (and the agent layer) can:
- Detect mixed-version results in one response (different modalities show different `parse_version`)
- Skip a retrieval modality that's currently FAILED/INDEXING
- Decide whether to wait + retry vs proceed with partial coverage

Schema-wise this is a small extension of the D10.h `SearchResultMetadata` allowlist (already locked at chunk_id / section_path / heading_anchor / source / title / collection_id / document_id / asset_id / mimetype / page_idx / url / modality; v3 adds `parse_version` / `index_modality` / `index_state_per_modality`).

---

## §H. Multi-tenant isolation — simple now, organization-ready later

earayu2 拍板（`msg=cc0a00d7`）: "未来我考虑引入 organization；额度管理需要，但是目前可以做简单点，不要锁死未来灵活性就好了。"

v2 的多租户策略：现在做**最小化**实现，但所有边界都按"未来要加 organization 一层"对齐，不挖坑。

### H.1. 当前层级（locked）

```
User --(owns)--> Collection --(owns)--> Document --> document_index rows
```

Tenant 边界在 `Collection.user`。`document_index` 没有显式 user 列（通过 `document_id → collection.user_id` 间接拥有），但**所有 worker 操作以 collection_id 为隔离单位**。

### H.2. 未来 organization 层（forward-compat hook）

预期演进路径（不在 v2 实现，但 v2 不阻挡）：

```
Organization --(owns)--> User --(belongs)--> Organization
            \--(owns directly)--> Collection (org-shared)
```

为此 v2 在以下两处留 hook：
- `document_index` 表新增 `tenant_scope_key VARCHAR` 字段（v2 默认填 `user:<user_id>`；未来填 `org:<org_id>` 即切到组织级）。Worker / reconciler / cleanup 都按这个字段做 quota / fairness 维度。
- 配额表 `tenant_quota` (§H.5) 用 `tenant_scope_key` 做主键，不绑定到 user。

引入 organization 时只需：
1. 写一次性脚本把 `tenant_scope_key` 从 `user:X` 重写为 `org:Y`
2. 不动 worker / reconciler 代码

### H.3. Required (always-on)

- 所有 queue message 携带 `tenant_scope_key`
- 所有 backend 写入在 metadata 里 tag `tenant_scope_key + collection_id`
- HTTP layer 鉴权：requestor 必须能访问 `collection_id` 所属的 tenant_scope

### H.4. Deferred (do not implement v2 unless symptom shows)

- Per-tenant **fairness queueing**（一个 noisy collection 不能把队列吃满）
- Per-tenant **concurrency cap**（单 tenant 同时最多 N 个 in-flight）
- Per-tenant **资源配额** beyond simple LLM token bucket

Add when observability (§J) shows queue starvation or cross-tenant lag variance. 私有化部署里这些信号很少出现 —— 多租户私有化通常 = 一个客户的多个团队。

### H.5. 简单配额管理（earayu2 "需要但简单"）

Single-knob 实现：

```python
# Redis keys:
#   quota:llm:{tenant_scope_key}:tokens
#   quota:embedding:{tenant_scope_key}:tokens
#   quota:llm:default:tokens   ← 兜底配额，所有 tenant 共享

# Per resource class, per tenant:
#   capacity     = e.g. 60 tokens (60 LLM calls / minute)
#   refill_rate  = e.g. 1 token / second
```

#### H.5.1. Redis logical db assignment（Wave 4 T8 chunk 4 amendment, architect msg=baf6618e）

ApeRAG 在多个 subsystem 用 Redis，单一 host 共享 keyspace 有 key collision 风险（celery / Wave 1 entity lock / Wave 4 WorkQueue / Wave 4 Quota）。chunk 4 lock 以下 logical-db 分隔：

| Logical DB | Subsystem | Key prefix |
|---|---|---|
| 0 | Celery broker (`CELERY_BROKER_URL`) | `celery-task-meta-*` etc. |
| 1 | Memory backend (`MEMORY_REDIS_URL`) | `memory:*` |
| 2 | Indexing WorkQueue (`INDEXING_QUEUE_REDIS_URL`) | `q:parse`, `q:vector`, `q:fulltext`, `q:graph`, `q:summary`, `q:vision` |
| 3 | Quota / EntityLock | `quota:<class>:<tenant>:tokens`, `indexing:graph:entity:<slot>` |

`aperag/config.py` 的 default-derive 路径会从单个 `REDIS_HOST` / `REDIS_PORT` 自动衍生这四个 URL，分别绑定 db=0/1/2/3。Operator 可以单独覆盖任一 URL 走外部 Redis 集群。**production 部署必须保 4 个 logical db 不重叠**，否则 BLPOP queues 会和 cache / broker 互相 RPOP/RPUSH。

#### H.5.2. Nebula graph backend multi-process EntityLock invariant（Wave 4 T8 chunk 4 amendment, architect msg=87e2b187）

如 `collection.config.graph_backend_type == "nebula"` 且 worker pool 是 multi-process（worker pool concurrency >= 2 进程）, **`INDEXING_QUEUE_REDIS_URL` MUST be set** — Nebula 缺乏 native list ops，`upsert_*_with_lineage` 必须 read-modify-write，多进程并发 upsert 同一 entity 必须 serialise 才不丢 lineage 元素。`worker_factory._resolve_entity_lock(backend_type="nebula")` 走 `RedisEntityLock`（绑定 `indexing_queue_redis_url`）保证 cross-process 锁定。

如 Redis 未配，fallback 到 `InMemoryEntityLock`（process-local `asyncio.Lock`），仅适用于 single-process test/dev — production multi-process 跑 Nebula 会丢 lineage 元素（race window）。

Postgres + Neo4j backend **不需要** Redis EntityLock — 它们的 `upsert_*_with_lineage` 单 SQL/Cypher 语句下 strip-then-append（PG `INSERT ON CONFLICT … COALESCE + jsonb_agg` / Neo4j `MERGE … WITH … SET` 都在 row-lock 内 atomic），cross-process race 由 backend native row-lock 兜底。

Worker 在调用 LLM / embedding 之前 `acquire_token(scope, resource_class)`：
- 优先从 `quota:<class>:<tenant_scope_key>:tokens` 拿
- 没有就从 `quota:<class>:default:tokens` 拿（共享池）
- 都空就 wait

写一个 `aperag/indexing/quota.py` ~80 行实现整个配额逻辑。tenant 配置可读 `tenant_quota` 表（schema：`tenant_scope_key | resource_class | capacity | refill_rate_per_sec`）；缺记录走 default。

**为什么不锁死未来灵活性**：
- 数据模型支持 per-tenant per-class 调参，新增 resource class 只是加一行
- 未来加 fairness queueing（按 tenant_scope_key 分独立队列）或 priority lane，是上面再加一层，不需要改 quota 这一层
- 切到 organization 维度只是改 `tenant_scope_key` 的填写规则

### H.6. Bulkhead — defense in depth

每个 worker process 设硬上限（与 tenant 无关）：
- 内存上限（Linux cgroup / Docker）
- LLM API 调用超时（默认 60s）
- Embedding API 超时（默认 30s）
- 上传文档大小上限（默认 50MB）

这些是现有代码就有的 pattern，v2 保留并集中到一个 `aperag/indexing/limits.py`。

---

## §I. Failure recovery

### I.1. Three failure modes

| Mode | Detection | Response |
|---|---|---|
| **Worker crash mid-task** | `last_heartbeat < now - 60s` AND status = RUNNING | Reconciler resets status to PENDING; next reconcile cycle re-dispatches |
| **Backend unavailable (transient)** | Worker catches exception, sets status = FAILED with `retry_after = now + backoff` | Reconciler picks up FAILED rows where `retry_after < now`; re-dispatches |
| **Backend permanently broken** | retry_count exceeds threshold (default 5) | Status remains FAILED; flagged for operator attention; document never enters atomic flip eligible state |

### I.2. Retry policy

- Initial retry delay: 30 seconds
- Exponential backoff: 30s → 60s → 120s → 240s → 480s
- After 5 retries: stop retrying, flag for operator
- LLM rate-limit specifically: separate retry policy that watches for 429 responses, longer backoff (5 minutes), no retry count cap (but emits an alert at retry 10+)

### I.3. Reconciler implementation (target ~80 lines)

```python
async def reconcile_loop(db_pool, redis):
    while True:
        async with db_pool.acquire() as conn:
            # PENDING: never run
            pending = await conn.fetch("""
                SELECT id, document_id, parse_version, modality
                FROM document_index
                WHERE status = 'PENDING'
                ORDER BY created_at
                LIMIT 100
            """)
            for row in pending:
                await redis.rpush(f"q:{row.modality}", json.dumps({
                    'index_id': row.id,
                    'document_id': row.document_id,
                    'parse_version': row.parse_version,
                }))

            # FAILED with retry_after passed
            retryable = await conn.fetch("""
                SELECT id, ...
                FROM document_index
                WHERE status = 'FAILED' AND retry_after < NOW() AND retry_count < 5
            """)
            for row in retryable:
                # Reset to PENDING for next cycle
                await conn.execute("UPDATE document_index SET status = 'PENDING' WHERE id = $1", row.id)

            # RUNNING with stale heartbeat
            stale = await conn.fetch("""
                SELECT id FROM document_index
                WHERE status = 'RUNNING' AND last_heartbeat < NOW() - INTERVAL '60 seconds'
            """)
            for row in stale:
                await conn.execute("UPDATE document_index SET status = 'PENDING' WHERE id = $1", row.id)

        await asyncio.sleep(30)
```

That's the core. Plus per-document atomic-flip check after each modality's transition to ACTIVE — also ~30 lines.

### I.4. Backpressure for LLM rate-limits

A simple Redis-based token bucket per resource:

```python
async def acquire_llm_token(redis, bucket='llm', capacity=10, refill_rate=10/60.0):
    """10 LLM calls per minute, refilled smoothly."""
    while True:
        now = time.time()
        # Lua-script atomic refill + acquire
        result = await redis.eval(BUCKET_LUA, keys=[bucket], args=[capacity, refill_rate, now])
        if result == 1:
            return  # got token
        await asyncio.sleep(0.5)  # wait and retry
```

This is invoked at the top of `graph.derive()` and `summary.derive()` (which call LLM). When the bucket is empty, the worker quietly waits — no exceptions, no retries. This eliminates the retry storm under rate-limiting.

### I.5. What we explicitly don't do

- **Reconciler-trigger / outbox / CDC**: deferred indefinitely; the 30s polling cycle is fine for 100 concurrent docs and pre-launch scale. If latency becomes a problem at much larger scale, revisit then.
- **Cross-modality compensation**: if vector succeeds and graph fails, vector ACTIVE rows stay; graph FAILED rows retry independently. Document remains "partially indexed" (`index_state` discriminator surfaces this) until graph succeeds. No rollback of vector.
- **Distributed tracing**: not in this design pack. Add later if debugging requires it.

---

## §J. Observability

### J.1. The four SLI minimum

```
1. index_lag_seconds{collection_id, modality}
   - From document upload time to (modality, parse_version) ACTIVE
   - p50, p95, p99
   - Surfaces: how long does indexing take?

2. index_failure_total{collection_id, modality}  +  index_success_total{collection_id, modality}
   - Counter pair (monotonic). Worker emits one bump per terminal transition.
   - Rate computed downstream by aggregator (PromQL: rate(failure[5m]) / (rate(failure[5m]) + rate(success[5m]))).
   - Why counter pair instead of single rate gauge: counter pair is OTLP-idiomatic, preserves raw events, re-aggregates cleanly across workers, no per-worker sliding-window state. Amended 2026-04-26 per huangheng Wave 1 CR finding (msg=8e67bf0e) + architect ruling.
   - Surfaces: which modality is broken (rate downstream); which worker burned the failures (raw counter).

3. queue_depth{modality}
   - Redis LLEN of each queue
   - Surfaces: are workers keeping up?

4. worker_utilization{modality}
   - fraction of time worker is doing actual work vs waiting
   - Surfaces: are we under-provisioning workers?
```

### J.2. Implementation

- Workers emit metrics to OTLP (already on the backlog as PR #1702)
- Reconciler emits queue depths on every poll cycle
- Index lag computed at atomic-flip time: `flip_at - upload_at`

### J.3. Dashboard (deferred)

The four SLI are emitted to OTLP. Dashboard composition is an ops decision (Grafana / DataDog / etc.) and not in scope of this design pack. The architect recommends one row per modality showing all four metrics; that's the only thing operators need to know.

### J.4. PR #1702 OTLP alignment opportunity

PR #1702 introduces OTLP infrastructure but is awaiting earayu2 routing decision. This redesign's Phase 1 (observability) is an excellent host for that infrastructure landing — combine them.

---

## §K. Migration plan — 3 waves, parallel-friendly

**earayu2 拍板（`msg=cc0a00d7`）**: "搞大 PR，少 PR，这样可能完成的快一点，少一点上下文切换；如果任务之间没太多依赖，可以并行让多人写代码。"

v2 把 v1 的 7 个细粒度 PR 收成 **3 个 wave**（≈3 个大 PR）。每个 wave 内部可拆 commit 但合一次 review、合一次 merge。Hard cut 一次完成，不留 feature flag 长期共存。

### K.1. 3 个 wave 总览

| Wave | 主题 | 是否可并行 | 估算 diff |
|---|---|---|---|
| **Wave 1** — Foundation | schema + Modality ABC + idempotency + derived/ 落盘 + 对象存储 adapter + observability primitives | ✅ Wave 内 5 modalities + observability + storage adapter 可并行 | +2200 / -1500 |
| **Wave 2** — Runtime | Redis queue + asyncio worker pool + reconciler + per-modality cutover + cleanup worker + simple quota | ✅ workers / reconciler / cleanup / quota 可并行 | +1400 / -300 |
| **Wave 3** — Cutover | hard-delete Celery layer + index_state 暴露到 SearchResultMetadata + 私有化部署默认 inline mode 文档 + 100-doc 负载测试 | ⚠ 需 Wave 2 落地后再做（依赖） | +250 / -3200 (delete-heavy) |

净增减：**+3850 / -5000 ≈ 净减 1150 行**（v2 版减得更多，因为去掉 atomic-flip orchestration 多砍 ~150 行）。

### K.2. Wave 1 — Foundation

**目标**: 新代码全 ready，但还没有 worker 拉它跑。可以与 main Celery 系统共存而不冲突（新代码独立路径）。

**含 4 块并行可写的内容**:

1. **Schema + Modality ABC**（1 人，~400 行）
   - 新建 `aperag/indexing/base.py`（Modality ABC, `derive(...) / sync(...)` 抽象）
   - 新建 `aperag/indexing/schema.py`（`document_index` 新表 alembic migration，含 `is_serving` 列、`tenant_scope_key`、`derived_artifact_path`）
   - 新建 `aperag/indexing/object_store.py`（LocalFS / S3-compatible adapter）

2. **5 个 Modality 实现**（最多 5 人并行，~150-300 行/模态）
   - `aperag/indexing/{vector,fulltext,graph,summary,vision}.py`
   - 每个实现 `derive` + `sync`，`sync` 必须 DELETE-before-INSERT，附幂等自测
   - **`graph.py` 合并 `aperag/graphindex/v2/*` 的逻辑，删除 7-hop indirection**（earayu2 specific complaint 修复）
   - **`graph.py` 修 nebula append-on-conflict bug**（§A.6 / §D.3）

3. **Parser → derived/**（1 人，~300 行）
   - `aperag/indexing/parser.py`：parse → `markdown.md` + `outline.json` + `chunks.jsonl`
   - parse_version 计算复用 D10.g §E.2 既有逻辑

4. **Observability primitives + OTLP 对齐 #1702**（1 人，~200 行）
   - `index_lag_seconds` / `index_failure_total` + `index_success_total` (counter pair) / `queue_depth` / `worker_utilization` 4 SLI
   - 与 PR #1702 OTLP infra 合在一起（在 #celery 单独 ack 时讨论）

**Wave 1 不做**:
- 不接 Redis 队列（modality 独立可单测）
- 不删 Celery（新旧并存，Celery 仍走原路径）
- 不改 search 路径

**Wave 1 完工的 acceptance**: 5 modality 都能 `pytest tests/unit/indexing/` 跑通幂等自测；ABC 接口 frozen。

### K.3. Wave 2 — Runtime

**目标**: 把 Wave 1 的 modality 接到 Redis 异步 worker pool，跑通 100-doc burst。Celery 仍未删，但默认流量改走新路径。

**含 4 块并行可写的内容**:

1. **Worker pool + Redis queue**（1 人，~500 行）
   - `aperag/indexing/orchestrator.py`：BLPOP loop + asyncio semaphore + heartbeat
   - 5 个 worker entrypoint（小，~50 行/个，复用 orchestrator）

2. **Reconciler + cleanup**（1 人，~250 行）
   - `aperag/indexing/reconciler.py`：30s loop（PENDING dispatch + FAILED retry + RUNNING reclaim + per-modality cutover trigger）
   - `aperag/indexing/cleanup.py`：5min loop（GC 旧 parse_version + 已删 document）

3. **Per-modality cutover transaction**（与 reconciler 同一人或者切给 modality owner）
   - §F.3 三语句事务封进 `Modality.commit_active(...)` helper

4. **Simple quota + bulkhead**（1 人，~150 行）
   - `aperag/indexing/quota.py`：Redis token bucket per `(resource_class, tenant_scope_key)`
   - `aperag/indexing/limits.py`：超时 / 上传大小 / 内存上限的统一配置

**Wave 2 不做**:
- 不删 Celery（默认 INDEXING_MODE=async 切到新系统，但 Celery 代码还在）
- 不暴露 `parse_version` / `index_state_per_modality` 到 search 出参（Wave 3）
- 不写 deploy-and-forget 文档

**Wave 2 完工的 acceptance**:
- Synthetic 100-doc burst 跑通：所有 modality ACTIVE within 30min（graph 是瓶颈）
- 主路径切换：`INDEXING_MODE=async`（默认）走新系统；`INDEXING_MODE=inline` 单进程同步走（§E.5）

### K.4. Wave 3 — Cutover & cleanup

**目标**: hard-cut 删除 Celery + 全部老代码 + 暴露 search 元数据 + 出私有化部署文档。

**Wave 3 内容**（顺序，因为依赖 Wave 2 跑稳）:

1. **删 Celery 层**（~200 行新建配置 / -3000 行删除）
   - `aperag/tasks/{collection,document,models,processing_lease,reconciler,scheduler,utils}.py` — 整层删除
   - `aperag/domains/indexing/{tasks,orchestration,manager}.py` — 删（被 `aperag/indexing/orchestrator.py` 替代）
   - `aperag/concurrent_control/redis_lock.py` — 删（无 caller）
   - `aperag/graphindex/v2/*` — 删（被 `aperag/indexing/graph.py` 吞并）
   - `aperag/domains/indexing/{vector,fulltext,graph,summary,vision}_index.py` — 删（被 `aperag/indexing/<modality>.py` 替代）
   - Celery + kombu 依赖从 `pyproject.toml` 移除

2. **`SearchResultMetadata` 加 `parse_version` + `index_state_per_modality`**（~80 行新增 / 50 行测试）
   - 既有 D10.h allowlist 模式扩展
   - 在 `aperag/api/search.py` 出参补两字段（读自 `document_index.is_serving=TRUE` 行 + `status` 投影）

3. **私有化部署文档 + inline mode**（~100 行新增 + ~150 行 docs）
   - `docs/private-deployment.md`：单机最小 stack（SQLite + LocalFS + INDEXING_MODE=inline + 无 Redis）→ §L 落地
   - `INDEXING_MODE=inline` 路径：HTTP handler 直接调 `derive` + `sync`，跳过 RPUSH

4. **Synthetic 100-doc load test 进 CI**（~200 行）
   - tests/load/test_100_doc_burst.py：并发 upload 100 文档，断言所有 modality ACTIVE within 30min

**Wave 3 完工的 acceptance**:
- Celery 完全删干净（grep 验证）
- search API 出参带 `parse_version` + `index_state_per_modality`
- 私有化部署文档可读 + 单机部署能跑 demo
- 100-doc burst CI gate 通过

### K.5. 并行度估算

| Wave | 同时可开工人数 |
|---|---|
| Wave 1 | 4-7 人（5 modality + parser + observability + storage adapter，互不依赖） |
| Wave 2 | 3-4 人（worker / reconciler / cleanup / quota） |
| Wave 3 | 1-2 人（Celery 删 + SearchResultMetadata 扩展 + 部署文档，少量耦合） |

总人时估算（如果 5 人并行可写）：Wave 1 ≈ 1 周，Wave 2 ≈ 1 周，Wave 3 ≈ 0.5 周；总 **2.5 周**。如果 2 人并行：3 周内可控。

### K.6. PR 拆分边界规则（少 PR / 少上下文切换）

每个 wave = **一个大 PR**（不是 3 个 PR 集合）。Wave 内并行写代码 → push 到同一个分支 → 一次 review、一次 merge。

理由（earayu2 directive 落地）：
- 少上下文切换：reviewer 一次看完一整层
- 少 PR：3 次 review cycle vs 7 次
- 并行不冲突：5 modality 各自独立文件，不动同一个文件，git 合并干净
- Hard cut 一次到位：Wave 3 一个 PR 删 3000 行老代码，没有半程共存的复杂度

### K.7. Wave 4 — Production-readiness（Wave 3 fix-cycle 教训, architect msg=baf6618e amendment）

**目标**: 把 Wave 1+2 ship 的 placeholder layers (InMemoryWorkQueue / InMemoryQuotaBackend / NoopMetricsEmitter / InMemoryLineageGraphStore / no-op LLM extractor / no-op multimodal vision-LLM) wire 到真后端，让新 indexing pipeline 真正 production-ready。

**Wave 4 11 项 backlog (PR #1731)**:

1. T1 — Real graph LLM extractor (chunks → entities/relations)
2. T2 — Cleanup loop 5 modality singleton fan-out（per-row worker_factory）
3. T3 — Real parser (PDF/Word/image via DocParser/Marker/OCR)
4. T4 — Real Redis WorkQueue backend (替 InMemoryWorkQueue)
5. T5 — Real Redis QuotaBackend Lua atomic (替 InMemoryQuotaBackend)
6. T6 — OTLP MetricsEmitter wire-in production (替 NoopMetricsEmitter)
7. T7 — Real multimodal vision-LLM + vision modality production wiring
8. T8 — graph 3 backend adapter wiring (Postgres / Neo4j / Nebula `LineageGraphStore`)
9. T9 — fulltext multi-backend adapter dispatch (`collection.config.fulltext_backend_type`)

**T8 chunk 4 acceptance 9 项**：alembic ORM mirror / drop relation `description` / async cross-event-loop / 6-case cross-backend contract test / EntityLock injection / factory dispatch on `graph_backend_type` / **grep-zero verify (narrowed scope, architect msg=87e2b187 chunk 4d ruling)** / spec amendments §C.1+§D.3.5+§H.5×2 / Phase 1 e2e production smoke。

**chunk 4d narrowed scope (architect msg=87e2b187 ruling Option C)**：grep-zero verify NEW indexing pipeline (`aperag/indexing/*`) **不 cross-reference** legacy `aperag/domains/knowledge_graph/graphindex/storage/{base,postgres,neo4j,nebula,connector}.py`。**不删 legacy file** — legacy `graphindex` package 整体淘汰是 cross-cutting refactor，移交 Wave 5。Wave 4 chunk 4d 只 lock invariant：新 pipeline 内部不能再回头依赖 legacy storage。

**Wave 4 production-readiness invariant pattern (per `feedback_production_readiness_invariant.md`)**: each layer 在 Wave-close 前 spec 显式列出 must-be-real / may-be-gated / fully-resolves，gate placeholder 用 `WorkerFactoryError` + self-disable detection（chunk 4b T1-extractor gate / vision multimodal gate 是这条 pattern 的现行实例）。

### K.8. Wave 5 — Legacy graphindex 淘汰 + retrieval/curation 迁移（Wave 4 close-out 后 follow-up）

**目标**: 完成 Wave 4 chunk 4d 推迟的 legacy `graphindex` package 整体淘汰 + retrieval/curation 调用迁移到 §G.5 read primitives。

**scope (per architect msg=87e2b187 chunk 4d ruling Option C deferral)**:

- delete `aperag/domains/knowledge_graph/graphindex/storage/{base,postgres,neo4j,nebula,connector}.py` (Wave 3 hard-cut 第二轮 deferred)
- delete `aperag/domains/knowledge_graph/graphindex/{__init__,service,integration}.py` + `engine/`
- migrate callers:
  - `aperag/domains/retrieval/pipeline.py:85` → §G.5 read primitives
  - `aperag/domains/knowledge_graph/service.py:69+` → graph CRUD via new `LineageGraphStore` find/get methods
  - `aperag/graph_curation/service.py:37` + `integration.py:21` → curation reads via §D.3 lineage-aware path
  - `aperag/indexing/worker_factory.py:696` `build_collection_llm_callable` → relocate to `aperag/indexing/llm.py`
  - `aperag/service/prompt_template_service.py:161` `ENTITY_RELATION_EXTRACTION` → relocate to T1 LLM extractor module
- delete tests:
  - `tests/unit_test/graphindex/test_connector.py`
  - `tests/unit_test/graphindex/test_nebula_store.py`
  - `tests/integration/compat/test_graph_compat.py`

**为什么 Wave 5 而非 Wave 4 chunk 4d**: legacy `GraphStore` Protocol (24-method LightRAG-style flat-graph) 与新 `LineageGraphStore` Protocol (10-method §D.3 lineage-aware) **API 不同**，不是 1-to-1 替换。retrieval/curation 调用迁移涉及 LightRAG-style flow → §G.5 lineage-aware read primitives 的语义重写，是 cross-cutting refactor 而非 hard-cut。Wave 4 chunk 4d 强行做会触发 Wave 3 fix-cycle 同款 risk (per huangheng msg=87e2b187 CR analysis lesson #9 reference)。

**Wave 5 其他 backlog**（accumulated during Wave 4）:

- per-collection store-instance TTL cache (when collection count > 10K; architect msg=95179f2a Design point 2)
- W5-perf-graph-lineage: parallel-list O(N) alternative encoding for high-cardinality entities (>10k docs/entity)
- W5-neo4j-label-namespace: prefix `aperag_LineageEntity` / `aperag_LineageRelation` to avoid user-namespace collision
- W5-cypher-type-keyword: rename `n.type` property (Cypher `TYPE()` keyword shadow); cross-backend rename also in Postgres/Nebula
- W5-otlp-config-cross-check: lifespan startup cross-check `INDEXING_METRICS_EMITTER=otlp` ⇔ `APERAG_OBSERVABILITY_MODE=otlp`
- W5 reconciler "document 创建 N 分钟无 document_index rows → re-enqueue parse" (T3 chunk 2 obs A failure semantic)
- W5 parse_orchestrator short-circuit on existing parse_version artefact (T3 chunk 2 obs B)
- W5 `tenant_scope_key` org-prefix forward-compat (T3 chunk 2 obs C)
- W5 `_resolve_cleanup_worker` narrow exception types (T2 obs A)
- W5 cleanup builder share helpers with dispatch builders (T2 obs B drift risk)
- W5 e2e-http-compose lane stub model-provider fixture + Layer 2 full-pipeline test activation (chunk 4e Layer 2 tests are currently `pytest.skip(...)` stubs because vector/fulltext/summary embedders need a configured model-provider that local dev does not have; the e2e-http-compose lane has the scaffolding but the stub fixture must be wired so Layer 2 can run for real instead of skip — per architect msg=87e2b187 chunk 4d/4e ratify decision condition #3)

### K.9. Wave 5 — Indexing program close-out + 16 backlog items（per architect msg=11442bbf reframe + earayu2 msg=eced858d "大PR 快速落地" directive）

**目标**: 把 Wave 4 close-out 后剩余 16 backlog items 一次性 implement，single Wave 5 PR with 5-phase chunked-rotation commits。Wave 4 同款 batch pattern (PR #1731 推 11 items)。

**5-phase commit roadmap (16 acceptance items)**:

#### Phase 1 (narrowed scope per architect ruling 2026-04-27 post-cascade scope check) — `aperag/indexing/llm.py` relocate + import re-route

**Scope evolution** (per Bryce scope reality check msg=30c7e994 + architect ruling Option (a) adoption msg-post-2026-04-27-18:10):

The original Phase 1 scope (delete legacy ``graphindex/{storage, service, integration, engine, __init__}.py`` + migrate retrieval/curation callers to §G.5 read primitives) collided with an unresolved design gap: legacy ``GraphIndexService.query_context()`` is a 24-method LightRAG-style query API, while the new ``LineageGraphStore`` Protocol exposes only 4 read methods (``find_entity_ids_with_lineage`` / ``find_relation_keys_with_lineage`` / ``get_entity`` / ``get_relation``) bound to ``(document_id, parse_version)`` lineage scans + single-key fetches. There is no by-keyword / by-semantic / by-graph-traversal API surface on the new pipeline yet — that is itself a 1-2 week design + implementation effort (build LightRAG-style query layer on ``LineageGraphStore`` with vector recall + traversal).

Per architect Option (a) ruling: **Phase 1 narrowed to relocate-only scope**. Full legacy ``graphindex`` package elimination and retrieval/curation cascade migration are deferred to **Wave 6** as a coordinated cross-cutting refactor PR alongside the new query layer design.

**Phase 1 actual scope (Wave 5 chunks 1+2 already shipped)**:

- relocate ``build_collection_llm_callable`` + ``render_extraction_prompt`` + ``ENTITY_RELATION_EXTRACTION`` template + ``LLMCall`` type alias → new ``aperag/indexing/llm.py`` (Wave 5 PR commit ``11113acb``)
- legacy ``graphindex/integration.py`` and ``graphindex/prompts.py`` keep deprecation-shim re-exports during the Wave 6 deprecation window so legacy retrieval/curation callers do not break mid-Wave-5
- migrate ``aperag/indexing/graph_extractor.py`` import to new location (Wave 5 PR commit ``11113acb``)
- migrate ``aperag/service/prompt_template_service.py`` ``get_default_prompt(prompt_type="graph")`` import to new location (Wave 5 PR commit ``e0707165``)

**production-readiness 三类 layer (revised)**:
- must-be-real: ``aperag.indexing.llm`` is the canonical home for the relocated helpers; Wave 4 indexing pipeline (``graph_extractor`` + ``worker_factory`` chunk 4b graph dispatch) imports from new location, no transitive legacy dependency
- may-be-gated: legacy ``graphindex/{integration,prompts}.py`` re-export shims are intentional Wave 6 deprecation-window devices, not gated defaults
- fully-resolves: §K.8 Wave 5 backlog item 3 (``aperag/indexing/llm.py`` relocate). Wave 5 backlog item 1 (legacy graphindex package elimination) **deferred to Wave 6** — see §K.10.

**Wave 6 entry condition** (per architect Option (a) ruling): a separate Wave 6 PR covers the legacy graphindex retrieval-side query layer migration. Phase 1 narrowed scope avoids the Wave 3 lesson #10 anti-pattern (silent regression by forcing a 1-week design refactor into a high-throughput PR) and follows the chunk 4d Option C precedent (narrowed scope + Wave N+1 cross-cutting refactor batch).

**pre-check pattern** (per `feedback_spec_lock_grep_verify_caller.md` Pattern 1):
- ``grep -rn "from aperag.domains.knowledge_graph.graphindex" aperag/indexing/`` → 0 matches (chunk 4d invariant maintained ✅)
- legacy callers (``retrieval/pipeline.py`` / ``knowledge_graph/service.py`` / ``graph_curation/*``) **intentionally** still import from legacy modules pending Wave 6 migration

**architect direct ratify scope**: chunk 1 (``11113acb``) ratified ✅; chunk 2 (``e0707165``) huangheng pass-1 ✅. No further Phase 1 architect direct ratify gate needed in Wave 5 — the cascade chunks (3-7) move to Wave 6 with their own architect ratify lane (cascade caller migration spec + completeness verify).

#### Phase 2 (post-Phase-1) — T7 multimodal vision-LLM 3-item bundle

per §G.2.5.1 spec amendment (Wave 4 chunk 4e follow-up `da576f62`):

- **item 1**: `aperag/llm/embed/embedding_service.py` 加 `embed_image(image_bytes: bytes, alt_text: str = "") -> list[float]` API surface; provider implementation 通过 v3 router resolve
- **item 2**: parser pipeline 加 image extraction → write `derived/parse_<v>/vision/images/<image_id>.<ext>` (per-page image bytes for PDF / single-image-input passthrough); 同 batch 实施 chunk_id schema unification (per huangheng T1 obs B)
- **item 3**: `aperag/domains/model_platform/api/providers_v3_routes.py` 加 multimodal capability flag (`is_multimodal=True` source of truth); operator can set on collection embedder spec
- chunk 4b vision gate **self-disables** when `is_multimodal()` flips True + provider flag set

**production-readiness 三类**:
- must-be-real: real multimodal LLM provider integration (CLIP / LLaVA / GPT-4V / etc. via v3 router)
- may-be-gated: 无 (full implementation lands)
- fully-resolves: §K.8 Wave 5 backlog item 4 (T7 multimodal full impl 3-item bundle)

**pre-check** (per Pattern 2): grep `embed_image` / `is_multimodal` 全 verify 现 code binding (chunk 4b gate `is_multimodal()` call site already exists)

**architect direct ratify scope**: §G.2.5.1 spec ↔ implementation align (3-item bundle full impl) + chunk 4b vision gate self-disable verify (`is_multimodal=True` 实测 + Layer 1 test rename `test_phase1_vision_modality_*` 改 expect ACTIVE)

#### Phase 3 (parallel with Phase 2) — Layer 2 e2e fixture wiring + sweep D activation

per chunk 4d+4e ratify msg=c279a0ff + sweep D Layer 2 stub (`4d36c7fb`):

- e2e-http-compose lane stub model-provider fixture wire
- activate `test_phase1_full_pipeline_vector_fulltext_summary_active_graph_vision_failed` Layer 2 stub
- activate `test_phase1_multi_keyword_fulltext_search_returns_hits` Layer 2 stub (sweep D minimum_should_match real verify)

**production-readiness 三类**:
- must-be-real: real Postgres / Redis / Qdrant / ES / OTel SDK + stub model-provider fixture
- may-be-gated: 无 (Layer 2 真激活)
- fully-resolves: §K.8 Wave 5 backlog item 2 (Layer 2 e2e fixture wiring)

**architect direct ratify scope**: Layer 2 fixture wiring canonical contract (vector+fulltext+summary ACTIVE + chunk 4b vision gate FAILED in Phase 2 deferred state until Phase 2 lands flips to ACTIVE) + delete+cleanup roundtrip backend artefact 0-count verify

#### Phase 4 (parallel with Phase 1) — P1 production robustness 3-pack

3 commits in same phase batch:

1. **T2 cleanup `_resolve_cleanup_worker` transient-vs-intentional error split** (per T2 architect ratify msg=6aa8ca88):
   - `WorkerFactoryError` (intentional gate, drop row) vs 其他 `Exception` (transient, NOT drop row, retry next cycle)
2. **T3 chunk 2 parse_orchestrator parse_version short-circuit** (per huangheng T3 chunk 2 obs B):
   - `if derived/parse_<v>/chunks.jsonl 已存在 → skip DocParser，直接 dispatch`
3. **T2 reconciler "document N min 无 document_index rows → re-enqueue parse"** (per huangheng T3 chunk 2 obs A):
   - cleanup loop 加 stuck document detection + parse re-enqueue

**production-readiness 三类**:
- must-be-real: 3 production fault-tolerance fixes (real e2e flow benefit)
- may-be-gated: 无
- fully-resolves: §K.8 Wave 5 backlog items 7 + 9 + 10

#### Phase 5A (post-Phase-4, Bryce batch) — P2 batch graph + perf polish

5 commits in same phase batch (Bryce ownership per Wave 4 chunk 4 / T1 / T8 expertise):

1. **T1 graph extractor per-collection config tunability** (per huangheng T1 obs A): `collection.config.knowledge_graph_config.{per_chunk_timeout, max_entities_per_chunk, max_relations_per_chunk}` per-collection override — **shipped `f5e454ed`**
2. ~~chunk 4b `_no_op_extractor` identity check → attribute marker~~ — **N/A: resolved by Wave 4 T1 land**. The placeholder was deleted when `build_collection_graph_extractor` replaced `_no_op_extractor` (see commit `19d3d70f`); there is no surviving identity-check site to harden, so this item is moot in current code.
3. ~~W5-perf-graph-lineage parallel-list O(N) alternative encoding~~ — **deferred to Wave 6**. Cross-backend perf rewrite touches Postgres / Nebula schema (Postgres switching from JSONB-array-of-objects to parallel `text[]` columns; Nebula re-modelling tags) plus alembic migration. High-cardinality (>10k docs/entity) is not a Wave 5 acceptance criterion — defer to dedicated perf wave once observed in real deployments.
4. **W5-neo4j-label-namespace prefix `aperag_LineageEntity` / `aperag_LineageRelation`** to avoid user-namespace collision — **shipped `42b5fa6a`** (Neo4j-only constant + comment rename, no schema migration needed since Wave 4 default-gated graph indexing meant no production data on legacy labels).
5. ~~W5-cypher-type-keyword rename `n.type` property~~ — **deferred to Wave 6**. Although `TYPE(n)` is a Cypher built-in for relationships not nodes (so node-property `n.type` is technically unambiguous), a forward-compat rename across all 3 backends (Postgres column rename + alembic migration + Cypher property rename + Nebula tag-prop rename + `EntityRecord.type` / `RelationRecord.type` Protocol surface) is non-trivial. Keep current naming until Wave 6 dedicated cross-backend rename batch.

**fully-resolves**: §K.8 P5A items 1 + 4 (items 2 / 3 / 5 redirected as noted)

#### Phase 5B (post-Phase-4, chenyexuan batch) — P2 batch infra + cleanup polish

5 commits in same phase batch (chenyexuan ownership per Wave 4 T2 / T3 / T9 expertise):

1. **T1 chunk_id schema unification on parser layer** (per huangheng T1 obs B; partially overlap with Phase 2 item 2)
2. **T2 cleanup builder share helper with dispatch builders** (per huangheng T2 obs B drift risk refactor)
3. **T3 chunk 2 `tenant_scope_key` org-prefix forward-compat** (per T3 chunk 2 obs C)
4. **chunk 4a `utc_now` Python default vs `CURRENT_TIMESTAMP` server_default unification** (per huangheng chunk 4a obs A)
5. **W5-otlp-config-cross-check** lifespan startup cross-check `INDEXING_METRICS_EMITTER=otlp` ⇔ `APERAG_OBSERVABILITY_MODE=otlp` (per huangheng T6 chunk 1 obs)

**fully-resolves**: §K.8 Wave 5 backlog items 6 + 8 + 11 + remaining accumulated obs

#### K.9.1. Wave 5 acceptance summary (16 items)

| # | Item | Phase | Owner |
|---|---|---|---|
| 1 | Legacy graphindex package 整体淘汰 | 1 | Bryce |
| 2 | Layer 2 e2e fixture wiring | 3 | chenyexuan |
| 3 | `aperag/indexing/llm.py` relocate | 1 | Bryce |
| 4 | T7 multimodal vision-LLM 3-item bundle | 2 | Bryce |
| 5 | T1 graph extractor per-collection config tunability | 5A | Bryce |
| 6 | T1 chunk_id schema unification | 2 / 5B | Bryce / chenyexuan |
| 7 | T2 cleanup transient-vs-intentional error split | 4 | chenyexuan |
| 8 | T2 cleanup builder share helper | 5B | chenyexuan |
| 9 | T2 reconciler stuck-document re-enqueue | 4 | chenyexuan |
| 10 | T3 chunk 2 parse_version short-circuit | 4 | chenyexuan |
| 11 | T3 chunk 2 tenant_scope_key org-prefix | 5B | chenyexuan |
| 12 | fulltext additional backends extension | 5A | Bryce |
| 13 | chunk 4a utc_now vs CURRENT_TIMESTAMP unify | 5B | chenyexuan |
| 14 | chunk 4b _no_op_extractor identity check robust | 5A | Bryce |
| 15 | W5-perf-graph-lineage parallel-list O(N) | 5A | Bryce |
| 16 | W5-otlp-config-cross-check + Neo4j label namespace + Cypher type rename | 5A / 5B | Bryce / chenyexuan |

#### K.9.2. Wave 5 architect direct ratify lane

per Wave 4 lane lock (chunk 4 / T1 / T5 / T7 ratify trail), Wave 5 architect direct ratify scope:

- **Phase 1**: legacy graphindex elimination caller migration completeness verify (grep-zero post-Phase-1 + `aperag/indexing/llm.py` relocate clean)
- **Phase 2**: T7 §G.2.5.1 spec ↔ implementation align (3-item bundle full impl) + chunk 4b vision gate self-disable verify
- **Phase 3**: Layer 2 e2e fixture wiring canonical activation
- **Phase 4 / 5A / 5B**: 走 huangheng pass-1 lane only; architect spot-check on architectural soundness。如出现 spec/state-binding drift → architect direct ratify trigger (per Wave 4 T5 §H.5.1 fix-forward 教训)

#### K.9.3. Pre-check pattern lock (per `feedback_spec_lock_grep_verify_caller.md` 双 pattern)

- **Pattern 1** (caller cascade): Phase 1 必跑 — grep `from aperag.domains.knowledge_graph.graphindex` 全 caller list verify migration scope
- **Pattern 2** (state binding): 任何新 `collection.config.*` field / `settings.*` setting / logical db / port / path lock — 必跑 grep `from settings import` / `getattr(cfg,...)` 现 code binding verify

#### K.9.4. Layer 1 same-session continuation directive (per `feedback_no_refresh_complete_all_tasks.md`)

Wave 5 implementation 同 Wave 4 directive:
- earayu2 不再 帮 fresh-restart agent session
- 各 implementer same-session continuation push 节奏 maintained
- Layer 1 metric goal: ≥ 90% same-session ratio (Wave 4 verified 65% same-session + 100% pass-1 一次过率)
- chunked-rotation within PR (multiple commits in same branch / same session)

### K.7. 测试策略

- **Unit tests**: 每个 modality 配幂等自测（Wave 1 gate）
- **Integration tests**: end-to-end upload → indexed → searchable（Wave 2 gate）
- **Synthetic load test**: 100-doc burst（Wave 3 gate，进 CI）
- **Smoke for inline mode**: 单机 SQLite + LocalFS（Wave 3，docs 配套）

### K.8. Implementation 分工建议

PM (燧木) 决定。架构师建议参考 D10 模式：
- 架构师 (符炫炜) — design canon、wave scope、line-by-line review、跨 modality 对齐
- 单 modality 写手（5 人 × Wave 1 一人一个 modality + Wave 2 worker / reconciler 等）
- Bryce — graph modality（最复杂的那个，且修 nebula append bug）+ idempotency 自测把关

### K.10. Wave 6 — Legacy graphindex elimination + retrieval/curation migration（per architect Option (a) ruling 2026-04-27 + Wave 5 Phase 1 scope-narrow handoff）

**目标**: 完成 Wave 5 Phase 1 narrowed scope 推迟的 legacy ``graphindex`` package 整体淘汰 + retrieval/curation flows 迁移。这是 cross-cutting refactor，与 Wave 5 single-PR 风格不同 — 需先设计新查询层 (LightRAG-style query layer on ``LineageGraphStore``)，再 cascade migrate 调用方。

**Wave 6 acceptance items** (per Wave 5 §K.9 scope-narrow handoff + Option C ruling msg=6fccd9ab narrowing post-chunk-3 caller-cascade gap):

1. **Design + implement LightRAG-style query layer** on ``LineageGraphStore`` Protocol — **✅ shipped via #33 chunks 1+2 (PR #1737 + #1741)**:
   - ✅ by-keyword entity lookup (`query_entities_by_keyword`)
   - ❌ ~~vector-recall augmented retrieval~~ — **deferred per chunk 2 Option A ruling msg=54eac595** (lineage schema 没 entity vector column; Wave 7+ if real evidence)
   - ✅ graph traversal API (`expand_neighbors_n_hops` with bounded `hops`)
   - ✅ query result types align with retrieval pipeline expectations (`EntityWithLineage` / `RelationWithLineage`)
2. **Migrate legacy ``graphindex`` callers to new query layer** — **partially shipped via #33 chunk 3 (PR #1742) per Option C ruling msg=6fccd9ab**:
   - ✅ ``aperag/domains/retrieval/pipeline.py`` `_graph_search` — 用新 `query_entities_by_keyword` + `expand_neighbors_n_hops` + `_render_graph_context_text` 替 ``GraphIndexService.query_context()``。retrieval/pipeline.py grep-zero verified post-chunk-3.
   - ❌ ~~``aperag/domains/knowledge_graph/service.py`` graph CRUD~~ — **deferred to evidence-based future Wave** (UI methods `get_labels` / `get_knowledge_graph` / `merge_entities` 不在 new Protocol surface — chunk 1 spec lock 时只 verify import-level + design `query_context` replacement，漏 enumerate UI/curation methods coverage; Pattern 1 强化 lesson sediment to `feedback_spec_lock_grep_verify_caller.md`)
   - ❌ ~~``aperag/graph_curation/service.py`` + ``integration.py`` + ``candidate_generation.py``~~ — **deferred to evidence-based future Wave** (依赖 graphindex.dto.Entity 类型 + LLM curation flows)
3. ❌ ~~**Delete legacy package**~~ — **deferred to evidence-based future Wave** (UI/curation flows 仍 reference legacy package; per simple-stable directive "if no evidence → defer"; UI/curation 现 work fine via legacy — 无 user-facing 急迫性)
4. ❌ ~~**Delete legacy tests**~~ — **deferred** (与 item 3 同 reason — UI/curation tests 仍 reference)
5. ❌ ~~**Final grep-zero verify** post-Wave-6, ``aperag/`` 全树无 graphindex import~~ — **deferred**; post-chunk-3 narrow grep-zero: `from aperag.domains.knowledge_graph.graphindex` 仅在 4 UI/curation files (knowledge_graph/service.py + 3 graph_curation/) — retrieval-side cutover ✅

**Wave 6 #33 close-out (narrowed)**: chunks 1+2+3 ship retrieval-side LightRAG-style query layer + retrieval/pipeline.py cutover。Legacy graphindex package retained for UI/curation。Wave 7+ task creation contingent on evidence-based real-world need (production user 报 issue / new UI feature 需 / refactor pressure 累积) — per simple-stable directive #1 "不无限扩范围" + audit-filter rule "if no evidence → defer"。

**Cross-backend lineage polish folded from Wave 5 P5A** (per Wave 5 P5A close-out
2026-04-27):

6. **W5-perf-graph-lineage parallel-list O(N) cross-backend** — Postgres
   switches the JSONB-array-of-objects to parallel ``text[]`` columns
   (matching Neo4j parallel-list encoding); Nebula re-models tags to
   parallel-list semantics; alembic migration for the column-shape
   change. Trigger when a deployment observes >10k docs/entity (per
   architect msg=39a74026 high-cardinality threshold).
7. **W5-cypher-type-keyword cross-backend rename** — rename
   ``EntityRecord.type`` / ``RelationRecord.type`` (and
   ``EntityWithLineage`` / ``RelationWithLineage``) to ``entity_type`` /
   ``relation_type`` across the §D.3 Protocol; Postgres column rename via
   alembic; Cypher ``n.type`` / ``r.type`` rewrite; Nebula tag-prop
   rename. Forward-compat hygiene (Cypher ``TYPE(r)`` is for relationships
   only so current code is technically unambiguous, but Protocol-surface
   rename frees future use of ``TYPE`` as a true Cypher keyword without
   ambiguity).

**production-readiness 三类 layer**:
- must-be-real: 新 LightRAG-style query layer ships behavior-equivalent retrieval (existing graph search results within ε tolerance)
- may-be-gated: 无 (full hard-cut)
- fully-resolves: §K.8 Wave 5 backlog item 1 deferred (legacy graphindex elimination) + supplementary cascade items

**effort estimate**: 1-2 weeks single-implementer or 0.5-1 week 2-implementer parallel (query layer design + migration + test rewrite + e2e behavior preservation verify)。

**architect direct ratify lane**: full Wave 6 — query layer Protocol design + migration completeness verify + behavior-preservation validation against retrieval / curation existing behaviors。

**why Wave 6 separate PR (not Wave 5 fold)**: Wave 5 PR (#1733) targets 16 polish items in single-PR fast-landing style. Wave 6 cross-cutting refactor needs (a) new design work upstream, (b) careful behavior validation against existing retrieval flows, (c) phased migration to avoid silent regression. Bundling into Wave 5 risked Wave 3 lesson #10 anti-pattern (rushed cross-cutting refactor in high-throughput PR → silent broken paths). Wave 6 separate PR with its own architect ratify lane is the disciplined path.

### K.11. Wave 6 — Full spec amendment (post-Wave-5 PR #1733 merge, 2026-04-27)

**目标**: §K.10 锁定 graphindex elimination 为 Wave 6 主项 (5 items + 2 perf 折叠)。Wave 5 close-out 后，task board 上的 Wave 6 实际 backlog 共 **7 acceptance items** (#33-#39)：§K.10 的 #33+#35+#36 三项，加上 4 项新发现的 polish items (#34 parser unify + #37 multimodal cache wiring + #38 cache cross-loop lazy-rebuild + #39 provider format variations)。本节 §K.11 即 Wave 6 完整 spec amendment — 在 §K.10 graphindex elimination 基础上扩展到全部 7 items 的 phase 拆分 + 同时锁 Wave 4-5 沉淀的 architect-side patterns (`feedback_simple_stable_zero_maintenance.md` / `feedback_spec_lock_grep_verify_caller.md` / `feedback_no_refresh_complete_all_tasks.md` / `feedback_production_readiness_invariant.md`)。

**与 §K.10 的关系**: §K.10 留作 graphindex elimination 单独 sub-spec (handoff from Wave 5 Phase 1 narrowed scope ruling)；§K.11 是 Wave 6 整体 spec — 7 items / 3 phase / 2 PR 战略 / 4 guardrail / 双 pre-check pattern lock。Wave 6 的 architect direct ratify lane 由 §K.11 lane lock 决定 (替 §K.10 单一 lane statement)。

#### K.11.1. Wave 6 acceptance items (7 items)

| # | Item | Owner | Phase | Effort | PR target |
|---|---|---|---|---|---|
| 33 | Legacy graphindex elimination + LightRAG-style query layer (foundational; per §K.10) | Bryce | A | 1-2 weeks | PR-A (graphindex) |
| 34 | Parser canonical schema unification (chunk_id field unify on parser layer + remaining utc_now/CURRENT_TIMESTAMP audit + spec amend) | chenyexuan | B | 1-2 days | PR-B (polish) |
| 35 | Lineage SET parallel-list O(N) cross-backend perf rewrite (Postgres JSONB→text[] + Nebula tag re-model + Neo4j parallel-list re-encode + alembic; per §K.10 item 6) | Bryce | B | 3-5 days | PR-B (polish) |
| 36 | Cypher TYPE() rename + EntityRecord.type Protocol surface rename (cross-backend column rename + §D.3 Protocol surface change; per §K.10 item 7) | Bryce | B | 2-3 days | PR-B (polish) |
| 37 | Multimodal embedding cache wiring (`EmbeddingService.embed_image` callsite 接 `aperag.cache.NAMESPACE_EMBEDDING`; image bytes → sha256 file_hash + provider/model + alt_text key shape; uses canonical PR #1734 cache infra per earayu2 directive msg=26d1509e) | 明书 | B | 1-2 days | PR-B (polish) |
| 38 | Application cache cross-loop lazy-rebuild + WARN log (fix `application_runtime.get_application_cache()` Noop fallback OR add metrics counter `cache_noop_due_to_loop_switch_total`; per huangheng PR #1734 sync point 7) | chenyexuan | B | 1-2 days | PR-B (polish) |
| 39 | Provider-specific multimodal embedder format variations (Voyage / Jina v3 / OpenAI multimodal SDK input format adjustments; Wave 5 chunk 1 用 LiteLLM documented shape per §G.2.5.1) | 明书 | C | 2-4 days | PR-B (polish), post-#37 |

#### K.11.2. 2-PR strategy (per `feedback_simple_stable_zero_maintenance.md` 4 guardrail #2 "尽快上线")

Wave 6 split 成 2 PR 是因为 #33 (graphindex elim + LightRAG-style query layer) 是 1-2 周 cross-cutting refactor，会自然拖住其他 6 个小型 polish items 的 ship pace。Bundle 进 single PR → 违反 simple-stable directive #2 "先把功能做实，希望尽快上线"。

| PR | Scope | Items | Branch | Owner driving |
|---|---|---|---|---|
| **PR-A "Wave 6 graphindex"** | foundational cross-cutting refactor — query layer Protocol design + cascade migrate retrieval/curation/service callers + delete legacy graphindex package | #33 | `bryce/celery-wave6-graphindex` | Bryce |
| **PR-B "Wave 6 polish batch"** | 6 bounded items: parser unify + perf rewrite + Cypher rename + multimodal cache + cross-loop rebuild + provider format variations | #34 + #35 + #36 + #37 + #38 + #39 | `wave6/celery-wave6-polish` | Bryce + chenyexuan + 明书 multi-implementer |

**Land order**: PR-B 先 ship (1 周内 land)，PR-A 后 ship (1-2 周 land；可与 PR-B parallel 推进，PR-B merge 不阻塞 PR-A 设计/迁移)。两 PR 之间无文件 overlap (#33 改 graphindex/* + retrieval/* + knowledge_graph/* + graph_curation/*; PR-B items 改 parser.py / graph_storage/* / embedding_service.py / cache/*) — mechanical merge 可预期。

#### K.11.3. Phase 拆分 + 依赖图

```
Phase A (PR-A):    #33 ─────────────────────────────────────────► merge
                     ↑ (parallel — no shared files with Phase B)
Phase B (PR-B):    ├─ #34 chunk_id unify (chenyexuan, 1-2 days)
                   ├─ #35 lineage perf rewrite (Bryce, 3-5 days)
                   ├─ #36 Cypher rename (Bryce, 2-3 days)
                   ├─ #37 multimodal cache wiring (明书, 1-2 days)
                   └─ #38 cache cross-loop rebuild (chenyexuan, 1-2 days)
Phase C (PR-B):    └── #39 provider format variations (明书, 2-4 days, post-#37)
                                                          │
                                                          ▼
                                                       PR-B merge
```

**Phase B 内部并行性**: 6 个 items 中 #34 / #35 / #36 / #37 / #38 互无 file overlap (parser.py / graph_storage/{postgres,neo4j,nebula}.py / embedding_service.py / aperag/cache/__init__.py + application_runtime.py)，可同步进行。#39 sequential after #37 是因为 cache wiring 落地后 provider format 才有 callsite stable。

#### K.11.4. Production-readiness 三类 layer (per `feedback_production_readiness_invariant.md`)

每个 acceptance item 必显式声明三类 layer scope:

| # | must-be-real | may-be-gated | fully-resolves |
|---|---|---|---|
| 33 | LightRAG-style query layer 真实实现 (by-keyword + traversal API) on `LineageGraphStore` + retrieval/pipeline.py cutover (`query_context()` → `query_entities_by_keyword` + `expand_neighbors_n_hops` + `_render_graph_context_text`)，行为等价 within ε tolerance for retrieval-side keyword + n-hop graph recall paths。**TWO honest defer**：(a) **Graph entity vector recall NOT in Wave 6 scope** — deferred to Wave 7+ if real evidence (per chunk 2 ruling msg=54eac595; lineage schema 没承担 entity vector storage); (b) **UI/curation flows migration NOT in Wave 6 scope** — `aperag/domains/knowledge_graph/service.py` + `aperag/graph_curation/*` 4 files 仍 reference legacy `graphindex/` package for `get_labels` / `get_knowledge_graph` / `merge_entities` UI methods (per chunk 3 ruling msg=6fccd9ab; chunk 1 spec lock 时只 verify import-level + design `query_context` replacement Protocol，漏 enumerate UI/curation methods coverage — Pattern 1 强化 lesson) | 无 (chunks 1+2+3 三 chunk 全 hard-cut applied within narrowed scope) | §K.8 Wave 5 backlog item 1 deferred + §K.10 sub-spec narrowed (items 1-2 ✅ shipped via chunks 1-3, items 3-5 deferred to evidence-based future Wave) |
| 34 | parser layer chunk_id 字段 schema 一致 + remaining utc_now → CURRENT_TIMESTAMP unify | parser legacy alt-shape transitional decoder OK to keep if needed for legacy fixtures | huangheng T1 obs B + Wave 5 P5B chunk_id 5th item 推迟项 |
| 35 | Postgres / Nebula / Neo4j 全 lineage SET storage 改 parallel-list O(N) encoding + alembic migration | 无 — hard-cut policy per earayu2 msg=30c81478 (无生产数据 → schema 直改) | §K.10 item 6 (`feedback_simple_stable_zero_maintenance.md` directive #4 "私有化部署免维护" align — operator deploy 后不管 schema migration since 自动 alembic upgrade) |
| 36 | `EntityRecord.type` → `entity_type` Protocol surface rename + Postgres column rename + Cypher property rewrite + Nebula tag-prop rename | 无 — hard-cut | §K.10 item 7 + §D.3 Protocol stability |
| 37 | `EmbeddingService.embed_image` 调用接 `aperag.cache.get_or_compute` (NAMESPACE_EMBEDDING) + sha256 file_hash key shape；hot-path real cache hit/miss measured | 无 — full impl, no shim. Cache miss 走 LiteLLM real call (chunk 1 落地的 `_embed_image_via_litellm` path) | post-#1734 (cache infra) + Wave 5 P2 chunk 4 callsite (`embed_image` API) prerequisite |
| 38 | `application_runtime.get_application_cache()` 不再静默返 Noop on loop switch；real lazy-rebuild OR explicit metrics counter `cache_noop_due_to_loop_switch_total` + WARN log | 第一次 loop switch 后 rebuild 仍可能小 cost (latency observation OK，但不静默 zero-cache) | huangheng PR #1734 sync point 7 / `feedback_announce_equals_landed.md` (silent zero-cache 是 narrative-truth violation) |
| 39 | Voyage / Jina v3 / OpenAI multimodal SDK input format 各自实现 (real provider tests OR documented capability matrix) | LiteLLM canonical shape 是 fallback；provider-specific format 是 polish item，可分 sub-batch incremental ship | §G.2.5.1 spec provider variations footnote |

#### K.11.5. Pre-check pattern lock (per `feedback_spec_lock_grep_verify_caller.md` 三 pattern)

**Pattern 1 (caller cascade — IMPORT count + METHOD coverage matrix)** 强制触发条件:

- **import-level pre-check** (Pattern 1 v1): `grep -rn "from X import" aperag/` 列 caller files 数量 + 分类 (test-only / new-code-self-ref / legacy-production)
- **method-level pre-check** (Pattern 1 v2 — added 2026-04-27 per Wave 6 #33 chunk 3 ruling msg=6fccd9ab lesson): FOR EACH caller file, enumerate USED METHODS — grep `legacy_X.METHOD(` calls。VERIFY EACH method 是否 在 new Protocol/API surface 内 covered。Method-level coverage 是 import-level coverage 的 superset — 单 import-level pass 不充分。

如 ANY caller method NOT covered → STOP: lock form 4 (extended)：narrow chunk to migrate only callers using covered methods; legacy package retained for un-covered methods; acceptance text 显式 declare un-covered methods + 决策 (add Protocol method same chunk / prior chunk / defer caller migration to next Wave)。

- **#33**: pre-implementation 必跑 (a) `grep -rn "from aperag.domains.knowledge_graph.graphindex" aperag/` 列全 caller graph (5 files: retrieval/pipeline.py + knowledge_graph/service.py + graph_curation/*); (b) FOR EACH caller, enumerate `GraphIndexService.METHOD(` calls — verify each method 在 new `LineageGraphStore` Protocol covered。**Wave 6 #33 retrospective miss**: chunk 1 spec 时只 verify (a) + design `query_context` replacement Protocol，未 enumerate (b) UI/curation methods (`get_labels` / `get_knowledge_graph` / `merge_entities`)。chunk 3 实施时 surface gap → Option C ruling narrow + UI/curation defer to evidence-based future Wave。
- **#36**: pre-implementation 必跑 `grep -rn "EntityRecord.type\|RelationRecord.type\|\.type=" aperag/indexing/ aperag/domains/` 列 Protocol surface caller graph + Cypher template caller graph + Postgres column caller graph，避免 rename 漏 site

**Pattern 2 (state binding — runtime config)** 强制触发条件:

- **#34**: pre-implementation 必跑 `grep -rn "chunk_id" aperag/indexing/parser*.py aperag/indexing/dispatcher.py aperag/indexing/cleanup.py` 验证 chunk_id 现 binding sites 是否全 align 新 schema
- **#35**: pre-implementation 必跑 `grep -rn "_lineage\|set_lineage\|sets_for" aperag/indexing/graph_storage/` 列 lineage SET binding sites + alembic head verify
- **#37**: pre-implementation 必跑 `grep -rn "embed_image\|get_or_compute" aperag/llm/embed/ aperag/cache/` 验证 cache infra (`NAMESPACE_EMBEDDING` / `get_or_compute`) 真实存在 (PR #1734 merged ✅) + `embed_image` API 真实存在 (Wave 5 P2 chunk 1 merged ✅)
- **#38**: pre-implementation 必跑 `grep -rn "get_application_cache\|Noop" aperag/cache/` 验证 Noop fallback site + loop-switch trigger condition

**Pattern 3 (state binding — Protocol method reads from state X)** 强制触发条件 (added 2026-04-27 per Wave 6 #33 chunk 2 architect ruling msg=54eac595 lesson):

- 任何 spec acceptance lock 形式 "Protocol method that reads from state X (column / table / index)" — **架构师 必先 grep-verify state X 现 schema 中是否存在/可加**。否则 implementer 实施时 surface state-binding gap，触发 architect ruling round-trip + chunk amend
- **#33 chunk 1 retrospective miss**: §K.11.11 chunk 1 sub-spec 锁 `query_entities_by_vector` Protocol method 时未 grep-verify entity vector storage state binding — Postgres `aperag_lineage_entity` 没 vector column (Wave 4 lineage schema 是 write-side only)。chunk 2 实施时 surface gap → architect Option A ruling msg=54eac595 → narrowed scope (delete `query_entities_by_vector` Protocol method + stub + 2 tests, defer vector recall to Wave 7+ if real evidence surfaces)。Lesson 沉淀到 `feedback_spec_lock_grep_verify_caller.md` Pattern 3
- **forward**: 未来 Wave/任 architect spec lock for "Protocol method reads state X" 必跑 `grep -rn "state_x_column\|state_x_index" alembic/migrations/ aperag/db/models/` 验证 schema 存在；如不存在 → 4 sub-options (in-chunk schema add / prior chunk add / separate service taking state dep / defer to next Wave)

未通过 pre-check 不允许 push — architect spec lock 时 implementer push 前自跑此 grep + 验证 caller graph / state binding / Protocol-method-state-binding 与 spec 一致。

#### K.11.6. simple-stable 4 guardrail lock (per `feedback_simple_stable_zero_maintenance.md`)

每个 Wave 6 acceptance item / chunk 必通过：

| Guardrail | Wave 6 application |
|---|---|
| **不无限扩范围** | 7 items locked 来源 = task board #33-#39 (per architect msg=c8cdb40d audit-filter conclusion 22 audit findings → 0 Wave 7 expansion)。Wave 6 内不增加 academic / "good-to-have" items；如 implementer 实施时 surface 新 obs，记 follow-up doc 段落但不 fold 进 chunk |
| **先把功能做实，尽快上线** | PR-B (6 polish items) 1 周内 ship (per `feedback_announce_equals_landed.md` PR-as-truth)；PR-A (#33) 1-2 周 ship 但不阻 PR-B；single-PR within phase, multi-implementer parallel, chunked-rotation commits 内 push 节奏 |
| **简单稳定优于复杂** | Wave 6 拒绝 distributed pattern (DLQ / pub-sub invalidation) / cross-region replication / quality eval framework / drift detection 类 over-engineering；KISS — cache wiring = 现有 pattern mirror, schema rewrite = alembic mechanical migration, query layer = LightRAG-style 已验证 patten |
| **私有化部署免维护** | 拒绝 operator CLI tooling / manual recovery script / runbook-required ops；#33 query layer 自带 vector-recall 自动 fallback；#37 cache 自带 sha256 file_hash 自动 dedupe；#38 cross-loop rebuild 自动 rebuild on loop switch (no manual reset); #35 alembic upgrade 自动 (operator deploy 即激活 perf rewrite) |

#### K.11.7. Layer 1 same-session continuation directive (per `feedback_no_refresh_complete_all_tasks.md`)

Wave 6 implementation 同 Wave 4-5 directive (earayu2 msg=24797e27)：
- earayu2 不再帮 fresh-restart agent session
- 各 implementer same-session continuation push 节奏 maintained
- Layer 1 metric goal: ≥ 90% same-session ratio (Wave 5 verified 13/13 = 100%)
- chunked-rotation within PR-B (multiple commits in same branch / same session per implementer)
- 跨 implementer 之间 PR-B branch share — 任意 implementer push 完后下个 implementer rebase 接

#### K.11.8. Architect direct ratify lane

Wave 6 architect direct ratify scope (per Wave 5 §K.9.2 lane lock pattern + simple-stable directive #4 "私有化部署免维护")：

- **#33 Phase A 全程**: query layer Protocol 设计 + cascade migration completeness verify + behavior preservation validation (existing retrieval / curation behaviors within ε tolerance) — 因 cross-cutting refactor 风险高
- **#34**: spec amendment text ↔ code state binding align verify (pre-check Pattern 2)
- **#35**: alembic migration cross-backend correctness + Wave 4 T5 §H.5.1 state binding pattern lesson 类 verify
- **#36**: §D.3 Protocol surface rename completeness (`EntityRecord.type` / `RelationRecord.type` 全 caller migrate) — 因 Protocol-surface change 是 Wave 1 lesson #5 类 risk 项
- **#37**: cache key shape ↔ cache README invariant align (raw bytes 不进 key) + sha256 file_hash 真实 dedupe verify
- **#38**: cross-loop lazy-rebuild correctness + silent-zero-cache anti-pattern 不再触发 verify
- **#39**: provider format variation 走 huangheng pass-1 lane only — architect spot-check on architectural soundness 即可 (small bounded items no architect direct ratify gate)

如出现 spec / state-binding drift 或 chunk-内 scope creep → architect direct ratify trigger (per Wave 4 T5 §H.5.1 fix-forward 教训 + chunk 4d Option C narrowed precedent msg=b26f64b2)。

#### K.11.9. Hard-cut policy lock (per earayu2 msg=30c81478)

> "数据库无数据，系统未上线，所有改动可以 hard-cut"

Wave 6 acceptance items 中所有 schema / state binding / Protocol surface change 均走 hard-cut path：

- #33 graphindex 删除 → 不留 deprecation shim window (Wave 5 Phase 1 已 keep shims as transitional, Wave 6 hard-cut 删 shim 并删 legacy package)
- #35 Postgres JSONB → text[] migration → 直接 alembic schema change，不留 migration backfill (无生产数据)
- #36 Postgres column rename → 直接 alembic rename，不留 dual-column transitional
- #34 chunk_id schema unify → 直接 hard-cut 老 schema 写入 path

无 forward-compat backfill 设计，无 migration runbook，operator 不需要任何手工步骤 (`feedback_simple_stable_zero_maintenance.md` directive #4 align)。

#### K.11.10. PR-B merge gate (per `feedback_announce_equals_landed.md` 硬证据 ack)

PR-B "Wave 6 polish batch" merge ack 必须含 (与 Wave 5 PR #1733 merge pattern 一致)：

- GitHub URL + head SHA + state=MERGED
- merge commit SHA on main
- task board 全 6 items moved to `done` (任何还在 in_progress / in_review 的 item 都 block PR merge)
- Layer 1 metric tally: pass-1 一次过率 vs total chunks
- CI 4/4 lanes green (lint-and-unit + graph-compat 3-backend + vector-compat 2-backend)

**not-acceptable signals**: narrative-only "🟢 done" without GitHub URL + SHA + state；跨 implementer 之间口头 handoff without push event evidence；implementer 报 "all tests pass" but PR CI 未跑。

#### K.11.11. PR-A (#33) sub-spec — 3-chunk decomposition (per Bryce context-build report msg=5a82e0d1)

PR-A "Wave 6 graphindex" 是 cross-cutting refactor，独立 architect direct ratify lane。Bryce caller-cascade pre-check (per §K.11.5 Pattern 1) 已 surface 实际 cascade scope:

**Caller cascade scope (5 production caller files)**:
1. `aperag/domains/retrieval/pipeline.py:85` — `make_service_for_collection` → `svc.query_context(collection_id, query, top_k)` returns `ctx.text` (LightRAG-style query→entities→context-text)
2. `aperag/domains/knowledge_graph/service.py:69, 94, 151` — graph CRUD service 3 callsites
3. `aperag/graph_curation/service.py:36-37, 199` — Entity DTO + GraphIndexService + LLMCall + make_service_for_collection
4. `aperag/graph_curation/integration.py:21` — make_service_for_collection
5. `aperag/graph_curation/candidate_generation.py:24` — Entity DTO

**Architectural gap**: new `LineageGraphStore` (Wave 4) is **write-side only** (`find_*_with_lineage` by-document-id + `get_*` single-fetch + `upsert_*` / `remove_*` / `gc_*`)。Legacy `query_context(query, top_k)` 是 by-vector + by-keyword + traversal — **没有 1-to-1 mapping**，需要全新设计 query layer。

**3-chunk decomposition**:

| Chunk | Scope | Owner | Estimated LOC | Status |
|---|---|---|---|---|
| **chunk 1** | LightRAG-style read Protocol stubs on `LineageGraphStore` — NO implementation, NO caller migration; spec ↔ Protocol surface align verify only | Bryce | ~200 LOC docs+protocol | ✅ MERGED PR #1737 commit `20b9071b` 2026-04-27 |
| **chunk 2** | Protocol implementations (`query_entities_by_keyword` + `expand_neighbors_n_hops`) across Postgres / Neo4j / Nebula backends + unit tests + chunk 1 amend (drop `query_entities_by_vector` per architect Option A ruling msg=54eac595) | Bryce | ~600 LOC + tests | 🔄 PR #1741 in_review |
| **chunk 3** (narrowed per Option C ruling msg=6fccd9ab) | retrieval/pipeline.py cutover only (`query_context()` → `query_entities_by_keyword` + `expand_neighbors_n_hops` + `_render_graph_context_text`) + retrieval-side legacy reference removal + retrieval/pipeline.py grep-zero verify。**DO NOT delete** legacy graphindex package + tests (UI/curation 4 files 仍 reference) | Bryce | ~50 LOC + 9 tests | ✅ PR #1742 in_review |

**Chunk 1 → chunk 2 amend note (architect ruling msg=54eac595 — Option A narrowed scope)**:
- chunk 1 originally declared 3 Protocol methods (keyword + vector + traversal). chunk 2 backend impl surface state-binding gap: lineage schema 没 entity vector column (Wave 4 design choice, write-side only)
- Architect ruling Option A (msg=54eac595): chunk 2 amends chunk 1 — drop `query_entities_by_vector` Protocol method declaration + InMemoryLineageGraphStore stub + 2 contract tests. Vector recall deferred to Wave 7+ if real evidence。
- Lesson: chunk 1 spec lock 时 architect 应先 grep-verify entity vector storage schema → Pattern 3 (Protocol method state binding) added to §K.11.5 + sediment 到 `feedback_spec_lock_grep_verify_caller.md`

**Chunk 2 → chunk 3 amend note #2 (architect ruling msg=6fccd9ab — Option C narrowed scope, 2026-04-27 second amend within #33)**:
- chunk 2 implementation 完 + chunk 3 caller-cascade pre-check 时 Bryce surface 第二 architectural gap: legacy `GraphIndexService` 4 methods 中只 `query_context()` 在 new `LineageGraphStore` Protocol covered；UI/curation methods (`get_labels` / `get_knowledge_graph` / `merge_entities`) 不在 surface — 4 caller files (knowledge_graph/service.py + graph_curation/*) 不能 migrate without new Graph CRUD API design。
- Architect ruling Option C (msg=6fccd9ab): chunk 3 narrowed to retrieval/pipeline.py cutover only。Legacy graphindex package retained for UI/curation。Wave 6 close-out 不再 include "delete legacy graphindex" — defer to evidence-based future Wave。
- Lesson #2: chunk 1 spec lock 时 architect 应 enumerate per-caller-per-method matrix (Pattern 1 v2 method-level coverage)，不仅 import count。Pattern 1 强化 added to §K.11.5 + sediment 到 `feedback_spec_lock_grep_verify_caller.md`。
- 双 architect-side own-ups 在同一 #33 task — chunk 1 spec lock 颗粒度 spec amend 模式 ratify trail：chunks 2+3 各自 amend chunk 1 acceptance text + memory feedback Pattern 沉淀。

**Each chunk lands independently within PR-A** (chunked-rotation per `feedback_no_refresh_complete_all_tasks.md` Layer 1 same-session continuation directive)。Each chunk has own architect direct ratify gate per §K.11.8 lane lock。

**Why 3 chunks (not 1 monolith)**: (a) chunk 1 spec/Protocol-only allows architect ratify to lock query layer surface before backend implementations; (b) chunk 2 backend impl can independently verify Protocol semantics on real engines (per `feedback_dataflow_review.md` dataflow trace each backend); (c) chunk 3 caller migration is pure mechanical refactor once Protocol stable — chunk 4d Option C precedent (msg=b26f64b2 narrowed scope) avoids "spec drift mid-cascade" anti-pattern。

PR-A 的 acceptance items: §K.10 items 1-2 (narrowed) ✅ shipped via chunks 1-3; items 3-5 (delete legacy package + tests + grep-zero) deferred to evidence-based future Wave per Option C ruling msg=6fccd9ab。amended #33 must-be-real per §K.11.4 (graph entity vector recall NOT in scope per chunk 2 ruling msg=54eac595; UI/curation flows migration NOT in scope per chunk 3 ruling msg=6fccd9ab; retrieval-side keyword + traversal cutover ✅ shipped)。

#### K.11.12. Wave 6 close-out gate

Wave 6 close-out (revised 2026-04-27 post-Option C chunk 3 ruling msg=6fccd9ab)：
- PR-B fast-track merges ✅ → 4 polish items shipped (#34 + #37 + #38 + #39 已 MERGED via PR #1735/#1738/#1739)
- PR-A chunks 1+2+3 merges ✅ → retrieval-side LightRAG-style query layer + retrieval/pipeline.py cutover shipped (#33 narrowed scope; UI/curation migration + legacy package delete deferred to evidence-based future Wave)
- 余 #35 (Bryce, Lineage parallel-list perf rewrite) + #36 (Bryce, Cypher TYPE() / EntityRecord.type rename) — 各自 fast-track PR per Wave 6 individual fast-track pattern
- task board #33-#39 全 done (七 task close-out)
- 架构师发 Wave 6 final review (mirror Wave 5 close-out msg=43df7dd8) 含 honest declare 双 deferred items (vector recall + UI/curation migration) for future Wave evidence-based reactivation
- Wave 7+ 创建前必通过 simple-stable 4 guardrail 重审 (architect msg=c8cdb40d audit-filter 模板 reusable) + 三 pre-check pattern lock (Pattern 1 v2 method-level coverage + Pattern 2 + Pattern 3) per `feedback_spec_lock_grep_verify_caller.md`

**why 双 PR 是 simple-stable**: 拒绝 single-PR 24-day land cycle (#33 阻 polish 6 items)；6 polish items 1 周 land + #33 设计完成后再 ship — 用户视角 "尽快上线" 实质化 (PR-B 各 modality 改进可下周即用)。

### K.12. Wave 7 — Legacy graphindex 全删 + 共享 graph 层最终态（启动于 2026-04-28，per earayu2 msg=3d3b2e68 "最后一波 graph 和 index 层修改" directive）

**目标**：删除整个 `aperag/domains/knowledge_graph/graphindex/` legacy 包，把 LightRAG 风格的语义召回（向量 + 图遍历）真正接通到新的 `LineageGraphStore` 体系，提供 UI 实体合并能力 + kg.jsonl 阶段自动检测合并候选，全链路（上传/索引/MCP/Agent/UI）打通。这是 graph + index 层最后一波修改，ship 后系统达到最终态。

**Wave 7 与 Wave 6 关系**：Wave 6 §K.10/§K.11 narrowed scope 把 retrieval-side cutover ship 了，但 UI/curation flows 与 LightRAG-style vector recall 都 deferred。Wave 7 是把这些 deferred 项一次性收口 + 删除 legacy。

#### K.12.1. 现状分析（2026-04-28 grep 验证的事实）

per architect msg=ccfc6114 grep 调研 + huangheng msg=f1e695a0 grep 验证：

1. **legacy `graphindex/` 包**：
   - 三套图数据库实现（postgres/neo4j/nebula），各自在 `graphindex_nodes` + `graphindex_edges` 表（与 `aperag_lineage_*` 表完全独立，不共享 schema）
   - `GraphIndexService.query_context()`（line 760-870）实施完整 LightRAG 风格查询：嵌入 query → Qdrant 向量召回（payload filter `indexer="graph_entity"`/`"graph_relation"`）→ 图遍历 → context text 渲染
   - `_sync_entity_relation_vectors()`（line 573）实施 entity/relation embedding 写入 Qdrant
   - `_delete_removed_shadow_vectors()`（snapshot diff 算法）实施 doc lifecycle vector 清理
   - `_compact_oversized_descriptions()`（line 158）+ `_summarize_description()` + `_fallback_truncate()` 实施 description 长度治理（LLM summary + 字符截断 fallback）
   - `merge_entities()` 实施用户合并（store-side topology + LLM 生成 unified description）
   - `get_knowledge_graph()` / `list_labels()`（已 #40 narrow-replace）实施 UI methods

2. **新 `LineageGraphStore` 体系（Wave 4-6）**：
   - per-doc lineage 追踪：每个 entity/relation 的 `description_parts` 自带 `(document_id, parse_version, chunk_ids)` 标签
   - 三个新 backend（`aperag/indexing/graph_storage/{postgres,neo4j,nebula}.py`），写入 `aperag_lineage_*` 表
   - Wave 6 #33 加 keyword + 1-hop traversal + chunk_id schema
   - **没** entity vector embedding（vector 完全没在新体系里）

3. **隐藏的功能性回归**：
   - 自 Wave 4 hard-cut 后 legacy `run_index_document_sync` 0 callers → Qdrant entity index 自 Wave 4 起一直空表
   - Wave 6 chunk 3 retrieval cutover 切到新 keyword + traversal，没接通向量召回
   - 老 `query_context` 一直查空 vector index → fallback 到 keyword
   - `enable_knowledge_graph=False` 默认值掩护着这个事实
   - **LightRAG-style 向量召回从 Wave 4 起在生产环境下缺席**

4. **UI 接口实际状态**：
   - `/graphs/labels` (`get_labels`) → Wave 6 #40 已迁，读新表 ✅
   - `/graphs?label=...` (`get_knowledge_graph`) → 仍走 legacy，老表空 → UI 看到空图谱 ❌
   - `/graphs/merge` (`merge_entities`) → 仍走 legacy，老表空 → 没东西可合并 ❌

5. **ApeRAG 已有的可复用基础设施**：
   - `aperag/vectorstore/connector.py` `VectorStoreConnectorAdaptor` 已抽象 Qdrant + pgvector + 未来其他 backend
   - `VectorPoint` / `QueryRequest` / `VectorFilter` / `Eq` filter DTO vector-store-agnostic
   - `aperag/domains/agent_runtime/tools/registry.py` MCP server registry 基础设施已 ship

Wave 7 实质：**激活向量召回写入 + 替换查询服务 + 加用户合并能力 + 删除 legacy 包**。Migrate-not-invent 模式（不重新设计，参考 legacy 算法 + 新组件命名 + 复用基础设施）。

#### K.12.2. 设计哲学：四层分离

per `feedback_simple_stable_zero_maintenance.md` 4 guardrail（不无限扩范围 / 尽快上线 / 简单稳定 / 私有化部署免维护）+ earayu2 msg=cf5186e2 + msg=f92b2584 + msg=aac6053d directive：

| 层 | Source of truth | 职责 | Doc lifecycle 行为 |
|---|---|---|---|
| **图数据层** | 图数据库 (`aperag_lineage_*` 表 / Neo4j 节点边 / Nebula tag) | 结构化数据 + per-doc lineage 追踪 (Wave 4 §C.3 契约) + 新加 `compacted_description` 字段 (LLM-summarized 派生) | strip per-doc parts → empty entity gc_orphan 自动级联 |
| **向量层** | 向量数据库 (Qdrant / pgvector / 未来其他, 通过 `VectorStoreConnectorAdaptor` 调度) | 与 chunk 向量同 collection, 靠 payload `indexer="graph_entity"/"graph_relation"/"chunk"` 区分 | snapshot-diff 删 + 重 embed 自动跟随 |
| **用户意图层** | `aperag_lineage_entity_alias` 表 | 持久化用户主动合并意图, 不被 doc lifecycle 影响 (orphan 保留, lossy resurrection acceptable) | 不变 — alias_map 独立 |
| **合并候选检测层** | `aperag_merge_candidate` 表 (新加) | kg.jsonl 生成完跑 detection, 综合字符串 + vector + (可选) LLM judge → 写候选给 UI | 候选过期/过时由 UI 端 dismiss |

**关键 invariant**：图数据层永远是 source of truth + 向量层和候选层是派生数据可重算 + 用户意图层独立持久化。文档 add/update/delete 时图数据层按 §C.3 自动级联 → 向量层通过 snapshot-diff 自动跟随 → 候选层重新检测。

#### K.12.3. 核心组件（去 LightRAG 命名）

- **`GraphSearchService`**（替代老 `GraphIndexService.query_context`）
  - 聚合 `LineageGraphStore` + `VectorStoreConnectorAdaptor` + `Embedder` + `LLMCall`
  - 接口：`search_entities(query, top_k)` / `search_relations(query, top_k)` / `get_subgraph(label, max_depth, max_nodes)` / `compose_context(query, top_k)`
  - 算法参考老 `query_context` line 760-870：embed query → Qdrant 召回 entity（payload filter `Eq("indexer", "graph_entity")`）→ Qdrant 召回 relation → 拿 entity_names 调 `LineageGraphStore.expand_neighbors_n_hops` → 组装 context

- **`GraphCurationService`**（替代老 `merge_entities`）
  - 聚合 `LineageGraphStore` + `VectorStoreConnectorAdaptor` + `Embedder` + `LLMCall`
  - 接口：`merge_entities(target, sources)` / `find_merge_candidates(entity_name, top_k)`（thin wrapper 走 store keyword search）
  - merge 流程（4 步幂等）：
    1. 写 alias_map（A→C, B→C），cycle flatten 1-hop redirect
    2. UNION：`C.description_parts = A.parts ∪ B.parts`，`C.source_lineage = A.lineage ∪ B.lineage`
    3. 调 LLM 生成 unified description → 触发 `GraphIndexCompactor` 长度治理 → 写 `compacted_description`
    4. embed `compacted_description` → upsert vector point + 删 A、B 的 vector point + 删 A、B lineage rows

- **`GraphIndexCompactor`**（新组件，移植老 `_compact_oversized_descriptions` 算法 + 去 LightRAG 命名）
  - 在 `GraphModalityWorker.sync()` 末尾跑（也被 `GraphCurationService.merge_entities` 调用）
  - 触发条件：`len(description_parts) >= summarize_at_fragments (默认 8)` 或 `total chars >= max_description_chars (默认 8000)`
  - LLM summary：调用 `render_summarization_prompt`（参考老 prompt 模板），target_chars ~3000
  - LLM 失败 fallback：字符截断 + 末尾 ` … [truncated]` 标记
  - 写到 `compacted_description` 字段（不污染 `description_parts` per-doc tracking）

- **`MergeCandidateDetector`**（新组件，老代码无对应）
  - 在 `GraphModalityWorker.sync()` 末尾跑（kg.jsonl 生成完）
  - 算法（综合三种信号）：
    1. 字符串相似度：lower-case + token-set Jaccard >= 0.6 双方都触发
    2. Vector 相似度：embed entity name + compacted_description → ANN top-5 → cosine >= 0.85
    3. 取并集，按 `max(string_score, vector_score)` 降序
    4. （可选 tier）LLM judge：高 confidence (vector >= 0.95) 才触发 LLM 二次确认；默认 off
  - 写候选到 `aperag_merge_candidate` 表 `(collection_id, entity_a, entity_b, similarity_score, source: 'string'|'vector'|'llm', detected_at)`
  - **不自动合并** — UI curator 主动决策

- **`GraphModalityWorker`**（扩展 Wave 4 现有组件）
  - `sync()` 末尾增加四步：
    1. 跑 `GraphIndexCompactor` → 长度超阈值的 entity 写 `compacted_description`
    2. 嵌入 entity / relation description（用 `compacted_description` 优先, 否则 `name + " " + concat(description_parts.text)`）→ 通过 `VectorStoreConnectorAdaptor.upsert` 写 Qdrant point
    3. snapshot-diff 删除 gc 的 entity/relation 对应 vector point（参考老 `_delete_removed_shadow_vectors`）
    4. 跑 `MergeCandidateDetector` → 写候选

#### K.12.4. 数据 schema 改动

**新加字段**（item 1）：

- `aperag_lineage_entity` 加 `compacted_description: TEXT NULL`
- `aperag_lineage_relation` 加 `compacted_description: TEXT NULL`
- 三个 backend（postgres/neo4j/nebula）各自的存储支持
- **DTO 落点**（lock 2026-04-28，per huangheng msg=4d93a6c5 + Bryce pre-check msg=2dbd5a6b + architect ratify msg=6926f1ff）:
  - 字段加在 `EntityWithLineage` / `RelationWithLineage`（storage view DTO），**不**改 `EntityRecord` / `RelationRecord`（kg.jsonl raw 抽取契约）
  - 理由: `compacted_description` 是 sync 末尾 GraphIndexCompactor 派生写入，属于 storage 层；kg.jsonl 是 graph_extractor raw 抽取输出，不应被派生数据污染（architecture invariant: L1 storage layer derived field, NOT L0 extraction layer）
- **写入路径**（Option B locked）: 走现有 `upsert_entity_with_lineage(..., compacted_description: str | None = None)` / `upsert_relation_with_lineage(...)` 加 nullable kwarg；**不**新增 Protocol method
  - 理由: 不扩 Protocol 表面（simple-stable directive #3）+ atomic single-write（forward-only retry safety）+ Wave 6 chunk 2 spec-impl gap 教训（不必要的 Protocol method 是 spec drift 高发区）
  - kwarg sentinel 语义: 不传 = 保留 existing；传 None = 显式清空；传 str = 写入；implementer 自选 sentinel 实现细节，但行为契约固定
- **safety gate test**（per huangheng msg=828c83cc，3 backend 各 1 case）: `test_compactor_write_preserves_all_lineage_fields` — 验证 compactor read-modify-write 不丢 `source_lineage` / `description_parts`，是 Option B 的 invariant 守门员

**新加表**（item 6 only）：

- `aperag_lineage_entity_alias`：`(collection_id, alias_name, canonical_name, merged_at, merged_by)` PK `(collection_id, alias_name)` + index `(collection_id, canonical_name)`（反向查询用）

**Item 4 候选检测 — 不新加表**（lock 2026-04-28，per cuiwenbo msg=0414287c + chenyexuan msg=0243e9bf grep + architect ratify msg=2be943f6 + 明书 grep msg=e5c35c05）:

- 写入现有 `GraphCurationSuggestion` 表（`aperag/domains/knowledge_graph/db/models.py:106`）
- 每次 sync 创建一个 `GraphCurationRun` 行：`status=COMPLETED`, `config_json={"source": "auto_detect", "sync_run_id": "<sync uuid>"}`, `user_id=collection.owner`
- 写 `GraphCurationSuggestion` 行 mapping:
  - `entity_ids = [entity_a, entity_b]` (canonical pair `(min, max)` alphabetical-first 强制)
  - `target_entity_id = min(entity_a, entity_b)` (deterministic alphabetical-first; user 在 accept 时仍可通过 `target_entity_data` override)
  - `confidence_score = composite(jaccard, vector_score)` (沿用现有 `_pair_score`)
  - `reason = "auto_detected"` (区分 user-driven 的 `reason` 文本)
  - `evidence = {"jaccard": ..., "vector_score": ..., "llm_judge_verdict": ...?}`
  - `status = "PENDING"`
- supersede 语义: 下次 sync detector 跑前必显式 supersede 前一次 auto-source 的 PENDING → SUPERSEDED (不留双 PENDING orphan)
- 0 alembic migration (复用现有表)

理由 (per `feedback_simple_stable_zero_maintenance.md` 4 guardrail 全 hit):
- 现有 `GraphCurationSuggestion` schema 已成熟 (batch / status / accept-reject / target_data override / `PENDING|ACCEPTED|REJECTED|EXPIRED|SUPERSEDED` lifecycle)
- 现有 `aperag/graph_curation/candidate_generation.py:build_candidate_pairs` (Jaccard + vector 复合 scoring) + `service.py:_adjudicate_pairs` (LLM judge) infra 已 production-validated
- frontend 0 改 (cuiwenbo task #9 工作量从 1 天降到 ~1h)
- task #4 实施降级到薄 wrapper, ETA 从 1-2 周到 1 天

**Detector 仍叫 `MergeCandidateDetector`** (新代码命名一致, "Candidate detection" = algorithmic 阶段 vs "Suggestion" = user-facing 持久化层, 两层概念清晰)。

**alembic migration**：item 1 + item 6 各 1 个 migration（链式; item 4 = 0 migration）。

**老表 drop**（item 10 last）：alembic drop `graphindex_nodes` + `graphindex_edges`（hard-cut，无生产数据）。

#### K.12.5. Application-layer 长度治理（不是 DB schema CHECK）

per earayu2 msg=421c4223 "不能很随意写代码，导致超列长度报错" + huangheng msg=11e95fb2 wording fix：

**关键澄清**: 下表所列限制是 **application-layer caps**（Compactor + graph_extractor + Curation Service 在写入前 enforce），**不是** DB schema CHECK constraint。Postgres TEXT / Neo4j string property / Nebula tag-prop string 在 schema 层都不写长度约束；application 层 truncate-not-fail 保证写入永远 ≤ 限制值。

| 限制项 | 默认值 | enforcement 位置 | 理由 |
|---|---|---|---|
| per-part 最大字符数 | 5000 chars | `graph_extractor` 输出阶段 truncate (kg.jsonl 写入前) | Nebula tag-prop string 兼容 boundary |
| per-entity 最大 parts 数 | 100 parts | `graph_extractor` 已 tunable knob (Wave 6 P5A) | 累积 description 总量上限保护 |
| `compacted_description` 最大字符数 | 8000 chars | `GraphIndexCompactor.compact_if_oversized` LLM target_chars 3000 + fallback truncate cap 8000 | embedder input safe |
| 全部超限处理 | truncate + ` … [truncated]` 标记 + log warn | 各 enforcement layer 内部 | 不让 production 因 oversize 报错 (per `feedback_simple_stable_zero_maintenance.md` directive #3) |

#### K.12.6. MCP 接口

注册到 `aperag/domains/agent_runtime/tools/registry.py`（item 7）：

- `query_graph_entities(query: str, top_k: int) -> list[Entity]` — 走 `GraphSearchService.search_entities`
- `expand_graph_subgraph(entity_names: list[str], hops: int) -> dict` — 走 `LineageGraphStore.expand_neighbors_n_hops`
- `get_entity_detail(name: str) -> Entity` — 走 `LineageGraphStore.get_entity` + 必要时 `get_unified_description_for_ui`

Agent 通过这三个 tool 实现 LightRAG 风格 RAG over graph：semantic search → traversal → detail lookup。

#### K.12.7. 4 项关键决策（locked）

| 决策 | 选项 | 选择 | 理由 |
|---|---|---|---|
| 索引时是否自动合并相似实体 | 自动合并 / 仅检测候选 / 完全不动 | **仅检测候选 (β option / Option K2)** | 自动合并 LLM 误判不可逆；检测候选给 UI 让用户决策 (per simple-stable directive #1 + #4) |
| alias_map 在 canonical entity gc 时如何处理 | 保留 orphan / 级联删除 | **保留 orphan (option X)** | 用户合并意图比 doc lifecycle 优先级高；alias 表行数极小；同名 doc 重传可自动恢复 |
| 合并时是否调 LLM | 调 / 不调 | **调 LLM 生成 unified description + 触发 compaction (option α)** | 这是 vector embedding 质量来源；description 长度治理紧跟其后；与 LightRAG eager merge pattern 一致 |
| Vector store 选型 | Qdrant only / 用 ApeRAG 现有 abstraction | **复用 `VectorStoreConnectorAdaptor`** | per earayu2 msg=aac6053d "图 DB 不装 vector 插件"；已有抽象层覆盖 Qdrant + pgvector + 未来 backend；操作员部署一份 vector store 即可 |

#### K.12.8. Wave 7 acceptance items（10 项）

| # | 内容 | 负责人 | ETA |
|---|---|---|---|
| 1 | `LineageGraphStore` schema 加 `compacted_description: TEXT NULL` 字段 + alembic migration（Postgres）+ 三个 backend 各自存储支持 | Bryce | 1 天 |
| 2 | `GraphIndexCompactor` 组件：移植老 `_compact_oversized_descriptions` + `_summarize_description` + `_fallback_truncate` + `_should_summarize` 算法到新代码树（去 LightRAG 命名）+ unit tests | 明书 | 1 天 |
| 3 | `GraphModalityWorker.sync()` 扩展末尾四步（compactor → embed → vector upsert via `VectorStoreConnectorAdaptor` + snapshot-diff delete → 跑 detector）+ integration tests | Bryce（依赖 #1）| 2 天 |
| 4 | `MergeCandidateDetector` 薄 wrapper：复用现有 `build_candidate_pairs` (Jaccard + vector 复合) + 可选 LLM judge → 写现有 `GraphCurationSuggestion` 表（per-sync `GraphCurationRun` source=auto_detect）+ supersede 前一次 auto-PENDING + unit tests。**0 alembic migration** | 明书 | 1 天 |
| 5 | `GraphSearchService` 实现：聚合 + `search_entities` + `search_relations` + `get_subgraph` + `compose_context`，参考老 `query_context` line 760-870 算法 + unit tests | 明书 + architect | 2-3 天 |
| 6 | `GraphCurationService` 实现：alias_map 表（含 alembic + cycle flatten + per-collection cache）+ `merge_entities`（UNION + LLM unified + compaction + vector re-embed + remove sources）+ `find_merge_candidates`（thin wrapper）+ chunk 2 critical inseparability：`upsert_entity` 内部透明 alias redirect + integration tests | 明书 + Bryce 协助 graph backend（依赖 #1）| 2-3 天 |
| 7 | MCP 接口注册：`query_graph_entities` / `expand_graph_subgraph` / `get_entity_detail` 进 `agent_runtime/tools/registry.py` + integration tests | 明书 或 chenyexuan（依赖 #5）| 1 天 |
| 8 | `retrieval/pipeline.py:_graph_search` 升级走 `GraphSearchService`（恢复向量召回，不是 keyword-only）+ `aperag/graph_curation/*` 4 文件迁移（用 alias 替代老 Entity DTO）+ `domains/knowledge_graph/service.py` `merge_entities`+`get_knowledge_graph` internal cutover + 现有 REST routes (`POST/GET /graphs/merge-suggestions`、`POST /graphs/nodes/merge`、`GET /graphs?label=...`) 内部 cutover。**0 new endpoint** (复用 `MergeSuggestion*` REST per cuiwenbo lock) | chenyexuan（依赖 #5 + #6）| 1-2 天 |
| 9 | 前端验证 + OpenAPI 重新生成（response shape 不变则 0 工作量；否则 UI 微调显示 `compacted_description` / 候选合并 UI）| cuiwenbo（如需）| 0-1 天 |
| 10 | 删除整个 `aperag/domains/knowledge_graph/graphindex/` 包 + `tests/unit_test/graphindex/` 整目录 + `tests/integration/compat/test_graph_compat.py`（已被 `test_lineage_graph_compat.py` 替代）+ alembic drop `graphindex_nodes` + `graphindex_edges` 表 + grep-zero verify。**这步必须最后做** | Bryce（依赖前 9 项全 merged）| 0.5 天 |

**Total**：1.5-2 周（多人并行：item 1+2+4+5 可同时启动；item 3 + 6 依赖 item 1；item 7 + 8 依赖 item 5；item 9 依赖 item 8；item 10 必 last）。

#### K.12.9. Pre-check pattern lock（per `feedback_spec_lock_grep_verify_caller.md` + `feedback_architect_design_impl_gap_monitor.md`）

**强制要求**（per `feedback_cr_spec_crosscheck.md` hard gate）: 每个 PR description body 必 paste 4-pattern + 5 mini-pattern grep 输出 + 12 invariant cross-check 表 + simple-stable 4-guardrail 落点。CR 拒绝 LGTM 缺这三段的 PR。

每个 acceptance item 实施前必跑（implementer 自我审计 + post pre-check matrix gap analysis 给 architect 验证）：

- **Pattern 1 v1 + v2**: 
  - `grep -rn "from aperag.domains.knowledge_graph.graphindex" aperag/ tests/` — caller import-level
  - FOR EACH caller, enumerate `GraphIndexService.METHOD(` calls — method coverage matrix
- **Pattern 2** (state binding): 
  - `grep -rn "graphindex_nodes\|graphindex_edges" aperag/` — legacy table refs
  - `grep -rn "compacted_description\|aperag_lineage_entity_alias" aperag/migration/` — 新 schema binding
- **Pattern 3** (Protocol method state): 
  - `grep -rn "indexer.*graph_entity\|indexer.*graph_relation" aperag/vectorstore/` — vector payload filter binding
  - 验证 `description_parts.text` 列累积长度 vs `max_description_chars` 阈值
- **Mini #4** (Protocol DTO 名 grep): `grep -rn "class EntityRecord\|class EntityWithLineage" aperag/indexing/graph.py` — 派生字段必须落在 storage view DTO, 不污染 raw 抽取契约
- **Mini #5** (dependency 接口签名): `grep -rn "def search\|def query" aperag/vectorstore/base.py aperag/vectorstore/connector.py` — sync vs async, method 名 verify
- **Mini #6** (binding pattern conform): `grep -rn "def .*\\(self.*collection_id" aperag/indexing/graph.py` — 现有 Protocol 全 per-collection 绑定, 新 method 不引入 `collection_id` 参数
- **Mini #7** (新加 X 前 grep 现有 X-similar): `grep -rn "class.*Suggestion\|class.*Candidate\|build_.*pairs\|adjudicate" aperag/ tests/` — reuse > new default
- **Mini #8** (port legacy 必列 invariant feature): spec 写"port legacy 算法 X" 时必显式列 X 的 invariant feature list (例: DESCRIPTION_SUMMARIZATION 5 features = subject_label / language / relation framing / contradiction handling / length flexibility)

#### K.12.10. 命名约定（去 LightRAG）

| 老命名 | 新命名 |
|---|---|
| `LightRAGQueryService` (architect 早期 draft) | **`GraphSearchService`** |
| `LightRAGCurationService` (architect 早期 draft) | **`GraphCurationService`** |
| LightRAG-style query | "图谱搜索" / "vector + traversal 召回" |
| 注释里 LightRAG 引用 | 移除 |
| 老 `legacy graphindex` 类名 | 整个包删掉 |

代码 + REST API + MCP tool name 中完全去掉 LightRAG。仅本节"现状分析"提及"灵感来自 LightRAG 设计"。

#### K.12.10b. Cycle flatten 算法（alias_map redirect 1-hop guarantee）

每次 user merge `A → B` 时, 防 cycle 必跑 flatten:

```
def upsert_alias(collection_id, alias_name=A, target=B):
    # 1. resolve target's canonical (1-hop guarantee = 永远不嵌套)
    canonical = resolve_canonical(collection_id, target)  # 若 B 已 alias_of(C), canonical=C
    
    # 2. write (A, canonical) — alias_map 永远存终点, 不存中间链
    upsert_row(collection_id, alias=A, canonical=canonical, merged_at=now, merged_by=user)
    
    # 3. invalidate per-collection cache
    cache.invalidate(collection_id)


def resolve_canonical(collection_id, name) -> str:
    """ 永远 1-hop 查询. 若 alias_map 中存 (name, X), 返回 X; 否则返回 name 自身 """
    row = alias_map.get(collection_id, name)
    return row.canonical if row else name
```

**Invariant**: alias_map 中 `(alias_name, canonical_name)` 行的 `canonical_name` **永远是终点** (即不存在另一行 `(canonical_name, X)` where X != canonical_name)。upsert 时 flatten target chain 保证此 invariant。

**Edge case** (用户先 merge A→B, 再 merge B→C):
- step 1: alias_map 写 `(A, B)`
- step 2: merge B→C 触发 `upsert_alias(B → C)` + 必须扫描所有 alias_map 行 where canonical=B → 改成 canonical=C (flatten 已存在的 A→B 链 → A→C)
- 实施: `UPDATE alias_map SET canonical_name=C WHERE collection_id=cid AND canonical_name=B; INSERT (B, C);`

**为何 1-hop 而不是 transitive lookup**: 简单稳定 (per directive #3) + lookup O(1) per-collection cache + 不需递归 cycle detection (写时已防)。

**实施测试 case**:
- merge A→B 后查 A → B
- merge B→C 后查 A → C (transitive flatten)
- merge C→A (循环) → 拒绝 (cycle detection)

#### K.12.11. Architect direct ratify lane（per §K.9.2 / §K.11.8 lane lock pattern）

Wave 7 architect direct ratify scope：
- **Item 6**（GraphCurationService chunk 2 inseparability）：alias_map 表 + indexer integration + merge atomic 4-step orchestration —— 因 critical-path 且涉及 user intent 持久化
- **Item 5**（GraphSearchService）：聚合 4 个组件，是 retrieval / MCP / UI 共同入口 —— 因 system-wide query path
- **Item 10**（删 legacy 包）：grep-zero verify + alembic drop 老表 —— 因 hard-cut 不可逆
- 其它 item 走 huangheng pass-1 lane only（小 bounded items），但 huangheng 主动 CR 全部 PR

如出现 spec / state-binding drift 或 chunk-内 scope creep → architect direct ratify trigger（per Wave 6 #33 chunks 2/3 经验）。

#### K.12.12. Wave 7 close-out gate

Wave 7 close-out 必须满足：
- 全 10 acceptance items 全 merged（item 10 是 last）
- 全 PR CI 全绿
- task board 全 done
- grep-zero verify: `grep -rn "from aperag.domains.knowledge_graph.graphindex" aperag/ tests/` → 0 matches in production code
- alembic head 含 `graphindex_*` 老表 drop migration
- 架构师发 Wave 7 final review（mirror Wave 5/6 close-out msg pattern）含 "graph 层最终态" declaration
- earayu2 不再介入（per msg=3d3b2e68 directive），团队自驱完成

**Wave 7 完成态**：
- ✅ legacy `aperag/domains/knowledge_graph/graphindex/` 包 0 行代码
- ✅ alembic drop `graphindex_*` 老表
- ✅ 全 graph 功能（read + curation + visualization + indexing）走新 Protocol + Service
- ✅ LightRAG 风格语义召回真实接通生产环境
- ✅ UI 实体合并 + 候选自动检测体验完整
- ✅ Doc lifecycle add/update/delete 自动级联（per Wave 4 §C.3 + snapshot-diff vector cleanup）
- ✅ 私有化部署免维护（vector store 抽象 + 不依赖图 DB vector 插件）
- ✅ MCP 接口完整，agent 直接 LightRAG 风格查询
- ✅ Frontend backward-compat

---

## §L. Private / on-premise deployment — "deploy-and-forget"

earayu2（`msg=cc0a00d7`）: "我的系统是要私有化交付和部署的，我希望能做到交付后不管。"

v2 把"私有化交付 + 弱运维 + 客户拿到包就能跑、跑起来不用回头维护"作为**首要 deployment target**。架构必须直接服务这个目标。

### L.1. 部署形态分级

| 形态 | 流量假设 | Stack | 备注 |
|---|---|---|---|
| **Tier 1 — Single binary / inline** | < 10 docs/hour | SQLite + LocalFS + 单进程 + `INDEXING_MODE=inline`（§E.5） | 客户单机 demo / POC；无 Redis、无 PostgreSQL、无独立 worker |
| **Tier 2 — Single VM / async** | < 100 concurrent docs | PostgreSQL + LocalFS or MinIO + Redis + 5 worker processes（§E.2） | 客户单 VM 标准部署；docker-compose 一键拉起 |
| **Tier 3 — Multi VM / scale-out** | > 100 concurrent docs | PostgreSQL + S3-compatible + Redis + horizontal worker scaling | 客户跨机部署；架构无变化，扩 worker 进程数 |

**所有三层共用同一份代码**，差异仅在配置。私有化交付时根据客户规模选 Tier，不存在"小客户用一套代码、大客户用另一套"。

### L.2. 必须做到的属性

- **零云依赖**：不依赖 AWS / Aliyun / GCP 任何 managed service。MinIO 取代 S3，PostgreSQL 自带，Redis 自带，LLM 走客户的 endpoint（OpenAI 兼容协议）。
- **打包即可运行**：`docker-compose up` 启全栈；Tier 1 一行 `python -m aperag.cli serve` 即可起。
- **弱运维**：没有运维介入也不会"越跑越坏"——
  - cleanup worker 自动 GC 老 parse_version（§F.5）
  - reconciler 自动 retry 失败 + 自动 reclaim crashed worker（§I.3）
  - 配额 token bucket 自动 refill（§H.5）
  - 无需 cron 配置，无需手工清表，无需手工 reindex
- **可观测性自带**：4 个 SLI emit 到 OTLP，客户可挂自己的 collector；不挂也不影响系统运行。
- **兼容客户已有 LLM gateway**：所有 LLM / embedding 调用走配置文件指定的 endpoint，不硬编码任何云厂商。

### L.3. Deploy-and-forget 的具体落点

每个会随时间败坏的资源，在架构里都有一个自愈机制：

| 资源 | 不做兜底会怎样 | v2 自愈 |
|---|---|---|
| 旧 parse_version 在对象存储和 DB 里堆积 | 磁盘 OOM | cleanup worker 5min cycle，无需运维（§F.5） |
| Worker 进程崩溃 | 任务卡 RUNNING | reconciler 60s 后 reclaim → retry（§E.4） |
| LLM API rate-limit 超限 | 重试风暴 | Redis token bucket，超限自动 wait（§H.5） |
| 失败任务永不重试 | 卡 FAILED | reconciler exponential backoff retry（§I.2） |
| derived/ 半写文件 | 下次读到坏数据 | tmp+rename / multipart upload + complete（§C.7） |
| 已删文档残留索引 | backend 越堆越多 | cleanup worker 检测 `deleted_at` GC（§F.5） |
| 配额配置 drift | 部分租户被卡死 | 兜底走 default 池（§H.5） |

**没有需要"运维定期处理"的资源**。这是"deploy-and-forget"的硬要求。

### L.4. 单机最小 stack（Tier 1，"私有化最小心智成本"）

```
┌─ 单进程 Python ─────────────────────────────┐
│  FastAPI HTTP API                           │
│  inline mode: 上传后同步 derive + sync       │
│  (no Redis, no separate workers)            │
│                                             │
│  SQLite (~/.aperag/aperag.db)               │
│  LocalFS (~/.aperag/data/collections/...)   │
└─────────────────────────────────────────────┘
```

部署 = `pip install aperag && aperag serve`。客户不需要懂 Redis 不需要懂 PostgreSQL。

为什么可以这样：
- `document_index` 表在 SQLite 上 schema 完全一致，~10 行 PRAGMA + 索引就能跑
- `object_store.py` LocalFS adapter ≈ 30 行
- `INDEXING_MODE=inline` HTTP handler 同步调用 `derive` + `sync`（§E.5）
- Reconciler 在 inline mode 下变成"上传完同步重试"——错失瞬间崩溃可以下次 upload 时清

代价：单机吞吐受限（~10 docs/hour）。客户量大就升 Tier 2，docker-compose 拉起 PostgreSQL + Redis + workers，**代码完全不动**。

### L.5. 升级 / 数据迁移在私有化场景

私有化客户偶尔升级版本，可能 schema 也变。本设计的应对：
- `parse_version` 哈希函数升级 → 老 version 自然过期，新上传走新 version；老 derived/ 由 cleanup worker 在 1 小时后清。**无需迁移脚本**。
- Backend schema 变动（e.g., Qdrant payload 字段加） → 重新 `sync` 一次即可；DELETE-before-INSERT 幂等保证（§D.1）。可写一个 admin cmd 重跑 `sync` 对所有现存 `(document_id, parse_version)`。
- `document_index` schema 变动 → alembic migration（标准）。

### L.6. 与多客户多版本

私有化交付场景下，客户 A 跑 v1.2，客户 B 跑 v1.3，都是独立部署。架构师不需要在代码里支持 "v1.2 和 v1.3 兼容"——每个客户的部署是独立闭环。

这是私有化部署相对 SaaS 的**简化**：删掉了 SaaS 必须考虑的"全部租户同时升级 / 蓝绿 / 多版本共存"的复杂度。

---

## End of design pack v2

v2 的核心简化（相对 v1）：
- **删原子 flip** → per-modality 独立可见（§F.3-F.4）
- **删 Celery 决策矩阵** → 锁 Redis + asyncio（§E）
- **7 PR → 3 wave**（§K）
- **加 §C.6/C.7** 显式答 derived/ 内容 + 对象存储能力
- **加 §H** future organization forward-compat
- **加 §L** 私有化 deploy-and-forget

净效果：~200 行代码进一步减少（atomic flip orchestration 删除）+ 弱运维 contract 明确 + 私有化部署一等公民。

**遗留待 earayu2 复阅**:
- §F.4 inconsistency window 上限（v2 假设 ~25min，受 graph LLM 限制）—— 是否可接受
- §H.2 `tenant_scope_key` 列字段命名 —— 保留或换名
- §K.1 wave 划分 —— 是否进一步合并 Wave 2 + Wave 3

如 earayu2 通过 v2 整体方向，PM (燧木) 即可基于此 v2 拆 task board（3 个 wave 入 task list，按 §K.5 并行度分发 lane owner）。
