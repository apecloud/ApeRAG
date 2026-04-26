# ApeRAG Indexing Redesign — Design Pack

**Owner**: 符炫炜 (chief architect)
**Status**: Draft for earayu2 review
**Date**: 2026-04-26
**Trigger**: earayu2 directive `#celery msg=56812dd6` + `msg=d8080c08` — "可靠稳定的文档处理系统 / 100 concurrent docs / pre-launch / 简单 over 复杂 / 完整 redesign"

## Preface

ApeRAG is **pre-launch** (no users, no production data, no migration). The current Celery-based document indexing system grew incrementally and now exhibits architecture-level problems that would compound under load. earayu2 has authorized full redesign — including replacing Celery, rewriting the indexing layer, or even going HTTP-only — with the constraint: **simple over complex, code quality over feature breadth, no historical baggage**.

This design pack delivers (1) a comprehensive analysis of the current system, (2) a first-principles redesign that prioritizes simplicity and reliability, and (3) a phased migration plan.

The architect's recommendation, in one sentence: **drop Celery, adopt a filesystem-as-source-of-truth pattern with derived per-modality artifacts (jsonl, markdown, etc.), use a thin Redis-backed asyncio worker pool, and let the database hold the state machine on `(document_id, parse_version, modality)` triples** — this collapses three different ownership layers (Celery task / lease ledger / DB state) into one (DB), eliminates ~50% of the current code complexity, scales to 100+ concurrent documents on a single server, and makes per-modality reasoning concrete (each modality reads/writes one derived artifact file).

Sections:
- §A — Current system analysis (with file:line evidence)
- §B — First principles
- §C — Three-layer document model (source / derived / index)
- §D — Idempotency contract per modality
- §E — Concurrency model decision (HTTP-only vs lightweight task vs replace Celery)
- §F — State machine + atomic flip
- §G — Multi-modal unified pipeline
- §H — Multi-tenant isolation (recommend simple)
- §I — Failure recovery
- §J — Observability
- §K — Migration plan + phase sequence

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

### C.4. Object store choice

The object store is whatever ApeRAG already uses (S3 / MinIO / local filesystem in dev). This design pack does not impose a new dependency — the directory layout overlays the existing object store.

### C.5. What does not live in this layout

- Cache: `aperag/cache/` (D10.g) keeps its own L1/L2 read-primitive cache. It reads from object store but does not own state.
- Index state machine: lives in the database (`DocumentIndex` table), not on disk.
- Tenancy: `collection_id` is part of the path, but the database is the authority on who owns the collection.

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

### D.3. Graph indexer fix (the existing bug)

`aperag/graphindex/storage/nebula.py:354 upsert_entities` must change from append-on-conflict to:

```python
def sync_graph_from_jsonl(jsonl_path, document_id, parse_version):
    # Step 1: delete any previously-synced rows for this (doc, version)
    nebula.execute(
        "DELETE VERTEX WHERE document_id == $doc AND parse_version == $ver",
        doc=document_id, ver=parse_version,
    )
    nebula.execute(
        "DELETE EDGE  WHERE document_id == $doc AND parse_version == $ver",
        ...
    )
    # Step 2: insert from the immutable jsonl artifact
    for line in open(jsonl_path):
        record = json.loads(line)
        nebula.upsert_vertex(... document_id=document_id, parse_version=parse_version)
        # or upsert_edge for relation records
```

`upsert_vertex` and `upsert_edge` in this redesign perform `INSERT ... ON CONFLICT REPLACE` semantics with `parse_version` as part of the unique key — never append.

For Neo4j the change is similar; Nebula GQL syntax differs but the semantics are identical.

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

## §E. Concurrency model

This is the most consequential architectural decision. earayu2 invited evaluation of three options:
1. HTTP-only (no async / no task system)
2. Lightweight task system (replace Celery)
3. Keep Celery but use it correctly

### E.1. Decision matrix

| Criterion | HTTP-only | Lightweight (RQ / dramatiq / custom) | Celery |
|---|---|---|---|
| Code lines (estimated indexing layer) | **400** | 700 | 2500 (current) |
| New external dependency | None | Redis (already used) | Redis + Celery + kombu |
| 100 concurrent docs feasible | Yes (with asyncio + worker process pool) | Yes | Yes |
| Worker crash recovery | Process supervisor (systemd / Kubernetes) | Reconciler reclaim | Reconciler + lease + token |
| Operations overhead | Lowest | Low | Medium-high |
| Visibility (operators reading state) | High (DB only) | High | Medium |
| LLM rate-limit handling | Backpressure in HTTP layer | Token bucket per worker pool | Retry + backoff (current) |
| Time to implement | 1-2 weeks | 2-3 weeks | already exists, just refactor |
| Future scale beyond 100 concurrent | Horizontal scale (multiple HTTP servers) | Same as Celery | Same |

### E.2. Recommendation: lightweight Redis-backed asyncio worker pool — NOT Celery, NOT pure HTTP

After weighing simplicity, correctness, and 100-concurrent target:

**HTTP-only** is too restrictive: long-running operations (graph extraction can take 30+ seconds with LLM calls) block the HTTP request, forcing the client to either hold the connection or poll. Polling is its own complexity. And LLM rate-limit retry over minutes is awkward to express as an HTTP response cycle.

**Celery** is too heavy: the current Celery architecture has the three-layer skew problem (§A.3), needs a Python lease thread, has chord-callback fragility, and earayu2 has stated dissatisfaction. Refactoring to "use Celery correctly" is possible but the simplification ceiling is bounded by Celery's design.

**Lightweight Redis-backed asyncio worker pool** balances:
- One process per modality (5 small worker processes), each running asyncio
- Each worker pulls from a Redis list (`BLPOP` with timeout) — no Celery, no chord, no lease
- State in DB; worker reads task → does work → updates DB
- Reconciler still exists (a single small loop) for retry-after-failure
- Per-modality concurrency limit set per worker process via asyncio semaphore
- Per-resource (LLM, embedding API) token bucket lives in a shared Redis key

This is the recommendation. Concrete sketch in §E.3.

### E.3. Recommended architecture

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

### E.4. How concurrency reaches 100 docs simultaneously

100 documents × 5 modalities = 500 in-flight operations at peak. Workers handle them as follows:

- `parse_worker` (concurrency 8): parses 8 docs at a time. With ~5s parse time, ~96 docs/minute throughput. 100 docs parsed in ~63 seconds.
- `vector_worker` (concurrency 16): embeds chunks. Constrained by embedding API rate limit (assumed 100 req/sec).
- `fulltext_worker` (concurrency 32): writes ES. ES handles thousands of writes/sec; 32 is generous.
- `graph_worker` (concurrency 4): LLM extraction is expensive (~30s each). 4 concurrent × 100 docs = 25 minutes total. **This is the bottleneck.**
- `summary_worker` (concurrency 4): LLM call, similar to graph but shorter.
- `vision_worker` (concurrency 4): GPU/embedding-bound.

End-to-end: a 100-doc burst takes about 25 minutes because of graph-modality LLM extraction. **All non-graph modalities complete in under 2 minutes.** Per `parse_version` atomic flip, the document becomes searchable on vector + fulltext modalities first, then on graph as it completes — `index_state` discriminator (§G.5) lets clients know.

If this is too slow, scale horizontally: add a second graph_worker process (concurrency 4 × 2 = 8 graph in-flight). This is the same pattern as Celery worker scaling but with 1/10th the code.

### E.5. No lease thread, no chord, no token games

The current system has:
- Python lease thread that renews `lease_expires_at` every 60s (§A.4)
- Celery chord callback that aggregates 5 parallel tasks
- Processing token validated at task entry to prevent ghost callbacks

The new system has:
- Worker writes `last_heartbeat = now()` at task start (one DB UPDATE)
- No chord; each modality reports its own status independently
- Reconciler reclaims any task with `last_heartbeat < now - 60s` AND status=CREATING

This is simpler and survives worker crashes naturally (heartbeat stops being updated → reconciler reclaims).

### E.6. What about HTTP-only?

For very small deployments (<10 docs/hour), HTTP-only with synchronous in-request processing works. The architect leaves this as a documented escape hatch — same code can run in synchronous mode by setting an env var that makes the HTTP handler call `sync_<modality>` directly instead of `RPUSH`. This is a 50-line addition.

But the recommendation is to ship the asynchronous Redis-queue version as the default; it scales to 100 concurrent without architectural change.

---

## §F. State machine + atomic flip

### F.1. New `DocumentIndex` schema

```sql
CREATE TABLE document_index (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR NOT NULL,
    parse_version VARCHAR(16) NOT NULL,
    modality VARCHAR NOT NULL,            -- 'vector' | 'fulltext' | 'graph' | 'summary' | 'vision'

    status VARCHAR NOT NULL,              -- see F.2 below
    error_message TEXT,
    retry_count INT DEFAULT 0,
    retry_after TIMESTAMPTZ,

    last_heartbeat TIMESTAMPTZ,           -- set by worker on each progress step
    derived_artifact_path TEXT,           -- e.g. 'collections/A/documents/D/derived/parse_v123/kg.jsonl'

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (document_id, parse_version, modality)   -- the convergence triple
);

CREATE TABLE document (
    id VARCHAR PRIMARY KEY,
    collection_id VARCHAR NOT NULL,
    active_parse_version VARCHAR(16),     -- the parse_version the search path uses
    pending_parse_version VARCHAR(16),    -- the parse_version currently being indexed (NULL if none)
    -- ... existing fields ...
);
```

The composite unique key `(document_id, parse_version, modality)` is the convergence triple from §B. The `document.active_parse_version` and `document.pending_parse_version` columns implement atomic flip.

### F.2. Status enum (simplified from current)

```
PENDING       — row created, work not yet started
RUNNING       — a worker is processing this triple; last_heartbeat is being updated
ACTIVE        — backend reflects this triple's derived artifact
FAILED        — terminal failure for this attempt; retry_after gives next try time
```

Note: removed `CREATING` / `DELETING` / `DELETION_IN_PROGRESS`. Deletion is handled by removing the row and letting a separate cleanup worker (§F.5) garbage-collect backend entries by `(doc, version)`. ACTIVE is the only "happy" state; FAILED is the only "sad" state; PENDING/RUNNING are transient.

### F.3. Atomic flip

When a document is uploaded or re-uploaded:

1. Compute `parse_version` from content
2. If `parse_version == document.active_parse_version`: nothing to do (re-upload of identical content)
3. Else:
   a. Set `document.pending_parse_version = parse_version`
   b. Insert 5 `document_index` rows for `(document_id, parse_version, *)` with status PENDING
   c. Reconciler dispatches them to workers; workers run; status becomes ACTIVE one-by-one
   d. When all 5 modalities are ACTIVE, transactionally:
      - `UPDATE document SET active_parse_version = pending_parse_version, pending_parse_version = NULL WHERE id = X`
      - Old `active_parse_version` rows become candidates for cleanup
   e. Cleanup worker garbage-collects backend entries for `(doc, old_version)` — see §F.5

### F.4. What "all 5 modalities ACTIVE" means with optional modalities

Some collections don't have all 5 modalities enabled (e.g., no graph index). The `Collection.config.indexers_enabled` set determines which modalities are required. Atomic flip flips when **all enabled modalities are ACTIVE**, ignoring disabled ones.

A modality stuck on FAILED indefinitely blocks the flip. To unstick:
- Operator can manually mark the modality as "skipped" (a special row state)
- Or: configure per-modality optional flag — non-blocking modalities don't gate the flip but are surfaced as `index_state` in search results (§G.5)

Recommendation: for a pre-launch system, **all enabled modalities block the flip by default** — keeps the model simple. Add per-modality optional flag as a phase-2 enhancement only if real operational pain emerges.

### F.5. Cleanup worker (replaces DELETING / DELETION_IN_PROGRESS)

A separate cleanup worker runs periodically (e.g., every 5 minutes) and deletes backend entries for any `(document_id, parse_version)` that is:
- Not the current `active_parse_version` AND
- Not the current `pending_parse_version` AND
- Older than 1 hour

This GC pattern is conceptually like a tombstone collector; it is independent from the main write path. Failure to clean up does not affect correctness — just consumes backend storage.

Document deletion (user requests removal) sets `document.active_parse_version = NULL, document.pending_parse_version = NULL, document.deleted_at = NOW()`. Cleanup worker garbage-collects all `(document_id, *)` backend entries.

### F.6. Why this is simpler than current

- 4 status values (PENDING / RUNNING / ACTIVE / FAILED) vs 6 in current
- No `processing_token` (replaced by heartbeat — simpler concept)
- No version field (parse_version is sufficient; version was redundant)
- Deletion is async cleanup, not a state transition — eliminates DELETING / DELETION_IN_PROGRESS / their reclaim logic
- Atomic flip is one DB UPDATE, not a multi-step orchestration

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

### G.5. `SearchResultItem.metadata.index_state` discriminator (D10.h amendment-#3 follow-up)

Each search result carries `index_state: {<modality>: ACTIVE | FAILED | NOT_ENABLED}` in its metadata so clients know which modality served the hit and which other modalities are healthy for follow-up calls. This was identified in the architect/Bryce alignment thread (msg=2ee66c89) as Phase 3 in the prior sequencing; it stays in the redesign as a small schema addition to `SearchResultMetadata`.

---

## §H. Multi-tenant isolation — recommend simple

earayu2's current message did not emphasize multi-tenancy fairness; the focus was reliability + simplicity + 100 concurrent. The architect recommends:

### H.1. Required (always-on)

- Tenant context (`collection_id`) is part of every queue message and every DB row
- Backend writes tagged with `collection_id` in metadata (existing pattern, retained)
- Cross-tenant validation at HTTP layer: requestor's auth must cover the collection

### H.2. Optional (recommend deferring)

- Per-tenant fairness queueing (Bryce's Phase B)
- Per-tenant concurrency cap
- Per-tenant resource quotas

These are deferred because:
- Pre-launch system has no real tenant load to distinguish
- The asyncio worker pool with per-modality concurrency limits is already a coarse fairness mechanism (no tenant can monopolize more than the worker's concurrency)
- Adding fairness machinery before observing real noisy-neighbor behavior is premature optimization

### H.3. When to add fairness

Add fairness machinery when observability (§J) shows:
- Per-tenant queue depth > 100 sustained for any single tenant
- Per-tenant index lag > 10 minutes for the median
- Cross-tenant variance in index lag > 5x

Until those signals appear, the simpler design is correct.

### H.4. Bulkhead — defense in depth

Independent of fairness, **resource isolation** does matter even at small scale: a malicious or buggy document (e.g., 100MB JSON, prompt injection in the LLM call) shouldn't crash a worker that affects all tenants. The recommendation:

- Each worker process has a hard memory limit (Linux cgroup or Docker) — overrun → process restart
- LLM API calls have a hard timeout (configurable, default 60s)
- Embedding API calls have a hard timeout (default 30s)
- Document upload size cap (configurable, default 50MB)

These are existing patterns in the current code; the redesign keeps them and consolidates into a single config file.

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

2. index_failure_rate{collection_id, modality}
   - failed transitions / total transitions over rolling 5min window
   - Surfaces: which modality is broken?

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

## §K. Migration plan + phase sequence

The current system is on `main` and works (post-D10.h cutover). The redesign is a substantial rewrite. Per earayu2's `pre-launch / no users / no migration` guidance, the migration is **hard-cut**: delete the old, ship the new, no compatibility window.

### K.1. PR sequence (7 PRs)

| # | Phase | Scope | Dependencies |
|---|---|---|---|
| **PR-A** | Observability primitives | Emit 4 SLI from current Celery system to OTLP (alignment with #1702) | None |
| **PR-B** | New schema + idempotent indexers | Add `parse_version` to `document_index`; rewrite all 5 indexers as `Modality` ABC implementations; fix graph DELETE-before-INSERT; add idempotency self-tests | None (can run alongside Celery) |
| **PR-C** | Object store layout | Implement source/derived directory structure; document parser writes derived artifacts; existing indexers still consume in-memory parts (compatibility shim) | PR-B |
| **PR-D** | Worker pool + Redis queue | Implement 5 modality workers + reconciler + cleanup; deploy in parallel with Celery (feature flag: dispatch goes to new system or old) | PR-B, PR-C |
| **PR-E** | Atomic flip + state machine | Implement document.active_parse_version / pending_parse_version; flip logic; cleanup worker | PR-D |
| **PR-F** | Cutover | Set feature flag default to new system; delete Celery + reconciler.py + processing_lease.py + tasks.py + indexing/orchestration.py + graphindex v2 indirection | PR-E |
| **PR-G** | Per-modality availability discriminator | Add `index_state` to `SearchResultMetadata`; D10.h amendment | PR-F |

### K.2. PR sizing

| PR | Estimated diff |
|---|---|
| PR-A | +200 / -0 |
| PR-B | +1500 / -1500 (rewrites in parallel) |
| PR-C | +400 / -100 |
| PR-D | +1200 / -0 |
| PR-E | +600 / -200 |
| PR-F | +100 / -3000 (delete-heavy) |
| PR-G | +150 / -50 |

Total: roughly +4150 / -4850. Net subtraction of ~700 lines despite a bigger feature set.

### K.3. Deletion list (PR-F)

Files removed in the cutover:
- `aperag/tasks/{collection,document,models,processing_lease,reconciler,scheduler,utils}.py` (Celery layer)
- `aperag/domains/indexing/{tasks,orchestration,manager}.py` (Celery-coupled orchestration)
- `aperag/concurrent_control/redis_lock.py` (unused after redesign)
- `aperag/graphindex/v2/*` (collapsed into `aperag/indexing/graph.py`)

Files reduced significantly:
- `aperag/domains/indexing/{vector,fulltext,graph,summary,vision}_index.py` rewritten as `aperag/indexing/{vector,fulltext,graph,summary,vision}.py`

### K.4. Rollback considerations

Pre-launch has no rollback considerations in the production sense. During development, PR-F can be reverted if the new system shows unexpected behavior under the synthetic 100-concurrent load test. The synthetic test runs in CI and is the merge gate for PR-F.

### K.5. Feature flag during PR-D / PR-E

A single env var `INDEXING_BACKEND=celery|new` switches between systems. Default during PR-D / PR-E is `celery` (old system stays canonical until PR-F). This lets developers exercise the new system in staging without affecting any other environment.

After PR-F merges, the feature flag and the old code path are deleted in the same PR.

### K.6. Test plan summary

- Unit tests: per-modality idempotency self-test (PR-B); reconciler dispatch logic (PR-D); atomic-flip semantics (PR-E)
- Integration tests: end-to-end document upload → indexed → searchable (each PR)
- Synthetic load test: 100 documents in parallel through new system; assert all 5 modalities ACTIVE within 30 minutes (PR-F gate)

### K.7. Implementation owners

Same model as D10:
- Architect (符炫炜) — design pack canon, PR scope decisions, line-by-line review
- Bryce / cuiwenbo / chenyexuan / 黄恒 / 明书 — implementation per PR claim

The 7 PRs can largely be parallelized after PR-A and PR-B land (PR-C through PR-E have a sequential dependency chain because each depends on the previous).

---

## End of design pack

This design pack proposes a **simpler, more reliable, more inspectable** indexing system that scales to 100+ concurrent documents on a single server, eliminates the three-layer ownership skew, fixes the graph idempotency bug, and reduces the indexing layer code count by approximately 700 lines while adding (not removing) functionality.

The architect recommends earayu2 review §E (concurrency model decision) and §K (PR sequence) first; those are the most consequential decisions. If the recommendations there are accepted, the rest of the design pack flows.

Open question for earayu2 to confirm or override:
- Is the recommended concurrency model (lightweight Redis-backed asyncio worker pool, dropping Celery entirely) acceptable, or do you prefer the Celery-refactor or HTTP-only paths?
- Is per-modality availability (atomic flip when *all enabled* modalities ACTIVE) the right contract, or do you want graceful degradation (flip per-modality independently)?
- Is the 7-PR sequence acceptable, or do you want to combine some?

The architect can revise the design pack based on earayu2's answers.
