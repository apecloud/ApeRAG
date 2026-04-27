# Private / on-premise deployment

ApeRAG is built so a customer can take a release archive, run one
command, and stop thinking about the system. This page is the
operator's guide to picking a deployment tier, getting it running,
and keeping it running with minimal intervention.

The shape of the indexing pipeline + the §F.5 cleanup contract make
this realistic: every resource that would otherwise rot over time
has a self-healing mechanism baked in, so an unattended deployment
does not slowly fall over.

> **Source of truth:** the architectural intent for these tiers is
> ``docs/modularization/indexing-redesign-design-pack.md`` §L. This
> page is the operator-facing guide; the design pack is the spec.

## Wave 3 release scope (read first)

Wave 3 ships the production-ready infrastructure for the
**vector**, **fulltext**, **summary**, and **vision** modalities.
Two surfaces are intentionally **gated** until Wave 4 to avoid
silently broken behaviour in production:

* **Knowledge-graph modality is gated.** The §D.3 lineage pipeline
  is structurally implemented but its production backend (the
  Nebula / PostgreSQL ``LineageGraphStore`` adapter) and the
  LLM-driven entity / relation extractor are Wave 4 scope.
  ``CollectionConfig.enable_knowledge_graph`` defaults to ``false``;
  any collection that opts in will get an explicit
  ``WorkerFactoryError`` on its graph row instead of an empty
  ``status=ACTIVE`` that would mislead operators into thinking
  graph search works. Wave 4 release flips the default back to
  ``true`` once the real backend is wired.

* **Parser supports UTF-8 markdown only.** PDF, Word, image, and
  other binary inputs raise ``ValueError`` from the indexing
  parser. Wave 4 wires the docparser / Marker / OCR pipelines that
  convert binary inputs to markdown before parsing. Until then,
  upload handlers must accept only markdown bodies.

The Wave 4 backlog is locked: real graph backend + extractor + real
parser integration + cleanup-loop modality fan-out + cross-modality
contract tests (already merged via PR #1730). See architect
msg=c79e9a3f for the full gap analysis that produced this scope cut.

## Pick a deployment tier

| Tier | Throughput | Stack | When to pick |
|---|---|---|---|
| **Tier 1 — Single binary / inline** | ~10 docs / hour | SQLite + LocalFS, single process, `INDEXING_MODE=inline` | Demo / POC; one-machine pilot; air-gapped sites with no Redis budget |
| **Tier 2 — Single VM / async** | ~100 concurrent docs | PostgreSQL + LocalFS or MinIO + Redis + 5 worker processes (one per modality) | Standard customer install; everything on a single VM via `docker-compose` |
| **Tier 3 — Multi-VM / scale-out** | > 100 concurrent docs | PostgreSQL + S3-compatible object store + Redis + horizontally scaled worker processes | Large customer; spreads workers across nodes |

**All three tiers run the same code.** The only differences are the
config values (database URL, object store backend, `INDEXING_MODE`)
and how many worker processes are running. There is no "small
customer fork" vs "enterprise fork".

## Tier 1 — `INDEXING_MODE=inline`

Tier 1 is the lightest deployment. The HTTP API process does
parsing, embedding, and the per-modality `derive` + `sync` calls
synchronously inside the request task. There is **no Redis, no
worker pool, no reconciler loop** — the upload handler returns when
the document is fully indexed for every requested modality.

### Stack

```
┌─ Single Python process ────────────────────────┐
│  FastAPI HTTP API                              │
│  + IndexingMode.INLINE upload handler          │
│                                                │
│  SQLite at ~/.aperag/aperag.db                 │
│  LocalFS at ~/.aperag/data/...                 │
└────────────────────────────────────────────────┘
```

No Redis. No PostgreSQL. No separate worker. One Python process,
two files on disk.

### Setup

```bash
pip install aperag
export INDEXING_MODE=inline
export APERAG_DB_URL='sqlite:///~/.aperag/aperag.db'
export APERAG_OBJECT_STORE='localfs:///~/.aperag/data'
aperag serve
```

That is the whole installation. The first request migrates the
SQLite schema; subsequent requests upload + index + serve in the
same process.

Constraints:

- Single-process throughput caps the tier at roughly 10
  documents / hour because graph LLM extraction + embedding calls
  block the request thread.
- A request that hits an LLM rate-limit waits in-process; there is
  no background retry. The `dispatch_indexing` call surfaces the
  exception to the HTTP client, who can re-upload.
- A worker crash mid-indexing means the upload handler returns an
  error; on next upload of the same document the dispatcher's
  §C.7 `read_or_none` contract picks up the half-finished derive
  artifact and re-syncs from there.

When throughput exceeds these limits, switch to Tier 2 — **no code
change**.

## Tier 2 — single-VM `INDEXING_MODE=async` via docker-compose

Tier 2 runs PostgreSQL, Redis, MinIO, and the five modality worker
processes alongside the HTTP API on a single VM. The HTTP handler
does an `INSERT … status=PENDING` per modality and returns
immediately; workers pick up the queue, run derive + sync, and the
worker-side cutover transaction promotes the row to
`is_serving=TRUE` (per §F.3).

### Stack

```
┌─ docker-compose ───────────────────────────────────────────────┐
│                                                                │
│  ┌─ aperag-api (FastAPI) ──┐    ┌─ aperag-worker-vector  ─┐    │
│  │ INDEXING_MODE=async     │    │ run_vector_worker        │    │
│  │ /upload → INSERT        │    └──────────────────────────┘    │
│  │   PENDING + RPUSH       │    ┌─ aperag-worker-fulltext ─┐    │
│  └─────────────────────────┘    │ run_fulltext_worker      │    │
│                                 └──────────────────────────┘    │
│  ┌─ postgres ──────────────┐    ┌─ aperag-worker-graph ────┐    │
│  │ document_index table    │    │ run_graph_worker         │    │
│  └─────────────────────────┘    └──────────────────────────┘    │
│  ┌─ redis ─────────────────┐    ┌─ aperag-worker-summary ──┐    │
│  │ q:vector / q:fulltext / │    │ run_summary_worker       │    │
│  │ q:graph / q:summary /   │    └──────────────────────────┘    │
│  │ q:vision                │    ┌─ aperag-worker-vision  ──┐    │
│  └─────────────────────────┘    │ run_vision_worker        │    │
│                                 └──────────────────────────┘    │
│  ┌─ minio ─────────────────┐    ┌─ aperag-reconciler ──────┐    │
│  │ collections/<cid>/…     │    │ 30s cycle (§I.3)         │    │
│  │ source/ + derived/      │    └──────────────────────────┘    │
│  └─────────────────────────┘    ┌─ aperag-cleanup ─────────┐    │
│                                 │ 5min cycle (§F.5)        │    │
│                                 └──────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

### Setup

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp examples/private-deployment/.env.example .env
# Edit .env: LLM endpoint URL, embedding endpoint URL, admin password.
docker compose -f examples/private-deployment/docker-compose.yml up -d
```

The compose file pulls one image, starts each component, and waits
for the API to come up. Initial schema migration runs from the API
container's startup hook.

Throughput budget: ~100 concurrent documents. The graph worker is
the bottleneck (LLM extraction averages ~25 minutes per 100 docs at
concurrency 4). A 30-minute SLO covers the worst case; the §J.1
`indexing.index_lag_seconds` gauge is the operational signal.

### `INDEXING_MODE` switch

| Mode | Behaviour |
|---|---|
| `inline` | Upload returns when indexed; no queue. Tier 1. |
| `async` | Upload returns immediately; worker pool catches up. Tier 2 / 3. |

Switching modes does not require schema changes. Operators can pin
`INDEXING_MODE=inline` for a developer laptop and lift it on the
production VM.

## Tier 3 — multi-VM scale-out

Tier 3 is Tier 2 with worker processes spread across multiple VMs
behind a shared Redis + PostgreSQL + S3-compatible object store. The
modality concurrency caps (§E.2: vector 16, fulltext 32, graph 4,
summary 4, vision 4) are per-process, so a customer needing 32
concurrent graph extractions runs eight graph-worker processes (or
two VMs with four each).

There is no architectural difference between Tier 2 and Tier 3 —
just more worker processes pointing at the same Redis and the same
object store.

## "Deploy and forget": what self-heals automatically

Every resource that would otherwise rot has a corresponding
self-healing mechanism. Operators do not need cron jobs, manual
clean-ups, or scheduled re-indexing.

| Resource that would rot | What v2 does | Reference |
|---|---|---|
| Old `parse_version` artifacts in object store + DB | Cleanup worker scans every 5 minutes (Path A: orphan parse_version GC) | §F.5 |
| Worker process crashes | Reconciler reclaims `RUNNING` rows whose heartbeat is > 60s stale (does NOT increment `retry_count` — worker death ≠ work failure) | §I.3 + §E.4 |
| LLM API rate limits | Per-`(resource_class, tenant_scope_key)` token bucket; over-limit wait + retry without surfacing to client | §H.5 |
| Permanently failing tasks | Exponential backoff (30s → 60s → 120s → 240s → 480s); after 5 retries flagged for operator | §I.2 |
| Half-written derived artifacts | LocalFS `tmp + fsync + rename`; S3 / MinIO `CompleteMultipartUpload` atomic visibility | §C.7 |
| Soft-deleted documents | Cleanup worker Path B: per-document deletion cascade (per-modality backend delete + graph lineage cleanup) | §F.5 |
| Soft-deleted collections | Cleanup worker Path C: collection-level cascade (find docs by collection, invoke Path B per child, then remove collection row) | §F.5 |
| Quota config drift across tenants | Per-tenant policy lookup falls back to `default` policy when no override exists; missing default is the only failure mode and surfaces immediately | §H.5 |

The §F.5 Path C cascade in particular is what makes "deploy and
forget" credible for collection lifecycle: deleting a collection is
a single `UPDATE collection SET deleted_at = NOW()` from the API,
and the cleanup worker idempotently drains every child document's
indexing state in subsequent cycles. If the cleanup worker dies
mid-cascade, the next cycle resumes from where it stopped — no
operator intervention.

## Observability

Four §J.1 SLIs emit to OTLP:

| Metric | Type | Description |
|---|---|---|
| `indexing.index_lag_seconds` | gauge | Time from `PENDING` row insert to `is_serving=TRUE`. Per-modality attribute. |
| `indexing.index_failure_total` | counter | Increments on every `_finalize_failed`. Per-modality + per-error_kind attributes. |
| `indexing.index_success_total` | counter | Increments on every `_finalize_active_with_cutover`. Per-modality. |
| `indexing.queue_depth` | gauge | Outstanding items in each modality's Redis queue. |

Operators can wire these to their own collector. If no collector is
configured the emit calls are no-ops; the system runs unchanged.

## Upgrades and migrations

A private customer upgrading versions runs the new release with the
same database / object store. The system handles the new version's
new `parse_version` automatically:

- New uploads use the new `parse_version`.
- Old `parse_version` artifacts age out via the 1-hour cool-down +
  cleanup worker (§F.5 Path A).
- No manual reindex command is required for a `parse_version` rev.

Backend schema changes (e.g., a new Qdrant payload field) follow
the §D.1 DELETE-before-INSERT contract — re-running `sync` for
existing `(document_id, parse_version, modality)` triples is the
only step. An admin command exposes a "re-sync everything" path for
the rare case it is needed.

`document_index` schema changes ride alembic in the standard way.

## Multi-customer deployments

Tier 1 / 2 / 3 deployments are independent stacks per customer; the
ApeRAG codebase does not maintain runtime cross-customer state. A
customer running v1.5 and another running v1.6 do not coordinate;
each upgrade is a closed loop.

This is the simplification private deployment buys over SaaS: there
is no "all tenants must upgrade simultaneously", no blue-green, no
multi-version reconciliation. Per-customer release-train cadence is
the customer's choice.

## When to escalate

The §F.5 self-healing mechanisms above cover the operational steady
state. The signals that warrant escalation:

- Any `indexing.index_failure_total` rate that does not drop after
  the §I.2 retry budget — usually means an LLM endpoint mis-
  configuration or quota exhaustion outside the token bucket's view.
- `indexing.queue_depth` that climbs without bound — usually means
  workers cannot keep up; check worker process count + LLM
  endpoint latency.
- Cleanup worker logs reporting "skipping backend delete" repeatedly
  for the same `(document_id, modality)` — usually means a backend
  is unreachable; the row stays in the DB until the backend recovers.

These are exception conditions; the steady state is silent.
