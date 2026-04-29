# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reconciler — celery T2.1.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §I.3, a
single asyncio process runs every :data:`RECONCILE_INTERVAL_SECONDS`
and performs three DB scans:

1. **PENDING dispatch** — push every PENDING row's payload onto its
   modality's Redis queue. The orchestrator's atomic ``UPDATE ...
   WHERE status='PENDING'`` is what actually claims the row, so
   re-dispatching the same row across reconciler cycles is harmless
   (the second BLPOP-er's claim simply loses).
2. **FAILED retry** — flip ``status='PENDING'`` for every FAILED row
   whose ``retry_after`` has elapsed AND ``retry_count <= MAX``. The
   PENDING dispatch above will then re-queue it next cycle.
3. **RUNNING reclaim** — flip ``status='PENDING'`` for every RUNNING
   row whose ``last_heartbeat`` is older than the §E.4 stale window
   (60s). The orchestrator's atomic claim ``UPDATE WHERE status IN
   ('PENDING','FAILED')`` will pick it up; the original worker
   process is presumed dead.

Per-modality cutover (§F.3) is intentionally NOT a reconciler scan:
the §F.3 three-statement transaction must run inside the worker's
own session immediately after ``sync()`` succeeds (architect ruling
msg=492315e8 Ruling 1 — splitting introduces an ACTIVE-but-not-
is_serving inconsistency window the spec explicitly forbids). See
``aperag.indexing.orchestrator._finalize_active_with_cutover``.

The three scans are intentionally idempotent: re-running a cycle
mid-flight produces the same end state, so a reconciler crash mid-
cycle is recoverable on next tick. The ``run_reconcile_loop`` wrapper
is the production entrypoint; the three ``reconcile_*`` functions are
the testable seams (drive each scan independently in unit tests).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import Engine, and_, insert, select, tuple_, update
from sqlalchemy.orm import Session

from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
from aperag.indexing.orchestrator import (
    MAX_RETRY_COUNT,
    DispatchPayload,
    WorkQueue,
)

logger = logging.getLogger(__name__)


# §I.3 reconcile cycle interval. Production runs the loop continuously
# at this cadence; tests call individual ``reconcile_*`` functions
# without the sleep.
RECONCILE_INTERVAL_SECONDS = 30

# §E.4 stale heartbeat threshold. RUNNING rows older than this are
# presumed crashed and reclaimed back to PENDING.
HEARTBEAT_STALE_SECONDS = 60

# Cap the per-cycle dispatch / retry / reclaim batch so a single
# cycle never floods Redis with thousands of pushes when the system
# wakes up backed up. Each batch caps at this many rows; the next
# cycle will pick up the rest.
RECONCILE_BATCH_SIZE = 100

# Wave 5 P4: cooldown before a document with zero ``document_index``
# rows is considered "stuck after upload" and re-enqueued onto
# ``q:parse``. Set to 5 minutes so a normal parse run (PDF ~30s,
# Office ~60s, OCR ~3min) never trips it. Operators can shorten the
# cooldown via ``stuck_parse_cooldown_seconds`` for tighter recovery
# on small-document workloads.
STUCK_PARSE_COOLDOWN_SECONDS = 300


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------
# (1) PENDING dispatch
# ---------------------------------------------------------------------


async def reconcile_pending_dispatch(
    *,
    engine: Engine,
    queue: WorkQueue,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """Push every PENDING row's payload onto its modality queue.

    Returns the number of payloads pushed (0 on empty board). Skips
    rows lacking ``source_path`` — they were created without dispatch
    metadata (back-compat with Wave 1 fixtures) and the orchestrator
    can't run them. A warning logs them for triage.
    """
    rows = _select_pending(engine, batch_size)
    pushed = 0
    for row in rows:
        if not row.source_path:
            logger.warning(
                "reconciler skipping PENDING row id=%d (modality=%s) — missing source_path",
                row.id,
                row.modality,
            )
            continue
        try:
            modality_enum = Modality(row.modality)
        except ValueError:
            logger.error(
                "reconciler unknown modality %r on row id=%d — leaving PENDING",
                row.modality,
                row.id,
            )
            continue
        payload = DispatchPayload(
            index_id=row.id,
            document_id=row.document_id,
            parse_version=row.parse_version,
            modality=modality_enum,
            source_path=row.source_path,
            collection_id=row.collection_id,
        )
        await queue.push(modality=modality_enum, payload=payload.to_dict())
        pushed += 1
    return pushed


def _select_pending(engine: Engine, batch_size: int) -> list[DocumentIndex]:
    with Session(engine) as session:
        stmt = (
            select(DocumentIndex)
            .where(DocumentIndex.status == IndexStatus.PENDING.value)
            .order_by(DocumentIndex.created_at)
            .limit(batch_size)
        )
        return list(session.scalars(stmt))


# ---------------------------------------------------------------------
# (2) FAILED retry
# ---------------------------------------------------------------------


def reconcile_failed_retry(
    *,
    engine: Engine,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """Flip FAILED → PENDING for every retryable row past its backoff.

    A retryable row is one with ``retry_count <= MAX_RETRY_COUNT``
    and ``retry_after <= now()``. Past-budget rows (``retry_count >
    MAX``) stay FAILED; the operator must intervene.

    Returns the number of rows flipped to PENDING.
    """
    now = _utcnow()
    with Session(engine) as session, session.begin():
        candidates = list(
            session.scalars(
                select(DocumentIndex.id)
                .where(
                    and_(
                        DocumentIndex.status == IndexStatus.FAILED.value,
                        DocumentIndex.retry_after.is_not(None),
                        DocumentIndex.retry_after <= now,
                        DocumentIndex.retry_count <= MAX_RETRY_COUNT,
                    )
                )
                .limit(batch_size)
            )
        )
        if not candidates:
            return 0
        result = session.execute(
            update(DocumentIndex)
            .where(DocumentIndex.id.in_(candidates))
            .values(
                status=IndexStatus.PENDING.value,
                retry_after=None,
                error_message=None,
            )
        )
    return result.rowcount or 0


# ---------------------------------------------------------------------
# (3) RUNNING reclaim — stale heartbeat → PENDING
# ---------------------------------------------------------------------


def reconcile_running_reclaim(
    *,
    engine: Engine,
    stale_seconds: int = HEARTBEAT_STALE_SECONDS,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """Flip RUNNING → PENDING for every row whose heartbeat is stale.

    A heartbeat older than ``stale_seconds`` (default 60s per §E.4)
    means the worker process is dead — orphaned RUNNING claims would
    otherwise block re-dispatch indefinitely. Returns the number of
    rows reclaimed.
    """
    threshold = _utcnow() - timedelta(seconds=stale_seconds)
    with Session(engine) as session, session.begin():
        candidates = list(
            session.scalars(
                select(DocumentIndex.id)
                .where(
                    and_(
                        DocumentIndex.status == IndexStatus.RUNNING.value,
                        DocumentIndex.last_heartbeat.is_not(None),
                        DocumentIndex.last_heartbeat < threshold,
                    )
                )
                .limit(batch_size)
            )
        )
        if not candidates:
            return 0
        result = session.execute(
            update(DocumentIndex)
            .where(DocumentIndex.id.in_(candidates))
            .values(
                status=IndexStatus.PENDING.value,
                last_heartbeat=None,
                # Don't bump retry_count — a stale-heartbeat reclaim
                # is "worker process died", not "the work itself
                # failed". Workers should be free to crash/restart
                # without burning the row's retry budget.
            )
        )
    return result.rowcount or 0


# Per-modality cutover (§F.3) is intentionally NOT in the reconciler.
# Per architect ruling msg=492315e8 (Ruling 1), the §F.3 three-statement
# transaction (status=ACTIVE → demote-old → promote-new) must run inside
# the worker's own DB session immediately after sync() succeeds — not
# split across reconciler cycles. Splitting introduces the orchestration
# §F.3 explicitly forbids and creates an ACTIVE-but-not-is_serving
# inconsistency window of ~30s (the reconciler interval). See
# ``aperag.indexing.orchestrator._finalize_active_with_cutover`` for the
# canonical 3-statement TX.


# Wave 10 §K.13 hard-cut: ``reconcile_collection_summaries_hook`` (the
# legacy Pattern B hook around the ``CollectionSummary`` state machine)
# is removed alongside the ``collection_summary_service`` deletion. The
# Wave 10 replacement (``reconcile_collection_descriptions_hook``)
# lands in Chunk E — it scans ``Collection`` rows directly using the
# new ``summary_updated_at`` / ``description_updated_at`` /
# ``regen_lease_*`` columns added in Chunk A, and dispatches Stage 1
# (agent-runtime explore) + Stage 2 (cheap derive) via fire-and-forget
# ``asyncio.create_task`` per spec §K.13 §2.2.


# ---------------------------------------------------------------------
# Wave 5 P4 — stuck-document parse re-enqueue (closes T3 chunk 2 obs A
# silent-drop loop).
# ---------------------------------------------------------------------
#
# T3 chunk 2 q:parse async promotion logs + drops parse failures
# (DocParser raise / source missing) without persisting any
# ``document_index`` row. Pre-Wave-5 this surfaced as ``document.status
# == PENDING`` forever — the operator had no signal to re-trigger.
# This reconciler scan closes the loop by detecting documents that
# uploaded > N minutes ago but never sprouted a ``document_index``
# row, then re-enqueueing a parse job.
#
# Architectural note: the scan is **at-most-once-per-cooldown**, not
# per cycle: every re-enqueue advances ``Document.gmt_updated`` so
# consecutive cycles do not multi-push. Without that throttle a
# stuck document would re-queue every 30s, drowning the parse
# worker pool.


async def reconcile_stuck_documents_for_parse_reenqueue(
    *,
    engine: Engine,
    queue: WorkQueue,
    cooldown_seconds: int = STUCK_PARSE_COOLDOWN_SECONDS,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """Re-enqueue ``q:parse`` for documents stuck without ``document_index`` rows.

    A document is "stuck" when:

    * ``Document.gmt_deleted IS NULL`` (still active)
    * ``Document.gmt_created < now - cooldown_seconds`` (uploaded long
      enough ago that a normal parse should have completed and
      dispatched)
    * **zero** ``document_index`` rows reference its ``document_id``
      (parse never reached :func:`dispatch_indexing` — the parse worker
      either died, the source was missing, or DocParser raised)
    * ``Document.gmt_updated < now - cooldown_seconds`` (haven't
      already been re-enqueued this cooldown window — prevents the
      30-s reconciler tick from drowning ``q:parse`` with the same
      stuck doc on every cycle)

    For each stuck document the function pushes a fresh
    :class:`ParseDispatchPayload` matching the upload handler's
    contract (per ``document_service._create_or_update_document_indexes``).
    The parse worker handles retries via its existing log-and-drop
    semantics; this scan is the **producer-side recovery loop** that
    closes T3 chunk 2 obs A.

    Returns the number of stuck documents re-enqueued. Failure to
    reach Redis logs + bumps to next cycle (queue may be transiently
    unhealthy; we never crash the loop).
    """
    payloads = await asyncio.to_thread(
        _select_stuck_documents_for_reenqueue,
        engine,
        cooldown_seconds,
        batch_size,
    )
    if not payloads:
        return 0

    pushed_ids: list[str] = []
    for payload_dict in payloads:
        try:
            await queue.push_parse(payload=payload_dict)
        except Exception as exc:  # noqa: BLE001 — transient Redis fault, retry next cycle
            logger.warning(
                "stuck-document parse re-enqueue failed for document=%s: %s — will retry next cycle",
                payload_dict.get("document_id", "<unknown>"),
                exc,
            )
            continue
        pushed_ids.append(str(payload_dict["document_id"]))

    if pushed_ids:
        await asyncio.to_thread(_mark_stuck_documents_reenqueued, engine, pushed_ids)
        logger.info(
            "reconciler stuck-parse: re-enqueued %d documents to q:parse",
            len(pushed_ids),
        )
    return len(pushed_ids)


def _select_stuck_documents_for_reenqueue(
    engine: Engine,
    cooldown_seconds: int,
    batch_size: int,
) -> list[dict[str, object]]:
    """Sync DB scan for stuck documents. Returns a list of fully-formed
    parse payload dicts ready for ``queue.push_parse``.

    Runs through ``asyncio.to_thread`` from the async caller so the
    SQLAlchemy / collection-config decoding cost does not block the
    event loop.
    """
    from aperag.domains.knowledge_base.db.models import Document

    threshold = _utcnow() - timedelta(seconds=cooldown_seconds)
    stuck_dicts: list[dict[str, object]] = []
    with Session(engine) as session:
        rows_for_doc = select(DocumentIndex.document_id).where(DocumentIndex.document_id == Document.id).exists()
        stmt = (
            select(Document)
            .where(
                and_(
                    Document.gmt_deleted.is_(None),
                    Document.gmt_created < threshold,
                    Document.gmt_updated < threshold,
                    ~rows_for_doc,
                )
            )
            .order_by(Document.gmt_created)
            .limit(batch_size)
        )
        for document in session.scalars(stmt):
            payload = _build_parse_payload_for_document(session, document)
            if payload is None:
                continue
            stuck_dicts.append(payload.to_dict())
    return stuck_dicts


def _build_parse_payload_for_document(session: Session, document):
    """Reconstruct the parse payload the upload handler would have
    pushed. Mirrors
    :func:`aperag.domains.knowledge_base.service.document_service._create_or_update_document_indexes`
    so the parse worker sees identical data shape regardless of which
    producer enqueued the job. Returns ``None`` to signal an
    unrecoverable document (missing object_path / no enabled modalities).
    """
    import json as _json

    from aperag.domains.knowledge_base.db.models import Collection
    from aperag.indexing.parse_orchestrator import ParseDispatchPayload

    metadata: dict[str, object] = {}
    if document.doc_metadata:
        try:
            metadata = _json.loads(document.doc_metadata) or {}
        except (TypeError, ValueError):
            metadata = {}
    object_path = metadata.get("object_path")
    if not object_path:
        logger.warning(
            "stuck-document parse re-enqueue: skipping document=%s — doc_metadata.object_path missing",
            document.id,
        )
        return None

    collection = session.get(Collection, document.collection_id) if document.collection_id else None
    parser_config = _resolve_collection_parser_config(collection)
    modalities = _resolve_collection_modalities(collection)
    if not modalities:
        logger.warning(
            "stuck-document parse re-enqueue: skipping document=%s — collection has no enabled modalities",
            document.id,
        )
        return None

    from aperag.indexing.parse_orchestrator import resolve_tenant_scope_key

    return ParseDispatchPayload(
        document_id=document.id,
        collection_id=document.collection_id or "",
        object_path=str(object_path),
        tenant_scope_key=resolve_tenant_scope_key(document=document, collection=collection),
        modalities=tuple(modalities),
        parser_config=parser_config,
        purge_existing_triples=True,
    )


def _resolve_collection_parser_config(collection) -> dict | None:
    """Mirror ``document_service._resolve_parser_config_for_collection``
    in sync context. Defensive on every shape (None / non-JSON /
    non-dict).
    """
    import json as _json

    if collection is None or not collection.config:
        return None
    try:
        config_dict = _json.loads(collection.config)
    except (TypeError, ValueError):
        return None
    if not isinstance(config_dict, dict):
        return None
    parser_config = config_dict.get("parser_config")
    if parser_config is None or not isinstance(parser_config, dict):
        return None
    return parser_config


def _resolve_collection_modalities(collection) -> list[str]:
    """Walk ``collection.config.{enable_vector, enable_fulltext,
    enable_knowledge_graph, enable_summary, enable_vision}`` to derive
    the modality list. Default to vector + fulltext + summary (the
    always-Wave-3-real subset) when the collection's config is
    missing or malformed.
    """
    import json as _json

    defaults = ["vector", "fulltext", "summary"]
    if collection is None or not collection.config:
        return defaults
    try:
        cfg = _json.loads(collection.config)
    except (TypeError, ValueError):
        return defaults
    if not isinstance(cfg, dict):
        return defaults

    modalities: list[str] = []
    if cfg.get("enable_vector", True):
        modalities.append("vector")
    if cfg.get("enable_fulltext", True):
        modalities.append("fulltext")
    if cfg.get("enable_knowledge_graph", False):
        modalities.append("graph")
    if cfg.get("enable_summary", False):
        modalities.append("summary")
    if cfg.get("enable_vision", False):
        modalities.append("vision")
    return modalities or defaults


def _mark_stuck_documents_reenqueued(engine: Engine, document_ids: list[str]) -> None:
    """Bump ``Document.gmt_updated`` for the re-enqueued documents so
    the next reconciler tick does not pick the same documents up
    again until the next cooldown window. Re-using ``gmt_updated``
    keeps the scan single-table and avoids an alembic migration for
    a recovery-loop bookkeeping field.
    """
    if not document_ids:
        return
    from aperag.domains.knowledge_base.db.models import Document

    with Session(engine) as session, session.begin():
        session.execute(update(Document).where(Document.id.in_(document_ids)).values(gmt_updated=_utcnow()))


# ---------------------------------------------------------------------
# Wave 10 §K.13 Chunk E — collection auto-description reconciler hook.
# ---------------------------------------------------------------------
#
# Scans ``Collection`` rows directly using the new columns added in
# Chunk A (``summary_updated_at`` / ``description_updated_at`` /
# ``regen_lease_*``) and dispatches Stage 1 / Stage 2 regen via fire-
# and-forget ``asyncio.create_task``. Three scenarios covered:
#
# 1. Collection has no summary → Stage 1 (build initial summary)
# 2. A document was added/edited/deleted after the last successful
#    Stage 1 run AND the most recent edit is at least
#    ``MIN_STALE_AGE`` old → Stage 1 (re-build summary)
# 3. Summary is newer than description → Stage 2 (refresh description)
#
# Lease conflict / collection-deleted / quality-gate failure are all
# absorbed by ``regen_summary`` / ``regen_description`` (return ``False``
# without raising). The reconciler only schedules; it never blocks on
# the regen result.


async def reconcile_collection_descriptions_hook(
    *,
    engine: Engine,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """Wave 10 §K.13 Chunk E — dispatch Stage 1 / Stage 2 regen for
    collections whose summary or description is stale.

    Returns the total number of regen tasks dispatched (Stage 1 +
    Stage 2). The hook itself is best-effort: it dispatches via
    ``asyncio.create_task`` and returns immediately, so a failed
    regen does not block subsequent cycles.
    """
    from aperag.domains.knowledge_base.service.collection_regen_service import (
        regen_description,
        regen_summary,
    )
    from aperag.domains.knowledge_base.service.regen_constants import MIN_STALE_AGE

    summary_ids, description_ids = await asyncio.to_thread(
        _select_collections_needing_regen,
        engine,
        MIN_STALE_AGE,
        batch_size,
    )

    dispatched = 0
    for collection_id in summary_ids:
        asyncio.create_task(regen_summary(collection_id))
        dispatched += 1
    for collection_id in description_ids:
        asyncio.create_task(regen_description(collection_id))
        dispatched += 1
    if dispatched:
        logger.info(
            "reconciler collection-regen: scheduled stage1=%d stage2=%d",
            len(summary_ids),
            len(description_ids),
        )
    return dispatched


def _select_collections_needing_regen(
    engine: Engine,
    min_stale_age: timedelta,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """Sync DB scan for collections needing Stage 1 or Stage 2 regen.

    Returns ``(stage1_ids, stage2_ids)``. Lease ownership is treated
    as "in flight" — the regen service handles its own lease
    semantics, but skipping owned-lease rows here saves a wasted
    ``regen_*`` call per cycle. Soft-deleted collections are
    excluded.
    """
    from aperag.domains.knowledge_base.db.models import Collection, Document

    now = _utcnow()
    stale_threshold = now - min_stale_age

    with Session(engine) as session:
        # Stage 1 scenario 1: summary is missing entirely.
        missing_summary_stmt = (
            select(Collection.id)
            .where(
                and_(
                    Collection.gmt_deleted.is_(None),
                    Collection.summary.is_(None),
                    # No active or unexpired lease (someone else is
                    # working on it — skip this cycle).
                    (Collection.regen_lease_owner.is_(None)) | (Collection.regen_lease_expires_at < now),
                )
            )
            .order_by(Collection.gmt_created)
            .limit(batch_size)
        )
        stage1_ids: list[str] = list(session.scalars(missing_summary_stmt))

        # Stage 1 scenario 2: summary exists but a document edit
        # happened after summary_updated_at AND the latest edit is at
        # least MIN_STALE_AGE old (so we don't fire while the user is
        # still editing). Bound by batch_size so a single cycle never
        # explodes.
        if len(stage1_ids) < batch_size:
            stale_doc_subq = select(Document.collection_id).where(
                and_(
                    Document.gmt_deleted.is_(None),
                    Document.gmt_updated > Collection.summary_updated_at,
                    Document.gmt_updated < stale_threshold,
                )
            )
            stale_summary_stmt = (
                select(Collection.id)
                .where(
                    and_(
                        Collection.gmt_deleted.is_(None),
                        Collection.summary.is_not(None),
                        Collection.summary_updated_at.is_not(None),
                        (Collection.regen_lease_owner.is_(None)) | (Collection.regen_lease_expires_at < now),
                        Collection.id.in_(stale_doc_subq),
                    )
                )
                .order_by(Collection.summary_updated_at)
                .limit(batch_size - len(stage1_ids))
            )
            stage1_ids.extend(session.scalars(stale_summary_stmt))

        # Stage 2: summary newer than description (or description
        # missing while summary exists). Skip if Stage 1 already
        # picked the collection up — we'll dispatch Stage 2 next cycle
        # once Stage 1 finishes.
        stage1_set = set(stage1_ids)
        stage2_stmt = (
            select(Collection.id)
            .where(
                and_(
                    Collection.gmt_deleted.is_(None),
                    Collection.summary.is_not(None),
                    Collection.summary_updated_at.is_not(None),
                    (Collection.regen_lease_owner.is_(None)) | (Collection.regen_lease_expires_at < now),
                    (
                        (Collection.description_updated_at.is_(None))
                        | (Collection.description_updated_at < Collection.summary_updated_at)
                    ),
                )
            )
            .order_by(Collection.summary_updated_at)
            .limit(batch_size)
        )
        stage2_ids = [cid for cid in session.scalars(stage2_stmt) if cid not in stage1_set]

    return stage1_ids, stage2_ids


# ---------------------------------------------------------------------
# (5) Graph vectors enqueue — 任务 #5 设计文档 §4.4
# ---------------------------------------------------------------------


def reconcile_graph_vectors_enqueue(
    *,
    engine: Engine,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """事实层 ACTIVE 之后, 把缺失的向量层行 INSERT 成 PENDING.

    任务 #5 设计文档 §4.4 conservative serial scheduling: 上传时
    dispatcher 只插 ``graph_facts`` 一行; 当 ``graph_facts`` ACTIVE 之后,
    本函数扫到这行 + 同 ``(document_id, parse_version)`` 还没有对应的
    ``graph_vectors`` 行, 就 INSERT 一行 ``graph_vectors`` PENDING.
    ``source_path`` 直接复用 facts 行的 ``derived_artifact_path``, 让
    :class:`GraphVectorsWorker.derive` 不重跑 LLM extractor.

    新增的 PENDING 行会在下一轮 :func:`reconcile_pending_dispatch` 里被
    push 到 ``q:graph_vectors``, 进而被 worker 池消费.

    幂等: 已经存在 ``graph_vectors`` 行的 ``(doc, parse_v)`` 跳过, 所以
    多次循环不会重复 INSERT. 失败重试走标准
    :func:`reconcile_failed_retry` 路径.

    返回新 INSERT 的行数 (0 时为空板).
    """
    inserted = 0
    with Session(engine) as session, session.begin():
        facts_rows = list(
            session.scalars(
                select(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.modality == Modality.GRAPH_FACTS.value,
                        DocumentIndex.status == IndexStatus.ACTIVE.value,
                        DocumentIndex.derived_artifact_path.is_not(None),
                    )
                )
                .order_by(DocumentIndex.updated_at)
                .limit(batch_size)
            )
        )
        if not facts_rows:
            return 0

        # 一次查询找出所有已经存在的 graph_vectors 行 (按 (doc, v) 去重),
        # 避免 N 次 round-trip.
        pairs = [(r.document_id, r.parse_version) for r in facts_rows]
        existing_stmt = select(DocumentIndex.document_id, DocumentIndex.parse_version).where(
            and_(
                DocumentIndex.modality == Modality.GRAPH_VECTORS.value,
                tuple_(DocumentIndex.document_id, DocumentIndex.parse_version).in_(pairs),
            )
        )
        existing_pairs = {tuple(row) for row in session.execute(existing_stmt)}

        for facts_row in facts_rows:
            key = (facts_row.document_id, facts_row.parse_version)
            if key in existing_pairs:
                continue
            session.execute(
                insert(DocumentIndex).values(
                    document_id=facts_row.document_id,
                    parse_version=facts_row.parse_version,
                    modality=Modality.GRAPH_VECTORS.value,
                    status=IndexStatus.PENDING.value,
                    tenant_scope_key=facts_row.tenant_scope_key,
                    collection_id=facts_row.collection_id,
                    source_path=facts_row.derived_artifact_path,
                    is_serving=False,
                )
            )
            inserted += 1
    return inserted


# ---------------------------------------------------------------------
# Run loop — production entrypoint.
# ---------------------------------------------------------------------


async def run_reconcile_loop(
    *,
    engine: Engine,
    queue: WorkQueue,
    shutdown: asyncio.Event,
    interval_seconds: int = RECONCILE_INTERVAL_SECONDS,
    stale_seconds: int = HEARTBEAT_STALE_SECONDS,
) -> None:
    """Run the three reconcile scans + Pattern B hook every cycle until shutdown.

    Each cycle is best-effort: an exception in any of the scans is
    logged and the cycle continues to the next scan. A cycle that
    bombs entirely (e.g. DB unreachable) sleeps the interval and
    retries — better to keep the loop alive than to crash the process.

    Wave 10 §K.13 Chunk E: ``reconcile_collection_descriptions_hook``
    is wired here as the fourth scan; it dispatches collection-summary
    + description regen via fire-and-forget ``asyncio.create_task``.
    """
    while not shutdown.is_set():
        try:
            # 任务 #5 §4.4: graph_vectors 入队必须放在 reconcile_pending_dispatch
            # 之前, 这样新 INSERT 的 PENDING 行会在同一轮 reconcile 周期里
            # push 到队列, 缩短事实层 ACTIVE → 向量层 RUNNING 的等待.
            graph_vectors_enqueued = await asyncio.to_thread(
                reconcile_graph_vectors_enqueue,
                engine=engine,
            )
            pushed = await reconcile_pending_dispatch(engine=engine, queue=queue)
            retried = await asyncio.to_thread(reconcile_failed_retry, engine=engine)
            reclaimed = await asyncio.to_thread(
                reconcile_running_reclaim,
                engine=engine,
                stale_seconds=stale_seconds,
            )
            stuck_reenqueued = await reconcile_stuck_documents_for_parse_reenqueue(
                engine=engine,
                queue=queue,
            )
            collection_regens = await reconcile_collection_descriptions_hook(engine=engine)
            if pushed or retried or reclaimed or stuck_reenqueued or collection_regens or graph_vectors_enqueued:
                logger.info(
                    "reconciler cycle: dispatched=%d retried=%d reclaimed=%d "
                    "stuck_reenqueued=%d collection_regens=%d graph_vectors_enqueued=%d",
                    pushed,
                    retried,
                    reclaimed,
                    stuck_reenqueued,
                    collection_regens,
                    graph_vectors_enqueued,
                )
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            logger.exception("reconciler cycle failed: %s", exc)
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


__all__ = [
    "HEARTBEAT_STALE_SECONDS",
    "RECONCILE_BATCH_SIZE",
    "RECONCILE_INTERVAL_SECONDS",
    "STUCK_PARSE_COOLDOWN_SECONDS",
    "reconcile_collection_descriptions_hook",
    "reconcile_failed_retry",
    "reconcile_graph_vectors_enqueue",
    "reconcile_pending_dispatch",
    "reconcile_running_reclaim",
    "reconcile_stuck_documents_for_parse_reenqueue",
    "run_reconcile_loop",
]
