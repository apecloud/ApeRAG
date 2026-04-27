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

from sqlalchemy import Engine, and_, select, update
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


# ---------------------------------------------------------------------
# Pattern B periodic hook — collection summary reconciliation.
# ---------------------------------------------------------------------
#
# Per architect msg=3890c9d7 Pattern B ruling, the legacy
# ``CollectionSummaryReconciler.reconcile_all()`` (formerly a Celery
# beat task scheduled every 30s via ``django-celery-beat``) is merged
# into this 30-s reconciler loop as a sibling scan. The loop now also:
#
#   4. **Reclaim stale collection-summary leases** — flip
#      ``CollectionSummary.status='GENERATING' AND
#      lease_expires_at < now()`` back to ``PENDING`` so the next
#      reconciliation pass re-claims them.
#   5. **Dispatch pending collection summaries** — for every
#      ``CollectionSummary`` whose ``version != observed_version`` and
#      ``status='PENDING'``, atomically claim with a fresh
#      ``processing_token`` + ``lease_expires_at``, then fire-and-forget
#      ``collection_summary_task`` via ``asyncio.create_task(
#      asyncio.to_thread(...))`` (Pattern C dispatch).
#
# The dispatch is intentionally fire-and-forget (Pattern C):
# ``collection_summary_task`` is regenerable + idempotent (its own
# claim guard inside the task body re-validates ownership), so
# losing the dispatch on reconciler crash is recovered next cycle by
# the stale-lease reclaim. The hook never blocks the loop on summary
# generation duration.


async def reconcile_collection_summaries_hook(
    *,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> None:
    """Pattern B periodic hook — Wave 3 architect msg=3890c9d7.

    Replaces legacy ``aperag.tasks.reconciler.CollectionSummaryReconciler.
    reconcile_all()`` + the ``django-celery-beat`` 30-s schedule entry.
    Runs inside the existing 30-s reconciler loop and:

    1. Reclaims stale ``GENERATING`` summaries whose lease expired.
    2. Selects ``PENDING`` summaries whose ``version`` exceeds
       ``observed_version`` (work to do).
    3. Atomically claims each (fresh ``processing_token`` +
       ``lease_expires_at``).
    4. Fires ``collection_summary_task`` per claim as a Pattern C
       fire-and-forget background asyncio task — never blocks the loop.

    Imported lazily inside the function body to avoid the circular
    ``aperag.indexing.reconciler → aperag.domains.knowledge_base.{tasks,
    db.models} → aperag.indexing`` dependency at module-load time.
    """
    from aperag.config import get_sync_session
    from aperag.domains.knowledge_base.db.models import (
        CollectionSummary,
        CollectionSummaryStatus,
    )
    from aperag.domains.knowledge_base.tasks import (
        build_lease_expires_at,
        collection_summary_task,
        generate_processing_token,
    )

    def _reclaim_stale_and_claim_pending() -> list[tuple[str, str, int, str]]:
        """Sync DB-only worker. Returns list of claimed dispatch tuples."""
        from aperag.utils.utils import utc_now as _utc_now

        claimed_dispatches: list[tuple[str, str, int, str]] = []
        for session in get_sync_session():
            current_time = _utc_now()
            reclaim_stmt = (
                update(CollectionSummary)
                .where(
                    and_(
                        CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                        CollectionSummary.processing_token.is_not(None),
                        CollectionSummary.lease_expires_at.is_not(None),
                        CollectionSummary.lease_expires_at < current_time,
                    )
                )
                .values(
                    status=CollectionSummaryStatus.PENDING,
                    error_message="stale lease reclaimed",
                    processing_token=None,
                    lease_expires_at=None,
                    gmt_updated=current_time,
                    gmt_last_reconciled=current_time,
                )
            )
            reclaim_result = session.execute(reclaim_stmt)
            if reclaim_result.rowcount:
                session.commit()
                logger.warning(
                    "Reclaimed %s stale collection-summary leases back to PENDING",
                    reclaim_result.rowcount,
                )

            pending_stmt = (
                select(CollectionSummary)
                .where(
                    and_(
                        CollectionSummary.version != CollectionSummary.observed_version,
                        CollectionSummary.status == CollectionSummaryStatus.PENDING,
                    )
                )
                .limit(batch_size)
            )
            pending = list(session.scalars(pending_stmt))
            if not pending:
                return claimed_dispatches

            for summary in pending:
                token = generate_processing_token()
                claim_stmt = (
                    update(CollectionSummary)
                    .where(
                        and_(
                            CollectionSummary.id == summary.id,
                            CollectionSummary.status == CollectionSummaryStatus.PENDING,
                            CollectionSummary.version == summary.version,
                        )
                    )
                    .values(
                        status=CollectionSummaryStatus.GENERATING,
                        processing_token=token,
                        lease_expires_at=build_lease_expires_at(),
                        gmt_last_reconciled=_utc_now(),
                        gmt_updated=_utc_now(),
                    )
                )
                claim_result = session.execute(claim_stmt)
                if claim_result.rowcount:
                    session.commit()
                    claimed_dispatches.append((summary.id, summary.collection_id, summary.version, token))
                else:
                    session.rollback()
                    logger.debug(
                        "Skipping summary %s — could not claim (concurrent claim or version drift)",
                        summary.id,
                    )
            return claimed_dispatches
        return claimed_dispatches

    dispatches = await asyncio.to_thread(_reclaim_stale_and_claim_pending)
    for summary_id, collection_id, target_version, processing_token in dispatches:
        # Pattern C fire-and-forget — task body has its own ownership re-check.
        asyncio.create_task(
            asyncio.to_thread(
                collection_summary_task,
                summary_id,
                collection_id,
                target_version,
                processing_token,
            )
        )
    if dispatches:
        logger.info("collection-summary reconciler dispatched=%d", len(dispatches))


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

    return ParseDispatchPayload(
        document_id=document.id,
        collection_id=document.collection_id or "",
        object_path=str(object_path),
        tenant_scope_key=f"user:{document.user}",
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

    The Pattern B ``reconcile_collection_summaries_hook`` runs after
    the three index scans; a hook failure is logged but never crashes
    the loop.
    """
    while not shutdown.is_set():
        try:
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
            if pushed or retried or reclaimed or stuck_reenqueued:
                logger.info(
                    "reconciler cycle: dispatched=%d retried=%d reclaimed=%d stuck_reenqueued=%d",
                    pushed,
                    retried,
                    reclaimed,
                    stuck_reenqueued,
                )
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            logger.exception("reconciler cycle failed: %s", exc)
        try:
            await reconcile_collection_summaries_hook()
        except Exception as exc:  # noqa: BLE001 — Pattern B hook never crashes loop
            logger.exception("reconcile_collection_summaries_hook failed: %s", exc)
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


__all__ = [
    "HEARTBEAT_STALE_SECONDS",
    "RECONCILE_BATCH_SIZE",
    "RECONCILE_INTERVAL_SECONDS",
    "STUCK_PARSE_COOLDOWN_SECONDS",
    "reconcile_collection_summaries_hook",
    "reconcile_failed_retry",
    "reconcile_pending_dispatch",
    "reconcile_running_reclaim",
    "reconcile_stuck_documents_for_parse_reenqueue",
    "run_reconcile_loop",
]
