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
and performs four DB scans:

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
4. **Per-modality cutover trigger** — for every row that just went
   ACTIVE and is not yet ``is_serving=TRUE``, run the §F.3
   three-statement cutover transaction (UPDATE status=ACTIVE → demote
   prior serving row → promote new row). Per-modality, not
   document-wide (§F.6 — each modality flips independently).

The four scans are intentionally idempotent: re-running a cycle
mid-flight produces the same end state, so a reconciler crash mid-
cycle is recoverable on next tick. The ``run_reconcile_loop`` wrapper
is the production entrypoint; the four ``reconcile_*`` functions are
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


# ---------------------------------------------------------------------
# (4) Per-modality cutover trigger
# ---------------------------------------------------------------------


def reconcile_cutover(
    *,
    engine: Engine,
    batch_size: int = RECONCILE_BATCH_SIZE,
) -> int:
    """Run §F.3 three-statement cutover for every ACTIVE-but-not-serving row.

    For each candidate row, in a single transaction:

    1. UPDATE prior serving row for ``(document_id, modality)`` to
       ``is_serving=FALSE`` (the demote step).
    2. UPDATE this row to ``is_serving=TRUE`` (the promote step).

    The §F.1 partial unique index ``uniq_document_index_v2_serving``
    guarantees these two statements never leave two rows
    ``is_serving=TRUE`` for the same ``(document_id, modality)`` even
    under concurrent reconcilers — the second transaction would
    detect the conflict and abort.

    The full §F.3 contract calls for THREE statements (status→ACTIVE
    is the first); here the orchestrator already wrote status=ACTIVE
    on completion, so the reconciler runs the remaining two statements
    in the same transaction, which is semantically equivalent.

    Returns the number of rows promoted to ``is_serving=TRUE``.
    """
    with Session(engine) as session, session.begin():
        candidates = list(
            session.scalars(
                select(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.status == IndexStatus.ACTIVE.value,
                        DocumentIndex.is_serving.is_(False),
                    )
                )
                .order_by(DocumentIndex.updated_at)
                .limit(batch_size)
            )
        )
        promoted = 0
        for row in candidates:
            # Demote any prior serving row for the same (doc, modality).
            # Distinct from this row's id so we don't accidentally
            # demote-then-promote the same row in a no-op cycle.
            session.execute(
                update(DocumentIndex)
                .where(
                    and_(
                        DocumentIndex.document_id == row.document_id,
                        DocumentIndex.modality == row.modality,
                        DocumentIndex.is_serving.is_(True),
                        DocumentIndex.id != row.id,
                    )
                )
                .values(is_serving=False)
            )
            # Promote this row.
            session.execute(update(DocumentIndex).where(DocumentIndex.id == row.id).values(is_serving=True))
            promoted += 1
    return promoted


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
    """Run the four reconcile scans every ``interval_seconds`` until shutdown.

    Each cycle is best-effort: an exception in any of the four scans
    is logged and the cycle continues to the next scan. A cycle that
    bombs entirely (e.g. DB unreachable) sleeps the interval and
    retries — better to keep the loop alive than to crash the process.
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
            promoted = await asyncio.to_thread(reconcile_cutover, engine=engine)
            if pushed or retried or reclaimed or promoted:
                logger.info(
                    "reconciler cycle: dispatched=%d retried=%d reclaimed=%d promoted=%d",
                    pushed,
                    retried,
                    reclaimed,
                    promoted,
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
    "reconcile_cutover",
    "reconcile_failed_retry",
    "reconcile_pending_dispatch",
    "reconcile_running_reclaim",
    "run_reconcile_loop",
]
