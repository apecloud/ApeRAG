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

"""Cleanup worker — celery T2.1.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §F.5, a
single asyncio process runs every :data:`CLEANUP_INTERVAL_SECONDS`
and garbage-collects orphan ``(document_id, parse_version, modality)``
triples. A row is an orphan if all of:

- ``is_serving = FALSE``
- a *newer* ``parse_version`` exists for the same
  ``(document_id, modality)`` (i.e. this triple was superseded)
- ``updated_at < now() - 1 hour`` (cool-down so cutover races resolve
  before we delete)

For each orphan the cleanup worker:

1. Calls the modality's backend delete (via duck-typed
   ``delete_by_filter`` / ``delete_by_query`` on the backend or store)
   to remove the search-time tombstone.
2. Deletes the ``document_index_v2`` row itself.

The graph modality is the documented exception (§D.3): its backend
state is co-mingled with other documents' lineage members, so a flat
delete is unsafe. The graph cleanup hook is intentionally a no-op
warning in T2.1 — Bryce's T2.2 work extends ``GraphModalityWorker``
with a lineage-aware ``cleanup_backend_state`` (§D.3 lineage cleanup).

The cleanup-by-document-deletion path (``document.deleted_at`` set)
is the §F.5 second half: any rows for a tombstoned document have
``is_serving = FALSE`` set on demand, then the same orphan-GC pass
above sweeps the rows. T2.1 handles only the parse_version-supersede
path; the document-tombstone scan is left for the cutover lane (T2.2)
because it touches the ``document`` table outside the indexing
domain's write set.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import Engine, and_, delete, func, select
from sqlalchemy.orm import Session

from aperag.indexing.base import ModalityWorker
from aperag.indexing.models import DocumentIndex, Modality

logger = logging.getLogger(__name__)


# §F.5 cleanup cycle interval. Production runs every 5 minutes;
# tests call ``cleanup_orphan_parse_versions`` directly.
CLEANUP_INTERVAL_SECONDS = 300

# §F.5 cool-down between supersede and GC — gives the cutover swap
# (§F.3) time to land before the cleanup worker starts deleting
# backend state. Conservative; in practice cutover lands within
# seconds, so this is a safety margin for slow runs / transient DB
# outages.
ORPHAN_COOLDOWN_SECONDS = 3600

CLEANUP_BATCH_SIZE = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------
# Backend cleanup dispatch — duck-typed across the 4 Wave 1 backends.
# ---------------------------------------------------------------------


def _backend_delete_callable(worker: ModalityWorker):
    """Return ``delete(*, document_id, parse_version) -> None`` or ``None``.

    Walks the worker's exposed backend / store attribute and looks
    for a ``delete_by_filter`` (vector, summary, vision) or
    ``delete_by_query`` (fulltext) method. The graph worker's
    ``_store`` does NOT expose either — graph cleanup is the
    documented T2.1 no-op (deferred to T2.2 §D.3 lineage cleanup).

    Returning ``None`` signals "no clean delete path" and the
    cleanup worker logs + skips the row's backend delete (the row
    itself is still garbage-collected from ``document_index_v2``).
    """
    backend = getattr(worker, "_backend", None) or getattr(worker, "_store", None)
    if backend is None:
        return None
    for name in ("delete_by_filter", "delete_by_query"):
        fn = getattr(backend, name, None)
        if callable(fn):
            return fn
    return None


# ---------------------------------------------------------------------
# (1) Find orphan rows — superseded by a newer parse_version.
# ---------------------------------------------------------------------


def find_orphan_parse_versions(
    *,
    engine: Engine,
    cooldown_seconds: int = ORPHAN_COOLDOWN_SECONDS,
    batch_size: int = CLEANUP_BATCH_SIZE,
) -> list[DocumentIndex]:
    """Return rows whose ``parse_version`` was superseded > cooldown ago.

    A row is orphan iff a newer ``updated_at`` row exists for the
    same ``(document_id, modality)`` AND this row is not currently
    serving AND the row's ``updated_at`` is older than ``now() -
    cooldown_seconds``. Caller is responsible for invoking the
    backend delete + DB delete; this function is the read-only seam.
    """
    threshold = _utcnow() - timedelta(seconds=cooldown_seconds)
    with Session(engine) as session:
        # Subquery: max(updated_at) per (document_id, modality) — the
        # "latest parse_version slot" surrogate. Any row whose
        # updated_at < that max is a candidate orphan.
        latest = (
            select(
                DocumentIndex.document_id.label("did"),
                DocumentIndex.modality.label("mod"),
                func.max(DocumentIndex.updated_at).label("max_updated"),
            )
            .group_by(DocumentIndex.document_id, DocumentIndex.modality)
            .subquery()
        )
        stmt = (
            select(DocumentIndex)
            .join(
                latest,
                and_(
                    DocumentIndex.document_id == latest.c.did,
                    DocumentIndex.modality == latest.c.mod,
                ),
            )
            .where(
                and_(
                    DocumentIndex.is_serving.is_(False),
                    DocumentIndex.updated_at < latest.c.max_updated,
                    DocumentIndex.updated_at < threshold,
                )
            )
            .order_by(DocumentIndex.updated_at)
            .limit(batch_size)
        )
        return list(session.scalars(stmt))


# ---------------------------------------------------------------------
# (2) Cleanup execution — call backend delete + drop the row.
# ---------------------------------------------------------------------


async def cleanup_orphan_parse_versions(
    *,
    engine: Engine,
    workers: Mapping[Modality, ModalityWorker],
    cooldown_seconds: int = ORPHAN_COOLDOWN_SECONDS,
    batch_size: int = CLEANUP_BATCH_SIZE,
) -> dict[str, int]:
    """Garbage-collect every orphan triple visible right now.

    ``workers`` is the per-modality registry the orchestrator already
    uses; cleanup looks up each row's modality to find the worker and
    call its backend delete. Returns a dict ``{"backend_deleted": N,
    "rows_deleted": N, "backend_skipped": N}`` for telemetry / tests.
    """
    rows = await asyncio.to_thread(
        find_orphan_parse_versions,
        engine=engine,
        cooldown_seconds=cooldown_seconds,
        batch_size=batch_size,
    )
    counts = {"backend_deleted": 0, "rows_deleted": 0, "backend_skipped": 0}

    delete_ids: list[int] = []
    for row in rows:
        try:
            modality = Modality(row.modality)
        except ValueError:
            logger.error(
                "cleanup unknown modality %r on row id=%d — skipping",
                row.modality,
                row.id,
            )
            continue

        worker = workers.get(modality)
        if worker is None:
            logger.warning(
                "cleanup no worker registered for modality=%s row id=%d — skipping backend delete",
                row.modality,
                row.id,
            )
            counts["backend_skipped"] += 1
        else:
            delete_fn = _backend_delete_callable(worker)
            if delete_fn is None:
                logger.warning(
                    "cleanup modality=%s exposes no delete_by_filter/query — skipping backend delete for row id=%d (T2.2 graph follow-up)",
                    row.modality,
                    row.id,
                )
                counts["backend_skipped"] += 1
            else:
                try:
                    await asyncio.to_thread(
                        delete_fn,
                        document_id=row.document_id,
                        parse_version=row.parse_version,
                    )
                    counts["backend_deleted"] += 1
                except Exception as exc:  # noqa: BLE001 — log + leave row for next cycle
                    logger.exception(
                        "cleanup backend delete failed modality=%s row id=%d: %s",
                        row.modality,
                        row.id,
                        exc,
                    )
                    continue

        delete_ids.append(row.id)

    if delete_ids:
        await asyncio.to_thread(_delete_rows, engine, delete_ids)
        counts["rows_deleted"] = len(delete_ids)

    return counts


def _delete_rows(engine: Engine, ids: list[int]) -> None:
    with Session(engine) as session, session.begin():
        session.execute(delete(DocumentIndex).where(DocumentIndex.id.in_(ids)))


# ---------------------------------------------------------------------
# Run loop — production entrypoint.
# ---------------------------------------------------------------------


async def run_cleanup_loop(
    *,
    engine: Engine,
    workers: Mapping[Modality, ModalityWorker],
    shutdown: asyncio.Event,
    interval_seconds: int = CLEANUP_INTERVAL_SECONDS,
    cooldown_seconds: int = ORPHAN_COOLDOWN_SECONDS,
) -> None:
    """Run :func:`cleanup_orphan_parse_versions` every ``interval_seconds``.

    A cycle that throws is logged and the loop continues — DB
    unreachable / Redis blip should not crash the cleanup process.
    """
    while not shutdown.is_set():
        try:
            counts = await cleanup_orphan_parse_versions(
                engine=engine,
                workers=workers,
                cooldown_seconds=cooldown_seconds,
            )
            if any(counts.values()):
                logger.info(
                    "cleanup cycle: backend_deleted=%d rows_deleted=%d backend_skipped=%d",
                    counts["backend_deleted"],
                    counts["rows_deleted"],
                    counts["backend_skipped"],
                )
        except Exception as exc:  # noqa: BLE001 — keep loop alive
            logger.exception("cleanup cycle failed: %s", exc)
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


__all__ = [
    "CLEANUP_BATCH_SIZE",
    "CLEANUP_INTERVAL_SECONDS",
    "ORPHAN_COOLDOWN_SECONDS",
    "cleanup_orphan_parse_versions",
    "find_orphan_parse_versions",
    "run_cleanup_loop",
]


# Suppress unused-import warning for type-only narrowing.
_: Any = None
