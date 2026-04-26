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

Per ``docs/modularization/indexing-redesign-design-pack.md`` §F.5, the
cleanup worker has TWO trigger paths with different semantics for the
graph modality (per architect ruling msg=492315e8 Ruling 3):

(A) **Orphan parse_version GC** — :func:`cleanup_orphan_parse_versions`,
    runs every :data:`CLEANUP_INTERVAL_SECONDS`. A row is orphan if all of:

    - ``is_serving = FALSE``
    - a newer ``parse_version`` exists for the same
      ``(document_id, modality)`` (this triple was superseded)
    - ``updated_at < now() - 1 hour`` (cool-down so cutover races
      resolve before we delete)

    For non-graph modalities (vector / fulltext / summary / vision):
    call the backend's flat delete (``delete_by_filter`` /
    ``delete_by_query``) to remove the search-time tombstone, then
    drop the ``document_index_v2`` row.

    For the graph modality: the backend lineage was *already* cleaned
    by the §D.3.6 sync supersede semantic when the new parse_version
    was written (per design pack §D.3.2 amended canonical, head
    ``a0a47994`` — sync removes ALL lineage members for the document,
    not just the new parse_version's). So orphan parse_version GC is
    a backend-level no-op for graph; we still drop the DB row to
    shrink the index.

(B) **Document deletion** — :func:`cleanup_for_deleted_documents`,
    invoked by callers when a document is removed (e.g. user delete).
    For non-graph modalities: same flat delete per ``(document_id,
    parse_version)`` for every row of that document; then drop the
    rows. For the graph modality: invoke the lineage cleanup path on
    the worker's underlying ``LineageGraphStore`` to remove every
    ``LineageMember`` referencing the document — entities go orphan
    → garbage-collected (per §D.3 lineage model). Each entity is
    serialized through its :class:`EntityLock` so a concurrent graph
    sync cannot race the cleanup.

The two entry points share :func:`_delete_document_index_rows` and
the orchestrator's ``Modality`` registry; callers wire whichever fits
their lifecycle (orphan GC = scheduled loop, document deletion = on
user-initiated delete).
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


def _flat_backend_delete_callable(worker: ModalityWorker):
    """Return ``delete(*, document_id, parse_version) -> None`` or ``None``.

    Walks the worker's exposed backend attribute for a
    ``delete_by_filter`` (vector, summary, vision) or
    ``delete_by_query`` (fulltext) method — the four §D.1 flat-delete
    Wave 1 modalities. Graph workers expose ``_store`` (a
    :class:`LineageGraphStore`) without either method; cleanup must
    take the lineage-aware path instead, which this function signals
    by returning ``None``.
    """
    backend = getattr(worker, "_backend", None)
    if backend is None:
        return None
    for name in ("delete_by_filter", "delete_by_query"):
        fn = getattr(backend, name, None)
        if callable(fn):
            return fn
    return None


def _is_graph_worker(worker: ModalityWorker) -> bool:
    """Detect the graph modality without importing GraphModalityWorker.

    Imports of ``aperag.indexing.graph`` from cleanup would create a
    cleanup → graph import dependency that conflicts with graph's own
    optional Nebula / Neo4j extras (graph imports those lazily). We
    duck-type instead: only the graph worker exposes ``_store`` (a
    LineageGraphStore) AND ``_entity_lock``.
    """
    return getattr(worker, "modality", None) is Modality.GRAPH


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
    """Garbage-collect every orphan triple visible right now (path A).

    ``workers`` is the per-modality registry the orchestrator already
    uses; cleanup looks up each row's modality to find the worker.
    Returns a dict ``{"backend_deleted": N, "rows_deleted": N,
    "graph_noop": N, "backend_skipped": N}`` for telemetry / tests.

    **Graph behaviour** (per architect ruling msg=492315e8 Ruling 3):
    the §D.3.6 sync supersede semantic already removed the old
    parse_version's lineage members when the new parse_version was
    written (per amended canonical §D.3.2 — sync clears lineage by
    document_id, not by parse_version). So orphan parse_version GC is
    a backend-level no-op for graph; we still drop the DB row to
    shrink the index. Tracked under ``graph_noop`` for visibility.
    """
    rows = await asyncio.to_thread(
        find_orphan_parse_versions,
        engine=engine,
        cooldown_seconds=cooldown_seconds,
        batch_size=batch_size,
    )
    counts = {
        "backend_deleted": 0,
        "rows_deleted": 0,
        "graph_noop": 0,
        "backend_skipped": 0,
    }

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
        elif _is_graph_worker(worker):
            # Graph backend lineage was already cleared by sync's
            # supersede semantic — see §D.3.2 amended canonical (PR
            # #1725 head a0a47994). The orphan parse_version GC path
            # is a backend no-op for graph; still drop the DB row.
            logger.debug(
                "cleanup graph orphan parse_version row id=%d — backend no-op (sync supersede already cleared per §D.3.2 amended)",
                row.id,
            )
            counts["graph_noop"] += 1
        else:
            delete_fn = _flat_backend_delete_callable(worker)
            if delete_fn is None:
                logger.warning(
                    "cleanup modality=%s exposes no delete_by_filter/query — skipping backend delete for row id=%d",
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


# ---------------------------------------------------------------------
# (3) Document-deletion cleanup — path B (caller-driven).
# ---------------------------------------------------------------------


async def cleanup_for_deleted_documents(
    *,
    engine: Engine,
    workers: Mapping[Modality, ModalityWorker],
    document_ids: list[str],
) -> dict[str, int]:
    """Garbage-collect every triple for the given deleted documents (path B).

    Caller-driven: the upstream document-delete handler passes the
    document IDs that should be GC'd. For each row of every document:

    - **Non-graph modalities** (vector / fulltext / summary / vision)
      — call the backend's flat ``delete_by_filter`` /
      ``delete_by_query`` per ``(document_id, parse_version)``.
    - **Graph modality** — call the worker's underlying lineage-aware
      delete via :class:`LineageGraphStore` per architect ruling
      msg=492315e8 Ruling 3. Each entity is removed under its
      :class:`EntityLock` so a concurrent graph sync cannot race.

    All ``document_index_v2`` rows for the requested documents are
    dropped at the end (one batched DELETE).

    Returns ``{"backend_deleted": N, "graph_lineage_cleaned": N,
    "rows_deleted": N, "backend_skipped": N}``. ``graph_lineage_cleaned``
    counts documents (not rows) since one document's graph cleanup
    covers all parse_versions in one call.
    """
    if not document_ids:
        return {
            "backend_deleted": 0,
            "graph_lineage_cleaned": 0,
            "rows_deleted": 0,
            "backend_skipped": 0,
        }

    rows = await asyncio.to_thread(_select_rows_for_documents, engine, document_ids)
    counts = {
        "backend_deleted": 0,
        "graph_lineage_cleaned": 0,
        "rows_deleted": 0,
        "backend_skipped": 0,
    }

    # Per-document, per-modality dedup so graph lineage cleanup runs
    # at most once per (document, graph) regardless of how many
    # parse_versions the document has.
    graph_done: set[str] = set()
    delete_ids: list[int] = []

    for row in rows:
        try:
            modality = Modality(row.modality)
        except ValueError:
            logger.error(
                "cleanup unknown modality %r on row id=%d (document=%s) — skipping",
                row.modality,
                row.id,
                row.document_id,
            )
            continue

        worker = workers.get(modality)
        if worker is None:
            logger.warning(
                "cleanup no worker for modality=%s row id=%d document=%s — skipping backend",
                row.modality,
                row.id,
                row.document_id,
            )
            counts["backend_skipped"] += 1
        elif _is_graph_worker(worker):
            if row.document_id not in graph_done:
                try:
                    await _cleanup_graph_lineage_for_document(worker, row.document_id)
                    counts["graph_lineage_cleaned"] += 1
                    graph_done.add(row.document_id)
                except Exception as exc:  # noqa: BLE001 — log + leave row for next cycle
                    logger.exception(
                        "cleanup graph lineage failed document=%s row id=%d: %s",
                        row.document_id,
                        row.id,
                        exc,
                    )
                    continue
            # graph lineage cleanup is per-document; multiple
            # parse_version rows for the same doc share one call
            # but we still queue each row id for DB deletion.
        else:
            delete_fn = _flat_backend_delete_callable(worker)
            if delete_fn is None:
                logger.warning(
                    "cleanup modality=%s exposes no delete_by_filter/query — skipping backend for row id=%d",
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


def _select_rows_for_documents(engine: Engine, document_ids: list[str]) -> list[DocumentIndex]:
    with Session(engine) as session:
        return list(session.scalars(select(DocumentIndex).where(DocumentIndex.document_id.in_(document_ids))))


async def _cleanup_graph_lineage_for_document(worker: ModalityWorker, document_id: str) -> None:
    """Remove all graph lineage members for ``document_id``.

    Implements the §D.3 lineage cleanup at the storage layer. The
    graph worker exposes its :class:`LineageGraphStore` as ``_store``
    and its :class:`EntityLock` as ``_entity_lock`` — both are
    Wave 1 conventions used by the graph worker's own ``sync`` Phase
    1. We re-use them here instead of duplicating the cleanup loop
    inside ``GraphModalityWorker``, keeping graph.py untouched (per
    architect ruling msg=492315e8 Ruling 3 which has cleanup in this
    module).
    """
    store = getattr(worker, "_store", None)
    entity_lock = getattr(worker, "_entity_lock", None)
    if store is None or entity_lock is None:
        raise RuntimeError(
            "graph cleanup requires worker._store + worker._entity_lock (Wave 1 GraphModalityWorker convention)"
        )

    # Phase A — entities. Per §D.3.2 amended canonical (PR #1725 head
    # a0a47994), lineage cleanup is by document_id only; parse_version
    # is not needed because deletion supersedes ALL parse versions.
    entity_names = await store.find_entity_ids_with_lineage(document_id=document_id)
    for entity_name in entity_names:
        async with entity_lock.acquire(entity_name):
            await store.remove_entity_lineage_member(
                entity_name=entity_name,
                document_id=document_id,
            )
            # Hook for stores that GC entities once their lineage set
            # is empty. Wave 1 InMemoryLineageGraphStore + the Nebula
            # impl both expose this; missing method = leave entity
            # row in place (still valid graph state, just orphan).
            gc = getattr(store, "gc_entity_if_orphan", None)
            if callable(gc):
                await gc(entity_name=entity_name)

    # Phase B — relations. Same shape; graph workers cleanup their
    # relations the same way (find → remove member → optional GC).
    find_relations = getattr(store, "find_relation_keys_with_lineage", None)
    remove_relation = getattr(store, "remove_relation_lineage_member", None)
    if callable(find_relations) and callable(remove_relation):
        relation_keys = await find_relations(document_id=document_id)
        for relation_key in relation_keys:
            await remove_relation(
                relation_key=relation_key,
                document_id=document_id,
            )
            gc_rel = getattr(store, "gc_relation_if_orphan", None)
            if callable(gc_rel):
                await gc_rel(relation_key=relation_key)


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
    "cleanup_for_deleted_documents",
    "cleanup_orphan_parse_versions",
    "find_orphan_parse_versions",
    "run_cleanup_loop",
]


# Suppress unused-import warning for type-only narrowing.
_: Any = None
