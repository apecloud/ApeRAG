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

"""Cleanup worker — celery T2.1 (extended in T3.1 with path C).

Per ``docs/modularization/indexing-redesign-design-pack.md`` §F.5, the
cleanup worker has THREE trigger paths with different semantics for
the graph modality (per architect ruling msg=492315e8 Ruling 3 +
msg=3890c9d7 Pattern A):

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

(C) **Collection deletion cascade** —
    :func:`cleanup_for_deleted_collections`, the T3.1 path C added
    per architect msg=3890c9d7 Pattern A. Invoked by the
    Pattern-A-synchronous HTTP handler for the ``DELETE /collection``
    endpoint AND by the periodic Pattern-B reconciler scan that
    sweeps tombstoned collections (``WHERE Collection.deleted_at IS
    NOT NULL``). For each deleted collection:

    1. Find all distinct ``document_id`` values whose
       ``document_index`` rows reference that collection.
    2. Cascade to path B (:func:`cleanup_for_deleted_documents`) for
       those documents — that path already handles modality fan-out
       (graph lineage cleanup vs flat backend delete).
    3. Sweep any remaining ``document_index`` rows for the
       collection (covers the edge case where a document had no
       indexed modalities yet but the row was created before delete).

    Path C is idempotent: a partial cascade that crashes mid-way is
    resumed on the next scan because the per-row state machine still
    leaves the un-GC'd rows discoverable by collection_id.

The three entry points share the same :func:`_delete_rows` helper
and the orchestrator's ``Modality`` registry; callers wire whichever
fits their lifecycle (orphan GC = scheduled loop, document deletion
= on user-initiated delete, collection deletion = on user-initiated
collection delete + reconciler sweep).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional

from sqlalchemy import Engine, and_, delete, func, or_, select
from sqlalchemy.orm import Session

from aperag.indexing.base import ModalityWorker
from aperag.indexing.models import DocumentIndex, Modality
from aperag.indexing.worker_factory import WorkerFactoryError

logger = logging.getLogger(__name__)


# Wave 4 T2: per-row worker factory shape. Production wires
# :meth:`aperag.indexing.worker_factory.ProductionWorkerFactory.build_for_cleanup_row`;
# tests inject a closure that returns the right cleanup view per
# ``(row.collection_id, row.modality)`` against InMemory backends.
# A factory may raise :class:`WorkerFactoryError` for a row whose
# backend is intentionally gated (Wave 4 #9 vision multimodal /
# Wave 4 #2 graph extractor): the cleanup loop catches that and
# counts the row as ``backend_skipped`` while still dropping the DB
# row so the index does not grow unboundedly.
WorkerFactoryForRow = Callable[[DocumentIndex], Awaitable[ModalityWorker]]


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

    任务 #5: 老 ``GRAPH`` 单段拆成 ``GRAPH_FACTS`` + ``GRAPH_VECTORS``.
    三个值都映射到同一个 lineage store, 所以 cleanup 也都走 graph 路径.
    """
    return getattr(worker, "modality", None) in {
        Modality.GRAPH,
        Modality.GRAPH_FACTS,
        Modality.GRAPH_VECTORS,
    }


@dataclass(frozen=True)
class CleanupWorkerResolution:
    """Wave 5 P4 T2 — outcome of :func:`_resolve_cleanup_worker`.

    Distinguishes the two failure modes that pre-Wave-5 collapsed into
    a single ``None`` return:

    * **intentional gate** (``worker is None`` AND ``transient is False``)
      — :class:`WorkerFactoryError` raised because the modality is
      Wave-N-gated by design (graph extractor not wired / vision
      multimodal not configured), the operator deliberately disabled
      the modality via ``collection.config``, or the row's modality
      string is unknown. The cleanup loop should drop the DB row so
      the index does not grow unboundedly while the gate is active.

    * **transient infrastructure error** (``worker is None`` AND
      ``transient is True``) — DB connection blip / Qdrant
      unreachable / ES unhealthy / Redis network glitch. The cleanup
      loop must NOT drop the DB row — the next cycle (5 min later)
      retries automatically once the backend recovers. Pre-Wave-5
      this collapsed into the gate path and silently lost the retry
      signal.

    * **resolved worker** (``worker is not None``, ``transient ignored``)
      — happy path; caller proceeds with backend cleanup.
    """

    worker: Optional[ModalityWorker]
    transient: bool


@dataclass(frozen=True)
class DeletedDocumentCleanupTarget:
    """Durable DB-backed cleanup intent for task #17.

    API delete only tombstones the ``Document`` row. The worker cleanup
    loop reconstructs the object-store prefix from the DB row and uses
    remaining ``DocumentIndex`` rows as the retry signal for backend
    cleanup.
    """

    document_id: str
    object_store_prefix: str


async def _resolve_cleanup_worker(
    *,
    workers: Optional[Mapping[Modality, ModalityWorker]],
    worker_factory: Optional[WorkerFactoryForRow],
    row: DocumentIndex,
) -> CleanupWorkerResolution:
    """Resolve the cleanup worker for a row from factory or static map.

    Wave 4 T2 + Wave 5 P4 (transient-vs-intentional split): production
    wires the factory (per-(collection, modality) lazy materialisation);
    tests typically pass a pre-built ``workers`` mapping. When both
    are provided the factory wins — production deployments override
    the legacy mapping with the per-row factory.

    Returns a :class:`CleanupWorkerResolution`:

    * ``worker is not None``: backend cleanup proceeds.
    * ``worker is None, transient=False``: intentional gate / unknown
      modality / no source — caller drops DB row.
    * ``worker is None, transient=True``: transient infrastructure
      error — caller skips DB row drop so next cycle retries.

    Pre-Wave-5 this returned ``None`` for both failure modes,
    causing transient errors to silently lose their retry signal.
    """
    if worker_factory is not None:
        try:
            return CleanupWorkerResolution(worker=await worker_factory(row), transient=False)
        except WorkerFactoryError as exc:
            logger.warning(
                "cleanup worker_factory gate raised modality=%s row id=%d collection=%s: %s — "
                "counting as backend_skipped (intentional gate, dropping DB row)",
                row.modality,
                row.id,
                row.collection_id,
                exc,
            )
            return CleanupWorkerResolution(worker=None, transient=False)
        except Exception as exc:  # noqa: BLE001 — transient infra error, retry next cycle
            logger.warning(
                "cleanup worker_factory transient failure modality=%s row id=%d collection=%s: %s — "
                "skipping DB row drop, will retry next cycle",
                row.modality,
                row.id,
                row.collection_id,
                exc,
            )
            return CleanupWorkerResolution(worker=None, transient=True)
    if workers is None:
        return CleanupWorkerResolution(worker=None, transient=False)
    try:
        modality = Modality(row.modality)
    except ValueError:
        return CleanupWorkerResolution(worker=None, transient=False)
    return CleanupWorkerResolution(worker=workers.get(modality), transient=False)


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
    workers: Optional[Mapping[Modality, ModalityWorker]] = None,
    worker_factory: Optional[WorkerFactoryForRow] = None,
    cooldown_seconds: int = ORPHAN_COOLDOWN_SECONDS,
    batch_size: int = CLEANUP_BATCH_SIZE,
) -> dict[str, int]:
    """Garbage-collect every orphan triple visible right now (path A).

    Wave 4 T2: ``worker_factory`` is the per-row lazy resolver
    production lifespan installs; ``workers`` is the legacy per-modality
    static map kept for backward-compat with existing tests. When both
    are passed the factory wins. Returns a dict ``{"backend_deleted":
    N, "rows_deleted": N, "graph_noop": N, "backend_skipped": N}``
    for telemetry / tests.

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
        "transient_deferred": 0,
    }

    delete_ids: list[int] = []
    for row in rows:
        try:
            Modality(row.modality)
        except ValueError:
            logger.error(
                "cleanup unknown modality %r on row id=%d — skipping",
                row.modality,
                row.id,
            )
            continue

        resolution = await _resolve_cleanup_worker(
            workers=workers,
            worker_factory=worker_factory,
            row=row,
        )
        if resolution.transient:
            # Wave 5 P4: transient infra error — skip both backend
            # delete AND DB row drop so the next cleanup cycle (5 min
            # later) retries automatically once the backend recovers.
            counts["transient_deferred"] += 1
            continue
        worker = resolution.worker
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
    workers: Optional[Mapping[Modality, ModalityWorker]] = None,
    worker_factory: Optional[WorkerFactoryForRow] = None,
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

    Wave 4 T2: ``worker_factory`` resolves the worker per-row from the
    persisted ``DocumentIndex`` (collection_id + modality); ``workers``
    is the legacy static map kept for tests. When both are passed the
    factory wins.

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
        "transient_deferred": 0,
    }

    # Per-document, per-modality dedup so graph lineage cleanup runs
    # at most once per (document, graph) regardless of how many
    # parse_versions the document has.
    graph_done: set[str] = set()
    delete_ids: list[int] = []

    for row in rows:
        try:
            Modality(row.modality)
        except ValueError:
            logger.error(
                "cleanup unknown modality %r on row id=%d (document=%s) — skipping",
                row.modality,
                row.id,
                row.document_id,
            )
            continue

        resolution = await _resolve_cleanup_worker(
            workers=workers,
            worker_factory=worker_factory,
            row=row,
        )
        if resolution.transient:
            # Wave 5 P4: transient infra error — skip backend delete
            # AND DB row drop so the caller can retry on a later cycle
            # / re-invocation once the backend recovers.
            counts["transient_deferred"] += 1
            continue
        worker = resolution.worker
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


def find_deleted_document_cleanup_targets(
    *,
    engine: Engine,
    batch_size: int = CLEANUP_BATCH_SIZE,
) -> list[DeletedDocumentCleanupTarget]:
    """Return deleted documents that still have index rows to clean.

    Task #17 moves heavy document delete cleanup out of the API request
    path. The durable intent source is the DB:

    - ``Document.status = DELETED`` or ``Document.gmt_deleted IS NOT NULL``
    - at least one remaining ``DocumentIndex`` row for that document

    Redis cleanup wakeups, if any are added later, are only transport.
    This scan must be sufficient on its own after Redis loss.
    """
    from aperag.domains.knowledge_base.db.models import Document, DocumentStatus

    with Session(engine) as session:
        indexed_documents = (
            select(
                DocumentIndex.document_id.label("document_id"),
                func.min(DocumentIndex.updated_at).label("oldest_index_updated_at"),
            )
            .group_by(DocumentIndex.document_id)
            .subquery()
        )
        stmt = (
            select(Document)
            .join(indexed_documents, indexed_documents.c.document_id == Document.id)
            .where(
                or_(
                    Document.status == DocumentStatus.DELETED,
                    Document.gmt_deleted.is_not(None),
                )
            )
            .order_by(indexed_documents.c.oldest_index_updated_at, Document.id)
            .limit(batch_size)
        )
        return [
            DeletedDocumentCleanupTarget(
                document_id=document.id,
                object_store_prefix=document.object_store_base_path(),
            )
            for document in session.scalars(stmt)
        ]


async def cleanup_deleted_document_intents(
    *,
    engine: Engine,
    workers: Optional[Mapping[Modality, ModalityWorker]] = None,
    worker_factory: Optional[WorkerFactoryForRow] = None,
    batch_size: int = CLEANUP_BATCH_SIZE,
    object_store: Any | None = None,
) -> dict[str, int]:
    """Cleanup tombstoned documents discovered from DB state.

    This is task #17's worker-owned replacement for the previous API
    request-path cleanup. Object-store prefix deletion runs first; if it
    fails, the ``DocumentIndex`` rows are intentionally left in place so
    the next cleanup cycle can retry from the same DB intent.
    """
    targets = await asyncio.to_thread(
        find_deleted_document_cleanup_targets,
        engine=engine,
        batch_size=batch_size,
    )
    counts = {
        "documents_seen": len(targets),
        "object_store_deleted": 0,
        "object_store_deferred": 0,
        "backend_deleted": 0,
        "graph_lineage_cleaned": 0,
        "rows_deleted": 0,
        "backend_skipped": 0,
        "transient_deferred": 0,
    }
    if not targets:
        return counts

    if object_store is None:
        from aperag.objectstore.base import get_object_store

        object_store = await asyncio.to_thread(get_object_store)

    ready_document_ids: list[str] = []
    for target in targets:
        try:
            await asyncio.to_thread(object_store.delete_objects_by_prefix, target.object_store_prefix)
            counts["object_store_deleted"] += 1
            ready_document_ids.append(target.document_id)
        except Exception as exc:  # noqa: BLE001 — leave rows for retry
            logger.exception(
                "cleanup object-store prefix delete failed document=%s prefix=%s: %s",
                target.document_id,
                target.object_store_prefix,
                exc,
            )
            counts["object_store_deferred"] += 1

    if not ready_document_ids:
        return counts

    sub_counts = await cleanup_for_deleted_documents(
        engine=engine,
        workers=workers,
        worker_factory=worker_factory,
        document_ids=ready_document_ids,
    )
    for key in (
        "backend_deleted",
        "graph_lineage_cleaned",
        "rows_deleted",
        "backend_skipped",
        "transient_deferred",
    ):
        counts[key] += sub_counts[key]
    return counts


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
# (4) Collection-deletion cascade — path C (T3.1 architect msg=3890c9d7).
# ---------------------------------------------------------------------


async def cleanup_for_deleted_collections(
    *,
    engine: Engine,
    workers: Optional[Mapping[Modality, ModalityWorker]] = None,
    worker_factory: Optional[WorkerFactoryForRow] = None,
    collection_ids: list[str],
) -> dict[str, int]:
    """Cascade-cleanup every triple for the given deleted collections (path C).

    Caller-driven, invoked by both:

    - The Pattern-A synchronous HTTP handler for ``DELETE /collection``
      (must be synchronous because Celery is gone in Wave 3 and a
      collection-delete failure mid-cascade would leave orphan
      ``document_index`` rows + orphan source/derived storage —
      ``asyncio.create_task()`` is unsafe here per architect ruling
      msg=3890c9d7).

    - A periodic Pattern-B reconciler scan that sweeps tombstoned
      collections (e.g. ``WHERE Collection.deleted_at IS NOT NULL``)
      so a Pattern-A crash mid-cascade is recovered on the next loop.

    For each collection_id:

    1. Find all distinct ``document_id`` values in
       ``document_index`` rows referencing it.
    2. Cascade to :func:`cleanup_for_deleted_documents` (path B) —
       that path already handles modality fan-out (graph lineage
       cleanup vs flat backend delete).
    3. Sweep any remaining ``document_index`` rows for the
       collection. Covers the edge case where a row was created with
       a collection_id but no document_id ever got indexed (or
       all parse_versions were already orphan-GC'd by path A).

    Returns a counts dict with the path-B keys plus
    ``"collections_cleaned": len(collection_ids)``.

    Idempotent: a partial cascade that crashes mid-way is resumed on
    the next call because the per-row state machine still leaves
    un-GC'd rows discoverable by ``collection_id``.
    """
    counts = {
        "backend_deleted": 0,
        "graph_lineage_cleaned": 0,
        "rows_deleted": 0,
        "backend_skipped": 0,
        "transient_deferred": 0,
        "collections_cleaned": 0,
    }
    if not collection_ids:
        return counts

    document_ids = await asyncio.to_thread(
        _select_distinct_document_ids_for_collections,
        engine,
        collection_ids,
    )

    if document_ids:
        sub_counts = await cleanup_for_deleted_documents(
            engine=engine,
            workers=workers,
            worker_factory=worker_factory,
            document_ids=document_ids,
        )
        for key in (
            "backend_deleted",
            "graph_lineage_cleaned",
            "rows_deleted",
            "backend_skipped",
            "transient_deferred",
        ):
            counts[key] += sub_counts[key]

    # Sweep any rows that path B did not catch (no document_id match
    # because the row was orphaned earlier or the collection had
    # rows queued before any document made it past PENDING).
    extras = await asyncio.to_thread(_delete_rows_for_collections, engine, collection_ids)
    counts["rows_deleted"] += extras
    counts["collections_cleaned"] = len(collection_ids)
    return counts


def _select_distinct_document_ids_for_collections(engine: Engine, collection_ids: list[str]) -> list[str]:
    with Session(engine) as session:
        rows = session.scalars(
            select(DocumentIndex.document_id).where(DocumentIndex.collection_id.in_(collection_ids)).distinct()
        )
        return list(rows)


def _delete_rows_for_collections(engine: Engine, collection_ids: list[str]) -> int:
    with Session(engine) as session, session.begin():
        result = session.execute(delete(DocumentIndex).where(DocumentIndex.collection_id.in_(collection_ids)))
        return result.rowcount or 0


# ---------------------------------------------------------------------
# Run loop — production entrypoint.
# ---------------------------------------------------------------------


async def run_cleanup_loop(
    *,
    engine: Engine,
    workers: Optional[Mapping[Modality, ModalityWorker]] = None,
    worker_factory: Optional[WorkerFactoryForRow] = None,
    shutdown: asyncio.Event,
    interval_seconds: int = CLEANUP_INTERVAL_SECONDS,
    cooldown_seconds: int = ORPHAN_COOLDOWN_SECONDS,
) -> None:
    """Run cleanup scans every ``interval_seconds``.

    Three scans per cycle (Wave 3 Pattern B integration per architect
    msg=3890c9d7):

    - :func:`cleanup_orphan_parse_versions` — orphan parse_v GC (path A)
    - :func:`cleanup_deleted_document_intents` — task #17 DB-backed
      document delete cleanup, including object-store prefix deletion
      and backend cleanup outside the API request path
    - :func:`cleanup_expired_documents_hook` — soft-delete documents
      stuck in UPLOADED status > 1 day (replaces legacy
      ``cleanup_expired_documents_task`` Celery beat schedule)

    Wave 4 T2: production lifespan injects ``worker_factory`` (per-row
    lazy resolver against the existing ``ProductionWorkerFactory``);
    tests typically inject a static ``workers`` map. When both are
    given the factory wins.

    A cycle that throws is logged and the loop continues — DB
    unreachable / Redis blip should not crash the cleanup process.
    """
    while not shutdown.is_set():
        try:
            counts = await cleanup_orphan_parse_versions(
                engine=engine,
                workers=workers,
                worker_factory=worker_factory,
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
            counts = await cleanup_deleted_document_intents(
                engine=engine,
                workers=workers,
                worker_factory=worker_factory,
            )
            if any(counts.values()):
                logger.info(
                    "cleanup deleted documents: documents_seen=%d object_store_deleted=%d "
                    "object_store_deferred=%d backend_deleted=%d graph_lineage_cleaned=%d "
                    "rows_deleted=%d backend_skipped=%d transient_deferred=%d",
                    counts["documents_seen"],
                    counts["object_store_deleted"],
                    counts["object_store_deferred"],
                    counts["backend_deleted"],
                    counts["graph_lineage_cleaned"],
                    counts["rows_deleted"],
                    counts["backend_skipped"],
                    counts["transient_deferred"],
                )
        except Exception as exc:  # noqa: BLE001 — keep loop alive
            logger.exception("cleanup deleted-document cycle failed: %s", exc)
        try:
            await cleanup_expired_documents_hook()
        except Exception as exc:  # noqa: BLE001 — Pattern B hook never crashes loop
            logger.exception("cleanup_expired_documents_hook failed: %s", exc)
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


async def cleanup_expired_documents_hook() -> None:
    """Pattern B periodic hook — Wave 3 architect msg=3890c9d7.

    Thin async wrapper over the legacy-equivalent
    ``aperag.domains.knowledge_base.tasks.cleanup_expired_documents_task``
    body (sync SQL tombstone scan). Imported lazily to avoid the
    circular ``cleanup → knowledge_base → cleanup`` dependency at
    module load time.
    """
    from aperag.domains.knowledge_base.tasks import cleanup_expired_documents_task

    await asyncio.to_thread(cleanup_expired_documents_task)


__all__ = [
    "CLEANUP_BATCH_SIZE",
    "CLEANUP_INTERVAL_SECONDS",
    "ORPHAN_COOLDOWN_SECONDS",
    "WorkerFactoryForRow",
    "cleanup_deleted_document_intents",
    "cleanup_expired_documents_hook",
    "cleanup_for_deleted_collections",
    "cleanup_for_deleted_documents",
    "cleanup_orphan_parse_versions",
    "find_deleted_document_cleanup_targets",
    "find_orphan_parse_versions",
    "run_cleanup_loop",
]


# Suppress unused-import warning for type-only narrowing.
_: Any = None
