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

"""Cleanup worker_factory fan-out — celery Wave 4 T2.

Pin the contract that the cleanup loop materialises the right
per-(collection, modality) worker on every row instead of relying on
the legacy ``workers={}`` empty static map (which silently skipped
backend cleanup for every row in production after Wave 3 hard-cut).

Three layers covered:

1. **factory wins over workers map** — when both ``worker_factory``
   and ``workers`` are passed, the factory is consulted per row;
   the static map is only the fallback when no factory is given.

2. **per-row dispatch across collections** — a factory closure
   returning a different ``CleanupWorkerView`` per row triggers
   the right backend delete per (collection_id, modality) tuple.

3. **factory raises ⇒ backend_skipped + row still dropped** —
   ``WorkerFactoryError`` from the factory (Wave 4 T1 graph extractor
   gate / Wave 4 #9 vision multimodal gate) does not stop the cleanup
   loop; the row is dropped from ``document_index`` so the index does
   not grow unboundedly while the operator triages the gate.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, insert, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.indexing import (
    FulltextModality,
    InMemoryFulltextBackend,
    InMemoryObjectStore,
    InMemoryVectorBackend,
    Modality,
    VectorModality,
    cleanup_for_deleted_collections,
    cleanup_for_deleted_documents,
    cleanup_orphan_parse_versions,
)
from aperag.indexing.cleanup import _utcnow
from aperag.indexing.models import DocumentIndex, IndexStatus
from aperag.indexing.worker_factory import CleanupWorkerView, WorkerFactoryError

# -----------------------------------------------------------------------
# Fixtures — live ORM mirror of the production schema (matches the
# alembic head after the Wave 3 hard-cut migration). Mirrors the
# pattern used by ``tests/unit_test/indexing/test_t2_1_runtime.py``.
# -----------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


def _insert_row(
    engine: Engine,
    *,
    document_id: str,
    parse_version: str,
    modality: Modality,
    collection_id: str = "col-1",
    status: IndexStatus = IndexStatus.ACTIVE,
    is_serving: bool = True,
) -> int:
    with Session(engine) as session, session.begin():
        result = session.execute(
            insert(DocumentIndex)
            .values(
                document_id=document_id,
                parse_version=parse_version,
                modality=modality.value,
                status=status.value,
                tenant_scope_key="user:test",
                source_path=f"collections/{collection_id}/documents/{document_id}/derived/parse_{parse_version}/chunks.jsonl",
                collection_id=collection_id,
                is_serving=is_serving,
            )
            .returning(DocumentIndex.id)
        )
        return int(result.scalar_one())


def _set_updated_at(engine: Engine, row_id: int, when) -> None:
    with Session(engine) as session, session.begin():
        session.execute(update(DocumentIndex).where(DocumentIndex.id == row_id).values(updated_at=when))


# -----------------------------------------------------------------------
# Layer 1: factory takes precedence over workers map
# -----------------------------------------------------------------------


def test_worker_factory_wins_over_workers_map_when_both_passed(engine):
    """When both ``worker_factory`` and ``workers`` are provided, the
    factory is consulted per row. The map is the legacy fallback."""

    fallback_backend = InMemoryVectorBackend()
    fallback_worker = VectorModality(backend=fallback_backend, store=InMemoryObjectStore())

    factory_backend = InMemoryVectorBackend()
    factory_worker = VectorModality(backend=factory_backend, store=InMemoryObjectStore())
    factory_calls = {"count": 0}

    async def factory_closure(row: DocumentIndex):
        factory_calls["count"] += 1
        return factory_worker

    pv_a = "doc-del-a-pv0001"[:16]
    _insert_row(engine, document_id="doc-del", parse_version=pv_a, modality=Modality.VECTOR)
    factory_backend.upsert_point(
        chunk_id="chunk-fac",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-del",
            "parse_version": pv_a,
            "modality": "vector",
            "chunk_id": "chunk-fac",
            "text": "x",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )
    fallback_backend.upsert_point(
        chunk_id="chunk-fallback",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-del",
            "parse_version": pv_a,
            "modality": "vector",
            "chunk_id": "chunk-fallback",
            "text": "x",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            workers={Modality.VECTOR: fallback_worker},
            worker_factory=factory_closure,
            document_ids=["doc-del"],
        )
    )
    assert counts["backend_deleted"] == 1
    assert counts["rows_deleted"] == 1
    assert factory_calls["count"] == 1
    # factory_backend point removed — factory worker is what cleanup hit.
    assert factory_backend.points_for_document("doc-del") == []
    # fallback_backend point untouched — fallback worker was never used.
    assert len(fallback_backend.points_for_document("doc-del")) == 1


# -----------------------------------------------------------------------
# Layer 2: per-row dispatch across collections — factory closure picks
# the right (collection_id, modality) backend per row.
# -----------------------------------------------------------------------


def test_factory_dispatches_per_collection_and_modality(engine):
    """Multiple collections + multiple modalities → factory is called
    per row, each row's correct backend gets the delete.
    """

    backends: dict[tuple[str, Modality], Any] = {
        ("col-A", Modality.VECTOR): InMemoryVectorBackend(),
        ("col-A", Modality.FULLTEXT): InMemoryFulltextBackend(),
        ("col-B", Modality.VECTOR): InMemoryVectorBackend(),
        ("col-B", Modality.FULLTEXT): InMemoryFulltextBackend(),
    }
    workers_per_pair: dict[tuple[str, Modality], Any] = {
        ("col-A", Modality.VECTOR): VectorModality(
            backend=backends[("col-A", Modality.VECTOR)], store=InMemoryObjectStore()
        ),
        ("col-A", Modality.FULLTEXT): FulltextModality(
            backend=backends[("col-A", Modality.FULLTEXT)],
            store=InMemoryObjectStore(),
            collection_id="col-A",
        ),
        ("col-B", Modality.VECTOR): VectorModality(
            backend=backends[("col-B", Modality.VECTOR)], store=InMemoryObjectStore()
        ),
        ("col-B", Modality.FULLTEXT): FulltextModality(
            backend=backends[("col-B", Modality.FULLTEXT)],
            store=InMemoryObjectStore(),
            collection_id="col-B",
        ),
    }

    async def factory_closure(row: DocumentIndex):
        modality = Modality(row.modality)
        return workers_per_pair[(row.collection_id, modality)]

    pv = "perrowparseverv1"[:16]
    document_ids = []
    for col, modality, chunk_id in (
        ("col-A", Modality.VECTOR, "ca-vec"),
        ("col-A", Modality.FULLTEXT, "ca-ft"),
        ("col-B", Modality.VECTOR, "cb-vec"),
        ("col-B", Modality.FULLTEXT, "cb-ft"),
    ):
        # A document belongs to exactly one collection in production
        # (the §F.1 partial unique index enforces "one serving per
        # (document_id, modality)"). Use per-collection document ids
        # to model realistic delete-by-collection scope.
        document_id = f"doc-{col}"
        document_ids.append(document_id)
        _insert_row(
            engine,
            document_id=document_id,
            parse_version=pv,
            modality=modality,
            collection_id=col,
        )
        backend = backends[(col, modality)]
        if modality is Modality.VECTOR:
            backend.upsert_point(
                chunk_id=chunk_id,
                embedding=[0.0] * 16,
                payload={
                    "document_id": document_id,
                    "parse_version": pv,
                    "modality": "vector",
                    "chunk_id": chunk_id,
                    "text": "x",
                    "section_path": None,
                    "heading_anchor": None,
                    "page_idx": None,
                },
            )
        else:
            backend.bulk_index(
                documents=[
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "parse_version": pv,
                        "collection_id": col,
                        "text": "x",
                        "section_path": None,
                        "heading_anchor": None,
                    }
                ]
            )

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            worker_factory=factory_closure,
            document_ids=list(set(document_ids)),
        )
    )
    assert counts["backend_deleted"] == 4
    assert counts["rows_deleted"] == 4

    # Every backend got cleared for its corresponding document.
    for (col, modality), backend in backends.items():
        document_id = f"doc-{col}"
        if modality is Modality.VECTOR:
            assert backend.points_for_document(document_id) == [], f"vector residual in {col}"


# -----------------------------------------------------------------------
# Layer 3: factory raises ⇒ backend_skipped + row dropped.
# -----------------------------------------------------------------------


def test_factory_raises_worker_factory_error_skips_backend_drops_row(engine):
    """A factory that raises ``WorkerFactoryError`` (Wave 4 T1 graph
    extractor gate / Wave 4 #9 vision multimodal gate) must:

    - count the row as ``backend_skipped`` (operator-visible signal)
    - still drop the DB row so the cleanup index does not grow
      unboundedly while the gate is active
    """

    pv = "wfegateparsever1"[:16]
    _insert_row(engine, document_id="doc-gated", parse_version=pv, modality=Modality.GRAPH)

    raise_count = {"value": 0}

    async def gated_factory(row: DocumentIndex):
        raise_count["value"] += 1
        raise WorkerFactoryError("Wave 4 T1 extractor not wired")

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            worker_factory=gated_factory,
            document_ids=["doc-gated"],
        )
    )
    assert counts["backend_skipped"] == 1
    assert counts["rows_deleted"] == 1
    assert counts["graph_lineage_cleaned"] == 0
    assert raise_count["value"] == 1
    with Session(engine) as session:
        remaining = list(session.scalars(select(DocumentIndex.id).where(DocumentIndex.document_id == "doc-gated")))
    assert remaining == []


# -----------------------------------------------------------------------
# Layer 4: orphan parse_version GC accepts worker_factory.
# -----------------------------------------------------------------------


def test_orphan_parse_version_gc_uses_worker_factory(engine):
    """``cleanup_orphan_parse_versions`` honours ``worker_factory`` so
    the production lifespan can wire it the same way as path B."""

    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=InMemoryObjectStore())

    pv_old = "orphanversionold"[:16]
    pv_new = "orphanversionnew"[:16]
    old_id = _insert_row(
        engine,
        document_id="doc-orphan",
        parse_version=pv_old,
        modality=Modality.VECTOR,
        is_serving=False,
    )
    _insert_row(
        engine,
        document_id="doc-orphan",
        parse_version=pv_new,
        modality=Modality.VECTOR,
        is_serving=True,
    )
    _set_updated_at(engine, old_id, _utcnow() - timedelta(hours=2))

    backend.upsert_point(
        chunk_id="chunk-orphan",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-orphan",
            "parse_version": pv_old,
            "modality": "vector",
            "chunk_id": "chunk-orphan",
            "text": "x",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )

    factory_calls = {"count": 0}

    async def factory_closure(row: DocumentIndex):
        factory_calls["count"] += 1
        return worker

    counts = asyncio.run(
        cleanup_orphan_parse_versions(
            engine=engine,
            worker_factory=factory_closure,
        )
    )
    assert counts["backend_deleted"] == 1
    assert counts["rows_deleted"] == 1
    assert factory_calls["count"] == 1
    assert backend.points_for_document("doc-orphan") == []


# -----------------------------------------------------------------------
# Layer 5: collection-deletion cascade routes through factory too.
# -----------------------------------------------------------------------


def test_collection_deletion_cascade_uses_worker_factory(engine):
    """Path C delegates to path B internally; the factory plumbed
    through that chain so collection-deletion cleanup also runs the
    real per-(collection, modality) backend delete.
    """

    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=InMemoryObjectStore())

    pv = "coldelparsever01"[:16]
    _insert_row(
        engine,
        document_id="doc-col",
        parse_version=pv,
        modality=Modality.VECTOR,
        collection_id="col-doomed",
    )
    backend.upsert_point(
        chunk_id="chunk-col",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-col",
            "parse_version": pv,
            "modality": "vector",
            "chunk_id": "chunk-col",
            "text": "x",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )

    factory_calls = {"count": 0}

    async def factory_closure(row: DocumentIndex):
        factory_calls["count"] += 1
        return worker

    counts = asyncio.run(
        cleanup_for_deleted_collections(
            engine=engine,
            worker_factory=factory_closure,
            collection_ids=["col-doomed"],
        )
    )
    assert counts["backend_deleted"] == 1
    assert counts["rows_deleted"] == 1
    assert counts["collections_cleaned"] == 1
    assert factory_calls["count"] == 1
    assert backend.points_for_document("doc-col") == []


# -----------------------------------------------------------------------
# Layer 6: CleanupWorkerView itself — derive/sync stubs raise loudly.
# -----------------------------------------------------------------------


def test_cleanup_worker_view_derive_and_sync_raise_not_implemented():
    """The view is cleanup-only: any production path that wires it
    into dispatch by mistake must surface loudly instead of silently
    dropping work."""

    backend = InMemoryVectorBackend()
    view = CleanupWorkerView(modality=Modality.VECTOR, backend=backend)
    assert view.modality is Modality.VECTOR
    assert view._backend is backend

    async def _try_calls():
        with pytest.raises(NotImplementedError, match="cleanup-only"):
            await view.derive(document_id="d", parse_version="v", source_path="s")
        with pytest.raises(NotImplementedError, match="cleanup-only"):
            await view.sync(document_id="d", parse_version="v", derived_artifact_path="s")

    asyncio.run(_try_calls())


# -----------------------------------------------------------------------
# Layer 7: backward compat — workers-map-only path still works.
# -----------------------------------------------------------------------


def test_workers_map_only_path_unchanged_for_existing_callers(engine):
    """Pre-T2 callers that pass only ``workers={...}`` continue to
    work — the factory parameter is opt-in.
    """

    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=InMemoryObjectStore())

    pv = "compatparsever01"[:16]
    _insert_row(engine, document_id="doc-compat", parse_version=pv, modality=Modality.VECTOR)
    backend.upsert_point(
        chunk_id="chunk-compat",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-compat",
            "parse_version": pv,
            "modality": "vector",
            "chunk_id": "chunk-compat",
            "text": "x",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            workers={Modality.VECTOR: worker},
            document_ids=["doc-compat"],
        )
    )
    assert counts["backend_deleted"] == 1
    assert counts["rows_deleted"] == 1
    assert backend.points_for_document("doc-compat") == []
