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

"""T3.1 dispatcher + cleanup path C contract tests.

Locks the §K Wave 3 acceptance gates for the new T3.1 wire-in
helpers (dispatcher) + the architect msg=3890c9d7 Pattern A path C
cleanup extension:

1. **Dispatcher async mode** — INSERTs N PENDING rows + pushes
   payloads to the per-modality queue; returns the inserted row ids.
2. **Dispatcher inline mode** — INSERTs N PENDING rows + invokes
   ``process_one_task`` synchronously per modality; rows end up
   ACTIVE + is_serving=TRUE in one TX (§F.3).
3. **Dispatcher mode validation** — fail-fast on missing queue
   (async) or missing workers (inline).
4. **Path C cleanup** — cascades via path B per document, sweeps any
   collection-only rows, returns counts dict with ``collections_cleaned``.
5. **Dispatcher modality subset** — collection that opts out of e.g.
   vision still INSERTs + dispatches the requested subset only.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import (
    Engine,
    create_engine,
    insert,
    select,
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.indexing import (
    DispatchRequest,
    IndexingMode,
    InMemoryObjectStore,
    InMemoryVectorBackend,
    InMemoryWorkQueue,
    Modality,
    VectorModality,
    cleanup_for_deleted_collections,
    dispatch_indexing,
    drain_queue_sync,
    modalities_for_collection,
    parse_document,
)
from aperag.indexing.models import DocumentIndex, IndexStatus


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


def _seed_chunks(store: InMemoryObjectStore) -> tuple[str, str, str]:
    parsed = parse_document(
        store=store,
        collection_id="col-T3",
        document_id="doc-T3",
        source_bytes=b"# T3.1\n\nContent for dispatcher tests.",
    )
    return "doc-T3", parsed.parse_version, parsed.chunks_path


# ---------------------------------------------------------------------
# Dispatcher async mode
# ---------------------------------------------------------------------


def test_dispatcher_async_inserts_rows_and_pushes_payloads(engine):
    store = InMemoryObjectStore()
    doc_id, parse_version, chunks_path = _seed_chunks(store)
    queue = InMemoryWorkQueue()

    request = DispatchRequest(
        collection_id="col-T3",
        document_id=doc_id,
        parse_version=parse_version,
        source_path=chunks_path,
        tenant_scope_key="user:test",
        modalities=(Modality.VECTOR, Modality.FULLTEXT),
    )
    row_ids = asyncio.run(
        dispatch_indexing(
            engine=engine,
            queue=queue,
            workers=None,
            request=request,
            mode=IndexingMode.ASYNC,
        )
    )
    assert len(row_ids) == 2

    # DB rows are PENDING with the right scoping fields.
    with Session(engine) as session:
        rows = list(session.scalars(select(DocumentIndex).where(DocumentIndex.document_id == doc_id)))
    assert {r.modality for r in rows} == {Modality.VECTOR.value, Modality.FULLTEXT.value}
    for r in rows:
        assert r.status == IndexStatus.PENDING.value
        assert r.collection_id == "col-T3"
        assert r.source_path == chunks_path
        assert r.tenant_scope_key == "user:test"
        assert r.is_serving is False

    # Queue has both payloads.
    vec_payloads = drain_queue_sync(queue, Modality.VECTOR)
    ft_payloads = drain_queue_sync(queue, Modality.FULLTEXT)
    assert len(vec_payloads) == 1
    assert len(ft_payloads) == 1
    assert vec_payloads[0]["index_id"] in row_ids
    assert ft_payloads[0]["index_id"] in row_ids


def test_dispatcher_async_requires_queue(engine):
    request = DispatchRequest(
        collection_id="c",
        document_id="d",
        parse_version="x" * 16,
        source_path="p",
        tenant_scope_key="user:test",
        modalities=(Modality.VECTOR,),
    )
    with pytest.raises(ValueError, match="ASYNC.*queue"):
        asyncio.run(
            dispatch_indexing(
                engine=engine,
                queue=None,
                workers=None,
                request=request,
                mode=IndexingMode.ASYNC,
            )
        )


# ---------------------------------------------------------------------
# Dispatcher inline mode
# ---------------------------------------------------------------------


def test_dispatcher_inline_inserts_runs_and_finalizes_active_serving(engine):
    """Inline mode = single coroutine drives derive + sync + cutover.
    End state: row is ACTIVE + is_serving=TRUE in one TX (§F.3)."""
    store = InMemoryObjectStore()
    doc_id, parse_version, chunks_path = _seed_chunks(store)
    backend = InMemoryVectorBackend()
    workers = {Modality.VECTOR: VectorModality(backend=backend, store=store)}

    request = DispatchRequest(
        collection_id="col-T3",
        document_id=doc_id,
        parse_version=parse_version,
        source_path=chunks_path,
        tenant_scope_key="user:test",
        modalities=(Modality.VECTOR,),
    )
    row_ids = asyncio.run(
        dispatch_indexing(
            engine=engine,
            queue=None,
            workers=workers,
            request=request,
            mode=IndexingMode.INLINE,
        )
    )
    assert len(row_ids) == 1

    with Session(engine) as session:
        row = session.scalars(select(DocumentIndex).where(DocumentIndex.id == row_ids[0])).one()
    assert row.status == IndexStatus.ACTIVE.value
    assert row.is_serving is True
    assert backend.points_for_document(doc_id, parse_version)


def test_dispatcher_inline_requires_workers(engine):
    request = DispatchRequest(
        collection_id="c",
        document_id="d",
        parse_version="x" * 16,
        source_path="p",
        tenant_scope_key="user:test",
        modalities=(Modality.VECTOR,),
    )
    with pytest.raises(ValueError, match="INLINE.*workers"):
        asyncio.run(
            dispatch_indexing(
                engine=engine,
                queue=None,
                workers={},
                request=request,
                mode=IndexingMode.INLINE,
            )
        )


# ---------------------------------------------------------------------
# Modality subset
# ---------------------------------------------------------------------


def test_modalities_for_collection_helper_yields_canonical_subset_order():
    assert modalities_for_collection() == (
        Modality.VECTOR,
        Modality.FULLTEXT,
        Modality.GRAPH,
        Modality.SUMMARY,
        Modality.VISION,
    )
    assert modalities_for_collection(enable_vision=False) == (
        Modality.VECTOR,
        Modality.FULLTEXT,
        Modality.GRAPH,
        Modality.SUMMARY,
    )
    assert modalities_for_collection(
        enable_vector=True,
        enable_fulltext=False,
        enable_graph=False,
        enable_summary=True,
        enable_vision=False,
    ) == (Modality.VECTOR, Modality.SUMMARY)


# ---------------------------------------------------------------------
# Path C — cleanup_for_deleted_collections (architect msg=3890c9d7)
# ---------------------------------------------------------------------


def _insert_row(
    engine: Engine,
    *,
    document_id: str,
    parse_version: str,
    modality: Modality,
    collection_id: str,
    is_serving: bool = False,
) -> int:
    with Session(engine) as session, session.begin():
        result = session.execute(
            insert(DocumentIndex)
            .values(
                document_id=document_id,
                parse_version=parse_version,
                modality=modality.value,
                status=IndexStatus.ACTIVE.value,
                tenant_scope_key="user:test",
                source_path="ignored",
                collection_id=collection_id,
                is_serving=is_serving,
            )
            .returning(DocumentIndex.id)
        )
        return int(result.scalar_one())


def test_path_c_cascades_via_path_b_and_sweeps_collection_rows(engine):
    """Path C: deleted collections → cascade path B for each document
    → sweep remaining rows by collection_id. End state: 0 rows for the
    collection, backend tombstones removed for all parse_versions."""
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    # Two docs in one collection, each with one vector row, plus one
    # extra vector row whose document_id is "ghost" (e.g., row was
    # created but doc never indexed past PENDING) so we cover the
    # path-C sweep edge case.
    _insert_row(
        engine, document_id="docA", parse_version="paaaaaaaaaaaaaa1", modality=Modality.VECTOR, collection_id="col-X"
    )
    _insert_row(
        engine, document_id="docB", parse_version="pbbbbbbbbbbbbbb1", modality=Modality.VECTOR, collection_id="col-X"
    )
    # ghost row — collection_id matches, document_id has no other rows
    _insert_row(
        engine,
        document_id="docGhost",
        parse_version="pgggggggggggggg1",
        modality=Modality.VECTOR,
        collection_id="col-X",
    )
    # row in another collection — must NOT be touched
    other_id = _insert_row(
        engine,
        document_id="docOther",
        parse_version="poooooooooooooo1",
        modality=Modality.VECTOR,
        collection_id="col-Y",
    )

    # Pre-populate backend so we can assert path B fanned out.
    for chunk_id, doc, pv in (
        ("chunk-A", "docA", "paaaaaaaaaaaaaa1"),
        ("chunk-B", "docB", "pbbbbbbbbbbbbbb1"),
        ("chunk-G", "docGhost", "pgggggggggggggg1"),
        ("chunk-O", "docOther", "poooooooooooooo1"),
    ):
        backend.upsert_point(
            point_id=chunk_id,
            embedding=[0.0] * 16,
            payload={
                "document_id": doc,
                "parse_version": pv,
                "modality": "vector",
                "chunk_id": chunk_id,
                "text": "x",
                "section_path": None,
                "heading_anchor": None,
                "page_idx": None,
            },
        )

    counts = asyncio.run(
        cleanup_for_deleted_collections(
            engine=engine,
            workers={Modality.VECTOR: worker},
            collection_ids=["col-X"],
        )
    )
    assert counts["collections_cleaned"] == 1
    assert counts["backend_deleted"] == 3  # docA, docB, docGhost
    assert counts["rows_deleted"] == 3
    assert counts["graph_lineage_cleaned"] == 0  # no graph workers
    assert counts["backend_skipped"] == 0

    # Backend: only the other-collection chunk survives.
    surviving_chunks = {p["point_id"] for p in backend.all_points()}
    assert surviving_chunks == {"chunk-O"}

    # DB: only the other-collection row survives.
    with Session(engine) as session:
        remaining_ids = list(session.scalars(select(DocumentIndex.id)))
    assert remaining_ids == [other_id]


def test_path_c_handles_empty_input(engine):
    counts = asyncio.run(
        cleanup_for_deleted_collections(
            engine=engine,
            workers={},
            collection_ids=[],
        )
    )
    assert counts == {
        "backend_deleted": 0,
        "graph_lineage_cleaned": 0,
        "rows_deleted": 0,
        "backend_skipped": 0,
        "transient_deferred": 0,
        "collections_cleaned": 0,
    }


def test_path_c_idempotent_on_re_run(engine):
    """A second call with the same collection_ids returns zero counts
    (no rows left to clean) — proves the cascade is idempotent."""
    store = InMemoryObjectStore()
    backend = InMemoryVectorBackend()
    worker = VectorModality(backend=backend, store=store)

    _insert_row(
        engine, document_id="docZ", parse_version="pzzzzzzzzzzzzzz1", modality=Modality.VECTOR, collection_id="col-Z"
    )
    backend.upsert_point(
        point_id="chunk-Z",
        embedding=[0.0] * 16,
        payload={
            "document_id": "docZ",
            "parse_version": "pzzzzzzzzzzzzzz1",
            "modality": "vector",
            "chunk_id": "chunk-Z",
            "text": "x",
            "section_path": None,
            "heading_anchor": None,
            "page_idx": None,
        },
    )

    first = asyncio.run(
        cleanup_for_deleted_collections(
            engine=engine,
            workers={Modality.VECTOR: worker},
            collection_ids=["col-Z"],
        )
    )
    assert first["rows_deleted"] == 1

    second = asyncio.run(
        cleanup_for_deleted_collections(
            engine=engine,
            workers={Modality.VECTOR: worker},
            collection_ids=["col-Z"],
        )
    )
    assert second["rows_deleted"] == 0
    assert second["backend_deleted"] == 0
    assert second["collections_cleaned"] == 1
