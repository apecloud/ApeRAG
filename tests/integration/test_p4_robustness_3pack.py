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

"""Wave 5 Phase 4 — production robustness 3-pack.

Three latent issues from Wave 4 huangheng + architect ratify obs surface
silent-drop / wasted-work bugs that this phase locks down with explicit
behaviour:

1. **T2 cleanup transient-vs-intentional split** — pre-Wave-5
   ``_resolve_cleanup_worker`` collapsed any factory exception into
   "drop the DB row". This loses the retry signal when a transient
   infra error (Qdrant blip, ES network glitch) makes a
   ``WorkerFactoryError``-look-alike. Wave 5 P4 distinguishes the two
   so transient errors keep the row for next-cycle retry; intentional
   gates still drop the row to keep the index from growing.

2. **T3 parse_orchestrator parse_version short-circuit** — pre-Wave-5
   :func:`parse_document` always re-runs DocParser even when the
   resulting artifact directory is byte-identical (parse_version is
   content-derived). Wave 5 P4 short-circuits when all three derived
   artifacts are already present in the object store.

3. **T2 reconciler stuck-document parse re-enqueue** — pre-Wave-5
   parse failures (DocParser raise / source missing) silently dropped
   the document_id; the operator had no recovery path. Wave 5 P4 adds
   a reconciler scan that re-enqueues ``q:parse`` for documents that
   uploaded > N min ago without sprouting any ``document_index`` rows.

Tests use SQLAlchemy in-memory SQLite + InMemoryWorkQueue +
InMemoryObjectStore so the suite stays fast (~ms per test) without
external dependencies.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import Engine, create_engine, insert, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.db.base import Base
from aperag.indexing import (
    InMemoryObjectStore,
    InMemoryWorkQueue,
    Modality,
    parse_document,
    reconcile_stuck_documents_for_parse_reenqueue,
)
from aperag.indexing.cleanup import (
    CleanupWorkerResolution,
    _resolve_cleanup_worker,
    cleanup_for_deleted_documents,
)
from aperag.indexing.models import DocumentIndex, IndexStatus
from aperag.indexing.worker_factory import WorkerFactoryError

# -----------------------------------------------------------------------
# Fixtures — SQLite mirror of the production schema for both
# DocumentIndex and Document tables (P4-3 needs the Document table).
# -----------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Create only the tables we need for these tests so we don't drag
    # the whole schema in. Document table is required for the
    # stuck-parse reconciler scan; DocumentIndex table for the
    # zero-rows predicate.
    from aperag.domains.knowledge_base.db.models import Collection, Document

    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    Base.metadata.create_all(eng, tables=[Collection.__table__, Document.__table__])
    return eng


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------------------------------------------------
# Layer 1 — P4-1 cleanup transient-vs-intentional split
# -----------------------------------------------------------------------


def _stub_row(modality: Modality = Modality.VECTOR) -> DocumentIndex:
    return DocumentIndex(
        id=1,
        document_id="doc-stub",
        parse_version="pv0",
        modality=modality.value,
        status=IndexStatus.ACTIVE.value,
        is_serving=True,
        tenant_scope_key="user:t",
        collection_id="col-stub",
        source_path="x",
    )


def test_p4_resolve_cleanup_worker_intentional_gate_drops_row():
    """``WorkerFactoryError`` from the factory → ``transient=False``.

    The cleanup loop should drop the DB row so the index does not
    grow unboundedly while the gate is active (Wave 4 T2 surgical-
    gate corollary preserved).
    """

    async def gated_factory(_row: DocumentIndex):
        raise WorkerFactoryError("Wave 4 wiring T1 not yet implemented")

    async def _run() -> None:
        result = await _resolve_cleanup_worker(
            workers=None,
            worker_factory=gated_factory,
            row=_stub_row(),
        )
        assert result == CleanupWorkerResolution(worker=None, transient=False)

    asyncio.run(_run())


def test_p4_resolve_cleanup_worker_transient_error_keeps_row_for_retry():
    """Non-:class:`WorkerFactoryError` exception → ``transient=True``.

    Operator-visible signal: the next cleanup cycle (5 min later)
    will retry. Pre-Wave-5 this collapsed into the same path as the
    intentional gate, silently dropping the row + losing the retry
    signal.
    """

    async def transient_factory(_row: DocumentIndex):
        raise ConnectionError("Qdrant cluster unhealthy")

    async def _run() -> None:
        result = await _resolve_cleanup_worker(
            workers=None,
            worker_factory=transient_factory,
            row=_stub_row(),
        )
        assert result.worker is None
        assert result.transient is True

    asyncio.run(_run())


def test_p4_cleanup_for_deleted_documents_skips_drop_on_transient_error(engine):
    """Path B — ``transient=True`` resolution skips DB row drop.

    The row stays in ``document_index`` so the next cleanup cycle
    sees it again. Counts the deferred row under
    ``transient_deferred`` so operators can track the recovery rate.
    """
    pv = "stuckxyz0000000a"[:16]
    with Session(engine) as session, session.begin():
        session.execute(
            insert(DocumentIndex).values(
                document_id="doc-transient",
                parse_version=pv,
                modality=Modality.VECTOR.value,
                status=IndexStatus.ACTIVE.value,
                tenant_scope_key="user:t",
                collection_id="col-transient",
                source_path=f"collections/col-transient/documents/doc-transient/derived/parse_{pv}/chunks.jsonl",
                is_serving=True,
            )
        )

    async def transient_factory(_row: DocumentIndex):
        raise ConnectionError("Qdrant cluster unhealthy")

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            worker_factory=transient_factory,
            document_ids=["doc-transient"],
        )
    )
    assert counts["transient_deferred"] == 1
    assert counts["rows_deleted"] == 0
    assert counts["backend_skipped"] == 0
    assert counts["backend_deleted"] == 0

    with Session(engine) as session:
        rows = list(session.scalars(select(DocumentIndex).where(DocumentIndex.document_id == "doc-transient")))
    assert len(rows) == 1, "transient infra error must NOT drop the DB row"


def test_p4_cleanup_for_deleted_documents_drops_row_on_intentional_gate(engine):
    """Path B — ``transient=False`` from a :class:`WorkerFactoryError`
    intentional gate still drops the DB row (the gate is by-design;
    keeping the row would let the index grow unboundedly).
    """
    pv = "gatedxyz000000a"[:15] + "x"
    with Session(engine) as session, session.begin():
        session.execute(
            insert(DocumentIndex).values(
                document_id="doc-gated",
                parse_version=pv,
                modality=Modality.GRAPH.value,
                status=IndexStatus.ACTIVE.value,
                tenant_scope_key="user:t",
                collection_id="col-gated",
                source_path=f"collections/col-gated/documents/doc-gated/derived/parse_{pv}/chunks.jsonl",
                is_serving=True,
            )
        )

    async def gated_factory(_row: DocumentIndex):
        raise WorkerFactoryError("Wave 4 wiring T1 not yet")

    counts = asyncio.run(
        cleanup_for_deleted_documents(
            engine=engine,
            worker_factory=gated_factory,
            document_ids=["doc-gated"],
        )
    )
    assert counts["backend_skipped"] == 1
    assert counts["transient_deferred"] == 0
    assert counts["rows_deleted"] == 1

    with Session(engine) as session:
        rows = list(session.scalars(select(DocumentIndex).where(DocumentIndex.document_id == "doc-gated")))
    assert rows == [], "intentional gate must drop the DB row to bound index growth"


# -----------------------------------------------------------------------
# Layer 2 — P4-2 parse_version short-circuit
# -----------------------------------------------------------------------


def test_p4_parse_document_short_circuits_when_artifacts_already_present():
    """Pre-Wave-5 :func:`parse_document` re-runs DocParser even when
    artifacts are byte-identical. Wave 5 P4 short-circuits if all
    three canonical artifacts already exist in the store under the
    canonical ``derived/parse_<version>/`` path.
    """
    store = InMemoryObjectStore()
    body = b"# Cached Doc\n\nFirst paragraph.\n\n## Sub\n\nSecond paragraph.\n"

    # First parse — populates artifacts.
    first = parse_document(
        store=store,
        collection_id="col",
        document_id="doc",
        source_bytes=body,
    )
    snapshot = dict(store._objects)  # noqa: SLF001 — test introspection

    # Hijack store.put to surface unintended writes during the second call.
    write_calls = {"count": 0}
    original_put = store.put

    def tracking_put(path, data):  # noqa: ANN001
        write_calls["count"] += 1
        original_put(path, data)

    store.put = tracking_put  # type: ignore[method-assign]

    # Second parse — must short-circuit (no writes, no DocParser).
    second = parse_document(
        store=store,
        collection_id="col",
        document_id="doc",
        source_bytes=body,
    )
    assert second == first
    assert write_calls["count"] == 0, "short-circuit must not re-write artifacts"
    assert dict(store._objects) == snapshot, "artifacts must be byte-identical"


def test_p4_parse_document_does_not_short_circuit_when_chunks_missing():
    """If ``chunks.jsonl`` is missing the previous parse was interrupted;
    the short-circuit predicate must return False so DocParser re-runs.
    """
    store = InMemoryObjectStore()
    body = b"# Doc\n\nText.\n"

    # First parse — populates all three artifacts.
    parse_document(
        store=store,
        collection_id="col",
        document_id="doc",
        source_bytes=body,
    )

    # Manually delete chunks.jsonl to simulate an interrupted parse.
    chunks_keys = [k for k in store._objects if k.endswith("chunks.jsonl")]
    assert chunks_keys, "fixture should have produced chunks.jsonl"
    for k in chunks_keys:
        del store._objects[k]  # noqa: SLF001

    # Second parse — must NOT short-circuit; it must re-write chunks.
    parse_document(
        store=store,
        collection_id="col",
        document_id="doc",
        source_bytes=body,
    )
    chunks_keys = [k for k in store._objects if k.endswith("chunks.jsonl")]
    assert chunks_keys, "interrupted parse must be recovered, not skipped"


def test_p4_parse_document_short_circuit_can_be_disabled():
    """Tests / debugging callers can pass
    ``short_circuit_if_artifacts_exist=False`` to force a re-parse.
    """
    store = InMemoryObjectStore()
    body = b"# Doc\n\ntext\n"

    parse_document(
        store=store,
        collection_id="col",
        document_id="doc",
        source_bytes=body,
    )

    write_calls = {"count": 0}
    original_put = store.put

    def tracking_put(path, data):  # noqa: ANN001
        write_calls["count"] += 1
        original_put(path, data)

    store.put = tracking_put  # type: ignore[method-assign]

    parse_document(
        store=store,
        collection_id="col",
        document_id="doc",
        source_bytes=body,
        short_circuit_if_artifacts_exist=False,
    )
    assert write_calls["count"] >= 3, "disabling short-circuit must re-write all artifacts"


# -----------------------------------------------------------------------
# Layer 3 — P4-3 reconciler stuck-document parse re-enqueue
# -----------------------------------------------------------------------


def _insert_document(
    engine: Engine,
    *,
    document_id: str,
    collection_id: str = "col-stuck",
    object_path: str | None = "collections/col-stuck/documents/doc/source/upload.pdf",
    age: timedelta = timedelta(minutes=10),
    user: str = "u1",
) -> None:
    """Insert a Document row aged ``age`` ago. The age controls whether
    the row is past the cooldown window for re-enqueue.
    """
    import json as _json

    from aperag.domains.knowledge_base.db.models import Document

    created = _utcnow() - age
    doc_metadata = _json.dumps({"object_path": object_path}) if object_path else None
    with Session(engine) as session, session.begin():
        session.add(
            Document(
                id=document_id,
                name="doc.pdf",
                user=user,
                collection_id=collection_id,
                status="UPLOADED",
                size=1234,
                content_hash="abc",
                object_path=object_path,
                doc_metadata=doc_metadata,
                gmt_created=created,
                gmt_updated=created,
            )
        )


def _insert_collection(engine: Engine, *, collection_id: str = "col-stuck") -> None:
    import json as _json

    from aperag.domains.knowledge_base.db.models import Collection

    with Session(engine) as session, session.begin():
        session.add(
            Collection(
                id=collection_id,
                title="t",
                user="u1",
                type="document",
                status="ACTIVE",
                config=_json.dumps(
                    {
                        "enable_vector": True,
                        "enable_fulltext": True,
                        "enable_summary": True,
                    }
                ),
            )
        )


def test_p4_reconciler_re_enqueues_stuck_document(engine):
    """A document uploaded > cooldown ago with zero ``document_index``
    rows must be re-enqueued onto ``q:parse`` so the parse worker
    gets another chance.
    """
    _insert_collection(engine)
    _insert_document(engine, document_id="doc-stuck", age=timedelta(minutes=10))

    queue = InMemoryWorkQueue()
    pushed = asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=60,  # 60s cooldown — doc is 10 min old, easily past
        )
    )
    assert pushed == 1
    assert queue.parse_qsize() == 1


def test_p4_reconciler_skips_documents_within_cooldown(engine):
    """A freshly-uploaded document (within cooldown) must NOT be
    re-enqueued — the parse worker may simply be in the middle of
    its first run.
    """
    _insert_collection(engine)
    _insert_document(engine, document_id="doc-fresh", age=timedelta(seconds=10))

    queue = InMemoryWorkQueue()
    pushed = asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=300,  # 5 min — doc is 10s old
        )
    )
    assert pushed == 0
    assert queue.parse_qsize() == 0


def test_p4_reconciler_skips_documents_with_existing_index_rows(engine):
    """A document that already has ``document_index`` rows is being
    successfully indexed; the reconciler must not interfere.
    """
    _insert_collection(engine)
    _insert_document(engine, document_id="doc-indexing", age=timedelta(minutes=10))
    pv = "indexpv00000000a"[:15] + "a"
    with Session(engine) as session, session.begin():
        session.execute(
            insert(DocumentIndex).values(
                document_id="doc-indexing",
                parse_version=pv,
                modality=Modality.VECTOR.value,
                status=IndexStatus.PENDING.value,
                tenant_scope_key="user:u1",
                collection_id="col-stuck",
                source_path=f"collections/col-stuck/documents/doc-indexing/derived/parse_{pv}/chunks.jsonl",
                is_serving=False,
            )
        )

    queue = InMemoryWorkQueue()
    pushed = asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=60,
        )
    )
    assert pushed == 0
    assert queue.parse_qsize() == 0


def test_p4_reconciler_at_most_once_per_cooldown(engine):
    """Two consecutive reconciler ticks must not re-push the same
    stuck document — gmt_updated bumps after first push so the
    cooldown predicate filters it out.
    """
    _insert_collection(engine)
    _insert_document(engine, document_id="doc-bump", age=timedelta(minutes=10))

    queue = InMemoryWorkQueue()
    first = asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=60,
        )
    )
    assert first == 1
    second = asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=60,
        )
    )
    assert second == 0, (
        "second tick must not re-push within the cooldown window — Document.gmt_updated was bumped on the first push"
    )
    assert queue.parse_qsize() == 1


def test_p4_reconciler_skips_documents_without_object_path(engine):
    """A stuck document whose ``doc_metadata.object_path`` is missing
    cannot be re-parsed — the parse worker has nothing to read.
    The reconciler must skip it (operator must fix the upload).
    """
    _insert_collection(engine)
    _insert_document(
        engine,
        document_id="doc-missing-path",
        age=timedelta(minutes=10),
        object_path=None,
    )

    queue = InMemoryWorkQueue()
    pushed = asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=60,
        )
    )
    assert pushed == 0


def test_p4_reconciler_pushes_correct_payload_shape(engine):
    """The payload pushed by the reconciler must match the upload
    handler's contract (per ``ParseDispatchPayload`` shape) so the
    parse worker sees identical data regardless of producer.
    """
    _insert_collection(engine)
    _insert_document(
        engine,
        document_id="doc-payload",
        collection_id="col-stuck",
        age=timedelta(minutes=10),
    )

    queue = InMemoryWorkQueue()
    asyncio.run(
        reconcile_stuck_documents_for_parse_reenqueue(
            engine=engine,
            queue=queue,
            cooldown_seconds=60,
        )
    )

    payload = asyncio.run(queue.pop_parse(timeout_seconds=0.5))
    assert payload is not None
    assert payload["document_id"] == "doc-payload"
    assert payload["collection_id"] == "col-stuck"
    assert payload["object_path"] == "collections/col-stuck/documents/doc/source/upload.pdf"
    assert payload["tenant_scope_key"] == "user:u1"
    assert set(payload["modalities"]) == {"vector", "fulltext", "summary"}
    assert payload["purge_existing_triples"] is True
