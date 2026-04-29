from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy import Engine, create_engine, insert, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.db.base import Base
from aperag.domains.knowledge_base.db.models import (
    Collection,
    CollectionStatus,
    CollectionType,
    Document,
    DocumentStatus,
)
from aperag.indexing import InMemoryObjectStore, InMemoryVectorBackend, Modality, VectorModality
from aperag.indexing.cleanup import cleanup_deleted_document_intents, find_deleted_document_cleanup_targets
from aperag.indexing.models import DocumentIndex, IndexStatus


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng, tables=[Collection.__table__, Document.__table__])
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


class FailingObjectStore(InMemoryObjectStore):
    def delete_objects_by_prefix(self, prefix: str) -> None:
        raise RuntimeError(f"object store down for {prefix}")


def _insert_collection(session: Session, *, collection_id: str = "col-del") -> None:
    session.add(
        Collection(
            id=collection_id,
            title="Task 17 cleanup",
            description="cleanup test collection",
            user="user|test",
            status=CollectionStatus.ACTIVE,
            type=CollectionType.DOCUMENT,
            config="{}",
        )
    )


def _insert_deleted_document(
    engine: Engine,
    *,
    document_id: str = "doc-del",
    collection_id: str = "col-del",
    status: DocumentStatus = DocumentStatus.DELETED,
) -> str:
    with Session(engine) as session, session.begin():
        _insert_collection(session, collection_id=collection_id)
        document = Document(
            id=document_id,
            name=f"{document_id}.txt",
            user="user|test",
            collection_id=collection_id,
            status=status,
            size=12,
            content_hash="hash-task-17",
            object_path=f"source/{document_id}.txt",
            gmt_deleted=datetime.now(timezone.utc),
        )
        session.add(document)
        session.flush()
        object_prefix = document.object_store_base_path()
        session.execute(
            insert(DocumentIndex).values(
                document_id=document_id,
                parse_version="pvtask17clean1",
                modality=Modality.VECTOR.value,
                status=IndexStatus.ACTIVE.value,
                tenant_scope_key="user:test",
                source_path=f"{object_prefix}/derived/parse_pvtask17clean1/chunks.jsonl",
                collection_id=collection_id,
                is_serving=True,
            )
        )
        return object_prefix


def test_find_deleted_document_cleanup_targets_uses_db_intent(engine: Engine):
    object_prefix = _insert_deleted_document(engine)

    targets = find_deleted_document_cleanup_targets(engine=engine)

    assert len(targets) == 1
    assert targets[0].document_id == "doc-del"
    assert targets[0].object_store_prefix == object_prefix


def test_cleanup_deleted_document_intents_deletes_object_store_backend_and_rows(engine: Engine):
    object_prefix = _insert_deleted_document(engine)
    object_store = InMemoryObjectStore()
    object_store.put(f"{object_prefix}/original.txt", b"source")
    object_store.put(f"{object_prefix}/derived/parse_pvtask17clean1/chunks.jsonl", b"{}")
    object_store.put("user-user-test/col-del/other-doc/original.txt", b"keep")

    backend = InMemoryVectorBackend()
    backend.upsert_point(
        point_id="chunk-task-17",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-del",
            "parse_version": "pvtask17clean1",
            "modality": Modality.VECTOR.value,
            "chunk_id": "chunk-task-17",
            "text": "hello",
        },
    )
    worker = VectorModality(backend=backend, store=object_store)

    counts = asyncio.run(
        cleanup_deleted_document_intents(
            engine=engine,
            workers={Modality.VECTOR: worker},
            object_store=object_store,
        )
    )

    assert counts["documents_seen"] == 1
    assert counts["object_store_deleted"] == 1
    assert counts["object_store_deferred"] == 0
    assert counts["backend_deleted"] == 1
    assert counts["rows_deleted"] == 1
    assert object_store.list_objects_by_prefix(object_prefix) == []
    assert object_store.obj_exists("user-user-test/col-del/other-doc/original.txt")
    assert backend.points_for_document("doc-del") == []
    with Session(engine) as session:
        assert session.scalar(select(DocumentIndex).where(DocumentIndex.document_id == "doc-del")) is None


def test_cleanup_deleted_document_intents_defers_backend_when_object_store_delete_fails(engine: Engine):
    object_prefix = _insert_deleted_document(engine)
    object_store = FailingObjectStore()
    object_store.put(f"{object_prefix}/original.txt", b"source")

    backend = InMemoryVectorBackend()
    backend.upsert_point(
        point_id="chunk-task-17",
        embedding=[0.0] * 16,
        payload={
            "document_id": "doc-del",
            "parse_version": "pvtask17clean1",
            "modality": Modality.VECTOR.value,
            "chunk_id": "chunk-task-17",
            "text": "hello",
        },
    )
    worker = VectorModality(backend=backend, store=object_store)

    counts = asyncio.run(
        cleanup_deleted_document_intents(
            engine=engine,
            workers={Modality.VECTOR: worker},
            object_store=object_store,
        )
    )

    assert counts["documents_seen"] == 1
    assert counts["object_store_deleted"] == 0
    assert counts["object_store_deferred"] == 1
    assert counts["backend_deleted"] == 0
    assert counts["rows_deleted"] == 0
    assert backend.points_for_document("doc-del")
    with Session(engine) as session:
        assert session.scalar(select(DocumentIndex).where(DocumentIndex.document_id == "doc-del")) is not None
    assert object_store.obj_exists(f"{object_prefix}/original.txt")
