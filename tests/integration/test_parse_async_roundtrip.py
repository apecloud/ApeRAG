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

"""Parse worker async roundtrip — celery Wave 4 T3 chunk 2.

Pin the production-critical contract that the upload handler's
``push_parse`` call lands in the parse worker pool, parses the source
through :func:`parse_document`, and dispatches the per-modality jobs
without the HTTP request blocking on parse latency.

Three layers covered here:

1. **Single-task happy path** (:func:`process_one_parse_task`) — the
   parse worker reads source bytes from the object store, runs
   :func:`parse_document`, INSERTs the per-modality PENDING rows, and
   pushes one payload per modality onto ``q:<modality>``.

2. **Failure paths** — missing source / DocParser raise / unknown
   modality each return a distinct status string and never advance
   to dispatch (zero modality rows + zero queue payloads).

3. **End-to-end roundtrip** — push parse → parse worker → modality
   workers → ``status=ACTIVE`` for every requested modality. Verifies
   the chunk 2 promise that an upload returns 202 and the indexing
   completes asynchronously without further HTTP involvement.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.indexing import (
    DEFAULT_PARSE_CONCURRENCY,
    FulltextModality,
    InMemoryFulltextBackend,
    InMemoryObjectStore,
    InMemoryVectorBackend,
    InMemoryWorkQueue,
    Modality,
    OrchestratorConfig,
    ParseDispatchPayload,
    ParseOrchestratorConfig,
    VectorModality,
    drain_queue_sync,
    process_one_parse_task,
    run_parse_worker_loop,
    run_worker_loop,
)
from aperag.indexing.models import DocumentIndex, IndexStatus
from aperag.indexing.object_store import source_artifact

COLLECTION_ID = "col-parse-async"
DOCUMENT_ID = "doc-parse-async"
TENANT_SCOPE_KEY = "user:parse-async"

SOURCE_MARKDOWN = b"""# Async Roundtrip

The parse worker promotes parsing off the HTTP request thread.

## Section A

A second paragraph keeps the chunker honest about paragraph breaks
so the chunk count is at least 2.

## Section B

A third paragraph makes the chunk count comfortably non-trivial.
"""


def _make_engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


def _seed_source(store: InMemoryObjectStore, body: bytes) -> str:
    """Write the source artifact at the canonical upload path and return it."""
    path = source_artifact(
        collection_id=COLLECTION_ID,
        document_id=DOCUMENT_ID,
        filename="original.md",
    )
    store.put(path, body)
    return path


# -----------------------------------------------------------------------
# Layer 1: process_one_parse_task happy + degenerate paths
# -----------------------------------------------------------------------


def test_process_one_parse_task_happy_path_inserts_rows_and_enqueues_modalities():
    """Parse worker single-task drive:

    - Source is read from the object store.
    - ``parse_document`` writes the canonical artifacts.
    - Vector + fulltext PENDING rows land with the real ``parse_version``.
    - One payload lands on each modality queue with the chunks.jsonl path.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()
            object_path = _seed_source(store, SOURCE_MARKDOWN)
            queue = InMemoryWorkQueue()

            payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path=object_path,
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=(Modality.VECTOR.value, Modality.FULLTEXT.value),
            )

            outcome = await process_one_parse_task(
                engine=engine,
                queue=queue,
                object_store=store,
                payload=payload,
            )
            assert outcome == "completed"

            # --- DB rows landed
            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex).order_by(DocumentIndex.modality)).scalars())
            assert len(rows) == 2
            for row in rows:
                assert row.status == IndexStatus.PENDING.value
                assert row.document_id == DOCUMENT_ID
                assert row.collection_id == COLLECTION_ID
                assert row.tenant_scope_key == TENANT_SCOPE_KEY
                assert row.parse_version  # 16-char hex from compute_parse_version
                assert "chunks.jsonl" in row.source_path

            # --- queue payloads landed
            vector_payloads = drain_queue_sync(queue, Modality.VECTOR)
            fulltext_payloads = drain_queue_sync(queue, Modality.FULLTEXT)
            assert len(vector_payloads) == 1
            assert len(fulltext_payloads) == 1
            for raw in vector_payloads + fulltext_payloads:
                assert raw["document_id"] == DOCUMENT_ID
                assert raw["collection_id"] == COLLECTION_ID
                assert "chunks.jsonl" in raw["source_path"]
                assert raw["parse_version"] == rows[0].parse_version
        finally:
            engine.dispose()

    asyncio.run(_run())


def test_process_one_parse_task_no_modalities_is_parse_only_no_dispatch():
    """Empty modality tuple → parse runs but no dispatch / no rows.

    Useful for a future "parse-only" upload mode (e.g. an admin
    re-parse without re-indexing). The artifacts must still be
    written so downstream re-dispatch sees the canonical
    ``derived/parse_<v>/chunks.jsonl``.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()
            object_path = _seed_source(store, SOURCE_MARKDOWN)
            queue = InMemoryWorkQueue()

            payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path=object_path,
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=(),
            )
            outcome = await process_one_parse_task(
                engine=engine,
                queue=queue,
                object_store=store,
                payload=payload,
            )
            assert outcome == "completed"
            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex)).scalars())
            assert rows == []
            # Artifacts still written so a follow-up dispatch can re-use them.
            assert any("chunks.jsonl" in path for path in store._objects)  # noqa: SLF001 — test introspection
        finally:
            engine.dispose()

    asyncio.run(_run())


def test_process_one_parse_task_purge_existing_triples_handles_rebuild():
    """``purge_existing_triples=True`` lets a rebuild dispatch land
    even when the prior ``(document_id, parse_version, modality)`` rows
    still exist (same content → same parse_version → would otherwise
    trip the ``uq_document_index_triple`` UNIQUE constraint).
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()
            object_path = _seed_source(store, SOURCE_MARKDOWN)
            queue = InMemoryWorkQueue()

            payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path=object_path,
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=(Modality.VECTOR.value,),
                purge_existing_triples=True,
            )
            # First pass — clean DB.
            outcome = await process_one_parse_task(engine=engine, queue=queue, object_store=store, payload=payload)
            assert outcome == "completed"
            # Second pass — rebuild path. Without purge this would
            # IntegrityError on INSERT.
            outcome = await process_one_parse_task(engine=engine, queue=queue, object_store=store, payload=payload)
            assert outcome == "completed"

            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex)).scalars())
            # Single VECTOR row — purge dropped the prior, INSERT
            # re-added; second dispatch did not stack a duplicate.
            assert len(rows) == 1
            assert rows[0].modality == Modality.VECTOR.value
        finally:
            engine.dispose()

    asyncio.run(_run())


# -----------------------------------------------------------------------
# Layer 2: failure paths
# -----------------------------------------------------------------------


def test_process_one_parse_task_missing_source_returns_failed_read():
    """No source artifact → ``failed_read`` + zero rows + zero queue payloads."""

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()  # source intentionally NOT seeded
            queue = InMemoryWorkQueue()
            payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path="collections/x/documents/y/source/missing.md",
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=(Modality.VECTOR.value,),
            )
            outcome = await process_one_parse_task(engine=engine, queue=queue, object_store=store, payload=payload)
            assert outcome == "failed_read"

            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex)).scalars())
            assert rows == []
            assert drain_queue_sync(queue, Modality.VECTOR) == []
        finally:
            engine.dispose()

    asyncio.run(_run())


def test_process_one_parse_task_unknown_modality_returns_failed_dispatch():
    """Unknown modality value → ``failed_dispatch`` + zero rows.

    Ensures a malformed payload (e.g. a future modality name pushed by
    a newer client) does not silently no-op or partially dispatch — we
    surface the failure through the return code so the run loop logs
    + drops cleanly.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()
            object_path = _seed_source(store, SOURCE_MARKDOWN)
            queue = InMemoryWorkQueue()
            payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path=object_path,
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=("not-a-real-modality",),
            )
            outcome = await process_one_parse_task(engine=engine, queue=queue, object_store=store, payload=payload)
            assert outcome == "failed_dispatch"

            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex)).scalars())
            assert rows == []
        finally:
            engine.dispose()

    asyncio.run(_run())


def test_process_one_parse_task_unsupported_extension_returns_failed_parse():
    """Source extension DocParser does not accept → ``failed_parse``.

    DocParser raises ``ValueError`` per Wave 4 T3 chunk 1 (no silent
    simulator fallback). The parse worker turns that into
    ``failed_parse`` so the operator can triage from logs without
    half-dispatched modality rows.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()
            unknown_ext_path = source_artifact(
                collection_id=COLLECTION_ID,
                document_id=DOCUMENT_ID,
                filename="weird.unknownext",
            )
            store.put(unknown_ext_path, b"opaque bytes")
            queue = InMemoryWorkQueue()
            payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path=unknown_ext_path,
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=(Modality.VECTOR.value,),
            )
            outcome = await process_one_parse_task(engine=engine, queue=queue, object_store=store, payload=payload)
            assert outcome == "failed_parse"

            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex)).scalars())
            assert rows == []
        finally:
            engine.dispose()

    asyncio.run(_run())


# -----------------------------------------------------------------------
# Layer 3: full async roundtrip — push_parse → parse worker → modality
# workers → ACTIVE.
# -----------------------------------------------------------------------


def test_push_parse_async_roundtrip_reaches_active_for_each_modality():
    """End-to-end async pipeline:

    upload handler ``push_parse`` → parse worker BLPOP → parse +
    dispatch → vector + fulltext workers BLPOP → derive + sync +
    cutover → ``status=ACTIVE`` AND ``is_serving=TRUE``.

    The HTTP request never participates in parse latency in this
    flow — that is the whole point of chunk 2.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()
            object_path = _seed_source(store, SOURCE_MARKDOWN)
            queue = InMemoryWorkQueue()
            shutdown = asyncio.Event()

            # Upload handler analog: push_parse, return immediately.
            upload_payload = ParseDispatchPayload(
                document_id=DOCUMENT_ID,
                collection_id=COLLECTION_ID,
                object_path=object_path,
                tenant_scope_key=TENANT_SCOPE_KEY,
                modalities=(Modality.VECTOR.value, Modality.FULLTEXT.value),
            )
            await queue.push_parse(payload=upload_payload.to_dict())

            # Parse worker run loop — short poll so shutdown is responsive.
            async def _store_factory() -> InMemoryObjectStore:
                return store

            parse_task = asyncio.create_task(
                run_parse_worker_loop(
                    config=ParseOrchestratorConfig(
                        concurrency=DEFAULT_PARSE_CONCURRENCY,
                        poll_timeout_seconds=0.05,
                    ),
                    engine=engine,
                    queue=queue,
                    object_store_factory=_store_factory,
                    shutdown=shutdown,
                )
            )

            # Modality worker factories — bind the per-modality backend
            # to the shared object store so derive() sees the chunks
            # the parse worker just wrote.
            vector_modality = VectorModality(backend=InMemoryVectorBackend(), store=store)
            fulltext_modality = FulltextModality(backend=InMemoryFulltextBackend(), store=store)

            async def _vector_factory(_payload):
                return vector_modality

            async def _fulltext_factory(_payload):
                return fulltext_modality

            vector_task = asyncio.create_task(
                run_worker_loop(
                    config=OrchestratorConfig(
                        modality=Modality.VECTOR,
                        concurrency=2,
                        poll_timeout_seconds=0.05,
                        heartbeat_interval_seconds=0,
                    ),
                    engine=engine,
                    queue=queue,
                    worker_factory=_vector_factory,
                    shutdown=shutdown,
                )
            )
            fulltext_task = asyncio.create_task(
                run_worker_loop(
                    config=OrchestratorConfig(
                        modality=Modality.FULLTEXT,
                        concurrency=2,
                        poll_timeout_seconds=0.05,
                        heartbeat_interval_seconds=0,
                    ),
                    engine=engine,
                    queue=queue,
                    worker_factory=_fulltext_factory,
                    shutdown=shutdown,
                )
            )

            # Wait until both modality rows reach ACTIVE — bounded
            # spin so a regression does not hang the test runner.
            async def _await_active() -> list[DocumentIndex]:
                deadline = asyncio.get_event_loop().time() + 5.0
                while True:
                    with Session(engine) as session:
                        rows = list(session.execute(select(DocumentIndex)).scalars())
                    if (
                        len(rows) == 2
                        and all(r.status == IndexStatus.ACTIVE.value for r in rows)
                        and all(r.is_serving for r in rows)
                    ):
                        return rows
                    if asyncio.get_event_loop().time() > deadline:
                        return rows
                    await asyncio.sleep(0.05)

            rows = await _await_active()
            shutdown.set()
            await asyncio.gather(parse_task, vector_task, fulltext_task, return_exceptions=True)

            assert len(rows) == 2, f"expected 2 rows, got: {[(r.modality, r.status) for r in rows]}"
            statuses = sorted((r.modality, r.status, r.is_serving) for r in rows)
            assert statuses == sorted(
                [
                    (Modality.VECTOR.value, IndexStatus.ACTIVE.value, True),
                    (Modality.FULLTEXT.value, IndexStatus.ACTIVE.value, True),
                ]
            )
        finally:
            engine.dispose()

    asyncio.run(_run())


# -----------------------------------------------------------------------
# Layer 4: payload roundtrip — protect the JSON dispatch shape.
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "parser_config",
    [None, {"use_mineru": True, "mineru_api_token": "secret"}],
)
def test_parse_dispatch_payload_to_from_dict_roundtrip(parser_config):
    """``ParseDispatchPayload.to_dict / from_dict`` round-trip preserves
    every field — pinned because the payload is the wire format both
    the upload handler (push) and the parse worker (pop) depend on.
    """
    p = ParseDispatchPayload(
        document_id="doc-1",
        collection_id="col-1",
        object_path="collections/col-1/documents/doc-1/source/upload.pdf",
        tenant_scope_key="user:42",
        modalities=(Modality.VECTOR.value, Modality.GRAPH.value),
        parser_config=parser_config,
        purge_existing_triples=True,
    )
    encoded = p.to_dict()
    decoded = ParseDispatchPayload.from_dict(encoded)
    assert decoded == p


def test_parse_dispatch_payload_rejects_non_dict_parser_config():
    """A malformed ``parser_config`` (e.g. a string from a sloppy
    client) is rejected at decode time so the parse worker never
    reaches DocParser with something that would crash the parser
    chain mid-run.
    """
    bad = {
        "document_id": "doc-1",
        "collection_id": "col-1",
        "object_path": "x",
        "tenant_scope_key": "user:1",
        "modalities": [Modality.VECTOR.value],
        "parser_config": "not-a-dict",
    }
    with pytest.raises(TypeError, match="parser_config must be a dict"):
        ParseDispatchPayload.from_dict(bad)
