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

"""Parse-then-dispatch integration test — celery T3.1 post-pass-8.

Per architect msg=c605037e ruling: document upload must invoke
``parse_document`` synchronously before the dispatcher creates
modality rows, so the canonical ``derived/parse_<v>/chunks.jsonl``
artifact exists by the time vector / fulltext workers pull their
payload off the queue. Celery's ``process_document_task`` used to be
that step; chunk 2 deleted the task layer without replacing the
caller, leaving every modality worker stuck on ``derive-incomplete``
in production (see PM msg=4159d7a1 root-cause).

This test pins the canonical post-fix flow:

1. Parse runs in-process and writes
   ``derived/parse_<v>/chunks.jsonl`` to the object store.
2. The dispatcher INSERTs vector + fulltext PENDING rows with
   ``source_path = parsed.chunks_path``.
3. Modality workers (driven through ``IndexingMode.INLINE`` here so
   the test does not need the full lifespan / async queue) read the
   chunks.jsonl that the parse step just wrote and reach
   ``status=ACTIVE`` AND ``is_serving=TRUE``.

A regression that re-routes the dispatcher's ``source_path`` back to
``document.object_store_base_path()`` would put the workers back on
the empty-derive reschedule loop, which trips this test on the
first iteration.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.indexing import (
    DispatchRequest,
    FulltextModality,
    IndexingMode,
    InMemoryFulltextBackend,
    InMemoryObjectStore,
    InMemoryVectorBackend,
    Modality,
    VectorModality,
    dispatch_indexing,
    parse_document,
)
from aperag.indexing.models import DocumentIndex, IndexStatus

COLLECTION_ID = "col-parse-then-dispatch"
DOCUMENT_ID = "doc-parse-then-dispatch"
TENANT_SCOPE_KEY = "user:parse-test"

SOURCE_MARKDOWN = b"""# Parse Then Dispatch

This is the first paragraph that the parser turns into a chunk so
the vector and fulltext workers have something to consume.

## Section A

A second paragraph keeps the chunker honest about paragraph breaks
so the chunk count is at least 2.
"""


def _make_engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    return eng


def test_parse_then_dispatch_reaches_active_for_chunks_modalities():
    """Parsing first, then dispatching the modality workers, takes
    each row from PENDING through to ``status=ACTIVE`` AND
    ``is_serving=TRUE``. This is the pin for the architect's
    "parse-as-first-stage in HTTP handler" wiring.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            store = InMemoryObjectStore()

            # Step 1 — parse first. The parser writes chunks.jsonl
            # into the object store under the canonical
            # ``derived/parse_<v>/`` path.
            parsed = parse_document(
                store=store,
                collection_id=COLLECTION_ID,
                document_id=DOCUMENT_ID,
                source_bytes=SOURCE_MARKDOWN,
            )
            assert parsed.chunks_path
            chunks_stream = store.get(parsed.chunks_path)
            assert chunks_stream is not None
            with chunks_stream:
                chunks_blob = chunks_stream.read()
            assert chunks_blob and chunks_blob.endswith(b"\n")
            assert chunks_blob.count(b"\n") >= 1  # at least one chunk line

            # Step 2 — wire vector + fulltext workers to the same
            # object store, then dispatch.
            workers = {
                Modality.VECTOR: VectorModality(backend=InMemoryVectorBackend(), store=store),
                Modality.FULLTEXT: FulltextModality(backend=InMemoryFulltextBackend(), store=store),
            }
            requested = (Modality.VECTOR, Modality.FULLTEXT)

            row_ids = await dispatch_indexing(
                engine=engine,
                queue=None,
                workers=workers,
                request=DispatchRequest(
                    collection_id=COLLECTION_ID,
                    document_id=DOCUMENT_ID,
                    parse_version=parsed.parse_version,
                    source_path=parsed.chunks_path,
                    tenant_scope_key=TENANT_SCOPE_KEY,
                    modalities=requested,
                ),
                mode=IndexingMode.INLINE,
            )
            assert len(row_ids) == 2

            # Step 3 — every row reached ACTIVE + is_serving.
            with Session(engine) as session:
                rows = list(session.execute(select(DocumentIndex).order_by(DocumentIndex.id)).scalars())
            assert len(rows) == 2
            for row in rows:
                assert row.status == IndexStatus.ACTIVE.value, (
                    f"row id={row.id} modality={row.modality} status={row.status} error={row.error_message}"
                )
                assert row.is_serving is True
                assert row.derived_artifact_path
        finally:
            engine.dispose()

    asyncio.run(_run())
