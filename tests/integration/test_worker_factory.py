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

"""Production worker factory test — celery T3.1 follow-up.

Per architect msg=7782ebe0 spec gap fix + PM msg=dc13c4a5 root cause:
the FastAPI lifespan was wiring ``run_*_worker`` with a placeholder
that raised :class:`NotImplementedError` on every dispatch, so
async-mode documents stalled at PENDING forever and e2e-http-provider
gate failed on ``wait_for_document_indexes``.

Two contract-level invariants this test pins down (the e2e-http-
provider docker-compose covers full Qdrant / ES round-trip end-to-
end; this file covers the in-process invariants that don't need
external services):

1. **Factory failure → orchestrator §I.2 retry, not silent drop.**
   When ``worker_factory(payload)`` raises (broken collection config,
   missing collection row, transient backend error), the orchestrator
   runner must claim the row and finalise it ``FAILED`` with the
   error stashed in ``error_message``. Otherwise the row sits at
   ``PENDING`` forever and the reconciler keeps re-dispatching the
   same broken payload.

2. **Collection-not-found is a catchable WorkerFactoryError.**
   The factory must not crash with bare ``KeyError`` /
   ``AttributeError``; it should wrap the failure in
   :class:`WorkerFactoryError` so the orchestrator's
   ``except Exception`` catches it cleanly and the operator sees a
   meaningful ``error_message``.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import Engine, create_engine, insert
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.domains.knowledge_base.db.models import Collection
from aperag.indexing import InMemoryWorkQueue
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
from aperag.indexing.orchestrator import (
    DispatchPayload,
    OrchestratorConfig,
    run_worker_loop,
)
from aperag.indexing.worker_factory import (
    ProductionWorkerFactory,
    WorkerFactoryError,
)


def _make_engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    Collection.metadata.create_all(eng, tables=[Collection.__table__])
    return eng


def _seed_pending_row(engine: Engine, *, modality: Modality) -> int:
    with Session(engine) as session, session.begin():
        result = session.execute(
            insert(DocumentIndex)
            .values(
                document_id="doc-broken",
                parse_version="parse-v1",
                modality=modality.value,
                status=IndexStatus.PENDING.value,
                tenant_scope_key="user:t",
                collection_id="col-broken",
                source_path="source/path",
                is_serving=False,
            )
            .returning(DocumentIndex.id)
        )
        return int(result.scalar_one())


def test_orchestrator_finalises_failed_when_worker_factory_raises():
    """If ``worker_factory`` raises before ``process_one_task`` runs,
    the orchestrator must still claim the row and write FAILED so the
    §I.2 reconciler retry path picks it up. Without this, factory
    failures silently leak the row at PENDING forever and the
    reconciler keeps re-dispatching the same broken payload — the
    exact symptom that produced the e2e-http-provider stall (PM
    msg=dc13c4a5).
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            row_id = _seed_pending_row(engine, modality=Modality.VECTOR)
            queue = InMemoryWorkQueue()

            payload = DispatchPayload(
                index_id=row_id,
                document_id="doc-broken",
                parse_version="parse-v1",
                modality=Modality.VECTOR,
                source_path="source/path",
                collection_id="col-broken",
            )
            await queue.push(modality=Modality.VECTOR, payload=payload.to_dict())

            async def _failing_factory(_payload: DispatchPayload):
                raise WorkerFactoryError("collection col-broken not found")

            shutdown = asyncio.Event()

            async def _drive_one_then_shutdown() -> None:
                # Wait for the row to land in FAILED (the runner
                # claims + finalises asynchronously).
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    with Session(engine) as session:
                        row = session.get(DocumentIndex, row_id)
                        assert row is not None
                        if row.status == IndexStatus.FAILED.value:
                            break
                shutdown.set()

            await asyncio.gather(
                run_worker_loop(
                    config=OrchestratorConfig(modality=Modality.VECTOR, poll_timeout_seconds=0.05),
                    engine=engine,
                    queue=queue,
                    worker_factory=_failing_factory,
                    shutdown=shutdown,
                ),
                _drive_one_then_shutdown(),
            )

            with Session(engine) as session:
                row = session.get(DocumentIndex, row_id)
                assert row is not None
                assert row.status == IndexStatus.FAILED.value
                assert row.error_message and "worker_factory failed" in row.error_message
                assert row.retry_count == 1
                # §I.2: retry_after must be set so the reconciler picks it up
                # within the backoff window (30s for first failure).
                assert row.retry_after is not None
        finally:
            engine.dispose()

    asyncio.run(_run())


def test_production_factory_raises_when_collection_missing():
    """The production factory must wrap "collection not found" as
    :class:`WorkerFactoryError` so the orchestrator's except clause
    catches it and finalises FAILED. A bare exception type would still
    be caught (the orchestrator uses broad ``except Exception``), but
    the operator needs a meaningful error_message — that's what
    ``WorkerFactoryError`` provides.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            # Mark a row so the factory has a real ``index_id`` to
            # potentially hit on tenant_scope_key resolution. The
            # factory should fail BEFORE reaching tenant resolution
            # because the Collection lookup fails first.
            row_id = _seed_pending_row(engine, modality=Modality.VECTOR)
            payload = DispatchPayload(
                index_id=row_id,
                document_id="doc-broken",
                parse_version="parse-v1",
                modality=Modality.VECTOR,
                source_path="source/path",
                collection_id="col-does-not-exist",
            )

            factory = ProductionWorkerFactory(engine=engine, object_store=object())
            with pytest.raises(WorkerFactoryError) as exc_info:
                await factory(payload)
            assert "col-does-not-exist" in str(exc_info.value)
        finally:
            engine.dispose()

    asyncio.run(_run())


def test_production_factory_raises_when_collection_id_missing():
    """A payload without ``collection_id`` cannot resolve the
    collection-specific embedder / Qdrant tenant — the factory must
    fail fast with :class:`WorkerFactoryError` instead of papering
    over a malformed payload.
    """

    async def _run() -> None:
        engine = _make_engine()
        try:
            payload = DispatchPayload(
                index_id=1,
                document_id="doc-x",
                parse_version="parse-v1",
                modality=Modality.VECTOR,
                source_path="source/path",
                collection_id=None,
            )
            factory = ProductionWorkerFactory(engine=engine, object_store=object())
            with pytest.raises(WorkerFactoryError) as exc_info:
                await factory(payload)
            assert "collection_id" in str(exc_info.value)
        finally:
            engine.dispose()

    asyncio.run(_run())
