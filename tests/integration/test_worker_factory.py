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
from typing import Any

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


# ---------------------------------------------------------------------
# Wave 4 T8 chunk 4b — graph_backend_type dispatch + T1-extractor gate.
# ---------------------------------------------------------------------


def _make_collection_stub(*, config_obj: Any = None) -> Any:
    """Lightweight stub that the dispatch helpers can read off."""

    class _Stub:
        id = "col-4b"
        config = config_obj

    return _Stub()


def test_resolve_graph_backend_type_defaults_to_postgres():
    """A collection with no ``config`` or no ``graph_backend_type`` field
    falls back to the default ``postgres`` backend (the §D.3.5 reference
    adapter; new collections that opt into knowledge graph automatically
    use the application's own PostgreSQL without extra infra).
    """

    from aperag.indexing.worker_factory import _resolve_graph_backend_type

    assert _resolve_graph_backend_type(_make_collection_stub(config_obj=None)) == "postgres"

    class _ConfigNoBackend:
        graph_backend_type = None

    assert _resolve_graph_backend_type(_make_collection_stub(config_obj=_ConfigNoBackend())) == "postgres"


@pytest.mark.parametrize("backend", ["postgres", "neo4j", "nebula"])
def test_resolve_graph_backend_type_reads_from_pydantic_attr(backend: str):
    """A pydantic-shaped ``CollectionConfig`` exposes
    ``graph_backend_type`` as an attribute; the resolver reads it
    straight off."""

    from aperag.indexing.worker_factory import _resolve_graph_backend_type

    class _Config:
        graph_backend_type = backend

    assert _resolve_graph_backend_type(_make_collection_stub(config_obj=_Config())) == backend


def test_resolve_graph_backend_type_reads_from_dict_config():
    """``Collection.config`` may be persisted as a JSON dict;
    the resolver also handles the Mapping shape."""

    from aperag.indexing.worker_factory import _resolve_graph_backend_type

    cfg = {"graph_backend_type": "neo4j"}
    assert _resolve_graph_backend_type(_make_collection_stub(config_obj=cfg)) == "neo4j"


def test_resolve_graph_backend_type_reads_from_json_string():
    """Some legacy rows persisted ``Collection.config`` as a JSON
    string; the resolver decodes it."""

    from aperag.indexing.worker_factory import _resolve_graph_backend_type

    cfg = '{"graph_backend_type": "nebula"}'
    assert _resolve_graph_backend_type(_make_collection_stub(config_obj=cfg)) == "nebula"


def test_resolve_graph_backend_type_rejects_unknown():
    """Typos / unsupported backends raise a clear
    :class:`WorkerFactoryError` so the orchestrator finalises FAILED
    with operator-facing diagnostics."""

    from aperag.indexing.worker_factory import _resolve_graph_backend_type

    class _Config:
        graph_backend_type = "duckdb"

    with pytest.raises(WorkerFactoryError) as exc:
        _resolve_graph_backend_type(_make_collection_stub(config_obj=_Config()))
    assert "duckdb" in str(exc.value)
    assert "postgres" in str(exc.value)


def test_resolve_entity_lock_returns_inmemory_for_postgres_and_neo4j():
    """Postgres + Neo4j use single-statement strip-then-append under
    native row locks, so an in-process :class:`InMemoryEntityLock` is
    sufficient. Tests asserting "no Redis dependency for these
    backends" pin the architect msg=f2921ae0 invariant.
    """

    from aperag.indexing.graph import InMemoryEntityLock
    from aperag.indexing.worker_factory import (
        _reset_graph_backend_singletons_for_tests,
        _resolve_entity_lock,
    )

    _reset_graph_backend_singletons_for_tests()
    try:
        for backend in ("postgres", "neo4j"):
            lock = _resolve_entity_lock(backend_type=backend)
            assert isinstance(lock, InMemoryEntityLock), (
                f"backend={backend} must use InMemoryEntityLock (no Redis dependency)"
            )
    finally:
        _reset_graph_backend_singletons_for_tests()


def test_build_graph_worker_raises_when_completion_model_missing(monkeypatch: pytest.MonkeyPatch):
    """Post-T1 invariant: the chunk 4b ``"Wave 4 wiring T1"``
    extractor-symbol-identity gate has self-disabled because T1 wired
    the real LLM extractor. The new gate covers a Wave 3 lesson #10
    failure mode: a collection that opts into knowledge graph but
    lacks a configured completion model now surfaces a clear
    :class:`WorkerFactoryError` from the extractor builder so the
    orchestrator finalises the graph row FAILED with operator-facing
    diagnostics (no silent ACTIVE-with-empty-graph).
    """

    from aperag.indexing import worker_factory as wf
    from aperag.indexing.graph import InMemoryEntityLock, InMemoryLineageGraphStore

    # Stub the backend dispatch so the extractor builder is reached
    # without needing a live Postgres / Neo4j / Nebula. The collection
    # stub has no ``completion`` config so ``build_collection_llm_callable``
    # fails — the extractor builder wraps it in WorkerFactoryError.
    wf._reset_graph_backend_singletons_for_tests()
    monkeypatch.setattr(
        wf,
        "_build_lineage_graph_store",
        lambda *, backend_type, collection: InMemoryLineageGraphStore(),
    )
    monkeypatch.setattr(
        wf,
        "_resolve_entity_lock",
        lambda *, backend_type: InMemoryEntityLock(),
    )

    class _Config:
        graph_backend_type = "postgres"

    collection = _make_collection_stub(config_obj=_Config())
    payload = DispatchPayload(
        index_id=1,
        document_id="doc",
        parse_version="parse-v1",
        modality=Modality.GRAPH,
        source_path="source/path",
        collection_id="col-4b",
    )

    # Stub tenant-scope-key resolution so the extractor builder is the
    # only failure mode reachable.
    monkeypatch.setattr(wf, "_resolve_tenant_scope_key", lambda *, payload: "user:t")

    with pytest.raises(WorkerFactoryError) as exc:
        wf._build_graph_worker(collection=collection, object_store=object(), payload=payload)
    msg = str(exc.value)
    assert "completion model" in msg or "graph extractor" in msg


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


# ---------------------------------------------------------------------
# Wave 4 T9 — fulltext_backend_type dispatch (mirrors chunk 4b shape).
# ---------------------------------------------------------------------


def test_resolve_fulltext_backend_type_defaults_to_elasticsearch():
    """A collection with no ``config`` or no ``fulltext_backend_type``
    field falls back to ``elasticsearch`` so existing collections (created
    before T9) keep their pre-T9 behaviour without any migration step.
    """

    from aperag.indexing.worker_factory import _resolve_fulltext_backend_type

    assert _resolve_fulltext_backend_type(_make_collection_stub(config_obj=None)) == "elasticsearch"

    class _ConfigNoBackend:
        fulltext_backend_type = None

    assert _resolve_fulltext_backend_type(_make_collection_stub(config_obj=_ConfigNoBackend())) == "elasticsearch"


@pytest.mark.parametrize("backend", ["elasticsearch", "opensearch"])
def test_resolve_fulltext_backend_type_reads_from_pydantic_attr(backend: str):
    """A pydantic-shaped ``CollectionConfig`` exposes
    ``fulltext_backend_type`` as an attribute; the resolver reads it
    straight off."""

    from aperag.indexing.worker_factory import _resolve_fulltext_backend_type

    class _Config:
        fulltext_backend_type = backend

    assert _resolve_fulltext_backend_type(_make_collection_stub(config_obj=_Config())) == backend


def test_resolve_fulltext_backend_type_reads_from_dict_config():
    """Dict-shaped ``Collection.config`` also resolves; mirrors the
    chunk 4b graph dispatch handling for legacy persisted forms."""

    from aperag.indexing.worker_factory import _resolve_fulltext_backend_type

    cfg = {"fulltext_backend_type": "opensearch"}
    assert _resolve_fulltext_backend_type(_make_collection_stub(config_obj=cfg)) == "opensearch"


def test_resolve_fulltext_backend_type_reads_from_json_string():
    """JSON-string ``Collection.config`` (legacy persisted shape)
    decoded defensively just like the graph dispatch resolver."""

    from aperag.indexing.worker_factory import _resolve_fulltext_backend_type

    cfg = '{"fulltext_backend_type": "opensearch"}'
    assert _resolve_fulltext_backend_type(_make_collection_stub(config_obj=cfg)) == "opensearch"


def test_resolve_fulltext_backend_type_rejects_unknown():
    """Unknown values raise a clear :class:`WorkerFactoryError` with
    the supported backends embedded so the operator can fix the
    collection config without log-spelunking."""

    from aperag.indexing.worker_factory import _resolve_fulltext_backend_type

    class _Config:
        fulltext_backend_type = "meilisearch"

    with pytest.raises(WorkerFactoryError) as exc:
        _resolve_fulltext_backend_type(_make_collection_stub(config_obj=_Config()))
    assert "meilisearch" in str(exc.value)
    assert "elasticsearch" in str(exc.value)


def test_build_fulltext_backend_dispatches_to_elasticsearch(monkeypatch: pytest.MonkeyPatch):
    """``backend_type=elasticsearch`` constructs the
    :class:`Elasticsearch` client and wraps it in the shared
    ``_ElasticsearchFulltextBackend`` adapter. Patches the client
    constructor so the test does not need a live ES cluster.
    """

    import aperag.indexing.worker_factory as wf

    class _FakeES:
        def __init__(self, host, **kwargs):
            self.host = host
            self.kwargs = kwargs

    monkeypatch.setattr(wf, "_build_elasticsearch_client", lambda: _FakeES("http://es:9200"))

    backend = wf._build_fulltext_backend(
        backend_type="elasticsearch",
        index_name="aperag_doc_col-1",
    )
    assert isinstance(backend, wf._ElasticsearchFulltextBackend)
    assert backend._index == "aperag_doc_col-1"
    assert isinstance(backend._client, _FakeES)


def test_build_fulltext_backend_opensearch_gates_on_missing_driver(monkeypatch: pytest.MonkeyPatch):
    """``backend_type=opensearch`` requires the optional ``opensearch-py``
    dependency. When it is absent (the default for this repo's lock
    file) the factory raises a clear :class:`WorkerFactoryError`
    pointing operators at the ``fulltext-opensearch`` extra — mirrors
    the ``graph-neo4j`` / ``graph-nebula`` extras gating in chunk 4b.
    """

    import sys

    import aperag.indexing.worker_factory as wf
    from aperag.config import settings as _settings

    monkeypatch.setattr(_settings, "es_host", "http://os:9200", raising=False)
    # Force the lazy import to fail even if the test host happens to
    # have ``opensearch-py`` installed (CI runners usually do not).
    monkeypatch.setitem(sys.modules, "opensearchpy", None)

    with pytest.raises(WorkerFactoryError) as exc:
        wf._build_fulltext_backend(
            backend_type="opensearch",
            index_name="aperag_doc_col-1",
        )
    assert "opensearch-py" in str(exc.value)
    assert "fulltext-opensearch" in str(exc.value)


def test_build_fulltext_backend_elasticsearch_requires_es_host(monkeypatch: pytest.MonkeyPatch):
    """Both backends gate on ``settings.es_host`` since the same env
    variable feeds either driver. An unset ``ES_HOST`` raises a clear
    :class:`WorkerFactoryError` so the operator never gets a confusing
    half-built client.
    """

    import aperag.indexing.worker_factory as wf
    from aperag.config import settings as _settings

    monkeypatch.setattr(_settings, "es_host", "", raising=False)

    with pytest.raises(WorkerFactoryError) as exc:
        wf._build_fulltext_backend(
            backend_type="elasticsearch",
            index_name="aperag_doc_col-1",
        )
    assert "ES_HOST" in str(exc.value)
    assert "elasticsearch" in str(exc.value)
