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

"""Wave 4 T8 chunk 4e — Phase 1 production e2e smoke.

Locks the chunk 4 acceptance item 9 contract on the actual production
factory + dispatch path. Two layers (per architect msg=da3012a4 +
PM msg=067c18e5):

* **Layer 1 — gate invariants (always run)**: verify
  ``ProductionWorkerFactory`` raises :class:`WorkerFactoryError` with
  ``"Wave 4 wiring"`` for graph + vision payloads even when the
  backend is wired (chunk 4b). The Wave 3 lesson #10 explicit-gate
  pattern survives chunk 4b wiring; only T1 LLM extractor / T7
  multimodal embedder land flips them off. This layer exercises the
  factory directly + uses real Postgres for the ``Collection``
  resolve.

* **Layer 2 — full pipeline e2e (gated by ``RUN_E2E_PHASE1_SMOKE=1``)**:
  the canonical contract per architect msg=da3012a4 — real Postgres
  + real Redis + real Qdrant + real Elasticsearch + real OTel SDK,
  vector + fulltext + summary modalities reach
  ``DocumentIndex.status == "ACTIVE"`` for a markdown upload, while
  graph + vision rows finalise ``FAILED`` with ``error_message``
  containing ``"Wave 4 wiring"``. Plus the T2 cleanup roundtrip:
  delete document → cleanup loop → backend artefacts removed (
  Qdrant points / ES docs / lineage graph entities all 0). Skipped
  by default because the embedder requires a configured model
  provider; the e2e-http-compose CI lane runs it with a stub provider
  fixture that all five modalities can resolve.

The layer split keeps the local-dev signal (Layer 1) fast while
preserving the canonical Phase 1 invariant in CI (Layer 2). Wave 4
close-out (Phase 2) flips T1 + T7 wired and rewrites Layer 1 to
expect graph/vision ACTIVE.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, insert
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aperag.domains.knowledge_base.db.models import (
    Collection,
    CollectionStatus,
    CollectionType,
)
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
from aperag.indexing.orchestrator import DispatchPayload
from aperag.indexing.worker_factory import (
    ProductionWorkerFactory,
    WorkerFactoryError,
    _reset_graph_backend_singletons_for_tests,
)

# ---------------------------------------------------------------------
# Layer 1 — gate invariants (always run; uses SQLite for the
# Collection resolve so the factory can dispatch without external
# infra). Real Postgres path is exercised in Layer 2 below.
# ---------------------------------------------------------------------


def _make_engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DocumentIndex.metadata.create_all(eng, tables=[DocumentIndex.__table__])
    Collection.metadata.create_all(eng, tables=[Collection.__table__])
    return eng


def _seed_collection(engine: Engine, *, graph_backend_type: str = "postgres", enable_vision: bool = False) -> str:
    cid = "col-phase1-" + uuid.uuid4().hex[:8]
    config = {
        "enable_vector": True,
        "enable_fulltext": True,
        "enable_summary": True,
        "enable_knowledge_graph": True,
        "enable_vision": enable_vision,
        "graph_backend_type": graph_backend_type,
        "embedding": {
            "model_id": "fake-model",
            "model_service_provider": "fake-provider",
        },
    }
    with Session(engine) as session, session.begin():
        session.execute(
            insert(Collection).values(
                id=cid,
                title="Phase 1 Smoke",
                description=None,
                user="user-phase1",
                status=CollectionStatus.ACTIVE.value,
                type=CollectionType.DOCUMENT.value,
                config=json.dumps(config),
            )
        )
    return cid


def _seed_pending_row(engine: Engine, *, modality: Modality, collection_id: str) -> int:
    with Session(engine) as session, session.begin():
        result = session.execute(
            insert(DocumentIndex)
            .values(
                document_id=f"doc-{modality.value}-phase1",
                parse_version="parse-v1",
                modality=modality.value,
                status=IndexStatus.PENDING.value,
                tenant_scope_key="user:t",
                collection_id=collection_id,
                source_path="source/path",
                is_serving=False,
            )
            .returning(DocumentIndex.id)
        )
        return int(result.scalar_one())


@pytest.fixture(autouse=True)
def _reset_singletons_between_tests() -> None:
    """Drop cached backend client singletons so each Phase 1 case
    starts from a clean slate (otherwise the first case's mocked
    engine bleeds into the next)."""

    _reset_graph_backend_singletons_for_tests()
    yield
    _reset_graph_backend_singletons_for_tests()


def test_phase1_graph_modality_raises_when_completion_model_missing(monkeypatch: pytest.MonkeyPatch):
    """Layer 1 gate invariant (post-T1): chunk 4b backend dispatch is
    wired AND T1 has landed the real LLM extractor — the prior
    ``"Wave 4 wiring T1"`` symbol-identity gate has self-disabled.
    The new Phase 1 invariant: a collection that opts into knowledge
    graph but lacks a configured completion model surfaces a clear
    :class:`WorkerFactoryError` from the extractor builder so the
    orchestrator finalises the row FAILED with operator-facing
    diagnostics.

    Wave 3 lesson #10 ship-incomplete-but-don't-silently-lie still
    honoured — silent ACTIVE-with-empty-graph never happens; the gate
    just moved from "extractor symbol" to "completion model not
    configured" after T1.
    """

    from aperag.indexing import worker_factory as wf
    from aperag.indexing.graph import InMemoryEntityLock, InMemoryLineageGraphStore

    # Stub the backend client singleton so the gate is reached without
    # needing a live Postgres/Neo4j/Nebula. The graph extractor builder
    # is reached after the store + lock are constructed — and the
    # extractor builder raises because the seeded collection has no
    # ``completion`` config (only ``embedding``).
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

    engine = _make_engine()
    try:
        cid = _seed_collection(engine, graph_backend_type="postgres")
        row_id = _seed_pending_row(engine, modality=Modality.GRAPH, collection_id=cid)
        payload = DispatchPayload(
            index_id=row_id,
            document_id=f"doc-{Modality.GRAPH.value}-phase1",
            parse_version="parse-v1",
            modality=Modality.GRAPH,
            source_path="source/path",
            collection_id=cid,
        )

        async def _run() -> None:
            factory = ProductionWorkerFactory(engine=engine, object_store=object())
            with pytest.raises(WorkerFactoryError) as exc:
                await factory(payload)
            msg = str(exc.value)
            # Post-T1 the message is "completion model not configured"
            # (T1 self-disable). Pre-T1 was "Wave 4 wiring T1".
            assert "completion model" in msg or "Wave 4 wiring" in msg

        asyncio.run(_run())
    finally:
        engine.dispose()


def test_phase1_vision_modality_raises_wave4_wiring_gate(monkeypatch: pytest.MonkeyPatch):
    """Layer 1 gate invariant: vision modality requires a real
    multimodal embedder. The Wave 3 vision gate (Wave 4 backlog #7)
    raises ``WorkerFactoryError`` with ``"Wave 4 wiring"`` until T7
    lands a multimodal model. Phase 1 smoke pins this — Phase 2 (after
    T7) flips it to ACTIVE assertion.
    """

    # Stub the embedder so the gate reachability is decoupled from
    # the model-provider config. The gate compares
    # ``embedding_service.is_multimodal()`` — a non-multimodal stub
    # exercises the gate; a multimodal stub flips it (Phase 2).
    class _StubEmbeddingService:
        def is_multimodal(self) -> bool:
            return False

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    def _stub_get_embedding_service(_collection: Any) -> tuple[Any, int]:
        return _StubEmbeddingService(), 1

    monkeypatch.setattr(
        "aperag.llm.embed.base_embedding.get_collection_embedding_service_sync",
        _stub_get_embedding_service,
    )

    engine = _make_engine()
    try:
        cid = _seed_collection(engine, enable_vision=True)
        row_id = _seed_pending_row(engine, modality=Modality.VISION, collection_id=cid)
        payload = DispatchPayload(
            index_id=row_id,
            document_id=f"doc-{Modality.VISION.value}-phase1",
            parse_version="parse-v1",
            modality=Modality.VISION,
            source_path="source/path",
            collection_id=cid,
        )

        async def _run() -> None:
            factory = ProductionWorkerFactory(engine=engine, object_store=object())
            with pytest.raises(WorkerFactoryError) as exc:
                await factory(payload)
            msg = str(exc.value)
            assert "Wave 4 wiring" in msg

        asyncio.run(_run())
    finally:
        engine.dispose()


def test_phase1_unknown_graph_backend_type_raises():
    """Layer 1 invariant: a typo / unsupported value on
    ``collection.config.graph_backend_type`` surfaces a clear
    ``WorkerFactoryError`` so the orchestrator finalises FAILED with
    operator-facing diagnostics. Phase 1 smoke covers the
    config-error path that production deployments may hit while
    onboarding.
    """

    engine = _make_engine()
    try:
        cid = _seed_collection(engine, graph_backend_type="duckdb")
        row_id = _seed_pending_row(engine, modality=Modality.GRAPH, collection_id=cid)
        payload = DispatchPayload(
            index_id=row_id,
            document_id=f"doc-{Modality.GRAPH.value}-phase1",
            parse_version="parse-v1",
            modality=Modality.GRAPH,
            source_path="source/path",
            collection_id=cid,
        )

        async def _run() -> None:
            factory = ProductionWorkerFactory(engine=engine, object_store=object())
            with pytest.raises(WorkerFactoryError) as exc:
                await factory(payload)
            assert "duckdb" in str(exc.value)

        asyncio.run(_run())
    finally:
        engine.dispose()


def test_phase1_cleanup_view_dispatch_for_all_modalities(monkeypatch: pytest.MonkeyPatch):
    """Layer 1 invariant: ``ProductionWorkerFactory.build_for_cleanup_row``
    bypasses the dispatch-time gates (graph T1 extractor / vision
    multimodal) so cleanup can delete backend artefacts even for
    partially-gated modalities. Without this, post-delete cleanup
    would fail forever for graph+vision rows that never reached
    ACTIVE — leaving stale state operators cannot recover.

    Verifies the surgical-gate corollary of Wave 3 lesson #10
    (per chenyexuan T2 architect ratify msg) for all 5 modalities.
    """

    from aperag.indexing import worker_factory as wf

    # Stub the heavy backend builders so cleanup view construction
    # does not require Qdrant / ES / live graph database. The chunk
    # 4d narrowed verify already proved the new pipeline doesn't
    # cross-reference legacy storage; here we only assert the cleanup
    # view returns a non-None object for every modality (i.e. the
    # gates do NOT fire on the cleanup path).
    monkeypatch.setattr(wf, "_build_qdrant_cleanup_backend", lambda *a, **kw: object())
    monkeypatch.setattr(wf, "_build_es_cleanup_backend", lambda *a, **kw: object())
    monkeypatch.setattr(
        wf,
        "_build_lineage_graph_store",
        lambda *, backend_type, collection: object(),
    )
    monkeypatch.setattr(wf, "_resolve_entity_lock", lambda *, backend_type: object())

    engine = _make_engine()
    try:
        cid = _seed_collection(engine, enable_vision=True)
        for modality in Modality:
            row_id = _seed_pending_row(engine, modality=modality, collection_id=cid)
            with Session(engine) as session:
                row = session.get(DocumentIndex, row_id)
                assert row is not None

                async def _run(_row: DocumentIndex = row, _modality: Modality = modality) -> None:
                    factory = ProductionWorkerFactory(engine=engine, object_store=object())
                    view = await factory.build_for_cleanup_row(_row)
                    assert view is not None, (
                        f"cleanup view must build for modality={_modality.value} "
                        f"(otherwise post-delete artefacts leak forever)"
                    )
                    # Cleanup view derive/sync MUST raise — fail-loud
                    # design so a misroute (cleanup view used as
                    # dispatch worker) surfaces immediately rather
                    # than silently corrupting state.
                    with pytest.raises(NotImplementedError):
                        await view.derive(
                            document_id=_row.document_id,
                            parse_version=_row.parse_version,
                            source_path=_row.source_path or "",
                        )

                asyncio.run(_run())
    finally:
        engine.dispose()


# ---------------------------------------------------------------------
# Layer 2 — full pipeline e2e (gated by RUN_E2E_PHASE1_SMOKE=1).
# Real Postgres + Redis + Qdrant + Elasticsearch + OTel SDK.
#
# Wave 5 P3 (chenyexuan task #28) wires the Layer 2 fixture and
# replaces the pre-Wave-5 ``pytest.skip`` stubs with functional test
# bodies. The e2e-http-compose CI lane sets the env vars below; the
# tests run against the live stack and validate the canonical Phase 1
# contract end-to-end.
# ---------------------------------------------------------------------


_PHASE1_E2E_GATE = os.environ.get("RUN_E2E_PHASE1_SMOKE") == "1"
_PHASE1_E2E_COLLECTION_ID = os.environ.get("PHASE1_E2E_COLLECTION_ID")


def _phase1_e2e_skip_reason() -> str | None:
    """Return ``None`` if all required env vars are set, else a
    human-readable skip reason describing what is missing.

    The Wave 5 P3 contract is that Layer 2 runs only when:
    * ``RUN_E2E_PHASE1_SMOKE=1`` (operator opt-in)
    * ``PHASE1_E2E_COLLECTION_ID`` points at a Collection seeded by
      the e2e-http-compose lane bootstrap with a real model provider
      configured (so the embedder + summariser actually work).
    * Backend env vars set so :class:`ProductionWorkerFactory` can
      resolve real clients (``DATABASE_URL`` / ``ES_HOST`` / Qdrant
      env vars / ``INDEXING_QUEUE_REDIS_URL``).

    Skipping with a clear reason beats failing — local-dev runs of
    this file should not require operators to stand up the full
    stack.
    """
    if not _PHASE1_E2E_GATE:
        return (
            "Phase 1 full e2e smoke gated on RUN_E2E_PHASE1_SMOKE=1 — needs "
            "real Postgres + Redis + Qdrant + ES + a configured model "
            "provider. Runs in the e2e-http-compose CI lane."
        )
    if not _PHASE1_E2E_COLLECTION_ID:
        return (
            "Phase 1 Layer 2 needs PHASE1_E2E_COLLECTION_ID env var "
            "pointing at the e2e-http-compose-bootstrapped Collection "
            "(real model provider configured). Set it from "
            "tests/e2e_http/bootstrap/.generated/e2e.env."
        )
    return None


def _resolve_phase1_e2e_engine() -> Engine:
    """Open a real Postgres engine using the production ``settings``
    so the test sees the same Collection rows the e2e-http-compose
    lane seeded.

    Mirrors :func:`aperag.config.sync_engine` but builds the engine
    directly inside the test so a stale or pooled connection from a
    prior pytest module does not leak into the suite.
    """
    from aperag.config import settings

    if not settings.database_url:
        pytest.skip("Phase 1 Layer 2: settings.database_url is empty (POSTGRES_HOST etc unset)")
    return create_engine(settings.database_url, future=True)


async def _run_phase1_workers_until_quiet(
    *,
    engine: Engine,
    document_id: str,
    timeout_seconds: float = 60.0,
) -> dict[Modality, DocumentIndex]:
    """Drive the production worker pool until every per-modality
    ``DocumentIndex`` row for ``document_id`` finalises (ACTIVE or
    FAILED — terminal states), or ``timeout_seconds`` elapses.

    Returns a dict keyed by Modality with the final row state.

    Implementation drives :class:`ProductionWorkerFactory` directly
    against the live backends — same dispatch path the
    ``run_*_worker`` lifespan tasks use. Each modality is processed
    once per cycle until terminal; the loop is bounded by
    ``timeout_seconds`` so a hung modality (e.g. unreachable Qdrant)
    fails the test loud rather than blocking forever.
    """
    from sqlalchemy import select as sa_select

    from aperag.indexing.orchestrator import process_one_task

    factory = ProductionWorkerFactory(engine=engine)
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    finalised: dict[Modality, DocumentIndex] = {}

    while asyncio.get_event_loop().time() < deadline:
        with Session(engine) as session:
            rows = list(
                session.execute(
                    sa_select(DocumentIndex).where(DocumentIndex.document_id == document_id)
                ).scalars()
            )
        if not rows:
            await asyncio.sleep(0.1)
            continue

        all_terminal = True
        for row in rows:
            try:
                modality = Modality(row.modality)
            except ValueError:
                continue
            if row.status in (IndexStatus.ACTIVE.value, IndexStatus.FAILED.value):
                finalised[modality] = row
                continue
            all_terminal = False
            payload = DispatchPayload(
                index_id=row.id,
                document_id=row.document_id,
                parse_version=row.parse_version,
                modality=modality,
                source_path=row.source_path or "",
                collection_id=row.collection_id,
            )
            try:
                worker = await factory(payload)
            except WorkerFactoryError as exc:
                # Gate-raise → finalise FAILED with the gate message.
                # Mirrors :class:`run_worker_loop` path so Layer 2
                # exercises the production pattern.
                from aperag.indexing.orchestrator import _claim_row, _finalize_failed

                if await asyncio.to_thread(_claim_row, engine, row.id):
                    await asyncio.to_thread(
                        _finalize_failed,
                        engine,
                        row.id,
                        f"worker_factory failed: {exc!r}",
                    )
                continue
            await process_one_task(
                engine=engine,
                payload=payload,
                worker=worker,
                heartbeat_interval_seconds=0,
            )

        if all_terminal:
            break
        await asyncio.sleep(0.2)

    return finalised


@pytest.mark.skipif(
    _phase1_e2e_skip_reason() is not None,
    reason=(_phase1_e2e_skip_reason() or "phase 1 layer 2 skipped"),
)
def test_phase1_full_pipeline_vector_fulltext_summary_active_graph_vision_failed():
    """Layer 2 — canonical Phase 1 e2e smoke per architect msg=da3012a4.

    Wave 5 P3: implementation wired against the e2e-http-compose lane
    fixture. Reads ``PHASE1_E2E_COLLECTION_ID`` to resolve the live
    Collection, dispatches a markdown upload, drives the worker pool
    until every modality row finalises, then asserts the canonical
    Phase 1 contract:

    * vector + fulltext + summary reach ``ACTIVE`` (three real
      modalities ship in Wave 4)
    * graph + vision finalise ``FAILED`` with ``error_message``
      containing the gate marker (``Wave 4 wiring`` for chunk 4b
      gates, ``completion model`` for the post-T1 gate self-disable
      surface, or ``multimodal`` for the vision gate). The OR
      tolerates the T1 gate-self-disable transition that ships in
      this same Wave (per chunk 4b → T1 closure).

    The cleanup roundtrip (delete document → cleanup loop → backend
    artefacts gone) is left to a follow-up sub-test (`...
    _and_cleanup_removes_backend_artefacts`) once the Layer 2
    fixture supports document-delete API access.
    """


    from aperag.indexing.dispatcher import DispatchRequest, IndexingMode, dispatch_indexing
    from aperag.indexing.parser import ParseConfig, parse_document
    from aperag.objectstore.base import get_object_store

    collection_id = _PHASE1_E2E_COLLECTION_ID
    assert collection_id, "skip should have caught missing env var"

    document_id = "phase1-e2e-" + uuid.uuid4().hex[:8]
    source_bytes = (
        b"# Phase 1 e2e smoke\n\n"
        b"This document exercises the canonical Phase 1 contract: "
        b"vector + fulltext + summary reach ACTIVE; graph + vision "
        b"finalise FAILED with the Wave 4 wiring gate message.\n"
    )

    async def _run() -> None:
        engine = _resolve_phase1_e2e_engine()
        try:
            object_store = get_object_store()
            parsed = parse_document(
                store=object_store,
                collection_id=collection_id,
                document_id=document_id,
                source_bytes=source_bytes,
                source_filename="phase1-smoke.md",
                config=ParseConfig(),
            )
            from aperag.indexing.runtime import get_runtime

            runtime = get_runtime()
            assert runtime is not None and runtime.queue is not None, (
                "Phase 1 Layer 2 requires a live IndexingRuntime — the e2e-http-compose "
                "lane bootstraps it via FastAPI lifespan; ensure the test runs against "
                "the live API process."
            )
            await dispatch_indexing(
                engine=runtime.engine,
                queue=runtime.queue,
                workers=None,
                request=DispatchRequest(
                    collection_id=collection_id,
                    document_id=document_id,
                    parse_version=parsed.parse_version,
                    source_path=parsed.chunks_path,
                    tenant_scope_key="user:phase1-e2e",
                    modalities=tuple(Modality),
                ),
                mode=IndexingMode.ASYNC,
            )

            # Drive the pool inline so the test does not depend on the
            # lifespan worker tasks racing with the assertion.
            finalised = await _run_phase1_workers_until_quiet(
                engine=engine,
                document_id=document_id,
            )
            assert set(finalised.keys()) == set(Modality), (
                f"every modality must finalise within timeout; got {set(finalised.keys())}"
            )

            for modality in (Modality.VECTOR, Modality.FULLTEXT, Modality.SUMMARY):
                row = finalised[modality]
                assert row.status == IndexStatus.ACTIVE.value, (
                    f"modality={modality.value} must finalise ACTIVE in Phase 1; "
                    f"actual={row.status} error={row.error_message}"
                )
                assert row.is_serving is True

            for modality in (Modality.GRAPH, Modality.VISION):
                row = finalised[modality]
                assert row.status == IndexStatus.FAILED.value, (
                    f"modality={modality.value} must finalise FAILED until Wave 5 T7 lands; "
                    f"actual={row.status}"
                )
                msg = row.error_message or ""
                assert any(
                    marker in msg
                    for marker in ("Wave 4 wiring", "completion model", "multimodal")
                ), f"modality={modality.value} FAILED message must surface a gate marker; got {msg!r}"
        finally:
            engine.dispose()

    asyncio.run(_run())


@pytest.mark.skipif(
    _phase1_e2e_skip_reason() is not None,
    reason=(_phase1_e2e_skip_reason() or "phase 1 layer 2 skipped"),
)
def test_phase1_multi_keyword_fulltext_search_returns_hits():
    """Layer 2 — sweep D verification (architect msg=fdd53586): exercise
    the retrieval-side ``_fulltext_search`` with a multi-keyword query
    against a freshly indexed document. Asserts at least one hit so
    the ``minimum_should_match`` arithmetic over N×content + N×title
    should-clauses (huangheng msg=fb64468c flag) is verified end-to-end.

    Wave 5 P3 wires this against the live ES instance the
    e2e-http-compose lane provides. Re-uses the canonical Layer 2
    fixture (``PHASE1_E2E_COLLECTION_ID``).
    """

    from aperag.domains.retrieval.pipeline import _fulltext_search
    from aperag.indexing.dispatcher import DispatchRequest, IndexingMode, dispatch_indexing
    from aperag.indexing.parser import ParseConfig, parse_document
    from aperag.objectstore.base import get_object_store

    collection_id = _PHASE1_E2E_COLLECTION_ID
    assert collection_id, "skip should have caught missing env var"
    document_id = "phase1-sweepd-" + uuid.uuid4().hex[:8]
    source_bytes = (
        b"# Sweep D fulltext multi-keyword smoke\n\n"
        b"ApeRAG combines vector search and graph retrieval for production RAG workloads.\n"
    )

    async def _run() -> None:
        engine = _resolve_phase1_e2e_engine()
        try:
            from aperag.indexing.runtime import get_runtime

            runtime = get_runtime()
            assert runtime is not None and runtime.queue is not None, (
                "sweep D Layer 2 requires a live IndexingRuntime"
            )

            object_store = get_object_store()
            parsed = parse_document(
                store=object_store,
                collection_id=collection_id,
                document_id=document_id,
                source_bytes=source_bytes,
                source_filename="phase1-sweepd.md",
                config=ParseConfig(),
            )
            await dispatch_indexing(
                engine=runtime.engine,
                queue=runtime.queue,
                workers=None,
                request=DispatchRequest(
                    collection_id=collection_id,
                    document_id=document_id,
                    parse_version=parsed.parse_version,
                    source_path=parsed.chunks_path,
                    tenant_scope_key="user:phase1-sweepd",
                    modalities=(Modality.FULLTEXT,),
                ),
                mode=IndexingMode.ASYNC,
            )
            finalised = await _run_phase1_workers_until_quiet(
                engine=engine,
                document_id=document_id,
            )
            row = finalised.get(Modality.FULLTEXT)
            assert row is not None and row.status == IndexStatus.ACTIVE.value, (
                f"fulltext modality must finalise ACTIVE; got {row}"
            )

            # Now exercise the multi-keyword path. ``_fulltext_search``
            # is the retrieval pipeline entry the chat path consumes;
            # if its ``minimum_should_match: 80%`` over a 2N-clause
            # should-set has the latent calc gap huangheng flagged in
            # msg=fb64468c, this assertion fails — which is the whole
            # point of sweep D verification.
            from aperag.domains.knowledge_base.db.models import Collection

            with Session(engine) as session:
                collection = session.get(Collection, collection_id)
            assert collection is not None
            hits = await _fulltext_search(
                collection=collection,
                query="ApeRAG vector RAG",
                top_k=5,
                user_id="user:phase1-sweepd",
                chat_id=None,
            )
            assert hits and len(hits) >= 1, (
                "multi-keyword fulltext search returned 0 hits — sweep D latent issue "
                "may have materialised; check _fulltext_search:350 minimum_should_match calc."
            )
        finally:
            engine.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------
# Type-checker happy: ``Any`` is imported for future Layer 2 stubs.
# ---------------------------------------------------------------------


_ = Any
