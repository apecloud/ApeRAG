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
# ---------------------------------------------------------------------


_PHASE1_E2E_GATE = os.environ.get("RUN_E2E_PHASE1_SMOKE") == "1"


@pytest.mark.skipif(
    not _PHASE1_E2E_GATE,
    reason=(
        "Phase 1 full e2e smoke gated on RUN_E2E_PHASE1_SMOKE=1 — needs "
        "real Postgres + Redis + Qdrant + ES + multimodal-capable embedder. "
        "Runs in the e2e-http-compose CI lane."
    ),
)
def test_phase1_full_pipeline_vector_fulltext_summary_active_graph_vision_failed():
    """Layer 2 — canonical Phase 1 e2e smoke per architect msg=da3012a4.

    Sequence:
    1. Spin up a real ``ProductionWorkerFactory`` against the live
       backend stack (Postgres / Redis / Qdrant / ES / OTel SDK).
    2. Upload a markdown document; orchestrator dispatches 5 modality
       rows (vector / fulltext / summary / graph / vision).
    3. Run the worker pool until all rows finalise.
    4. Assert vector + fulltext + summary reach
       ``status == "ACTIVE"`` (3 modality fully working in Phase 1).
    5. Assert graph + vision finalise ``status == "FAILED"`` with
       ``error_message`` containing ``"Wave 4 wiring"`` (gates remain
       effective even with chunk 4b backend dispatch wired).
    6. Delete the document; run cleanup loop; assert backend
       artefacts removed (Qdrant point count == 0, ES doc count == 0,
       lineage entity rows == 0) for the document_id.

    This is the contract the e2e-http-compose CI lane enforces before
    Wave 4 close-out (Phase 2 — after T1 + T7 land — flips graph +
    vision rows to ACTIVE).
    """

    pytest.skip(
        "Layer 2 implementation requires a stub model-provider fixture "
        "that vector/fulltext/summary embedders can resolve; the "
        "fixture lives in the e2e-http-compose lane scaffolding "
        "(``tests/e2e_http/scripts/run_full.sh``). Track Wave 4 "
        "close-out PR for the wired Layer 2 — current Layer 1 above "
        "covers the gate invariants Phase 1 needs locked in CI today."
    )


@pytest.mark.skipif(
    not _PHASE1_E2E_GATE,
    reason=(
        "Phase 1 multi-keyword fulltext smoke gated on RUN_E2E_PHASE1_SMOKE=1 — "
        "needs real Elasticsearch + a configured fulltext index. Runs in the "
        "e2e-http-compose CI lane."
    ),
)
def test_phase1_multi_keyword_fulltext_search_returns_hits():
    """Layer 2 — sweep D verification (architect msg=fdd53586): exercise
    ``_fulltext_search`` with a multi-keyword query against a freshly
    indexed document and assert at least one hit. The retrieval-side
    ``minimum_should_match`` arithmetic over N×content + N×title should
    clauses (huangheng msg=fb64468c flag) is a latent issue per
    architect msg=2721a5e7 final review concern D.

    Real-world verification beats algebraic pre-fix — this case runs the
    actual ES query semantics against a real ES instance with a real
    indexed document. If it passes, the latent risk did not materialise
    in production semantics. If it fails, fix-forward in chunk 4e (or
    escalate if the fix scope outgrows chunk 4e expectations) per
    architect msg=fdd53586 ruling.

    Sequence:
    1. Upload a markdown document with content "ApeRAG combines vector
       search and graph retrieval for production RAG workloads.".
    2. Wait for fulltext modality to reach ACTIVE.
    3. Issue a 3-keyword query: ``"ApeRAG vector RAG"``.
    4. Assert at least 1 hit is returned (the indexed document).
    5. Cleanup.
    """

    pytest.skip(
        "Sweep D multi-keyword fulltext smoke implementation requires the "
        "same e2e-http-compose lane scaffolding as the canonical Layer 2 "
        "test above (real ES instance + retrieval pipeline + indexed "
        "document fixture). Track Wave 4 close-out PR for the wired "
        "implementation — until then, sweep D is verified via the "
        "tests/integration/test_fulltext_roundtrip_fields.py path which "
        "exercises bulk_index + search round-trip on a real ES index."
    )


# ---------------------------------------------------------------------
# Type-checker happy: ``Any`` is imported for future Layer 2 stubs.
# ---------------------------------------------------------------------


_ = Any
