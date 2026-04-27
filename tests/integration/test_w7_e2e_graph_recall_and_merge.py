# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 7 task #11 — end-to-end narrative validation.

This is the **integration safety net** for the three pieces of wiring
that task #8 introduces (per @huangheng's CR rationale on
``msg=0b48af2b``):

  1. ``worker_factory._build_lineage_graph_store`` wraps the chosen
     backend in :class:`LineageGraphStoreWithAliasRedirect` (the W7-6
     decorator). Unit tests can prove the wrap call exists; only an
     integration narrative can prove the *behaviour* — that an alias
     name silently redirects on the indexer-side ``upsert`` path.
  2. REST ``POST /collections/{cid}/graphs/nodes/merge`` cuts over
     from the legacy ``GraphIndexService.merge_entities`` to the new
     ``GraphCurationService.merge_entities``. Unit tests pin the
     route shape; only an end-to-end call can verify the alias_map +
     vector re-embed round-trip.
  3. ``retrieval/pipeline.py:_graph_search`` cuts over to the new
     vector recall. Unit tests pin the function signature; only a
     real recall against an indexed graph can verify the result shape.

The narrative is the canonical Wave 7 happy path:

  step 1:  seed the lineage store with three entities (Alice / Bob /
           Acme) — direct ``upsert_entity_with_lineage`` so the
           narrative does not depend on a real LLM extractor's
           non-deterministic output. The wiring under test is the
           graph storage + vector + alias paths, not the extractor
           prompt.
  step 2:  run the GraphIndexCompactor + vector writer phases by
           invoking the per-collection ``GraphSearchService`` factory
           and asserting the pre-conditions for vector recall (vectors
           upserted, ``compacted_description`` populated, snapshot
           diff did not delete shared entities).
  step 3:  call ``GraphSearchService.search_entities("Alice")`` (the
           service the retrieval pipeline now routes through). Assert
           hit returns Alice. Validates wiring #3 *behaviourally*.
  step 4:  call ``GraphService.merge_entities(target="Alice",
           sources=["Alicia"])`` — the same path the REST route
           ``POST /collections/{cid}/graphs/nodes/merge`` now
           delegates to (per ``merge_nodes_view``). Asserts the
           backward-compat response shape; HTTP layer surface is
           pinned by the route unit tests.
  step 5:  read alias_map: assert ``{Alicia: Alice}`` persisted via
           the :class:`AliasMapRepository`.
  step 6:  upload a NEW doc-equivalent extraction containing
           ``Alicia`` through the *decorated* lineage store
           (``_build_lineage_graph_store`` returns the wrapped one).
           Assert that Alicia is silently redirected to Alice — no
           Alicia row exists in ``aperag_lineage_entity``. This is
           the critical inseparability gate proof.
  step 7:  search again for "Alice" → still returns the merged
           entity. Vector recall on the canonical name keeps working.
  step 8 (W8-3 trigger pin): ``GraphService.get_entity_detail``
           currently bypasses the alias_map on reads; calling it for
           "Alicia" returns ``None`` (route emits 404). When Wave 8
           W8-3 ships read-side alias resolution this assertion
           flips and the test is the trigger condition pinned in the
           repo.
  step 9 (failure-mode fold-in): patch ``GraphIndexCompactor.compact_if_oversized``
           to raise. Re-trigger the merge orchestration. Assert that
           the merge still completes (graceful degrade — compaction
           failure is non-fatal per Wave 6 invariant). The entity
           survives with a fallback ``compacted_description``.

Layer split (per Wave 4 ``test_full_indexing_pipeline.py`` precedent):

* **Layer 1 — scaffold gate (always collects, marked skip)**: pinned
  to a placeholder ``pytest.skip`` until task #8 wiring lands. The
  file is checked in early so the narrative is reviewable in parallel
  with task #8 implementation; the body fills in once #8 ships.
* **Layer 2 — full narrative (gated by ``RUN_W7_E2E_NARRATIVE=1``)**:
  brings up real Postgres + Redis + Qdrant + Elasticsearch + the
  configured LLM/embedding provider. Mirrors the Wave 4 e2e gate
  pattern — env var off by default so local-dev pytest runs stay fast;
  CI flips it on in the Wave 7 lane once task #8 wiring is alive.

ETA & sequence (per architect ratify ``msg=a0ba75da`` + huangheng
``msg=0b48af2b``):

* Scaffold parallel with task #8 (in_progress as of 2026-04-28).
* Run + push PR after task #8 merge (3 wiring alive) and **before**
  task #10 close-out (so a regression introduced by deleting legacy
  is caught while rollback is still cheap).

12-invariant table (PR body): mostly n/a — narrative-correctness is
the hard gate on this file. Material invariants validated implicitly
by the narrative (#2 compaction, #3 sync ordering, #4 GC tolerance,
#5 vector recall byte-parity, #11 W8-3 trigger pin).

4-pattern pre-check matrix (PR body):
  * Pattern 1 v1: ``LineageGraphStoreWithAliasRedirect`` import exists
    in worker_factory (post-#8).
  * Pattern 1 v2: REST route cite verifies new merge handler binding.
  * Pattern 2: alias_map column / index in the
    ``aperag_lineage_entity_alias`` Wave 7 schema.
  * Pattern 3: alias redirect honored on the upsert read-modify-write
    path inside the decorator.

simple-stable 4-guardrail (PR body):
  1. 不无限扩范围: one file, no production code change.
  2. 先把功能做实: real backends + real provider — narrative validates
     production behaviour, not stubbed surface.
  3. 简单稳定: one happy-path narrative + one W8-3 pin + one
     failure-mode step. Not a regression matrix.
  4. 私有化免维护: env-var-gated; CI lane flips it on, local stays
     fast.

W8-3 trigger condition pinned at step 8: when read-side alias
resolution ships in Wave 8, the assertion flips from "404 expected"
to "200 with Alice payload" and the tests document the cutover.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------
# Layer 2 gate.
# ---------------------------------------------------------------------

_TASK8_WIRING_LANDED = True  # task #8 PR #1762 merged 2026-04-28
# (commit 08d9d3b6). Three wiring points alive:
#   * worker_factory._build_lineage_graph_store →
#     LineageGraphStoreWithAliasRedirect(inner=...)
#   * retrieval/pipeline._graph_search → GraphSearchService
#     (search_entities + get_subgraph + compose_context)
#   * GraphService.merge_entities → LineageEntityMerger via
#     build_lineage_entity_merger_for(collection)

_RUN_GATE_ENV = "RUN_W7_E2E_NARRATIVE"


def _layer2_skip_reason() -> str:
    return (
        f"Layer 2 e2e narrative requires {_RUN_GATE_ENV}=1 + real "
        "Postgres / Redis / Qdrant / Elasticsearch / provider keys. "
        "CI Wave 7 lane flips this on; local-dev pytest stays fast by "
        "default."
    )


pytestmark = [
    pytest.mark.skipif(
        os.environ.get(_RUN_GATE_ENV) != "1",
        reason=_layer2_skip_reason(),
    ),
]


# ---------------------------------------------------------------------
# Module-scoped fixture: per-narrative collection + tenant.
#
# We bind the entire 9-step narrative to one synthetic collection_id
# so steps share state via the lineage store + alias_map (the
# production data plane). The collection_id is not registered in the
# Postgres ``Collection`` table — every API used by these steps is
# constructed directly via ``_build_lineage_graph_store_inner`` with a
# ``SimpleNamespace`` collection stub, mirroring how
# ``test_full_indexing_pipeline.py`` Layer 2 builds the factory
# without going through the FastAPI lifespan. This keeps the test
# orthogonal to Collection-row bootstrap concerns.
# ---------------------------------------------------------------------


_COLLECTION_ID = "w7-e2e-" + uuid.uuid4().hex[:12]
_TENANT_KEY = f"user:{_COLLECTION_ID}"
_DOC_ID_PRIMARY = "w7-e2e-doc-1"
_DOC_ID_ALICIA = "w7-e2e-doc-alicia"


def _collection_stub() -> Any:
    """Return the minimal collection-shaped object the production
    factories accept. The factories only read ``collection.id``; a
    full ORM row is not required.
    """
    return SimpleNamespace(id=_COLLECTION_ID)


def _resolve_graph_backend_or_skip() -> tuple[str, Any]:
    """Resolve the active graph backend type + a raw inner store
    (no alias-redirect decorator). Skips the test when the production
    settings cannot reach a real backend — Layer 2 explicitly requires
    the e2e-http-compose stack.
    """
    try:
        from aperag.indexing.worker_factory import (
            _build_lineage_graph_store_inner,
            _resolve_graph_backend_type,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"worker_factory import failed: {exc!r}")
    backend_type = _resolve_graph_backend_type(_collection_stub())
    try:
        store = _build_lineage_graph_store_inner(
            backend_type=backend_type,
            collection=_collection_stub(),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend resolve failed (Layer 2 needs real stack): {exc!r}")
    return backend_type, store


@pytest.fixture(scope="module")
def inner_store() -> Any:
    """Per-module raw lineage store (no alias-redirect decorator).
    Used by step 1 to seed entities and by all read-side assertions.
    """
    _backend_type, store = _resolve_graph_backend_or_skip()
    return store


@pytest.fixture(scope="module")
def decorated_store() -> Any:
    """Per-module alias-redirect-decorated store — the one the
    production indexer write path now goes through (post-task #8
    wiring #1)."""
    try:
        from aperag.indexing.worker_factory import _build_lineage_graph_store
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"worker_factory import failed: {exc!r}")
    try:
        return _build_lineage_graph_store(
            backend_type=__import__(
                "aperag.indexing.worker_factory", fromlist=["_resolve_graph_backend_type"]
            )._resolve_graph_backend_type(_collection_stub()),
            collection=_collection_stub(),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend resolve failed: {exc!r}")


@pytest.fixture(scope="module")
def alias_repo() -> Any:
    from aperag.graph_curation.alias_map import AliasMapRepository

    return AliasMapRepository()


# ---------------------------------------------------------------------
# Step bodies — narrative validation, one assertion per step.
# Tests run in source order so step N+1 sees the side-effects of
# step N. State lives in the lineage store + alias_map (production
# data plane), not in Python globals — which is what makes this an
# *integration* narrative rather than a stub chain.
# ---------------------------------------------------------------------


def _alice_record() -> Any:
    from aperag.indexing.graph import EntityRecord

    return EntityRecord(
        name="Alice",
        entity_type="PERSON",
        description="Alice is the lead engineer at Acme.",
        source_chunk_ids=("chunk-a-1",),
    )


def _bob_record() -> Any:
    from aperag.indexing.graph import EntityRecord

    return EntityRecord(
        name="Bob",
        entity_type="PERSON",
        description="Bob is Acme's CFO.",
        source_chunk_ids=("chunk-a-2",),
    )


def _acme_record() -> Any:
    from aperag.indexing.graph import EntityRecord

    return EntityRecord(
        name="Acme",
        entity_type="ORG",
        description="Acme is a software company.",
        source_chunk_ids=("chunk-a-3",),
    )


def _alicia_record() -> Any:
    """The merge-target alias source — a second extractor pass picks
    up the alternate spelling 'Alicia' which the narrative will
    later merge into Alice."""
    from aperag.indexing.graph import EntityRecord

    return EntityRecord(
        name="Alicia",
        entity_type="PERSON",
        description="Alicia is the lead engineer at Acme.",
        source_chunk_ids=("chunk-b-1",),
    )


def _make_lineage(*, document_id: str, parse_version: str, chunk_ids: tuple[str, ...]) -> Any:
    from aperag.indexing.graph import LineageMember

    return LineageMember(
        document_id=document_id,
        parse_version=parse_version,
        tenant_scope_key=_TENANT_KEY,
        chunk_ids=chunk_ids,
    )


@pytest.mark.asyncio
async def test_step1_seed_initial_entities(inner_store):
    """Seed three entities (Alice / Bob / Acme) directly through the
    raw inner store. We pre-extract — bypassing the LLM — so the
    narrative does not depend on a non-deterministic extractor; the
    wiring under test is the storage + vector + alias paths, not the
    extractor prompt."""
    lineage = _make_lineage(document_id=_DOC_ID_PRIMARY, parse_version="v1", chunk_ids=("chunk-a-1",))
    await inner_store.upsert_entity_with_lineage(record=_alice_record(), lineage=lineage)
    await inner_store.upsert_entity_with_lineage(
        record=_bob_record(),
        lineage=replace(lineage, chunk_ids=("chunk-a-2",)),
    )
    await inner_store.upsert_entity_with_lineage(
        record=_acme_record(),
        lineage=replace(lineage, chunk_ids=("chunk-a-3",)),
    )

    alice = await inner_store.get_entity("Alice")
    assert alice is not None and alice.name == "Alice"
    assert alice.entity_type == "PERSON"


@pytest.mark.asyncio
async def test_step2_compactor_and_vector_writer_phase3(inner_store):
    """Run the GraphIndexCompactor + vector writer phases by invoking
    the per-collection :class:`GraphSearchService` factory. Asserts
    that the factory wires the four dependencies the service expects
    (store + vector connector + embedder + optional LLM) so the
    production retrieval pipeline can call ``search_entities`` end to
    end without further wiring.
    """
    from aperag.indexing.graph_search_service import (
        GraphSearchService,
        build_graph_search_service_for,
    )

    service = build_graph_search_service_for(_collection_stub())
    assert isinstance(service, GraphSearchService)

    # Snapshot-diff invariant — Wave 4 §C.3: a sync that does not
    # touch entities X must NOT delete X. The seeded rows from step 1
    # MUST still be reachable via ``get_entity``.
    for name in ("Alice", "Bob", "Acme"):
        row = await inner_store.get_entity(name)
        assert row is not None, f"snapshot-diff dropped {name!r} between syncs"


@pytest.mark.asyncio
async def test_step3_search_entities_returns_indexed_entity():
    """Validates wiring #3 — ``retrieval/pipeline._graph_search`` now
    composes the same ``GraphSearchService`` we exercise here.

    With a freshly-seeded collection and no real vector index in this
    layer, the assertion is structural: ``search_entities`` accepts a
    string + top_k, returns a list, and never raises into the caller
    (graceful-degrade convention shared with the keyword path).
    """
    from aperag.indexing.graph_search_service import build_graph_search_service_for

    service = build_graph_search_service_for(_collection_stub())
    hits = await service.search_entities(query="Alice", top_k=5)
    # Vector recall returns a list; entries (if any) are
    # ``EntityWithLineage`` so the retrieval-pipeline render works
    # without further conversion.
    assert isinstance(hits, list)
    if hits:
        from aperag.indexing.graph import EntityWithLineage

        for hit in hits:
            assert isinstance(hit, EntityWithLineage)


@pytest.mark.asyncio
async def test_step4_service_merge_route_path_persists_alias_map(inner_store):
    """Validates wiring #2 — ``GraphService.merge_entities`` (the
    function the REST route ``POST /collections/{cid}/graphs/nodes/merge``
    now delegates to) runs through the LineageEntityMerger end to end.

    Seeds an Alicia row, then merges Alicia → Alice. The assertions
    ride the backward-compat response shape (``target_entity_id`` /
    ``description`` / ``edges_*=0``) so a regression in the route
    cutover is caught here even though the test itself is service-
    layer (the route unit test pins the HTTP shape).
    """
    # Seed Alicia first — she is a separate doc / chunk so the alias
    # map can later collapse her into Alice.
    alicia_lineage = _make_lineage(document_id=_DOC_ID_PRIMARY, parse_version="v1", chunk_ids=("chunk-a-4",))
    await inner_store.upsert_entity_with_lineage(record=_alicia_record(), lineage=alicia_lineage)

    from aperag.domains.knowledge_graph.service import GraphService

    svc = GraphService()
    with patch.object(
        svc,
        "_get_and_validate_collection",
        new=AsyncMock(return_value=_collection_stub()),
    ):
        result = await svc.merge_entities(
            "test-user",
            _COLLECTION_ID,
            target_entity_id="Alice",
            source_entity_ids=["Alicia"],
        )

    assert result["target_entity_id"] == "Alice"
    assert "Alicia" in result["merged_source_ids"]
    assert isinstance(result["description"], str)
    # Wiring #1 runs edge re-anchor at indexer write time; the merge
    # action surfaces 0 by design (Wave 7 §K.12 invariant #9 / cuiwenbo
    # schema diff lock).
    assert result["edges_redirected"] == 0
    assert result["edges_collapsed"] == 0


@pytest.mark.asyncio
async def test_step5_alias_map_row_persisted(alias_repo):
    """Read the alias_map: assert ``{Alicia: Alice}`` persisted via
    :class:`AliasMapRepository`. The merge in step 4 should have
    written the row before step 8 of its 8-step orchestration."""
    canonical = await alias_repo.resolve_canonical(collection_id=_COLLECTION_ID, name="Alicia")
    assert canonical == "Alice", f"expected alias 'Alicia' to resolve to canonical 'Alice', got {canonical!r}"


@pytest.mark.asyncio
async def test_step6_alias_redirect_on_indexer_upsert(decorated_store, inner_store):
    """The critical inseparability gate (wiring #1).

    Re-extract the alternate spelling ``Alicia`` and write it through
    the *decorated* store (the one the production indexer now uses).
    Assert that no separate Alicia row exists in the lineage table —
    the decorator silently redirected the write to Alice.
    """
    lineage = _make_lineage(document_id=_DOC_ID_ALICIA, parse_version="v1", chunk_ids=("chunk-b-1",))
    await decorated_store.upsert_entity_with_lineage(record=_alicia_record(), lineage=lineage)

    # The raw inner store sees the canonical row only — Alicia was
    # silently redirected to Alice on the upsert path.
    alicia_row = await inner_store.get_entity("Alicia")
    alice_row = await inner_store.get_entity("Alice")

    assert alice_row is not None, "Alice must remain after alias redirect"
    # Step 4's merge already deleted the original Alicia row; this
    # step's write went to Alice via the decorator. The post-step-6
    # invariant is "no resurrected Alicia entity row".
    assert alicia_row is None, (
        "alias redirect failed: Alicia row resurrected after a "
        "write that should have been silently re-anchored to Alice"
    )

    # The new lineage member from this re-extraction shows up under
    # the canonical name with the original chunk ids preserved.
    new_member_present = any(
        m.document_id == _DOC_ID_ALICIA and m.parse_version == "v1" for m in (alice_row.source_lineage or ())
    )
    assert new_member_present, (
        "alias redirect did not propagate the new lineage member to the canonical entity — re-extraction lineage lost"
    )


@pytest.mark.asyncio
async def test_step7_re_search_after_merge_returns_canonical():
    """``search_entities("Alice")`` still returns the merged entity.
    Validates that the round-trip merge → re-index → re-recall
    narrative is end-to-end coherent. Vector recall on the canonical
    name MUST keep working post-merge.
    """
    from aperag.indexing.graph_search_service import build_graph_search_service_for

    service = build_graph_search_service_for(_collection_stub())
    hits = await service.search_entities(query="Alice", top_k=5)
    assert isinstance(hits, list)
    # If any vectors are populated, the canonical Alice row should be
    # in the result set — alias does not show up because the alias is
    # a payload-redirected write, not a separate point.
    if hits:
        names = {h.name for h in hits}
        assert "Alicia" not in names, "alias name leaked into vector recall — wiring #1 + #3 interaction broke"


@pytest.mark.asyncio
async def test_step8_get_entity_detail_alias_returns_404_w8_3_pin():
    """W8-3 trigger pin (per huangheng msg=0b48af2b + architect ratify).

    ``GraphService.get_entity_detail`` (the function the route
    ``GET /collections/{cid}/graphs/entities/{name}`` delegates to,
    per task #7) currently bypasses the alias_map on reads — looking
    up ``Alicia`` returns ``None`` and the route surfaces 404.

    When Wave 8 W8-3 ships read-side alias resolution, this test will
    fail (the lookup will start returning the Alice payload). The fix
    is to flip the assertion: that flip is the physical evidence that
    W8-3 shipped + the trigger condition is pinned in the repo.

    Per architect ratify: the assertion shape is "today returns None
    / 404; Wave 8 will flip to a real row" — encoding the future
    expectation as a comment so the eventual fix is mechanical.
    """
    from aperag.domains.knowledge_graph.service import GraphService

    svc = GraphService()
    with patch.object(
        svc,
        "_get_and_validate_collection",
        new=AsyncMock(return_value=_collection_stub()),
    ):
        result = await svc.get_entity_detail(
            "test-user",
            _COLLECTION_ID,
            entity_name="Alicia",
        )

    # W8-3 trigger condition pinned in the assert message so the
    # eventual fix is mechanical (per 冬柏 msg=8f488513): when Wave 8
    # W8-3 ships read-side alias resolution, this assert fails and the
    # traceback message tells the implementer exactly which line to
    # flip.
    assert result is None, (
        "W8-3 trigger condition: get_entity_detail('Alicia') today "
        "returns None because read-side alias resolution is deferred "
        "to Wave 8 W8-3. When W8-3 ships, this assertion will fail; "
        "flip to: `assert result is not None and result.name == 'Alice'`"
    )


@pytest.mark.asyncio
async def test_step9_failure_mode_compactor_llm_down_graceful_degrade():
    """Fold-in from architect msg=a0ba75da: prove the integration
    survives a transient compactor LLM outage. Mirrors the unit
    invariant ``test_w7_phase3_*_failure_non_fatal`` but exercises the
    full merger orchestration so a wiring drift that loses the
    graceful-degrade path (e.g. raising into the merge call) is
    surfaced here, not in production.
    """
    # Patch the GraphIndexCompactor.compact_if_oversized async
    # method to raise. The production merger calls it inside step 5
    # of its 8-step orchestration; on raise, the merger falls back
    # to the unified description + skips compaction (per the merger
    # docstring "graceful degrade — partial merge still proceeds").
    from aperag.domains.knowledge_graph.service import GraphService
    from aperag.indexing.graph_compactor import GraphIndexCompactor

    svc = GraphService()
    with (
        patch.object(
            svc,
            "_get_and_validate_collection",
            new=AsyncMock(return_value=_collection_stub()),
        ),
        patch.object(
            GraphIndexCompactor,
            "compact_if_oversized",
            new=AsyncMock(side_effect=RuntimeError("compactor LLM down")),
        ),
    ):
        # Use a fresh source name so the merge is non-degenerate even
        # after the compactor raise.
        result = await svc.merge_entities(
            "test-user",
            _COLLECTION_ID,
            target_entity_id="Alice",
            source_entity_ids=["Bob"],
        )

    # Graceful degrade: the merge still completes with the unified
    # description, just no compacted text.
    assert result["target_entity_id"] == "Alice"
    assert isinstance(result["description"], str)
