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

  step 1:  upload a document containing entities Alice, Bob, Acme +
           relations Alice—knows—Bob and Bob—works_at—Acme.
  step 2:  trigger sync (Phase 3 — entity vector compute, compaction,
           snapshot diff). Verify entity vectors are written, the
           compacted_description column populates, and snapshot diff
           does not silently delete shared entities.
  step 3:  call ``GraphSearchService.search_entities("Alice")``.
           Assert hit returns Alice. Validates wiring #3
           (retrieval/pipeline.py cutover to vector recall).
  step 4:  POST ``/collections/{cid}/graphs/nodes/merge`` with target
           "Alice" + sources ["Alicia"]. Validates wiring #2 (REST
           route swap end-to-end).
  step 5:  read alias_map: assert ``{Alicia: Alice}`` persisted.
           Vector point upserted to target. Source DescriptionParts
           re-anchored.
  step 6:  upload a new document with kg.jsonl containing "Alicia".
           Trigger sync. Assert the indexer-side ``upsert`` path
           silently redirected the write to "Alice" — Alicia must NOT
           appear as a separate entity. Validates wiring #1
           (LineageGraphStoreWithAliasRedirect decorator alive).
  step 7:  search again for "Alice" → still returns the merged entity.
           Search for "Alicia" → also returns Alice (alias hit).
  step 8 (W8-3 trigger pin): ``GET /collections/{cid}/graphs/nodes/
           {alicia_id}`` → expect 404 / NotFound. Wave 8 W8-3 will add
           read-side alias resolution; when it ships, this assertion
           flips from "404 expected" to "200 with Alice payload" and
           the test is the trigger condition pinned in the repo.
  step 9 (failure-mode fold-in): simulate the compactor LLM call
           raising. Re-trigger sync. Assert the document still
           reaches ``status=ACTIVE`` with compaction skipped (Wave 6
           graceful degrade preserved per
           ``test_w7_phase3_*_failure_non_fatal`` unit invariant);
           the entity row exists with an empty / fallback
           ``compacted_description`` instead of failing the document.

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

import pytest

# ---------------------------------------------------------------------
# Layer 1 — scaffold gate. Until task #8 lands the wiring, the body
# of every step has nothing real to call against; we pin the narrative
# shape now so review can happen in parallel with task #8 impl.
# ---------------------------------------------------------------------

_TASK8_WIRING_LANDED = False  # flip to True in the same PR that lands
# task #8 wiring (worker_factory decorator wrap +
# REST merge route swap + retrieval pipeline cutover).
# Until then every step skips with a pointer to task #8.

_RUN_GATE_ENV = "RUN_W7_E2E_NARRATIVE"


def _wiring_skip_reason(step: str) -> str:
    return (
        f"task #11 step '{step}' requires task #8 wiring "
        "(LineageGraphStoreWithAliasRedirect / REST merge route swap / "
        "retrieval pipeline cutover). Scaffold pinned per architect "
        "msg=a0ba75da; bodies fill in alongside the task #8 PR."
    )


def _layer2_skip_reason() -> str:
    return (
        f"Layer 2 e2e narrative requires {_RUN_GATE_ENV}=1 + real "
        "Postgres / Redis / Qdrant / Elasticsearch / provider keys. "
        "CI Wave 7 lane flips this on once task #8 wiring is alive; "
        "local-dev pytest stays fast by default."
    )


pytestmark = [
    pytest.mark.skipif(
        not _TASK8_WIRING_LANDED,
        reason=_wiring_skip_reason("module"),
    ),
    pytest.mark.skipif(
        os.environ.get(_RUN_GATE_ENV) != "1",
        reason=_layer2_skip_reason(),
    ),
]


# ---------------------------------------------------------------------
# Step bodies. Each test is one narrative step. Once task #8 lands
# the wiring + ``_TASK8_WIRING_LANDED`` flips True, these bodies are
# implemented per the docstrings; until then they are recognised
# contract pins so the file is reviewable as a scaffold.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step1_upload_document_with_kg_jsonl():
    """Upload a document whose ``kg.jsonl`` contains the Alice / Bob /
    Acme entity set + Alice—knows—Bob and Bob—works_at—Acme relations.
    The document upload returns a document_id used by step 2."""
    pytest.skip(_wiring_skip_reason("step 1 upload"))


@pytest.mark.asyncio
async def test_step2_sync_phase3_writes_entity_vectors_and_compaction():
    """Trigger sync (Phase 3). Assert:

    * ``aperag_lineage_entity.entity_vector`` populated for all 3
      entities (vector recall pre-condition).
    * ``aperag_lineage_entity.compacted_description`` populated
      (W7-1 chunk delivers this column).
    * Snapshot-diff does not delete shared entities across docs
      (T8 chunk 4 cross-event-loop invariant).
    """
    pytest.skip(_wiring_skip_reason("step 2 sync Phase 3"))


@pytest.mark.asyncio
async def test_step3_search_entities_returns_indexed_entity():
    """``GraphSearchService.search_entities("Alice")`` MUST return
    Alice. Validates wiring #3 — retrieval/pipeline.py:_graph_search
    cuts over to the new vector recall."""
    pytest.skip(_wiring_skip_reason("step 3 search via retrieval cutover"))


@pytest.mark.asyncio
async def test_step4_rest_merge_endpoint_persists_alias_map():
    """POST ``/collections/{cid}/graphs/nodes/merge`` with
    target="Alice", sources=["Alicia"]. Assert HTTP 200 + the
    response payload reports alias_map updated. Validates wiring #2
    — REST route swap from legacy ``GraphIndexService.merge_entities``
    to ``GraphCurationService.merge_entities``."""
    pytest.skip(_wiring_skip_reason("step 4 REST merge route swap"))


@pytest.mark.asyncio
async def test_step5_alias_map_row_persisted_and_vector_re_embedded():
    """Read ``aperag_lineage_entity_alias`` (Wave 7 W7-6 schema) and
    assert ``{Alicia: Alice}`` row exists. Vector point for the
    canonical Alice is the upsert target. Source DescriptionParts
    re-anchor preserved."""
    pytest.skip(_wiring_skip_reason("step 5 alias_map persistence"))


@pytest.mark.asyncio
async def test_step6_re_add_doc_with_alias_silently_redirects():
    """Re-upload a document with ``kg.jsonl`` containing ``Alicia``.
    Trigger sync. Assert that on the indexer-side ``upsert`` path
    the write was silently redirected to ``Alice``: ``Alicia`` MUST
    NOT appear as a separate entity row in
    ``aperag_lineage_entity``. This is the critical inseparability
    gate: it is the only test that proves the
    ``LineageGraphStoreWithAliasRedirect`` decorator (wiring #1) is
    not just present but produces the expected behaviour."""
    pytest.skip(_wiring_skip_reason("step 6 alias redirect on indexer upsert"))


@pytest.mark.asyncio
async def test_step7_re_search_still_returns_merged_entity():
    """``search_entities("Alice")`` still hits Alice.
    ``search_entities("Alicia")`` ALSO hits Alice (recall traverses
    the alias_map). Validates the round-trip merge → re-index →
    re-recall narrative end-to-end."""
    pytest.skip(_wiring_skip_reason("step 7 re-search after merge"))


@pytest.mark.asyncio
async def test_step8_get_entity_detail_alias_returns_404_w8_3_pin():
    """W8-3 trigger pin (per huangheng msg=0b48af2b + architect ratify):

    ``GET /collections/{cid}/graphs/nodes/{alicia_id}`` MUST return
    404 / NotFound today, because read-side alias resolution is
    deferred to Wave 8 W8-3.

    When Wave 8 W8-3 ships read-side alias resolution, this test will
    fail (because the endpoint will start returning 200 with the Alice
    payload). The fix is to flip the assertion: that flip is the
    physical evidence that W8-3 shipped + the trigger condition is
    pinned in the repo."""
    pytest.skip(_wiring_skip_reason("step 8 W8-3 trigger pin"))


@pytest.mark.asyncio
async def test_step9_failure_mode_compactor_llm_down_graceful_degrade():
    """failure-mode fold-in (per architect msg=a0ba75da):

    Patch the compactor LLM call to raise. Re-trigger sync on a fresh
    document. Assert:

    * ``DocumentIndex.status`` reaches ``ACTIVE`` (not ``FAILED``) —
      compaction failure is non-fatal per Wave 6 graceful degrade.
    * The entity row exists with an empty / fallback
      ``compacted_description`` (the body falls back to the longest
      DescriptionPart text per the Wave 6 contract).

    Complements ``tests/unit_test/.../test_w7_phase3_*_failure_non_fatal``
    by proving the same invariant survives the full e2e pipeline,
    not just the unit-level branch."""
    pytest.skip(_wiring_skip_reason("step 9 failure-mode graceful degrade"))
