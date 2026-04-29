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

"""T1.2 acceptance tests — graph modality (design pack §D.3 / §D.4).

Three coverage groups, mapping to the architect-locked acceptance
gates for the graph lane (msg=7f82bb71):

1. **§D.3.6 five-step idempotency**: doc_A v1 → doc_B v1 → doc_A v2
   overwrites old doc_A → doc_A delete → doc_B delete with full
   entity GC. The shared entity ``Linus`` must end the run as having
   exactly the surviving lineage members at every step. This is the
   regression Bryce surfaced in msg=7ccb176f #3 — testing it
   end-to-end pins the §D.3.2 algorithm against the design pack.

2. **§D.4 byte-equivalent re-sync**: a re-run of ``sync(doc, v,
   kg.jsonl)`` against the same artifact leaves the backend
   byte-equivalent to a single-call sync. This is the standard
   "DELETE-before-INSERT idempotency" gate every modality must pass.

3. **Nebula race condition** (msg=f2921ae0 architect-locked
   invariant): two concurrent ``sync`` calls touching the same
   shared entity, with cooperative scheduling that would interleave
   read-modify-write phases under a no-op lock, MUST produce the
   union of both lineage members under the real per-entity lock.
   Since the ``LineageGraphStore`` interface delegates lineage SET
   maintenance to the lock, exercising it on an in-memory backend
   exercises exactly the same race window that bites Nebula
   read-modify-write.

The tests use the in-memory reference store
(:class:`InMemoryLineageGraphStore`) plus an in-memory entity lock
plus an in-memory object store, so no external infra is required to
run them in CI. The §D.3.2 algorithm is backend-agnostic by design;
when Wave 2 wires real Nebula / Neo4j adapters they MUST satisfy the
same Protocol contract and therefore inherit pass/fail status from
this suite.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import pytest

from aperag.indexing.graph import (
    KG_ARTIFACT_FILENAME,
    DescriptionPart,
    EntityLock,
    EntityRecord,
    EntityWithLineage,
    GraphModalityWorker,
    InMemoryEntityLock,
    InMemoryLineageGraphStore,
    LineageMember,
    RelationRecord,
    parse_kg_jsonl,
    serialize_kg_jsonl,
)
from aperag.indexing.object_store import (
    InMemoryObjectStore,
    derived_artifact,
    write_atomic,
)
from aperag.indexing.parser import ParseConfig, parse_document

# ---------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------


COLLECTION_ID = "col-graph-test"
DEFAULT_TENANT = "user:test"


@pytest.fixture
def store() -> InMemoryLineageGraphStore:
    return InMemoryLineageGraphStore()


@pytest.fixture
def entity_lock() -> EntityLock:
    return InMemoryEntityLock()


@pytest.fixture
def object_store() -> InMemoryObjectStore:
    return InMemoryObjectStore()


def _doc_chunks(doc_id: str, parse_version: str) -> list[dict[str, Any]]:
    """Two synthetic chunks for ``doc_id`` at ``parse_version``.

    The chunk ids embed the doc + parse_version so the §D.3.6 trace
    of "which doc / parse_v contributed which chunk to the entity
    lineage" stays human-readable in test failures.
    """
    return [
        {
            "chunk_id": f"{doc_id}-{parse_version}-c0",
            "text": f"chunk 0 of {doc_id} parse_version={parse_version}",
        },
        {
            "chunk_id": f"{doc_id}-{parse_version}-c1",
            "text": f"chunk 1 of {doc_id} parse_version={parse_version}",
        },
    ]


def _write_doc_chunks_jsonl(
    *,
    object_store: InMemoryObjectStore,
    document_id: str,
    parse_version: str,
) -> None:
    """Pre-seed ``derived/parse_<v>/chunks.jsonl`` for a doc/version
    so ``GraphModalityWorker.derive`` has something to read.

    Bypasses the parser since ``parse_document`` requires a real
    markdown body and we only care about chunk-id flow into kg.jsonl
    for these tests."""
    import json

    body = ("\n".join(json.dumps(chunk) for chunk in _doc_chunks(document_id, parse_version)) + "\n").encode("utf-8")
    write_atomic(
        object_store,
        derived_artifact(
            collection_id=COLLECTION_ID,
            document_id=document_id,
            parse_version=parse_version,
            filename="chunks.jsonl",
        ),
        body,
    )


async def _stub_extractor_factory(
    entities_per_doc: dict[str, list[EntityRecord]],
    relations_per_doc: dict[str, list[RelationRecord]],
    document_id: str,
):
    """Build a :data:`GraphExtractor` stub that returns the canned
    ``(entities, relations)`` for ``document_id``.

    Tests pre-stage what the LLM extractor "would have" produced for
    each doc / parse_v under test; the actual LLM call is stubbed
    out so the algorithm under test is exercised without LLM
    flakiness.
    """

    async def extractor(
        chunks: Sequence[dict[str, Any]],
    ) -> tuple[list[EntityRecord], list[RelationRecord]]:
        del chunks  # the stub does not depend on chunks
        return (
            list(entities_per_doc.get(document_id, [])),
            list(relations_per_doc.get(document_id, [])),
        )

    return extractor


def _make_worker(
    *,
    store: InMemoryLineageGraphStore,
    entity_lock: EntityLock,
    object_store: InMemoryObjectStore,
    document_id: str,
    entities_per_doc: dict[str, list[EntityRecord]] | None = None,
    relations_per_doc: dict[str, list[RelationRecord]] | None = None,
    tenant_scope_key: str = DEFAULT_TENANT,
    entity_type_merger=None,
) -> GraphModalityWorker:
    async def extractor(
        chunks: Sequence[dict[str, Any]],
    ) -> tuple[list[EntityRecord], list[RelationRecord]]:
        del chunks
        return (
            list((entities_per_doc or {}).get(document_id, [])),
            list((relations_per_doc or {}).get(document_id, [])),
        )

    return GraphModalityWorker(
        store=store,
        extractor=extractor,
        entity_lock=entity_lock,
        object_store=object_store,
        collection_id=COLLECTION_ID,
        tenant_scope_key=tenant_scope_key,
        entity_type_merger=entity_type_merger,
    )


async def _derive_then_sync(
    *,
    worker: GraphModalityWorker,
    document_id: str,
    parse_version: str,
    object_store: InMemoryObjectStore,
) -> None:
    """Run derive + sync end-to-end for a single (doc, parse_v).

    Pre-condition: ``chunks.jsonl`` for ``(document_id, parse_version)``
    is already on the object store. Post-condition: the backend
    reflects the §D.3.2 lineage outcome.
    """
    _write_doc_chunks_jsonl(object_store=object_store, document_id=document_id, parse_version=parse_version)
    derive_result = await worker.derive(
        document_id=document_id,
        parse_version=parse_version,
        source_path="<irrelevant>",
    )
    await worker.sync(
        document_id=document_id,
        parse_version=parse_version,
        derived_artifact_path=derive_result.derived_artifact_path,
    )


def _lineage_keys(entity: EntityWithLineage) -> set[tuple[str, str]]:
    return {member.key() for member in entity.source_lineage}


def _description_keys(entity: EntityWithLineage) -> set[tuple[str, str]]:
    return {part.key() for part in entity.description_parts}


@pytest.mark.asyncio
async def test_derive_merges_entity_types_once_per_document(store, entity_lock, object_store):
    calls: list[tuple[str, list[str]]] = []

    async def _merge(document_id: str, entity_types: Sequence[str]) -> list[str]:
        calls.append((document_id, list(entity_types)))
        return list(entity_types)

    worker = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={
            "doc_A": [
                EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
                EntityRecord(name="Bob", entity_type="person", description="b", source_chunk_ids=("c1",)),
                EntityRecord(name="Acme", entity_type="组织", description="o", source_chunk_ids=("c1",)),
            ]
        },
        entity_type_merger=_merge,
    )

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")
    await worker.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")

    assert calls == [("doc_A", ["Person", "组织"])]


@pytest.mark.asyncio
async def test_derive_entity_type_merge_failure_is_non_fatal(store, entity_lock, object_store, caplog):
    async def _merge(_document_id: str, _entity_types: Sequence[str]) -> list[str]:
        raise RuntimeError("merge unavailable")

    worker = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={
            "doc_A": [EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",))]
        },
        entity_type_merger=_merge,
    )

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")
    with caplog.at_level(logging.WARNING):
        result = await worker.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")

    assert object_store.obj_exists(result.derived_artifact_path)
    assert "graph entity type merge failed" in caplog.text


# ---------------------------------------------------------------------
# Group 1: §D.3.6 five-step idempotency self-test
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d3_6_step1_doc_a_v1_inserts_initial_lineage(store, entity_lock, object_store):
    """Step 1: doc_A v1 → entity ``Linus`` lineage = {(doc_A, v1)}."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="kernel hacker per doc_A",
                source_chunk_ids=("doc_A-v1-c0",),
            )
        ]
    }
    worker = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )

    entity = await store.get_entity("Linus")
    assert entity is not None
    assert _lineage_keys(entity) == {("doc_A", "v1")}
    assert _description_keys(entity) == {("doc_A", "v1")}


@pytest.mark.asyncio
async def test_d3_6_step2_doc_b_v1_adds_to_shared_entity_lineage(store, entity_lock, object_store):
    """Step 2: doc_B v1 → entity ``Linus`` lineage = {(doc_A,v1), (doc_B,v1)}."""
    entities_per_doc = {
        "doc_A": [EntityRecord("Linus", "Person", "doc_A says", ("doc_A-v1-c0",))],
        "doc_B": [EntityRecord("Linus", "Person", "doc_B says", ("doc_B-v1-c0",))],
    }
    for doc_id in ("doc_A", "doc_B"):
        worker = _make_worker(
            store=store,
            entity_lock=entity_lock,
            object_store=object_store,
            document_id=doc_id,
            entities_per_doc=entities_per_doc,
        )
        await _derive_then_sync(
            worker=worker,
            document_id=doc_id,
            parse_version="v1",
            object_store=object_store,
        )

    entity = await store.get_entity("Linus")
    assert entity is not None
    assert _lineage_keys(entity) == {("doc_A", "v1"), ("doc_B", "v1")}
    # Both descriptions preserved per §D.3.3 Option A.
    assert _description_keys(entity) == {("doc_A", "v1"), ("doc_B", "v1")}
    descriptions_text = {p.text for p in entity.description_parts}
    assert descriptions_text == {"doc_A says", "doc_B says"}


@pytest.mark.asyncio
async def test_d3_6_step3_doc_a_v2_overwrites_old_doc_a_lineage(store, entity_lock, object_store):
    """Step 3: doc_A v2 supersedes old doc_A v1 lineage member.

    Final state: lineage = {(doc_A, v2), (doc_B, v1)} — doc_B v1
    untouched, doc_A v1 GC'd, doc_A v2 fresh.
    """
    entities_per_doc = {
        "doc_A": [EntityRecord("Linus", "Person", "doc_A v1 says", ("doc_A-v1-c0",))],
        "doc_B": [EntityRecord("Linus", "Person", "doc_B v1 says", ("doc_B-v1-c0",))],
    }
    # Step 1 + Step 2 (build initial state).
    for doc_id, parse_v in (("doc_A", "v1"), ("doc_B", "v1")):
        worker = _make_worker(
            store=store,
            entity_lock=entity_lock,
            object_store=object_store,
            document_id=doc_id,
            entities_per_doc=entities_per_doc,
        )
        await _derive_then_sync(
            worker=worker,
            document_id=doc_id,
            parse_version=parse_v,
            object_store=object_store,
        )

    # Step 3: doc_A v2 with new description and new chunk ids.
    entities_per_doc["doc_A"] = [
        EntityRecord("Linus", "Person", "doc_A v2 says (revised)", ("doc_A-v2-c0", "doc_A-v2-c1"))
    ]
    worker_v2 = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
    )
    await _derive_then_sync(
        worker=worker_v2,
        document_id="doc_A",
        parse_version="v2",
        object_store=object_store,
    )

    entity = await store.get_entity("Linus")
    assert entity is not None
    # doc_A v1 GC'd, doc_A v2 in, doc_B v1 untouched.
    assert _lineage_keys(entity) == {("doc_A", "v2"), ("doc_B", "v1")}
    # doc_A v2 chunk ids reflect the v2 parse, not the v1 ones.
    doc_a_v2_member = next(m for m in entity.source_lineage if m.document_id == "doc_A")
    assert doc_a_v2_member.chunk_ids == ("doc_A-v2-c0", "doc_A-v2-c1")
    # description_parts shows the revised doc_A description plus the
    # untouched doc_B description.
    description_text_by_doc = {p.document_id: p.text for p in entity.description_parts}
    assert description_text_by_doc == {
        "doc_A": "doc_A v2 says (revised)",
        "doc_B": "doc_B v1 says",
    }


@pytest.mark.asyncio
async def test_d3_6_step4_delete_doc_a_keeps_doc_b_lineage(store, entity_lock, object_store):
    """Step 4: deleting doc_A leaves entity ``Linus`` with lineage = {(doc_B, v1)}.

    "Delete doc_A" is modelled as syncing an EMPTY kg.jsonl for the
    final ``(doc_A, v_final)`` parse_version, which is precisely the
    flow Wave 2's reconciler emits when the document moves to the
    DELETION state per §F.5.
    """
    entities_per_doc = {
        "doc_A": [EntityRecord("Linus", "Person", "doc_A says", ("doc_A-v1-c0",))],
        "doc_B": [EntityRecord("Linus", "Person", "doc_B says", ("doc_B-v1-c0",))],
    }
    for doc_id, parse_v in (("doc_A", "v1"), ("doc_B", "v1")):
        worker = _make_worker(
            store=store,
            entity_lock=entity_lock,
            object_store=object_store,
            document_id=doc_id,
            entities_per_doc=entities_per_doc,
        )
        await _derive_then_sync(
            worker=worker,
            document_id=doc_id,
            parse_version=parse_v,
            object_store=object_store,
        )

    # Step 4: doc_A deletion = sync empty kg.jsonl for doc_A v1.
    entities_per_doc["doc_A"] = []
    worker_delete = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
    )
    await _derive_then_sync(
        worker=worker_delete,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )

    entity = await store.get_entity("Linus")
    assert entity is not None
    assert _lineage_keys(entity) == {("doc_B", "v1")}
    descriptions = {p.document_id: p.text for p in entity.description_parts}
    assert descriptions == {"doc_B": "doc_B says"}


@pytest.mark.asyncio
async def test_d3_6_step5_delete_doc_b_garbage_collects_entity(store, entity_lock, object_store):
    """Step 5: deleting doc_B leaves no contributors → entity row GC'd."""
    entities_per_doc = {
        "doc_A": [EntityRecord("Linus", "Person", "doc_A says", ("doc_A-v1-c0",))],
        "doc_B": [EntityRecord("Linus", "Person", "doc_B says", ("doc_B-v1-c0",))],
    }
    # Build then delete both docs.
    for doc_id, parse_v in (("doc_A", "v1"), ("doc_B", "v1")):
        worker = _make_worker(
            store=store,
            entity_lock=entity_lock,
            object_store=object_store,
            document_id=doc_id,
            entities_per_doc=entities_per_doc,
        )
        await _derive_then_sync(
            worker=worker,
            document_id=doc_id,
            parse_version=parse_v,
            object_store=object_store,
        )

    for doc_id in ("doc_A", "doc_B"):
        entities_per_doc[doc_id] = []
        worker_delete = _make_worker(
            store=store,
            entity_lock=entity_lock,
            object_store=object_store,
            document_id=doc_id,
            entities_per_doc=entities_per_doc,
        )
        await _derive_then_sync(
            worker=worker_delete,
            document_id=doc_id,
            parse_version="v1",
            object_store=object_store,
        )

    entity = await store.get_entity("Linus")
    assert entity is None  # entity row GC'd because lineage is empty


# ---------------------------------------------------------------------
# Group 2: relation lineage (§D.3.1 + §D.3.2 — relation symmetry)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relation_lineage_doc_a_then_doc_b_then_delete_doc_a(store, entity_lock, object_store):
    """Relation between shared entities has the same lineage semantics
    as the entity itself: doc_A v1 + doc_B v1 → both members; delete
    doc_A → only doc_B member; delete doc_B → relation GC'd."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord("Linus", "Person", "...", ("doc_A-v1-c0",)),
            EntityRecord("Linux", "Project", "...", ("doc_A-v1-c0",)),
        ],
        "doc_B": [
            EntityRecord("Linus", "Person", "...", ("doc_B-v1-c0",)),
            EntityRecord("Linux", "Project", "...", ("doc_B-v1-c0",)),
        ],
    }
    relations_per_doc = {
        "doc_A": [RelationRecord("Linus", "Linux", "created", "doc_A says", ("doc_A-v1-c0",))],
        "doc_B": [RelationRecord("Linus", "Linux", "created", "doc_B says", ("doc_B-v1-c0",))],
    }

    for doc_id, parse_v in (("doc_A", "v1"), ("doc_B", "v1")):
        worker = _make_worker(
            store=store,
            entity_lock=entity_lock,
            object_store=object_store,
            document_id=doc_id,
            entities_per_doc=entities_per_doc,
            relations_per_doc=relations_per_doc,
        )
        await _derive_then_sync(
            worker=worker,
            document_id=doc_id,
            parse_version=parse_v,
            object_store=object_store,
        )

    relation = await store.get_relation("Linus", "Linux", "created")
    assert relation is not None
    assert {(m.document_id, m.parse_version) for m in relation.evidence_lineage} == {
        ("doc_A", "v1"),
        ("doc_B", "v1"),
    }

    # Delete doc_A: relation stays with only doc_B's evidence.
    entities_per_doc["doc_A"] = []
    relations_per_doc["doc_A"] = []
    worker_delete_a = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        relations_per_doc=relations_per_doc,
    )
    await _derive_then_sync(
        worker=worker_delete_a,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    relation = await store.get_relation("Linus", "Linux", "created")
    assert relation is not None
    assert {(m.document_id, m.parse_version) for m in relation.evidence_lineage} == {("doc_B", "v1")}

    # Delete doc_B: relation row GC'd.
    entities_per_doc["doc_B"] = []
    relations_per_doc["doc_B"] = []
    worker_delete_b = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_B",
        entities_per_doc=entities_per_doc,
        relations_per_doc=relations_per_doc,
    )
    await _derive_then_sync(
        worker=worker_delete_b,
        document_id="doc_B",
        parse_version="v1",
        object_store=object_store,
    )
    relation = await store.get_relation("Linus", "Linux", "created")
    assert relation is None


# ---------------------------------------------------------------------
# Group 3: §D.4 byte-equivalent re-sync idempotency
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d4_byte_equivalent_resync(store, entity_lock, object_store):
    """Re-syncing the same kg.jsonl twice yields the same backend state.

    This is the cross-modality idempotency contract; the §D.3.2
    lineage algorithm must satisfy it the same as the simpler
    DELETE-by-(doc, parse_v) modalities.
    """
    entities_per_doc = {"doc_A": [EntityRecord("Linus", "Person", "...", ("doc_A-v1-c0",))]}
    worker = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
    )

    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    entity_after_first = await store.get_entity("Linus")
    assert entity_after_first is not None

    # Replay the same sync; the backend state must remain byte-equivalent.
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    entity_after_second = await store.get_entity("Linus")
    assert entity_after_second is not None
    assert _lineage_keys(entity_after_second) == _lineage_keys(entity_after_first)
    assert _description_keys(entity_after_second) == _description_keys(entity_after_first)
    # And specifically: lineage member chunk_ids unchanged.
    assert sorted(m.chunk_ids for m in entity_after_second.source_lineage) == sorted(
        m.chunk_ids for m in entity_after_first.source_lineage
    )


# ---------------------------------------------------------------------
# Group 4: Nebula-style race condition under per-entity lock
# ---------------------------------------------------------------------


class _RaceProvocateurStore(InMemoryLineageGraphStore):
    """In-memory store that simulates a Nebula-style read-modify-write
    race between the find / remove / upsert phases.

    The base store is fully serial (single asyncio guard). The
    provocateur subclass introduces a deterministic
    :class:`asyncio.Event`-based barrier inside
    :meth:`upsert_entity_with_lineage` so the race window opens at
    the EXACT same point regardless of asyncio scheduler quirks under
    CI load. ``race_count`` controls how many concurrent writers must
    reach the barrier before any may proceed:

    * ``race_count=1`` (default) — no barrier; the provocateur
      behaves like a normal store with a single ``asyncio.sleep(0)``
      yield. Used for the lock-protected test where the per-entity
      lock guarantees only one writer is in flight at a time, so a
      ``race_count=2`` barrier would deadlock.
    * ``race_count=2`` — both writers must complete their read phase
      (compute ``current_keys`` from the same stale snapshot) before
      EITHER writer is allowed to write back. This makes the
      "scheduler-dependent" race deterministic and pins the failure
      mode the no-lock negative-control asserts.

    Without this barrier the test :func:`test_nebula_race_without_lock_loses_a_writer`
    flakes under heavy CI load because ``asyncio.sleep(0)`` yields
    only once and the scheduler may resume the same writer before
    the other gets to its read phase (huangheng msg=2b20974b
    informational + architect msg=8420f12a follow-up).
    """

    def __init__(self, *, race_count: int = 1) -> None:
        super().__init__()
        self._race_count = race_count
        self._readers_at_barrier = 0
        self._barrier_event = asyncio.Event()

    async def _maybe_wait_at_barrier(self) -> None:
        """Block until ``race_count`` writers have completed their read
        phase. With ``race_count=1`` this is a no-op (still need a
        single yield to emulate Nebula round-trip).
        """
        if self._race_count <= 1:
            await asyncio.sleep(0)
            return
        async with self._guard:
            self._readers_at_barrier += 1
            if self._readers_at_barrier >= self._race_count:
                self._barrier_event.set()
        await self._barrier_event.wait()

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        # ``compacted_description`` 透传给父类的 in-memory store,
        # 保持与 :class:`LineageGraphStore` Protocol 一致 (新模态拆分
        # 后父类 sync 总会传这个 kwarg).
        del compacted_description  # provocateur 关注 race window, 不读这个字段
        async with self._guard:
            row = self._entities.get(record.name)
            # Yield AFTER reading current state but BEFORE writing
            # the new lineage member back. This emulates the
            # round-trip latency of Nebula's read-modify-write.
            current_keys = set() if row is None else set(row.source_lineage.keys())

        # Wait at the deterministic barrier so a concurrent ``upsert``
        # has guaranteed read its own ``current_keys`` snapshot
        # before either writer proceeds. Lock-protected tests bypass
        # the barrier with ``race_count=1``.
        await self._maybe_wait_at_barrier()

        async with self._guard:
            row = self._entities.get(record.name)
            if row is None:
                # Note: the outer ``type(row)(...)`` call uses Python's
                # builtin ``type()``; the ``entity_type=`` kwarg is the
                # new dataclass field name (Wave 6 #36).
                row = type(row)(name=record.name, entity_type=record.entity_type) if row is not None else None
            from aperag.indexing.graph import _InMemoryEntityRow  # noqa: PLC0415

            if row is None:
                row = _InMemoryEntityRow(name=record.name, entity_type=record.entity_type)
                self._entities[record.name] = row
            else:
                row.entity_type = record.entity_type
            # Without the ``EntityLock`` the second writer's
            # ``current_keys`` may have been computed before the
            # first writer's mutation, but since we still merge into
            # the SAME row dict, both writes ultimately stick. The
            # race that bites Nebula is at the network round-trip
            # level: ``current_keys`` would be sent back as a list
            # replacement, clobbering the other writer's member.
            #
            # To emulate that more faithfully we rebuild the lineage
            # SET from the snapshot we read, i.e., DROP any members
            # the other writer might have inserted concurrently.
            new_lineage_keys = current_keys | {lineage.key()}
            preserved = {
                key: member
                for key, member in row.source_lineage.items()
                if key in new_lineage_keys and key != lineage.key()
            }
            preserved[lineage.key()] = lineage
            row.source_lineage = preserved
            row.description_parts[lineage.key()] = DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )


@pytest.mark.asyncio
async def test_nebula_race_under_per_entity_lock_preserves_both_writes():
    """Two concurrent ``sync`` calls on the SAME entity under
    :class:`InMemoryEntityLock` end up with both lineage members.

    Without the lock, the :class:`_RaceProvocateurStore` would
    deterministically lose one writer's lineage member. With the
    lock, the second writer enters its critical section AFTER the
    first writer's commit, observes the up-to-date lineage SET,
    and merges its own member on top. The lock therefore turns the
    Nebula read-modify-write window from racy into safe.
    """
    store = _RaceProvocateurStore()
    object_store = InMemoryObjectStore()
    entity_lock = InMemoryEntityLock()

    entity_record_a = EntityRecord("Linus", "Person", "doc_A says", ("doc_A-v1-c0",))
    entity_record_b = EntityRecord("Linus", "Person", "doc_B says", ("doc_B-v1-c0",))

    worker_a = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={"doc_A": [entity_record_a]},
    )
    worker_b = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_B",
        entities_per_doc={"doc_B": [entity_record_b]},
    )

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")
    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_B", parse_version="v1")

    derive_a = await worker_a.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")
    derive_b = await worker_b.derive(document_id="doc_B", parse_version="v1", source_path="<irrelevant>")

    await asyncio.gather(
        worker_a.sync(
            document_id="doc_A",
            parse_version="v1",
            derived_artifact_path=derive_a.derived_artifact_path,
        ),
        worker_b.sync(
            document_id="doc_B",
            parse_version="v1",
            derived_artifact_path=derive_b.derived_artifact_path,
        ),
    )

    entity = await store.get_entity("Linus")
    assert entity is not None
    # Both lineage members preserved per architect msg=f2921ae0
    # invariant: per-entity lock serialises read-modify-write.
    assert {(m.document_id, m.parse_version) for m in entity.source_lineage} == {
        ("doc_A", "v1"),
        ("doc_B", "v1"),
    }


@pytest.mark.asyncio
async def test_nebula_race_without_lock_loses_a_writer():
    """Negative control: with a no-op lock, the race provocateur
    deterministically drops one writer's lineage member.

    This proves the lock is load-bearing. If a future regression makes
    :class:`InMemoryEntityLock` no-op (e.g., someone deletes the
    asyncio.Lock around the registry), the previous test would still
    pass on a "happy path" interleaving but fail intermittently. The
    no-op lock test below pins the failure mode so we have a clear
    sentinel that the lock is what makes the previous test pass.
    """

    class _NoOpLock:
        def acquire(self, entity_id: str):
            from contextlib import nullcontext

            del entity_id
            return nullcontext()

    # ``race_count=2`` opens a deterministic Event barrier so both
    # writers MUST complete their read phase before either writes
    # back, regardless of asyncio scheduler quirks under CI load.
    # This pins the race deterministically — without the barrier the
    # ``asyncio.sleep(0)`` yield was scheduler-dependent and the test
    # flaked under heavy concurrent CI runs (huangheng msg=2b20974b
    # + architect msg=8420f12a follow-up directive).
    store = _RaceProvocateurStore(race_count=2)
    object_store = InMemoryObjectStore()

    worker_a = _make_worker(
        store=store,
        entity_lock=_NoOpLock(),
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={"doc_A": [EntityRecord("Linus", "Person", "doc_A says", ("doc_A-v1-c0",))]},
    )
    worker_b = _make_worker(
        store=store,
        entity_lock=_NoOpLock(),
        object_store=object_store,
        document_id="doc_B",
        entities_per_doc={"doc_B": [EntityRecord("Linus", "Person", "doc_B says", ("doc_B-v1-c0",))]},
    )

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")
    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_B", parse_version="v1")

    derive_a = await worker_a.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")
    derive_b = await worker_b.derive(document_id="doc_B", parse_version="v1", source_path="<irrelevant>")

    await asyncio.gather(
        worker_a.sync(
            document_id="doc_A",
            parse_version="v1",
            derived_artifact_path=derive_a.derived_artifact_path,
        ),
        worker_b.sync(
            document_id="doc_B",
            parse_version="v1",
            derived_artifact_path=derive_b.derived_artifact_path,
        ),
    )

    entity = await store.get_entity("Linus")
    assert entity is not None
    # Without the lock the race provocateur drops one writer's
    # lineage member: only one of (doc_A, v1) / (doc_B, v1) survives.
    surviving = {(m.document_id, m.parse_version) for m in entity.source_lineage}
    assert len(surviving) == 1, f"expected exactly one writer to win the race, got {surviving}"


# ---------------------------------------------------------------------
# Group 5: kg.jsonl serialization round-trip
# ---------------------------------------------------------------------


def test_kg_jsonl_round_trip_preserves_records():
    entities = [
        EntityRecord("Linus", "Person", "kernel hacker", ("c0",)),
        EntityRecord("Linux", "Project", "kernel", ("c0", "c1")),
    ]
    relations = [
        RelationRecord("Linus", "Linux", "created", "rel desc", ("c0",)),
    ]

    body = serialize_kg_jsonl(entities, relations)
    parsed_entities, parsed_relations = parse_kg_jsonl(body)

    assert parsed_entities == entities
    assert parsed_relations == relations


def test_kg_jsonl_skips_unknown_kinds_gracefully():
    # Forward-compatible: an older worker reads a newer artifact and
    # silently skips kinds it does not know yet.
    body = (
        b'{"kind": "entity", "name": "X", "entity_type": "Y", "description": "", "source_chunk_ids": []}\n'
        b'{"kind": "future_kind", "data": "..."}\n'
        b'{"kind": "relation", "source": "X", "target": "Z", "relation_type": "rel", "description": "", "source_chunk_ids": []}\n'
    )
    entities, relations = parse_kg_jsonl(body)
    assert len(entities) == 1
    assert len(relations) == 1


def test_kg_jsonl_empty_body_round_trips():
    body = serialize_kg_jsonl([], [])
    # Always at least one byte so the §C.7 "empty == derive not
    # finished" sentinel cannot collide with a deliberate "no records"
    # payload that the deletion flow publishes.
    assert body == b"\n"
    entities, relations = parse_kg_jsonl(body)
    assert entities == []
    assert relations == []


# ---------------------------------------------------------------------
# Group 6: derive's contract — read chunks.jsonl + write kg.jsonl atomically
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_writes_kg_jsonl_under_canonical_path():
    object_store = InMemoryObjectStore()
    store = InMemoryLineageGraphStore()
    entity_lock = InMemoryEntityLock()

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")

    worker = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={"doc_A": [EntityRecord("Linus", "Person", "...", ("doc_A-v1-c0",))]},
    )

    result = await worker.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")

    expected_path = derived_artifact(
        collection_id=COLLECTION_ID,
        document_id="doc_A",
        parse_version="v1",
        filename=KG_ARTIFACT_FILENAME,
    )
    assert result.derived_artifact_path == expected_path
    assert object_store.obj_exists(expected_path)


@pytest.mark.asyncio
async def test_sync_with_missing_artifact_is_a_noop():
    """Per §C.7, a missing derived artifact is treated as
    'derive not yet finished' — sync must no-op rather than raise.
    """
    object_store = InMemoryObjectStore()
    store = InMemoryLineageGraphStore()
    entity_lock = InMemoryEntityLock()

    worker = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
    )

    missing_path = derived_artifact(
        collection_id=COLLECTION_ID,
        document_id="doc_A",
        parse_version="v_missing",
        filename=KG_ARTIFACT_FILENAME,
    )
    # No object at this path; the call should return cleanly.
    await worker.sync(
        document_id="doc_A",
        parse_version="v_missing",
        derived_artifact_path=missing_path,
    )

    # The store remains empty.
    assert await store.get_entity("anything") is None


# ---------------------------------------------------------------------
# Group 6.5: tenant_scope_key propagation (§H.2 + architect msg=c3b0ba5b)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tenant_scope_key_propagates_into_lineage_members(store, entity_lock, object_store):
    """Architect msg=c3b0ba5b lock: graph worker captures the
    orchestrator-supplied ``tenant_scope_key`` into every lineage SET
    element, not at entity row level. This test pins that placement
    so a future regression that drops the field or moves it to
    entity row level fails loudly.
    """
    entities_per_doc = {
        "doc_alice": [EntityRecord("Linus", "Person", "Alice doc", ("doc_alice-v1-c0",))],
        "doc_bob": [EntityRecord("Linus", "Person", "Bob doc", ("doc_bob-v1-c0",))],
    }

    # Two different tenants citing the same shared entity.
    worker_alice = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_alice",
        entities_per_doc=entities_per_doc,
        tenant_scope_key="user:alice",
    )
    await _derive_then_sync(
        worker=worker_alice,
        document_id="doc_alice",
        parse_version="v1",
        object_store=object_store,
    )

    worker_bob = _make_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_bob",
        entities_per_doc=entities_per_doc,
        tenant_scope_key="user:bob",
    )
    await _derive_then_sync(
        worker=worker_bob,
        document_id="doc_bob",
        parse_version="v1",
        object_store=object_store,
    )

    entity = await store.get_entity("Linus")
    assert entity is not None
    # Both lineage members preserved (shared-entity model intact)
    # AND each carries its originating tenant_scope_key.
    by_doc = {m.document_id: m for m in entity.source_lineage}
    assert set(by_doc.keys()) == {"doc_alice", "doc_bob"}
    assert by_doc["doc_alice"].tenant_scope_key == "user:alice"
    assert by_doc["doc_bob"].tenant_scope_key == "user:bob"


# ---------------------------------------------------------------------
# Group 7: end-to-end with the real T1.1 parser
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_with_real_parser_chunks():
    """A small end-to-end run: parse a tiny markdown body via the
    T1.1 parser, then run derive + sync with a deterministic
    extractor that reads chunk ids from the parsed output. Pins
    that graph.derive cooperates with the chunks.jsonl shape T1.1
    actually produces (no hidden coupling beyond the §C.6 contract).
    """
    object_store = InMemoryObjectStore()
    store = InMemoryLineageGraphStore()
    entity_lock = InMemoryEntityLock()

    parse_result = parse_document(
        store=object_store,
        collection_id=COLLECTION_ID,
        document_id="doc_e2e",
        source_bytes=b"# Linus\n\nKernel hacker.\n\n# Linux\n\nKernel project.\n",
        config=ParseConfig(),
    )

    # Extractor that emits one entity per chunk it reads (verifies
    # the integration with read_chunks).
    async def extractor(chunks):
        return (
            [
                EntityRecord(
                    name=f"E_{c['chunk_id']}",
                    entity_type="Test",
                    description=c["text"],
                    source_chunk_ids=(c["chunk_id"],),
                )
                for c in chunks
            ],
            [],
        )

    worker = GraphModalityWorker(
        store=store,
        extractor=extractor,
        entity_lock=entity_lock,
        object_store=object_store,
        collection_id=COLLECTION_ID,
        tenant_scope_key=DEFAULT_TENANT,
    )

    derive_result = await worker.derive(
        document_id="doc_e2e",
        parse_version=parse_result.parse_version,
        source_path="<irrelevant>",
    )
    await worker.sync(
        document_id="doc_e2e",
        parse_version=parse_result.parse_version,
        derived_artifact_path=derive_result.derived_artifact_path,
    )

    # At least one entity created from a real chunk id.
    body = object_store.get(derive_result.derived_artifact_path)
    assert body is not None
    entities, _relations = parse_kg_jsonl(body.read())
    assert len(entities) >= 1
    assert all(e.name.startswith("E_") for e in entities)


# ---------------------------------------------------------------------
# Group 5: Wave 7 W7-1 — ``compacted_description`` field + unconditional
# ``delete_entity`` / ``delete_relation`` against the InMemory reference
# implementation. The cross-backend versions of these tests live in
# ``tests/integration/compat/test_lineage_graph_compat.py``; the unit
# tests here pin the reference oracle (the InMemory store is the
# canonical correctness target every backend must match).
# ---------------------------------------------------------------------


_LINEAGE_W7_DOC_A_V1 = LineageMember(
    document_id="doc_A",
    parse_version="v1",
    tenant_scope_key="tenant-X",
    chunk_ids=("c0",),
)

_LINEAGE_W7_DOC_A_V2 = LineageMember(
    document_id="doc_A",
    parse_version="v2",
    tenant_scope_key="tenant-X",
    chunk_ids=("c1",),
)


def _record(name: str = "Alice", *, description: str = "raw text", chunk: str = "c0") -> EntityRecord:
    return EntityRecord(
        name=name,
        entity_type="Person",
        description=description,
        source_chunk_ids=(chunk,),
    )


def _relation_record(
    source: str = "Alice",
    target: str = "Bob",
    *,
    relation_type: str = "knows",
    description: str = "they know each other",
    chunk: str = "c0",
) -> RelationRecord:
    return RelationRecord(
        source=source,
        target=target,
        relation_type=relation_type,
        description=description,
        source_chunk_ids=(chunk,),
    )


@pytest.mark.asyncio
async def test_w7_compacted_description_defaults_to_none():
    store = InMemoryLineageGraphStore()
    await store.upsert_entity_with_lineage(record=_record(), lineage=_LINEAGE_W7_DOC_A_V1)
    got = await store.get_entity("Alice")
    assert got is not None
    assert got.compacted_description is None


@pytest.mark.asyncio
async def test_w7_compacted_description_round_trip():
    store = InMemoryLineageGraphStore()
    await store.upsert_entity_with_lineage(
        record=_record(),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="LLM summary text",
    )
    got = await store.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == "LLM summary text"


@pytest.mark.asyncio
async def test_w7_compacted_description_preserved_on_subsequent_none_kwarg():
    """COALESCE invariant on the InMemory reference store — ``None``
    kwarg on a subsequent upsert MUST preserve the existing value."""
    store = InMemoryLineageGraphStore()
    await store.upsert_entity_with_lineage(
        record=_record(),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="v1 summary",
    )
    # Indexer-side re-sync — same lineage key, no compacted kwarg.
    await store.upsert_entity_with_lineage(
        record=_record(description="re-extracted"),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    got = await store.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == "v1 summary"


@pytest.mark.asyncio
async def test_w7_compacted_description_overwritten_on_non_none_kwarg():
    store = InMemoryLineageGraphStore()
    await store.upsert_entity_with_lineage(
        record=_record(),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="v1",
    )
    await store.upsert_entity_with_lineage(
        record=_record(),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="v2",
    )
    got = await store.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == "v2"


@pytest.mark.asyncio
async def test_w7_compacted_write_does_not_clobber_lineage_state():
    """huangheng safety gate (msg=828c83cc): the Compactor write path
    must preserve every other lineage field on the row."""
    store = InMemoryLineageGraphStore()
    await store.upsert_entity_with_lineage(record=_record(description="v1 text"), lineage=_LINEAGE_W7_DOC_A_V1)
    await store.upsert_entity_with_lineage(
        record=_record(description="v2 text", chunk="c1"),
        lineage=_LINEAGE_W7_DOC_A_V2,
    )
    pre = await store.get_entity("Alice")
    assert pre is not None
    pre_lineage_keys = {(m.document_id, m.parse_version) for m in pre.source_lineage}
    pre_part_keys = {(p.document_id, p.parse_version, p.text) for p in pre.description_parts}

    # Compactor-style write — passes the v1 record again with a
    # compacted summary covering both v1 and v2.
    await store.upsert_entity_with_lineage(
        record=_record(description="v1 text"),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="v1+v2 unified summary",
    )
    after = await store.get_entity("Alice")
    assert after is not None
    after_lineage_keys = {(m.document_id, m.parse_version) for m in after.source_lineage}
    after_part_keys = {(p.document_id, p.parse_version, p.text) for p in after.description_parts}

    assert after_lineage_keys == pre_lineage_keys
    assert after_part_keys == pre_part_keys
    assert after.compacted_description == "v1+v2 unified summary"


@pytest.mark.asyncio
async def test_w7_relation_compacted_description_round_trip():
    store = InMemoryLineageGraphStore()
    await store.upsert_relation_with_lineage(
        record=_relation_record(),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="rel summary",
    )
    got = await store.get_relation("Alice", "Bob", "knows")
    assert got is not None
    assert got.compacted_description == "rel summary"


@pytest.mark.asyncio
async def test_w7_delete_entity_unconditionally_removes():
    store = InMemoryLineageGraphStore()
    await store.upsert_entity_with_lineage(record=_record(), lineage=_LINEAGE_W7_DOC_A_V1)
    deleted = await store.delete_entity("Alice")
    assert deleted is True
    assert await store.get_entity("Alice") is None


@pytest.mark.asyncio
async def test_w7_delete_entity_returns_false_when_absent():
    store = InMemoryLineageGraphStore()
    deleted = await store.delete_entity("DoesNotExist")
    assert deleted is False


@pytest.mark.asyncio
async def test_w7_delete_relation_unconditionally_removes():
    store = InMemoryLineageGraphStore()
    await store.upsert_relation_with_lineage(record=_relation_record(), lineage=_LINEAGE_W7_DOC_A_V1)
    deleted = await store.delete_relation("Alice", "Bob", "knows")
    assert deleted is True
    assert await store.get_relation("Alice", "Bob", "knows") is None


@pytest.mark.asyncio
async def test_w7_delete_relation_returns_false_when_absent():
    store = InMemoryLineageGraphStore()
    deleted = await store.delete_relation("Alice", "Bob", "knows")
    assert deleted is False


# ---------------------------------------------------------------------
# Group 6: Wave 7 W7-3 — ``GraphModalityWorker.sync()`` Phase 3
# extension (compactor → embed → vector upsert → snapshot-diff
# delete → merge candidate detector). Covers the InMemory reference
# store; cross-backend coverage of the storage primitives lives in
# ``tests/integration/compat/test_lineage_graph_compat.py`` (W7-1).
# ---------------------------------------------------------------------


from uuid import NAMESPACE_DNS, uuid5  # noqa: E402 — Wave 7 W7-3 group section import

from aperag.vectorstore.dto import VectorPoint  # noqa: E402 — same


class _StubCompactor:
    """Stub :class:`GraphIndexCompactor` — returns whatever the test
    wired into ``response`` (None to opt-out, str to overwrite)."""

    def __init__(self, response: str | None = None, raise_after: int = 0) -> None:
        self.response = response
        self.calls: list[list[str]] = []
        self.kwarg_calls: list[dict[str, str]] = []
        self._raise_after = raise_after

    async def compact_if_oversized(
        self,
        parts: list[str],
        *,
        subject_kind: str,
        subject_label: str,
        language: str = "en",
    ) -> str | None:
        self.calls.append(list(parts))
        self.kwarg_calls.append(
            {
                "subject_kind": subject_kind,
                "subject_label": subject_label,
                "language": language,
            }
        )
        if self._raise_after and len(self.calls) >= self._raise_after:
            raise RuntimeError("simulated compactor failure")
        return self.response


class _StubVectorConnector:
    """Captures upsert / delete calls against a deterministic embedder."""

    def __init__(self, raise_on_upsert: bool = False) -> None:
        self.upserts: list[list[VectorPoint]] = []
        self.deletes: list[list[str]] = []
        self._raise_on_upsert = raise_on_upsert

    def upsert(self, points: list[VectorPoint]) -> list[str]:
        if self._raise_on_upsert:
            raise RuntimeError("simulated vector upsert failure")
        self.upserts.append(list(points))
        return [p.id for p in points]

    def delete(self, ids: list[str]) -> None:
        self.deletes.append(list(ids))


def _stub_embedder(text: str) -> list[float]:
    # Deterministic 4-dim embedding so tests can compare equality.
    h = abs(hash(text)) % 1000
    return [float(h % 7) / 7.0, float(h % 11) / 11.0, float(h % 13) / 13.0, float(h % 17) / 17.0]


class _StubMergeDetector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def detect_for_sync(self, *, sync_run_id: str, affected_entity_names: Sequence[str]) -> int:
        self.calls.append((sync_run_id, list(affected_entity_names)))
        return len(affected_entity_names)


def _phase3_worker(
    *,
    store: InMemoryLineageGraphStore,
    entity_lock: EntityLock,
    object_store: InMemoryObjectStore,
    document_id: str,
    entities_per_doc: dict[str, list[EntityRecord]] | None = None,
    relations_per_doc: dict[str, list[RelationRecord]] | None = None,
    compactor: Any = None,
    embedder: Any = None,
    vector_connector: Any = None,
    merge_detector: Any = None,
) -> GraphModalityWorker:
    async def extractor(
        chunks: Sequence[dict[str, Any]],
    ) -> tuple[list[EntityRecord], list[RelationRecord]]:
        del chunks
        return (
            list((entities_per_doc or {}).get(document_id, [])),
            list((relations_per_doc or {}).get(document_id, [])),
        )

    return GraphModalityWorker(
        store=store,
        extractor=extractor,
        entity_lock=entity_lock,
        object_store=object_store,
        collection_id=COLLECTION_ID,
        tenant_scope_key=DEFAULT_TENANT,
        compactor=compactor,
        embedder=embedder,
        vector_connector=vector_connector,
        merge_detector=merge_detector,
    )


def _expected_entity_id(name: str) -> str:
    return str(uuid5(NAMESPACE_DNS, f"graph_entity:{COLLECTION_ID}:{name}"))


def _expected_relation_id(source: str, target: str, type_: str) -> str:
    return str(uuid5(NAMESPACE_DNS, f"graph_relation:{COLLECTION_ID}:{source}->{target}:{type_}"))


@pytest.mark.asyncio
async def test_w7_phase3_skipped_when_vector_deps_unwired(store, entity_lock, object_store):
    """Wave 6 backward-compat: a worker with no embedder /
    vector_connector behaves exactly as the lineage-only Wave 6
    version — no Phase 3 side effects."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="kernel hacker",
                source_chunk_ids=("doc_A-v1-c0",),
            )
        ]
    }
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        # All Phase 3 deps deliberately None.
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    # Lineage rebuild still happened (Wave 6 path).
    entity = await store.get_entity("Linus")
    assert entity is not None
    assert entity.compacted_description is None  # Phase 3 skipped — no compactor write.


@pytest.mark.asyncio
async def test_w7_phase3_writes_vector_point_with_3_field_payload(store, entity_lock, object_store):
    """Spec §K.12.5 lock: vector point payload is 3 fields exactly
    (``indexer`` / ``entity_name`` / ``entity_type``); no
    ``collection_id`` payload (that lives in the uuid5 id instead so
    a shared backing store still has cross-collection unique ids)."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="kernel hacker",
                source_chunk_ids=("doc_A-v1-c0",),
            )
        ]
    }
    vector = _StubVectorConnector()
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )

    assert len(vector.upserts) == 1
    [point] = vector.upserts[0]
    assert point.id == _expected_entity_id("Linus")
    assert point.payload == {
        "indexer": "graph_entity",
        "entity_name": "Linus",
        "entity_type": "Person",
    }
    # 3 fields strict — no collection_id leakage.
    assert "collection_id" not in point.payload


@pytest.mark.asyncio
async def test_w7_phase3_uuid5_id_is_deterministic(store, entity_lock, object_store):
    """Same (collection, name) MUST produce identical uuid5 across
    calls so vector upsert overwrites the existing point instead of
    leaving stale duplicates (forward-only retry safety)."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="rev1",
                source_chunk_ids=("c0",),
            )
        ]
    }
    vector = _StubVectorConnector()
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    # First sync.
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    first_id = vector.upserts[0][0].id
    # Re-sync same doc same content — id MUST match (deterministic).
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    assert vector.upserts[1][0].id == first_id
    assert first_id == _expected_entity_id("Linus")


@pytest.mark.asyncio
async def test_w7_phase3_compactor_runs_before_embed(store, entity_lock, object_store):
    """Spec §K.12.3 invariant #3 ordering: compactor MUST run before
    the vector embed step so the embedded text is the LLM-summarised
    version, not the raw concat. We assert by wiring a compactor that
    returns ``"COMPACTED"`` and checking that the embed input matches."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="raw text",
                source_chunk_ids=("c0",),
            )
        ]
    }
    compactor = _StubCompactor(response="COMPACTED")
    vector = _StubVectorConnector()
    captured_embed_inputs: list[str] = []

    def capturing_embedder(text: str) -> list[float]:
        captured_embed_inputs.append(text)
        return _stub_embedder(text)

    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        compactor=compactor,
        embedder=capturing_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    # Compactor was called with the per-doc parts text.
    assert compactor.calls == [["raw text"]]
    # Embed input is the compacted summary, not the raw text.
    assert captured_embed_inputs == ["COMPACTED"]
    # Storage row also carries the compacted value for downstream readers.
    entity = await store.get_entity("Linus")
    assert entity is not None
    assert entity.compacted_description == "COMPACTED"


@pytest.mark.asyncio
async def test_w7_phase3_compactor_receives_entity_and_relation_context(
    store,
    entity_lock,
    object_store,
):
    """Regression for #1866: GraphModalityWorker must pass the
    compactor's required subject context for both entity and relation
    compaction calls."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Alice",
                entity_type="Person",
                description="Alice description",
                source_chunk_ids=("c0",),
            ),
            EntityRecord(
                name="Bob",
                entity_type="Person",
                description="Bob description",
                source_chunk_ids=("c0",),
            ),
        ]
    }
    relations_per_doc = {
        "doc_A": [
            RelationRecord(
                source="Alice",
                target="Bob",
                relation_type="knows",
                description="Alice knows Bob",
                source_chunk_ids=("c0",),
            )
        ]
    }
    compactor = _StubCompactor(response=None)
    vector = _StubVectorConnector()
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        relations_per_doc=relations_per_doc,
        compactor=compactor,
        embedder=_stub_embedder,
        vector_connector=vector,
    )

    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )

    assert compactor.calls == [
        ["Alice description"],
        ["Bob description"],
        ["Alice knows Bob"],
    ]
    assert compactor.kwarg_calls == [
        {
            "subject_kind": "entity",
            "subject_label": "Alice",
            "language": "en",
        },
        {
            "subject_kind": "entity",
            "subject_label": "Bob",
            "language": "en",
        },
        {
            "subject_kind": "relation",
            "subject_label": "Alice -> Bob",
            "language": "en",
        },
    ]


@pytest.mark.asyncio
async def test_w7_phase3_compactor_none_falls_back_to_raw_text(store, entity_lock, object_store):
    """Compactor returning ``None`` (below threshold) leaves the
    storage column untouched and the embedder gets the
    ``name + raw parts`` fallback string."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="short",
                source_chunk_ids=("c0",),
            )
        ]
    }
    compactor = _StubCompactor(response=None)
    vector = _StubVectorConnector()
    captured: list[str] = []

    def capturing(text: str) -> list[float]:
        captured.append(text)
        return _stub_embedder(text)

    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        compactor=compactor,
        embedder=capturing,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    assert captured == ["Linus\n\nshort"]
    entity = await store.get_entity("Linus")
    assert entity is not None
    assert entity.compacted_description is None  # COALESCE preserve.


@pytest.mark.asyncio
async def test_w7_phase3_snapshot_diff_deletes_gc_vector_points(store, entity_lock, object_store):
    """Doc-delete cascade: the entity gc'd by Phase 1 MUST have its
    vector point deleted in Phase 3 step C — using the lineage-store
    name set diff (not an ANN list-all per invariant #7)."""
    # Sync 1: doc_A produces ``Linus``.
    entities_per_doc_v1 = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="kernel hacker",
                source_chunk_ids=("c0",),
            )
        ]
    }
    vector = _StubVectorConnector()
    worker_v1 = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc_v1,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker_v1,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    assert len(vector.upserts) == 1  # Linus upserted.

    # Sync 2: doc_A re-parsed but no longer mentions Linus → entity
    # gc'd → vector point should be deleted by snapshot-diff.
    entities_per_doc_v2 = {"doc_A": []}
    worker_v2 = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc_v2,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker_v2,
        document_id="doc_A",
        parse_version="v2",
        object_store=object_store,
    )
    # No new upsert (kg.jsonl empty).
    assert len(vector.upserts) == 1
    # Snapshot-diff issued a delete for Linus' vector point.
    assert vector.deletes == [[_expected_entity_id("Linus")]]


@pytest.mark.asyncio
async def test_w7_phase3_snapshot_diff_preserves_cross_doc_entity(store, entity_lock, object_store):
    """Cross-doc shared entity: when doc_A is re-parsed and drops
    Linus, but doc_B still mentions him, Phase 3 MUST NOT delete the
    vector point (post_sync set still contains Linus from doc_B)."""
    # Seed both docs.
    vector = _StubVectorConnector()
    entities_per_doc = {
        "doc_A": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="kernel hacker per doc_A",
                source_chunk_ids=("c0",),
            )
        ],
        "doc_B": [
            EntityRecord(
                name="Linus",
                entity_type="Person",
                description="kernel hacker per doc_B",
                source_chunk_ids=("c10",),
            )
        ],
    }
    worker_a = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    worker_b = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_B",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker_a,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    await _derive_then_sync(
        worker=worker_b,
        document_id="doc_B",
        parse_version="v1",
        object_store=object_store,
    )
    # Re-sync doc_A with no entities — Linus survives via doc_B.
    entities_per_doc_v2 = {"doc_A": [], "doc_B": entities_per_doc["doc_B"]}
    worker_a_v2 = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc_v2,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker_a_v2,
        document_id="doc_A",
        parse_version="v2",
        object_store=object_store,
    )
    # No deletes — Linus still alive via doc_B's lineage.
    assert vector.deletes == []


@pytest.mark.asyncio
async def test_w7_phase3_relation_vector_upsert_payload_and_id(store, entity_lock, object_store):
    """Relation Phase 3 mirrors entity: 3-field payload with
    ``indexer="graph_relation"``, uuid5 id formatted
    ``graph_relation:<cid>:<src>-><tgt>:<type>``."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
            EntityRecord(name="Bob", entity_type="Person", description="b", source_chunk_ids=("c0",)),
        ]
    }
    relations_per_doc = {
        "doc_A": [
            RelationRecord(
                source="Alice",
                target="Bob",
                relation_type="knows",
                description="Alice knows Bob",
                source_chunk_ids=("c0",),
            )
        ]
    }
    vector = _StubVectorConnector()
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        relations_per_doc=relations_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    # 3 upserts: 2 entities + 1 relation.
    upserted_ids = {pts[0].id for pts in vector.upserts}
    upserted_payloads = {pts[0].payload["indexer"] for pts in vector.upserts}
    assert _expected_relation_id("Alice", "Bob", "knows") in upserted_ids
    assert "graph_relation" in upserted_payloads
    # Find the relation point and validate payload shape.
    relation_points = [pts[0] for pts in vector.upserts if pts[0].payload["indexer"] == "graph_relation"]
    assert len(relation_points) == 1
    rel_payload = relation_points[0].payload
    assert rel_payload == {
        "indexer": "graph_relation",
        "entity_name": "Alice->Bob",
        "entity_type": "knows",
    }


@pytest.mark.asyncio
async def test_w7_phase3_merge_detector_invoked_with_affected_names(store, entity_lock, object_store):
    """Step D wiring: when a detector is wired, Phase 3 calls
    ``detect_for_sync`` exactly once per sync, with the entity names
    just touched by this sync (per D-3 — incremental detection only)."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
            EntityRecord(name="Bob", entity_type="Person", description="b", source_chunk_ids=("c0",)),
        ]
    }
    detector = _StubMergeDetector()
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=_StubVectorConnector(),
        merge_detector=detector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    assert len(detector.calls) == 1
    sync_run_id, names = detector.calls[0]
    assert sync_run_id == f"{COLLECTION_ID}:doc_A:v1"
    assert sorted(names) == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_w7_phase3_merge_detector_failure_is_non_fatal(store, entity_lock, object_store):
    """Detector throwing must NOT abort sync — Phase 3 step D is
    best-effort (write-only auxiliary, not on the lineage critical
    path). The lineage rebuild + vector upsert that ran before it
    must remain intact."""

    class _RaisingDetector:
        async def detect_for_sync(self, *, sync_run_id, affected_entity_names):
            raise RuntimeError("simulated detector failure")

    entities_per_doc = {
        "doc_A": [
            EntityRecord(name="Linus", entity_type="Person", description="raw", source_chunk_ids=("c0",)),
        ]
    }
    vector = _StubVectorConnector()
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
        merge_detector=_RaisingDetector(),
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    # Lineage + vector upsert succeeded.
    assert (await store.get_entity("Linus")) is not None
    assert len(vector.upserts) == 1


@pytest.mark.asyncio
async def test_w7_phase3_compactor_failure_falls_back(store, entity_lock, object_store):
    """Compactor failing (LLM flake) must NOT abort sync — vector
    upsert continues with the raw description fallback (forward-only
    retry safety)."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(name="Linus", entity_type="Person", description="raw", source_chunk_ids=("c0",)),
        ]
    }
    compactor = _StubCompactor(raise_after=1)
    vector = _StubVectorConnector()
    captured: list[str] = []

    def capturing(text: str) -> list[float]:
        captured.append(text)
        return _stub_embedder(text)

    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        compactor=compactor,
        embedder=capturing,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    # Embedder ran with the fallback.
    assert captured == ["Linus\n\nraw"]
    # Vector upsert still happened.
    assert len(vector.upserts) == 1


@pytest.mark.asyncio
async def test_w7_phase3_vector_upsert_failure_is_non_fatal(store, entity_lock, object_store):
    """Vector connector raising on upsert must NOT abort sync — the
    failure is logged and lineage state is unaffected."""
    entities_per_doc = {
        "doc_A": [
            EntityRecord(name="Linus", entity_type="Person", description="raw", source_chunk_ids=("c0",)),
        ]
    }
    vector = _StubVectorConnector(raise_on_upsert=True)
    worker = _phase3_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc=entities_per_doc,
        embedder=_stub_embedder,
        vector_connector=vector,
    )
    await _derive_then_sync(
        worker=worker,
        document_id="doc_A",
        parse_version="v1",
        object_store=object_store,
    )
    # Lineage row still present.
    assert (await store.get_entity("Linus")) is not None
    # No upsert recorded (raised before append).
    assert vector.upserts == []


# ---------------------------------------------------------------------
# Group 7: Wave 7 W7-10 — ``LineageGraphStore.list_entities`` Protocol
# method (paginated entity list with optional label filter).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w7_list_entities_returns_empty_for_fresh_store():
    s = InMemoryLineageGraphStore()
    assert await s.list_entities() == []


@pytest.mark.asyncio
async def test_w7_list_entities_sorted_alphabetically():
    s = InMemoryLineageGraphStore()
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Zara", entity_type="Person", description="z", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Bob", entity_type="Person", description="b", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    rows = await s.list_entities()
    assert [r.name for r in rows] == ["Alice", "Bob", "Zara"]


@pytest.mark.asyncio
async def test_w7_list_entities_label_filter():
    s = InMemoryLineageGraphStore()
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Acme", entity_type="Organization", description="o", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    persons = await s.list_entities(label="Person")
    assert [r.name for r in persons] == ["Alice"]
    orgs = await s.list_entities(label="Organization")
    assert [r.name for r in orgs] == ["Acme"]


@pytest.mark.asyncio
async def test_w7_list_entities_pagination_via_offset_limit():
    s = InMemoryLineageGraphStore()
    for name in ("Alice", "Bob", "Carol", "Dave", "Eve"):
        await s.upsert_entity_with_lineage(
            record=EntityRecord(name=name, entity_type="Person", description=name, source_chunk_ids=("c0",)),
            lineage=_LINEAGE_W7_DOC_A_V1,
        )
    page1 = await s.list_entities(limit=2, offset=0)
    page2 = await s.list_entities(limit=2, offset=2)
    page3 = await s.list_entities(limit=2, offset=4)
    assert [r.name for r in page1] == ["Alice", "Bob"]
    assert [r.name for r in page2] == ["Carol", "Dave"]
    assert [r.name for r in page3] == ["Eve"]


@pytest.mark.asyncio
async def test_w7_list_entities_zero_or_negative_limit_returns_empty():
    s = InMemoryLineageGraphStore()
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    assert await s.list_entities(limit=0) == []
    assert await s.list_entities(limit=-5) == []


@pytest.mark.asyncio
async def test_w7_list_entities_negative_offset_treated_as_zero():
    s = InMemoryLineageGraphStore()
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Alice", entity_type="Person", description="a", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
    )
    rows = await s.list_entities(offset=-10)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_w7_list_entities_returns_compacted_description():
    s = InMemoryLineageGraphStore()
    await s.upsert_entity_with_lineage(
        record=EntityRecord(name="Alice", entity_type="Person", description="raw", source_chunk_ids=("c0",)),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="LLM summary",
    )
    [row] = await s.list_entities()
    assert row.compacted_description == "LLM summary"


# ---------------------------------------------------------------------
# Wave 8 W8-2 — bulk_upsert_entity_with_lineage_parts contract on the
# InMemory reference store. Cross-backend integration tests live in
# the separate ``tests/integration/graph_storage`` cross-backend
# fixture and reuse the same scenarios.
# ---------------------------------------------------------------------


_LINEAGE_DOC_B_V1 = LineageMember(
    document_id="doc_B",
    parse_version="v1",
    tenant_scope_key="tenant-X",
    chunk_ids=("c2",),
)


@pytest.mark.asyncio
async def test_w8_bulk_upsert_empty_parts_is_noop():
    s = InMemoryLineageGraphStore()
    await s.bulk_upsert_entity_with_lineage_parts(parts=[])
    assert await s.list_entities() == []


@pytest.mark.asyncio
async def test_w8_bulk_upsert_creates_entity_with_all_parts():
    s = InMemoryLineageGraphStore()
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_record(description="part-A"), _LINEAGE_W7_DOC_A_V1),
            (_record(description="part-B"), _LINEAGE_W7_DOC_A_V2),
            (_record(description="part-C"), _LINEAGE_DOC_B_V1),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    lineage_keys = {(m.document_id, m.parse_version) for m in got.source_lineage}
    assert lineage_keys == {("doc_A", "v1"), ("doc_A", "v2"), ("doc_B", "v1")}
    part_texts = {(p.document_id, p.parse_version, p.text) for p in got.description_parts}
    assert part_texts == {
        ("doc_A", "v1", "part-A"),
        ("doc_A", "v2", "part-B"),
        ("doc_B", "v1", "part-C"),
    }


@pytest.mark.asyncio
async def test_w8_bulk_upsert_replaces_existing_keys_and_keeps_others():
    """Strip-then-append semantic — same dedup key as single upsert."""
    s = InMemoryLineageGraphStore()
    # Seed with two pre-existing parts.
    await s.upsert_entity_with_lineage(record=_record(description="old-A"), lineage=_LINEAGE_W7_DOC_A_V1)
    await s.upsert_entity_with_lineage(record=_record(description="old-B"), lineage=_LINEAGE_W7_DOC_A_V2)
    # Bulk re-upsert hits ``doc_A/v1`` (replaces) + adds ``doc_B/v1``.
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_record(description="new-A"), _LINEAGE_W7_DOC_A_V1),
            (_record(description="new-B"), _LINEAGE_DOC_B_V1),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    part_texts = {(p.document_id, p.parse_version, p.text) for p in got.description_parts}
    assert part_texts == {
        # Original ``doc_A/v1`` replaced with ``new-A``.
        ("doc_A", "v1", "new-A"),
        # ``doc_A/v2`` untouched.
        ("doc_A", "v2", "old-B"),
        # ``doc_B/v1`` newly added.
        ("doc_B", "v1", "new-B"),
    }


@pytest.mark.asyncio
async def test_w8_bulk_upsert_rejects_mismatched_names():
    s = InMemoryLineageGraphStore()
    with pytest.raises(ValueError):
        await s.bulk_upsert_entity_with_lineage_parts(
            parts=[
                (_record(name="Alice"), _LINEAGE_W7_DOC_A_V1),
                (_record(name="Bob"), _LINEAGE_W7_DOC_A_V2),
            ],
        )


@pytest.mark.asyncio
async def test_w8_bulk_upsert_dedups_within_input_last_wins():
    """Two parts in ``parts`` sharing the same dedup key — last one wins."""
    s = InMemoryLineageGraphStore()
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_record(description="first"), _LINEAGE_W7_DOC_A_V1),
            (_record(description="second"), _LINEAGE_W7_DOC_A_V1),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    [part] = got.description_parts
    assert part.text == "second"


@pytest.mark.asyncio
async def test_w8_bulk_upsert_does_not_clobber_compacted_description():
    """Bulk path NEVER touches compacted_description (preserves existing)."""
    s = InMemoryLineageGraphStore()
    await s.upsert_entity_with_lineage(
        record=_record(description="seed"),
        lineage=_LINEAGE_W7_DOC_A_V1,
        compacted_description="LLM summary survives bulk",
    )
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[(_record(description="bulk-add"), _LINEAGE_DOC_B_V1)],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == "LLM summary survives bulk"


@pytest.mark.asyncio
async def test_w8_bulk_upsert_picks_up_last_entity_type():
    """Mirror single-upsert "most recently observed type wins" rule."""
    s = InMemoryLineageGraphStore()
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (
                EntityRecord(name="Alice", entity_type="Person", description="d1", source_chunk_ids=("c0",)),
                _LINEAGE_W7_DOC_A_V1,
            ),
            (
                EntityRecord(name="Alice", entity_type="Researcher", description="d2", source_chunk_ids=("c1",)),
                _LINEAGE_W7_DOC_A_V2,
            ),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.entity_type == "Researcher"


# ---------------------------------------------------------------------
# Task #5 — GraphFactsWorker / GraphVectorsWorker 拆分
# ---------------------------------------------------------------------


def _make_facts_worker(
    *,
    store: InMemoryLineageGraphStore,
    entity_lock: EntityLock,
    object_store: InMemoryObjectStore,
    document_id: str,
    entities_per_doc: dict[str, list[EntityRecord]] | None = None,
    relations_per_doc: dict[str, list[RelationRecord]] | None = None,
):
    from aperag.indexing.graph import GraphFactsWorker

    async def extractor(chunks):
        del chunks
        return (
            list((entities_per_doc or {}).get(document_id, [])),
            list((relations_per_doc or {}).get(document_id, [])),
        )

    return GraphFactsWorker(
        store=store,
        extractor=extractor,
        entity_lock=entity_lock,
        object_store=object_store,
        collection_id=COLLECTION_ID,
        tenant_scope_key=DEFAULT_TENANT,
    )


@pytest.mark.asyncio
async def test_graph_facts_worker_writes_lineage_but_clears_description(store, entity_lock, object_store):
    """事实层 worker 完成 sync 之后, entity 应该有 lineage member, 但
    description_parts 里的文本应该是空 (compactor / search 路径会用
    ``if p.text`` 过滤掉空 part).
    """
    facts_worker = _make_facts_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={
            "doc_A": [
                EntityRecord(
                    name="Alice",
                    entity_type="Person",
                    description="原始抽取出来的描述,事实层应该把它清空",
                    source_chunk_ids=("c0",),
                )
            ]
        },
    )

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")
    derive_result = await facts_worker.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")
    await facts_worker.sync(
        document_id="doc_A",
        parse_version="v1",
        derived_artifact_path=derive_result.derived_artifact_path,
    )

    got = await store.get_entity("Alice")
    assert got is not None
    # Phase 2 lineage rebuild 仍然写了 (doc_A, v1) member
    assert _lineage_keys(got) == {("doc_A", "v1")}
    # description_parts 还是有一个 (doc_A, v1) part, 但文本为空
    parts = list(got.description_parts)
    assert len(parts) == 1
    assert parts[0].key() == ("doc_A", "v1")
    assert parts[0].text == ""


@pytest.mark.asyncio
async def test_graph_facts_worker_overwrites_legacy_compacted_description(store, entity_lock, object_store):
    """事实层 worker 应该把老 GRAPH 模态遗留的 compacted_description
    覆盖成空串. 设计文档 §4.3 第 3 点.
    """
    # 先模拟老 GRAPH 模态留下来的 compacted_description
    await store.upsert_entity_with_lineage(
        record=EntityRecord(
            name="Alice",
            entity_type="Person",
            description="老描述",
            source_chunk_ids=("c0",),
        ),
        lineage=LineageMember(
            document_id="doc_A",
            parse_version="v0",
            tenant_scope_key=DEFAULT_TENANT,
            chunk_ids=("c0",),
        ),
        compacted_description="老 GRAPH 模态留下的 LLM summary",
    )

    facts_worker = _make_facts_worker(
        store=store,
        entity_lock=entity_lock,
        object_store=object_store,
        document_id="doc_A",
        entities_per_doc={
            "doc_A": [
                EntityRecord(
                    name="Alice",
                    entity_type="Person",
                    description="新描述",
                    source_chunk_ids=("c0",),
                )
            ]
        },
    )

    _write_doc_chunks_jsonl(object_store=object_store, document_id="doc_A", parse_version="v1")
    derive_result = await facts_worker.derive(document_id="doc_A", parse_version="v1", source_path="<irrelevant>")
    await facts_worker.sync(
        document_id="doc_A",
        parse_version="v1",
        derived_artifact_path=derive_result.derived_artifact_path,
    )

    got = await store.get_entity("Alice")
    assert got is not None
    # 老 compacted_description 被显式覆盖为空串
    assert got.compacted_description == ""


@pytest.mark.asyncio
async def test_graph_vectors_worker_derive_reuses_facts_artifact(store, entity_lock, object_store):
    """GraphVectorsWorker.derive 应该把传入的 source_path (facts 服务行
    的 derived_artifact_path) 直接当成自己的 artifact 返回, 不再重跑
    extractor. source_path 为空时应该报错让 reconciler 重排.
    """
    from aperag.indexing.graph import GraphVectorsWorker

    async def extractor(chunks):  # pragma: no cover — 不应被调用
        del chunks
        raise AssertionError("GraphVectorsWorker.derive 不应该重跑 extractor")

    vectors_worker = GraphVectorsWorker(
        store=store,
        extractor=extractor,
        entity_lock=entity_lock,
        object_store=object_store,
        collection_id=COLLECTION_ID,
        tenant_scope_key=DEFAULT_TENANT,
    )

    facts_artifact_path = derived_artifact(
        collection_id=COLLECTION_ID,
        document_id="doc_A",
        parse_version="v1",
        filename=KG_ARTIFACT_FILENAME,
    )
    derive_result = await vectors_worker.derive(
        document_id="doc_A",
        parse_version="v1",
        source_path=facts_artifact_path,
    )
    assert derive_result.derived_artifact_path == facts_artifact_path

    with pytest.raises(ValueError, match="not ready"):
        await vectors_worker.derive(
            document_id="doc_A",
            parse_version="v1",
            source_path="",
        )


@pytest.mark.asyncio
async def test_graph_vectors_worker_sync_skips_phase_1_2(store, entity_lock, object_store):
    """GraphVectorsWorker.sync 不会清理 / 重建 lineage. 即使 store
    里没有该实体的 lineage, sync 不会自己调 upsert (因为 vectors 层
    依赖事实层的 ACTIVE 状态).

    在没有 vector_connector / embedder 时, Phase 3 也短路 (Wave 6
    backward-compat), 所以 sync 是 no-op (除了读 kg.jsonl).
    """
    from aperag.indexing.graph import GraphVectorsWorker

    async def extractor(chunks):
        del chunks
        return ([], [])

    vectors_worker = GraphVectorsWorker(
        store=store,
        extractor=extractor,
        entity_lock=entity_lock,
        object_store=object_store,
        collection_id=COLLECTION_ID,
        tenant_scope_key=DEFAULT_TENANT,
    )

    # 准备一个最小的 kg.jsonl (1 entity, 0 relation)
    body = serialize_kg_jsonl(
        [EntityRecord(name="Alice", entity_type="Person", description="d", source_chunk_ids=("c0",))],
        [],
    )
    artifact_path = derived_artifact(
        collection_id=COLLECTION_ID,
        document_id="doc_A",
        parse_version="v1",
        filename=KG_ARTIFACT_FILENAME,
    )
    write_atomic(object_store, artifact_path, body)

    # store 里没有任何 entity (没跑 facts worker)
    assert await store.get_entity("Alice") is None

    await vectors_worker.sync(
        document_id="doc_A",
        parse_version="v1",
        derived_artifact_path=artifact_path,
    )

    # 仍然没有 — vectors 层不写 lineage
    assert await store.get_entity("Alice") is None
