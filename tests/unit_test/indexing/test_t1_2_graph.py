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
                type="Person",
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
    ) -> None:
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
                row = type(row)(name=record.name, type=record.type) if row is not None else None
            from aperag.indexing.graph import _InMemoryEntityRow  # noqa: PLC0415

            if row is None:
                row = _InMemoryEntityRow(name=record.name, type=record.type)
                self._entities[record.name] = row
            else:
                row.type = record.type
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
        b'{"kind": "entity", "name": "X", "type": "Y", "description": "", "source_chunk_ids": []}\n'
        b'{"kind": "future_kind", "data": "..."}\n'
        b'{"kind": "relation", "source": "X", "target": "Z", "type": "rel", "description": "", "source_chunk_ids": []}\n'
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
                    type="Test",
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
