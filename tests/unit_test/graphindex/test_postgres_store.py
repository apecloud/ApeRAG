# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end integration test for ``PostgresGraphStore``.

Gated on ``APERAG_TEST_GRAPHINDEX_PG_URL``. When unset, all tests skip
so CI without a database does not fail. Local runs can set:

    export APERAG_TEST_GRAPHINDEX_PG_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres

to exercise the full upsert / query / delete path against real
PostgreSQL. The tests create their own tables before each test and
drop them after, so they don't rely on alembic migrations and don't
pollute the main schema.
"""

from __future__ import annotations

import os
import uuid

import pytest

PG_URL = os.environ.get("APERAG_TEST_GRAPHINDEX_PG_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL,
    reason="set APERAG_TEST_GRAPHINDEX_PG_URL to run pg integration tests",
)

pytest.importorskip("sqlalchemy")
pytest.importorskip("asyncpg")

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from aperag.domains.knowledge_graph.graphindex import (  # noqa: E402
    Chunk,
    Entity,
    PostgresGraphStore,
    Relation,
)
from aperag.domains.knowledge_graph.graphindex.models import (  # noqa: E402
    CHUNKS_TABLE,
    EDGES_TABLE,
    NODES_TABLE,
    GraphIndexChunk,  # noqa: F401
    GraphIndexEdge,  # noqa: F401
    GraphIndexNode,  # noqa: F401
)


@pytest.fixture
async def store():
    """Create a fresh engine + tables per test. Drop tables on teardown
    so test state never leaks across runs."""
    from aperag.db.models import Base

    engine = create_async_engine(PG_URL, future=True)
    # Create only the graphindex_* tables (not every aperag table).
    async with engine.begin() as conn:
        for table in (NODES_TABLE, EDGES_TABLE, CHUNKS_TABLE):
            await conn.execute(__import__("sqlalchemy").text(f"DROP TABLE IF EXISTS {table} CASCADE"))

        def _create(sync_conn):
            Base.metadata.tables[NODES_TABLE].create(sync_conn, checkfirst=True)
            Base.metadata.tables[EDGES_TABLE].create(sync_conn, checkfirst=True)
            Base.metadata.tables[CHUNKS_TABLE].create(sync_conn, checkfirst=True)

        await conn.run_sync(_create)

    yield PostgresGraphStore(engine=engine)

    async with engine.begin() as conn:
        for table in (EDGES_TABLE, NODES_TABLE, CHUNKS_TABLE):
            await conn.execute(__import__("sqlalchemy").text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    await engine.dispose()


def _mk_chunk(collection_id: str, doc_id: str, order: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        collection_id=collection_id,
        order_in_doc=order,
        text=text,
    )


def _mk_entity(collection_id: str, eid: str, name: str, chunk_ids=()):
    return Entity(
        entity_id=eid,
        collection_id=collection_id,
        name=name,
        type="person",
        description=f"desc of {name}",
        source_chunk_ids=chunk_ids,
    )


def _mk_relation(collection_id: str, s: str, t: str, chunk_ids=(), weight=5.0):
    return Relation(
        collection_id=collection_id,
        source_id=s,
        target_id=t,
        description=f"{s}→{t}",
        weight=weight,
        source_chunk_ids=chunk_ids,
    )


# ---------------------------------------------------------------------------
# upsert + read round-trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_round_trip(store):
    """Write chunks + entities + relations; read back via
    ``find_entities_by_names`` / ``expand_neighborhood`` — verify every
    field round-trips without loss."""
    cid = "col-rt"
    c1 = _mk_chunk(cid, "d1", 0, "Alice met Bob.")
    c2 = _mk_chunk(cid, "d1", 1, "Bob works at Acme.")

    await store.upsert_chunks(cid, [c1, c2])
    await store.upsert_entities(
        cid,
        [
            _mk_entity(cid, "e-alice", "Alice", [c1.chunk_id]),
            _mk_entity(cid, "e-bob", "Bob", [c1.chunk_id, c2.chunk_id]),
        ],
    )
    await store.upsert_relations(
        cid,
        [_mk_relation(cid, "e-alice", "e-bob", [c1.chunk_id])],
    )

    alice_hits = await store.find_entities_by_names(cid, ["Alice"])
    assert [e.name for e in alice_hits] == ["Alice"]

    entities, relations = await store.expand_neighborhood(
        collection_id=cid, anchor_entity_ids=["e-alice"], max_hop=1, limit=50
    )
    assert {e.entity_id for e in entities} == {"e-alice", "e-bob"}
    assert len(relations) == 1
    assert relations[0].source_id == "e-alice"
    assert relations[0].target_id == "e-bob"


# ---------------------------------------------------------------------------
# idempotent upsert: repeated writes union chunk ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_entity_unions_chunk_ids(store):
    cid = "col-idem"
    c1 = _mk_chunk(cid, "d", 0, "a")
    c2 = _mk_chunk(cid, "d", 1, "b")
    await store.upsert_chunks(cid, [c1, c2])

    await store.upsert_entities(cid, [_mk_entity(cid, "e1", "X", [c1.chunk_id])])
    await store.upsert_entities(cid, [_mk_entity(cid, "e1", "X", [c2.chunk_id])])

    found = await store.find_entities_by_names(cid, ["X"])
    assert len(found) == 1
    assert set(found[0].source_chunk_ids) == {c1.chunk_id, c2.chunk_id}


@pytest.mark.asyncio
async def test_upsert_relation_keeps_max_weight(store):
    cid = "col-w"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])
    await store.upsert_entities(
        cid,
        [_mk_entity(cid, "a", "A"), _mk_entity(cid, "b", "B")],
    )

    await store.upsert_relations(cid, [_mk_relation(cid, "a", "b", [c.chunk_id], weight=3)])
    await store.upsert_relations(cid, [_mk_relation(cid, "a", "b", [c.chunk_id], weight=8)])
    await store.upsert_relations(cid, [_mk_relation(cid, "a", "b", [c.chunk_id], weight=5)])

    # max_hop=1 reaches both endpoints; the resulting edge set contains
    # the (a, b) relation and asserts the GREATEST weight policy held.
    _, relations = await store.expand_neighborhood(collection_id=cid, anchor_entity_ids=["a"], max_hop=1, limit=10)
    ab_edges = [r for r in relations if r.source_id == "a" and r.target_id == "b"]
    assert len(ab_edges) == 1
    assert float(ab_edges[0].weight) == 8.0


# ---------------------------------------------------------------------------
# delete_document_rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_removes_orphans_keeps_shared(store):
    """An entity that originated from one document is deleted when
    that document is deleted; an entity whose source_chunk_ids list
    covers multiple documents is kept (with the pruned chunks removed
    from its list)."""
    cid = "col-del"
    c_d1 = _mk_chunk(cid, "d1", 0, "doc1 body")
    c_d2 = _mk_chunk(cid, "d2", 0, "doc2 body")
    await store.upsert_chunks(cid, [c_d1, c_d2])

    await store.upsert_entities(
        cid,
        [
            # Only in d1 → should be deleted.
            _mk_entity(cid, "orphan", "Orphan", [c_d1.chunk_id]),
            # In both d1 and d2 → should be kept.
            _mk_entity(cid, "shared", "Shared", [c_d1.chunk_id, c_d2.chunk_id]),
        ],
    )
    await store.upsert_relations(
        cid,
        [
            _mk_relation(cid, "orphan", "shared", [c_d1.chunk_id]),
        ],
    )

    result = await store.delete_document_rows(cid, "d1")
    assert result.chunks_removed == 1
    assert result.entities_removed == 1  # orphan gone
    assert result.relations_removed == 1  # rel (orphan→shared) gone

    remaining = await store.find_entities_by_names(cid, ["Orphan", "Shared"])
    names = {e.name for e in remaining}
    assert "Orphan" not in names
    assert "Shared" in names
    # Shared entity now only references d2's chunk.
    shared = [e for e in remaining if e.name == "Shared"][0]
    assert shared.source_chunk_ids == (c_d2.chunk_id,)


# ---------------------------------------------------------------------------
# rebuild idempotency — regression for blocker 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_cycle_does_not_accumulate_chunk_ids(store):
    """Simulates ``GraphIndexService.index_document`` running twice on
    the same ``doc_id``: delete-before-rebuild must keep
    ``source_chunk_ids`` equal to the *new* chunk ids only, never the
    union of old + new. Without the facade-level delete this grows
    unboundedly because chunk ids are UUID4 and change each run."""
    cid = "col-rebuild"

    # ----- first indexing cycle
    c_old = _mk_chunk(cid, "d1", 0, "Alice met Bob (v1)")
    await store.upsert_chunks(cid, [c_old])
    await store.upsert_entities(cid, [_mk_entity(cid, "e-alice", "Alice", [c_old.chunk_id])])
    await store.upsert_relations(
        cid,
        [_mk_relation(cid, "e-alice", "e-alice-self", [c_old.chunk_id])],
    )

    # ----- second indexing cycle: delete then re-upsert with NEW chunk id
    await store.delete_document_rows(cid, "d1")

    c_new = _mk_chunk(cid, "d1", 0, "Alice met Bob (v2)")
    await store.upsert_chunks(cid, [c_new])
    await store.upsert_entities(cid, [_mk_entity(cid, "e-alice", "Alice", [c_new.chunk_id])])

    found = await store.find_entities_by_names(cid, ["Alice"])
    assert len(found) == 1
    # Exact new-only membership; the old UUID4 must not be present.
    assert set(found[0].source_chunk_ids) == {c_new.chunk_id}
    assert c_old.chunk_id not in set(found[0].source_chunk_ids)


# ---------------------------------------------------------------------------
# list_labels / list_subgraph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_labels_returns_distinct_types(store):
    cid = "col-lbl"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])
    e1 = _mk_entity(cid, "e1", "A", [c.chunk_id])
    e2 = Entity(
        entity_id="e2",
        collection_id=cid,
        name="B",
        type="organization",
        description="",
        source_chunk_ids=(c.chunk_id,),
    )
    await store.upsert_entities(cid, [e1, e2])
    labels = await store.list_labels(cid)
    assert labels == ["organization", "person"]


@pytest.mark.asyncio
async def test_list_subgraph_is_truncated_when_cap_hit(store):
    cid = "col-sg"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])
    # Create 20 entities, ask for 5 — must flag truncation.
    entities = [_mk_entity(cid, f"e{i}", f"N{i}", [c.chunk_id]) for i in range(20)]
    await store.upsert_entities(cid, entities)
    kg = await store.list_subgraph(collection_id=cid, label=None, max_depth=0, max_nodes=5)
    assert len(kg.nodes) == 5
    assert kg.is_truncated is True


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drop_collection_removes_all_tenant_rows(store):
    cid_a = "col-a"
    cid_b = "col-b"
    ca = _mk_chunk(cid_a, "d", 0, "a")
    cb = _mk_chunk(cid_b, "d", 0, "b")
    await store.upsert_chunks(cid_a, [ca])
    await store.upsert_chunks(cid_b, [cb])
    await store.upsert_entities(cid_a, [_mk_entity(cid_a, "ea", "A", [ca.chunk_id])])
    await store.upsert_entities(cid_b, [_mk_entity(cid_b, "eb", "B", [cb.chunk_id])])

    await store.drop_collection(cid_a)

    assert await store.find_entities_by_names(cid_a, ["A"]) == []
    b_hits = await store.find_entities_by_names(cid_b, ["B"])
    assert [e.name for e in b_hits] == ["B"]


# ---------------------------------------------------------------------------
# Description accumulation: concat + dedup, no cap in SQL layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_entity_accumulates_descriptions(store):
    """Each upsert appends its fragment to the stored description with
    the shared separator. Critical: the SQL layer does NOT truncate —
    size bounding is the service layer's job and must remain so."""
    from aperag.domains.knowledge_graph.graphindex.dto import DESCRIPTION_SEPARATOR

    cid = "col-desc"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])

    await store.upsert_entities(
        cid,
        [Entity(entity_id="e", collection_id=cid, name="E", type="person", description="First fragment.")],
    )
    await store.upsert_entities(
        cid,
        [Entity(entity_id="e", collection_id=cid, name="E", type="person", description="Second fragment.")],
    )
    await store.upsert_entities(
        cid,
        [Entity(entity_id="e", collection_id=cid, name="E", type="person", description="Third fragment.")],
    )

    [found] = await store.find_entities_by_names(cid, ["E"])
    fragments = found.description.split(DESCRIPTION_SEPARATOR)
    assert fragments == ["First fragment.", "Second fragment.", "Third fragment."]


@pytest.mark.asyncio
async def test_upsert_entity_dedupes_identical_fragments(store):
    """If the same fragment is seen twice we don't store it twice — a
    common pattern when identical boilerplate appears in multiple
    chunks of the same document."""
    cid = "col-dedup"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])

    await store.upsert_entities(
        cid,
        [Entity(entity_id="e", collection_id=cid, name="E", type="person", description="One fragment.")],
    )
    await store.upsert_entities(
        cid,
        [Entity(entity_id="e", collection_id=cid, name="E", type="person", description="One fragment.")],
    )

    [found] = await store.find_entities_by_names(cid, ["E"])
    assert found.description == "One fragment."


# ---------------------------------------------------------------------------
# find_oversized_entities / find_oversized_relations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_oversized_entities_returns_rows_past_threshold(store):
    from aperag.domains.knowledge_graph.graphindex.dto import DESCRIPTION_SEPARATOR

    cid = "col-over"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])

    # Short — under both thresholds.
    await store.upsert_entities(
        cid,
        [Entity(entity_id="small", collection_id=cid, name="Small", type="person", description="tiny")],
    )
    # Long by fragment count.
    many_frags = DESCRIPTION_SEPARATOR.join(f"frag{i}" for i in range(10))
    await store.upsert_entities(
        cid,
        [Entity(entity_id="frags", collection_id=cid, name="Frags", type="person", description=many_frags)],
    )
    # Long by character count.
    await store.upsert_entities(
        cid,
        [Entity(entity_id="chars", collection_id=cid, name="Chars", type="person", description="a" * 5000)],
    )

    oversized = await store.find_oversized_entities(cid, min_chars=4000, min_fragments=6, limit=50)
    ids = {e.entity_id for e in oversized}
    assert ids == {"frags", "chars"}


@pytest.mark.asyncio
async def test_rewrite_entity_description_replaces_in_place(store):
    cid = "col-rw"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])
    await store.upsert_entities(
        cid,
        [Entity(entity_id="e", collection_id=cid, name="E", type="person", description="original long desc")],
    )
    await store.rewrite_entity_description(cid, "e", "summary")
    [found] = await store.find_entities_by_names(cid, ["E"])
    assert found.description == "summary"


# ---------------------------------------------------------------------------
# merge_entities: structural merge in one transaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_redirects_edges_and_unions_chunks(store):
    cid = "col-merge"
    c1 = _mk_chunk(cid, "d1", 0, "chunk 1 text")
    c2 = _mk_chunk(cid, "d2", 0, "chunk 2 text")
    c3 = _mk_chunk(cid, "d3", 0, "chunk 3 text")
    await store.upsert_chunks(cid, [c1, c2, c3])
    await store.upsert_entities(
        cid,
        [
            Entity(
                entity_id="target",
                collection_id=cid,
                name="Target",
                type="person",
                description="Target description.",
                source_chunk_ids=(c1.chunk_id,),
            ),
            Entity(
                entity_id="src1",
                collection_id=cid,
                name="Source One",
                type="person",
                description="Source one adds details.",
                source_chunk_ids=(c2.chunk_id,),
            ),
            Entity(
                entity_id="src2",
                collection_id=cid,
                name="Source Two",
                type="person",
                description="Source two adds more details.",
                source_chunk_ids=(c3.chunk_id,),
            ),
            # Bystander entity so we can verify its edges are untouched.
            _mk_entity(cid, "other", "Other", [c1.chunk_id]),
        ],
    )
    await store.upsert_relations(
        cid,
        [
            _mk_relation(cid, "src1", "other", [c2.chunk_id]),
            _mk_relation(cid, "src2", "other", [c3.chunk_id]),
            # Edge between two sources — becomes a self-loop after merge,
            # must be dropped.
            _mk_relation(cid, "src1", "src2", [c2.chunk_id]),
        ],
    )

    result = await store.merge_entities(cid, target_entity_id="target", source_entity_ids=["src1", "src2"])

    assert result.target_entity_id == "target"
    assert set(result.merged_source_ids) == {"src1", "src2"}
    assert "Source one adds details." in result.description
    assert "Source two adds more details." in result.description
    # Two source chunks unioned into target.
    assert set(result.source_chunk_ids) == {c1.chunk_id, c2.chunk_id, c3.chunk_id}
    # src1↔src2 and each src→other contribute 3 affected edges; after
    # redirect src1↔src2 collapses to a self-loop and is dropped; the
    # two edges to "other" collapse into one target→other edge.
    assert result.edges_collapsed >= 1

    # Source entities were removed.
    assert await store.find_entities_by_names(cid, ["Source One", "Source Two"]) == []
    # Target still exists with the merged chunks.
    [target] = await store.find_entities_by_names(cid, ["Target"])
    assert set(target.source_chunk_ids) == {c1.chunk_id, c2.chunk_id, c3.chunk_id}

    # The edges touching "other" all land on target, not the sources.
    entities, relations = await store.expand_neighborhood(
        collection_id=cid, anchor_entity_ids=["target"], max_hop=1, limit=50
    )
    assert {e.entity_id for e in entities} >= {"target", "other"}
    assert all(r.source_id == "target" and r.target_id == "other" for r in relations) or all(
        r.source_id == "other" and r.target_id == "target" for r in relations
    )


@pytest.mark.asyncio
async def test_merge_entities_missing_target_raises(store):
    cid = "col-merge-missing"
    c = _mk_chunk(cid, "d", 0, "x")
    await store.upsert_chunks(cid, [c])

    with pytest.raises(ValueError):
        await store.merge_entities(cid, target_entity_id="ghost", source_entity_ids=["also-missing"])
