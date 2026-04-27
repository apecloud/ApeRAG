# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cross-backend GraphStore compatibility tests.

Run the exact same deterministic scenario against every graph backend
that has a running instance. Each backend is gated on an env var so CI
can selectively enable whichever databases it spins up.

Env vars:
    COMPAT_PG_URL       = postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres
    COMPAT_NEO4J_URI    = bolt://127.0.0.1:7687
    COMPAT_NEO4J_USER   = neo4j
    COMPAT_NEO4J_PASS   = password
    COMPAT_NEBULA_HOSTS = 127.0.0.1:9669

Usage:
    # PG only (default docker-compose)
    COMPAT_PG_URL=... pytest tests/integration/compat/test_graph_compat.py -v

    # All three backends
    COMPAT_PG_URL=... COMPAT_NEO4J_URI=... COMPAT_NEBULA_HOSTS=... pytest ...

    # Via make
    make test-compat-graph
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest

from aperag.domains.knowledge_graph.graphindex.dto import Chunk, Entity, Relation

_BASE_COLLECTION_ID = "compat_test_base"

# --- fixtures: one per backend -------------------------------------------


def _make_pg_store():
    url = os.environ.get("COMPAT_PG_URL")
    if not url:
        return None
    from sqlalchemy.ext.asyncio import create_async_engine

    from aperag.domains.knowledge_graph.graphindex.storage.postgres import PostgresGraphStore

    engine = create_async_engine(url, future=True)
    return PostgresGraphStore(engine=engine), engine


def _make_neo4j_store():
    uri = os.environ.get("COMPAT_NEO4J_URI")
    if not uri:
        return None
    from aperag.domains.knowledge_graph.graphindex.storage.neo4j import Neo4jGraphStore

    return Neo4jGraphStore(
        uri=uri,
        username=os.environ.get("COMPAT_NEO4J_USER", "neo4j"),
        password=os.environ.get("COMPAT_NEO4J_PASS", "password"),
    ), None


def _make_nebula_store():
    hosts = os.environ.get("COMPAT_NEBULA_HOSTS")
    if not hosts:
        return None
    from aperag.domains.knowledge_graph.graphindex.storage.nebula import NebulaGraphStore

    return NebulaGraphStore(
        hosts=hosts,
        username=os.environ.get("COMPAT_NEBULA_USER", "root"),
        password=os.environ.get("COMPAT_NEBULA_PASS", "nebula"),
        space_prefix="compat_test",
    ), None


_BACKENDS = {
    "postgresql": _make_pg_store,
    "neo4j": _make_neo4j_store,
    "nebula": _make_nebula_store,
}


def _available_backend_names():
    available = []
    if os.environ.get("COMPAT_PG_URL"):
        available.append(pytest.param("postgresql", id="postgresql"))
    if os.environ.get("COMPAT_NEO4J_URI"):
        available.append(pytest.param("neo4j", id="neo4j"))
    if os.environ.get("COMPAT_NEBULA_HOSTS"):
        available.append(pytest.param("nebula", id="nebula"))
    return available


_available = _available_backend_names()

if not _available:
    pytestmark = pytest.mark.skip(
        reason="No graph backend env vars set (COMPAT_PG_URL / COMPAT_NEO4J_URI / COMPAT_NEBULA_HOSTS)"
    )


@pytest.fixture
def collection_id() -> str:
    return f"compat_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(params=_available if _available else [pytest.param(None, marks=pytest.mark.skip)])
async def store(request, collection_id):
    """Yield a (backend_name, store) tuple. Clean up after test."""
    name = request.param
    store_result = _BACKENDS[name]()
    assert store_result is not None
    store_obj, engine = store_result

    # For PG: ensure tables exist
    if name == "postgresql" and engine is not None:
        from aperag.db.models import Base
        from aperag.domains.knowledge_graph.graphindex.models import CHUNKS_TABLE, EDGES_TABLE, NODES_TABLE

        async with engine.begin() as conn:
            for table in (NODES_TABLE, EDGES_TABLE, CHUNKS_TABLE):
                await conn.execute(__import__("sqlalchemy").text(f"DROP TABLE IF EXISTS {table} CASCADE"))

            def _create(sync_conn):
                Base.metadata.tables[NODES_TABLE].create(sync_conn, checkfirst=True)
                Base.metadata.tables[EDGES_TABLE].create(sync_conn, checkfirst=True)
                Base.metadata.tables[CHUNKS_TABLE].create(sync_conn, checkfirst=True)

            await conn.run_sync(_create)

    yield name, store_obj

    # Cleanup
    try:
        await store_obj.drop_collection(collection_id)
    except Exception:
        pass
    try:
        close = getattr(store_obj, "close", None)
        if close is not None:
            await close()
    except Exception:
        pass
    if name == "postgresql" and engine is not None:
        await engine.dispose()


# --- deterministic test data ---------------------------------------------

CHUNK_1 = Chunk(
    chunk_id="c1", doc_id="d1", collection_id=_BASE_COLLECTION_ID, order_in_doc=0, text="Alice met Bob at Acme Labs."
)
CHUNK_2 = Chunk(
    chunk_id="c2", doc_id="d1", collection_id=_BASE_COLLECTION_ID, order_in_doc=1, text="Bob works at Acme Labs on AI."
)

ENTITY_ALICE = Entity(
    entity_id="e-alice",
    collection_id=_BASE_COLLECTION_ID,
    name="Alice",
    type="person",
    description="A researcher",
    source_chunk_ids=("c1",),
)
ENTITY_BOB = Entity(
    entity_id="e-bob",
    collection_id=_BASE_COLLECTION_ID,
    name="Bob",
    type="person",
    description="An engineer",
    source_chunk_ids=("c1", "c2"),
)
ENTITY_ACME = Entity(
    entity_id="e-acme",
    collection_id=_BASE_COLLECTION_ID,
    name="Acme Labs",
    type="organization",
    description="A research lab",
    source_chunk_ids=("c1", "c2"),
)

REL_ALICE_BOB = Relation(
    collection_id=_BASE_COLLECTION_ID,
    source_id="e-alice",
    target_id="e-bob",
    description="Alice met Bob",
    weight=7.0,
    source_chunk_ids=("c1",),
)
REL_BOB_ACME = Relation(
    collection_id=_BASE_COLLECTION_ID,
    source_id="e-bob",
    target_id="e-acme",
    description="Bob works at Acme",
    weight=8.0,
    source_chunk_ids=("c1", "c2"),
)


def _graph_data(collection_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_1=replace(CHUNK_1, collection_id=collection_id),
        chunk_2=replace(CHUNK_2, collection_id=collection_id),
        entity_alice=replace(ENTITY_ALICE, collection_id=collection_id),
        entity_bob=replace(ENTITY_BOB, collection_id=collection_id),
        entity_acme=replace(ENTITY_ACME, collection_id=collection_id),
        rel_alice_bob=replace(REL_ALICE_BOB, collection_id=collection_id),
        rel_bob_acme=replace(REL_BOB_ACME, collection_id=collection_id),
    )


# --- the tests -----------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_and_find_entities(store, collection_id):
    """Write entities, read them back by name — verify fields round-trip."""
    name, s = store
    graph = _graph_data(collection_id)
    await s.upsert_chunks(collection_id, [graph.chunk_1, graph.chunk_2])
    await s.upsert_entities(collection_id, [graph.entity_alice, graph.entity_bob, graph.entity_acme])

    found = await s.find_entities_by_names(collection_id, ["Alice", "Bob"])
    names = {e.name for e in found}
    assert "Alice" in names, f"[{name}] Alice not found"
    assert "Bob" in names, f"[{name}] Bob not found"

    found_by_id = await s.find_entities_by_ids(collection_id, ["e-alice", "e-acme"])
    ids = {e.entity_id for e in found_by_id}
    assert "e-alice" in ids, f"[{name}] find_entities_by_ids failed for e-alice"
    assert "e-acme" in ids, f"[{name}] find_entities_by_ids failed for e-acme"


@pytest.mark.asyncio
async def test_upsert_relations_and_expand(store, collection_id):
    """Write relations, BFS-expand from an anchor."""
    name, s = store
    graph = _graph_data(collection_id)
    await s.upsert_chunks(collection_id, [graph.chunk_1, graph.chunk_2])
    await s.upsert_entities(collection_id, [graph.entity_alice, graph.entity_bob, graph.entity_acme])
    await s.upsert_relations(collection_id, [graph.rel_alice_bob, graph.rel_bob_acme])

    entities, relations = await s.expand_neighborhood(collection_id, ["e-alice"], max_hop=2, limit=50)
    entity_ids = {e.entity_id for e in entities}
    assert "e-alice" in entity_ids, f"[{name}] anchor not in expansion"
    assert "e-bob" in entity_ids, f"[{name}] 1-hop neighbor not found"
    assert len(relations) >= 1, f"[{name}] no relations in expansion"


@pytest.mark.asyncio
async def test_description_accumulation(store, collection_id):
    """Multiple upserts for the same entity should accumulate descriptions."""
    name, s = store
    graph = _graph_data(collection_id)
    await s.upsert_chunks(collection_id, [graph.chunk_1])
    await s.upsert_entities(collection_id, [graph.entity_alice])
    # Second upsert with different description
    updated = Entity(
        entity_id="e-alice",
        collection_id=collection_id,
        name="Alice",
        type="person",
        description="Also a teacher",
        source_chunk_ids=("c2",),
    )
    await s.upsert_entities(collection_id, [updated])

    found = await s.find_entities_by_names(collection_id, ["Alice"])
    assert len(found) == 1, f"[{name}] expected 1 entity, got {len(found)}"
    assert "researcher" in found[0].description.lower() or "teacher" in found[0].description.lower(), (
        f"[{name}] description should contain at least one fragment"
    )


@pytest.mark.asyncio
async def test_merge_entities_collapses_preexisting_target_edge(store, collection_id):
    """Merging sources into a target must also merge with the target's
    pre-existing edge, not just collapse redirected source edges."""
    name, s = store
    chunks = [
        Chunk(chunk_id="mc1", doc_id="d1", collection_id=collection_id, order_in_doc=0, text="target edge"),
        Chunk(chunk_id="mc2", doc_id="d2", collection_id=collection_id, order_in_doc=0, text="source 1 edge"),
        Chunk(chunk_id="mc3", doc_id="d3", collection_id=collection_id, order_in_doc=0, text="source 2 edge"),
    ]
    await s.upsert_chunks(collection_id, chunks)
    await s.upsert_entities(
        collection_id,
        [
            Entity(
                entity_id="e-target",
                collection_id=collection_id,
                name="Target",
                type="person",
                description="target entity",
                source_chunk_ids=("mc1",),
            ),
            Entity(
                entity_id="e-src1",
                collection_id=collection_id,
                name="Source One",
                type="person",
                description="source entity one",
                source_chunk_ids=("mc2",),
            ),
            Entity(
                entity_id="e-src2",
                collection_id=collection_id,
                name="Source Two",
                type="person",
                description="source entity two",
                source_chunk_ids=("mc3",),
            ),
            Entity(
                entity_id="e-other",
                collection_id=collection_id,
                name="Other",
                type="person",
                description="other entity",
                source_chunk_ids=("mc1", "mc2", "mc3"),
            ),
        ],
    )
    await s.upsert_relations(
        collection_id,
        [
            Relation(
                collection_id=collection_id,
                source_id="e-target",
                target_id="e-other",
                description="target pre-existing edge",
                weight=5.0,
                source_chunk_ids=("mc1",),
            ),
            Relation(
                collection_id=collection_id,
                source_id="e-src1",
                target_id="e-other",
                description="source one redirected edge",
                weight=7.0,
                source_chunk_ids=("mc2",),
            ),
            Relation(
                collection_id=collection_id,
                source_id="e-src2",
                target_id="e-other",
                description="source two redirected edge",
                weight=8.0,
                source_chunk_ids=("mc3",),
            ),
        ],
    )

    result = await s.merge_entities(collection_id, target_entity_id="e-target", source_entity_ids=["e-src1", "e-src2"])
    assert set(result.merged_source_ids) == {"e-src1", "e-src2"}, f"[{name}] expected both sources to merge"

    entities, relations = await s.expand_neighborhood(collection_id, ["e-target"], max_hop=1, limit=50)
    entity_ids = {e.entity_id for e in entities}
    assert {"e-target", "e-other"} <= entity_ids, f"[{name}] merged neighborhood missing expected entities"

    target_edges = [r for r in relations if r.source_id == "e-target" and r.target_id == "e-other"]
    assert len(target_edges) == 1, f"[{name}] expected exactly one merged target->other edge, got {relations}"

    [merged_edge] = target_edges
    assert merged_edge.weight == pytest.approx(8.0), f"[{name}] weight should keep max source/target edge weight"
    assert set(merged_edge.source_chunk_ids) == {"mc1", "mc2", "mc3"}, (
        f"[{name}] chunk provenance should be unioned across target and redirected edges"
    )
    assert "target pre-existing edge" in merged_edge.description, f"[{name}] target edge description was lost"
    assert "source one redirected edge" in merged_edge.description, f"[{name}] source 1 description was lost"
    assert "source two redirected edge" in merged_edge.description, f"[{name}] source 2 description was lost"


@pytest.mark.asyncio
async def test_delete_document_removes_orphans(store, collection_id):
    """Deleting a document should remove orphan entities/relations."""
    name, s = store
    graph = _graph_data(collection_id)
    await s.upsert_chunks(collection_id, [graph.chunk_1])
    await s.upsert_entities(
        collection_id,
        [
            Entity(
                entity_id="e-orphan",
                collection_id=collection_id,
                name="Orphan",
                type="person",
                description="only in d1",
                source_chunk_ids=("c1",),
            ),
        ],
    )
    result = await s.delete_document_rows(collection_id, "d1")
    assert result.chunks_removed >= 1, f"[{name}] no chunks removed"

    remaining = await s.find_entities_by_names(collection_id, ["Orphan"])
    assert len(remaining) == 0, f"[{name}] orphan entity should have been deleted"


@pytest.mark.asyncio
async def test_get_chunks_by_ids(store, collection_id):
    """Chunk rehydration should return the stored text."""
    name, s = store
    graph = _graph_data(collection_id)
    await s.upsert_chunks(collection_id, [graph.chunk_1, graph.chunk_2])
    chunks = await s.get_chunks_by_ids(collection_id, ["c1", "c2"])
    texts = {c.text for c in chunks}
    assert "Alice met Bob at Acme Labs." in texts, f"[{name}] chunk text not found"


# ``test_list_labels`` removed in Wave 6 #40 narrow replacement —
# the legacy ``GraphStore.list_labels`` was retired in favour of
# ``LineageGraphStore.list_entity_labels`` (per architect ruling
# msg=3efdf906). The new method is exercised by
# ``tests/unit_test/indexing/test_lineage_query_protocol.py`` plus the
# real-engine integration suites under
# ``tests/integration/test_{neo4j,nebula}_lineage_graph_store.py``.


@pytest.mark.asyncio
async def test_drop_collection_cleans_everything(store, collection_id):
    """After drop_collection, nothing should remain."""
    name, s = store
    graph = _graph_data(collection_id)
    await s.upsert_chunks(collection_id, [graph.chunk_1])
    await s.upsert_entities(collection_id, [graph.entity_alice])
    await s.drop_collection(collection_id)

    found = await s.find_entities_by_names(collection_id, ["Alice"])
    assert len(found) == 0, f"[{name}] entities survived drop_collection"
