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

import pytest

from aperag.graphindex.dto import Chunk, Entity, Relation

COLLECTION_ID = f"compat_test_{uuid.uuid4().hex[:8]}"

# --- fixtures: one per backend -------------------------------------------


def _make_pg_store():
    url = os.environ.get("COMPAT_PG_URL")
    if not url:
        return None
    from sqlalchemy.ext.asyncio import create_async_engine

    from aperag.graphindex.storage.postgres import PostgresGraphStore

    engine = create_async_engine(url, future=True)
    return PostgresGraphStore(engine=engine), engine


def _make_neo4j_store():
    uri = os.environ.get("COMPAT_NEO4J_URI")
    if not uri:
        return None
    from aperag.graphindex.storage.neo4j import Neo4jGraphStore

    return Neo4jGraphStore(
        uri=uri,
        username=os.environ.get("COMPAT_NEO4J_USER", "neo4j"),
        password=os.environ.get("COMPAT_NEO4J_PASS", "password"),
    ), None


def _make_nebula_store():
    hosts = os.environ.get("COMPAT_NEBULA_HOSTS")
    if not hosts:
        return None
    from aperag.graphindex.storage.nebula import NebulaGraphStore

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


def _available_backends():
    available = []
    for name, factory in _BACKENDS.items():
        result = factory()
        if result is not None:
            available.append(pytest.param((name, result[0], result[1]), id=name))
    return available


_available = _available_backends()

if not _available:
    pytestmark = pytest.mark.skip(
        reason="No graph backend env vars set (COMPAT_PG_URL / COMPAT_NEO4J_URI / COMPAT_NEBULA_HOSTS)"
    )


@pytest.fixture(params=_available if _available else [pytest.param(None, marks=pytest.mark.skip)])
async def store(request):
    """Yield a (backend_name, store) tuple. Clean up after test."""
    name, store_obj, engine = request.param

    # For PG: ensure tables exist
    if name == "postgresql" and engine is not None:
        from aperag.db.models import Base
        from aperag.graphindex.models import CHUNKS_TABLE, EDGES_TABLE, NODES_TABLE

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
        await store_obj.drop_collection(COLLECTION_ID)
    except Exception:
        pass
    if name == "postgresql" and engine is not None:
        await engine.dispose()


# --- deterministic test data ---------------------------------------------

CHUNK_1 = Chunk(
    chunk_id="c1", doc_id="d1", collection_id=COLLECTION_ID, order_in_doc=0, text="Alice met Bob at Acme Labs."
)
CHUNK_2 = Chunk(
    chunk_id="c2", doc_id="d1", collection_id=COLLECTION_ID, order_in_doc=1, text="Bob works at Acme Labs on AI."
)

ENTITY_ALICE = Entity(
    entity_id="e-alice",
    collection_id=COLLECTION_ID,
    name="Alice",
    type="person",
    description="A researcher",
    source_chunk_ids=("c1",),
)
ENTITY_BOB = Entity(
    entity_id="e-bob",
    collection_id=COLLECTION_ID,
    name="Bob",
    type="person",
    description="An engineer",
    source_chunk_ids=("c1", "c2"),
)
ENTITY_ACME = Entity(
    entity_id="e-acme",
    collection_id=COLLECTION_ID,
    name="Acme Labs",
    type="organization",
    description="A research lab",
    source_chunk_ids=("c1", "c2"),
)

REL_ALICE_BOB = Relation(
    collection_id=COLLECTION_ID,
    source_id="e-alice",
    target_id="e-bob",
    description="Alice met Bob",
    weight=7.0,
    source_chunk_ids=("c1",),
)
REL_BOB_ACME = Relation(
    collection_id=COLLECTION_ID,
    source_id="e-bob",
    target_id="e-acme",
    description="Bob works at Acme",
    weight=8.0,
    source_chunk_ids=("c1", "c2"),
)


# --- the tests -----------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_and_find_entities(store):
    """Write entities, read them back by name — verify fields round-trip."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1, CHUNK_2])
    await s.upsert_entities(COLLECTION_ID, [ENTITY_ALICE, ENTITY_BOB, ENTITY_ACME])

    found = await s.find_entities_by_names(COLLECTION_ID, ["Alice", "Bob"])
    names = {e.name for e in found}
    assert "Alice" in names, f"[{name}] Alice not found"
    assert "Bob" in names, f"[{name}] Bob not found"

    found_by_id = await s.find_entities_by_ids(COLLECTION_ID, ["e-alice", "e-acme"])
    ids = {e.entity_id for e in found_by_id}
    assert "e-alice" in ids, f"[{name}] find_entities_by_ids failed for e-alice"
    assert "e-acme" in ids, f"[{name}] find_entities_by_ids failed for e-acme"


@pytest.mark.asyncio
async def test_upsert_relations_and_expand(store):
    """Write relations, BFS-expand from an anchor."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1, CHUNK_2])
    await s.upsert_entities(COLLECTION_ID, [ENTITY_ALICE, ENTITY_BOB, ENTITY_ACME])
    await s.upsert_relations(COLLECTION_ID, [REL_ALICE_BOB, REL_BOB_ACME])

    entities, relations = await s.expand_neighborhood(COLLECTION_ID, ["e-alice"], max_hop=2, limit=50)
    entity_ids = {e.entity_id for e in entities}
    assert "e-alice" in entity_ids, f"[{name}] anchor not in expansion"
    assert "e-bob" in entity_ids, f"[{name}] 1-hop neighbor not found"
    assert len(relations) >= 1, f"[{name}] no relations in expansion"


@pytest.mark.asyncio
async def test_description_accumulation(store):
    """Multiple upserts for the same entity should accumulate descriptions."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1])
    await s.upsert_entities(COLLECTION_ID, [ENTITY_ALICE])
    # Second upsert with different description
    updated = Entity(
        entity_id="e-alice",
        collection_id=COLLECTION_ID,
        name="Alice",
        type="person",
        description="Also a teacher",
        source_chunk_ids=("c2",),
    )
    await s.upsert_entities(COLLECTION_ID, [updated])

    found = await s.find_entities_by_names(COLLECTION_ID, ["Alice"])
    assert len(found) == 1, f"[{name}] expected 1 entity, got {len(found)}"
    assert "researcher" in found[0].description.lower() or "teacher" in found[0].description.lower(), (
        f"[{name}] description should contain at least one fragment"
    )


@pytest.mark.asyncio
async def test_delete_document_removes_orphans(store):
    """Deleting a document should remove orphan entities/relations."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1])
    await s.upsert_entities(
        COLLECTION_ID,
        [
            Entity(
                entity_id="e-orphan",
                collection_id=COLLECTION_ID,
                name="Orphan",
                type="person",
                description="only in d1",
                source_chunk_ids=("c1",),
            ),
        ],
    )
    result = await s.delete_document_rows(COLLECTION_ID, "d1")
    assert result.chunks_removed >= 1, f"[{name}] no chunks removed"

    remaining = await s.find_entities_by_names(COLLECTION_ID, ["Orphan"])
    assert len(remaining) == 0, f"[{name}] orphan entity should have been deleted"


@pytest.mark.asyncio
async def test_get_chunks_by_ids(store):
    """Chunk rehydration should return the stored text."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1, CHUNK_2])
    chunks = await s.get_chunks_by_ids(COLLECTION_ID, ["c1", "c2"])
    texts = {c.text for c in chunks}
    assert "Alice met Bob at Acme Labs." in texts, f"[{name}] chunk text not found"


@pytest.mark.asyncio
async def test_list_labels(store):
    """list_labels should return distinct entity types."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1])
    await s.upsert_entities(COLLECTION_ID, [ENTITY_ALICE, ENTITY_ACME])
    labels = await s.list_labels(COLLECTION_ID)
    assert "person" in labels, f"[{name}] 'person' label missing"
    assert "organization" in labels, f"[{name}] 'organization' label missing"


@pytest.mark.asyncio
async def test_drop_collection_cleans_everything(store):
    """After drop_collection, nothing should remain."""
    name, s = store
    await s.upsert_chunks(COLLECTION_ID, [CHUNK_1])
    await s.upsert_entities(COLLECTION_ID, [ENTITY_ALICE])
    await s.drop_collection(COLLECTION_ID)

    found = await s.find_entities_by_names(COLLECTION_ID, ["Alice"])
    assert len(found) == 0, f"[{name}] entities survived drop_collection"
