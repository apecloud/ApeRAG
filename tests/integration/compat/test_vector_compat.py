# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cross-backend VectorStore compatibility tests.

Run the same deterministic scenario against Qdrant and pgvector. Each
backend is gated on an env var.

Env vars:
    COMPAT_QDRANT_URL   = http://127.0.0.1:6333
    COMPAT_PGVECTOR_URL = postgresql://postgres:postgres@127.0.0.1:5432/postgres

Usage:
    COMPAT_QDRANT_URL=... COMPAT_PGVECTOR_URL=... pytest tests/integration/compat/test_vector_compat.py -v
    make test-compat-vector
"""

from __future__ import annotations

import os
import uuid

import pytest

COLLECTION = f"compat_vec_{uuid.uuid4().hex[:8]}"
VECTOR_SIZE = 8


def _make_qdrant_connector():
    url = os.environ.get("COMPAT_QDRANT_URL")
    if not url:
        return None
    from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

    ctx = {
        "url": url,
        "collection": COLLECTION,
        "vector_size": VECTOR_SIZE,
        "distance": "Cosine",
        "multitenant": True,
    }
    adaptor = VectorStoreConnectorAdaptor("qdrant", ctx=ctx)
    adaptor.connector.ensure_collection()
    return adaptor.connector


def _make_pgvector_connector():
    url = os.environ.get("COMPAT_PGVECTOR_URL")
    if not url:
        return None
    from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

    ctx = {
        "pgvector_database_url": url,
        "collection": COLLECTION,
        "vector_size": VECTOR_SIZE,
        "distance": "Cosine",
        "multitenant": True,
        "pgvector_hnsw_m": 16,
        "pgvector_hnsw_ef_construction": 64,
        "pgvector_hnsw_ef_search": 40,
    }
    adaptor = VectorStoreConnectorAdaptor("pgvector", ctx=ctx)
    adaptor.connector.ensure_collection()
    return adaptor.connector


_BACKEND_FACTORIES = {
    "qdrant": _make_qdrant_connector,
    "pgvector": _make_pgvector_connector,
}


def _available_backends():
    available = []
    for name, factory in _BACKEND_FACTORIES.items():
        try:
            conn = factory()
            if conn is not None:
                available.append(pytest.param((name, conn), id=name))
        except Exception:
            pass
    return available


_available = _available_backends()

if not _available:
    pytestmark = pytest.mark.skip(reason="No vector backend env vars set (COMPAT_QDRANT_URL / COMPAT_PGVECTOR_URL)")


@pytest.fixture(params=_available if _available else [pytest.param(None, marks=pytest.mark.skip)])
def connector(request):
    name, conn = request.param
    yield name, conn
    try:
        conn.drop_tenant()
    except Exception:
        pass


# --- test data ---


def _point(id_: str, vec_seed: float, **payload):
    from aperag.vectorstore.dto import VectorPoint

    return VectorPoint(id=id_, vector=[vec_seed] * VECTOR_SIZE, payload=payload)


# --- tests ---


def test_upsert_and_search(connector):
    """Basic upsert + similarity search round-trip."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest

    points = [
        _point("p1", 0.1, indexer="vector", text="hello"),
        _point("p2", 0.9, indexer="vector", text="world"),
    ]
    conn.upsert(points)

    hits = conn.search(
        QueryRequest(
            embedding=[0.1] * VECTOR_SIZE,
            top_k=2,
            score_threshold=0.0,
        )
    )
    assert len(hits) >= 1, f"[{name}] no search results"
    ids = {h.id for h in hits}
    assert "p1" in ids or "p2" in ids, f"[{name}] expected at least one hit"


def test_upsert_graph_entity_and_filter(connector):
    """Graph entity shadow vectors should be searchable with indexer filter."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest
    from aperag.vectorstore.filters import Eq

    points = [
        _point("ge_e1", 0.5, indexer="graph_entity", entity_name="Alice", collection_id="test"),
        _point("p_chunk", 0.5, indexer="vector", text="chunk text"),
    ]
    conn.upsert(points)

    hits = conn.search(
        QueryRequest(
            embedding=[0.5] * VECTOR_SIZE,
            top_k=10,
            flt=Eq("indexer", "graph_entity"),
            score_threshold=0.0,
        )
    )
    ids = {h.id for h in hits}
    assert "ge_e1" in ids, f"[{name}] graph entity not found via filter"
    assert "p_chunk" not in ids, f"[{name}] chunk vector should be excluded by filter"


def test_delete_by_filter(connector):
    """delete_by_filter should remove only matching vectors."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest
    from aperag.vectorstore.filters import And, Eq

    points = [
        _point("ge_del1", 0.3, indexer="graph_entity", collection_id="col1"),
        _point("ge_del2", 0.3, indexer="graph_entity", collection_id="col2"),
        _point("p_keep", 0.3, indexer="vector", collection_id="col1"),
    ]
    conn.upsert(points)

    conn.delete_by_filter(And(parts=(Eq("indexer", "graph_entity"), Eq("collection_id", "col1"))))

    hits = conn.search(
        QueryRequest(
            embedding=[0.3] * VECTOR_SIZE,
            top_k=10,
            score_threshold=0.0,
        )
    )
    ids = {h.id for h in hits}
    assert "ge_del1" not in ids, f"[{name}] deleted vector still found"
    assert "p_keep" in ids, f"[{name}] non-matching vector was deleted"


def test_retrieve_by_ids(connector):
    """retrieve() should return vectors by id."""
    name, conn = connector
    points = [
        _point("ret1", 0.2, indexer="vector", text="a"),
        _point("ret2", 0.8, indexer="vector", text="b"),
    ]
    conn.upsert(points)

    results = conn.retrieve(["ret1", "ret2"])
    ids = {r.id for r in results}
    assert "ret1" in ids, f"[{name}] ret1 not retrieved"
    assert "ret2" in ids, f"[{name}] ret2 not retrieved"
