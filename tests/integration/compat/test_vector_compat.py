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


def _available_backend_names():
    available = []
    if os.environ.get("COMPAT_QDRANT_URL"):
        available.append(pytest.param("qdrant", id="qdrant"))
    if os.environ.get("COMPAT_PGVECTOR_URL"):
        available.append(pytest.param("pgvector", id="pgvector"))
    return available


_available = _available_backend_names()

if not _available:
    pytestmark = pytest.mark.skip(reason="No vector backend env vars set (COMPAT_QDRANT_URL / COMPAT_PGVECTOR_URL)")


@pytest.fixture(params=_available if _available else [pytest.param(None, marks=pytest.mark.skip)])
def connector(request):
    name = request.param
    conn = _BACKEND_FACTORIES[name]()
    assert conn is not None
    yield name, conn
    try:
        conn.drop_tenant()
    except Exception:
        pass


# --- test data ---


def _point(id_: str, vec_seed: float, **payload):
    from aperag.vectorstore.dto import VectorPoint

    return VectorPoint(id=_point_id(id_), vector=[vec_seed] * VECTOR_SIZE, payload=payload)


def _point_id(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"compat-vector:{seed}"))


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
    assert _point_id("p1") in ids or _point_id("p2") in ids, f"[{name}] expected at least one hit"


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
    assert _point_id("ge_e1") in ids, f"[{name}] graph entity not found via filter"
    assert _point_id("p_chunk") not in ids, f"[{name}] chunk vector should be excluded by filter"


def test_delete_by_filter(connector):
    """delete_by_filter should remove only matching vectors."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest
    from aperag.vectorstore.filters import And, Eq

    points = [
        _point("ge_del1", 0.3, indexer="graph_entity", shadow_scope="col1"),
        _point("ge_del2", 0.3, indexer="graph_entity", shadow_scope="col2"),
        _point("p_keep", 0.3, indexer="vector", shadow_scope="col1"),
    ]
    conn.upsert(points)

    conn.delete_by_filter(And(parts=(Eq("indexer", "graph_entity"), Eq("shadow_scope", "col1"))))

    hits = conn.search(
        QueryRequest(
            embedding=[0.3] * VECTOR_SIZE,
            top_k=10,
            score_threshold=0.0,
        )
    )
    ids = {h.id for h in hits}
    assert _point_id("ge_del1") not in ids, f"[{name}] deleted vector still found"
    assert _point_id("p_keep") in ids, f"[{name}] non-matching vector was deleted"


def test_retrieve_by_ids(connector):
    """retrieve() should return vectors by id."""
    name, conn = connector
    points = [
        _point("ret1", 0.2, indexer="vector", text="a"),
        _point("ret2", 0.8, indexer="vector", text="b"),
    ]
    conn.upsert(points)

    results = conn.retrieve([_point_id("ret1"), _point_id("ret2")])
    ids = {r.id for r in results}
    assert _point_id("ret1") in ids, f"[{name}] ret1 not retrieved"
    assert _point_id("ret2") in ids, f"[{name}] ret2 not retrieved"


# ---------------------------------------------------------------------------
# task #61 P0-A: filter fail-loud (cross-backend invariant)
# ---------------------------------------------------------------------------


def test_search_unsupported_filter_raises_unsupported_filter_error(connector):
    """Both backends must raise the cross-adapter ``UnsupportedFilterError``
    when the caller hands a filter shape neither translator understands.
    Silent fall-through to an unfiltered query is a correctness footgun
    (task #61 P0-A)."""
    name, conn = connector
    from aperag.vectorstore.base import UnsupportedFilterError
    from aperag.vectorstore.dto import QueryRequest

    class _BogusNode:
        pass

    with pytest.raises(UnsupportedFilterError):
        conn.search(
            QueryRequest(
                embedding=[0.1] * VECTOR_SIZE,
                top_k=5,
                flt=_BogusNode(),  # type: ignore[arg-type]
                score_threshold=0.0,
            )
        )

    # Subclass of TypeError, so existing ``except TypeError`` callers
    # keep working.
    with pytest.raises(TypeError):
        conn.search(
            QueryRequest(
                embedding=[0.1] * VECTOR_SIZE,
                top_k=5,
                flt=_BogusNode(),  # type: ignore[arg-type]
                score_threshold=0.0,
            )
        )


# ---------------------------------------------------------------------------
# task #61 P0-B: score normalization (cross-backend invariant on cosine)
# ---------------------------------------------------------------------------


def test_search_score_in_unit_interval(connector):
    """Per the §5 normalized-score contract, every ``SearchHit.score`` must
    lie in ``[0, 1]`` regardless of the underlying distance metric. The
    SearchHit DTO validator already enforces this, but we keep the
    assertion here so a regression that bypassed the DTO (e.g. a future
    direct-build code path) would surface in compat-test rather than
    quietly drifting."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest

    points = [
        _point("sc1", 0.1, indexer="vector"),
        _point("sc2", 0.9, indexer="vector"),
    ]
    conn.upsert(points)

    hits = conn.search(QueryRequest(embedding=[0.1] * VECTOR_SIZE, top_k=5, score_threshold=0.0))
    assert hits, f"[{name}] expected non-empty hits"
    for h in hits:
        assert 0.0 <= h.score <= 1.0, f"[{name}] score {h.score} outside [0, 1]"


def test_search_score_threshold_filters_low_scores(connector):
    """``score_threshold`` is on the same normalized [0, 1] scale; raising
    it must monotonically reduce or preserve the result count, never
    expand it. Pinned cross-backend so PGVector / Qdrant agree on the
    direction of the threshold."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest

    points = [
        _point("th_close", 0.10, indexer="vector"),
        _point("th_far", 0.90, indexer="vector"),
    ]
    conn.upsert(points)

    lo = conn.search(QueryRequest(embedding=[0.10] * VECTOR_SIZE, top_k=10, score_threshold=0.0))
    hi = conn.search(QueryRequest(embedding=[0.10] * VECTOR_SIZE, top_k=10, score_threshold=0.99))
    # Higher threshold must shrink (or preserve) the result set.
    assert len(hi) <= len(lo), f"[{name}] threshold direction inverted"
    # Every survivor of the high-threshold filter must clear the cutoff.
    for h in hi:
        assert h.score >= 0.99, f"[{name}] threshold not enforced (score={h.score})"


def test_search_top_k_ranking_higher_is_better(connector):
    """Ranking is monotone in normalized score: the top hit is the most
    similar, scores decrease (or stay equal) with rank. Cross-backend
    pinned because score-direction confusion was the root cause of
    cuiwenbo's FE display regression (task #70)."""
    name, conn = connector
    from aperag.vectorstore.dto import QueryRequest

    points = [_point(f"r{i}", 0.1 * i, indexer="vector") for i in range(1, 6)]
    conn.upsert(points)

    hits = conn.search(QueryRequest(embedding=[0.10] * VECTOR_SIZE, top_k=5, score_threshold=0.0))
    assert len(hits) >= 2, f"[{name}] need at least 2 hits to assert ordering"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), (
        f"[{name}] top-k ordering must be descending in normalized score, got {scores}"
    )
