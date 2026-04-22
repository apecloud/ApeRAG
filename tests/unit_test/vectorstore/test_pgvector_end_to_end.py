# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end integration test for the pgvector backend.

Gated on the ``APERAG_TEST_PGVECTOR_URL`` env var. Set it to a reachable
Postgres DSN (with pgvector installable — i.e. superuser *or* with the
extension already created) and these tests exercise the full write →
search → delete path.

Example local run:

.. code-block:: bash

    export APERAG_TEST_PGVECTOR_URL="postgresql://postgres:postgres@127.0.0.1:5432/aperag_test"
    pytest tests/unit_test/vectorstore/test_pgvector_end_to_end.py -v

The tests deliberately use their own uniquely-shaped vector size
(``size=5`` in a custom schema) so they can't interfere with production
data in a shared database.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("pgvector")
pytest.importorskip("sqlalchemy")

PGVECTOR_URL = os.environ.get("APERAG_TEST_PGVECTOR_URL")

pytestmark = pytest.mark.skipif(
    not PGVECTOR_URL,
    reason="set APERAG_TEST_PGVECTOR_URL to run pgvector integration tests",
)

from sqlalchemy import text  # noqa: E402

from aperag.vectorstore import pgvector_connector as pg  # noqa: E402
from aperag.vectorstore.dto import QueryRequest, VectorPoint  # noqa: E402
from aperag.vectorstore.filters import Eq, all_of  # noqa: E402
from aperag.vectorstore.pgvector_connector import (  # noqa: E402
    PgvectorVectorStoreConnector,
)

# Use 5-dim vectors so we never collide with real aperag_vectors_* tables
# that use production vector sizes (1024, 1536, 3072, …).
TEST_VECTOR_SIZE = 5
VEC_A = [1.0, 0.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _clean_slate():
    """Drop the test table + caches before every test so we start fresh.

    The module-level ``_ENSURED_TABLES`` cache only matters *within* a
    process; wiping it along with the DB table keeps the ensure-DDL path
    covered on every run.
    """
    pg._reset_engine_cache()
    engine = pg._get_or_create_engine(PGVECTOR_URL)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS aperag_vectors_{TEST_VECTOR_SIZE}_cosine CASCADE"))
    yield
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS aperag_vectors_{TEST_VECTOR_SIZE}_cosine CASCADE"))
    pg._reset_engine_cache()


def _make_connector(tenant_id: str) -> PgvectorVectorStoreConnector:
    return PgvectorVectorStoreConnector(
        {
            "collection": tenant_id,
            "vector_size": TEST_VECTOR_SIZE,
            "distance": "Cosine",
            "pgvector_database_url": PGVECTOR_URL,
        }
    )


# ---------------------------------------------------------------------------
# ensure_collection idempotency
# ---------------------------------------------------------------------------


def test_ensure_collection_creates_table_and_indexes():
    a = _make_connector("col_a")
    # Table was created + cached.
    assert (a._database_url, a.table_name) in pg._ENSURED_TABLES
    # Second call is a cache hit (no DDL).
    a.ensure_collection()  # no exception, no DDL


# ---------------------------------------------------------------------------
# upsert → search round-trip
# ---------------------------------------------------------------------------


def test_upsert_then_search_returns_own_point():
    a = _make_connector("col_a")
    pid = str(uuid.uuid4())
    a.upsert(
        [
            VectorPoint(
                id=pid,
                vector=VEC_A,
                payload={"text": "hello", "metadata": {"source": "unit"}},
            )
        ]
    )
    hits = a.search(QueryRequest(embedding=VEC_A, top_k=5, score_threshold=-1.0))
    assert len(hits) == 1
    assert hits[0].id == pid
    # Cosine similarity of a vector with itself is ~1.
    assert hits[0].score > 0.99
    assert hits[0].payload.get("text") == "hello"
    assert hits[0].payload.get("collection_id") == "col_a"


# ---------------------------------------------------------------------------
# tenant isolation: two connectors on same shape, different tenants
# ---------------------------------------------------------------------------


def test_search_is_tenant_isolated():
    a = _make_connector("col_aa")
    b = _make_connector("col_bb")
    a.upsert([VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={"indexer": "vector"})])
    b.upsert([VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={"indexer": "vector"})])

    hits_a = a.search(QueryRequest(embedding=VEC_A, top_k=10, score_threshold=-1.0))
    hits_b = b.search(QueryRequest(embedding=VEC_A, top_k=10, score_threshold=-1.0))

    assert {h.payload["collection_id"] for h in hits_a} == {"col_aa"}
    assert {h.payload["collection_id"] for h in hits_b} == {"col_bb"}


# ---------------------------------------------------------------------------
# retrieve: tenant guard blocks foreign-id lookups
# ---------------------------------------------------------------------------


def test_retrieve_drops_foreign_tenant_ids():
    a = _make_connector("col_aa")
    b = _make_connector("col_bb")
    a_id = str(uuid.uuid4())
    b_id = str(uuid.uuid4())
    a.upsert([VectorPoint(id=a_id, vector=VEC_A, payload={})])
    b.upsert([VectorPoint(id=b_id, vector=VEC_B, payload={})])

    # A requests both ids; only its own comes back.
    got = a.retrieve([a_id, b_id])
    assert {p.id for p in got} == {a_id}


# ---------------------------------------------------------------------------
# delete by ids: tenant-guarded
# ---------------------------------------------------------------------------


def test_delete_by_ids_ignores_foreign_ids():
    a = _make_connector("col_aa")
    b = _make_connector("col_bb")
    a_id = str(uuid.uuid4())
    b_id = str(uuid.uuid4())
    a.upsert([VectorPoint(id=a_id, vector=VEC_A, payload={})])
    b.upsert([VectorPoint(id=b_id, vector=VEC_B, payload={})])

    a.delete([a_id, b_id])

    # A's row gone; B's row survives (the tenant guard kept B intact).
    assert a.retrieve([a_id]) == []
    assert len(b.retrieve([b_id])) == 1


# ---------------------------------------------------------------------------
# delete_by_filter: AND-merged with tenant guard
# ---------------------------------------------------------------------------


def test_delete_by_filter_respects_tenant_guard():
    a = _make_connector("col_aa")
    b = _make_connector("col_bb")
    a.upsert([VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={"indexer": "vector"})])
    b.upsert([VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={"indexer": "vector"})])

    a.delete_by_filter(Eq(key="indexer", value="vector"))

    # A's row gone.
    assert a.search(QueryRequest(embedding=VEC_A, top_k=10, score_threshold=-1.0)) == []
    # B's row survives.
    assert len(b.search(QueryRequest(embedding=VEC_A, top_k=10, score_threshold=-1.0))) == 1


def test_delete_by_filter_empty_is_rejected():
    a = _make_connector("col_aa")
    with pytest.raises(ValueError, match="non-empty filter"):
        a.delete_by_filter(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# search with DSL filter
# ---------------------------------------------------------------------------


def test_search_applies_dsl_filter():
    a = _make_connector("col_aa")
    a.upsert(
        [
            VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={"indexer": "vector"}),
            VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={"indexer": "vision"}),
        ]
    )
    hits = a.search(
        QueryRequest(
            embedding=VEC_A,
            top_k=10,
            flt=Eq(key="indexer", value="vision"),
            score_threshold=-1.0,
        )
    )
    assert len(hits) == 1
    assert hits[0].payload["indexer"] == "vision"


# ---------------------------------------------------------------------------
# drop_tenant: single shard
# ---------------------------------------------------------------------------


def test_drop_tenant_leaves_other_tenants_intact():
    a = _make_connector("col_aa")
    b = _make_connector("col_bb")
    a.upsert([VectorPoint(id=str(uuid.uuid4()), vector=VEC_A, payload={})])
    b.upsert([VectorPoint(id=str(uuid.uuid4()), vector=VEC_B, payload={})])

    a.drop_tenant()

    # A has nothing; B still has its row.
    assert a.search(QueryRequest(embedding=VEC_A, top_k=10, score_threshold=-1.0)) == []
    assert len(b.search(QueryRequest(embedding=VEC_B, top_k=10, score_threshold=-1.0))) == 1


# ---------------------------------------------------------------------------
# composed filter matches Qdrant semantics: all_of(Eq, Eq)
# ---------------------------------------------------------------------------


def test_composed_all_of_filter_matches_both_predicates():
    a = _make_connector("col_aa")
    a.upsert(
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=VEC_A,
                payload={"indexer": "vector", "chat_id": "c1"},
            ),
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=VEC_A,
                payload={"indexer": "vector", "chat_id": "c2"},
            ),
        ]
    )
    hits = a.search(
        QueryRequest(
            embedding=VEC_A,
            top_k=10,
            flt=all_of(Eq(key="indexer", value="vector"), Eq(key="chat_id", value="c1")),
            score_threshold=-1.0,
        )
    )
    assert len(hits) == 1
    assert hits[0].payload["chat_id"] == "c1"
