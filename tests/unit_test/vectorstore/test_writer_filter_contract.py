# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration tests that pin the contract between the real writer path
(``TextNode -> nodes_to_vector_points -> connector.upsert``) and the
filter DSL used by search / delete.

This exists because the prior regression coverage seeded payloads by
hand as ``{"indexer": ...}`` flat dicts, which silently masked a
writer/filter split: real adapter output puts filterable keys under
``metadata.*``. If the adapter ever stops mirroring those keys at the
top level, ``Eq("indexer", ...)`` / ``Eq("chat_id", ...)`` on new writes
will silently return zero rows, and the only thing that will notice is
production.

We intentionally run against the Qdrant ``:memory:`` backend: it's the
only connector that exercises real translator + real storage in-process
with no external dependency.
"""

from __future__ import annotations

from typing import List

import pytest
from llama_index.core.schema import TextNode

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as rest  # noqa: E402

from aperag.vectorstore import qdrant_connector as qc  # noqa: E402
from aperag.vectorstore.dto import (  # noqa: E402
    QueryRequest,
    TenantRef,
    VectorShape,
)
from aperag.vectorstore.filters import Eq, In, IsEmpty, all_of, any_of  # noqa: E402  # noqa: E402
from aperag.vectorstore.llama_index_adapter import nodes_to_vector_points  # noqa: E402
from aperag.vectorstore.qdrant_connector import (  # noqa: E402
    QdrantVectorStoreConnector,
    global_collection_name,
)

VECTOR_SIZE = 4
VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0]
VEC_C = [0.0, 0.0, 1.0, 0.0]


@pytest.fixture(autouse=True)
def _reset_ensured_cache():
    qc._ENSURED_COLLECTIONS.clear()
    yield
    qc._ENSURED_COLLECTIONS.clear()


@pytest.fixture
def shared_client():
    return QdrantClient(":memory:")


def _make_connector(tenant_id: str, client: QdrantClient) -> QdrantVectorStoreConnector:
    """Bind a connector to a shared in-memory client. Mirrors the helper in
    ``test_qdrant_multitenancy_integration.py`` but intentionally duplicated
    so a future refactor doesn't accidentally couple these test files."""
    conn = QdrantVectorStoreConnector.__new__(QdrantVectorStoreConnector)
    ctx = {
        "url": ":memory:",
        "collection": tenant_id,
        "vector_size": VECTOR_SIZE,
        "distance": "Cosine",
        "multitenant": True,
        "quantization_enabled": False,
        "hnsw_on_disk": False,
    }
    conn.ctx = ctx
    conn.cfg = ctx
    conn.multitenant = True
    conn._tenant = TenantRef(id=tenant_id)
    conn._shape = VectorShape(size=VECTOR_SIZE, distance="cosine")
    conn.url = ":memory:"
    conn.port = 6333
    conn.grpc_port = 6334
    conn.prefer_grpc = False
    conn.https = False
    conn.timeout = 300
    conn.client = client
    conn.collection_name = global_collection_name(VECTOR_SIZE, "cosine")
    conn.ensure_collection()
    return conn


def _node(text: str, *, embedding: List[float], metadata: dict) -> TextNode:
    n = TextNode(text=text, metadata=metadata)
    n.embedding = embedding
    return n


def _seed_via_real_writer(conn: QdrantVectorStoreConnector, nodes: List[TextNode]) -> List[str]:
    points = nodes_to_vector_points(nodes, tenant_id=conn.tenant.id)
    return conn.upsert(points)


# ---------------------------------------------------------------------------
# query filter: Eq / In / IsEmpty contract survives real writer payloads
# ---------------------------------------------------------------------------


def test_eq_indexer_filter_matches_real_writer_payload(shared_client):
    """The regression that motivated this file: ``Eq("indexer", "vision")``
    must only match points the writer actually stamped as vision. If the
    adapter stops lifting ``indexer`` to top level, this test goes red."""
    conn = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    _seed_via_real_writer(
        conn,
        [
            _node("vec1", embedding=VEC_A, metadata={"indexer": "vector"}),
            _node("vec2", embedding=VEC_B, metadata={"indexer": "vector"}),
            _node("vis1", embedding=VEC_A, metadata={"indexer": "vision"}),
            _node("sum1", embedding=VEC_A, metadata={"indexer": "summary"}),
        ],
    )

    hits = conn.search(
        QueryRequest(
            embedding=VEC_A,
            top_k=10,
            flt=Eq(key="indexer", value="vision"),
            score_threshold=-1.0,
        )
    )
    assert len(hits) == 1
    assert hits[0].payload.get("indexer") == "vision"


def test_index_types_filter_does_not_leak_across_indexers(shared_client):
    """``index_types=["vector"]`` is implemented as
    ``In(indexer, ["vector"]) OR IsEmpty(indexer)`` — the ``IsEmpty`` half
    exists to keep pre-migration data visible. If new writes leave the
    top-level ``indexer`` empty (because it only lives in
    ``metadata.indexer``), IsEmpty matches every new point, and
    ``index_types=["vector"]`` starts bleeding summary/vision results in.
    This test is the direct regression guard for that.
    """
    conn = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    _seed_via_real_writer(
        conn,
        [
            _node("v1", embedding=VEC_A, metadata={"indexer": "vector"}),
            _node("s1", embedding=VEC_A, metadata={"indexer": "summary"}),
            _node("vi1", embedding=VEC_A, metadata={"indexer": "vision"}),
        ],
    )

    hits = conn.search(
        QueryRequest(
            embedding=VEC_A,
            top_k=10,
            flt=any_of(
                In(key="indexer", values=("vector",)),
                IsEmpty(key="indexer"),
            ),
            score_threshold=-1.0,
        )
    )

    indexers = {h.payload.get("indexer") for h in hits}
    assert indexers == {"vector"}, (
        f"index_types=['vector'] leaked across indexers: {indexers}. "
        "Writer probably stopped lifting `indexer` to top-level payload."
    )


def test_chat_id_eq_filter_matches_real_writer_payload(shared_client):
    """``chat_id`` is the other filterable key whose contract lives in
    ``ContextManager._create_combined_filter``. Same risk: if it only sits
    under ``metadata.chat_id``, every chat-scoped retrieval silently
    returns nothing."""
    conn = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    _seed_via_real_writer(
        conn,
        [
            _node(
                "n1",
                embedding=VEC_A,
                metadata={"indexer": "vector", "chat_id": "chat_111"},
            ),
            _node(
                "n2",
                embedding=VEC_A,
                metadata={"indexer": "vector", "chat_id": "chat_222"},
            ),
        ],
    )

    hits = conn.search(
        QueryRequest(
            embedding=VEC_A,
            top_k=10,
            flt=all_of(
                Eq(key="indexer", value="vector"),
                Eq(key="chat_id", value="chat_111"),
            ),
            score_threshold=-1.0,
        )
    )
    assert len(hits) == 1
    assert hits[0].payload.get("chat_id") == "chat_111"


# ---------------------------------------------------------------------------
# delete_by_filter must hit real writer payloads
# ---------------------------------------------------------------------------


def test_delete_by_filter_on_real_writer_payload_removes_only_matches(shared_client):
    """The original review blocker explicitly called out that
    ``delete_by_filter(Eq("indexer", ...))`` was only validated against
    hand-seeded flat payloads. This is the direct regression guard:
    write via the real adapter, then ensure the delete hits the right
    points and leaves the rest alone."""
    conn = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)

    _seed_via_real_writer(
        conn,
        [
            _node("v1", embedding=VEC_A, metadata={"indexer": "vector"}),
            _node("v2", embedding=VEC_B, metadata={"indexer": "vector"}),
            _node("vi1", embedding=VEC_C, metadata={"indexer": "vision"}),
        ],
    )

    # All three should be visible before the delete.
    tenant_filter = rest.Filter(
        must=[rest.FieldCondition(key="collection_id", match=rest.MatchValue(value=conn.tenant.id))]
    )
    before, _ = conn.client.scroll(collection_name=conn.collection_name, scroll_filter=tenant_filter, limit=100)
    assert len(before) == 3

    conn.delete_by_filter(Eq(key="indexer", value="vector"))

    remaining, _ = conn.client.scroll(collection_name=conn.collection_name, scroll_filter=tenant_filter, limit=100)
    remaining_indexers = [p.payload.get("indexer") for p in remaining]
    assert remaining_indexers == ["vision"], (
        f"delete_by_filter(Eq indexer=vector) did not match real writer payload: "
        f"remaining={remaining_indexers}. Writer probably stopped lifting "
        f"`indexer` to top-level payload."
    )
