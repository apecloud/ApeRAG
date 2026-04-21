# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration tests for the multitenant Qdrant connector using ``:memory:``.

The reviewer's most important concern was "are two tenants actually isolated?"
— a property that can only be validated end-to-end against a running Qdrant.
The ``qdrant_client.QdrantClient(":memory:")`` local mode gives us a real
point-store implementation (with scroll / upsert / filter semantics) without
requiring a server process, so these tests execute in <1 s each.

Note: payload indexes are a no-op in local mode (the client prints a warning
to that effect). That's fine — we're exercising correctness of the
filter-based isolation, not the segment-level defragmentation benefit.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as rest  # noqa: E402

from aperag.query.query import QueryWithEmbedding  # noqa: E402
from aperag.vectorstore import qdrant_connector as qc  # noqa: E402
from aperag.vectorstore.qdrant_connector import (  # noqa: E402
    TENANT_PAYLOAD_KEY,
    QdrantVectorStoreConnector,
    global_collection_name,
)

VECTOR_SIZE = 4
VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _reset_ensured_cache():
    """Clear the module-level "already ensured" cache between tests.

    The cache is a process-level optimization that would normally skip the
    create_collection RPC for (url, collection_name) tuples we've seen
    before. In tests we spin up a fresh in-memory QdrantClient per test —
    without a reset, the second test's connector would skip ensure and then
    talk to a client that doesn't have the collection, blowing up at the
    first upsert.
    """
    qc._ENSURED_COLLECTIONS.clear()
    yield
    qc._ENSURED_COLLECTIONS.clear()


@pytest.fixture
def shared_client():
    """One in-memory Qdrant client shared by all connectors in a test.

    In production each connector instance talks to the same remote Qdrant,
    so they see each other's writes. In tests we must mimic that by having
    every connector reuse the same client — otherwise ``tenant A`` and
    ``tenant B`` would each get their own isolated :memory: store and the
    whole "are they isolated?" question becomes trivially true for the
    wrong reason.
    """
    return QdrantClient(":memory:")


def _make_connector(
    tenant_id: str,
    client: QdrantClient,
    multitenant: bool = True,
) -> QdrantVectorStoreConnector:
    """Spin up a connector bound to a caller-provided client.

    We patch ``_ENSURED_COLLECTIONS`` via fixture (so ensure runs), and we
    monkey-patch the client reference right after construction — because the
    connector's ``__init__`` builds its own client from the ctx; we want all
    connectors in a test to share one in-memory store.
    """
    ctx = {
        "url": ":memory:",
        "collection": tenant_id,
        "vector_size": VECTOR_SIZE,
        "distance": "Cosine",
        "multitenant": multitenant,
        # Quantization is meaningless for 4-dim local mode; disable to keep
        # create_collection valid in the in-memory backend (which doesn't
        # implement all quantization variants).
        "quantization_enabled": False,
        "hnsw_on_disk": False,
    }
    # Strategy: pre-populate the shared client by running ensure_collection
    # logic through ONE connector, then force subsequent connectors to reuse
    # the same client.
    conn = QdrantVectorStoreConnector.__new__(QdrantVectorStoreConnector)
    conn.ctx = ctx
    conn.multitenant = multitenant
    conn.cfg = ctx
    if multitenant and not ctx.get("collection"):
        raise ValueError("QdrantVectorStoreConnector(multitenant=True) requires ctx['collection']")
    conn.tenant_id = str(tenant_id)
    conn.url = ":memory:"
    conn.port = 6333
    conn.grpc_port = 6334
    conn.prefer_grpc = False
    conn.https = False
    conn.timeout = 300
    conn.vector_size = VECTOR_SIZE
    conn.distance = "Cosine"
    conn.client = client

    if multitenant:
        conn.collection_name = global_collection_name(VECTOR_SIZE, "Cosine")
        conn._ensure_collection()
    else:
        conn.collection_name = tenant_id
        # Legacy mode: caller explicitly creates the physical collection.
        if not client.collection_exists(conn.collection_name):
            client.create_collection(
                collection_name=conn.collection_name,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
            )

    # llama_index store — not used in these tests, but the connector API
    # expects the attribute to be set. Importing here avoids the cost when
    # tests don't need it.
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    conn.store = QdrantVectorStore(
        client=client,
        collection_name=conn.collection_name,
        vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
    )
    return conn


def _seed_points(
    conn: QdrantVectorStoreConnector,
    vectors: List[List[float]],
    tenant_payload: str,
) -> List[str]:
    """Write raw points straight into the connector's physical collection.

    We deliberately bypass ``store.add()`` because llama_index serializes the
    entire node into ``_node_content`` and adds other derived fields, which
    muddies tests focused on tenant-payload filtering. Direct upsert with
    ``rest.PointStruct`` mirrors what the migration script does.
    """
    ids = [str(uuid.uuid4()) for _ in vectors]
    points = [
        rest.PointStruct(id=pid, vector=vec, payload={TENANT_PAYLOAD_KEY: tenant_payload})
        for pid, vec in zip(ids, vectors)
    ]
    conn.client.upsert(collection_name=conn.collection_name, points=points, wait=True)
    return ids


# ---------------------------------------------------------------------------
# tenant isolation on search
# ---------------------------------------------------------------------------


def test_multitenant_search_is_isolated_between_tenants(shared_client):
    """Two tenants writing to the same physical collection must not see each
    other's points through the connector's search path, because the connector
    silently adds a ``collection_id`` ``must`` clause."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    assert a.collection_name == b.collection_name == global_collection_name(VECTOR_SIZE, "Cosine"), (
        "Both tenants must be routed to the same physical global collection"
    )

    _seed_points(a, [VEC_A, VEC_A], tenant_payload=a.tenant_id)
    _seed_points(b, [VEC_A, VEC_B], tenant_payload=b.tenant_id)

    # Tenant A queries with VEC_A: should hit only its own 2 points.
    q = QueryWithEmbedding(query="", top_k=10, embedding=VEC_A)
    res_a = a.search(q, score_threshold=0.0)
    res_b = b.search(q, score_threshold=0.0)

    tenants_seen_by_a = {r.metadata.get(TENANT_PAYLOAD_KEY) for r in res_a.results if r.metadata}
    tenants_seen_by_b = {r.metadata.get(TENANT_PAYLOAD_KEY) for r in res_b.results if r.metadata}

    assert tenants_seen_by_a == {a.tenant_id}, f"tenant A saw foreign points: {tenants_seen_by_a}"
    assert tenants_seen_by_b == {b.tenant_id}, f"tenant B saw foreign points: {tenants_seen_by_b}"
    assert len(res_a.results) == 2
    # B has one VEC_A and one VEC_B; VEC_B's cosine similarity to the query VEC_A is 0 so
    # it may be filtered out by score_threshold=0 depending on implementation, but both
    # points belong to B regardless.
    assert len(res_b.results) >= 1


# ---------------------------------------------------------------------------
# delete(ids=...) must not cross tenants
# ---------------------------------------------------------------------------


def test_delete_by_ids_does_not_cross_tenants_even_if_ids_leak(shared_client):
    """If A somehow passes B's point ids into ``delete(ids=...)``, the
    connector's defense-in-depth filter (``HasIdCondition`` +
    ``FieldCondition(tenant)``) must refuse to delete them."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    ids_a = _seed_points(a, [VEC_A], tenant_payload=a.tenant_id)
    ids_b = _seed_points(b, [VEC_B], tenant_payload=b.tenant_id)

    # A tries to delete B's id (simulating a confused caller).
    a.delete(ids=ids_b)

    # B's point must still exist; A's own point is untouched.
    remaining_b = a.client.retrieve(collection_name=b.collection_name, ids=ids_b)
    remaining_a = a.client.retrieve(collection_name=a.collection_name, ids=ids_a)
    assert len(remaining_b) == 1, "Cross-tenant delete must be a no-op, B's point should survive"
    assert len(remaining_a) == 1, "A's own data should be unaffected when misusing delete"

    # Sanity: A's own ids still delete correctly.
    a.delete(ids=ids_a)
    remaining_a2 = a.client.retrieve(collection_name=a.collection_name, ids=ids_a)
    assert len(remaining_a2) == 0


# ---------------------------------------------------------------------------
# delete_collection() = per-tenant purge, not physical drop
# ---------------------------------------------------------------------------


def test_delete_collection_multitenant_purges_only_own_tenant(shared_client):
    """``delete_collection`` on tenant A must wipe A's points but keep the
    global collection alive and B's points intact."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    _seed_points(a, [VEC_A, VEC_A, VEC_A], tenant_payload=a.tenant_id)
    ids_b = _seed_points(b, [VEC_B, VEC_B], tenant_payload=b.tenant_id)

    a.delete_collection()

    # Global collection must still exist (other tenants depend on it).
    assert a.client.collection_exists(a.collection_name)

    # Every A point gone; every B point still present.
    a_points, _ = a.client.scroll(
        collection_name=a.collection_name,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(key=TENANT_PAYLOAD_KEY, match=rest.MatchValue(value=a.tenant_id))]
        ),
        limit=100,
    )
    assert a_points == [], "delete_collection must remove ALL of tenant A's points"

    remaining_b = b.client.retrieve(collection_name=b.collection_name, ids=ids_b)
    assert len(remaining_b) == 2, "delete_collection(A) must not touch tenant B's points"


# ---------------------------------------------------------------------------
# retrieve() defense-in-depth
# ---------------------------------------------------------------------------


def test_retrieve_drops_foreign_tenant_points(shared_client):
    """``retrieve(ids=...)`` in multitenant mode must post-filter out points
    whose ``collection_id`` payload doesn't match the connector's tenant_id,
    even if the caller passes in an id that happens to exist under another
    tenant."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    ids_a = _seed_points(a, [VEC_A], tenant_payload=a.tenant_id)
    ids_b = _seed_points(b, [VEC_B], tenant_payload=b.tenant_id)

    # A calls retrieve with a mixture of its own and B's ids.
    mixed = ids_a + ids_b
    points = a.retrieve(ids=mixed, with_payload=True)

    # Only A's ids survive the post-filter.
    retrieved_ids = {str(p.id) for p in points}
    assert retrieved_ids == set(ids_a), f"leaked foreign points: {retrieved_ids - set(ids_a)}"


# ---------------------------------------------------------------------------
# legacy mode still works
# ---------------------------------------------------------------------------


def test_legacy_mode_uses_tenant_name_as_physical_collection(shared_client):
    """``multitenant=False`` must preserve the historical layout: the physical
    Qdrant collection is named after the tenant id, and no tenant filter is
    applied."""
    legacy = _make_connector("col_legacy_legacyxxxx", client=shared_client, multitenant=False)
    assert legacy.collection_name == "col_legacy_legacyxxxx"
    assert legacy.multitenant is False

    _seed_points(legacy, [VEC_A], tenant_payload="irrelevant_in_legacy")

    # Legacy search does NOT add any tenant filter, so the point comes back
    # even though its payload.collection_id is unrelated to the tenant.
    q = QueryWithEmbedding(query="", top_k=5, embedding=VEC_A)
    res = legacy.search(q, score_threshold=0.0)
    assert len(res.results) == 1
    # DocumentWithScore doesn't expose id/embedding directly; the metadata
    # fallback path (for non-llama_index-shaped payloads) preserves the
    # tenant-payload key so we can check it made it through the connector.
    assert res.results[0].metadata.get(TENANT_PAYLOAD_KEY) == "irrelevant_in_legacy"


# ---------------------------------------------------------------------------
# ctor sanity: multitenant without tenant id is rejected
# ---------------------------------------------------------------------------


def test_multitenant_without_tenant_id_raises():
    with pytest.raises(ValueError, match="requires ctx\\['collection'\\]"):
        QdrantVectorStoreConnector(
            {
                "url": ":memory:",
                "collection": "",  # empty -> rejected
                "vector_size": 4,
                "multitenant": True,
            }
        )
