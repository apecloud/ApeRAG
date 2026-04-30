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

Note: payload indexes are a no-op in local mode (the client prints a
warning to that effect). That's fine — we're exercising correctness of
the filter-based isolation, not the segment-level defragmentation benefit.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as rest  # noqa: E402

from aperag.vectorstore import qdrant_connector as qc  # noqa: E402
from aperag.vectorstore.dto import (  # noqa: E402
    QueryRequest,
    TenantRef,
    VectorPoint,
    VectorShape,
)
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
    """One in-memory Qdrant client shared by all connectors in a test."""
    return QdrantClient(":memory:")


def _make_connector(
    tenant_id: str,
    client: QdrantClient,
    multitenant: bool = True,
) -> QdrantVectorStoreConnector:
    """Spin up a connector bound to a caller-provided client.

    The trick: we use ``__new__`` to bypass ``__init__`` (which would build
    a fresh ``:memory:`` client, not share ours) and then populate the
    fields by hand. The connector's contract from the rest of the codebase
    is unchanged — external callers still go through ``__init__``.
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
    if multitenant and not ctx.get("collection"):
        raise ValueError("QdrantVectorStoreConnector(multitenant=True) requires ctx['collection']")

    conn = QdrantVectorStoreConnector.__new__(QdrantVectorStoreConnector)
    conn.ctx = ctx
    conn.multitenant = multitenant
    conn.cfg = ctx
    conn._tenant = TenantRef(id=str(tenant_id))
    # VectorShape canonicalizes distance to lowercase.
    conn._shape = VectorShape(size=VECTOR_SIZE, distance="cosine")
    conn.url = ":memory:"
    conn.port = 6333
    conn.grpc_port = 6334
    conn.prefer_grpc = False
    conn.https = False
    conn.timeout = 300
    conn.client = client

    # opclass / score expr aren't used by Qdrant but the constructor
    # leaves them absent in this path; nothing reads them.

    if multitenant:
        conn.collection_name = global_collection_name(VECTOR_SIZE, "cosine")
        conn.ensure_collection()
    else:
        conn.collection_name = tenant_id
        if not client.collection_exists(conn.collection_name):
            client.create_collection(
                collection_name=conn.collection_name,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
            )

    return conn


def _seed_points(
    conn: QdrantVectorStoreConnector,
    vectors: List[List[float]],
    tenant_payload: str,
) -> List[str]:
    """Write raw points into the connector's physical collection.

    We bypass ``upsert()`` because that method stamps ``collection_id``
    from the connector's own ``tenant`` — which is exactly the behavior
    under test for some scenarios. For "what if a foreign tenant's id
    leaks in?" cases we need to plant payloads with arbitrary tenant ids,
    which only direct ``client.upsert`` can do.
    """
    ids = [str(uuid.uuid4()) for _ in vectors]
    points = [
        rest.PointStruct(id=pid, vector=vec, payload={TENANT_PAYLOAD_KEY: tenant_payload})
        for pid, vec in zip(ids, vectors)
    ]
    conn.client.upsert(collection_name=conn.collection_name, points=points, wait=True)
    return ids


def _query(conn: QdrantVectorStoreConnector, vec: List[float], top_k: int = 10) -> list:
    """Tiny helper to call ``search`` with just an embedding and a non-
    restrictive threshold."""
    return conn.search(QueryRequest(embedding=vec, top_k=top_k, score_threshold=-1.0))


# ---------------------------------------------------------------------------
# upsert goes native (no LlamaIndex)
# ---------------------------------------------------------------------------


def test_upsert_writes_tenant_stamped_points(shared_client):
    """Round-trip via the new ``upsert()`` API: the connector must stamp
    ``collection_id`` on every point (matching the tenant), write
    successfully, and make the points visible through ``search``."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    points = [
        VectorPoint(
            id=str(uuid.uuid4()),
            vector=VEC_A,
            payload={"text": "hello", "metadata": {"source": "unit-test"}},
        ),
    ]
    ids = a.upsert(points)
    assert ids == [points[0].id]

    hits = _query(a, VEC_A)
    assert len(hits) == 1
    # The stamped tenant id must land in the payload so downstream
    # filter guards have something to match against.
    assert hits[0].payload.get(TENANT_PAYLOAD_KEY) == a.tenant.id
    # Payload round-trips.
    assert hits[0].payload.get("text") == "hello"


# ---------------------------------------------------------------------------
# tenant isolation on search
# ---------------------------------------------------------------------------


def test_multitenant_search_is_isolated_between_tenants(shared_client):
    """Two tenants writing to the same physical collection must not see each
    other's points through the connector's search path."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    assert a.collection_name == b.collection_name == global_collection_name(VECTOR_SIZE, "cosine")

    _seed_points(a, [VEC_A, VEC_A], tenant_payload=a.tenant.id)
    _seed_points(b, [VEC_A, VEC_B], tenant_payload=b.tenant.id)

    res_a = _query(a, VEC_A)
    res_b = _query(b, VEC_A)

    tenants_seen_by_a = {h.payload.get(TENANT_PAYLOAD_KEY) for h in res_a}
    tenants_seen_by_b = {h.payload.get(TENANT_PAYLOAD_KEY) for h in res_b}

    assert tenants_seen_by_a == {a.tenant.id}, f"tenant A saw foreign points: {tenants_seen_by_a}"
    assert tenants_seen_by_b == {b.tenant.id}, f"tenant B saw foreign points: {tenants_seen_by_b}"
    assert len(res_a) == 2
    assert len(res_b) >= 1


# ---------------------------------------------------------------------------
# delete(ids=...) must not cross tenants
# ---------------------------------------------------------------------------


def test_delete_by_ids_does_not_cross_tenants_even_if_ids_leak(shared_client):
    """If A somehow passes B's point ids into ``delete(ids=...)``, the
    connector's defense-in-depth filter must refuse to delete them."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    ids_a = _seed_points(a, [VEC_A], tenant_payload=a.tenant.id)
    ids_b = _seed_points(b, [VEC_B], tenant_payload=b.tenant.id)

    # A tries to delete B's id (simulating a confused caller).
    a.delete(ids_b)

    # B's point must still exist; A's own point is untouched.
    remaining_b = a.client.retrieve(collection_name=b.collection_name, ids=ids_b)
    remaining_a = a.client.retrieve(collection_name=a.collection_name, ids=ids_a)
    assert len(remaining_b) == 1, "Cross-tenant delete must be a no-op"
    assert len(remaining_a) == 1, "A's own data should be unaffected"

    # Sanity: A's own ids still delete correctly.
    a.delete(ids_a)
    remaining_a2 = a.client.retrieve(collection_name=a.collection_name, ids=ids_a)
    assert len(remaining_a2) == 0


# ---------------------------------------------------------------------------
# drop_tenant = per-tenant purge, not physical drop
# ---------------------------------------------------------------------------


def test_drop_tenant_multitenant_purges_only_own_tenant(shared_client):
    """``drop_tenant`` on tenant A must wipe A's points but keep the
    global collection alive and B's points intact."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    _seed_points(a, [VEC_A, VEC_A, VEC_A], tenant_payload=a.tenant.id)
    ids_b = _seed_points(b, [VEC_B, VEC_B], tenant_payload=b.tenant.id)

    a.drop_tenant()

    # Global collection must still exist (other tenants depend on it).
    assert a.client.collection_exists(a.collection_name)

    # Every A point gone; every B point still present.
    a_points, _ = a.client.scroll(
        collection_name=a.collection_name,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(key=TENANT_PAYLOAD_KEY, match=rest.MatchValue(value=a.tenant.id))]
        ),
        limit=100,
    )
    assert a_points == [], "drop_tenant must remove ALL of tenant A's points"

    remaining_b = b.client.retrieve(collection_name=b.collection_name, ids=ids_b)
    assert len(remaining_b) == 2, "drop_tenant(A) must not touch tenant B's points"


# ---------------------------------------------------------------------------
# retrieve() defense-in-depth
# ---------------------------------------------------------------------------


def test_retrieve_drops_foreign_tenant_points(shared_client):
    """``retrieve(ids=...)`` in multitenant mode must post-filter out points
    whose ``collection_id`` payload doesn't match the connector's tenant id,
    even if the caller passes in an id that happens to exist under another
    tenant."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    ids_a = _seed_points(a, [VEC_A], tenant_payload=a.tenant.id)
    ids_b = _seed_points(b, [VEC_B], tenant_payload=b.tenant.id)

    mixed = ids_a + ids_b
    points = a.retrieve(mixed)

    retrieved_ids = {p.id for p in points}
    assert retrieved_ids == set(ids_a), f"leaked foreign points: {retrieved_ids - set(ids_a)}"


def test_retrieve_multitenant_mode_strict_requires_payload_key(shared_client):
    """Pinned by task #61 P1-V4 BLOCKER from Weston msg=910cad66:
    in multitenant mode the payload-level tenant filter must be
    **strict** — a row MUST carry ``TENANT_PAYLOAD_KEY`` matching
    this tenant. No "no payload key → pass through" branch in
    multitenant mode, because the physical collection is shared,
    and a missing-key row would otherwise expose to every tenant
    on a ``retrieve(ids=...)`` call.

    The legacy-mode permissive branch (no payload key passes
    through) is exercised by
    :func:`test_retrieve_legacy_mode_filters_stray_foreign_payload`;
    this test pins the multitenant strict counterpart so a future
    refactor can't unify them and silently re-open the leak.
    """
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)

    # Seed three points directly on the shared multitenant collection:
    # 1. ``{}`` payload (no TENANT_PAYLOAD_KEY at all — the stray
    #    case Weston caught).
    # 2. Own-tenant payload — must pass through.
    # 3. Foreign-tenant payload — must be dropped.
    no_payload_id = str(uuid.uuid4())
    own_payload_id = str(uuid.uuid4())
    foreign_payload_id = str(uuid.uuid4())
    shared_client.upsert(
        collection_name=a.collection_name,
        points=[
            rest.PointStruct(id=no_payload_id, vector=VEC_A, payload={}),
            rest.PointStruct(
                id=own_payload_id,
                vector=VEC_A,
                payload={TENANT_PAYLOAD_KEY: a.tenant.id},
            ),
            rest.PointStruct(
                id=foreign_payload_id,
                vector=VEC_A,
                payload={TENANT_PAYLOAD_KEY: "col_some_other_tenant"},
            ),
        ],
        wait=True,
    )

    out = a.retrieve([no_payload_id, own_payload_id, foreign_payload_id])
    out_ids = {p.id for p in out}

    # Strict: only own-tenant row passes.
    assert no_payload_id not in out_ids, (
        "multitenant mode must NOT pass-through rows missing TENANT_PAYLOAD_KEY — "
        "shared collection means a missing key would leak to every tenant"
    )
    assert own_payload_id in out_ids, "own-tenant payload must pass through"
    assert foreign_payload_id not in out_ids, "foreign-tenant payload must be dropped"


def test_retrieve_legacy_mode_filters_stray_foreign_payload(shared_client):
    """Pinned by task #61 P1-V4: even in legacy (per-tenant physical
    collection) mode the ``retrieve()`` post-filter must drop any
    point that carries a ``TENANT_PAYLOAD_KEY`` payload whose value
    doesn't match the connector's tenant id.

    Primary isolation in legacy mode is the physical collection name
    (``collection_name == tenant_id``) — a stray foreign-tenant point
    can't normally reach a legacy collection. But tooling drift /
    migration mistakes / re-ingest scripts could accidentally write
    one. The defense-in-depth filter catches that case before it
    surfaces as a cross-tenant data leak.

    Legacy rows that don't carry ``TENANT_PAYLOAD_KEY`` at all (the
    common case in legacy mode) pass through unchanged — backward-
    compatible with the historical legacy data.
    """
    legacy = _make_connector("col_legacy_owner", client=shared_client, multitenant=False)

    # Seed 3 points: 1 with no payload key (typical legacy row), 1
    # with the matching tenant id, 1 with a *foreign* tenant id (the
    # stray case the filter must catch).
    no_payload_id = str(uuid.uuid4())
    own_payload_id = str(uuid.uuid4())
    foreign_payload_id = str(uuid.uuid4())
    shared_client.upsert(
        collection_name=legacy.collection_name,
        points=[
            rest.PointStruct(id=no_payload_id, vector=VEC_A, payload={}),
            rest.PointStruct(
                id=own_payload_id,
                vector=VEC_A,
                payload={TENANT_PAYLOAD_KEY: legacy.tenant.id},
            ),
            rest.PointStruct(
                id=foreign_payload_id,
                vector=VEC_A,
                payload={TENANT_PAYLOAD_KEY: "col_some_other_tenant"},
            ),
        ],
        wait=True,
    )

    out = legacy.retrieve([no_payload_id, own_payload_id, foreign_payload_id])
    out_ids = {p.id for p in out}

    # The no-payload + own-payload rows pass through; the foreign-
    # payload row is filtered.
    assert no_payload_id in out_ids, "legacy row without TENANT_PAYLOAD_KEY must pass through (backward compat)"
    assert own_payload_id in out_ids, "own-tenant payload must pass through"
    assert foreign_payload_id not in out_ids, "stray foreign-tenant payload must be dropped (P1-V4 defense-in-depth)"


# ---------------------------------------------------------------------------
# delete_by_filter
# ---------------------------------------------------------------------------


def test_delete_by_filter_tenant_guarded(shared_client):
    """``delete_by_filter`` must AND-merge with the tenant guard so a
    filter that would match B's points still leaves them alone when A
    issues it."""
    from aperag.vectorstore.filters import Eq

    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    b = _make_connector("col_bbbbbbbbbbbbb_b", client=shared_client)

    # Both tenants seed a point with indexer=vector.
    a_id = str(uuid.uuid4())
    b_id = str(uuid.uuid4())
    shared_client.upsert(
        collection_name=a.collection_name,
        points=[
            rest.PointStruct(id=a_id, vector=VEC_A, payload={TENANT_PAYLOAD_KEY: a.tenant.id, "indexer": "vector"}),
            rest.PointStruct(id=b_id, vector=VEC_A, payload={TENANT_PAYLOAD_KEY: b.tenant.id, "indexer": "vector"}),
        ],
        wait=True,
    )

    # A deletes every ``indexer=vector`` row — should only touch its own.
    a.delete_by_filter(Eq(key="indexer", value="vector"))

    # A's row gone, B's row survives.
    assert a.client.retrieve(collection_name=a.collection_name, ids=[a_id]) == []
    assert len(b.client.retrieve(collection_name=b.collection_name, ids=[b_id])) == 1


def test_delete_by_filter_empty_filter_rejected(shared_client):
    """An empty / None filter must never be accepted — it would DELETE
    every point in the shared global collection. We prefer an explicit
    raise to silent truncation."""
    a = _make_connector("col_aaaaaaaaaaaaa_a", client=shared_client)
    with pytest.raises(ValueError, match="non-empty filter"):
        a.delete_by_filter(None)  # type: ignore[arg-type]


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

    # Legacy search does NOT add any tenant filter, so the point comes
    # back even though its payload.collection_id is unrelated.
    hits = _query(legacy, VEC_A, top_k=5)
    assert len(hits) == 1
    assert hits[0].payload.get(TENANT_PAYLOAD_KEY) == "irrelevant_in_legacy"


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


# ---------------------------------------------------------------------------
# purge_all_shards
# ---------------------------------------------------------------------------


def _make_shard(client: QdrantClient, vector_size: int, distance: str = "cosine") -> str:
    name = global_collection_name(vector_size, distance)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
    return name


def _seed_into_shard(client: QdrantClient, shard: str, tenant: str, vectors: List[List[float]]) -> List[str]:
    ids = [str(uuid.uuid4()) for _ in vectors]
    points = [
        rest.PointStruct(id=pid, vector=vec, payload={TENANT_PAYLOAD_KEY: tenant}) for pid, vec in zip(ids, vectors)
    ]
    client.upsert(collection_name=shard, points=points, wait=True)
    return ids


def test_purge_all_shards_removes_tenant_from_every_global_collection(shared_client):
    """The exact scenario behind ``drop_tenant(purge_all_shards=True)``:
    an ApeRAG collection's embedding provider is removed, vector_size
    cannot be resolved, so the normal drop routes to a default shard and
    leaves orphans in the shard that actually holds data."""
    shard4 = _make_shard(shared_client, 4)
    shard3 = _make_shard(shared_client, 3)

    ids_a4 = _seed_into_shard(shared_client, shard4, "col_a", [[1.0, 0.0, 0.0, 0.0]])
    ids_a3 = _seed_into_shard(shared_client, shard3, "col_a", [[1.0, 0.0, 0.0]])
    ids_b4 = _seed_into_shard(shared_client, shard4, "col_b", [[0.0, 1.0, 0.0, 0.0]])

    a = _make_connector("col_a", client=shared_client)
    a.drop_tenant(purge_all_shards=True)

    for shard, ids in ((shard4, ids_a4), (shard3, ids_a3)):
        survivors = shared_client.retrieve(collection_name=shard, ids=ids)
        assert survivors == [], f"purge_all_shards left {len(survivors)} of A in {shard}"

    remaining_b = shared_client.retrieve(collection_name=shard4, ids=ids_b4)
    assert len(remaining_b) == 1

    assert shared_client.collection_exists(shard4)
    assert shared_client.collection_exists(shard3)


def test_purge_all_shards_ignores_legacy_named_collections(shared_client):
    """``_purge_tenant_from_all_global_collections`` must scan only names
    starting with ``aperag_vectors_``. Legacy / unrelated collections in
    the same Qdrant cluster must be left completely alone."""
    shard4 = _make_shard(shared_client, 4)
    ids_a4 = _seed_into_shard(shared_client, shard4, "col_a", [[1.0, 0.0, 0.0, 0.0]])

    legacy_name = "col_legacy_unrelated_x"
    shared_client.create_collection(
        collection_name=legacy_name,
        vectors_config=rest.VectorParams(size=4, distance=rest.Distance.COSINE),
    )
    legacy_ids = _seed_into_shard(shared_client, legacy_name, "col_a", [[0.0, 1.0, 0.0, 0.0]])

    a = _make_connector("col_a", client=shared_client)
    a.drop_tenant(purge_all_shards=True)

    assert shared_client.retrieve(collection_name=shard4, ids=ids_a4) == []
    still_there = shared_client.retrieve(collection_name=legacy_name, ids=legacy_ids)
    assert len(still_there) == 1, "purge must not touch non-aperag_vectors_* collections"
