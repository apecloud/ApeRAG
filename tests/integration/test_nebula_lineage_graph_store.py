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

"""Integration tests for ``NebulaLineageGraphStore`` — Wave 4 T8 chunk 3.

Pin the §D.3.5 lineage SET semantics on a real Nebula 3.x instance.
Six contract scenarios mirror the Postgres + Neo4j chunks (chunk 1 +
chunk 2): roundtrip / multi-doc lineage / doc re-parse v1→v2 via
``remove → upsert`` flow / orphan GC / relation-independent lineage /
per-collection_id tenant isolation.

Tests skip when ``COMPAT_NEBULA_HOSTS`` is unset so the lint-and-unit
CI lane stays green; the e2e-http-compose lane spins Nebula up and
runs them.
"""

from __future__ import annotations

import os
import uuid

import pytest

from aperag.indexing.graph import (
    EntityRecord,
    InMemoryEntityLock,
    LineageMember,
    RelationRecord,
)
from aperag.indexing.graph_storage.nebula import NebulaLineageGraphStore

_NEBULA_HOSTS = os.environ.get("COMPAT_NEBULA_HOSTS") or os.environ.get("TEST_LINEAGE_NEBULA_HOSTS")
_NEBULA_USER = os.environ.get("COMPAT_NEBULA_USER", "root")
_NEBULA_PASS = os.environ.get("COMPAT_NEBULA_PASS", "nebula")


def _nebula_reachable(hosts: str | None) -> bool:
    """Synchronous reachability probe."""

    if not hosts:
        return False
    try:
        from nebula3.Config import Config as NebulaConfig
        from nebula3.gclient.net import ConnectionPool
    except ImportError:  # pragma: no cover — nebula3-python ships under graph-nebula extra
        return False

    try:
        config = NebulaConfig()
        config.max_connection_pool_size = 4
        config.timeout = 5000
        host_pairs = []
        for part in hosts.split(","):
            part = part.strip()
            if ":" in part:
                h, p = part.rsplit(":", 1)
                host_pairs.append((h, int(p)))
            else:
                host_pairs.append((part, 9669))
        pool = ConnectionPool()
        if not pool.init(host_pairs, config):
            pool.close()
            return False
        session = pool.get_session(_NEBULA_USER, _NEBULA_PASS)
        try:
            r = session.execute("SHOW SPACES")
            ok = r.is_succeeded()
        finally:
            session.release()
        pool.close()
        return ok
    except Exception:
        return False


_NEBULA_OK = _nebula_reachable(_NEBULA_HOSTS)
pytestmark = pytest.mark.skipif(
    not _NEBULA_OK,
    reason=(f"Nebula at {_NEBULA_HOSTS or '<unset>'} unreachable; skipping NebulaLineageGraphStore integration suite"),
)


def _make_member(*, doc: str, version: str, chunks: tuple[str, ...] = ()) -> LineageMember:
    return LineageMember(
        document_id=doc,
        parse_version=version,
        tenant_scope_key="public",
        chunk_ids=chunks,
    )


def _build_pool():
    from nebula3.Config import Config as NebulaConfig
    from nebula3.gclient.net import ConnectionPool

    config = NebulaConfig()
    config.max_connection_pool_size = 8
    config.timeout = 60000
    host_pairs = []
    for part in (_NEBULA_HOSTS or "").split(","):
        part = part.strip()
        if ":" in part:
            h, p = part.rsplit(":", 1)
            host_pairs.append((h, int(p)))
        else:
            host_pairs.append((part, 9669))
    pool = ConnectionPool()
    pool.init(host_pairs, config)
    return pool


@pytest.fixture
async def store():
    """Per-test ``NebulaLineageGraphStore`` bound to a unique
    ``collection_id``. The bound space is dropped on teardown so
    leftover vertices don't leak into the next test.
    """

    pool = _build_pool()
    cid = f"lineage_test_{uuid.uuid4().hex[:8]}"
    s = NebulaLineageGraphStore(
        pool=pool,
        username=_NEBULA_USER,
        password=_NEBULA_PASS,
        collection_id=cid,
        entity_lock=InMemoryEntityLock(),
        space_prefix="aperag_lineage_test",
    )
    await s.ensure_schema()
    try:
        yield s
    finally:
        # Drop the test space so subsequent tests start clean. Do this
        # via a fresh session because the store-instance keeps state
        # bound to the dropped space.
        session = pool.get_session(_NEBULA_USER, _NEBULA_PASS)
        try:
            session.execute(f"DROP SPACE IF EXISTS `{s._space}`")
        finally:
            session.release()
        pool.close()


@pytest.mark.asyncio
async def test_roundtrip_entity_with_one_lineage_member(store):
    """Single upsert — read-back returns the canonical lineage view."""

    record = EntityRecord(
        name="Linus Torvalds",
        type="person",
        description="Created Linux.",
        source_chunk_ids=("chunk-1",),
    )
    member = _make_member(doc="doc-A", version="v1", chunks=("chunk-1",))
    await store.upsert_entity_with_lineage(record=record, lineage=member)

    fetched = await store.get_entity("Linus Torvalds")
    assert fetched is not None
    assert fetched.name == "Linus Torvalds"
    assert fetched.type == "person"
    assert len(fetched.source_lineage) == 1
    assert fetched.source_lineage[0].document_id == "doc-A"
    assert fetched.source_lineage[0].parse_version == "v1"
    assert fetched.source_lineage[0].chunk_ids == ("chunk-1",)
    assert len(fetched.description_parts) == 1
    assert fetched.description_parts[0].text == "Created Linux."


@pytest.mark.asyncio
async def test_two_documents_cite_same_entity_preserves_both_lineage(store):
    """§D.3 cross-doc lineage invariant — two docs upsert same entity →
    both members coexist; one's retry must not drop the other.
    """

    record_base = EntityRecord(name="Python", type="language", description="", source_chunk_ids=())
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record_base.__dict__, "description": "Created by Guido."}),
        lineage=_make_member(doc="doc-A", version="v1", chunks=("a-1",)),
    )
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record_base.__dict__, "description": "Dynamically typed."}),
        lineage=_make_member(doc="doc-B", version="v1", chunks=("b-1",)),
    )

    fetched = await store.get_entity("Python")
    assert fetched is not None
    doc_ids = {m.document_id for m in fetched.source_lineage}
    assert doc_ids == {"doc-A", "doc-B"}
    parts_by_doc = {p.document_id: p.text for p in fetched.description_parts}
    assert parts_by_doc == {"doc-A": "Created by Guido.", "doc-B": "Dynamically typed."}


@pytest.mark.asyncio
async def test_doc_re_parse_replaces_old_parse_version_member(store):
    """§D.3.6 step 3 — doc_A v2 supersedes doc_A v1.

    Orchestrator workflow:
    1. cleanup phase — ``remove_entity_lineage_member(name, doc_A)``
    2. rebuild phase — ``upsert_entity_with_lineage(record, doc_A v2)``

    After this two-step flow only the v2 slice for doc_A must remain.
    Also verifies that a same-(doc, parse_v) repeat upsert dedups
    rather than appends — locking the (document_id, parse_version)
    composite as the SET dedup key on Nebula too.
    """

    record = EntityRecord(name="Rust", type="language", description="memory-safe", source_chunk_ids=())
    await store.upsert_entity_with_lineage(
        record=record,
        lineage=_make_member(doc="doc-A", version="v1", chunks=("v1-1",)),
    )
    await store.remove_entity_lineage_member(entity_name="Rust", document_id="doc-A")
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "memory-safe + concurrent"}),
        lineage=_make_member(doc="doc-A", version="v2", chunks=("v2-1",)),
    )

    fetched = await store.get_entity("Rust")
    assert fetched is not None
    versions = {m.parse_version for m in fetched.source_lineage if m.document_id == "doc-A"}
    assert versions == {"v2"}
    parts = [p for p in fetched.description_parts if p.document_id == "doc-A"]
    assert len(parts) == 1
    assert parts[0].parse_version == "v2"
    assert parts[0].text == "memory-safe + concurrent"

    # Repeat upsert with same (doc-A, v2) — must dedup, not append.
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "memory-safe + concurrent"}),
        lineage=_make_member(doc="doc-A", version="v2", chunks=("v2-1",)),
    )
    fetched = await store.get_entity("Rust")
    assert fetched is not None
    doc_a_members = [m for m in fetched.source_lineage if m.document_id == "doc-A"]
    assert len(doc_a_members) == 1


@pytest.mark.asyncio
async def test_remove_then_gc_orphan_entity(store):
    """§D.3.2 phase-1 cleanup → phase-2 GC.

    Stripping a member when other docs cite the entity must NOT make
    it eligible for GC. Stripping all → GC actually deletes.
    """

    record = EntityRecord(name="ApeRAG", type="project", description="", source_chunk_ids=())
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "RAG framework"}),
        lineage=_make_member(doc="doc-A", version="v1"),
    )
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "by ApeCloud"}),
        lineage=_make_member(doc="doc-B", version="v1"),
    )

    await store.remove_entity_lineage_member(entity_name="ApeRAG", document_id="doc-A")
    deleted = await store.gc_entity_if_orphan("ApeRAG")
    assert deleted is False
    fetched = await store.get_entity("ApeRAG")
    assert fetched is not None
    remaining = {m.document_id for m in fetched.source_lineage}
    assert remaining == {"doc-B"}

    await store.remove_entity_lineage_member(entity_name="ApeRAG", document_id="doc-B")
    deleted = await store.gc_entity_if_orphan("ApeRAG")
    assert deleted is True
    fetched = await store.get_entity("ApeRAG")
    assert fetched is None


@pytest.mark.asyncio
async def test_relation_lineage_set_independent_from_entity(store):
    """Relations carry their own evidence_lineage SET; the strip /
    upsert / GC semantics mirror entities. Verify a relation lineage
    round-trip independently of any entity row.
    """

    rel = RelationRecord(
        source="Linus Torvalds",
        target="Linux",
        type="created",
        description="Linus created Linux in 1991.",
        source_chunk_ids=("c-1",),
    )
    await store.upsert_relation_with_lineage(
        record=rel,
        lineage=_make_member(doc="doc-A", version="v1", chunks=("c-1",)),
    )
    await store.upsert_relation_with_lineage(
        record=rel,
        lineage=_make_member(doc="doc-B", version="v1", chunks=("c-1",)),
    )

    keys = await store.find_relation_keys_with_lineage(document_id="doc-A")
    assert ("Linus Torvalds", "Linux", "created") in keys

    fetched = await store.get_relation("Linus Torvalds", "Linux", "created")
    assert fetched is not None
    assert {m.document_id for m in fetched.evidence_lineage} == {"doc-A", "doc-B"}

    await store.remove_relation_lineage_member(
        source="Linus Torvalds", target="Linux", type="created", document_id="doc-A"
    )
    deleted = await store.gc_relation_if_orphan("Linus Torvalds", "Linux", "created")
    assert deleted is False

    await store.remove_relation_lineage_member(
        source="Linus Torvalds", target="Linux", type="created", document_id="doc-B"
    )
    deleted = await store.gc_relation_if_orphan("Linus Torvalds", "Linux", "created")
    assert deleted is True


@pytest.mark.asyncio
async def test_tenant_isolation_collection_id_filters_all_queries(store):
    """Two store instances bound to different collection_ids must not
    see each other's vertices. Nebula realises this via per-collection
    SPACE; the store-instance binds to ``collection_id`` at
    construction so the SPACE selection is automatic.
    """

    other_pool = _build_pool()
    other_cid = f"lineage_test_other_{uuid.uuid4().hex[:8]}"
    other_store = NebulaLineageGraphStore(
        pool=other_pool,
        username=_NEBULA_USER,
        password=_NEBULA_PASS,
        collection_id=other_cid,
        entity_lock=InMemoryEntityLock(),
        space_prefix="aperag_lineage_test",
    )
    await other_store.ensure_schema()

    try:
        record = EntityRecord(
            name="Shared Entity",
            type="thing",
            description="from other tenant",
            source_chunk_ids=(),
        )
        await other_store.upsert_entity_with_lineage(
            record=record,
            lineage=_make_member(doc="other-doc", version="v1"),
        )
        # Sanity — the OTHER tenant CAN see its own row.
        fetched_other = await other_store.get_entity("Shared Entity")
        assert fetched_other is not None

        # The PRIMARY store (different collection_id → different SPACE)
        # MUST NOT see the other tenant's row.
        leaked = await store.get_entity("Shared Entity")
        assert leaked is None
        names = await store.find_entity_ids_with_lineage(document_id="other-doc")
        assert names == []
    finally:
        session = other_pool.get_session(_NEBULA_USER, _NEBULA_PASS)
        try:
            session.execute(f"DROP SPACE IF EXISTS `{other_store._space}`")
        finally:
            session.release()
        other_pool.close()
