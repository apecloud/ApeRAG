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

"""Integration tests for ``Neo4jLineageGraphStore`` — Wave 4 T8 chunk 2.

Pin the §D.3.5 lineage SET semantics on a real Neo4j 5.x instance. The
parallel-list encoding (``source_lineage`` / ``source_lineage_doc_ids``
/ ``source_lineage_parse_versions``) realises a JSONB-shaped SET on a
backend that does not support ``LIST<MAP>`` properties; these tests
lock the backend's correctness against the same six contract scenarios
the chunk 4 cross-backend suite will use.

Tests skip when ``COMPAT_NEO4J_URI`` is unset so the lint-and-unit CI
lane (no Neo4j) stays green; the e2e-http-compose lane spins Neo4j up
and runs them.
"""

from __future__ import annotations

import os
import uuid

import pytest

from aperag.indexing.graph import (
    EntityRecord,
    LineageMember,
    RelationRecord,
)
from aperag.indexing.graph_storage.neo4j import Neo4jLineageGraphStore

_NEO4J_URI = os.environ.get("COMPAT_NEO4J_URI") or os.environ.get(
    "TEST_LINEAGE_NEO4J_URI",
)
_NEO4J_USER = os.environ.get("COMPAT_NEO4J_USER", "neo4j")
_NEO4J_PASS = os.environ.get("COMPAT_NEO4J_PASS", "password")


def _neo4j_reachable(uri: str | None) -> bool:
    """Synchronous reachability probe so the skip decision is taken
    before pytest tries to schedule the async test on the event loop.
    """

    if not uri:
        return False
    try:
        from neo4j import GraphDatabase
    except ImportError:  # pragma: no cover — `neo4j` ships under graph-neo4j extra
        return False

    try:
        driver = GraphDatabase.driver(uri, auth=(_NEO4J_USER, _NEO4J_PASS))
        with driver.session() as session:
            session.run("RETURN 1").consume()
        driver.close()
        return True
    except Exception:
        return False


_NEO4J_OK = _neo4j_reachable(_NEO4J_URI)
pytestmark = pytest.mark.skipif(
    not _NEO4J_OK,
    reason=(f"Neo4j at {_NEO4J_URI or '<unset>'} unreachable; skipping Neo4jLineageGraphStore integration suite"),
)


def _make_member(*, doc: str, version: str, chunks: tuple[str, ...] = ()) -> LineageMember:
    return LineageMember(
        document_id=doc,
        parse_version=version,
        tenant_scope_key="public",
        chunk_ids=chunks,
    )


@pytest.fixture
async def store():
    """A fresh per-test ``Neo4jLineageGraphStore`` bound to a unique
    ``collection_id`` so concurrent runs don't see each other's nodes.
    """

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
    cid = f"lineage_test_{uuid.uuid4().hex[:8]}"
    s = Neo4jLineageGraphStore(driver=driver, collection_id=cid)
    await s.ensure_schema()
    try:
        yield s
    finally:
        # Drop everything for this test's collection so leftover nodes
        # don't accumulate; constraint stays for the next test (cheap
        # idempotent CREATE … IF NOT EXISTS).
        async with driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.collection_id = $cid DETACH DELETE n",
                cid=cid,
            )
        await driver.close()


@pytest.mark.asyncio
async def test_roundtrip_entity_with_one_lineage_member(store):
    """Single upsert — read-back returns the canonical lineage view
    with one member matching the upsert input.
    """

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
    """§D.3 cross-doc lineage invariant — two docs upsert the same
    entity → both members coexist; one's retry must not drop the other.
    """

    record_base = EntityRecord(
        name="Python",
        type="language",
        description="",
        source_chunk_ids=(),
    )
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
    """§D.3.6 step 3 — doc_A v2 supersedes doc_A v1 on the same entity.

    The orchestrator workflow is:
    1. Cleanup phase — ``remove_entity_lineage_member(entity, doc_A)``
       strips ALL doc_A members regardless of parse_version.
    2. Rebuild phase — ``upsert_entity_with_lineage(record, doc_A v2)``
       writes the new slice.

    After this two-step flow the entity must carry only the v2 slice
    for doc_A; v1 must be gone. This locks both the strip-by-doc
    semantics and the upsert add semantics together — which is the
    contract :class:`GraphModalityWorker` relies on.

    Also verifies the upsert's own dedup key: a same-(doc, parse_v)
    repeat upsert (e.g. a retry of phase 2) must coexist as a single
    member, not duplicate.
    """

    record = EntityRecord(
        name="Rust",
        type="language",
        description="memory-safe",
        source_chunk_ids=(),
    )
    # v1 lands first.
    await store.upsert_entity_with_lineage(
        record=record,
        lineage=_make_member(doc="doc-A", version="v1", chunks=("v1-1",)),
    )

    # §D.3.2 phase 1: orchestrator strips the doc on a re-parse.
    await store.remove_entity_lineage_member(entity_name="Rust", document_id="doc-A")

    # §D.3.2 phase 2: orchestrator writes the new (doc_A, v2) slice.
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "memory-safe + concurrent"}),
        lineage=_make_member(doc="doc-A", version="v2", chunks=("v2-1",)),
    )

    fetched = await store.get_entity("Rust")
    assert fetched is not None
    versions = {m.parse_version for m in fetched.source_lineage if m.document_id == "doc-A"}
    assert versions == {"v2"}, "after remove+upsert flow, v1 must be gone and only v2 must remain"
    parts = [p for p in fetched.description_parts if p.document_id == "doc-A"]
    assert len(parts) == 1
    assert parts[0].parse_version == "v2"
    assert parts[0].text == "memory-safe + concurrent"

    # Same-(doc, parse_v) repeat upsert (orchestrator retry) must
    # dedup — the SET must still contain exactly one v2 member.
    await store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "memory-safe + concurrent"}),
        lineage=_make_member(doc="doc-A", version="v2", chunks=("v2-1",)),
    )
    fetched = await store.get_entity("Rust")
    assert fetched is not None
    doc_a_members = [m for m in fetched.source_lineage if m.document_id == "doc-A"]
    assert len(doc_a_members) == 1, "(doc, parse_v) repeat upsert must dedup, not append"


@pytest.mark.asyncio
async def test_remove_then_gc_orphan_entity(store):
    """§D.3.2 phase-1 cleanup → phase-2 GC.

    ``remove_entity_lineage_member(doc_A)`` strips ALL members whose
    document_id matches; if the SET goes empty, ``gc_entity_if_orphan``
    actually deletes the row. Stripping a member when other docs cite
    the entity must NOT make it eligible for GC.
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

    # Strip doc-A → doc-B still cites the entity → GC must NOT delete.
    await store.remove_entity_lineage_member(entity_name="ApeRAG", document_id="doc-A")
    deleted = await store.gc_entity_if_orphan("ApeRAG")
    assert deleted is False
    fetched = await store.get_entity("ApeRAG")
    assert fetched is not None
    remaining = {m.document_id for m in fetched.source_lineage}
    assert remaining == {"doc-B"}

    # Strip doc-B → SET goes empty → GC deletes.
    await store.remove_entity_lineage_member(entity_name="ApeRAG", document_id="doc-B")
    deleted = await store.gc_entity_if_orphan("ApeRAG")
    assert deleted is True
    fetched = await store.get_entity("ApeRAG")
    assert fetched is None


@pytest.mark.asyncio
async def test_relation_lineage_set_independent_from_entity(store):
    """Relations carry their own evidence_lineage SET; the strip/upsert
    semantics mirror entities. Verify a relation upsert + strip + GC
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
    assert deleted is False, "doc-B still cites; relation must not be GC'd"

    await store.remove_relation_lineage_member(
        source="Linus Torvalds", target="Linux", type="created", document_id="doc-B"
    )
    deleted = await store.gc_relation_if_orphan("Linus Torvalds", "Linux", "created")
    assert deleted is True


@pytest.mark.asyncio
async def test_tenant_isolation_collection_id_filters_all_queries(store):
    """Two store instances bound to different collection_ids must not
    see each other's rows — even though both write the same entity
    name, ``find_entity_ids_with_lineage`` and ``get_entity`` filter
    on the bound ``collection_id``.

    This is the per-store-instance binding half of the §H.2 tenant
    double-layer (the other half is the composite-key uniqueness
    constraint, exercised by ensure_schema).
    """

    from neo4j import AsyncGraphDatabase

    other_cid = f"lineage_test_other_{uuid.uuid4().hex[:8]}"
    other_driver = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
    other_store = Neo4jLineageGraphStore(driver=other_driver, collection_id=other_cid)
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

        # The PRIMARY store (bound to a different collection_id) MUST
        # NOT see the other tenant's row.
        leaked = await store.get_entity("Shared Entity")
        assert leaked is None
        names = await store.find_entity_ids_with_lineage(document_id="other-doc")
        assert names == []
    finally:
        async with other_driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.collection_id = $cid DETACH DELETE n",
                cid=other_cid,
            )
        await other_driver.close()
