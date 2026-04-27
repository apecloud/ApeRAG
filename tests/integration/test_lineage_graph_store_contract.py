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

"""Cross-backend ``LineageGraphStore`` contract — Wave 4 T8 chunk 4c.

Locks the §D.3.5 Protocol semantics across all three production
backends (Postgres reference adapter / Neo4j parallel-list mirror /
Nebula JSON-string mirror) with a single 6-case fixture so a backend
that drifts from the spec breaks here, regardless of which adapter
it is.

The six cases are the canonical contract bar (per architect
msg=baf6618e + huangheng msg=b6f20096 chunk 4 acceptance item 4):

1. ``roundtrip_entity_with_one_lineage_member`` — basic upsert + read
2. ``two_documents_cite_same_entity_preserves_both_lineage`` —
   §D.3 cross-doc lineage SET
3. ``doc_re_parse_replaces_old_parse_version_member`` — §D.3.6 step 3
   "doc_A v2 supersedes doc_A v1" remove + upsert + dedup composite
   key ``(document_id, parse_version)``
4. ``remove_then_gc_orphan_entity`` — §D.3.2 phase-1 strip → phase-2
   GC; orphan-only deletion
5. ``relation_lineage_set_independent_from_entity`` — relation
   evidence_lineage SET independent of entity source_lineage
6. ``tenant_isolation_collection_id_filters_all_queries`` — §H.2
   per-store-instance binding tenant double-layer

Each backend is reached behind an env-var skip so the lint-and-unit CI
lane (no Postgres / Neo4j / Nebula) stays green; the e2e-http-compose
lane spins all three and runs them.

The per-backend tests in ``test_neo4j_lineage_graph_store.py`` and
``test_nebula_lineage_graph_store.py`` keep backend-specific extras
(parallel-list encoding shape, Nebula schema-visibility retry) that
cannot be expressed as Protocol-level invariants; this file covers
the Protocol-level contract that all three backends must obey.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Awaitable, Callable

import pytest

from aperag.indexing.graph import (
    EntityRecord,
    InMemoryEntityLock,
    LineageMember,
    RelationRecord,
)

# ---------------------------------------------------------------------
# Backend reachability probes — skip a backend's parametrize cell if
# the env-var is unset (lint-and-unit CI lane stays green).
# ---------------------------------------------------------------------


_POSTGRES_DSN = os.environ.get(
    "COMPAT_POSTGRES_DSN",
    os.environ.get(
        "TEST_LINEAGE_POSTGRES_DSN",
        "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres",
    ),
)


def _postgres_reachable() -> bool:
    if not _POSTGRES_DSN:
        return False
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return False

    async def _probe() -> bool:
        engine = create_async_engine(_POSTGRES_DSN)
        try:
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            return True
        finally:
            await engine.dispose()

    try:
        return asyncio.run(_probe())
    except Exception:
        return False


_NEO4J_URI = os.environ.get("COMPAT_NEO4J_URI") or os.environ.get("TEST_LINEAGE_NEO4J_URI")
_NEO4J_USER = os.environ.get("COMPAT_NEO4J_USER", "neo4j")
_NEO4J_PASS = os.environ.get("COMPAT_NEO4J_PASS", "password")


def _neo4j_reachable() -> bool:
    if not _NEO4J_URI:
        return False
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return False
    try:
        driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
        with driver.session() as session:
            session.run("RETURN 1").consume()
        driver.close()
        return True
    except Exception:
        return False


_NEBULA_HOSTS = os.environ.get("COMPAT_NEBULA_HOSTS") or os.environ.get("TEST_LINEAGE_NEBULA_HOSTS")
_NEBULA_USER = os.environ.get("COMPAT_NEBULA_USER", "root")
_NEBULA_PASS = os.environ.get("COMPAT_NEBULA_PASS", "nebula")


def _nebula_reachable() -> bool:
    if not _NEBULA_HOSTS:
        return False
    try:
        from nebula3.Config import Config as _NebulaConfig
        from nebula3.gclient.net import ConnectionPool
    except ImportError:
        return False
    hosts: list[tuple[str, int]] = []
    for raw_host in _NEBULA_HOSTS.split(","):
        host_part = raw_host.strip()
        if not host_part:
            continue
        host, _, port_str = host_part.partition(":")
        hosts.append((host, int(port_str or "9669")))
    if not hosts:
        return False
    try:
        config = _NebulaConfig()
        pool = ConnectionPool()
        if not pool.init(hosts, config):
            return False
        pool.close()
        return True
    except Exception:
        return False


_POSTGRES_OK = _postgres_reachable()
_NEO4J_OK = _neo4j_reachable()
_NEBULA_OK = _nebula_reachable()


# ---------------------------------------------------------------------
# Per-backend store builder + cleanup. Each returns an async context-
# manager-style ``(store, cleanup)`` so the parametrize fixture can use
# the same lifecycle for any backend.
# ---------------------------------------------------------------------


async def _build_postgres_store(collection_id: str) -> tuple[Any, Callable[[], Awaitable[None]]]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from aperag.indexing.graph_storage.postgres import PostgresLineageGraphStore

    engine = create_async_engine(_POSTGRES_DSN)
    store = PostgresLineageGraphStore(engine=engine, collection_id=collection_id)
    # Tables are owned by alembic in production; ensure_schema is the
    # test-fallback. Safe + idempotent so we always call it in tests.
    await store.ensure_schema()

    async def _cleanup() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM aperag_lineage_entity WHERE collection_id = :c"),
                {"c": collection_id},
            )
            await conn.execute(
                text("DELETE FROM aperag_lineage_relation WHERE collection_id = :c"),
                {"c": collection_id},
            )
        await engine.dispose()

    return store, _cleanup


async def _build_neo4j_store(collection_id: str) -> tuple[Any, Callable[[], Awaitable[None]]]:
    from neo4j import AsyncGraphDatabase

    from aperag.indexing.graph_storage.neo4j import Neo4jLineageGraphStore

    driver = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
    store = Neo4jLineageGraphStore(driver=driver, collection_id=collection_id)
    await store.ensure_schema()

    async def _cleanup() -> None:
        async with driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.collection_id = $cid DETACH DELETE n",
                cid=collection_id,
            )
        await driver.close()

    return store, _cleanup


async def _build_nebula_store(collection_id: str) -> tuple[Any, Callable[[], Awaitable[None]]]:
    from nebula3.Config import Config as _NebulaConfig
    from nebula3.gclient.net import ConnectionPool

    from aperag.indexing.graph_storage.nebula import NebulaLineageGraphStore, _space_name

    hosts: list[tuple[str, int]] = []
    for raw_host in _NEBULA_HOSTS.split(","):
        host_part = raw_host.strip()
        if not host_part:
            continue
        host, _, port_str = host_part.partition(":")
        hosts.append((host, int(port_str or "9669")))
    config = _NebulaConfig()
    pool = ConnectionPool()
    assert pool.init(hosts, config), f"nebula pool init({hosts}) failed"
    space_prefix = "aperag_lineage_test"
    store = NebulaLineageGraphStore(
        pool=pool,
        username=_NEBULA_USER,
        password=_NEBULA_PASS,
        collection_id=collection_id,
        entity_lock=InMemoryEntityLock(),
        space_prefix=space_prefix,
    )
    await store.ensure_schema()

    async def _cleanup() -> None:
        # DROP the per-collection SPACE — fastest tenant wipe.
        space = _space_name(space_prefix, collection_id)
        session = pool.get_session(_NEBULA_USER, _NEBULA_PASS)
        try:
            session.execute(f"DROP SPACE IF EXISTS `{space}`")
        finally:
            session.release()
        pool.close()

    return store, _cleanup


_BACKEND_BUILDERS: dict[str, tuple[bool, Callable[[str], Awaitable[tuple[Any, Callable[[], Awaitable[None]]]]]]] = {
    "postgres": (_POSTGRES_OK, _build_postgres_store),
    "neo4j": (_NEO4J_OK, _build_neo4j_store),
    "nebula": (_NEBULA_OK, _build_nebula_store),
}


@pytest.fixture(params=["postgres", "neo4j", "nebula"])
async def lineage_store(request: pytest.FixtureRequest):
    """Yield a fresh per-test :class:`LineageGraphStore` for each
    parametrized backend. Skips the cell when the backend is
    unreachable (env-var unset / connection refused)."""

    backend = request.param
    reachable, builder = _BACKEND_BUILDERS[backend]
    if not reachable:
        pytest.skip(f"backend={backend} unreachable; skipping cross-backend contract case")

    cid = f"lineage_contract_{backend}_{uuid.uuid4().hex[:8]}"
    store, cleanup = await builder(cid)
    try:
        yield store
    finally:
        await cleanup()


# ---------------------------------------------------------------------
# Helpers — shared across the 6 cases.
# ---------------------------------------------------------------------


def _make_member(*, doc: str, version: str, chunks: tuple[str, ...] = ()) -> LineageMember:
    return LineageMember(
        document_id=doc,
        parse_version=version,
        tenant_scope_key="public",
        chunk_ids=chunks,
    )


# ---------------------------------------------------------------------
# 6 contract cases — each runs against every backend the parametrize
# fixture yields.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_roundtrip_entity_with_one_lineage_member(lineage_store):
    """Case 1: basic upsert + read — the canonical lineage view returns
    one member with the upsert input echoed back."""

    record = EntityRecord(
        name="Linus Torvalds",
        type="person",
        description="Created Linux.",
        source_chunk_ids=("chunk-1",),
    )
    member = _make_member(doc="doc-A", version="v1", chunks=("chunk-1",))
    await lineage_store.upsert_entity_with_lineage(record=record, lineage=member)

    fetched = await lineage_store.get_entity("Linus Torvalds")
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
async def test_two_documents_cite_same_entity_preserves_both_lineage(lineage_store):
    """Case 2: §D.3 cross-doc lineage — two docs upsert the same entity
    must coexist as two SET members; one document's retry must not drop
    the other's contribution."""

    record_base = EntityRecord(
        name="Python",
        type="language",
        description="",
        source_chunk_ids=(),
    )
    await lineage_store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record_base.__dict__, "description": "Created by Guido."}),
        lineage=_make_member(doc="doc-A", version="v1", chunks=("a-1",)),
    )
    await lineage_store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record_base.__dict__, "description": "Dynamically typed."}),
        lineage=_make_member(doc="doc-B", version="v1", chunks=("b-1",)),
    )

    fetched = await lineage_store.get_entity("Python")
    assert fetched is not None
    doc_ids = {m.document_id for m in fetched.source_lineage}
    assert doc_ids == {"doc-A", "doc-B"}
    parts_by_doc = {p.document_id: p.text for p in fetched.description_parts}
    assert parts_by_doc == {"doc-A": "Created by Guido.", "doc-B": "Dynamically typed."}


@pytest.mark.asyncio
async def test_doc_re_parse_replaces_old_parse_version_member(lineage_store):
    """Case 3: §D.3.6 step 3 + composite key dedup — the
    ``remove(doc_A) → upsert(doc_A, v2)`` orchestrator flow leaves only
    the v2 slice for doc_A; the v1 member is gone. A same-(doc, parse_v)
    repeat upsert (orchestrator retry) must dedup, not duplicate.
    """

    record = EntityRecord(
        name="Rust",
        type="language",
        description="memory-safe",
        source_chunk_ids=(),
    )
    await lineage_store.upsert_entity_with_lineage(
        record=record,
        lineage=_make_member(doc="doc-A", version="v1", chunks=("v1-1",)),
    )

    await lineage_store.remove_entity_lineage_member(entity_name="Rust", document_id="doc-A")

    await lineage_store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "memory-safe + concurrent"}),
        lineage=_make_member(doc="doc-A", version="v2", chunks=("v2-1",)),
    )

    fetched = await lineage_store.get_entity("Rust")
    assert fetched is not None
    versions = {m.parse_version for m in fetched.source_lineage if m.document_id == "doc-A"}
    assert versions == {"v2"}, "after remove+upsert flow, v1 must be gone and only v2 must remain"
    parts = [p for p in fetched.description_parts if p.document_id == "doc-A"]
    assert len(parts) == 1
    assert parts[0].parse_version == "v2"
    assert parts[0].text == "memory-safe + concurrent"

    # Same-(doc, parse_v) repeat upsert (orchestrator retry) must dedup.
    await lineage_store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "memory-safe + concurrent"}),
        lineage=_make_member(doc="doc-A", version="v2", chunks=("v2-1",)),
    )
    fetched = await lineage_store.get_entity("Rust")
    assert fetched is not None
    doc_a_members = [m for m in fetched.source_lineage if m.document_id == "doc-A"]
    assert len(doc_a_members) == 1, "(doc, parse_v) repeat upsert must dedup, not append"


@pytest.mark.asyncio
async def test_remove_then_gc_orphan_entity(lineage_store):
    """Case 4: §D.3.2 phase-1 strip → phase-2 GC. Orphan-only deletion;
    other-doc citations protect the row from GC."""

    record = EntityRecord(name="ApeRAG", type="project", description="", source_chunk_ids=())
    await lineage_store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "RAG framework"}),
        lineage=_make_member(doc="doc-A", version="v1"),
    )
    await lineage_store.upsert_entity_with_lineage(
        record=EntityRecord(**{**record.__dict__, "description": "by ApeCloud"}),
        lineage=_make_member(doc="doc-B", version="v1"),
    )

    await lineage_store.remove_entity_lineage_member(entity_name="ApeRAG", document_id="doc-A")
    deleted = await lineage_store.gc_entity_if_orphan("ApeRAG")
    assert deleted is False, "doc-B still cites; entity must not be GC'd"
    fetched = await lineage_store.get_entity("ApeRAG")
    assert fetched is not None
    remaining = {m.document_id for m in fetched.source_lineage}
    assert remaining == {"doc-B"}

    await lineage_store.remove_entity_lineage_member(entity_name="ApeRAG", document_id="doc-B")
    deleted = await lineage_store.gc_entity_if_orphan("ApeRAG")
    assert deleted is True
    fetched = await lineage_store.get_entity("ApeRAG")
    assert fetched is None


@pytest.mark.asyncio
async def test_relation_lineage_set_independent_from_entity(lineage_store):
    """Case 5: relation evidence_lineage SET semantics mirror entity
    source_lineage but on a different SET; relation upsert + strip + GC
    work independently of any entity row."""

    rel = RelationRecord(
        source="Linus Torvalds",
        target="Linux",
        type="created",
        description="Linus created Linux in 1991.",
        source_chunk_ids=("c-1",),
    )
    await lineage_store.upsert_relation_with_lineage(
        record=rel,
        lineage=_make_member(doc="doc-A", version="v1", chunks=("c-1",)),
    )
    await lineage_store.upsert_relation_with_lineage(
        record=rel,
        lineage=_make_member(doc="doc-B", version="v1", chunks=("c-1",)),
    )

    keys = await lineage_store.find_relation_keys_with_lineage(document_id="doc-A")
    assert ("Linus Torvalds", "Linux", "created") in keys

    fetched = await lineage_store.get_relation("Linus Torvalds", "Linux", "created")
    assert fetched is not None
    assert {m.document_id for m in fetched.evidence_lineage} == {"doc-A", "doc-B"}

    await lineage_store.remove_relation_lineage_member(
        source="Linus Torvalds", target="Linux", type="created", document_id="doc-A"
    )
    deleted = await lineage_store.gc_relation_if_orphan("Linus Torvalds", "Linux", "created")
    assert deleted is False, "doc-B still cites; relation must not be GC'd"

    await lineage_store.remove_relation_lineage_member(
        source="Linus Torvalds", target="Linux", type="created", document_id="doc-B"
    )
    deleted = await lineage_store.gc_relation_if_orphan("Linus Torvalds", "Linux", "created")
    assert deleted is True


@pytest.mark.asyncio
async def test_tenant_isolation_collection_id_filters_all_queries(lineage_store, request: pytest.FixtureRequest):
    """Case 6: §H.2 tenant double-layer — two store instances bound to
    different collection_ids must not see each other's rows even when
    both write the same entity name. Confirms ``find_*_with_lineage``
    and ``get_*`` filter on the bound ``collection_id``."""

    backend = request.node.callspec.params["lineage_store"]
    other_cid = f"lineage_contract_other_{backend}_{uuid.uuid4().hex[:8]}"
    _, builder = _BACKEND_BUILDERS[backend]
    other_store, other_cleanup = await builder(other_cid)

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
        # Sanity — the other tenant CAN see its own row.
        fetched_other = await other_store.get_entity("Shared Entity")
        assert fetched_other is not None

        # The PRIMARY store (bound to a different collection_id) MUST
        # NOT see the other tenant's row.
        leaked = await lineage_store.get_entity("Shared Entity")
        assert leaked is None
        names = await lineage_store.find_entity_ids_with_lineage(document_id="other-doc")
        assert names == []
    finally:
        await other_cleanup()


# ---------------------------------------------------------------------
# Case 7 (architect msg=87e2b187 chunk 4c amendment): explicit cross-
# event-loop scenario. Pin acceptance lock item 3 ("async driver wire
# cross-event-loop verify; no asyncio.run near factory") at the
# contract layer rather than relying on driver-vendor lazy bind
# behaviour. Forward-prevent for Wave 3 evaluation cross-loop bug
# msg=e1f23258 if a future driver upgrade flips to eager bind in
# ``__init__``.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres", "neo4j", "nebula"])
async def test_cross_event_loop_construct_then_call(backend: str):
    """Construct the per-collection store via ``asyncio.to_thread`` —
    mirroring ``ProductionWorkerFactory.__call__`` which builds the
    worker on a thread but the worker's ``sync(...)`` then awaits on
    the orchestrator event loop. The store / driver must bind lazily
    on first use so the cross-thread-to-loop handoff is safe.

    A driver that eagerly binds inside ``__init__`` would surface here
    as ``RuntimeError: ... attached to a different loop`` or
    ``RuntimeError: Event loop is closed`` once we await the upsert.

    The test mirrors the production worker_factory flow exactly:
    1. Create the per-process backend client SYNC (no awaits — same as
       ``_postgres_async_engine_singleton`` / ``_neo4j_async_driver_singleton``
       / ``_nebula_pool_singleton``). Loop binding deferred to first use.
    2. Construct the per-collection adapter inside ``asyncio.to_thread``
       (sync ``__init__`` only — same as ``_build_lineage_graph_store``).
    3. Await the adapter's async methods on the orchestrator loop.
    """

    reachable, _ = _BACKEND_BUILDERS[backend]
    if not reachable:
        pytest.skip(f"backend={backend} unreachable; skipping cross-event-loop case")

    cid = f"lineage_xloop_{backend}_{uuid.uuid4().hex[:8]}"

    # Step 1 — sync client creation (no event loop binding yet).
    if backend == "postgres":
        from sqlalchemy.ext.asyncio import create_async_engine

        from aperag.indexing.graph_storage.postgres import PostgresLineageGraphStore

        client = create_async_engine(_POSTGRES_DSN)

        def _construct() -> Any:
            return PostgresLineageGraphStore(engine=client, collection_id=cid)

        async def _cleanup_xloop() -> None:
            from sqlalchemy import text as _text

            async with client.begin() as conn:
                await conn.execute(
                    _text("DELETE FROM aperag_lineage_entity WHERE collection_id = :c"),
                    {"c": cid},
                )
                await conn.execute(
                    _text("DELETE FROM aperag_lineage_relation WHERE collection_id = :c"),
                    {"c": cid},
                )
            await client.dispose()

    elif backend == "neo4j":
        from neo4j import AsyncGraphDatabase

        from aperag.indexing.graph_storage.neo4j import Neo4jLineageGraphStore

        client = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))

        def _construct() -> Any:
            return Neo4jLineageGraphStore(driver=client, collection_id=cid)

        async def _cleanup_xloop() -> None:
            async with client.session() as session:
                await session.run(
                    "MATCH (n) WHERE n.collection_id = $cid DETACH DELETE n",
                    cid=cid,
                )
            await client.close()

    else:  # nebula
        from nebula3.Config import Config as _NebulaConfig
        from nebula3.gclient.net import ConnectionPool

        from aperag.indexing.graph_storage.nebula import NebulaLineageGraphStore, _space_name

        hosts: list[tuple[str, int]] = []
        for raw_host in _NEBULA_HOSTS.split(","):
            host_part = raw_host.strip()
            if not host_part:
                continue
            host, _, port_str = host_part.partition(":")
            hosts.append((host, int(port_str or "9669")))
        pool = ConnectionPool()
        assert pool.init(hosts, _NebulaConfig())
        space_prefix = "aperag_lineage_test_xloop"
        client = pool

        def _construct() -> Any:
            return NebulaLineageGraphStore(
                pool=client,
                username=_NEBULA_USER,
                password=_NEBULA_PASS,
                collection_id=cid,
                entity_lock=InMemoryEntityLock(),
                space_prefix=space_prefix,
            )

        async def _cleanup_xloop() -> None:
            space = _space_name(space_prefix, cid)
            session = client.get_session(_NEBULA_USER, _NEBULA_PASS)
            try:
                session.execute(f"DROP SPACE IF EXISTS `{space}`")
            finally:
                session.release()
            client.close()

    # Step 2 — adapter constructor runs on a worker thread (no loop).
    store = await asyncio.to_thread(_construct)
    try:
        # Step 3 — first async op runs on the test event loop. The
        # driver MUST lazily bind to this loop, not the (non-existent)
        # builder-thread loop.
        await store.ensure_schema()
        record = EntityRecord(
            name="XLoop Entity",
            type="thing",
            description="cross-loop ok",
            source_chunk_ids=(),
        )
        await store.upsert_entity_with_lineage(
            record=record,
            lineage=_make_member(doc="doc-xloop", version="v1"),
        )
        fetched = await store.get_entity("XLoop Entity")
        assert fetched is not None
        assert fetched.source_lineage[0].document_id == "doc-xloop"
    finally:
        await _cleanup_xloop()
