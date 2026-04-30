# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cross-backend ``LineageGraphStore`` compatibility tests.

Mirrors the legacy ``test_graph_compat.py`` pattern but targets the
new Wave 4 T8 + Wave 6 #33 ``LineageGraphStore`` Protocol
(``aperag/indexing/graph.py``) and its three production backends:

  - ``PostgresLineageGraphStore`` (``aperag/indexing/graph_storage/postgres.py``)
  - ``Neo4jLineageGraphStore``    (``aperag/indexing/graph_storage/neo4j.py``)
  - ``NebulaLineageGraphStore``   (``aperag/indexing/graph_storage/nebula.py``)

Each test runs the same deterministic scenario against every backend
that has a running instance. Backends are gated on env vars so CI can
selectively enable whichever databases it spins up — the same env vars
the legacy ``test_graph_compat.py`` already uses.

Env vars:
    COMPAT_PG_URL       = postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres
    COMPAT_NEO4J_URI    = bolt://127.0.0.1:7687
    COMPAT_NEO4J_USER   = neo4j
    COMPAT_NEO4J_PASS   = password
    COMPAT_NEBULA_HOSTS = 127.0.0.1:9669

Usage:
    # All available backends (skip when env var unset)
    pytest tests/integration/compat/test_lineage_graph_compat.py -v

    # Via make
    make test-compat-graph
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from aperag.graph_curation.lineage_merge import (
    CURATION_MERGE_DOCUMENT_ID,
    LineageEntityMerger,
)
from aperag.indexing.graph import (
    EntityRecord,
    InMemoryEntityLock,
    LineageMember,
    RelationRecord,
)

# --- backend factories ----------------------------------------------------


def _make_pg_store(collection_id: str):
    url = os.environ.get("COMPAT_PG_URL")
    if not url:
        return None
    from sqlalchemy.ext.asyncio import create_async_engine

    from aperag.indexing.graph_storage.postgres import PostgresLineageGraphStore

    engine = create_async_engine(url, future=True)
    store = PostgresLineageGraphStore(engine=engine, collection_id=collection_id)
    return store, engine


def _make_neo4j_store(collection_id: str):
    uri = os.environ.get("COMPAT_NEO4J_URI")
    if not uri:
        return None
    from neo4j import AsyncGraphDatabase

    from aperag.indexing.graph_storage.neo4j import Neo4jLineageGraphStore

    driver = AsyncGraphDatabase.driver(
        uri,
        auth=(
            os.environ.get("COMPAT_NEO4J_USER", "neo4j"),
            os.environ.get("COMPAT_NEO4J_PASS", "password"),
        ),
    )
    store = Neo4jLineageGraphStore(driver=driver, collection_id=collection_id)
    return store, driver


def _make_nebula_store(collection_id: str):
    hosts = os.environ.get("COMPAT_NEBULA_HOSTS")
    if not hosts:
        return None
    from nebula3.Config import Config as NebulaConfig
    from nebula3.gclient.net import ConnectionPool

    from aperag.indexing.graph_storage.nebula import NebulaLineageGraphStore

    pool = ConnectionPool()
    parsed = []
    for host_str in hosts.split(","):
        host, _, port = host_str.strip().partition(":")
        parsed.append((host, int(port) if port else 9669))
    pool.init(parsed, NebulaConfig())
    store = NebulaLineageGraphStore(
        pool=pool,
        username=os.environ.get("COMPAT_NEBULA_USER", "root"),
        password=os.environ.get("COMPAT_NEBULA_PASS", "nebula"),
        collection_id=collection_id,
        entity_lock=InMemoryEntityLock(),
    )
    return store, pool


_BACKENDS = {
    "postgresql": _make_pg_store,
    "neo4j": _make_neo4j_store,
    "nebula": _make_nebula_store,
}


def _available_backend_names():
    available = []
    if os.environ.get("COMPAT_PG_URL"):
        available.append(pytest.param("postgresql", id="postgresql"))
    if os.environ.get("COMPAT_NEO4J_URI"):
        available.append(pytest.param("neo4j", id="neo4j"))
    if os.environ.get("COMPAT_NEBULA_HOSTS"):
        available.append(pytest.param("nebula", id="nebula"))
    return available


_available = _available_backend_names()

if not _available:
    pytestmark = pytest.mark.skip(
        reason="No graph backend env vars set (COMPAT_PG_URL / COMPAT_NEO4J_URI / COMPAT_NEBULA_HOSTS)"
    )


# --- per-test fixtures ----------------------------------------------------


@pytest.fixture
def collection_id() -> str:
    # Per-test fresh collection id keeps fixtures isolated even when a
    # prior test leaks rows; backends bind to ``collection_id`` at
    # construction time and ``ensure_schema`` is idempotent.
    return f"compat_lineage_{uuid.uuid4().hex[:8]}"


@pytest.fixture(params=_available if _available else [pytest.param(None, marks=pytest.mark.skip)])
async def store(request, collection_id):
    """Yield a (backend_name, lineage_store) tuple. Provisions per-backend
    schema and tears down the per-test collection at exit.
    """
    name = request.param
    factory_result = _BACKENDS[name](collection_id)
    assert factory_result is not None, f"factory for backend {name!r} returned None despite env var set"
    store_obj, owned_resource = factory_result

    # All three production backends expose ``ensure_schema()`` for test
    # bring-up (alembic / DDL is the production path; tests just create
    # what they need on the fly).
    await store_obj.ensure_schema()

    yield name, store_obj

    # Cleanup. Each backend has its own teardown idiom; missing methods
    # are tolerated so the fixture is robust to backend-API drift.
    try:
        drop = getattr(store_obj, "drop_collection", None)
        if drop is not None:
            await drop()
    except Exception:
        pass
    try:
        close = getattr(store_obj, "close", None)
        if close is not None:
            res = close()
            # Some backends return a coroutine, others a None.
            if hasattr(res, "__await__"):
                await res
    except Exception:
        pass

    # Per-backend resource disposal: SQLAlchemy engine, Neo4j driver,
    # Nebula connection pool. These are caller-owned in production
    # too — the worker_factory caches them — so the fixture mirrors
    # that lifecycle here.
    if name == "postgresql":
        await owned_resource.dispose()
    elif name == "neo4j":
        await owned_resource.close()
    elif name == "nebula":
        owned_resource.close()


# --- shared deterministic test data ---------------------------------------

# Use stable lineage members so assertions are easy to read; the
# Protocol's contract is "dedup by (document_id, parse_version)" so
# we exercise both replace-on-same-key and add-on-distinct-key paths.

_LM_A_V1 = LineageMember(
    document_id="doc-A",
    parse_version="v1",
    tenant_scope_key="tenant-X",
    chunk_ids=("c1", "c2"),
)

_LM_A_V2 = LineageMember(
    document_id="doc-A",
    parse_version="v2",
    tenant_scope_key="tenant-X",
    chunk_ids=("c3",),
)

_LM_B_V1 = LineageMember(
    document_id="doc-B",
    parse_version="v1",
    tenant_scope_key="tenant-X",
    chunk_ids=("c10",),
)


def _entity(name: str, *, entity_type: str = "person", description: str = "desc") -> EntityRecord:
    return EntityRecord(
        name=name,
        entity_type=entity_type,
        description=description,
        source_chunk_ids=("c1",),
    )


def _relation(source: str, target: str, *, relation_type: str = "knows") -> RelationRecord:
    return RelationRecord(
        source=source,
        target=target,
        relation_type=relation_type,
        description=f"{source} -[{relation_type}]-> {target}",
        source_chunk_ids=("c1",),
    )


class _AliasRepoDouble:
    def __init__(self, *, raise_on_upsert: Exception | None = None) -> None:
        self.raise_on_upsert = raise_on_upsert
        self.resolve_calls: list[tuple[str, str]] = []
        self.upsert_calls: list[dict[str, Any]] = []

    async def resolve_canonical(self, *, collection_id: str, name: str) -> str:
        self.resolve_calls.append((collection_id, name))
        return name

    async def upsert_alias(
        self,
        *,
        collection_id: str,
        alias_name: str,
        target: str,
        merged_by: str | None = None,
    ) -> str:
        if self.raise_on_upsert is not None:
            raise self.raise_on_upsert
        self.upsert_calls.append(
            {
                "collection_id": collection_id,
                "alias_name": alias_name,
                "target": target,
                "merged_by": merged_by,
            }
        )
        return target


class _ForbiddenDescriptionCollaborator:
    """Fail fast if the description-free variant touches legacy surfaces."""

    async def compact_if_oversized(self, *_args: Any, **_kwargs: Any) -> str | None:
        raise AssertionError("description-free apply must not call GraphIndexCompactor")

    def embed_query(self, *_args: Any, **_kwargs: Any) -> list[float]:
        raise AssertionError("description-free apply must not embed description text")

    def upsert(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("description-free apply must not upsert vector points")

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("description-free apply must not delete vector points")


async def _forbidden_llm(_prompt: str) -> str:
    raise AssertionError("description-free apply must not call LLM")


def _description_free_merger(store, collection_id: str, alias_repo: _AliasRepoDouble | None = None):
    forbidden = _ForbiddenDescriptionCollaborator()
    return LineageEntityMerger(
        store=store,
        alias_repo=alias_repo or _AliasRepoDouble(),
        compactor=forbidden,
        vector_connector=forbidden,
        embedder=forbidden,
        llm=_forbidden_llm,
        collection_id=collection_id,
    )


def _lineage_keys(entity) -> set[tuple[str, str]]:
    return {(member.document_id, member.parse_version) for member in entity.source_lineage}


def _description_texts_by_key(entity) -> dict[tuple[str, str], str]:
    return {(part.document_id, part.parse_version): part.text for part in entity.description_parts}


# --- tests: lineage write + read round-trip -------------------------------


@pytest.mark.asyncio
async def test_upsert_entity_then_get_entity_round_trip(store, collection_id):
    """``upsert_entity_with_lineage`` followed by ``get_entity`` MUST
    return the canonical lineage view with the exact lineage member
    that was written."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)

    got = await s.get_entity("Alice")
    assert got is not None, "entity must exist after upsert"
    assert got.name == "Alice"
    assert got.entity_type == "person"
    assert any(lm.document_id == "doc-A" and lm.parse_version == "v1" for lm in got.source_lineage), (
        "source_lineage must contain the just-written member"
    )


@pytest.mark.asyncio
async def test_upsert_entity_dedup_by_doc_parse_version(store, collection_id):
    """Two upserts with the same ``(document_id, parse_version)`` MUST
    NOT produce two coexisting members — Protocol invariant
    (``LineageGraphStore`` docstring lines 425-428)."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Alice", description="desc-2"), lineage=_LM_A_V1)

    got = await s.get_entity("Alice")
    assert got is not None
    matching = [lm for lm in got.source_lineage if lm.document_id == "doc-A" and lm.parse_version == "v1"]
    assert len(matching) == 1, f"expected exactly one (doc-A, v1) member; got {len(matching)}"


@pytest.mark.asyncio
async def test_upsert_entity_distinct_parse_versions_coexist(store, collection_id):
    """Two upserts with different ``parse_version`` for the same
    ``document_id`` MUST coexist as separate members."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V2)

    got = await s.get_entity("Alice")
    assert got is not None
    versions = {lm.parse_version for lm in got.source_lineage if lm.document_id == "doc-A"}
    assert versions == {"v1", "v2"}


@pytest.mark.asyncio
async def test_upsert_relation_then_get_relation_round_trip(store, collection_id):
    """Symmetric to entity round-trip for relations."""

    _, s = store
    # Endpoints must exist before the relation upsert on graph backends
    # that enforce referential integrity (Postgres FK, Neo4j MERGE on
    # endpoint match). Nebula doesn't require it but the call is harmless.
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(record=_relation("Alice", "Bob"), lineage=_LM_A_V1)

    got = await s.get_relation("Alice", "Bob", "knows")
    assert got is not None, "relation must exist after upsert"
    assert (got.source, got.target, got.relation_type) == ("Alice", "Bob", "knows")
    assert any(lm.document_id == "doc-A" for lm in got.evidence_lineage)


# --- tests: lineage scan helpers (pre-rebuild phase) ----------------------


@pytest.mark.asyncio
async def test_find_entity_ids_with_lineage_returns_only_matching_doc(store, collection_id):
    """``find_entity_ids_with_lineage(document_id=X)`` MUST return
    exactly the entities whose lineage SET contains a member with
    ``document_id=X`` (any parse_version)."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V2)
    await s.upsert_entity_with_lineage(record=_entity("Carol"), lineage=_LM_B_V1)

    ids_for_a = await s.find_entity_ids_with_lineage(document_id="doc-A")
    assert sorted(ids_for_a) == ["Alice", "Bob"]

    ids_for_b = await s.find_entity_ids_with_lineage(document_id="doc-B")
    assert ids_for_b == ["Carol"]


@pytest.mark.asyncio
async def test_find_relation_keys_with_lineage_returns_matching_doc(store, collection_id):
    """Symmetric scan helper for relations."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Carol"), lineage=_LM_B_V1)
    await s.upsert_relation_with_lineage(record=_relation("Alice", "Bob"), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(record=_relation("Carol", "Bob"), lineage=_LM_B_V1)

    keys_for_a = await s.find_relation_keys_with_lineage(document_id="doc-A")
    assert ("Alice", "Bob", "knows") in keys_for_a
    assert ("Carol", "Bob", "knows") not in keys_for_a


# --- tests: lineage member removal + GC -----------------------------------


@pytest.mark.asyncio
async def test_remove_entity_lineage_member_clears_only_target_doc(store, collection_id):
    """``remove_entity_lineage_member`` MUST clear ALL members whose
    ``document_id`` matches (across any parse_version) and leave other
    documents' lineage intact."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V2)
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_B_V1)

    await s.remove_entity_lineage_member(entity_name="Alice", document_id="doc-A")

    got = await s.get_entity("Alice")
    assert got is not None, "entity row must persist; remove only clears lineage members"
    docs = {lm.document_id for lm in got.source_lineage}
    assert docs == {"doc-B"}, f"only doc-B lineage should remain; got {docs}"


@pytest.mark.asyncio
async def test_remove_relation_lineage_member_clears_only_target_doc(store, collection_id):
    """Task #61 P1-G1: relation lineage removal mirrors the entity
    rewrite invariant.

    Removing ``document_id=doc-A`` MUST clear every matching
    parse-version slice from both ``evidence_lineage`` and
    ``description_parts`` while preserving unrelated doc-B evidence.
    The relation row itself remains; GC is the separate orphan cleanup
    step tested below.
    """

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(
        record=_relation("Alice", "Bob", relation_type="knows"),
        lineage=_LM_A_V1,
    )
    await s.upsert_relation_with_lineage(
        record=_relation("Alice", "Bob", relation_type="knows"),
        lineage=_LM_A_V2,
    )
    await s.upsert_relation_with_lineage(
        record=_relation("Alice", "Bob", relation_type="knows"),
        lineage=_LM_B_V1,
    )

    await s.remove_relation_lineage_member(
        source="Alice",
        target="Bob",
        type="knows",
        document_id="doc-A",
    )

    got = await s.get_relation("Alice", "Bob", "knows")
    assert got is not None, "relation row must persist; remove only clears lineage members"
    assert {(member.document_id, member.parse_version) for member in got.evidence_lineage} == {("doc-B", "v1")}
    assert {(part.document_id, part.parse_version) for part in got.description_parts} == {("doc-B", "v1")}


@pytest.mark.asyncio
async def test_remove_relation_lineage_member_missing_relation_is_noop(store, collection_id):
    """Missing relation cleanup MUST be idempotent/no-op across
    backends. Rebuild scans call removal from scan results that can race
    with prior GC, so an absent edge must not raise."""

    _, s = store
    await s.remove_relation_lineage_member(
        source="Alice",
        target="Bob",
        type="knows",
        document_id="doc-A",
    )
    assert await s.get_relation("Alice", "Bob", "knows") is None


@pytest.mark.asyncio
async def test_gc_entity_if_orphan_deletes_only_when_lineage_empty(store, collection_id):
    """``gc_entity_if_orphan`` returns True iff a delete actually ran
    AND only when ``source_lineage`` is empty after upstream cleanup."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)

    # Lineage non-empty → GC must NOT run.
    deleted = await s.gc_entity_if_orphan("Alice")
    assert deleted is False
    assert await s.get_entity("Alice") is not None

    # Clear lineage, then GC should run.
    await s.remove_entity_lineage_member(entity_name="Alice", document_id="doc-A")
    deleted = await s.gc_entity_if_orphan("Alice")
    assert deleted is True
    assert await s.get_entity("Alice") is None


@pytest.mark.asyncio
async def test_gc_relation_if_orphan_deletes_only_when_lineage_empty(store, collection_id):
    """Symmetric GC contract for relations."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(record=_relation("Alice", "Bob"), lineage=_LM_A_V1)

    deleted = await s.gc_relation_if_orphan("Alice", "Bob", "knows")
    assert deleted is False

    await s.remove_relation_lineage_member(
        source="Alice",
        target="Bob",
        type="knows",
        document_id="doc-A",
    )
    deleted = await s.gc_relation_if_orphan("Alice", "Bob", "knows")
    assert deleted is True
    assert await s.get_relation("Alice", "Bob", "knows") is None


# --- tests: Wave 6 #33 retrieval query layer ------------------------------


@pytest.mark.asyncio
async def test_query_entities_by_keyword_returns_match(store, collection_id):
    """``query_entities_by_keyword`` MUST recall an entity whose name
    contains the query term (best-effort lexical recall, backend-defined)."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice Wonderland"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)

    hits = await s.query_entities_by_keyword(query="Alice", top_k=10)
    names = {h.name for h in hits}
    assert "Alice Wonderland" in names


@pytest.mark.asyncio
async def test_query_entities_by_keyword_empty_query_returns_nothing(store, collection_id):
    """Empty / whitespace-only query MUST return ``[]`` — no spurious
    recall (Protocol docstring lines 562-563)."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)

    assert await s.query_entities_by_keyword(query="", top_k=10) == []
    assert await s.query_entities_by_keyword(query="   ", top_k=10) == []


@pytest.mark.asyncio
async def test_query_entities_by_keyword_top_k_zero_returns_nothing(store, collection_id):
    """``top_k <= 0`` MUST return ``[]`` (Protocol docstring line 563)."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)

    assert await s.query_entities_by_keyword(query="Alice", top_k=0) == []


@pytest.mark.asyncio
async def test_expand_neighbors_n_hops_returns_seed_and_relations(store, collection_id):
    """``hops=1`` MUST return immediate neighbours plus connecting
    relations. The result UNION includes the seed entities themselves
    (Protocol docstring lines 575-578)."""

    _, s = store
    # Build a small graph: Alice -- knows -- Bob -- works_at -- Acme
    for n in ("Alice", "Bob", "Acme"):
        await s.upsert_entity_with_lineage(record=_entity(n), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(record=_relation("Alice", "Bob"), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(
        record=_relation("Bob", "Acme", relation_type="works_at"),
        lineage=_LM_A_V1,
    )

    entities, relations = await s.expand_neighbors_n_hops(entity_names=["Alice"], hops=1)
    names = {e.name for e in entities}
    assert "Alice" in names, "seed entity must be in result"
    assert "Bob" in names, "1-hop neighbour Bob must be in result"

    rel_keys = {(r.source, r.target, r.relation_type) for r in relations}
    assert ("Alice", "Bob", "knows") in rel_keys


@pytest.mark.asyncio
async def test_expand_neighbors_zero_hops_returns_only_seed(store, collection_id):
    """``hops <= 0`` MUST return just the seed entities (no relations),
    per Protocol docstring lines 583-585."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)
    await s.upsert_relation_with_lineage(record=_relation("Alice", "Bob"), lineage=_LM_A_V1)

    entities, relations = await s.expand_neighbors_n_hops(entity_names=["Alice"], hops=0)
    names = {e.name for e in entities}
    assert names == {"Alice"}
    assert relations == []


@pytest.mark.asyncio
async def test_expand_neighbors_empty_seed_returns_empty(store, collection_id):
    """Empty ``entity_names`` MUST return ``([], [])``
    (Protocol docstring line 586)."""

    _, s = store
    entities, relations = await s.expand_neighbors_n_hops(entity_names=[], hops=2)
    assert entities == []
    assert relations == []


# --- tests: description-parts accumulation --------------------------------


@pytest.mark.asyncio
async def test_description_parts_accumulate_across_distinct_parses(store, collection_id):
    """Per §D.3.3 Option A the read path must return the FULL set of
    ``DescriptionPart`` rows; two upserts with distinct
    ``(document_id, parse_version)`` keys MUST coexist as separate
    parts so the upstream LLM dedup / summarise step sees each
    document's contribution verbatim."""

    _, s = store
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="from doc-A v1"),
        lineage=_LM_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="from doc-A v2"),
        lineage=_LM_A_V2,
    )

    got = await s.get_entity("Alice")
    assert got is not None
    parts = {(p.document_id, p.parse_version): p.text for p in got.description_parts}
    assert parts.get(("doc-A", "v1")) == "from doc-A v1"
    assert parts.get(("doc-A", "v2")) == "from doc-A v2"


# --- tests: list_entity_labels (Wave 6 #40 narrow replacement) ------------


@pytest.mark.asyncio
async def test_list_entity_labels_returns_distinct_sorted(store, collection_id):
    """``list_entity_labels`` MUST return the sorted distinct
    ``EntityRecord.entity_type`` values present in this collection.
    Wave 6 #40 contract: collection-scoped per-instance store, no
    ``collection_id`` parameter at this layer."""

    _, s = store
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", entity_type="person"),
        lineage=_LM_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=_entity("Bob", entity_type="person"),
        lineage=_LM_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=_entity("Acme", entity_type="organization"),
        lineage=_LM_A_V1,
    )

    labels = await s.list_entity_labels()
    # Distinct and sorted (Wave 6 #40 contract).
    assert labels == sorted(set(labels)), "labels must be distinct + sorted"
    assert set(labels) == {"person", "organization"}


@pytest.mark.asyncio
async def test_list_entity_labels_empty_collection_returns_empty(store, collection_id):
    """Empty collection (no entities indexed yet) MUST return ``[]`` —
    per Protocol docstring "Returns ``[]`` when the collection has not
    been indexed yet — that is the *correct* answer, not a cue to fall
    back to the legacy graphindex path." (lines 594-596)."""

    _, s = store
    labels = await s.list_entity_labels()
    assert labels == []


@pytest.mark.asyncio
async def test_list_entity_labels_excludes_orphan_after_gc(store, collection_id):
    """After ``gc_entity_if_orphan`` deletes the only entity of a given
    type, ``list_entity_labels`` MUST no longer include that type.
    Pins the read path against a real backend GC interaction (the
    InMemory reference impl can't catch SQL/Cypher/nGQL-specific
    DISTINCT-after-DELETE issues)."""

    _, s = store
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", entity_type="person"),
        lineage=_LM_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=_entity("Acme", entity_type="organization"),
        lineage=_LM_A_V1,
    )
    assert set(await s.list_entity_labels()) == {"person", "organization"}

    # Clear lineage on Acme then GC — collection now has no orgs.
    await s.remove_entity_lineage_member(entity_name="Acme", document_id="doc-A")
    deleted = await s.gc_entity_if_orphan("Acme")
    assert deleted is True

    labels = await s.list_entity_labels()
    assert labels == ["person"]


# --- tests: task #61 P1-G2 list_entities stable capability ----------------


@pytest.mark.asyncio
async def test_list_entities_returns_stable_name_order_across_pages(store, collection_id):
    """Task #61 P1-G2: ``list_entities`` declares a stable name order
    capability so paged callers can concatenate pages without missing
    or duplicating rows."""

    _, s = store
    for name in ("Zara", "Alice", "Bob", "Aaron", "Carol"):
        await s.upsert_entity_with_lineage(record=_entity(name), lineage=_LM_A_V1)

    rows = await s.list_entities()
    names = [row.name for row in rows]
    assert names == ["Aaron", "Alice", "Bob", "Carol", "Zara"]

    page1 = await s.list_entities(limit=2, offset=0)
    page2 = await s.list_entities(limit=2, offset=2)
    page3 = await s.list_entities(limit=2, offset=4)
    assert [row.name for row in [*page1, *page2, *page3]] == names
    assert await s.list_entities(limit=2, offset=99) == []


@pytest.mark.asyncio
async def test_list_entities_label_filter_paginates_in_stable_name_order(store, collection_id):
    """Label-filtered pages use the same deterministic ``name`` order
    and exact ``entity_type`` filtering across all graph backends."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Zara", entity_type="person"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Acme", entity_type="organization"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob", entity_type="person"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Alice", entity_type="person"), lineage=_LM_A_V1)

    page1 = await s.list_entities(label="person", limit=2, offset=0)
    page2 = await s.list_entities(label="person", limit=2, offset=2)
    assert [row.name for row in page1] == ["Alice", "Bob"]
    assert [row.name for row in page2] == ["Zara"]
    assert [row.name for row in await s.list_entities(label="organization")] == ["Acme"]


@pytest.mark.asyncio
async def test_list_entities_limit_offset_edges(store, collection_id):
    """``limit <= 0`` returns an empty page and negative offsets are
    normalized to zero. These edges are part of the Protocol
    declaration, not backend-specific behaviour."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.upsert_entity_with_lineage(record=_entity("Bob"), lineage=_LM_A_V1)

    assert await s.list_entities(limit=0) == []
    assert await s.list_entities(limit=-5) == []
    rows = await s.list_entities(limit=1, offset=-10)
    assert [row.name for row in rows] == ["Alice"]


# --- tests: Wave 7 W7-1 ``compacted_description`` field --------------------
#
# These pin the cross-backend behaviour of the new W7-1 column added by
# the alembic migration (Postgres) / Neo4j property / Nebula tag-prop:
#
#   - default value is ``None`` after a Wave 4-style indexer upsert
#     (kwarg not supplied)
#   - explicit non-None kwarg writes through and is read back
#   - ``None`` kwarg on a subsequent upsert PRESERVES the existing
#     compacted value (the COALESCE invariant — the per-chunk indexer
#     hot path MUST NOT clobber a Compactor-written value)
#   - 100k-char roundtrip (huangheng safety gate: verifies driver
#     buffers don't truncate / corrupt at large sizes)
#   - lineage-preserve safety gate (huangheng msg=828c83cc 1131981c):
#     setting compacted via upsert MUST NOT clobber other lineage state
#   - ``delete_entity`` / ``delete_relation`` unconditional remove


@pytest.mark.asyncio
async def test_compacted_description_defaults_to_none_when_kwarg_omitted(store, collection_id):
    """A Wave 4-style upsert (no ``compacted_description`` kwarg)
    leaves the column at its default ``None``."""

    _, s = store
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="raw extracted text"),
        lineage=_LM_A_V1,
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.compacted_description is None


@pytest.mark.asyncio
async def test_compacted_description_round_trip_when_supplied(store, collection_id):
    """Explicit non-None kwarg writes through and reads back unchanged."""

    _, s = store
    summary = "Alice is a senior engineer working on graph indexing."
    await s.upsert_entity_with_lineage(
        record=_entity("Alice"),
        lineage=_LM_A_V1,
        compacted_description=summary,
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == summary


@pytest.mark.asyncio
async def test_compacted_description_preserved_on_subsequent_none_upsert(store, collection_id):
    """The COALESCE invariant: once the Compactor writes a non-None
    value, a subsequent Wave 4 indexer upsert (no kwarg → ``None``
    default) MUST preserve it. Without this contract every per-chunk
    upsert in the indexer hot path would clear the Compactor cache and
    force a re-summary on every doc add. This test pins the
    cross-backend behaviour (Postgres COALESCE, Neo4j COALESCE,
    Nebula in-Python preserve)."""

    _, s = store
    summary = "compacted v1 — synthesised by GraphIndexCompactor"
    await s.upsert_entity_with_lineage(
        record=_entity("Alice"),
        lineage=_LM_A_V1,
        compacted_description=summary,
    )
    # Indexer-side upsert with the same lineage key but no compacted
    # kwarg — emulates a re-sync of the same doc/parse_version.
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="re-extracted text"),
        lineage=_LM_A_V1,
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == summary  # preserved


@pytest.mark.asyncio
async def test_compacted_description_overwritten_on_subsequent_non_none_upsert(store, collection_id):
    """The Compactor refreshing a stale summary MUST overwrite the
    existing value. (Compactor passes a non-None kwarg deliberately.)"""

    _, s = store
    await s.upsert_entity_with_lineage(
        record=_entity("Alice"),
        lineage=_LM_A_V1,
        compacted_description="v1 summary",
    )
    await s.upsert_entity_with_lineage(
        record=_entity("Alice"),
        lineage=_LM_A_V1,
        compacted_description="v2 summary",
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.compacted_description == "v2 summary"


@pytest.mark.asyncio
async def test_compacted_description_supports_100k_chars_roundtrip(store, collection_id):
    """100k-char roundtrip safety gate (per huangheng msg=4d93a6c5
    gate #1). The Postgres ``Text`` / Neo4j ``STRING`` / Nebula
    ``string`` types are theoretically unbounded but driver-level
    buffers (asyncpg / neo4j-python-driver / nebula-python) have
    historically clipped large payloads silently. The application
    layer caps at 8000 chars (per spec §K.12.5), but the schema layer
    MUST tolerate the 12.5× margin — otherwise a Compactor bug that
    drops the cap could surface as a database-level corruption rather
    than an obvious truncation. This test pins the schema-side margin
    cross-backend."""

    _, s = store
    big = "x" * 100_000
    await s.upsert_entity_with_lineage(
        record=_entity("Bob"),
        lineage=_LM_A_V1,
        compacted_description=big,
    )
    got = await s.get_entity("Bob")
    assert got is not None
    assert got.compacted_description is not None
    assert len(got.compacted_description) == 100_000
    assert got.compacted_description == big


@pytest.mark.asyncio
async def test_compacted_description_does_not_clobber_lineage_on_compactor_write(store, collection_id):
    """huangheng safety gate (msg=828c83cc): writing
    ``compacted_description`` via the existing
    ``upsert_*_with_lineage`` kwarg path MUST preserve every other
    lineage field on the row. This is the principal risk of Option B
    (chosen over Option A's separate ``set_compacted_description``
    method): a Compactor that mis-constructs the EntityRecord could
    silently drop ``description_parts`` from other (doc, parse_v)
    slices. The test seeds multi-doc lineage, then simulates a
    Compactor write (re-passing the same lineage), and asserts ALL
    fields intact."""

    _, s = store
    # Seed two distinct (doc, parse_v) lineage slices with distinct
    # description text. The compactor will be invoked once, with
    # lineage A-v1; A-v2 must remain untouched.
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="from doc-A v1"),
        lineage=_LM_A_V1,
    )
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="from doc-A v2"),
        lineage=_LM_A_V2,
    )
    pre = await s.get_entity("Alice")
    assert pre is not None
    pre_lineage = sorted(pre.source_lineage, key=lambda m: (m.document_id, m.parse_version))
    pre_parts = sorted(pre.description_parts, key=lambda p: (p.document_id, p.parse_version))
    assert {(m.document_id, m.parse_version) for m in pre_lineage} == {("doc-A", "v1"), ("doc-A", "v2")}

    # Compactor write — passes the EXISTING (doc-A, v1) record (text
    # unchanged) plus a compacted_description. After this, both
    # lineage slices MUST still be present, neither description_parts
    # text has changed, and compacted_description is set.
    await s.upsert_entity_with_lineage(
        record=_entity("Alice", description="from doc-A v1"),
        lineage=_LM_A_V1,
        compacted_description="LLM-merged summary across both v1 and v2",
    )
    after = await s.get_entity("Alice")
    assert after is not None
    after_lineage = sorted(after.source_lineage, key=lambda m: (m.document_id, m.parse_version))
    after_parts = sorted(after.description_parts, key=lambda p: (p.document_id, p.parse_version))
    assert after_lineage == pre_lineage
    # description_parts text values for v1/v2 must be unchanged.
    assert {(p.document_id, p.parse_version, p.text) for p in after_parts} == {
        (p.document_id, p.parse_version, p.text) for p in pre_parts
    }
    assert after.compacted_description == "LLM-merged summary across both v1 and v2"


@pytest.mark.asyncio
async def test_compacted_description_relation_round_trip(store, collection_id):
    """Symmetric coverage for relations — the W7-1 schema is added on
    both ``aperag_lineage_entity`` and ``aperag_lineage_relation``."""

    _, s = store
    await s.upsert_relation_with_lineage(
        record=_relation("Alice", "Bob"),
        lineage=_LM_A_V1,
        compacted_description="Alice and Bob co-authored the paper.",
    )
    got = await s.get_relation("Alice", "Bob", "knows")
    assert got is not None
    assert got.compacted_description == "Alice and Bob co-authored the paper."


# --- tests: Wave 7 W7-1 ``delete_entity`` / ``delete_relation`` ------------


@pytest.mark.asyncio
async def test_delete_entity_unconditionally_removes(store, collection_id):
    """``delete_entity`` removes the row regardless of lineage state.
    Distinguishes from ``gc_entity_if_orphan`` which only deletes when
    ``source_lineage`` is empty."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    deleted = await s.delete_entity("Alice")
    assert deleted is True
    assert await s.get_entity("Alice") is None
    # Idempotent: a second delete returns False (row no longer exists).
    again = await s.delete_entity("Alice")
    assert again is False


@pytest.mark.asyncio
async def test_delete_entity_returns_false_when_absent(store, collection_id):
    """``delete_entity`` is idempotent: returns False when the row was
    not present to begin with."""

    _, s = store
    deleted = await s.delete_entity("DoesNotExist")
    assert deleted is False


@pytest.mark.asyncio
async def test_delete_relation_unconditionally_removes(store, collection_id):
    _, s = store
    await s.upsert_relation_with_lineage(record=_relation("Alice", "Bob"), lineage=_LM_A_V1)
    deleted = await s.delete_relation("Alice", "Bob", "knows")
    assert deleted is True
    assert await s.get_relation("Alice", "Bob", "knows") is None


@pytest.mark.asyncio
async def test_delete_relation_returns_false_when_absent(store, collection_id):
    _, s = store
    deleted = await s.delete_relation("Alice", "Bob", "knows")
    assert deleted is False


# --- task #61 P0 — bulk_upsert_entity_with_lineage_parts cross-backend -----
#
# This Protocol method (Wave 8 W8-2) was previously not exercised by any
# cross-backend test (冬柏 task #67 audit msg=3e93bb64). Bulk write paths
# are exactly where backend differences emerge: batch limits / atomicity /
# error handling / dedup contract. PM @不穷 elevated this to P0 in
# msg=10b753e8. The Protocol contract (`aperag/indexing/graph.py:575+`):
#   * All record.name MUST share the same string — raise ValueError otherwise.
#   * Empty parts is a no-op.
#   * Per-part record.entity_type follows last-wins.
#   * compacted_description is intentionally NOT a parameter.
#   * Same dedup-by-(document_id, parse_version) key as single upsert.
#   * Forward-only retry safety: per-part dedup so replays are idempotent.


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_empty_is_noop(store, collection_id):
    """Per Protocol contract `empty parts is a no-op`. The store MUST
    not raise and MUST not create any row."""

    _, s = store
    await s.bulk_upsert_entity_with_lineage_parts(parts=[])
    # Round-trip: get_entity should return None for a name that was never
    # written — the empty bulk must not implicitly create any entity.
    assert await s.get_entity("Alice") is None


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_rejects_mixed_names(store, collection_id):
    """Per Protocol contract `all record.name values MUST share the same
    string — backends MAY assert and raise ValueError if they don't`.
    All 3 production backends MUST raise to keep the bulk path's atomic
    guarantee honest (a mixed-name bulk would silently fan out to N
    different entity rows — atomicity meaningless)."""

    _, s = store
    with pytest.raises(ValueError):
        await s.bulk_upsert_entity_with_lineage_parts(
            parts=[
                (_entity("Alice"), _LM_A_V1),
                (_entity("Bob"), _LM_B_V1),
            ],
        )
    # Per @huangheng msg=99b5ffd5 + @ziang msg=84f5c3cc NIT — Lesson
    # #12 v6.4 (aggregation chain): a backend that raised AFTER writing
    # the first row would silently leak partial state. Pin
    # "raise-then-zero-side-effect" so any backend that swaps validation
    # order regresses loudly.
    assert await s.get_entity("Alice") is None, "raise must occur before any row write"
    assert await s.get_entity("Bob") is None, "raise must occur before any row write"


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_round_trip(store, collection_id):
    """Bulk write 3 distinct (document_id, parse_version) parts for the
    same entity name; all 3 lineage members + description parts must be
    visible after a single round-trip."""

    _, s = store
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_entity("Alice", description="from-doc-A-v1"), _LM_A_V1),
            (_entity("Alice", description="from-doc-A-v2"), _LM_A_V2),
            (_entity("Alice", description="from-doc-B-v1"), _LM_B_V1),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    keys = {(lm.document_id, lm.parse_version) for lm in got.source_lineage}
    assert keys == {
        ("doc-A", "v1"),
        ("doc-A", "v2"),
        ("doc-B", "v1"),
    }, f"all 3 (document_id, parse_version) members must be visible after bulk; got {keys}"
    # Per @huangzhangshu msg=5bbc5d1a — description_parts text must
    # round-trip alongside lineage keys; a backend that wrote the
    # lineage member but dropped the description text would silently
    # break agent context retrieval.
    parts_by_key = {(p.document_id, p.parse_version): p.text for p in got.description_parts}
    assert parts_by_key[("doc-A", "v1")] == "from-doc-A-v1"
    assert parts_by_key[("doc-A", "v2")] == "from-doc-A-v2"
    assert parts_by_key[("doc-B", "v1")] == "from-doc-B-v1"


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_dedup_last_wins_within_bulk(store, collection_id):
    """Per Protocol contract `parts sharing the same key collapse
    last-wins`. Two parts in the same bulk with the same
    (document_id, parse_version) MUST collapse to one row, with the
    second part's description winning."""

    _, s = store
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_entity("Alice", description="first-write"), _LM_A_V1),
            (_entity("Alice", description="last-write"), _LM_A_V1),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    matching = [lm for lm in got.source_lineage if lm.document_id == "doc-A" and lm.parse_version == "v1"]
    assert len(matching) == 1, f"same-key parts must collapse to one member; got {len(matching)}"
    # Per @huangzhangshu msg=5bbc5d1a — last-wins is on description
    # text not just lineage member. A backend that collapsed to one
    # member but kept the FIRST description text would silently break
    # the contract.
    parts_by_key = {(p.document_id, p.parse_version): p.text for p in got.description_parts}
    assert parts_by_key[("doc-A", "v1")] == "last-write", (
        f"same-key dedup MUST keep last description text; got {parts_by_key.get(('doc-A', 'v1'))!r}"
    )


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_replaces_existing_same_key(store, collection_id):
    """An existing single upsert with key (doc-A, v1) must be replaced
    when a subsequent bulk write contains the same key — same dedup
    contract as single upsert. Per Protocol: bulk strips existing rows
    whose key matches any incoming part, then appends the new parts."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice", description="single"), lineage=_LM_A_V1)
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[(_entity("Alice", description="bulk"), _LM_A_V1)],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    matching = [lm for lm in got.source_lineage if lm.document_id == "doc-A" and lm.parse_version == "v1"]
    assert len(matching) == 1, "single + bulk on same key must collapse to one member"
    # Per @huangzhangshu msg=5bbc5d1a — bulk's strip-then-append MUST
    # keep the bulk-side description text (not the prior single
    # upsert's). A backend that kept the single-side text would silently
    # fail the strip semantics.
    parts_by_key = {(p.document_id, p.parse_version): p.text for p in got.description_parts}
    assert parts_by_key[("doc-A", "v1")] == "bulk", (
        f"bulk write MUST overwrite single-write description; got {parts_by_key.get(('doc-A', 'v1'))!r}"
    )


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_appends_distinct_keys(store, collection_id):
    """An existing single upsert with key (doc-A, v1) must coexist with
    a subsequent bulk write containing distinct keys — bulk must NOT
    wipe unrelated lineage members. This is the cross-backend dedup
    invariant the Protocol pins."""

    _, s = store
    await s.upsert_entity_with_lineage(record=_entity("Alice"), lineage=_LM_A_V1)
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_entity("Alice"), _LM_A_V2),
            (_entity("Alice"), _LM_B_V1),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    keys = {(lm.document_id, lm.parse_version) for lm in got.source_lineage}
    assert keys == {
        ("doc-A", "v1"),
        ("doc-A", "v2"),
        ("doc-B", "v1"),
    }, f"bulk with distinct keys MUST NOT wipe pre-existing lineage; got {keys}"


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_entity_type_last_wins(store, collection_id):
    """Per Protocol contract `per-part record.entity_type follows the
    single-upsert "most recently observed value wins" rule (last
    tuple's type is the post-write entity_type for the row)`. The
    final row's entity_type MUST match the last bulk part's type."""

    _, s = store
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[
            (_entity("Alice", entity_type="person"), _LM_A_V1),
            (_entity("Alice", entity_type="researcher"), _LM_A_V2),
        ],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    assert got.entity_type == "researcher", f"last bulk part's entity_type wins; got {got.entity_type}"


@pytest.mark.asyncio
async def test_bulk_upsert_entity_with_lineage_parts_replay_is_idempotent(store, collection_id):
    """Per Protocol contract `Forward-only retry safety: the dedup key
    is per-part, so a mid-flight crash + retry replays each tuple
    idempotently`. Two consecutive bulk calls with the same key MUST
    collapse to one lineage member (idempotent) AND the second call's
    description MUST overwrite the first (last-wins on replay).

    Per @huangheng msg=99b5ffd5 + @ziang msg=84f5c3cc NIT — replay
    safety is the critical retry-correctness invariant; a backend that
    appended on replay (instead of dedup-then-replace) would silently
    duplicate lineage members under retry.
    """

    _, s = store
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[(_entity("Alice", description="v1-first"), _LM_A_V1)],
    )
    await s.bulk_upsert_entity_with_lineage_parts(
        parts=[(_entity("Alice", description="v1-replay"), _LM_A_V1)],
    )
    got = await s.get_entity("Alice")
    assert got is not None
    matching = [lm for lm in got.source_lineage if lm.document_id == "doc-A" and lm.parse_version == "v1"]
    assert len(matching) == 1, f"replay must be idempotent (no duplicate member); got {len(matching)}"
    # Per @huangzhangshu msg=5bbc5d1a — replay safety is not just about
    # member dedup; the replay's description text MUST overwrite the
    # first call's (last-wins on replay). A backend that kept the first
    # text would silently fail forward-progress on retry.
    parts_by_key = {(p.document_id, p.parse_version): p.text for p in got.description_parts}
    assert parts_by_key[("doc-A", "v1")] == "v1-replay", (
        f"replay MUST overwrite description (last-wins); got {parts_by_key.get(('doc-A', 'v1'))!r}"
    )


# --- task #31 B2 — description-free LineageEntityMerger cross-backend -----
#
# task #31 A3 introduced ``LineageEntityMerger.merge_entities_apply_description_free``
# as the async accept-apply path. This is application-layer logic built from
# ``LineageGraphStore`` primitives, so the cross-backend contract belongs here
# (not on a nonexistent ``LineageGraphStore.merge_entities`` method).
#
# Contract pinned across PostgreSQL / Neo4j / Nebula:
#   * source lineage members are re-anchored under the canonical target
#     without copying stale source description text;
#   * sources are deleted from L1, but the vector layer is untouched;
#   * no ``__curation_merge__`` sentinel lineage is written;
#   * replay after the sources are already consumed is idempotent;
#   * failures before graph-store writes leave entity rows unchanged.


@pytest.mark.asyncio
async def test_lineage_entity_merger_description_free_apply_reanchors_lineage_cross_backend(store, collection_id):
    """Description-free accept-apply preserves source lineage without
    leaking source descriptions or writing the legacy curation sentinel.

    This is the cross-backend test requested by task #31 § 3.1.4 /
    § 5.2.b and Bryce's P1 gap: it exercises
    ``LineageEntityMerger`` behaviour over the real backend stores,
    while replacing LLM / Compactor / vector collaborators with
    fail-fast doubles so any description-bearing legacy call regresses
    loudly.
    """

    _, s = store
    alias_repo = _AliasRepoDouble()
    merger = _description_free_merger(s, collection_id, alias_repo)

    target_lineage = LineageMember(
        document_id="target-doc",
        parse_version="v1",
        tenant_scope_key="tenant-X",
        chunk_ids=("target-c1",),
    )
    source_a_v1 = LineageMember(
        document_id="source-doc-a",
        parse_version="v1",
        tenant_scope_key="tenant-X",
        chunk_ids=("source-a-c1",),
    )
    source_a_v2 = LineageMember(
        document_id="source-doc-a",
        parse_version="v2",
        tenant_scope_key="tenant-X",
        chunk_ids=("source-a-c2",),
    )
    source_b_v1 = LineageMember(
        document_id="source-doc-b",
        parse_version="v1",
        tenant_scope_key="tenant-X",
        chunk_ids=("source-b-c1", "source-b-c2"),
    )

    await s.upsert_entity_with_lineage(
        record=EntityRecord(
            name="Apple Inc.",
            entity_type="organization",
            description="",
            source_chunk_ids=("target-c1",),
        ),
        lineage=target_lineage,
        compacted_description="legacy-target-summary",
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(
            name="Apple",
            entity_type="organization",
            description="SHOULD_NOT_LEAK_source_a_v1",
            source_chunk_ids=("source-a-c1",),
        ),
        lineage=source_a_v1,
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(
            name="Apple",
            entity_type="organization",
            description="SHOULD_NOT_LEAK_source_a_v2",
            source_chunk_ids=("source-a-c2",),
        ),
        lineage=source_a_v2,
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(
            name="AAPL",
            entity_type="organization",
            description="SHOULD_NOT_LEAK_source_b_v1",
            source_chunk_ids=("source-b-c1", "source-b-c2"),
        ),
        lineage=source_b_v1,
    )

    result = await merger.merge_entities_apply_description_free(
        target_name="Apple Inc.",
        source_names=["Apple", "AAPL"],
        merged_by="reviewer-1",
    )

    assert result.final_target == "Apple Inc."
    assert result.merged_source_ids == ["Apple", "AAPL"]
    assert result.unified_description == ""
    assert result.compacted_description is None
    assert alias_repo.upsert_calls == [
        {
            "collection_id": collection_id,
            "alias_name": "Apple",
            "target": "Apple Inc.",
            "merged_by": "reviewer-1",
        },
        {
            "collection_id": collection_id,
            "alias_name": "AAPL",
            "target": "Apple Inc.",
            "merged_by": "reviewer-1",
        },
    ]

    target_after = await s.get_entity("Apple Inc.")
    assert target_after is not None
    assert await s.get_entity("Apple") is None
    assert await s.get_entity("AAPL") is None

    expected_keys = {
        ("target-doc", "v1"),
        ("source-doc-a", "v1"),
        ("source-doc-a", "v2"),
        ("source-doc-b", "v1"),
    }
    assert _lineage_keys(target_after) == expected_keys
    assert all(member.document_id != CURATION_MERGE_DOCUMENT_ID for member in target_after.source_lineage), (
        "description-free apply must not write the legacy __curation_merge__ sentinel lineage"
    )

    descriptions = _description_texts_by_key(target_after)
    assert descriptions[("source-doc-a", "v1")] == ""
    assert descriptions[("source-doc-a", "v2")] == ""
    assert descriptions[("source-doc-b", "v1")] == ""
    assert not any("SHOULD_NOT_LEAK" in text for text in descriptions.values())

    # Replay after the queue redelivers the same accept-apply payload:
    # source rows are already gone, so the method should not duplicate
    # target lineage or resurrect descriptions.
    await merger.merge_entities_apply_description_free(
        target_name="Apple Inc.",
        source_names=["Apple", "AAPL"],
        merged_by="reviewer-1",
    )
    replayed = await s.get_entity("Apple Inc.")
    assert replayed is not None
    assert _lineage_keys(replayed) == expected_keys
    replayed_descriptions = _description_texts_by_key(replayed)
    assert replayed_descriptions[("source-doc-a", "v1")] == ""
    assert replayed_descriptions[("source-doc-a", "v2")] == ""
    assert replayed_descriptions[("source-doc-b", "v1")] == ""


@pytest.mark.asyncio
async def test_lineage_entity_merger_description_free_apply_alias_failure_has_no_graph_side_effect(
    store,
    collection_id,
):
    """If alias validation fails before graph-store writes, backend
    entity rows must remain unchanged.

    This pins the zero-side-effect-on-raise edge of the application
    contract without requiring alias-map DB setup in every graph
    backend. The real alias repository owns its own transaction; this
    test proves the merger does not touch the backend store after an
    alias-layer exception.
    """

    _, s = store
    alias_repo = _AliasRepoDouble(raise_on_upsert=RuntimeError("alias cycle"))
    merger = _description_free_merger(s, collection_id, alias_repo)

    await s.upsert_entity_with_lineage(
        record=EntityRecord(
            name="Canonical",
            entity_type="organization",
            description="",
            source_chunk_ids=("target-c1",),
        ),
        lineage=LineageMember(
            document_id="target-doc",
            parse_version="v1",
            tenant_scope_key="tenant-X",
            chunk_ids=("target-c1",),
        ),
    )
    await s.upsert_entity_with_lineage(
        record=EntityRecord(
            name="Alias",
            entity_type="organization",
            description="SHOULD_STILL_EXIST",
            source_chunk_ids=("source-c1",),
        ),
        lineage=LineageMember(
            document_id="source-doc",
            parse_version="v1",
            tenant_scope_key="tenant-X",
            chunk_ids=("source-c1",),
        ),
    )

    with pytest.raises(RuntimeError, match="alias cycle"):
        await merger.merge_entities_apply_description_free(
            target_name="Canonical",
            source_names=["Alias"],
            merged_by="reviewer-1",
        )

    target_after = await s.get_entity("Canonical")
    source_after = await s.get_entity("Alias")
    assert target_after is not None
    assert source_after is not None
    assert _lineage_keys(target_after) == {("target-doc", "v1")}
    assert _lineage_keys(source_after) == {("source-doc", "v1")}
    assert _description_texts_by_key(source_after)[("source-doc", "v1")] == "SHOULD_STILL_EXIST"
