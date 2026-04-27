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

"""Wave 6 #33 — Protocol surface + in-memory implementation for the
LightRAG-style query layer on :class:`aperag.indexing.graph.LineageGraphStore`.

Pin the Protocol method signatures so chunk 3 caller migration can
rely on a stable surface, and pin the in-memory query/traversal
semantics that production backends must match.

Per architect ruling msg=54eac595 (Option A) the read API is:
* ``query_entities_by_keyword(query, top_k)`` — lexical recall
* ``expand_neighbors_n_hops(entity_names, hops)`` — graph traversal

``query_entities_by_vector`` was originally declared in chunk 1 but
dropped in chunk 2 because the Wave 4 lineage schema does not carry an
entity-vector column. Wave 7 may re-introduce vector recall via a
separate ``LightRAGQueryService`` if real evidence surfaces.
"""

from __future__ import annotations

import asyncio
import inspect

from aperag.indexing.graph import (
    EntityRecord,
    InMemoryLineageGraphStore,
    LineageGraphStore,
    LineageMember,
    RelationRecord,
)


def test_lineage_query_protocol_exposes_two_read_methods():
    """The new query layer surface must declare exactly the two
    LightRAG-style read methods agreed in §K.11.11 amend."""

    members = set(LineageGraphStore.__dict__)
    assert "query_entities_by_keyword" in members
    assert "expand_neighbors_n_hops" in members
    # Vector recall was removed by architect ruling msg=54eac595 —
    # no entity-vector column in Wave 4 lineage schema.
    assert "query_entities_by_vector" not in members


def test_query_by_keyword_signature_takes_query_and_top_k():
    sig = inspect.signature(LineageGraphStore.query_entities_by_keyword)
    params = sig.parameters
    assert "query" in params
    assert "top_k" in params
    # kw-only by spec — the retrieval pipeline always names the args.
    assert params["query"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["top_k"].kind is inspect.Parameter.KEYWORD_ONLY


def test_expand_neighbors_signature_takes_entity_names_and_hops():
    sig = inspect.signature(LineageGraphStore.expand_neighbors_n_hops)
    params = sig.parameters
    assert "entity_names" in params
    assert "hops" in params
    assert params["entity_names"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["hops"].kind is inspect.Parameter.KEYWORD_ONLY
    # Default hops=1 is the simple-stable-directive choice — caller
    # must opt into multi-hop explicitly.
    assert params["hops"].default == 1


def _seed_minimal_graph(store: InMemoryLineageGraphStore) -> None:
    """Three entities (Alice / Bob / Carol) connected by two
    relations: Alice -knows-> Bob, Bob -manages-> Carol. Used by the
    in-memory contract tests below."""

    async def _run() -> None:
        members = LineageMember(
            document_id="doc-1",
            parse_version="v1",
            tenant_scope_key="user:test",
            chunk_ids=("chunk-1",),
        )
        for name in ("Alice", "Bob", "Carol"):
            await store.upsert_entity_with_lineage(
                record=EntityRecord(
                    name=name,
                    entity_type="person",
                    description=f"{name} description",
                    source_chunk_ids=("chunk-1",),
                ),
                lineage=members,
            )
        await store.upsert_relation_with_lineage(
            record=RelationRecord(
                source="Alice",
                target="Bob",
                relation_type="knows",
                description="knows",
                source_chunk_ids=("chunk-1",),
            ),
            lineage=members,
        )
        await store.upsert_relation_with_lineage(
            record=RelationRecord(
                source="Bob",
                target="Carol",
                relation_type="manages",
                description="manages",
                source_chunk_ids=("chunk-1",),
            ),
            lineage=members,
        )

    asyncio.run(_run())


def test_in_memory_query_by_keyword_finds_substring_matches():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        hits = await store.query_entities_by_keyword(query="ali", top_k=5)
        assert [e.name for e in hits] == ["Alice"]

        hits = await store.query_entities_by_keyword(query="A", top_k=5)
        # Substring "A" matches "Alice" + "Carol"; sorted by name.
        assert [e.name for e in hits] == ["Alice", "Carol"]

    asyncio.run(_run())


def test_in_memory_query_by_keyword_empty_or_zero_top_k_returns_empty():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        assert await store.query_entities_by_keyword(query="", top_k=5) == []
        assert await store.query_entities_by_keyword(query="   ", top_k=5) == []
        assert await store.query_entities_by_keyword(query="A", top_k=0) == []

    asyncio.run(_run())


def test_in_memory_expand_neighbors_one_hop_includes_seed_and_neighbours():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        entities, relations = await store.expand_neighbors_n_hops(entity_names=["Alice"], hops=1)
        names = {e.name for e in entities}
        assert names == {"Alice", "Bob"}
        rel_keys = {(r.source, r.target, r.relation_type) for r in relations}
        assert rel_keys == {("Alice", "Bob", "knows")}

    asyncio.run(_run())


def test_in_memory_expand_neighbors_two_hops_walks_further():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        entities, relations = await store.expand_neighbors_n_hops(entity_names=["Alice"], hops=2)
        names = {e.name for e in entities}
        # 2-hop reaches Carol via Alice→Bob→Carol.
        assert names == {"Alice", "Bob", "Carol"}
        rel_keys = {(r.source, r.target, r.relation_type) for r in relations}
        assert rel_keys == {("Alice", "Bob", "knows"), ("Bob", "Carol", "manages")}

    asyncio.run(_run())


def test_in_memory_expand_neighbors_zero_hops_returns_seed_only():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        entities, relations = await store.expand_neighbors_n_hops(entity_names=["Bob"], hops=0)
        assert {e.name for e in entities} == {"Bob"}
        assert relations == []

    asyncio.run(_run())


def test_in_memory_expand_neighbors_empty_input_returns_empty_pair():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        result = await store.expand_neighbors_n_hops(entity_names=[], hops=1)
        assert result == ([], [])

    asyncio.run(_run())


def test_in_memory_expand_neighbors_unknown_seed_returns_empty_entities():
    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        entities, relations = await store.expand_neighbors_n_hops(entity_names=["Unknown"], hops=1)
        assert entities == []
        assert relations == []

    asyncio.run(_run())


# ---------------------------------------------------------------------
# Wave 6 #40 — ``list_entity_labels`` Protocol method (narrow
# replacement for the legacy ``GraphIndexService.list_labels`` UI path
# per architect ruling msg=3efdf906 / Option B). Pin the surface +
# in-memory semantics; backend integration tests cover the SQL/Cypher
# path against real engines.
# ---------------------------------------------------------------------


def test_list_entity_labels_protocol_method_exists():
    """The narrow-replacement Protocol method MUST exist on
    ``LineageGraphStore`` after Wave 6 #40 — pinned so future Protocol
    refactors can't silently drop it.
    """

    assert "list_entity_labels" in LineageGraphStore.__dict__


def test_list_entity_labels_signature_takes_no_args():
    """``LineageGraphStore`` is collection-scoped per-instance, so
    ``list_entity_labels`` does NOT take a ``collection_id`` argument
    (unlike the legacy ``GraphIndexService.list_labels(collection_id)``
    which was global).
    """

    sig = inspect.signature(LineageGraphStore.list_entity_labels)
    # Only ``self`` — no other parameters.
    assert list(sig.parameters) == ["self"]


def test_in_memory_list_entity_labels_returns_sorted_distinct_types():
    """Three entities (Alice/Bob/Carol) all of type ``person`` → one
    label. Then upsert one entity of a different type → two sorted
    labels.
    """

    store = InMemoryLineageGraphStore()
    _seed_minimal_graph(store)

    async def _run() -> None:
        labels = await store.list_entity_labels()
        assert labels == ["person"]

        # Add a new entity with a different type — list must update
        # to reflect the new distinct value, sorted lexicographically.
        await store.upsert_entity_with_lineage(
            record=EntityRecord(
                name="ACME Corp",
                entity_type="organization",
                description="company",
                source_chunk_ids=("chunk-1",),
            ),
            lineage=LineageMember(
                document_id="doc-1",
                parse_version="v1",
                tenant_scope_key="user:test",
                chunk_ids=("chunk-1",),
            ),
        )
        labels = await store.list_entity_labels()
        assert labels == ["organization", "person"]

    asyncio.run(_run())


def test_in_memory_list_entity_labels_returns_empty_for_fresh_store():
    """A collection that has not been indexed yet returns ``[]`` — that
    is the correct answer per docstring contract, NOT a cue to fall
    back to legacy graphindex.
    """

    store = InMemoryLineageGraphStore()

    async def _run() -> None:
        labels = await store.list_entity_labels()
        assert labels == []

    asyncio.run(_run())


def test_in_memory_list_entity_labels_skips_empty_type_values():
    """Defensive: an entity row whose ``type`` is the empty string
    must not pollute the dropdown — UI label list shows only
    meaningful values.
    """

    store = InMemoryLineageGraphStore()

    async def _run() -> None:
        await store.upsert_entity_with_lineage(
            record=EntityRecord(
                name="Mystery",
                entity_type="",
                description="no type",
                source_chunk_ids=("chunk-1",),
            ),
            lineage=LineageMember(
                document_id="doc-1",
                parse_version="v1",
                tenant_scope_key="user:test",
                chunk_ids=("chunk-1",),
            ),
        )
        labels = await store.list_entity_labels()
        assert labels == []

    asyncio.run(_run())
