# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``GraphIndexService`` — the public facade.

These focus on *delegation* behavior: the facade should hand off to
store / engine without transforming data. Business logic that lives in
the engine (or the store) is covered in its own tests."""

from __future__ import annotations

from typing import Optional

import pytest

from aperag.graphindex import (
    DeleteDocumentResult,
    Entity,
    GraphContext,
    GraphIndexService,
    KnowledgeGraph,
    Relation,
)


class _StubStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.find_result: list[Entity] = []
        self.expand_result: tuple[list[Entity], list[Relation]] = ([], [])
        self.labels_result: list[str] = []
        self.subgraph_result = KnowledgeGraph(nodes=(), edges=(), is_truncated=False)
        self.delete_result: Optional[DeleteDocumentResult] = None

    async def ensure_schema(self) -> None:
        self.calls.append(("ensure_schema", (), {}))

    async def drop_collection(self, collection_id: str) -> None:
        self.calls.append(("drop_collection", (collection_id,), {}))

    async def upsert_chunks(self, collection_id, chunks):
        self.calls.append(("upsert_chunks", (collection_id, list(chunks)), {}))

    async def upsert_entities(self, collection_id, entities):
        self.calls.append(("upsert_entities", (collection_id, list(entities)), {}))

    async def upsert_relations(self, collection_id, relations):
        self.calls.append(("upsert_relations", (collection_id, list(relations)), {}))

    async def delete_document_rows(self, collection_id, doc_id):
        self.calls.append(("delete_document_rows", (collection_id, doc_id), {}))
        return self.delete_result or DeleteDocumentResult(
            doc_id=doc_id,
            chunks_removed=0,
            entities_removed=0,
            relations_removed=0,
        )

    async def find_entities_by_names(self, collection_id, names):
        self.calls.append(("find_entities_by_names", (collection_id, list(names)), {}))
        return list(self.find_result)

    async def expand_neighborhood(self, collection_id, anchor_entity_ids, max_hop, limit):
        self.calls.append(
            (
                "expand_neighborhood",
                (collection_id, list(anchor_entity_ids), max_hop, limit),
                {},
            )
        )
        return self.expand_result

    async def list_labels(self, collection_id):
        self.calls.append(("list_labels", (collection_id,), {}))
        return list(self.labels_result)

    async def list_subgraph(self, collection_id, label, max_depth, max_nodes):
        self.calls.append(("list_subgraph", (collection_id, label, max_depth, max_nodes), {}))
        return self.subgraph_result


async def _null_llm(_prompt: str) -> str:
    return "{}"


# ---------------------------------------------------------------------------
# read-path delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_labels_delegates_to_store():
    store = _StubStore()
    store.labels_result = ["person", "organization"]
    svc = GraphIndexService(store=store, llm=_null_llm)

    got = await svc.get_labels(collection_id="c")
    assert got == ["person", "organization"]
    assert ("list_labels", ("c",), {}) in store.calls


@pytest.mark.asyncio
async def test_get_knowledge_graph_passes_all_params():
    store = _StubStore()
    svc = GraphIndexService(store=store, llm=_null_llm)

    await svc.get_knowledge_graph(collection_id="c", label="person", max_depth=3, max_nodes=77)
    assert ("list_subgraph", ("c", "person", 3, 77), {}) in store.calls


@pytest.mark.asyncio
async def test_drop_collection_delegates():
    store = _StubStore()
    svc = GraphIndexService(store=store, llm=_null_llm)
    await svc.drop_collection(collection_id="c-42")
    assert ("drop_collection", ("c-42",), {}) in store.calls


@pytest.mark.asyncio
async def test_delete_document_returns_store_result():
    store = _StubStore()
    store.delete_result = DeleteDocumentResult(doc_id="d", chunks_removed=2, entities_removed=1, relations_removed=3)
    svc = GraphIndexService(store=store, llm=_null_llm)
    result = await svc.delete_document(collection_id="c", doc_id="d")
    assert result.chunks_removed == 2
    assert result.entities_removed == 1
    assert result.relations_removed == 3


# ---------------------------------------------------------------------------
# query_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_context_empty_query_returns_empty_context():
    """Empty queries are a real shape (caller didn't pre-trim). Must
    not hit the store unnecessarily and must return an empty result,
    not raise."""
    store = _StubStore()
    svc = GraphIndexService(store=store, llm=_null_llm)
    ctx = await svc.query_context(collection_id="c", query="")
    assert isinstance(ctx, GraphContext)
    assert ctx.text == ""
    # Did NOT call expand_neighborhood:
    assert all(c[0] != "expand_neighborhood" for c in store.calls)


@pytest.mark.asyncio
async def test_query_context_renders_entities_and_relations():
    store = _StubStore()
    alice = Entity(
        entity_id="e1",
        collection_id="c",
        name="Alice",
        type="person",
        description="A researcher",
    )
    bob = Entity(
        entity_id="e2",
        collection_id="c",
        name="Bob",
        type="person",
        description="A collaborator",
    )
    rel = Relation(
        collection_id="c",
        source_id="e1",
        target_id="e2",
        description="Works with",
        weight=7.0,
    )
    store.find_result = [alice]
    store.expand_result = ([alice, bob], [rel])

    svc = GraphIndexService(store=store, llm=_null_llm)
    ctx = await svc.query_context(collection_id="c", query="Alice")

    assert "Alice" in ctx.text
    assert "Bob" in ctx.text
    assert "Works with" in ctx.text
    assert len(ctx.entities) == 2
    assert len(ctx.relations) == 1
