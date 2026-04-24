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

from aperag.domains.knowledge_graph.graphindex import (
    DeleteDocumentResult,
    Entity,
    GraphContext,
    GraphIndexService,
    IndexDocumentResult,
    KnowledgeGraph,
    Relation,
)
from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import DESCRIPTION_SEPARATOR, MergeEntitiesResult


class _StubStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.find_result: list[Entity] = []
        self.expand_result: tuple[list[Entity], list[Relation]] = ([], [])
        self.labels_result: list[str] = []
        self.subgraph_result = KnowledgeGraph(nodes=(), edges=(), is_truncated=False)
        self.delete_result: Optional[DeleteDocumentResult] = None
        # Normalization / merge surface.
        self.oversized_entities: list[Entity] = []
        self.oversized_relations: list[Relation] = []
        self.merge_result: Optional[MergeEntitiesResult] = None

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

    async def find_entities_by_ids(self, collection_id, entity_ids):
        self.calls.append(("find_entities_by_ids", (collection_id, list(entity_ids)), {}))
        return list(self.find_result)

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

    async def get_chunks_by_ids(self, collection_id, chunk_ids):
        from aperag.domains.knowledge_graph.graphindex.dto import Chunk

        return [
            Chunk(chunk_id=cid, doc_id="d", collection_id=collection_id, order_in_doc=0, text=f"text of {cid}")
            for cid in chunk_ids
        ]

    async def merge_entities(self, collection_id, *, target_entity_id, source_entity_ids):
        self.calls.append(("merge_entities", (collection_id, target_entity_id, list(source_entity_ids)), {}))
        assert self.merge_result is not None, "test must set merge_result before calling merge_entities"
        return self.merge_result

    async def find_oversized_entities(self, collection_id, *, min_chars, min_fragments, limit=200):
        self.calls.append(
            ("find_oversized_entities", (collection_id,), {"min_chars": min_chars, "min_fragments": min_fragments})
        )
        return list(self.oversized_entities)

    async def find_oversized_relations(self, collection_id, *, min_chars, min_fragments, limit=200):
        self.calls.append(
            ("find_oversized_relations", (collection_id,), {"min_chars": min_chars, "min_fragments": min_fragments})
        )
        return list(self.oversized_relations)

    async def rewrite_entity_description(self, collection_id, entity_id, description):
        self.calls.append(("rewrite_entity_description", (collection_id, entity_id, description), {}))

    async def rewrite_relation_description(self, collection_id, source_id, target_id, description):
        self.calls.append(("rewrite_relation_description", (collection_id, source_id, target_id, description), {}))


class _StubVectorConnector:
    def __init__(self) -> None:
        self.deleted_batches: list[list[str]] = []
        self.upsert_batches: list[list[str]] = []

    def delete(self, ids):
        self.deleted_batches.append(list(ids))

    def upsert(self, points):
        self.upsert_batches.append([p.id for p in points])


def _stub_store_factory() -> _StubStore:
    """Return a stub with the normalization/merge fields initialised so
    it can stand in anywhere the real store is expected."""
    store = _StubStore()
    store.oversized_entities = []
    store.oversized_relations = []
    return store


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


# ---------------------------------------------------------------------------
# write-path rebuild idempotency (blocker 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_document_wipes_existing_rows_before_extracting():
    """Consecutive ``index_document`` calls for the same ``doc_id`` must
    first delete the prior rows so ``source_chunk_ids`` does not grow
    unboundedly across rebuilds. The delete must happen **before** any
    upsert — checked by position in ``store.calls``."""
    store = _StubStore()
    svc = GraphIndexService(store=store, llm=_null_llm)

    await svc.index_document(collection_id="c", doc_id="d1", content="", file_path="")
    await svc.index_document(collection_id="c", doc_id="d1", content="", file_path="")

    delete_calls = [i for i, (name, *_) in enumerate(store.calls) if name == "delete_document_rows"]
    assert len(delete_calls) == 2, f"expected two delete calls, saw {store.calls}"
    # Each delete must target the right (collection_id, doc_id)
    for idx in delete_calls:
        assert store.calls[idx][1] == ("c", "d1")
    # Any upsert for this doc must occur after at least one delete
    upsert_names = {"upsert_chunks", "upsert_entities", "upsert_relations"}
    first_delete_idx = delete_calls[0]
    upsert_positions = [i for i, (name, *_) in enumerate(store.calls) if name in upsert_names]
    if upsert_positions:
        assert min(upsert_positions) > first_delete_idx


@pytest.mark.asyncio
async def test_index_document_rebuild_deletes_stale_shadow_vectors(monkeypatch):
    class _LifecycleStore(_StubStore):
        def __init__(self) -> None:
            super().__init__()
            self.entity_state = [
                Entity(
                    entity_id="old-e",
                    collection_id="c",
                    name="Old",
                    type="person",
                    description="old desc",
                ),
                Entity(
                    entity_id="other-e",
                    collection_id="c",
                    name="Other",
                    type="person",
                    description="other desc",
                ),
            ]
            self.relation_state = [
                Relation(
                    collection_id="c",
                    source_id="old-e",
                    target_id="other-e",
                    description="old relation",
                )
            ]

        async def find_oversized_entities(self, collection_id, *, min_chars, min_fragments, limit=200):
            self.calls.append(
                ("find_oversized_entities", (collection_id,), {"min_chars": min_chars, "min_fragments": min_fragments})
            )
            if min_chars == 0 and min_fragments == 0:
                return list(self.entity_state)
            return []

        async def find_oversized_relations(self, collection_id, *, min_chars, min_fragments, limit=200):
            self.calls.append(
                ("find_oversized_relations", (collection_id,), {"min_chars": min_chars, "min_fragments": min_fragments})
            )
            if min_chars == 0 and min_fragments == 0:
                return list(self.relation_state)
            return []

    store = _LifecycleStore()
    vector = _StubVectorConnector()

    async def embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    async def fake_index_document(*, store, llm, config, collection_id, doc_id, content, file_path):
        store.entity_state = [
            Entity(
                entity_id="new-e",
                collection_id=collection_id,
                name="New",
                type="person",
                description="new desc",
            ),
            Entity(
                entity_id="other-e",
                collection_id=collection_id,
                name="Other",
                type="person",
                description="other desc",
            ),
        ]
        store.relation_state = [
            Relation(
                collection_id=collection_id,
                source_id="new-e",
                target_id="other-e",
                description="new relation",
            )
        ]
        return IndexDocumentResult(doc_id=doc_id, chunks_created=1, entities_extracted=2, relations_extracted=1)

    import aperag.domains.knowledge_graph.graphindex.service as service_module

    monkeypatch.setattr(service_module, "index_document", fake_index_document)

    svc = GraphIndexService(store=store, llm=_null_llm, embed_texts=embed_texts, vector_connector=vector)
    await svc.index_document(collection_id="c", doc_id="d1", content="", file_path="")

    assert vector.deleted_batches == [["ge_old-e", "gr_old-e_other-e"]]
    upserted_ids = [point_id for batch in vector.upsert_batches for point_id in batch]
    assert "ge_new-e" in upserted_ids
    assert "ge_other-e" in upserted_ids
    assert "gr_new-e_other-e" in upserted_ids
    assert "ge_old-e" not in upserted_ids
    assert "gr_old-e_other-e" not in upserted_ids


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


# ---------------------------------------------------------------------------
# Normalization: LLM summarization of oversized descriptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_document_summarizes_oversized_entities_via_llm():
    """After writing a document, the service must ask the store for any
    entity whose description is above threshold and call the LLM to
    produce a compact summary, then persist the summary via
    ``rewrite_entity_description``.

    This is the v2 replacement for LightRAG's
    ``force_llm_summary_on_merge`` behaviour. Critically, the summary
    must come from the LLM, not from any client-side truncation — that
    would lose information, which is exactly the failure mode this test
    protects against.
    """
    store = _stub_store_factory()
    long_desc = DESCRIPTION_SEPARATOR.join(f"Fragment {i}: Alice does something." for i in range(10))
    store.oversized_entities = [
        Entity(entity_id="e1", collection_id="c", name="Alice", type="person", description=long_desc)
    ]

    captured_prompts: list[str] = []

    async def llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Alice is a person who has done many things across the document."

    svc = GraphIndexService(store=store, llm=llm, config=GraphIndexConfig(summarize_at_fragments=6))

    await svc.index_document(collection_id="c", doc_id="d", content="", file_path="")

    rewrites = [c for c in store.calls if c[0] == "rewrite_entity_description"]
    assert len(rewrites) == 1, f"expected one rewrite, saw: {store.calls}"
    rewritten = rewrites[0][1][2]
    assert "Alice" in rewritten
    assert DESCRIPTION_SEPARATOR not in rewritten, "summary must not contain raw fragments"
    # The LLM was asked with a prompt containing each fragment.
    assert captured_prompts, "LLM was not invoked — summarization must go through the LLM"
    assert "Fragment 0" in captured_prompts[0]


@pytest.mark.asyncio
async def test_summarization_falls_back_to_truncation_only_when_llm_fails():
    """If the LLM raises, the service must still keep the description
    bounded so the DB doesn't accumulate megabytes. The fallback is
    word-boundary truncation; it is explicitly marked so operators can
    find capped rows and treat them as degraded."""
    store = _stub_store_factory()
    long_desc = "A " * 5000  # 10000 chars, way above the cap
    store.oversized_entities = [
        Entity(entity_id="e1", collection_id="c", name="Alice", type="person", description=long_desc)
    ]

    async def failing_llm(_prompt: str) -> str:
        raise RuntimeError("simulated LLM outage")

    svc = GraphIndexService(
        store=store,
        llm=failing_llm,
        config=GraphIndexConfig(summarize_at_fragments=6, max_description_chars=4000),
    )
    await svc.index_document(collection_id="c", doc_id="d", content="", file_path="")

    rewrites = [c for c in store.calls if c[0] == "rewrite_entity_description"]
    assert len(rewrites) == 1
    written = rewrites[0][1][2]
    assert len(written) <= 4000
    assert "[truncated]" in written, "fallback must be clearly marked so operators can audit"


@pytest.mark.asyncio
async def test_index_document_skips_summary_when_no_oversized_rows():
    """Happy path: small document, no description grew beyond the
    threshold, no LLM calls beyond extraction. Guards against the
    summarization pass becoming an unconditional expense."""
    store = _stub_store_factory()
    # oversized_entities / oversized_relations are already empty

    llm_calls: list[str] = []

    async def llm(prompt: str) -> str:
        llm_calls.append(prompt)
        return "{}"

    svc = GraphIndexService(store=store, llm=llm)
    await svc.index_document(collection_id="c", doc_id="d", content="", file_path="")

    assert not any(c[0] == "rewrite_entity_description" for c in store.calls)
    assert not any(c[0] == "rewrite_relation_description" for c in store.calls)


# ---------------------------------------------------------------------------
# Merge entities (LLM-summarized description, not plain concat)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_summarizes_merged_description():
    """After the SQL merge produces a long patchwork description, the
    service must call the LLM to collapse it into one coherent
    paragraph BEFORE returning. The failure mode this protects against
    is "merged entity has a 10-fragment description that no user can
    read" — the specific regression the user flagged when reviewing
    the first rewrite attempt."""
    store = _stub_store_factory()
    patchwork = DESCRIPTION_SEPARATOR.join(f"Claim {i} about the entity." for i in range(8))
    store.merge_result = MergeEntitiesResult(
        target_entity_id="A",
        merged_source_ids=("B", "C"),
        description=patchwork,
        source_chunk_ids=("chunk1", "chunk2", "chunk3"),
        edges_redirected=3,
        edges_collapsed=1,
    )

    summary = "A is the merged entity and does these things."

    async def llm(prompt: str) -> str:
        assert "Claim 0" in prompt, "summarization prompt must contain the fragments verbatim"
        return summary

    svc = GraphIndexService(store=store, llm=llm, config=GraphIndexConfig(summarize_at_fragments=6))

    result = await svc.merge_entities(collection_id="c", target_entity_id="A", source_entity_ids=["B", "C"])

    assert result.description == summary
    # The summary must have been persisted.
    rewrites = [c for c in store.calls if c[0] == "rewrite_entity_description"]
    assert any(call[1] == ("c", "A", summary) for call in rewrites)


@pytest.mark.asyncio
async def test_merge_entities_skips_summary_on_short_description():
    """If the merged description is already small, skip the LLM round
    trip. Prevents the merge API from paying a summarization cost on
    every call, which would make the UI feel sluggish."""
    store = _stub_store_factory()
    store.merge_result = MergeEntitiesResult(
        target_entity_id="A",
        merged_source_ids=("B",),
        description="A short merged description.",
        source_chunk_ids=("chunk1",),
        edges_redirected=0,
        edges_collapsed=0,
    )

    async def llm(_prompt: str) -> str:
        raise AssertionError("LLM must not be called when description is already compact")

    svc = GraphIndexService(store=store, llm=llm, config=GraphIndexConfig(summarize_at_fragments=6))

    result = await svc.merge_entities(collection_id="c", target_entity_id="A", source_entity_ids=["B"])

    assert result.description == "A short merged description."
    assert not any(c[0] == "rewrite_entity_description" for c in store.calls)
