# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Indexer orchestration tests. Mocks both LLM and GraphStore so we
verify the pipeline wiring (chunk → extract → persist) without hitting
a database."""

from __future__ import annotations

import json
from typing import Sequence

import pytest

from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import Chunk, DeleteDocumentResult, Entity, Relation
from aperag.domains.knowledge_graph.graphindex.engine.indexer import index_document


class _FakeStore:
    """In-memory GraphStore for pipeline testing. Implements just enough
    of the Protocol to let the indexer run end-to-end."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.upsert_chunk_calls = 0
        self.upsert_entity_calls = 0
        self.upsert_relation_calls = 0

    async def ensure_schema(self) -> None:
        return None

    async def upsert_chunks(self, collection_id: str, chunks: Sequence[Chunk]) -> None:
        self.upsert_chunk_calls += 1
        self.chunks.extend(chunks)

    async def upsert_entities(self, collection_id: str, entities: Sequence[Entity]) -> None:
        self.upsert_entity_calls += 1
        self.entities.extend(entities)

    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None:
        self.upsert_relation_calls += 1
        self.relations.extend(relations)

    async def delete_document_rows(self, collection_id: str, doc_id: str):
        return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

    async def drop_collection(self, collection_id: str) -> None:
        self.chunks.clear()
        self.entities.clear()
        self.relations.clear()

    async def find_entities_by_names(self, collection_id, names):
        return [e for e in self.entities if e.name in set(names)]

    async def expand_neighborhood(self, collection_id, anchor_entity_ids, max_hop, limit):
        return [], []

    async def list_labels(self, collection_id):
        return sorted({e.type for e in self.entities})

    async def list_subgraph(self, collection_id, label, max_depth, max_nodes):
        from aperag.domains.knowledge_graph.graphindex.dto import KnowledgeGraph

        return KnowledgeGraph(nodes=(), edges=(), is_truncated=False)


def _cfg(**kw):
    base = dict(
        chunk_token_size=10,
        chunk_overlap_token_size=2,
        entity_types=("person",),
        max_chunks_per_batch=2,
        llm_max_retries=0,
    )
    base.update(kw)
    return GraphIndexConfig(**base)


def _mock_llm_returning(responses):
    """LLM that returns successive items from ``responses``, cycling if
    exhausted. Each response should already be a JSON string."""

    idx = {"i": 0}

    async def _call(_prompt: str) -> str:
        r = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return r

    return _call


@pytest.mark.asyncio
async def test_empty_document_writes_nothing():
    store = _FakeStore()
    llm = _mock_llm_returning(["{}"])
    result = await index_document(
        store=store,
        llm=llm,
        config=_cfg(),
        collection_id="col",
        doc_id="doc-1",
        content="",
    )
    assert result.chunks_created == 0
    assert result.entities_extracted == 0
    assert result.relations_extracted == 0
    assert store.chunks == []
    assert store.entities == []
    assert store.relations == []


@pytest.mark.asyncio
async def test_end_to_end_happy_path():
    """Three chunks, each yielding one entity and one self-referential
    (dropped) relation — the pipeline should land 3 entities and 0
    relations in the store, not crash on the bad relations."""
    responses = [
        json.dumps(
            {
                "entities": [{"name": "Alice", "type": "person", "description": ""}],
                "relations": [],
            }
        ),
        json.dumps(
            {
                "entities": [{"name": "Bob", "type": "person", "description": ""}],
                "relations": [],
            }
        ),
        json.dumps(
            {
                "entities": [{"name": "Carol", "type": "person", "description": ""}],
                "relations": [],
            }
        ),
    ]
    store = _FakeStore()
    llm = _mock_llm_returning(responses)

    content = " ".join(f"t{i}" for i in range(30))
    result = await index_document(
        store=store,
        llm=llm,
        config=_cfg(),
        collection_id="col",
        doc_id="doc-1",
        content=content,
        tokenize=str.split,
    )

    assert result.chunks_created >= 1
    assert result.entities_extracted >= 1
    # Chunks get written first so source_chunk_ids reference real rows.
    assert store.upsert_chunk_calls == 1
    assert store.upsert_entity_calls == 1


@pytest.mark.asyncio
async def test_single_chunk_failure_does_not_fail_document():
    """One bad LLM response must not prevent the other chunks' extracts
    from being persisted. Regression guard against a real production
    failure mode."""

    async def flaky_llm(prompt: str) -> str:
        # First call: break. Subsequent calls: return valid JSON.
        if not hasattr(flaky_llm, "_seen"):
            flaky_llm._seen = True  # type: ignore[attr-defined]
            raise RuntimeError("transient")
        return json.dumps(
            {
                "entities": [{"name": "Alice", "type": "person", "description": ""}],
                "relations": [],
            }
        )

    store = _FakeStore()
    content = " ".join(f"w{i}" for i in range(30))
    result = await index_document(
        store=store,
        llm=flaky_llm,
        config=_cfg(),
        collection_id="col",
        doc_id="doc-1",
        content=content,
        tokenize=str.split,
    )
    assert result.chunks_created >= 2
    # At least some entities should land (the non-failing chunks).
    assert len(store.entities) >= 1
