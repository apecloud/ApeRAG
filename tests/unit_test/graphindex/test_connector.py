# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``GraphStoreAdaptor`` dispatch and the vector-recall
query pipeline in ``GraphIndexService``."""

from __future__ import annotations

import pytest

from aperag.domains.knowledge_graph.graphindex.storage.connector import GraphStoreAdaptor


def test_adaptor_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported"):
        GraphStoreAdaptor("redis", ctx={})


def test_adaptor_postgresql_requires_engine():
    with pytest.raises(ValueError, match="engine"):
        GraphStoreAdaptor("postgresql", ctx={})


def test_adaptor_postgresql_creates_store():
    """Smoke test: passing a mock engine should instantiate
    PostgresGraphStore without error."""
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    adaptor = GraphStoreAdaptor("postgresql", ctx={"engine": mock_engine})
    assert adaptor.store is not None
    assert adaptor.graph_db_type == "postgresql"


class TestVectorRecallQueryPipeline:
    """Test the vector-based anchor resolution in GraphIndexService.

    These tests use stubs for both the graph store and the vector
    connector, validating that the service correctly delegates to both
    and merges the results.
    """

    @pytest.fixture
    def service(self):
        from aperag.domains.knowledge_graph.graphindex import Entity, GraphIndexService, KnowledgeGraph
        from aperag.domains.knowledge_graph.graphindex.dto import Chunk, DeleteDocumentResult, MergeEntitiesResult

        class StubStore:
            def __init__(self):
                self.calls = []
                self.entities_by_name = {}

            async def ensure_schema(self):
                pass

            async def drop_collection(self, cid):
                pass

            async def upsert_chunks(self, cid, chunks):
                pass

            async def upsert_entities(self, cid, entities):
                pass

            async def upsert_relations(self, cid, relations):
                pass

            async def delete_document_rows(self, cid, doc_id):
                return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

            async def merge_entities(self, cid, *, target_entity_id, source_entity_ids):
                return MergeEntitiesResult(
                    target_entity_id=target_entity_id,
                    merged_source_ids=(),
                    description="",
                    source_chunk_ids=(),
                    edges_redirected=0,
                    edges_collapsed=0,
                )

            async def find_oversized_entities(self, cid, *, min_chars, min_fragments, limit=200):
                return []

            async def find_oversized_relations(self, cid, *, min_chars, min_fragments, limit=200):
                return []

            async def rewrite_entity_description(self, cid, eid, desc):
                pass

            async def rewrite_relation_description(self, cid, src, tgt, desc):
                pass

            async def find_entities_by_ids(self, collection_id, entity_ids):
                out = []
                for eid in entity_ids:
                    for e in self.entities_by_name.values():
                        if e.entity_id == eid:
                            out.append(e)
                return out

            async def find_entities_by_names(self, collection_id, names):
                self.calls.append(("find_entities_by_names", names))
                out = []
                for n in names:
                    if n in self.entities_by_name:
                        out.append(self.entities_by_name[n])
                return out

            async def expand_neighborhood(self, collection_id, anchor_entity_ids, max_hop, limit):
                entities = []
                for eid in anchor_entity_ids:
                    for e in self.entities_by_name.values():
                        if e.entity_id == eid:
                            entities.append(e)
                return entities, []

            async def list_labels(self, cid):
                return []

            async def list_subgraph(self, cid, label, max_depth, max_nodes):
                return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

            async def get_chunks_by_ids(self, collection_id, chunk_ids):
                return [
                    Chunk(
                        chunk_id=cid_,
                        doc_id="d1",
                        collection_id=collection_id,
                        order_in_doc=0,
                        text=f"Chunk text for {cid_}",
                    )
                    for cid_ in chunk_ids
                ]

        store = StubStore()
        alice = Entity(
            entity_id="e-alice",
            collection_id="c",
            name="Alice",
            type="person",
            description="A researcher",
            source_chunk_ids=("chunk1",),
        )
        store.entities_by_name["Alice"] = alice

        class StubVectorConnector:
            def __init__(self):
                self.search_calls = []

            def search(self, request):
                self.search_calls.append(request)
                from aperag.vectorstore.dto import SearchHit

                flt_val = None
                if hasattr(request, "flt") and request.flt:
                    flt_val = getattr(request.flt, "value", None)

                if flt_val == "graph_entity":
                    return [
                        SearchHit(
                            id="ge_e-alice",
                            score=0.95,
                            payload={
                                "indexer": "graph_entity",
                                "entity_id": "e-alice",
                                "entity_name": "Alice",
                            },
                        )
                    ]
                elif flt_val == "graph_relation":
                    return []
                return []

            def upsert(self, points):
                pass

            def delete(self, ids):
                pass

            def delete_by_filter(self, flt):
                pass

        vc = StubVectorConnector()

        async def embed_query(text):
            return [0.1] * 128

        async def embed_texts(texts):
            return [[0.1] * 128 for _ in texts]

        async def null_llm(_prompt):
            return "{}"

        svc = GraphIndexService(
            store=store,
            llm=null_llm,
            embed_query=embed_query,
            embed_texts=embed_texts,
            vector_connector=vc,
        )
        return svc, store, vc

    @pytest.mark.asyncio
    async def test_query_context_uses_vector_recall(self, service):
        """When embed_query + vector_connector are wired, the service
        must use them for entity recall instead of naive name-match."""
        svc, store, vc = service

        ctx = await svc.query_context(collection_id="c", query="Tell me about Alice")

        assert "Alice" in ctx.text
        assert len(vc.search_calls) == 2
        assert any("graph_entity" in str(c) for c in vc.search_calls)
        assert len(ctx.entities) >= 1

    @pytest.mark.asyncio
    async def test_query_context_includes_chunks(self, service):
        """Graph context must include rehydrated chunk text (the
        LightRAG 'Document Chunks' section)."""
        svc, store, vc = service

        ctx = await svc.query_context(collection_id="c", query="Tell me about Alice")

        assert len(ctx.chunks) >= 1
        assert "Document Chunks" in ctx.text

    @pytest.mark.asyncio
    async def test_query_context_fallback_when_no_vector(self):
        """When no vector connector is configured, the service falls
        back to name-match — must not crash."""
        from aperag.domains.knowledge_graph.graphindex import GraphIndexService, KnowledgeGraph
        from aperag.domains.knowledge_graph.graphindex.dto import DeleteDocumentResult

        class MinimalStore:
            async def ensure_schema(self):
                pass

            async def drop_collection(self, cid):
                pass

            async def upsert_chunks(self, cid, chunks):
                pass

            async def upsert_entities(self, cid, entities):
                pass

            async def upsert_relations(self, cid, relations):
                pass

            async def delete_document_rows(self, cid, doc_id):
                return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

            async def find_oversized_entities(self, cid, *, min_chars, min_fragments, limit=200):
                return []

            async def find_oversized_relations(self, cid, *, min_chars, min_fragments, limit=200):
                return []

            async def rewrite_entity_description(self, cid, eid, desc):
                pass

            async def rewrite_relation_description(self, cid, src, tgt, desc):
                pass

            async def find_entities_by_ids(self, collection_id, entity_ids):
                return []

            async def find_entities_by_names(self, collection_id, names):
                return []

            async def expand_neighborhood(self, collection_id, anchor_entity_ids, max_hop, limit):
                return [], []

            async def list_labels(self, collection_id):
                return []

            async def list_subgraph(self, collection_id, label, max_depth, max_nodes):
                return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

            async def get_chunks_by_ids(self, collection_id, chunk_ids):
                return []

            async def merge_entities(self, collection_id, *, target_entity_id, source_entity_ids):
                pass

        async def null_llm(_p):
            return "{}"

        svc = GraphIndexService(store=MinimalStore(), llm=null_llm)
        ctx = await svc.query_context(collection_id="c", query="anything")
        assert ctx.text == ""


class TestRelationOnlyRecall:
    """Regression: relation hits must resolve to real Entity objects
    via find_entities_by_ids, not be silently dropped."""

    @pytest.mark.asyncio
    async def test_relation_hit_resolves_to_entity(self):
        from aperag.domains.knowledge_graph.graphindex import Entity, GraphIndexService, KnowledgeGraph
        from aperag.domains.knowledge_graph.graphindex.dto import DeleteDocumentResult, MergeEntitiesResult
        from aperag.vectorstore.dto import SearchHit

        bob = Entity(entity_id="e-bob", collection_id="c", name="Bob", type="person", description="A person")

        class StoreWithBob:
            async def ensure_schema(self):
                pass

            async def drop_collection(self, cid):
                pass

            async def upsert_chunks(self, cid, chunks):
                pass

            async def upsert_entities(self, cid, entities):
                pass

            async def upsert_relations(self, cid, relations):
                pass

            async def delete_document_rows(self, cid, doc_id):
                return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

            async def merge_entities(self, cid, *, target_entity_id, source_entity_ids):
                return MergeEntitiesResult(
                    target_entity_id=target_entity_id,
                    merged_source_ids=(),
                    description="",
                    source_chunk_ids=(),
                    edges_redirected=0,
                    edges_collapsed=0,
                )

            async def find_oversized_entities(self, cid, *, min_chars, min_fragments, limit=200):
                return []

            async def find_oversized_relations(self, cid, *, min_chars, min_fragments, limit=200):
                return []

            async def rewrite_entity_description(self, cid, eid, desc):
                pass

            async def rewrite_relation_description(self, cid, src, tgt, desc):
                pass

            async def find_entities_by_ids(self, collection_id, entity_ids):
                return [bob] if "e-bob" in entity_ids else []

            async def find_entities_by_names(self, collection_id, names):
                return []

            async def expand_neighborhood(self, collection_id, anchor_entity_ids, max_hop, limit):
                if "e-bob" in anchor_entity_ids:
                    return [bob], []
                return [], []

            async def list_labels(self, cid):
                return []

            async def list_subgraph(self, cid, lbl, md, mn):
                return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

            async def get_chunks_by_ids(self, cid, chunk_ids):
                return []

        class RelationOnlyVectorConnector:
            def search(self, request):
                flt_val = getattr(request.flt, "value", None)
                if flt_val == "graph_entity":
                    return []
                elif flt_val == "graph_relation":
                    return [
                        SearchHit(
                            id="gr_e-bob_e-other",
                            score=0.9,
                            payload={
                                "indexer": "graph_relation",
                                "source_id": "e-bob",
                                "target_id": "e-other",
                            },
                        )
                    ]
                return []

            def upsert(self, points):
                pass

            def delete(self, ids):
                pass

            def delete_by_filter(self, flt):
                pass

        async def embed_query(text):
            return [0.1] * 128

        async def null_llm(_p):
            return "{}"

        svc = GraphIndexService(
            store=StoreWithBob(),
            llm=null_llm,
            embed_query=embed_query,
            vector_connector=RelationOnlyVectorConnector(),
        )
        ctx = await svc.query_context(collection_id="c", query="something about Bob")

        assert "Bob" in ctx.text, (
            "Relation-only recall must resolve the entity from the "
            "relation hit's source_id/target_id via find_entities_by_ids"
        )
        assert len(ctx.entities) >= 1
