from types import SimpleNamespace

from aperag.domains.knowledge_graph.graphindex.dto import Entity, Relation
from aperag.domains.knowledge_graph.schemas import KnowledgeGraph
from aperag.domains.knowledge_graph.service import GraphService, _adapt_edges, _adapt_nodes
from aperag.domains.retrieval.schemas import SearchResultItem


def test_search_result_metadata_is_public_allowlist():
    item = SearchResultItem(
        rank=1,
        score=0.9,
        content="result",
        source="doc.pdf",
        recall_type="vision_search",
        metadata={
            "source": "doc.pdf",
            "name": "ignored-fallback.pdf",
            "title": "Document title",
            "collection_id": "col-1",
            "document_id": "doc-1",
            "asset_id": "asset-1",
            "mimetype": "image/png",
            "page_idx": "2",
            "url": "https://example.com/doc.pdf",
            "indexer": "vision",
            "index_method": "vision_to_text",
            "chat_id": "chat-1",
            "object_path": "internal/object/path",
            "_node_content": {"private": True},
        },
    )

    dumped = item.model_dump(exclude_none=True)

    assert dumped["metadata"] == {
        "source": "doc.pdf",
        "title": "Document title",
        "collection_id": "col-1",
        "document_id": "doc-1",
        "asset_id": "asset-1",
        "mimetype": "image/png",
        "page_idx": 2,
        "url": "https://example.com/doc.pdf",
        "modality": "image",
        # Wave 3 T3.2 (Bryce commit 5325788) §G.5 SearchResultMetadata
        # extension: ``index_modality`` is derived from the raw
        # ``indexer`` field via ``SearchResultMetadata.from_raw``.
        "index_modality": "vision",
    }


def test_graphindex_adapter_exposes_public_graph_properties_only():
    node = Entity(
        entity_id="entity-1",
        collection_id="col-1",
        name="ApeRAG",
        type="product",
        description="Graph RAG product",
        source_chunk_ids=("chunk-1", "chunk-2"),
    )
    edge = Relation(
        collection_id="col-1",
        source_id="entity-1",
        target_id="entity-2",
        description="relates to",
        weight=8,
        source_chunk_ids=("chunk-3",),
    )

    service = GraphService.__new__(GraphService)
    graph = KnowledgeGraph.model_validate(
        service._to_ui_dict(_adapt_nodes([node]), _adapt_edges([edge]), is_truncated=False)
    )
    dumped = graph.model_dump(exclude_none=True)

    node_props = dumped["nodes"][0]["properties"]
    assert node_props == {
        "entity_id": "entity-1",
        "entity_name": "ApeRAG",
        "entity_type": "product",
        "description": "Graph RAG product",
        "source_chunk_count": 2,
    }

    edge_props = dumped["edges"][0]["properties"]
    assert edge_props == {
        "weight": 8.0,
        "description": "relates to",
        "keywords": "",
        "source_chunk_count": 1,
    }


def test_legacy_graph_properties_are_sanitized_before_public_response():
    legacy_node = SimpleNamespace(
        id="entity-1",
        labels=["entity-1"],
        properties={
            "entity_id": "entity-1",
            "entity_name": "ApeRAG",
            "entity_type": "product",
            "description": "Graph RAG product",
            "source_id": "chunk-1,chunk-2",
            "file_path": "/private/doc.pdf",
            "created_at": 123,
        },
    )
    legacy_edge = SimpleNamespace(
        id="entity-1->entity-2",
        type="DIRECTED",
        source="entity-1",
        target="entity-2",
        properties={
            "weight": 3.0,
            "description": "relates to",
            "keywords": "graph",
            "source_id": "chunk-3",
            "file_path": "/private/doc.pdf",
            "created_at": 456,
        },
    )

    service = GraphService.__new__(GraphService)
    graph = KnowledgeGraph.model_validate(service._to_ui_dict([legacy_node], [legacy_edge], is_truncated=False))
    dumped = graph.model_dump(exclude_none=True)

    assert dumped["nodes"][0]["properties"] == {
        "entity_id": "entity-1",
        "entity_name": "ApeRAG",
        "entity_type": "product",
        "description": "Graph RAG product",
    }
    assert dumped["edges"][0]["properties"] == {
        "weight": 3.0,
        "description": "relates to",
        "keywords": "graph",
    }
