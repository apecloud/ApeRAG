from types import SimpleNamespace

from aperag.domains.knowledge_graph.schemas import KnowledgeGraph
from aperag.domains.knowledge_graph.service import (
    GraphService,
    _adapt_lineage_entities,
    _adapt_lineage_relations,
)
from aperag.domains.retrieval.schemas import SearchResultItem
from aperag.indexing.graph import (
    DescriptionPart,
    EntityWithLineage,
    LineageMember,
    RelationWithLineage,
)


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


def test_lineage_adapter_exposes_public_graph_properties_only():
    """Wave 7 W7-10: pin the ``EntityWithLineage`` /
    ``RelationWithLineage`` → UI properties projection so the public
    ``KnowledgeGraph`` shape stays stable across the legacy-graphindex
    deletion (``_adapt_nodes`` / ``_adapt_edges`` removed; new
    ``_adapt_lineage_entities`` / ``_adapt_lineage_relations`` take
    over)."""
    lineage_member = LineageMember(
        document_id="doc-1",
        parse_version="v1",
        tenant_scope_key="tenant-X",
        chunk_ids=("chunk-1", "chunk-2"),
    )
    description_part = DescriptionPart(
        document_id="doc-1",
        parse_version="v1",
        text="Graph RAG product",
    )
    node = EntityWithLineage(
        name="ApeRAG",
        entity_type="product",
        source_lineage=(lineage_member,),
        description_parts=(description_part,),
    )
    edge_evidence = LineageMember(
        document_id="doc-1",
        parse_version="v1",
        tenant_scope_key="tenant-X",
        chunk_ids=("chunk-3",),
    )
    edge_part = DescriptionPart(document_id="doc-1", parse_version="v1", text="relates to")
    edge = RelationWithLineage(
        source="ApeRAG",
        target="entity-2",
        relation_type="related_to",
        evidence_lineage=(edge_evidence,),
        description_parts=(edge_part,),
    )

    service = GraphService.__new__(GraphService)
    graph = KnowledgeGraph.model_validate(
        service._to_ui_dict(
            _adapt_lineage_entities([node]),
            _adapt_lineage_relations([edge]),
            is_truncated=False,
        )
    )
    dumped = graph.model_dump(exclude_none=True)

    node_props = dumped["nodes"][0]["properties"]
    assert node_props == {
        "entity_id": "ApeRAG",
        "entity_name": "ApeRAG",
        "entity_type": "product",
        "description": "Graph RAG product",
        "source_chunk_count": 2,
    }

    edge_props = dumped["edges"][0]["properties"]
    assert edge_props == {
        "weight": 1.0,
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
