from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aperag.domains.knowledge_graph.service import GraphService
from aperag.indexing.graph import LineageMember


def _entity(name: str, entity_type: str = "entity") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        entity_type=entity_type,
        compacted_description=None,
        description_parts=[],
        source_lineage=[],
    )


def _relation(source: str, target: str) -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        target=target,
        relation_type="related_to",
        compacted_description=None,
        description_parts=[],
        evidence_lineage=[],
    )


async def test_get_knowledge_graph_uses_store_one_hop_for_visualization(monkeypatch):
    subgraph_calls = []

    class FakeStore:
        async def list_entities(self, *, label, limit):
            assert label is None
            assert limit == 20
            return [_entity("A", "person"), _entity("B", "organization")]

        async def expand_neighbors_n_hops(self, *, entity_names, hops):
            assert entity_names == ["A", "B"]
            assert hops == 1
            subgraph_calls.append((tuple(entity_names), hops))
            return (
                [_entity("A", "person"), _entity("B", "organization")],
                [
                    _relation("A", "B"),
                    _relation("A", "outside"),
                ],
            )

    from aperag.indexing import worker_factory

    monkeypatch.setattr(worker_factory, "_resolve_graph_backend_type", lambda collection: "postgres")
    monkeypatch.setattr(
        worker_factory,
        "_build_lineage_graph_store",
        lambda *, backend_type, collection: FakeStore(),
    )

    service = GraphService()
    service._get_and_validate_collection = AsyncMock(return_value=SimpleNamespace(id="col1"))

    with patch("aperag.indexing.graph_search_service.build_graph_search_service_for") as build_search:
        graph = await service.get_knowledge_graph(
            user_id="user1",
            collection_id="col1",
            label="*",
            max_depth=3,
            max_nodes=10,
        )

    assert [node.id for node in graph.nodes] == ["A", "B"]
    assert [(edge.source, edge.target) for edge in graph.edges] == [("A", "B")]
    assert subgraph_calls == [(("A", "B"), 1)]
    build_search.assert_not_called()


async def test_get_hybrid_graph_joins_projection_and_lineage_metadata(monkeypatch):
    class FakeStore:
        async def expand_neighbors_n_hops(self, *, entity_names, hops):
            assert entity_names == ["A", "B"]
            assert hops == 1
            return (
                [_entity("A", "person"), _entity("B", "organization")],
                [
                    _relation("A", "B"),
                    _relation("A", "outside"),
                ],
            )

    from aperag.domains.knowledge_graph.schemas import GraphEmbeddingPoint
    from aperag.indexing import worker_factory

    monkeypatch.setattr(worker_factory, "_resolve_graph_backend_type", lambda collection: "postgres")
    monkeypatch.setattr(
        worker_factory,
        "_build_lineage_graph_store",
        lambda *, backend_type, collection: FakeStore(),
    )

    service = GraphService()
    service._get_and_validate_collection = AsyncMock(return_value=SimpleNamespace(id="col1"))

    async def fake_projection(*, backend_type, db_collection, collection_id, store, max_entities):
        assert backend_type == "postgres"
        assert collection_id == "col1"
        assert max_entities == 1000
        assert isinstance(store, FakeStore)
        return (
            [
                GraphEmbeddingPoint(
                    name="A",
                    entity_type="person",
                    cluster=0,
                    x=1.0,
                    y=2.0,
                    source_chunk_count=0,
                ),
                GraphEmbeddingPoint(
                    name="B",
                    entity_type="organization",
                    cluster=1,
                    x=3.0,
                    y=4.0,
                    source_chunk_count=0,
                ),
            ],
            {"0": "person", "1": "organization"},
            [_entity("A", "person"), _entity("B", "organization")],
            False,
        )

    service._get_embedding_projection = fake_projection

    graph = await service.get_hybrid_graph(
        user_id="user1",
        collection_id="col1",
        max_entities=1000,
    )

    assert [(node.id, node.x, node.y, node.cluster, node.value) for node in graph.nodes] == [
        ("A", 1.0, 2.0, 0, 9),
        ("B", 3.0, 4.0, 1, 9),
    ]
    assert [(edge.source, edge.target) for edge in graph.edges] == [("A", "B")]
    assert graph.edges[0].properties.relation_type == "related_to"
    assert graph.cluster_labels == {"0": "person", "1": "organization"}
    assert graph.layout_from_cache is False
    assert graph.layout_source == "embedding"


async def test_get_hybrid_graph_falls_back_to_facts_layout_without_vectors(monkeypatch):
    class FakeStore:
        async def list_entities(self, *, limit, offset):
            assert limit == 1000
            assert offset == 0
            return [_entity("A", "person"), _entity("B", "organization")]

        async def expand_neighbors_n_hops(self, *, entity_names, hops):
            assert entity_names == ["A", "B"]
            assert hops == 1
            return (
                [_entity("A", "person"), _entity("B", "organization")],
                [_relation("A", "B")],
            )

    from aperag.indexing import worker_factory

    monkeypatch.setattr(worker_factory, "_resolve_graph_backend_type", lambda collection: "postgres")
    monkeypatch.setattr(
        worker_factory,
        "_build_lineage_graph_store",
        lambda *, backend_type, collection: FakeStore(),
    )

    service = GraphService()
    service._get_and_validate_collection = AsyncMock(return_value=SimpleNamespace(id="col1"))

    async def fake_projection(*, backend_type, db_collection, collection_id, store, max_entities):
        return ([], {}, [_entity("A", "person"), _entity("B", "organization")], False)

    service._get_embedding_projection = fake_projection

    graph = await service.get_hybrid_graph(user_id="user1", collection_id="col1", max_entities=1000)

    assert [node.id for node in graph.nodes] == ["A", "B"]
    assert [(edge.source, edge.target) for edge in graph.edges] == [("A", "B")]
    assert graph.cluster_labels == {"0": "person", "1": "organization"}
    assert graph.layout_source == "facts"


async def test_get_entity_evidence_loads_bounded_chunks(monkeypatch):
    class FakeStore:
        async def get_entity(self, name):
            assert name == "A"
            return SimpleNamespace(
                source_lineage=[
                    LineageMember(
                        document_id="doc1",
                        parse_version="v1",
                        tenant_scope_key="tenant",
                        chunk_ids=("c1", "c2", "c1"),
                    ),
                    LineageMember(
                        document_id="missing",
                        parse_version="v1",
                        tenant_scope_key="tenant",
                        chunk_ids=("ignored",),
                    ),
                ]
            )

    from aperag.indexing import object_store as object_store_module
    from aperag.indexing import parser as parser_module
    from aperag.indexing import worker_factory
    from aperag.objectstore import base as objectstore_base

    monkeypatch.setattr(worker_factory, "_resolve_graph_backend_type", lambda collection: "postgres")
    monkeypatch.setattr(
        worker_factory,
        "_build_lineage_graph_store",
        lambda *, backend_type, collection: FakeStore(),
    )
    monkeypatch.setattr(
        object_store_module,
        "derived_artifact",
        lambda *, collection_id, document_id, parse_version, filename: (
            f"{collection_id}/{document_id}/{parse_version}/{filename}"
        ),
    )
    monkeypatch.setattr(objectstore_base, "get_object_store", lambda: object())
    monkeypatch.setattr(
        parser_module,
        "read_chunks",
        lambda store, path: [
            {
                "chunk_id": "c1",
                "text": "first " * 120,
                "section_path": "1",
                "heading_anchor": "intro",
                "page_idx": 3,
            },
            {
                "chunk_id": "c2",
                "text": "second",
                "section_path": None,
                "heading_anchor": None,
                "page_idx": "bad",
            },
        ],
    )

    service = GraphService()
    service._get_and_validate_collection = AsyncMock(return_value=SimpleNamespace(id="col1"))
    service.db_ops.query_document = AsyncMock(
        side_effect=[
            SimpleNamespace(name="Document One"),
            None,
        ]
    )

    response = await service.get_entity_evidence(user_id="user1", collection_id="col1", entity_name="A", limit=2)

    assert response.subject_type == "entity"
    assert response.entity_name == "A"
    assert response.total_source_chunks == 3
    assert [chunk.chunk_id for chunk in response.chunks] == ["c1", "c2"]
    assert response.chunks[0].document_name == "Document One"
    assert response.chunks[0].text.endswith("...")
    assert len(response.chunks[0].text) <= 503
    assert response.chunks[0].page_idx == 3
    assert response.chunks[1].page_idx is None


async def test_get_embedding_projection_uses_layout_cache(monkeypatch):
    from aperag.domains.knowledge_graph.schemas import GraphEmbeddingPoint
    from aperag.indexing import worker_factory

    class FakeStore:
        async def list_entities(self, *, limit, offset):
            assert limit == 1000
            assert offset == 0
            return [_entity("A", "person")]

    cached_points = [
        GraphEmbeddingPoint(
            name="A",
            entity_type="person",
            cluster=0,
            x=1.0,
            y=2.0,
            source_chunk_count=0,
        )
    ]

    service = GraphService()
    service._load_embedding_projection_cache = AsyncMock(return_value=(cached_points, {"0": "person"}))
    service._save_embedding_projection_cache = AsyncMock()
    monkeypatch.setattr(
        worker_factory,
        "_build_collection_qdrant_connector",
        lambda collection: (_ for _ in ()).throw(AssertionError("vector connector should not be built on cache hit")),
    )

    points, cluster_labels, entities, layout_from_cache = await service._get_embedding_projection(
        backend_type="postgres",
        db_collection=SimpleNamespace(id="col1"),
        collection_id="col1",
        store=FakeStore(),
        max_entities=1000,
    )

    assert points == cached_points
    assert cluster_labels == {"0": "person"}
    assert [entity.name for entity in entities] == ["A"]
    assert layout_from_cache is True
    service._save_embedding_projection_cache.assert_not_called()


async def test_get_embedding_projection_skips_cache_when_graph_vectors_partial():
    class FakeStore:
        async def list_entities(self, *, limit, offset):
            assert limit == 1000
            assert offset == 0
            return [_entity("A", "person")]

    service = GraphService()
    service._graph_vector_cache_state = AsyncMock(
        return_value={
            "available": True,
            "graph_facts_active_count": 10,
            "graph_vectors_active_count": 4,
        }
    )
    service._load_embedding_projection_cache = AsyncMock()
    service._save_embedding_projection_cache = AsyncMock()

    points, cluster_labels, entities, layout_from_cache = await service._get_embedding_projection(
        backend_type="postgres",
        db_collection=SimpleNamespace(id="col1"),
        collection_id="col1",
        store=FakeStore(),
        max_entities=1000,
    )

    assert points == []
    assert cluster_labels == {}
    assert [entity.name for entity in entities] == ["A"]
    assert layout_from_cache is False
    service._load_embedding_projection_cache.assert_not_called()
    service._save_embedding_projection_cache.assert_not_called()


def test_embedding_projection_cache_key_includes_graph_vector_state():
    service = GraphService()
    entities = [_entity("A", "person")]

    old_key = service._embedding_projection_cache_key(
        collection_id="col1",
        backend_type="postgres",
        max_entities=1000,
        collection_config={},
        entities=entities,
        graph_vector_cache_state={
            "graph_facts_active_count": 1,
            "graph_vectors_active_count": 1,
            "graph_vectors_latest_updated_at": "2026-04-29T12:00:00+00:00",
        },
    )
    new_key = service._embedding_projection_cache_key(
        collection_id="col1",
        backend_type="postgres",
        max_entities=1000,
        collection_config={},
        entities=entities,
        graph_vector_cache_state={
            "graph_facts_active_count": 1,
            "graph_vectors_active_count": 1,
            "graph_vectors_latest_updated_at": "2026-04-29T12:30:00+00:00",
        },
    )

    assert old_key != new_key


def test_graph_vector_state_blocks_partial_embedding_layout_cache():
    service = GraphService()

    assert (
        service._graph_vector_state_allows_embedding_layout(
            {
                "available": True,
                "graph_facts_active_count": 10,
                "graph_vectors_active_count": 4,
            }
        )
        is False
    )
    assert (
        service._graph_vector_state_allows_embedding_layout(
            {
                "available": True,
                "graph_facts_active_count": 10,
                "graph_vectors_active_count": 10,
            }
        )
        is True
    )
    assert (
        service._graph_vector_state_allows_embedding_layout(
            {
                "available": True,
                "graph_facts_active_count": 0,
                "graph_vectors_active_count": 0,
            }
        )
        is True
    )
