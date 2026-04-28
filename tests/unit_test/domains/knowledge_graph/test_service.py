from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aperag.domains.knowledge_graph.service import GraphService


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
