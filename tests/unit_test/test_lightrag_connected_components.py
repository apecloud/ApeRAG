"""
Unit tests for LightRAG connected components optimization
"""

from unittest.mock import AsyncMock, patch

import pytest

from aperag.graph.lightrag.lightrag import LightRAG


class TestConnectedComponents:
    """Test cases for connected components functionality"""

    @pytest.fixture
    def mock_lightrag(self):
        """Create a mock LightRAG instance"""
        with patch("aperag.graph.lightrag.lightrag.get_or_create_lock"):
            lightrag = LightRAG(embedding_func=AsyncMock(), llm_model_func=AsyncMock(), workspace="test_workspace")
            # Mock storage methods
            lightrag.chunk_entity_relation_graph = AsyncMock()
            lightrag.entities_vdb = AsyncMock()
            lightrag.relationships_vdb = AsyncMock()
            return lightrag

    def test_find_connected_components_single_component(self, mock_lightrag):
        """Test finding connected components with all entities connected"""
        chunk_results = [
            ({"A": [{"entity_name": "A"}], "B": [{"entity_name": "B"}]}, {("A", "B"): [{"src": "A", "tgt": "B"}]}),
            ({"C": [{"entity_name": "C"}]}, {("B", "C"): [{"src": "B", "tgt": "C"}]}),
        ]

        components = mock_lightrag._find_connected_components(chunk_results)

        assert len(components) == 1
        assert set(components[0]) == {"A", "B", "C"}

    def test_find_connected_components_multiple_components(self, mock_lightrag):
        """Test finding connected components with disconnected groups"""
        chunk_results = [
            (
                {
                    "A": [{"entity_name": "A"}],
                    "B": [{"entity_name": "B"}],
                    "X": [{"entity_name": "X"}],
                    "Y": [{"entity_name": "Y"}],
                },
                {("A", "B"): [{"src": "A", "tgt": "B"}], ("X", "Y"): [{"src": "X", "tgt": "Y"}]},
            )
        ]

        components = mock_lightrag._find_connected_components(chunk_results)

        assert len(components) == 2
        # Components can be in any order
        component_sets = [set(comp) for comp in components]
        assert {"A", "B"} in component_sets
        assert {"X", "Y"} in component_sets

    def test_find_connected_components_isolated_entities(self, mock_lightrag):
        """Test with isolated entities (no edges)"""
        chunk_results = [
            (
                {"A": [{"entity_name": "A"}], "B": [{"entity_name": "B"}], "C": [{"entity_name": "C"}]},
                {},  # No edges
            )
        ]

        components = mock_lightrag._find_connected_components(chunk_results)

        assert len(components) == 3
        # Each entity should be in its own component
        for comp in components:
            assert len(comp) == 1

    def test_find_connected_components_complex_graph(self, mock_lightrag):
        """Test with a more complex graph structure"""
        chunk_results = [
            (
                {f"E{i}": [{"entity_name": f"E{i}"}] for i in range(1, 8)},
                {
                    ("E1", "E2"): [{"src": "E1", "tgt": "E2"}],
                    ("E2", "E3"): [{"src": "E2", "tgt": "E3"}],
                    ("E4", "E5"): [{"src": "E4", "tgt": "E5"}],
                    ("E5", "E6"): [{"src": "E5", "tgt": "E6"}],
                    ("E4", "E6"): [{"src": "E4", "tgt": "E6"}],  # Creates a cycle
                },
            )
        ]

        components = mock_lightrag._find_connected_components(chunk_results)

        assert len(components) == 3
        component_sets = [set(comp) for comp in components]
        assert {"E1", "E2", "E3"} in component_sets
        assert {"E4", "E5", "E6"} in component_sets
        assert {"E7"} in component_sets

    @pytest.mark.asyncio
    async def test_process_entity_groups(self, mock_lightrag):
        """Test processing entity groups"""
        chunk_results = [
            ({"A": [{"entity_name": "A"}], "B": [{"entity_name": "B"}]}, {("A", "B"): [{"src": "A", "tgt": "B"}]}),
            ({"X": [{"entity_name": "X"}], "Y": [{"entity_name": "Y"}]}, {("X", "Y"): [{"src": "X", "tgt": "Y"}]}),
        ]

        components = [["A", "B"], ["X", "Y"]]

        # Mock the merge_nodes_and_edges function
        with patch("aperag.graph.lightrag.lightrag.merge_nodes_and_edges") as mock_merge:
            mock_merge.return_value = None

            result = await mock_lightrag._process_entity_groups(chunk_results, components, "test_collection")

        assert result["groups_processed"] == 2
        assert result["total_entities"] == 4
        assert result["total_relations"] == 2
        assert result["collection_id"] == "test_collection"

        # Verify merge_nodes_and_edges was called twice (once per component)
        assert mock_merge.call_count == 2

    @pytest.mark.asyncio
    async def test_process_entity_groups_filters_correctly(self, mock_lightrag):
        """Test that entity groups are filtered correctly"""
        chunk_results = [
            (
                {"A": [{"entity_name": "A"}], "B": [{"entity_name": "B"}], "C": [{"entity_name": "C"}]},
                {
                    ("A", "B"): [{"src": "A", "tgt": "B"}],
                    ("B", "C"): [{"src": "B", "tgt": "C"}],
                    ("A", "C"): [{"src": "A", "tgt": "C"}],  # Cross-component edge
                },
            )
        ]

        # Two separate components
        components = [["A", "B"], ["C"]]

        with patch("aperag.graph.lightrag.lightrag.merge_nodes_and_edges") as mock_merge:
            mock_merge.return_value = None

            await mock_lightrag._process_entity_groups(chunk_results, components, None)

        # Check that merge was called with correct filtered data
        assert mock_merge.call_count == 2

        # First call should have A, B and edge (A, B)
        first_call_chunks = mock_merge.call_args_list[0][1]["chunk_results"]
        assert len(first_call_chunks) == 1
        nodes, edges = first_call_chunks[0]
        assert set(nodes.keys()) == {"A", "B"}
        assert set(edges.keys()) == {("A", "B")}

        # Second call should have only C and no edges
        second_call_chunks = mock_merge.call_args_list[1][1]["chunk_results"]
        assert len(second_call_chunks) == 1
        nodes, edges = second_call_chunks[0]
        assert set(nodes.keys()) == {"C"}
        assert len(edges) == 0

    @pytest.mark.asyncio
    async def test_aprocess_graph_indexing_with_components(self, mock_lightrag):
        """Test the full aprocess_graph_indexing flow with component grouping"""
        chunks = {
            "chunk1": {"content": "Test content 1", "tokens": 100, "full_doc_id": "doc1"},
            "chunk2": {"content": "Test content 2", "tokens": 100, "full_doc_id": "doc1"},
        }

        # Mock extract_entities to return some results
        with patch("aperag.graph.lightrag.lightrag.extract_entities") as mock_extract:
            mock_extract.return_value = [
                (
                    {"Entity1": [{"entity_name": "Entity1"}], "Entity2": [{"entity_name": "Entity2"}]},
                    {("Entity1", "Entity2"): [{"src": "Entity1", "tgt": "Entity2"}]},
                ),
                ({"Entity3": [{"entity_name": "Entity3"}]}, {}),
            ]

            # Mock merge_nodes_and_edges
            with patch("aperag.graph.lightrag.lightrag.merge_nodes_and_edges"):
                result = await mock_lightrag.aprocess_graph_indexing(chunks, "test_collection")

        assert result["status"] == "success"
        assert result["chunks_processed"] == 2
        assert result["entities_extracted"] == 3
        assert result["relations_extracted"] == 1
        assert result["groups_processed"] == 2  # Two components: {Entity1, Entity2} and {Entity3}

    def test_find_connected_components_empty_input(self, mock_lightrag):
        """Test with empty input"""
        components = mock_lightrag._find_connected_components([])
        assert components == []

    def test_find_connected_components_no_edges(self, mock_lightrag):
        """Test with nodes but no edges"""
        chunk_results = [
            (
                {"A": [{"entity_name": "A"}], "B": [{"entity_name": "B"}]},
                {},  # No edges
            )
        ]

        components = mock_lightrag._find_connected_components(chunk_results)
        assert len(components) == 2
        assert all(len(comp) == 1 for comp in components)
