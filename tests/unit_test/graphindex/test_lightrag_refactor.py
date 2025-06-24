"""Test for LightRAG refactoring of _grouping_process_chunk_results and merge_nodes_and_edges"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.graph.lightrag.lightrag import LightRAG
from aperag.graph.lightrag.operate import merge_nodes_and_edges


@pytest.mark.asyncio
async def test_grouping_process_chunk_results_with_concurrency():
    """Test that _grouping_process_chunk_results processes components concurrently"""

    # Mock TiktokenTokenizer to avoid tiktoken dependency
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.decode = MagicMock(return_value="decoded text")

    # Mock storage classes
    mock_storage_class = MagicMock()
    mock_storage_class.return_value = MagicMock()

    # Create a LightRAG instance with mocked dependencies
    with patch("aperag.graph.lightrag.lightrag.check_storage_env_vars"):
        with patch("aperag.graph.lightrag.lightrag.verify_storage_implementation"):
            with patch("aperag.graph.lightrag.lightrag.TiktokenTokenizer", return_value=mock_tokenizer):
                # Mock _get_storage_class to return mock storage class
                with patch.object(LightRAG, "_get_storage_class", return_value=mock_storage_class):
                    lightrag = LightRAG(
                        llm_model_func=AsyncMock(),
                        embedding_func=AsyncMock(),
                        llm_model_max_async=3,  # Set concurrency limit
                        workspace="test_workspace",
                    )

    # Mock the storages
    lightrag.chunk_entity_relation_graph = AsyncMock()
    lightrag.entities_vdb = AsyncMock()
    lightrag.relationships_vdb = AsyncMock()

    # Create test chunk results with multiple connected components
    chunk_results = [
        # Component 1: A-B-C
        (
            {
                "A": [
                    {
                        "entity_name": "A",
                        "entity_type": "TYPE1",
                        "description": "desc_a",
                        "source_id": "chunk1",
                        "file_path": "file1",
                    }
                ],
                "B": [
                    {
                        "entity_name": "B",
                        "entity_type": "TYPE1",
                        "description": "desc_b",
                        "source_id": "chunk1",
                        "file_path": "file1",
                    }
                ],
            },
            {
                ("A", "B"): [
                    {
                        "src_id": "A",
                        "tgt_id": "B",
                        "description": "rel_ab",
                        "weight": 1.0,
                        "keywords": "ab",
                        "source_id": "chunk1",
                        "file_path": "file1",
                    }
                ]
            },
        ),
        (
            {
                "B": [
                    {
                        "entity_name": "B",
                        "entity_type": "TYPE1",
                        "description": "desc_b2",
                        "source_id": "chunk2",
                        "file_path": "file1",
                    }
                ],
                "C": [
                    {
                        "entity_name": "C",
                        "entity_type": "TYPE1",
                        "description": "desc_c",
                        "source_id": "chunk2",
                        "file_path": "file1",
                    }
                ],
            },
            {
                ("B", "C"): [
                    {
                        "src_id": "B",
                        "tgt_id": "C",
                        "description": "rel_bc",
                        "weight": 1.0,
                        "keywords": "bc",
                        "source_id": "chunk2",
                        "file_path": "file1",
                    }
                ]
            },
        ),
        # Component 2: D-E
        (
            {
                "D": [
                    {
                        "entity_name": "D",
                        "entity_type": "TYPE2",
                        "description": "desc_d",
                        "source_id": "chunk3",
                        "file_path": "file2",
                    }
                ],
                "E": [
                    {
                        "entity_name": "E",
                        "entity_type": "TYPE2",
                        "description": "desc_e",
                        "source_id": "chunk3",
                        "file_path": "file2",
                    }
                ],
            },
            {
                ("D", "E"): [
                    {
                        "src_id": "D",
                        "tgt_id": "E",
                        "description": "rel_de",
                        "weight": 1.0,
                        "keywords": "de",
                        "source_id": "chunk3",
                        "file_path": "file2",
                    }
                ]
            },
        ),
        # Component 3: F (isolated)
        (
            {
                "F": [
                    {
                        "entity_name": "F",
                        "entity_type": "TYPE3",
                        "description": "desc_f",
                        "source_id": "chunk4",
                        "file_path": "file3",
                    }
                ]
            },
            {},
        ),
    ]

    # Track concurrent executions
    concurrent_executions = []
    execution_order = []

    # Mock merge_nodes_and_edges with delay to test concurrency
    async def mock_merge_nodes_and_edges(*args, **kwargs):
        component = kwargs.get("component", [])
        start_time = asyncio.get_event_loop().time()
        concurrent_executions.append(start_time)
        execution_order.append(component)

        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Return mock statistics based on component size
        # Count actual entities and relations in component_chunk_results
        chunk_results = kwargs.get("chunk_results", [])
        entity_count = sum(len(nodes) for nodes, _ in chunk_results)
        relation_count = sum(len(edges) for _, edges in chunk_results)

        return {"entity_count": entity_count, "relation_count": relation_count}

    # Patch merge_nodes_and_edges
    with patch("aperag.graph.lightrag.operate.merge_nodes_and_edges", side_effect=mock_merge_nodes_and_edges):
        # Also patch it in the module where it's imported
        with patch("aperag.graph.lightrag.lightrag.merge_nodes_and_edges", side_effect=mock_merge_nodes_and_edges):
            # Call the method
            result = await lightrag._grouping_process_chunk_results(chunk_results, collection_id="test_collection")

    # Verify results
    assert result["groups_processed"] == 3  # 3 connected components
    assert result["total_entities"] == 7  # A, B(2 occurrences), B, C, D, E, F = 7
    assert result["total_relations"] == 3  # A-B, B-C, D-E
    assert result["collection_id"] == "test_collection"

    # Verify components were processed
    assert len(execution_order) == 3

    # Check that components are correct (order may vary due to concurrency)
    component_sets = [set(comp) for comp in execution_order]
    expected_components = [
        {"A", "B", "C"},  # Component 1
        {"D", "E"},  # Component 2
        {"F"},  # Component 3
    ]
    for expected in expected_components:
        assert expected in component_sets

    # Verify concurrent execution (tasks should start close to each other)
    if len(concurrent_executions) >= 2:
        time_differences = [
            concurrent_executions[i + 1] - concurrent_executions[i] for i in range(len(concurrent_executions) - 1)
        ]
        # Due to concurrency, time differences should be small (< 0.05 seconds)
        assert all(diff < 0.05 for diff in time_differences), f"Tasks did not execute concurrently: {time_differences}"


@pytest.mark.asyncio
async def test_merge_nodes_and_edges_with_locking():
    """Test that merge_nodes_and_edges properly manages locks"""

    # Create test component and chunk results
    component = ["entity1", "entity2", "entity3"]
    chunk_results = [
        (
            {
                "entity1": [
                    {
                        "entity_name": "entity1",
                        "entity_type": "TYPE1",
                        "description": "desc1",
                        "source_id": "chunk1",
                        "file_path": "file1",
                    }
                ],
                "entity2": [
                    {
                        "entity_name": "entity2",
                        "entity_type": "TYPE1",
                        "description": "desc2",
                        "source_id": "chunk1",
                        "file_path": "file1",
                    }
                ],
            },
            {
                ("entity1", "entity2"): [
                    {
                        "src_id": "entity1",
                        "tgt_id": "entity2",
                        "description": "rel12",
                        "weight": 1.0,
                        "keywords": "12",
                        "source_id": "chunk1",
                        "file_path": "file1",
                    }
                ]
            },
        ),
    ]

    # Mock dependencies
    mock_graph = AsyncMock()
    mock_graph.get_node = AsyncMock(return_value=None)
    mock_graph.has_edge = AsyncMock(return_value=False)
    mock_graph.has_node = AsyncMock(return_value=False)
    mock_graph.upsert_node = AsyncMock()
    mock_graph.upsert_edge = AsyncMock()

    mock_entity_vdb = AsyncMock()
    mock_entity_vdb.upsert = AsyncMock()

    mock_relations_vdb = AsyncMock()
    mock_relations_vdb.upsert = AsyncMock()

    mock_llm = AsyncMock()
    mock_tokenizer = MagicMock()

    # Track lock acquisitions
    acquired_locks = []

    # Mock get_or_create_lock
    def mock_get_or_create_lock(lock_id):
        lock = AsyncMock()
        lock.lock_id = lock_id
        acquired_locks.append(lock_id)
        return lock

    with patch("aperag.graph.lightrag.operate.get_or_create_lock", side_effect=mock_get_or_create_lock):
        # Mock MultiLock to verify it's used
        with patch("aperag.graph.lightrag.operate.MultiLock") as mock_multilock:
            mock_multilock.return_value.__aenter__ = AsyncMock()
            mock_multilock.return_value.__aexit__ = AsyncMock()

            # Call merge_nodes_and_edges with component
            from aperag.graph.lightrag.utils import create_lightrag_logger

            result = await merge_nodes_and_edges(
                chunk_results=chunk_results,
                component=component,
                workspace="test_workspace",
                knowledge_graph_inst=mock_graph,
                entity_vdb=mock_entity_vdb,
                relationships_vdb=mock_relations_vdb,
                llm_model_func=mock_llm,
                tokenizer=mock_tokenizer,
                llm_model_max_token_size=1000,
                summary_to_max_tokens=500,
                addon_params={"language": "English"},
                force_llm_summary_on_merge=5,
                lightrag_logger=create_lightrag_logger(workspace="test_workspace"),
            )

    # Verify locks were created for all entities in sorted order
    expected_locks = ["entity:entity1:test_workspace", "entity:entity2:test_workspace", "entity:entity3:test_workspace"]
    assert acquired_locks == expected_locks

    # Verify MultiLock was called with the correct locks
    mock_multilock.assert_called_once()
    assert len(mock_multilock.call_args[0][0]) == 3  # 3 locks

    # Verify result format
    assert "entity_count" in result
    assert "relation_count" in result
    assert result["entity_count"] == 2  # entity1 and entity2
    assert result["relation_count"] == 1  # entity1-entity2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
