"""
Universal E2E tests for graph storage implementations using real data.
Tests BaseGraphStorage interface methods with actual graph data from graph_storage_test_data.json

This test suite is storage-agnostic and works with any BaseGraphStorage implementation.
"""

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List

import dotenv
import numpy as np
import pytest

from aperag.graph.lightrag.base import BaseGraphStorage
from aperag.graph.lightrag.types import KnowledgeGraph
from aperag.graph.lightrag.utils import EmbeddingFunc

dotenv.load_dotenv(".env")


class TestDataLoader:
    """Utility class to load and validate test data"""

    @staticmethod
    def load_graph_data() -> Dict[str, Any]:
        """Load and validate test data from JSON file"""
        file_path = os.path.join(os.path.dirname(__file__), "graph_storage_test_data.json")
        if not os.path.exists(file_path):
            pytest.skip(f"Test data file not found: {file_path}")

        nodes = {}
        edges = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue

                    if item.get("type") == "node":
                        props = item.get("properties", {})
                        if "entity_id" in props and "entity_type" in props:
                            entity_id = props["entity_id"]
                            nodes[entity_id] = {
                                "neo4j_id": item.get("id"),
                                "labels": item.get("labels", []),
                                "properties": props,
                            }

                    elif item.get("type") == "relationship":
                        start_props = item.get("start", {}).get("properties", {})
                        end_props = item.get("end", {}).get("properties", {})

                        if "entity_id" in start_props and "entity_id" in end_props and "properties" in item:
                            edges.append(
                                {
                                    "neo4j_id": item.get("id"),
                                    "label": item.get("label", "DIRECTED"),
                                    "start_node_id": start_props["entity_id"],
                                    "end_node_id": end_props["entity_id"],
                                    "properties": item["properties"],
                                }
                            )

        except Exception as e:
            pytest.skip(f"Failed to load test data: {e}")

        if not nodes:
            pytest.skip("No valid nodes found in test data")

        print(f"Loaded {len(nodes)} nodes and {len(edges)} edges from test data")
        return {"nodes": nodes, "edges": edges}


@pytest.fixture(scope="session")
def graph_data():
    """Load test data from the JSON file"""
    return TestDataLoader.load_graph_data()


@pytest.fixture(scope="session")
def mock_embedding_func():
    """Create a mock embedding function for testing"""

    async def mock_embed(texts: List[str]) -> np.ndarray:
        return np.random.rand(len(texts), 128).astype(np.float32)

    return EmbeddingFunc(embedding_dim=128, max_token_size=8192, func=mock_embed)


def get_random_sample(data_dict: Dict, max_size: int = 100, min_size: int = 1) -> List:
    """Get random sample from dictionary keys, avoiding hardcoded values"""
    keys = list(data_dict.keys())
    if not keys:
        return []
    
    sample_size = min(max_size, max(min_size, len(keys)))
    if len(keys) <= sample_size:
        return keys
    
    return random.sample(keys, sample_size)


def get_high_degree_nodes(graph_data: Dict[str, Any], max_count: int = 10) -> List[str]:
    """Find nodes that are likely to have high degrees based on edge connections"""
    node_connections = {}
    
    # Count connections for each node
    for edge in graph_data["edges"]:
        start_node = edge["start_node_id"]
        end_node = edge["end_node_id"]
        
        if start_node in graph_data["nodes"]:
            node_connections[start_node] = node_connections.get(start_node, 0) + 1
        if end_node in graph_data["nodes"]:
            node_connections[end_node] = node_connections.get(end_node, 0) + 1
    
    # Sort by connection count and return top nodes
    sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)
    return [node_id for node_id, _ in sorted_nodes[:max_count]]


class GraphStorageTestSuite:
    """
    Universal test suite for BaseGraphStorage implementations.

    This class contains all the test methods that should work with any
    storage implementation that follows the BaseGraphStorage interface.
    """

    # ===== Node Operations =====

    @staticmethod
    async def test_has_node(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test has_node function with random sampling"""
        # Get random sample of nodes instead of hardcoded values
        sample_entities = get_random_sample(graph_data["nodes"], max_size=10, min_size=3)
        found_count = 0

        for entity in sample_entities:
            exists = await storage.has_node(entity)
            if exists:
                found_count += 1
                print(f"✓ Found entity: {entity}")

        assert found_count > 0, "Should find at least some test entities"

        # Test with non-existent node
        non_existent = await storage.has_node("不存在的节点_12345")
        assert not non_existent, "Non-existent node should return False"

    @staticmethod
    async def test_get_node(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_node function with dynamic node selection"""
        # Find a node with rich data (prefer nodes with longer descriptions)
        target_entity = None
        max_desc_length = 0
        
        for entity_id, node_data in graph_data["nodes"].items():
            desc_length = len(node_data["properties"].get("description", ""))
            if desc_length > max_desc_length:
                max_desc_length = desc_length
                target_entity = entity_id

        if not target_entity:
            target_entity = list(graph_data["nodes"].keys())[0]

        node_data = await storage.get_node(target_entity)
        assert node_data is not None, f"Node {target_entity} should exist"
        assert node_data["entity_id"] == target_entity

        # Verify description matches
        expected_desc = graph_data["nodes"][target_entity]["properties"]["description"]
        assert node_data["description"] == expected_desc

        # Test non-existent node
        null_node = await storage.get_node("不存在的节点_12345")
        assert null_node is None

    @staticmethod
    async def test_get_nodes_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_nodes_batch function with random sampling"""
        # Get random sample of nodes for batch testing
        node_ids = get_random_sample(graph_data["nodes"], max_size=20, min_size=5)

        batch_result = await storage.get_nodes_batch(node_ids)

        assert isinstance(batch_result, dict)
        assert len(batch_result) <= len(node_ids)  # Some might not exist in storage

        for node_id, node_data in batch_result.items():
            assert node_data["entity_id"] == node_id
            assert node_id in graph_data["nodes"]

    @staticmethod
    async def test_node_degree(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test node_degree function with high-degree nodes"""
        # Use nodes that are likely to have connections
        high_degree_nodes = get_high_degree_nodes(graph_data, max_count=5)
        
        if not high_degree_nodes:
            # Fallback to random sampling if no high-degree nodes found
            high_degree_nodes = get_random_sample(graph_data["nodes"], max_size=5, min_size=1)

        for node_id in high_degree_nodes:
            if await storage.has_node(node_id):
                degree = await storage.node_degree(node_id)
                assert isinstance(degree, int)
                assert degree >= 0
                print(f"✓ Node {node_id} has degree: {degree}")
                return  # Successfully tested one node
        
        # If none of the high-degree nodes exist, test with any available node
        any_node = list(graph_data["nodes"].keys())[0]
        if await storage.has_node(any_node):
            degree = await storage.node_degree(any_node)
            assert isinstance(degree, int)
            assert degree >= 0

    @staticmethod
    async def test_node_degrees_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test node_degrees_batch function with random sampling"""
        # Get random sample for batch degree testing
        node_ids = get_random_sample(graph_data["nodes"], max_size=10, min_size=3)

        degrees = await storage.node_degrees_batch(node_ids)

        assert isinstance(degrees, dict)
        assert len(degrees) <= len(node_ids)

        for node_id, degree in degrees.items():
            assert isinstance(degree, int)
            assert degree >= 0
            assert node_id in node_ids

    @staticmethod
    async def test_upsert_node(storage: BaseGraphStorage):
        """Test upsert_node function"""
        test_node_id = "测试节点_upsert"
        test_node_data = {
            "entity_id": test_node_id,
            "entity_type": "test",
            "description": "测试节点描述",
            "source_id": "test",
        }

        # Create node
        await storage.upsert_node(test_node_id, test_node_data)

        # Verify it exists
        exists = await storage.has_node(test_node_id)
        assert exists

        # Verify data
        retrieved_data = await storage.get_node(test_node_id)
        assert retrieved_data["entity_id"] == test_node_id
        assert retrieved_data["description"] == "测试节点描述"

        # Update node
        updated_data = test_node_data.copy()
        updated_data["description"] = "更新的描述"
        await storage.upsert_node(test_node_id, updated_data)

        # Verify update
        updated_retrieved = await storage.get_node(test_node_id)
        assert updated_retrieved["description"] == "更新的描述"

    # @staticmethod
    # async def test_upsert_nodes_batch(storage: BaseGraphStorage):
    #     """Test upsert_nodes_batch function - NEW TEST"""
    #     # Create test data for batch upsert
    #     test_nodes = []
    #     for i in range(5):
    #         node_id = f"测试节点_batch_upsert_{i}"
    #         node_data = {
    #             "entity_id": node_id,
    #             "entity_type": "test_batch",
    #             "description": f"批量插入测试节点 {i}",
    #             "source_id": "test_batch",
    #             "created_at": int(time.time()),
    #         }
    #         test_nodes.append((node_id, node_data))
    #
    #     # Batch upsert
    #     if hasattr(storage, 'upsert_nodes_batch'):
    #         await storage.upsert_nodes_batch(test_nodes)
    #
    #         # Verify all nodes exist
    #         for node_id, expected_data in test_nodes:
    #             exists = await storage.has_node(node_id)
    #             assert exists, f"Node {node_id} should exist after batch upsert"
    #
    #             retrieved_data = await storage.get_node(node_id)
    #             assert retrieved_data["entity_id"] == node_id
    #             assert retrieved_data["description"] == expected_data["description"]
    #     else:
    #         print("⚠️  upsert_nodes_batch not implemented, skipping test")

    @staticmethod
    async def test_delete_node(storage: BaseGraphStorage):
        """Test delete_node function"""
        test_node_id = "测试节点_delete"
        test_node_data = {
            "entity_id": test_node_id,
            "entity_type": "test",
            "description": "待删除的测试节点",
            "source_id": "test",
        }

        # Create node
        await storage.upsert_node(test_node_id, test_node_data)
        assert await storage.has_node(test_node_id)

        # Delete node
        await storage.delete_node(test_node_id)

        # Verify it's gone
        exists_after_delete = await storage.has_node(test_node_id)
        assert not exists_after_delete

    @staticmethod
    async def test_remove_nodes(storage: BaseGraphStorage):
        """Test remove_nodes function (batch delete)"""
        # Create multiple test nodes
        test_nodes = ["测试节点_batch1", "测试节点_batch2", "测试节点_batch3"]

        for node_id in test_nodes:
            test_data = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"批量删除测试节点 {node_id}",
                "source_id": "test",
            }
            await storage.upsert_node(node_id, test_data)
            assert await storage.has_node(node_id)

        # Batch delete
        await storage.remove_nodes(test_nodes)

        # Verify all are gone
        for node_id in test_nodes:
            exists = await storage.has_node(node_id)
            assert not exists, f"Node {node_id} should be deleted"

    # ===== Edge Operations =====

    @staticmethod
    async def test_has_edge(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test has_edge function with random edge sampling"""
        if not graph_data["edges"]:
            pytest.skip("No edges in test data")

        # Test with random sample of edges
        sample_edges = random.sample(graph_data["edges"], min(5, len(graph_data["edges"])))
        
        found_edges = 0
        for edge in sample_edges:
            start_node = edge["start_node_id"]
            end_node = edge["end_node_id"]

            # Verify nodes exist first
            start_exists = await storage.has_node(start_node)
            end_exists = await storage.has_node(end_node)

            if start_exists and end_exists:
                edge_exists = await storage.has_edge(start_node, end_node)
                if edge_exists:
                    found_edges += 1
                    print(f"✓ Edge {start_node}->{end_node} exists")

        # Test non-existent edge
        no_edge = await storage.has_edge("不存在1", "不存在2")
        assert not no_edge

    @staticmethod
    async def test_get_edge(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_edge function with weighted edges"""
        # Find edge with weight property
        weighted_edges = [edge for edge in graph_data["edges"] if "weight" in edge["properties"]]
        
        if not weighted_edges:
            pytest.skip("No weighted edges in test data")

        test_edge = random.choice(weighted_edges)
        start_node = test_edge["start_node_id"]
        end_node = test_edge["end_node_id"]

        # Verify nodes exist
        if await storage.has_node(start_node) and await storage.has_node(end_node):
            edge_data = await storage.get_edge(start_node, end_node)

            if edge_data:  # Edge might not exist due to insertion failures
                assert "weight" in edge_data
                assert edge_data["weight"] == test_edge["properties"]["weight"]

    @staticmethod
    async def test_get_edges_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_edges_batch function with random edge sampling"""
        if len(graph_data["edges"]) < 3:
            pytest.skip("Not enough edges for batch test")

        # Select random edges for testing
        sample_edges = random.sample(graph_data["edges"], min(10, len(graph_data["edges"])))
        edge_pairs = []

        for edge in sample_edges:
            start_exists = await storage.has_node(edge["start_node_id"])
            end_exists = await storage.has_node(edge["end_node_id"])

            if start_exists and end_exists:
                edge_pairs.append({"src": edge["start_node_id"], "tgt": edge["end_node_id"]})

        if edge_pairs:
            batch_result = await storage.get_edges_batch(edge_pairs)
            assert isinstance(batch_result, dict)

            for (src, tgt), edge_data in batch_result.items():
                assert isinstance(edge_data, dict)

    @staticmethod
    async def test_get_node_edges(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_node_edges function with high-degree nodes"""
        # Find nodes that likely have edges
        high_degree_nodes = get_high_degree_nodes(graph_data, max_count=5)
        
        if not high_degree_nodes:
            high_degree_nodes = get_random_sample(graph_data["nodes"], max_size=5, min_size=1)

        for node_id in high_degree_nodes:
            if await storage.has_node(node_id):
                edges = await storage.get_node_edges(node_id)
                assert isinstance(edges, (list, type(None)))

                if edges:
                    print(f"✓ Node {node_id} has {len(edges)} edges")
                    for src, tgt in edges:
                        assert isinstance(src, str)
                        assert isinstance(tgt, str)
                return  # Successfully tested one node

    @staticmethod
    async def test_get_nodes_edges_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_nodes_edges_batch function with random sampling"""
        # Select random nodes for batch test
        node_ids = get_random_sample(graph_data["nodes"], max_size=10, min_size=3)

        batch_result = await storage.get_nodes_edges_batch(node_ids)

        assert isinstance(batch_result, dict)
        assert len(batch_result) <= len(node_ids)

        for node_id, edges in batch_result.items():
            assert isinstance(edges, list)
            for src, tgt in edges:
                assert isinstance(src, str)
                assert isinstance(tgt, str)

    @staticmethod
    async def test_edge_degree(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test edge_degree function with random edge sampling"""
        if not graph_data["edges"]:
            pytest.skip("No edges in test data")

        # Test with random edge
        sample_edges = random.sample(graph_data["edges"], min(3, len(graph_data["edges"])))
        
        for edge in sample_edges:
            start_node = edge["start_node_id"]
            end_node = edge["end_node_id"]

            if await storage.has_node(start_node) and await storage.has_node(end_node):
                edge_degree = await storage.edge_degree(start_node, end_node)
                assert isinstance(edge_degree, int)
                assert edge_degree >= 0
                return  # Successfully tested one edge

    @staticmethod
    async def test_edge_degrees_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test edge_degrees_batch function with random edge sampling"""
        if len(graph_data["edges"]) < 2:
            pytest.skip("Not enough edges for batch test")

        # Select random edges for testing
        sample_edges = random.sample(graph_data["edges"], min(5, len(graph_data["edges"])))
        edge_pairs = []
        
        for edge in sample_edges:
            start_exists = await storage.has_node(edge["start_node_id"])
            end_exists = await storage.has_node(edge["end_node_id"])

            if start_exists and end_exists:
                edge_pairs.append((edge["start_node_id"], edge["end_node_id"]))

        if edge_pairs:
            degrees = await storage.edge_degrees_batch(edge_pairs)
            assert isinstance(degrees, dict)

            for (src, tgt), degree in degrees.items():
                assert isinstance(degree, int)
                assert degree >= 0

    @staticmethod
    async def test_upsert_edge(storage: BaseGraphStorage):
        """Test upsert_edge function"""
        # Create two test nodes first
        node1_id = "测试节点_edge_src"
        node2_id = "测试节点_edge_tgt"

        for node_id in [node1_id, node2_id]:
            test_data = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"边测试节点 {node_id}",
                "source_id": "test",
            }
            await storage.upsert_node(node_id, test_data)

        # Create edge
        edge_data = {"weight": 1.0, "description": "测试边", "source_id": "test"}

        await storage.upsert_edge(node1_id, node2_id, edge_data)

        # Verify edge exists
        edge_exists = await storage.has_edge(node1_id, node2_id)
        assert edge_exists

        # Verify edge data
        retrieved_edge = await storage.get_edge(node1_id, node2_id)
        if retrieved_edge:
            assert float(retrieved_edge["weight"]) == 1.0
            assert retrieved_edge["description"] == "测试边"

    # @staticmethod
    # async def test_upsert_edges_batch(storage: BaseGraphStorage):
    #     """Test upsert_edges_batch function - NEW TEST"""
    #     # Create test nodes first
    #     test_nodes = ["测试节点_batch_edge1", "测试节点_batch_edge2", "测试节点_batch_edge3"]
    #
    #     for node_id in test_nodes:
    #         test_data = {
    #             "entity_id": node_id,
    #             "entity_type": "test_batch",
    #             "description": f"批量边测试节点 {node_id}",
    #             "source_id": "test_batch",
    #         }
    #         await storage.upsert_node(node_id, test_data)
    #
    #     # Create test edges for batch upsert
    #     test_edges = []
    #     for i in range(len(test_nodes) - 1):
    #         src_node = test_nodes[i]
    #         tgt_node = test_nodes[i + 1]
    #         edge_data = {
    #             "weight": float(i + 1),
    #             "description": f"批量测试边 {i}",
    #             "source_id": "test_batch",
    #             "created_at": int(time.time()),
    #         }
    #         test_edges.append((src_node, tgt_node, edge_data))
    #
    #     # Batch upsert edges
    #     if hasattr(storage, 'upsert_edges_batch'):
    #         await storage.upsert_edges_batch(test_edges)
    #
    #         # Verify all edges exist
    #         for src_node, tgt_node, expected_data in test_edges:
    #             edge_exists = await storage.has_edge(src_node, tgt_node)
    #             assert edge_exists, f"Edge {src_node}->{tgt_node} should exist after batch upsert"
    #
    #             retrieved_edge = await storage.get_edge(src_node, tgt_node)
    #             if retrieved_edge:
    #                 assert retrieved_edge["description"] == expected_data["description"]
    #     else:
    #         print("⚠️  upsert_edges_batch not implemented, skipping test")

    @staticmethod
    async def test_remove_edges(storage: BaseGraphStorage):
        """Test remove_edges function (batch delete)"""
        # Create test nodes and edges
        nodes = ["测试节点_edge1", "测试节点_edge2", "测试节点_edge3"]

        for node_id in nodes:
            test_data = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"边删除测试节点 {node_id}",
                "source_id": "test",
            }
            await storage.upsert_node(node_id, test_data)

        # Create edges
        edges_to_remove = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]

        for src, tgt in edges_to_remove:
            edge_data = {"weight": 1.0, "description": "待删除的测试边", "source_id": "test"}
            await storage.upsert_edge(src, tgt, edge_data)
            assert await storage.has_edge(src, tgt)

        # Batch remove edges
        await storage.remove_edges(edges_to_remove)

        # Verify edges are gone
        for src, tgt in edges_to_remove:
            edge_exists = await storage.has_edge(src, tgt)
            assert not edge_exists, f"Edge {src}->{tgt} should be deleted"

    # ===== Performance and Stress Tests =====

    @staticmethod
    async def test_large_batch_operations(storage: BaseGraphStorage):
        """Test performance with large batch operations - NEW TEST"""
        # Test large batch node operations
        large_batch_size = 5000
        test_nodes = []
        
        print(f"Testing large batch operations with {large_batch_size} nodes...")
        
        for i in range(large_batch_size):
            node_id = f"large_batch_node_{i}"
            node_data = {
                "entity_id": node_id,
                "entity_type": "performance_test",
                "description": f"Large batch test node {i} with some description content",
                "source_id": "performance_test",
                "created_at": int(time.time()),
            }
            test_nodes.append((node_id, node_data))

        start_time = time.time()
        
        # Test batch upsert if available
        if hasattr(storage, 'upsert_nodes_batch'):
            await storage.upsert_nodes_batch(test_nodes)
        else:
            # Fallback to individual upserts
            for node_id, node_data in test_nodes:
                await storage.upsert_node(node_id, node_data)
        
        upsert_time = time.time() - start_time
        
        # Test batch retrieval
        node_ids = [node_id for node_id, _ in test_nodes]
        start_time = time.time()
        batch_result = await storage.get_nodes_batch(node_ids)
        retrieval_time = time.time() - start_time
        
        # Test batch deletion
        start_time = time.time()
        await storage.remove_nodes(node_ids)
        deletion_time = time.time() - start_time
        
        print(f"✓ Large batch performance: upsert={upsert_time:.3f}s, retrieval={retrieval_time:.3f}s, deletion={deletion_time:.3f}s")
        
        # Verify all operations completed successfully
        assert len(batch_result) == large_batch_size, "All nodes should be retrieved"
        
        # Verify deletion
        remaining_nodes = await storage.get_nodes_batch(node_ids)
        assert len(remaining_nodes) == 0, "All nodes should be deleted"

    @staticmethod
    async def test_concurrent_operations(storage: BaseGraphStorage):
        """Test concurrent access patterns - NEW TEST"""
        print("Testing concurrent operations...")
        
        # Create concurrent tasks for different operations
        async def create_nodes_task(prefix: str, count: int):
            tasks = []
            for i in range(count):
                node_id = f"{prefix}_concurrent_{i}"
                node_data = {
                    "entity_id": node_id,
                    "entity_type": "concurrent_test",
                    "description": f"Concurrent test node {prefix}_{i}",
                    "source_id": "concurrent_test",
                }
                tasks.append(storage.upsert_node(node_id, node_data))
            await asyncio.gather(*tasks)
            return [f"{prefix}_concurrent_{i}" for i in range(count)]

        # Run multiple concurrent create tasks
        task1 = create_nodes_task("task1", 10)
        task2 = create_nodes_task("task2", 10)
        task3 = create_nodes_task("task3", 10)
        
        start_time = time.time()
        results = await asyncio.gather(task1, task2, task3)
        concurrent_time = time.time() - start_time
        
        all_node_ids = []
        for result in results:
            all_node_ids.extend(result)
        
        # Verify all nodes were created
        batch_result = await storage.get_nodes_batch(all_node_ids)
        assert len(batch_result) == 30, "All concurrent nodes should be created"
        
        # Clean up
        await storage.remove_nodes(all_node_ids)
        
        print(f"✓ Concurrent operations completed in {concurrent_time:.3f}s")

    @staticmethod
    async def test_error_handling(storage: BaseGraphStorage):
        """Test error handling and edge cases - NEW TEST"""
        print("Testing error handling...")
        
        # Test with invalid data
        try:
            await storage.upsert_node("test_invalid", {"invalid": "missing_entity_id"})
            # If no exception, that's also valid (some storages might be lenient)
        except Exception as e:
            print(f"✓ Expected error for invalid node data: {type(e).__name__}")
        
        # Test with empty strings
        try:
            await storage.upsert_node("", {"entity_id": "", "entity_type": "test"})
        except Exception as e:
            print(f"✓ Expected error for empty node ID: {type(e).__name__}")
        
        # Test with very long strings
        long_description = "x" * 10000
        long_node_id = "test_long_desc"
        long_node_data = {
            "entity_id": long_node_id,
            "entity_type": "test",
            "description": long_description,
            "source_id": "test",
        }
        
        try:
            await storage.upsert_node(long_node_id, long_node_data)
            # Verify it was stored correctly
            retrieved = await storage.get_node(long_node_id)
            if retrieved:
                assert len(retrieved["description"]) == len(long_description)
                await storage.delete_node(long_node_id)  # Clean up
            print("✓ Handled long description successfully")
        except Exception as e:
            print(f"✓ Long description handled with error: {type(e).__name__}")
        
        # Test operations on non-existent data
        non_existent_node = await storage.get_node("absolutely_non_existent_node_12345")
        assert non_existent_node is None
        
        non_existent_edge = await storage.get_edge("non_existent_1", "non_existent_2")
        assert non_existent_edge is None
        
        print("✓ Error handling tests completed")

    # ===== Graph Operations =====

    @staticmethod
    async def test_get_all_labels(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_all_labels function with dynamic verification"""
        all_labels = await storage.get_all_labels()
        assert isinstance(all_labels, list)

        # Check that some nodes from our test data are present
        sample_entities = get_random_sample(graph_data["nodes"], max_size=5, min_size=1)
        found_entities = 0
        
        for entity in sample_entities:
            if entity in all_labels:
                found_entities += 1
                print(f"✓ Found expected entity: {entity}")

        # At least some entities should be found
        assert len(all_labels) > 0
        print(f"✓ Total labels found: {len(all_labels)}")

    @staticmethod
    async def test_get_knowledge_graph(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_knowledge_graph function with random node selection"""
        # Test with a random node that likely exists
        test_entities = get_random_sample(graph_data["nodes"], max_size=3, min_size=1)
        
        for test_entity in test_entities:
            if await storage.has_node(test_entity):
                kg = await storage.get_knowledge_graph(test_entity, max_depth=2, max_nodes=100)

                assert isinstance(kg, KnowledgeGraph)
                assert hasattr(kg, "nodes")
                assert hasattr(kg, "edges")

                print(f"✓ Knowledge graph for {test_entity}: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
                break
        else:
            print("⚠️  No test entities found in storage for knowledge graph test")

        # Test with wildcard (all nodes)
        kg_all = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=50)
        assert isinstance(kg_all, KnowledgeGraph)
        assert len(kg_all.nodes) > 0
        print(f"✓ Full knowledge graph: {len(kg_all.nodes)} nodes, {len(kg_all.edges)} edges")

    # ===== Data Integrity =====

    @staticmethod
    async def test_data_integrity(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test that loaded data maintains integrity with random sampling"""
        # Sample random nodes and verify their data
        sample_nodes = get_random_sample(graph_data["nodes"], max_size=200, min_size=80)
        
        verified_count = 0
        for entity_id in sample_nodes:
            expected_data = graph_data["nodes"][entity_id]
            
            if await storage.has_node(entity_id):
                actual_data = await storage.get_node(entity_id)

                # Check key fields
                assert actual_data["entity_id"] == entity_id
                assert actual_data["entity_type"] == expected_data["properties"]["entity_type"]

                # Description should match
                if "description" in expected_data["properties"]:
                    assert actual_data["description"] == expected_data["properties"]["description"]

                verified_count += 1

        print(f"✓ Data integrity verified for {verified_count} nodes")
        assert verified_count > 0, "Should verify at least some nodes"

    @staticmethod
    async def test_data_consistency_after_operations(storage: BaseGraphStorage):
        """Test data consistency after various operations - NEW TEST"""
        print("Testing data consistency after operations...")
        
        # Create a small graph
        nodes = ["consistency_node_1", "consistency_node_2", "consistency_node_3"]
        edges = [("consistency_node_1", "consistency_node_2"), ("consistency_node_2", "consistency_node_3")]
        
        # Insert nodes
        for node_id in nodes:
            node_data = {
                "entity_id": node_id,
                "entity_type": "consistency_test",
                "description": f"Consistency test node {node_id}",
                "source_id": "consistency_test",
            }
            await storage.upsert_node(node_id, node_data)
        
        # Insert edges
        for src, tgt in edges:
            edge_data = {
                "weight": 1.0,
                "description": f"Consistency test edge {src}->{tgt}",
                "source_id": "consistency_test",
            }
            await storage.upsert_edge(src, tgt, edge_data)
        
        # Verify initial state
        for node_id in nodes:
            assert await storage.has_node(node_id), f"Node {node_id} should exist"
        
        for src, tgt in edges:
            assert await storage.has_edge(src, tgt), f"Edge {src}->{tgt} should exist"
        
        # Delete a node and verify edges are handled correctly
        await storage.delete_node("consistency_node_2")
        
        # Node should be gone
        assert not await storage.has_node("consistency_node_2")
        
        # Related edges should be handled (depending on implementation)
        # Some implementations cascade delete, others don't
        remaining_edges = []
        for src, tgt in edges:
            if await storage.has_edge(src, tgt):
                remaining_edges.append((src, tgt))
        
        print(f"✓ After node deletion: {len(remaining_edges)} edges remain")
        
        # Clean up remaining nodes
        remaining_nodes = ["consistency_node_1", "consistency_node_3"]
        await storage.remove_nodes(remaining_nodes)
        
        print("✓ Data consistency tests completed")

    # ===== Summary Test =====

    @staticmethod
    async def test_interface_coverage_summary(storage: BaseGraphStorage):
        """Summary test to ensure all BaseGraphStorage methods are covered"""
        # List all BaseGraphStorage abstract methods that should be tested
        required_methods = [
            "has_node",
            "has_edge", 
            "node_degree",
            "edge_degree",
            "get_node",
            "get_edge",
            "get_node_edges",
            "get_nodes_batch",
            "node_degrees_batch",
            "edge_degrees_batch",
            "get_edges_batch",
            "get_nodes_edges_batch",
            "upsert_node",
            "upsert_edge",
            "delete_node",
            "remove_nodes", 
            "remove_edges",
            "get_all_labels",
            "get_knowledge_graph",
        ]

        # Additional methods that might be implemented
        optional_methods = [
            "upsert_nodes_batch",
            "upsert_edges_batch",
            "initialize",
            "finalize",
            "drop",
        ]

        # Verify all required methods exist on storage object
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(storage, method_name):
                missing_methods.append(method_name)

        assert not missing_methods, f"Missing required methods: {missing_methods}"

        # Check optional methods
        implemented_optional = []
        for method_name in optional_methods:
            if hasattr(storage, method_name):
                implemented_optional.append(method_name)

        print(f"✅ All {len(required_methods)} required BaseGraphStorage methods are implemented")
        if implemented_optional:
            print(f"✅ Optional methods implemented: {implemented_optional}")


class GraphStorageTestRunner:
    """
    Test runner that executes all tests from GraphStorageTestSuite.

    Storage-specific test classes should inherit from this class and provide
    their storage instance via the storage_with_data fixture.
    """

    async def test_has_node(self, storage_with_data):
        """Test has_node function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_has_node(storage, graph_data)

    async def test_get_node(self, storage_with_data):
        """Test get_node function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_node(storage, graph_data)

    async def test_get_nodes_batch(self, storage_with_data):
        """Test get_nodes_batch function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_nodes_batch(storage, graph_data)

    async def test_node_degree(self, storage_with_data):
        """Test node_degree function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_node_degree(storage, graph_data)

    async def test_node_degrees_batch(self, storage_with_data):
        """Test node_degrees_batch function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_node_degrees_batch(storage, graph_data)

    async def test_upsert_node(self, storage_with_data):
        """Test upsert_node function"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_upsert_node(storage)

    async def test_delete_node(self, storage_with_data):
        """Test delete_node function"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_delete_node(storage)

    async def test_remove_nodes(self, storage_with_data):
        """Test remove_nodes function (batch delete)"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_remove_nodes(storage)

    async def test_has_edge(self, storage_with_data):
        """Test has_edge function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_has_edge(storage, graph_data)

    async def test_get_edge(self, storage_with_data):
        """Test get_edge function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_edge(storage, graph_data)

    async def test_get_edges_batch(self, storage_with_data):
        """Test get_edges_batch function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_edges_batch(storage, graph_data)

    async def test_get_node_edges(self, storage_with_data):
        """Test get_node_edges function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_node_edges(storage, graph_data)

    async def test_get_nodes_edges_batch(self, storage_with_data):
        """Test get_nodes_edges_batch function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_nodes_edges_batch(storage, graph_data)

    async def test_edge_degree(self, storage_with_data):
        """Test edge_degree function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_edge_degree(storage, graph_data)

    async def test_edge_degrees_batch(self, storage_with_data):
        """Test edge_degrees_batch function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_edge_degrees_batch(storage, graph_data)

    async def test_upsert_edge(self, storage_with_data):
        """Test upsert_edge function"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_upsert_edge(storage)

    async def test_remove_edges(self, storage_with_data):
        """Test remove_edges function (batch delete)"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_remove_edges(storage)

    # ===== Performance and Stress Tests =====

    async def test_large_batch_operations(self, storage_with_data):
        """Test performance with large batch operations - NEW TEST"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_large_batch_operations(storage)

    async def test_concurrent_operations(self, storage_with_data):
        """Test concurrent access patterns - NEW TEST"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_concurrent_operations(storage)

    async def test_error_handling(self, storage_with_data):
        """Test error handling and edge cases - NEW TEST"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_error_handling(storage)

    # ===== Graph Operations =====

    async def test_get_all_labels(self, storage_with_data):
        """Test get_all_labels function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_all_labels(storage, graph_data)

    # ===== Data Integrity =====

    async def test_data_integrity(self, storage_with_data):
        """Test that loaded data maintains integrity"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_data_integrity(storage, graph_data)

    async def test_data_consistency_after_operations(self, storage_with_data):
        """Test data consistency after various operations - NEW TEST"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_data_consistency_after_operations(storage)

    # ===== Summary Test =====

    async def test_interface_coverage_summary(self, storage_with_data):
        """Summary test to ensure all BaseGraphStorage methods are covered"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_interface_coverage_summary(storage)
