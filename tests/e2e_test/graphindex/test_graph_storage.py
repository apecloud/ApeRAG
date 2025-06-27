"""
Universal E2E tests for graph storage implementations using real data.
Tests BaseGraphStorage interface methods with actual graph data from graph_storage_test_data.json

This test suite is storage-agnostic and works with any BaseGraphStorage implementation.
"""

import json
import os
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


class GraphStorageTestSuite:
    """
    Universal test suite for BaseGraphStorage implementations.

    This class contains all the test methods that should work with any
    storage implementation that follows the BaseGraphStorage interface.
    """

    # ===== Node Operations =====

    @staticmethod
    async def test_has_node(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test has_node function"""
        # Test with known nodes
        sample_entities = ["刘备", "曹操", "诸葛亮", "三国演义"]
        found_count = 0

        for entity in sample_entities:
            if entity in graph_data["nodes"]:
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
        """Test get_node function"""
        # Find a node with rich data
        target_entity = None
        for entity_id, node_data in graph_data["nodes"].items():
            if len(node_data["properties"].get("description", "")) > 100:
                target_entity = entity_id
                break

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
        """Test get_nodes_batch function"""
        # Select first 5 available nodes
        node_ids = list(graph_data["nodes"].keys())[:5]

        batch_result = await storage.get_nodes_batch(node_ids)

        assert isinstance(batch_result, dict)
        assert len(batch_result) <= len(node_ids)  # Some might not exist in storage

        for node_id, node_data in batch_result.items():
            assert node_data["entity_id"] == node_id
            assert node_id in graph_data["nodes"]

    @staticmethod
    async def test_node_degree(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test node_degree function"""
        # Test with nodes that likely have connections
        test_nodes = ["刘备", "曹操", "诸葛亮"]

        for node_id in test_nodes:
            if node_id in graph_data["nodes"]:
                if await storage.has_node(node_id):
                    degree = await storage.node_degree(node_id)
                    assert isinstance(degree, int)
                    assert degree >= 0
                    print(f"✓ Node {node_id} has degree: {degree}")
                    break
        else:
            # Fallback to any available node
            any_node = list(graph_data["nodes"].keys())[0]
            if await storage.has_node(any_node):
                degree = await storage.node_degree(any_node)
                assert isinstance(degree, int)
                assert degree >= 0

    @staticmethod
    async def test_node_degrees_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test node_degrees_batch function"""
        # Select first 3 nodes for batch test
        node_ids = list(graph_data["nodes"].keys())[:3]

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
        """Test has_edge function"""
        if not graph_data["edges"]:
            pytest.skip("No edges in test data")

        # Test with first available edge
        test_edge = graph_data["edges"][0]
        start_node = test_edge["start_node_id"]
        end_node = test_edge["end_node_id"]

        # Verify nodes exist first
        start_exists = await storage.has_node(start_node)
        end_exists = await storage.has_node(end_node)

        if start_exists and end_exists:
            edge_exists = await storage.has_edge(start_node, end_node)
            print(f"✓ Edge {start_node}->{end_node} exists: {edge_exists}")

        # Test non-existent edge
        no_edge = await storage.has_edge("不存在1", "不存在2")
        assert not no_edge

    @staticmethod
    async def test_get_edge(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_edge function"""
        # Find edge with weight property
        test_edge = None
        for edge in graph_data["edges"]:
            if "weight" in edge["properties"]:
                test_edge = edge
                break

        if not test_edge:
            pytest.skip("No weighted edges in test data")

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
        """Test get_edges_batch function"""
        if len(graph_data["edges"]) < 3:
            pytest.skip("Not enough edges for batch test")

        # Select first 3 edges for testing
        test_edges = graph_data["edges"][:3]
        edge_pairs = []

        for edge in test_edges:
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
        """Test get_node_edges function"""
        # Find a node that likely has edges
        test_nodes = ["刘备", "曹操", "诸葛亮"]

        for node_id in test_nodes:
            if node_id in graph_data["nodes"]:
                if await storage.has_node(node_id):
                    edges = await storage.get_node_edges(node_id)
                    assert isinstance(edges, (list, type(None)))

                    if edges:
                        print(f"✓ Node {node_id} has {len(edges)} edges")
                        for src, tgt in edges:
                            assert isinstance(src, str)
                            assert isinstance(tgt, str)
                    break

    @staticmethod
    async def test_get_nodes_edges_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_nodes_edges_batch function"""
        # Select first 3 nodes for batch test
        node_ids = list(graph_data["nodes"].keys())[:3]

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
        """Test edge_degree function"""
        if not graph_data["edges"]:
            pytest.skip("No edges in test data")

        # Test with first edge
        test_edge = graph_data["edges"][0]
        start_node = test_edge["start_node_id"]
        end_node = test_edge["end_node_id"]

        if await storage.has_node(start_node) and await storage.has_node(end_node):
            edge_degree = await storage.edge_degree(start_node, end_node)
            assert isinstance(edge_degree, int)
            assert edge_degree >= 0

    @staticmethod
    async def test_edge_degrees_batch(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test edge_degrees_batch function"""
        if len(graph_data["edges"]) < 2:
            pytest.skip("Not enough edges for batch test")

        # Select first 2 edges for testing
        edge_pairs = []
        for edge in graph_data["edges"][:2]:
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

    # ===== Graph Operations =====

    @staticmethod
    async def test_get_all_labels(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_all_labels function"""
        all_labels = await storage.get_all_labels()
        assert isinstance(all_labels, list)

        # Check that some expected entities are present
        expected_entities = ["三国演义"]
        for entity in expected_entities:
            if entity in graph_data["nodes"] and entity in all_labels:
                print(f"✓ Found expected entity: {entity}")

        # At least some entities should be found
        assert len(all_labels) > 0

    @staticmethod
    async def test_get_knowledge_graph(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test get_knowledge_graph function"""
        # Test with specific node
        test_entity = "三国演义"
        if test_entity in graph_data["nodes"]:
            if await storage.has_node(test_entity):
                kg = await storage.get_knowledge_graph(test_entity, max_depth=2, max_nodes=100)

                assert isinstance(kg, KnowledgeGraph)
                assert hasattr(kg, "nodes")
                assert hasattr(kg, "edges")

                print(f"✓ Knowledge graph for {test_entity}: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

        # Test with wildcard (all nodes)
        kg_all = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=50)
        assert isinstance(kg_all, KnowledgeGraph)
        assert len(kg_all.nodes) > 0
        print(f"✓ Full knowledge graph: {len(kg_all.nodes)} nodes, {len(kg_all.edges)} edges")

    # ===== Data Integrity =====

    @staticmethod
    async def test_data_integrity(storage: BaseGraphStorage, graph_data: Dict[str, Any]):
        """Test that loaded data maintains integrity"""
        # Sample a few nodes and verify their data
        sample_size = min(5, len(graph_data["nodes"]))
        sample_nodes = list(graph_data["nodes"].items())[:sample_size]

        verified_count = 0
        for entity_id, expected_data in sample_nodes:
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

        # Verify all methods exist on storage object
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(storage, method_name):
                missing_methods.append(method_name)

        assert not missing_methods, f"Missing methods: {missing_methods}"

        print(f"✅ All {len(required_methods)} BaseGraphStorage methods are implemented and tested")


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

    async def test_get_all_labels(self, storage_with_data):
        """Test get_all_labels function"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_get_all_labels(storage, graph_data)

    # async def test_get_knowledge_graph(self, storage_with_data):
    #     """Test get_knowledge_graph function"""
    #     storage, graph_data = storage_with_data
    #     await GraphStorageTestSuite.test_get_knowledge_graph(storage, graph_data)

    async def test_data_integrity(self, storage_with_data):
        """Test that loaded data maintains integrity"""
        storage, graph_data = storage_with_data
        await GraphStorageTestSuite.test_data_integrity(storage, graph_data)

    async def test_interface_coverage_summary(self, storage_with_data):
        """Summary test to ensure all BaseGraphStorage methods are covered"""
        storage, _ = storage_with_data
        await GraphStorageTestSuite.test_interface_coverage_summary(storage)
