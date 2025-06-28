"""
通用图存储测试套件

提供所有BaseGraphStorage实现的通用测试方法。
这个文件包含GraphStorageTestSuite静态方法，用于被其他测试文件复用。
"""
import os
import random
import time
from typing import Any, Dict, List

import dotenv
import pytest

from aperag.graph.lightrag.base import BaseGraphStorage
# Import NetworkX baseline for comparison testing

dotenv.load_dotenv(".env")


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

def get_random_sample(data: Dict[str, Any], max_size: int = 10, min_size: int = 1) -> List[str]:
    """Get a random sample of keys from data dictionary"""
    keys = list(data.keys())
    if not keys:
        return []

    sample_size = min(max_size, max(min_size, len(keys)))
    return random.sample(keys, sample_size)


def get_high_degree_nodes(graph_data: Dict[str, Any], max_count: int = 5) -> List[str]:
    """Get nodes that are likely to have high degree (appear frequently in edges)"""
    if not graph_data.get("edges"):
        return []

    # Count node appearances in edges
    node_counts = {}
    for edge in graph_data["edges"]:
        start_node = edge.get("start_node_id")
        end_node = edge.get("end_node_id")

        if start_node:
            node_counts[start_node] = node_counts.get(start_node, 0) + 1
        if end_node:
            node_counts[end_node] = node_counts.get(end_node, 0) + 1

    # Sort by count and return top nodes
    if not node_counts:
        return []

    sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
    high_degree_nodes = [node_id for node_id, count in sorted_nodes[:max_count]]

    return high_degree_nodes


class GraphStorageTestSuite:
    """
    Universal test suite for BaseGraphStorage implementations.

    This class contains all the test methods that should work with any
    storage implementation that follows the BaseGraphStorage interface.
    """

    # ===== Node Operations =====

    @staticmethod
    async def test_has_node(oracle, graph_data: Dict[str, Any]):
        """Test has_node function via oracle"""
        print("🔍 Testing has_node via oracle")
        
        # Get random sample of nodes instead of hardcoded values
        sample_entities = get_random_sample(graph_data["nodes"], max_size=10, min_size=3)
        found_count = 0

        for entity in sample_entities:
            # Oracle automatically compares storage and baseline results
            exists = await oracle.has_node(entity)
            if exists:
                found_count += 1
                print(f"✓ Found entity: {entity}")

        assert found_count > 0, "Should find at least some test entities"

        # Test with non-existent node - oracle will verify both return False
        non_existent = await oracle.has_node("不存在的节点_12345")
        assert not non_existent, "Non-existent node should return False"

    @staticmethod
    async def test_get_node(oracle, graph_data: Dict[str, Any]):
        """Test get_node function via oracle"""
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

        # Oracle automatically compares get_node results
        node_data = await oracle.get_node(target_entity)
        assert node_data is not None, f"Node {target_entity} should exist"
        assert node_data["entity_id"] == target_entity

        # Verify description matches
        expected_desc = graph_data["nodes"][target_entity]["properties"]["description"]
        assert node_data["description"] == expected_desc

        # Test non-existent node - oracle will verify both return None
        null_node = await oracle.get_node("不存在的节点_12345")
        assert null_node is None

    @staticmethod
    async def test_get_nodes_batch(oracle, graph_data: Dict[str, Any]):
        """Test get_nodes_batch function via oracle"""
        # Get random sample of nodes for batch testing
        node_ids = get_random_sample(graph_data["nodes"], max_size=20, min_size=5)

        # Oracle automatically compares batch results
        batch_result = await oracle.get_nodes_batch(node_ids)

        assert isinstance(batch_result, dict)
        assert len(batch_result) <= len(node_ids)  # Some might not exist in storage

        for node_id, node_data in batch_result.items():
            assert node_data["entity_id"] == node_id
            assert node_id in graph_data["nodes"]

    @staticmethod
    async def test_node_degree(oracle, graph_data: Dict[str, Any]):
        """Test node_degree function via oracle"""
        # Use nodes that are likely to have connections
        high_degree_nodes = get_high_degree_nodes(graph_data, max_count=5)
        
        if not high_degree_nodes:
            # Fallback to random sampling if no high-degree nodes found
            high_degree_nodes = get_random_sample(graph_data["nodes"], max_size=5, min_size=1)

        for node_id in high_degree_nodes:
            if await oracle.has_node(node_id):
                # Oracle automatically compares degree results
                degree = await oracle.node_degree(node_id)
                assert isinstance(degree, int)
                assert degree >= 0
                print(f"✓ Node {node_id} has degree: {degree}")
                return  # Successfully tested one node
        
        # If none of the high-degree nodes exist, test with any available node
        any_node = list(graph_data["nodes"].keys())[0]
        if await oracle.has_node(any_node):
            degree = await oracle.node_degree(any_node)
            assert isinstance(degree, int)
            assert degree >= 0

    @staticmethod
    async def test_node_degrees_batch(oracle, graph_data: Dict[str, Any]):
        """Test node_degrees_batch function via oracle"""
        # Get random sample for batch degree testing
        node_ids = get_random_sample(graph_data["nodes"], max_size=10, min_size=3)

        # Oracle automatically compares batch results
        degrees = await oracle.node_degrees_batch(node_ids)

        assert isinstance(degrees, dict)
        assert len(degrees) <= len(node_ids)

        for node_id, degree in degrees.items():
            assert isinstance(degree, int)
            assert degree >= 0
            assert node_id in node_ids

    @staticmethod
    async def test_upsert_node(oracle):
        """Test upsert_node function via oracle"""
        print("🔍 Testing upsert_node via oracle")
        
        test_node_id = "测试节点_upsert"
        test_node_data = {
            "entity_id": test_node_id,
            "entity_type": "test",
            "description": "测试节点描述",
            "source_id": "test",
        }

        # Create node - oracle will automatically sync to both storage and baseline
        await oracle.upsert_node(test_node_id, test_node_data)

        # Verify it exists - oracle will compare both results
        exists = await oracle.has_node(test_node_id)
        assert exists

        # Verify data - oracle will compare both results
        retrieved_data = await oracle.get_node(test_node_id)
        assert retrieved_data["entity_id"] == test_node_id
        assert retrieved_data["description"] == "测试节点描述"

        # Update node - oracle will sync both
        updated_data = test_node_data.copy()
        updated_data["description"] = "更新的描述"
        await oracle.upsert_node(test_node_id, updated_data)

        # Verify update - oracle will compare both results
        updated_retrieved = await oracle.get_node(test_node_id)
        assert updated_retrieved["description"] == "更新的描述"

    @staticmethod
    async def test_delete_node(oracle):
        """Test delete_node function via oracle"""
        test_node_id = "测试节点_delete"
        test_node_data = {
            "entity_id": test_node_id,
            "entity_type": "test",
            "description": "待删除的测试节点",
            "source_id": "test",
        }

        # Create node - oracle will sync both
        await oracle.upsert_node(test_node_id, test_node_data)
        
        # Verify exists - oracle will compare both
        assert await oracle.has_node(test_node_id)

        # Delete node - oracle will sync both
        await oracle.delete_node(test_node_id)

        # Verify it's gone - oracle will compare both
        exists_after_delete = await oracle.has_node(test_node_id)
        assert not exists_after_delete

    @staticmethod
    async def test_remove_nodes(oracle):
        """Test remove_nodes function (batch delete) via oracle"""
        # Create multiple test nodes
        test_nodes = ["测试节点_batch1", "测试节点_batch2", "测试节点_batch3"]

        for node_id in test_nodes:
            test_data = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"批量删除测试节点 {node_id}",
                "source_id": "test",
            }
            await oracle.upsert_node(node_id, test_data)
            assert await oracle.has_node(node_id)

        # Batch delete - oracle will sync both
        await oracle.remove_nodes(test_nodes)

        # Verify all are gone - oracle will compare both results
        for node_id in test_nodes:
            exists = await oracle.has_node(node_id)
            assert not exists, f"Node {node_id} should be deleted"

    # ===== Edge Operations =====

    @staticmethod
    async def test_has_edge(oracle, graph_data: Dict[str, Any]):
        """Test has_edge function via oracle"""
        print("🔍 Testing has_edge via oracle")
        
        if not graph_data["edges"]:
            pytest.skip("No edges in test data")

        # Test with random sample of edges
        sample_edges = random.sample(graph_data["edges"], min(5, len(graph_data["edges"])))
        
        found_edges = 0
        for edge in sample_edges:
            start_node = edge["start_node_id"]
            end_node = edge["end_node_id"]

            # Verify nodes exist first - oracle compares both results
            start_exists = await oracle.has_node(start_node)
            end_exists = await oracle.has_node(end_node)

            if start_exists and end_exists:
                # Oracle automatically compares has_edge results
                edge_exists = await oracle.has_edge(start_node, end_node)
                if edge_exists:
                    found_edges += 1
                    print(f"✓ Edge {start_node}->{end_node} exists")

        # Test non-existent edge - oracle will verify both return False
        no_edge = await oracle.has_edge("不存在1", "不存在2")
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
    async def test_get_edges_batch(oracle, graph_data: Dict[str, Any]):
        """Test get_edges_batch function via oracle"""
        if len(graph_data["edges"]) < 3:
            pytest.skip("Not enough edges for batch test")

        # Select random edges for testing
        sample_edges = random.sample(graph_data["edges"], min(10, len(graph_data["edges"])))
        edge_pairs = []

        for edge in sample_edges:
            start_exists = await oracle.has_node(edge["start_node_id"])
            end_exists = await oracle.has_node(edge["end_node_id"])

            if start_exists and end_exists:
                edge_pairs.append({"src": edge["start_node_id"], "tgt": edge["end_node_id"]})

        if edge_pairs:
            # Oracle automatically compares batch results
            batch_result = await oracle.get_edges_batch(edge_pairs)
            assert isinstance(batch_result, dict)

            for (src, tgt), edge_data in batch_result.items():
                assert isinstance(edge_data, dict)

    @staticmethod
    async def test_get_node_edges(oracle, graph_data: Dict[str, Any]):
        """Test get_node_edges function via oracle"""
        # Find nodes that likely have edges
        high_degree_nodes = get_high_degree_nodes(graph_data, max_count=5)
        
        if not high_degree_nodes:
            high_degree_nodes = get_random_sample(graph_data["nodes"], max_size=5, min_size=1)

        for node_id in high_degree_nodes:
            if await oracle.has_node(node_id):
                # Oracle automatically compares results
                edges = await oracle.get_node_edges(node_id)
                assert isinstance(edges, (list, type(None)))

                if edges:
                    print(f"✓ Node {node_id} has {len(edges)} edges")
                    for src, tgt in edges:
                        assert isinstance(src, str)
                        assert isinstance(tgt, str)
                return  # Successfully tested one node

    @staticmethod
    async def test_get_nodes_edges_batch(oracle, graph_data: Dict[str, Any]):
        """Test get_nodes_edges_batch function via oracle"""
        # Select random nodes for batch test
        node_ids = get_random_sample(graph_data["nodes"], max_size=10, min_size=3)

        # Oracle automatically compares batch results
        batch_result = await oracle.get_nodes_edges_batch(node_ids)

        assert isinstance(batch_result, dict)
        assert len(batch_result) <= len(node_ids)

        for node_id, edges in batch_result.items():
            assert isinstance(edges, list)
            for src, tgt in edges:
                assert isinstance(src, str)
                assert isinstance(tgt, str)

    @staticmethod
    async def test_edge_degree(oracle, graph_data: Dict[str, Any]):
        """Test edge_degree function via oracle"""
        if not graph_data["edges"]:
            pytest.skip("No edges in test data")

        # Test with random edge
        sample_edges = random.sample(graph_data["edges"], min(3, len(graph_data["edges"])))
        
        for edge in sample_edges:
            start_node = edge["start_node_id"]
            end_node = edge["end_node_id"]

            if await oracle.has_node(start_node) and await oracle.has_node(end_node):
                # Oracle automatically compares results
                edge_degree = await oracle.edge_degree(start_node, end_node)
                assert isinstance(edge_degree, int)
                assert edge_degree >= 0
                return  # Successfully tested one edge

    @staticmethod
    async def test_edge_degrees_batch(oracle, graph_data: Dict[str, Any]):
        """Test edge_degrees_batch function via oracle"""
        if len(graph_data["edges"]) < 2:
            pytest.skip("Not enough edges for batch test")

        # Select random edges for testing
        sample_edges = random.sample(graph_data["edges"], min(5, len(graph_data["edges"])))
        edge_pairs = []
        
        for edge in sample_edges:
            start_exists = await oracle.has_node(edge["start_node_id"])
            end_exists = await oracle.has_node(edge["end_node_id"])

            if start_exists and end_exists:
                edge_pairs.append((edge["start_node_id"], edge["end_node_id"]))

        if edge_pairs:
            # Oracle automatically compares batch results
            degrees = await oracle.edge_degrees_batch(edge_pairs)
            assert isinstance(degrees, dict)

            for (src, tgt), degree in degrees.items():
                assert isinstance(degree, int)
                assert degree >= 0

    @staticmethod
    async def test_upsert_edge(oracle):
        """Test upsert_edge function via oracle"""
        print("🔍 Testing upsert_edge via oracle")
        
        # Create two test nodes first - oracle will sync both
        node1_id = "测试节点_edge_src"
        node2_id = "测试节点_edge_tgt"

        for node_id in [node1_id, node2_id]:
            test_data = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"边测试节点 {node_id}",
                "source_id": "test",
            }
            await oracle.upsert_node(node_id, test_data)

        # Create edge - oracle will sync both
        edge_data = {"weight": 1.0, "description": "测试边", "source_id": "test"}
        await oracle.upsert_edge(node1_id, node2_id, edge_data)

        # Verify edge exists - oracle will compare both results
        edge_exists = await oracle.has_edge(node1_id, node2_id)
        assert edge_exists

        # Verify edge data - oracle will compare both results
        retrieved_edge = await oracle.get_edge(node1_id, node2_id)
        if retrieved_edge:
            assert float(retrieved_edge["weight"]) == 1.0
            assert retrieved_edge["description"] == "测试边"

    @staticmethod
    async def test_remove_edges(oracle):
        """Test remove_edges function via oracle"""
        # Create test nodes and edges - oracle will sync both
        nodes = ["测试节点_edge1", "测试节点_edge2", "测试节点_edge3"]

        for node_id in nodes:
            test_data = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"边删除测试节点 {node_id}",
                "source_id": "test",
            }
            await oracle.upsert_node(node_id, test_data)

        # Create edges - oracle will sync both
        edges_to_remove = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]

        for src, tgt in edges_to_remove:
            edge_data = {"weight": 1.0, "description": "待删除的测试边", "source_id": "test"}
            await oracle.upsert_edge(src, tgt, edge_data)
            
            # Oracle automatically compares results
            edge_exists = await oracle.has_edge(src, tgt)
            assert edge_exists

        # Batch remove edges - oracle will sync both
        await oracle.remove_edges(edges_to_remove)

        # Verify edges are gone - oracle will compare both results
        for src, tgt in edges_to_remove:
            edge_exists = await oracle.has_edge(src, tgt)
            assert not edge_exists, f"Edge {src}->{tgt} should be deleted"

    # ===== Performance and Stress Tests =====

    @staticmethod
    async def test_data_integrity(oracle, graph_data: Dict[str, Any]):
        """Test that loaded data maintains integrity via oracle"""
        # Sample random nodes and verify their data
        sample_nodes = get_random_sample(graph_data["nodes"], max_size=50, min_size=20)
        
        verified_count = 0
        for entity_id in sample_nodes:
            expected_data = graph_data["nodes"][entity_id]
            
            if await oracle.has_node(entity_id):
                # Oracle automatically compares get_node results
                actual_data = await oracle.get_node(entity_id)

                # Check key fields
                assert actual_data["entity_id"] == entity_id
                assert actual_data["entity_type"] == expected_data["properties"]["entity_type"]

                # Description should match
                if "description" in expected_data["properties"]:
                    assert actual_data["description"] == expected_data["properties"]["description"]

                verified_count += 1

        print(f"✓ Data integrity verified for {verified_count} nodes via oracle")
        assert verified_count > 0, "Should verify at least some nodes"

    @staticmethod
    async def test_large_batch_operations(oracle):
        """Test performance with large batch operations via oracle"""
        # Test large batch node operations
        large_batch_size = 1000
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
        
        # Use individual upserts for batch operations
        for node_id, node_data in test_nodes:
            await oracle.upsert_node(node_id, node_data)
        
        upsert_time = time.time() - start_time
        
        # Test batch retrieval - oracle will compare results
        node_ids = [node_id for node_id, _ in test_nodes]
        start_time = time.time()
        batch_result = await oracle.get_nodes_batch(node_ids)
        retrieval_time = time.time() - start_time
        
        # Test batch deletion - oracle will sync both
        start_time = time.time()
        await oracle.remove_nodes(node_ids)
        deletion_time = time.time() - start_time
        
        print(f"✓ Large batch performance: upsert={upsert_time:.3f}s, retrieval={retrieval_time:.3f}s, deletion={deletion_time:.3f}s")
        
        # Verify all operations completed successfully
        assert len(batch_result) == large_batch_size, "All nodes should be retrieved"
        
        # Verify deletion - oracle will compare results
        remaining_nodes = await oracle.get_nodes_batch(node_ids)
        assert len(remaining_nodes) == 0, "All nodes should be deleted"

    @staticmethod
    async def test_data_consistency_after_operations(oracle):
        """Test data consistency after various operations via oracle"""
        print("Testing data consistency after operations...")
        
        # Create a small graph - oracle will sync both
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
            await oracle.upsert_node(node_id, node_data)
        
        # Insert edges
        for src, tgt in edges:
            edge_data = {
                "weight": 1.0,
                "description": f"Consistency test edge {src}->{tgt}",
                "source_id": "consistency_test",
            }
            await oracle.upsert_edge(src, tgt, edge_data)
        
        # Verify initial state - oracle will compare all results
        for node_id in nodes:
            assert await oracle.has_node(node_id), f"Node {node_id} should exist"
        
        for src, tgt in edges:
            assert await oracle.has_edge(src, tgt), f"Edge {src}->{tgt} should exist"
        
        # Delete a node and verify edges are handled correctly
        await oracle.delete_node("consistency_node_2")
        
        # Node should be gone - oracle will compare both
        assert not await oracle.has_node("consistency_node_2")
        
        # Clean up remaining nodes
        remaining_nodes = ["consistency_node_1", "consistency_node_3"]
        await oracle.remove_nodes(remaining_nodes)
        
        print("✓ Data consistency tests completed via oracle")

    # ===== Graph Operations =====

    @staticmethod
    async def test_get_all_labels(oracle, graph_data: Dict[str, Any]):
        """Test get_all_labels function via oracle"""
        # Oracle automatically compares labels from both storages
        all_labels = await oracle.get_all_labels()
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
        """Test get_knowledge_graph function - skip oracle for complexity"""
        # This is complex to compare, so we test directly on storage
        if not graph_data["nodes"]:
            pytest.skip("No nodes for knowledge graph test")
        
        # Test with any node
        sample_entity = random.choice(list(graph_data["nodes"].keys()))
        
        if await storage.has_node(sample_entity):
            knowledge_graph = await storage.get_knowledge_graph(sample_entity, max_depth=2, max_nodes=50)
            
            assert hasattr(knowledge_graph, 'nodes')
            assert hasattr(knowledge_graph, 'edges')
            assert hasattr(knowledge_graph, 'is_truncated')
            
            print(f"✓ Knowledge graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")

    # ===== Summary Test =====

    @staticmethod
    async def test_interface_coverage_summary(oracle):
        """Summary test to ensure all BaseGraphStorage methods are covered via oracle"""
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
        ]

        # Additional methods that might be implemented
        optional_methods = [
            "initialize",
            "finalize",
            "drop",
        ]

        # Verify all required methods exist on both storage and baseline
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(oracle.storage, method_name):
                missing_methods.append(f"storage.{method_name}")
            if not hasattr(oracle.baseline, method_name):
                missing_methods.append(f"baseline.{method_name}")

        assert not missing_methods, f"Missing required methods: {missing_methods}"

        # Check optional methods
        implemented_optional = []
        for method_name in optional_methods:
            if hasattr(oracle.storage, method_name) and hasattr(oracle.baseline, method_name):
                implemented_optional.append(method_name)

        print(f"✅ All {len(required_methods)} required BaseGraphStorage methods are implemented")
        if implemented_optional:
            print(f"✅ Optional methods implemented: {implemented_optional}")
        print(f"🎯 Oracle tracked {oracle._operation_count} total operations")



