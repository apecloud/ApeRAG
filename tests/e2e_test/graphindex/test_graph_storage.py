"""
E2E tests for graph storage implementations (Neo4j and Nebula).
This module tests the BaseGraphStorage interface implementations.
"""

import asyncio
import os

import dotenv
import pytest
import pytest_asyncio
import numpy as np
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from aperag.graph.lightrag.utils import EmbeddingFunc
from aperag.graph.lightrag.types import KnowledgeGraph

dotenv.load_dotenv(".env")

# For testing, use existing Nebula space to avoid creation issues

class TestGraphStorage:
    """Test suite for graph storage implementations"""

    @pytest.fixture
    def mock_embedding_func(self):
        """Create a mock embedding function for testing"""
        async def mock_embed(texts: List[str]) -> np.ndarray:
            # Return dummy embeddings with consistent dimension
            return np.random.rand(len(texts), 128).astype(np.float32)
        
        return EmbeddingFunc(
            embedding_dim=128,
            max_token_size=8192,
            func=mock_embed
        )

    @pytest.fixture
    def test_workspace(self):
        """Provide a workspace ID for testing - use existing Nebula space"""
        # Use existing space name that we know works
        return "col0c8bcfd142ddbbab"  # Known existing Nebula space

    @pytest.fixture
    def test_namespace(self):
        """Provide a test namespace"""
        return "test_namespace"

    @pytest_asyncio.fixture
    async def neo4j_storage(self, test_namespace, test_workspace, mock_embedding_func):
        """Initialize Neo4j sync storage for testing"""
        try:
            from aperag.graph.lightrag.kg.neo4j_sync_impl import Neo4JSyncStorage
            
            storage = Neo4JSyncStorage(
                namespace=test_namespace,
                workspace=test_workspace,
                embedding_func=mock_embedding_func
            )
            await storage.initialize()
            yield storage
            
            # Cleanup after test
            try:
                await storage.drop()
            except Exception as e:
                print(f"Cleanup warning for Neo4j: {e}")
            finally:
                await storage.finalize()
                
        except ImportError:
            pytest.skip("Neo4j sync storage not available")

    @pytest_asyncio.fixture
    async def nebula_storage(self, test_namespace, test_workspace, mock_embedding_func):
        """Initialize Nebula sync storage for testing"""
        try:
            from aperag.graph.lightrag.kg.nebula_sync_impl import NebulaSyncStorage
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            
            storage = NebulaSyncStorage(
                namespace=test_namespace,
                workspace=test_workspace,
                embedding_func=mock_embedding_func
            )
            await storage.initialize()
            
            # Ensure schema exists in the space
            try:
                with NebulaSyncConnectionManager.get_session(space=storage._space_name) as session:
                    # Create base tag if not exists
                    tag_result = session.execute(
                        "CREATE TAG IF NOT EXISTS base ("
                        "entity_id string, "
                        "entity_type string, "
                        "description string, "
                        "source_id string, "
                        "file_path string"
                        ")"
                    )
                    # Create DIRECTED edge if not exists
                    edge_result = session.execute(
                        "CREATE EDGE IF NOT EXISTS DIRECTED ("
                        "weight double, "
                        "description string, "
                        "keywords string, "
                        "source_id string"
                        ")"
                    )
                    # Wait for schema to take effect
                    import time
                    time.sleep(2)
            except Exception as e:
                print(f"Schema setup warning: {e}")
            
            yield storage
            
            # Cleanup after test
            try:
                await storage.drop()
            except Exception as e:
                print(f"Cleanup warning for Nebula: {e}")
            finally:
                await storage.finalize()
                
        except ImportError:
            pytest.skip("Nebula sync storage not available")

    @pytest.fixture
    def sample_nodes(self):
        """Provide sample node data for testing"""
        return {
            "person_alice": {
                "entity_id": "person_alice",
                "entity_type": "person",
                "description": "Alice is a software engineer",
                "source_id": "doc1",
                "file_path": "test_document.txt"
            },
            "person_bob": {
                "entity_id": "person_bob", 
                "entity_type": "person",
                "description": "Bob is a data scientist",
                "source_id": "doc1",
                "file_path": "test_document.txt"
            },
            "company_techcorp": {
                "entity_id": "company_techcorp",
                "entity_type": "company",
                "description": "TechCorp is a technology company",
                "source_id": "doc2",
                "file_path": "test_document2.txt"
            }
        }

    @pytest.fixture
    def sample_edges(self):
        """Provide sample edge data for testing"""
        return [
            {
                "source": "person_alice",
                "target": "person_bob",
                "data": {
                    "weight": 0.8,
                    "description": "Alice and Bob are colleagues",
                    "keywords": "colleague, work",
                    "source_id": "doc1"
                }
            },
            {
                "source": "person_alice", 
                "target": "company_techcorp",
                "data": {
                    "weight": 0.9,
                    "description": "Alice works at TechCorp",
                    "keywords": "employment, work",
                    "source_id": "doc2"
                }
            }
        ]

    @pytest.mark.asyncio
    async def test_node_operations_neo4j(self, neo4j_storage, sample_nodes):
        """Test basic node operations with Neo4j storage"""
        await self._test_node_operations(neo4j_storage, sample_nodes)

    @pytest.mark.asyncio
    async def test_node_operations_nebula(self, nebula_storage, sample_nodes):
        """Test basic node operations with Nebula storage"""
        await self._test_node_operations(nebula_storage, sample_nodes)

    async def _test_node_operations(self, storage, sample_nodes):
        """Common test logic for node operations"""
        # Test node doesn't exist initially
        assert not await storage.has_node("person_alice")
        assert await storage.get_node("person_alice") is None

        # Test upsert node
        await storage.upsert_node("person_alice", sample_nodes["person_alice"])
        
        # Test node exists now
        assert await storage.has_node("person_alice")
        
        # Test get node
        node_data = await storage.get_node("person_alice")
        assert node_data is not None
        assert node_data["entity_id"] == "person_alice"
        assert node_data["entity_type"] == "person"
        assert "Alice is a software engineer" in node_data["description"]

        # Test update node (upsert with new data)
        updated_data = sample_nodes["person_alice"].copy()
        updated_data["description"] = "Alice is a senior software engineer"
        await storage.upsert_node("person_alice", updated_data)
        
        updated_node = await storage.get_node("person_alice")
        assert "senior software engineer" in updated_node["description"]

    @pytest.mark.asyncio
    async def test_edge_operations_neo4j(self, neo4j_storage, sample_nodes, sample_edges):
        """Test basic edge operations with Neo4j storage"""
        await self._test_edge_operations(neo4j_storage, sample_nodes, sample_edges)

    @pytest.mark.asyncio
    async def test_edge_operations_nebula(self, nebula_storage, sample_nodes, sample_edges):
        """Test basic edge operations with Nebula storage"""
        await self._test_edge_operations(nebula_storage, sample_nodes, sample_edges)

    async def _test_edge_operations(self, storage, sample_nodes, sample_edges):
        """Common test logic for edge operations"""
        # First create the nodes
        for node_id, node_data in sample_nodes.items():
            await storage.upsert_node(node_id, node_data)

        # Test edge doesn't exist initially
        assert not await storage.has_edge("person_alice", "person_bob")
        assert await storage.get_edge("person_alice", "person_bob") is None

        # Test upsert edge
        edge = sample_edges[0]
        await storage.upsert_edge(edge["source"], edge["target"], edge["data"])
        
        # Test edge exists now
        assert await storage.has_edge("person_alice", "person_bob")
        
        # Test get edge
        edge_data = await storage.get_edge("person_alice", "person_bob")
        assert edge_data is not None
        assert edge_data["weight"] == 0.8
        assert "colleagues" in edge_data["description"]
        assert edge_data["keywords"] == "colleague, work"

    @pytest.mark.asyncio
    async def test_degree_operations_neo4j(self, neo4j_storage, sample_nodes, sample_edges):
        """Test degree operations with Neo4j storage"""
        await self._test_degree_operations(neo4j_storage, sample_nodes, sample_edges)

    @pytest.mark.asyncio
    async def test_degree_operations_nebula(self, nebula_storage, sample_nodes, sample_edges):
        """Test degree operations with Nebula storage"""
        await self._test_degree_operations(nebula_storage, sample_nodes, sample_edges)

    async def _test_degree_operations(self, storage, sample_nodes, sample_edges):
        """Common test logic for degree operations"""
        # Create nodes and edges
        for node_id, node_data in sample_nodes.items():
            await storage.upsert_node(node_id, node_data)
        
        for edge in sample_edges:
            await storage.upsert_edge(edge["source"], edge["target"], edge["data"])

        # Test node degree (Alice should have degree 2: connected to Bob and TechCorp)
        alice_degree = await storage.node_degree("person_alice")
        assert alice_degree == 2

        # Test edge degree (Alice-Bob edge should be sum of both nodes' degrees)
        edge_degree = await storage.edge_degree("person_alice", "person_bob")
        bob_degree = await storage.node_degree("person_bob")
        expected_edge_degree = alice_degree + bob_degree
        assert edge_degree == expected_edge_degree

    @pytest.mark.asyncio
    async def test_batch_operations_neo4j(self, neo4j_storage, sample_nodes, sample_edges):
        """Test batch operations with Neo4j storage"""
        await self._test_batch_operations(neo4j_storage, sample_nodes, sample_edges)

    @pytest.mark.asyncio
    async def test_batch_operations_nebula(self, nebula_storage, sample_nodes, sample_edges):
        """Test batch operations with Nebula storage"""
        await self._test_batch_operations(nebula_storage, sample_nodes, sample_edges)

    async def _test_batch_operations(self, storage, sample_nodes, sample_edges):
        """Common test logic for batch operations"""
        # Create nodes and edges
        for node_id, node_data in sample_nodes.items():
            await storage.upsert_node(node_id, node_data)
        
        for edge in sample_edges:
            await storage.upsert_edge(edge["source"], edge["target"], edge["data"])

        # Test batch node retrieval
        node_ids = list(sample_nodes.keys())
        batch_nodes = await storage.get_nodes_batch(node_ids)
        assert len(batch_nodes) == len(sample_nodes)
        for node_id in node_ids:
            assert node_id in batch_nodes
            assert batch_nodes[node_id]["entity_id"] == node_id

        # Test batch degree retrieval
        batch_degrees = await storage.node_degrees_batch(node_ids)
        assert len(batch_degrees) == len(node_ids)
        for node_id in node_ids:
            assert node_id in batch_degrees
            assert isinstance(batch_degrees[node_id], int)

        # Test batch edge retrieval
        edge_pairs = [{"src": edge["source"], "tgt": edge["target"]} for edge in sample_edges]
        batch_edges = await storage.get_edges_batch(edge_pairs)
        assert len(batch_edges) >= 1  # At least one edge should be found

        # Test batch node edges
        batch_node_edges = await storage.get_nodes_edges_batch(["person_alice"])
        assert "person_alice" in batch_node_edges
        alice_edges = batch_node_edges["person_alice"]
        assert len(alice_edges) == 2  # Alice has 2 connections

    @pytest.mark.asyncio
    async def test_knowledge_graph_retrieval_neo4j(self, neo4j_storage, sample_nodes, sample_edges):
        """Test knowledge graph retrieval with Neo4j storage"""
        await self._test_knowledge_graph_retrieval(neo4j_storage, sample_nodes, sample_edges)

    @pytest.mark.asyncio
    async def test_knowledge_graph_retrieval_nebula(self, nebula_storage, sample_nodes, sample_edges):
        """Test knowledge graph retrieval with Nebula storage"""
        await self._test_knowledge_graph_retrieval(nebula_storage, sample_nodes, sample_edges)

    async def _test_knowledge_graph_retrieval(self, storage, sample_nodes, sample_edges):
        """Common test logic for knowledge graph retrieval"""
        # Create nodes and edges
        for node_id, node_data in sample_nodes.items():
            await storage.upsert_node(node_id, node_data)
        
        for edge in sample_edges:
            await storage.upsert_edge(edge["source"], edge["target"], edge["data"])

        # Test get all labels
        labels = await storage.get_all_labels()
        assert len(labels) >= 3
        assert "person_alice" in labels
        assert "person_bob" in labels
        assert "company_techcorp" in labels

        # Test get knowledge graph starting from Alice
        kg = await storage.get_knowledge_graph("person_alice", max_depth=2, max_nodes=10)
        assert isinstance(kg, KnowledgeGraph)
        assert len(kg.nodes) >= 2  # At least Alice and connected nodes
        assert len(kg.edges) >= 1  # At least one edge

        # Verify Alice is in the graph
        alice_found = False
        for node in kg.nodes:
            if node.id == "person_alice" or "person_alice" in node.labels:
                alice_found = True
                break
        assert alice_found

        # Test get all nodes (using "*")
        all_kg = await storage.get_knowledge_graph("*", max_nodes=5)
        assert len(all_kg.nodes) >= 3  # All our test nodes

    @pytest.mark.asyncio
    async def test_deletion_operations_neo4j(self, neo4j_storage, sample_nodes, sample_edges):
        """Test deletion operations with Neo4j storage"""
        await self._test_deletion_operations(neo4j_storage, sample_nodes, sample_edges)

    @pytest.mark.asyncio
    async def test_deletion_operations_nebula(self, nebula_storage, sample_nodes, sample_edges):
        """Test deletion operations with Nebula storage"""
        await self._test_deletion_operations(nebula_storage, sample_nodes, sample_edges)

    async def _test_deletion_operations(self, storage, sample_nodes, sample_edges):
        """Common test logic for deletion operations"""
        # Create nodes and edges
        for node_id, node_data in sample_nodes.items():
            await storage.upsert_node(node_id, node_data)
        
        for edge in sample_edges:
            await storage.upsert_edge(edge["source"], edge["target"], edge["data"])

        # Test single node deletion
        assert await storage.has_node("person_bob")
        await storage.delete_node("person_bob")
        assert not await storage.has_node("person_bob")

        # Test batch node deletion
        remaining_nodes = ["person_alice", "company_techcorp"]
        await storage.remove_nodes(remaining_nodes)
        for node_id in remaining_nodes:
            assert not await storage.has_node(node_id)

    @pytest.mark.asyncio
    async def test_error_handling_neo4j(self, neo4j_storage):
        """Test error handling with Neo4j storage"""
        await self._test_error_handling(neo4j_storage)

    @pytest.mark.asyncio
    async def test_error_handling_nebula(self, nebula_storage):
        """Test error handling with Nebula storage"""
        await self._test_error_handling(nebula_storage)

    async def _test_error_handling(self, storage):
        """Common test logic for error handling"""
        # Test operations on non-existent nodes
        assert not await storage.has_node("nonexistent_node")
        assert await storage.get_node("nonexistent_node") is None
        assert await storage.node_degree("nonexistent_node") == 0

        # Test operations on non-existent edges
        assert not await storage.has_edge("node1", "node2")
        assert await storage.get_edge("node1", "node2") is None

        # Test invalid node data (missing required fields should be handled gracefully)
        try:
            await storage.upsert_node("invalid_node", {"description": "missing entity_id"})
        except (ValueError, KeyError) as e:
            # This is expected behavior - either entity_id or entity_type is required
            assert "entity_id" in str(e) or "entity_type" in str(e)

    @pytest.mark.asyncio
    async def test_concurrent_operations_neo4j(self, neo4j_storage, sample_nodes):
        """Test concurrent operations with Neo4j storage"""
        await self._test_concurrent_operations(neo4j_storage, sample_nodes)

    @pytest.mark.asyncio
    async def test_concurrent_operations_nebula(self, nebula_storage, sample_nodes):
        """Test concurrent operations with Nebula storage"""
        await self._test_concurrent_operations(nebula_storage, sample_nodes)

    async def _test_concurrent_operations(self, storage, sample_nodes):
        """Common test logic for concurrent operations"""
        # Test concurrent node creation
        async def create_node(node_id, node_data):
            await storage.upsert_node(node_id, node_data)
            return await storage.has_node(node_id)

        tasks = [
            create_node(node_id, node_data) 
            for node_id, node_data in sample_nodes.items()
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(results)  # All nodes should be created successfully

        # Test concurrent node retrieval
        async def get_node(node_id):
            return await storage.get_node(node_id)

        tasks = [get_node(node_id) for node_id in sample_nodes.keys()]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            assert result is not None
            assert "entity_id" in result

    @pytest.mark.asyncio
    async def test_workspace_isolation_neo4j(self, test_namespace, mock_embedding_func):
        """Test workspace isolation with Neo4j storage"""
        await self._test_workspace_isolation(test_namespace, mock_embedding_func, "Neo4JSyncStorage")

    @pytest.mark.asyncio
    async def test_workspace_isolation_nebula(self, test_namespace, mock_embedding_func):
        """Test workspace isolation with Nebula storage"""
        await self._test_workspace_isolation(test_namespace, mock_embedding_func, "NebulaSyncStorage")

    async def _test_workspace_isolation(self, test_namespace, mock_embedding_func, storage_class_name):
        """Common test logic for workspace isolation"""
        try:
            if storage_class_name == "Neo4JSyncStorage":
                from aperag.graph.lightrag.kg.neo4j_sync_impl import Neo4JSyncStorage as StorageClass
            else:
                from aperag.graph.lightrag.kg.nebula_sync_impl import NebulaSyncStorage as StorageClass
                
            # Create two storage instances with different workspaces
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            storage1 = StorageClass(
                namespace=test_namespace,
                workspace=f"workspace1_{unique_id}",
                embedding_func=mock_embedding_func
            )
            storage2 = StorageClass(
                namespace=test_namespace,
                workspace=f"workspace2_{unique_id}", 
                embedding_func=mock_embedding_func
            )
            
            await storage1.initialize()
            await storage2.initialize()
            
            try:
                # Add node to workspace1
                node_data = {
                    "entity_id": "test_node",
                    "entity_type": "test",
                    "description": "Test node for isolation"
                }
                await storage1.upsert_node("test_node", node_data)
                
                # Verify node exists in workspace1 but not workspace2
                assert await storage1.has_node("test_node")
                assert not await storage2.has_node("test_node")
                
                assert await storage1.get_node("test_node") is not None
                assert await storage2.get_node("test_node") is None
                
            finally:
                # Cleanup
                try:
                    await storage1.drop()
                    await storage2.drop()
                except Exception as e:
                    print(f"Cleanup warning: {e}")
                finally:
                    await storage1.finalize()
                    await storage2.finalize()
                    
        except ImportError:
            pytest.skip(f"{storage_class_name} not available")

    @pytest.mark.asyncio 
    async def test_performance_batch_vs_individual_neo4j(self, neo4j_storage):
        """Test performance comparison between batch and individual operations with Neo4j"""
        await self._test_performance_batch_vs_individual(neo4j_storage)

    @pytest.mark.asyncio
    async def test_performance_batch_vs_individual_nebula(self, nebula_storage):
        """Test performance comparison between batch and individual operations with Nebula"""
        await self._test_performance_batch_vs_individual(nebula_storage)

    async def _test_performance_batch_vs_individual(self, storage):
        """Common test logic for performance comparison"""
        import time
        
        # Create test nodes
        test_nodes = {}
        for i in range(10):
            node_id = f"perf_test_node_{i}"
            test_nodes[node_id] = {
                "entity_id": node_id,
                "entity_type": "test",
                "description": f"Performance test node {i}"
            }
        
        # Create nodes
        for node_id, node_data in test_nodes.items():
            await storage.upsert_node(node_id, node_data)
        
        node_ids = list(test_nodes.keys())
        
        # Test individual node retrieval
        start_time = time.time()
        individual_results = {}
        for node_id in node_ids:
            individual_results[node_id] = await storage.get_node(node_id)
        individual_time = time.time() - start_time
        
        # Test batch node retrieval  
        start_time = time.time()
        batch_results = await storage.get_nodes_batch(node_ids)
        batch_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(individual_results) == len(batch_results)
        for node_id in node_ids:
            assert individual_results[node_id] is not None
            assert batch_results[node_id] is not None
            assert individual_results[node_id]["entity_id"] == batch_results[node_id]["entity_id"]
        
        # Log performance comparison (batch should generally be faster)
        print(f"Individual operations: {individual_time:.3f}s, Batch operations: {batch_time:.3f}s")
        
        # The test passes regardless of which is faster, as performance can vary
        # The important thing is that both methods return the same results
        assert True


# Integration test that requires both Neo4j and Nebula to be available
class TestGraphStorageCompatibility:
    """Test compatibility between different graph storage implementations"""

    @pytest.mark.asyncio
    async def test_data_format_compatibility(self, mock_embedding_func, test_workspace):
        """Test that data formats are compatible between Neo4j and Nebula storage"""
        try:
            from aperag.graph.lightrag.kg.neo4j_sync_impl import Neo4JSyncStorage
            from aperag.graph.lightrag.kg.nebula_sync_impl import NebulaSyncStorage
            
            neo4j_storage = Neo4JSyncStorage(
                namespace="compat_test",
                workspace=f"{test_workspace}_neo4j",
                embedding_func=mock_embedding_func
            )
            
            nebula_storage = NebulaSyncStorage(
                namespace="compat_test", 
                workspace=f"{test_workspace}_nebula",
                embedding_func=mock_embedding_func
            )
            
            await neo4j_storage.initialize()
            await nebula_storage.initialize()
            
            try:
                # Test that both storages can handle the same data format
                node_data = {
                    "entity_id": "compat_test_node",
                    "entity_type": "test",
                    "description": "Compatibility test node",
                    "source_id": "test_doc",
                    "file_path": "test.txt"
                }
                
                edge_data = {
                    "weight": 0.5,
                    "description": "Test relationship",
                    "keywords": "test, compatibility",
                    "source_id": "test_doc"
                }
                
                # Create nodes in both storages
                await neo4j_storage.upsert_node("compat_test_node", node_data)
                await nebula_storage.upsert_node("compat_test_node", node_data)
                
                # Verify both can retrieve the data
                neo4j_node = await neo4j_storage.get_node("compat_test_node")
                nebula_node = await nebula_storage.get_node("compat_test_node")
                
                assert neo4j_node is not None
                assert nebula_node is not None
                assert neo4j_node["entity_id"] == nebula_node["entity_id"]
                assert neo4j_node["entity_type"] == nebula_node["entity_type"]
                
            finally:
                # Cleanup
                try:
                    await neo4j_storage.drop()
                    await nebula_storage.drop()
                except Exception as e:
                    print(f"Cleanup warning: {e}")
                finally:
                    await neo4j_storage.finalize()
                    await nebula_storage.finalize()
                    
        except ImportError as e:
            pytest.skip(f"Storage implementations not available: {e}")

    @pytest.fixture
    def mock_embedding_func(self):
        """Create a mock embedding function for testing"""
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 128).astype(np.float32)
        
        return EmbeddingFunc(
            embedding_dim=128,
            max_token_size=8192,
            func=mock_embed
        )

    @pytest.fixture
    def test_workspace(self):
        """Provide a unique workspace ID for testing"""
        import uuid
        return f"test_workspace_{uuid.uuid4().hex[:8]}"
