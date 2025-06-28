"""
Neo4j-specific E2E tests using the universal graph storage test suite with Oracle verification.
This file provides Neo4j storage instances and inherits all tests from GraphStorageTestRunner.
"""

import asyncio
import uuid

import dotenv
import pytest
import pytest_asyncio

from aperag.graph.lightrag.kg.neo4j_sync_impl import Neo4JSyncStorage
from tests.e2e_test.graphindex.test_graph_storage import (
    GraphStorageTestRunner,
    GraphStorageTestSuite,
    graph_data, 
    mock_embedding_func
)
from tests.e2e_test.graphindex.graph_storage_oracle import GraphStorageOracle
from tests.e2e_test.graphindex.networkx_baseline_storage import NetworkXBaselineStorage

dotenv.load_dotenv(".env")


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="class")
async def neo4j_oracle_storage(mock_embedding_func, graph_data):
    """Create Oracle storage with Neo4j storage and NetworkX baseline using full test data.
    
    This fixture uses class scope, so data is imported once for all tests in TestNeo4jStorage.
    """
    
    # Generate unique workspace for this test class run
    workspace = f"test_neo4j_oracle_{uuid.uuid4().hex[:8]}"
    
    # Initialize Neo4j storage
    neo4j_storage = Neo4JSyncStorage(
        namespace="test_neo4j_oracle",
        workspace=workspace,
        embedding_func=mock_embedding_func,
    )
    
    # Initialize NetworkX baseline
    baseline_storage = NetworkXBaselineStorage(
        namespace="baseline_neo4j_test",
        workspace="baseline_neo4j_workspace",
        embedding_func=mock_embedding_func
    )
    
    # Create Oracle
    oracle = GraphStorageOracle(
        storage=neo4j_storage,
        baseline=baseline_storage,
        namespace="test_neo4j_oracle",
        workspace=workspace, 
        embedding_func=mock_embedding_func
    )
    
    try:
        # Initialize all storages - ONCE for all tests
        await oracle.initialize()
        print(f"🔗 Neo4j storage initialized with workspace: {workspace}")
        
        # Populate with FULL test data - all nodes - ONCE for all tests
        print(f"📂 Populating baseline with {len(graph_data['nodes'])} nodes...")
        for entity_id, node_data in graph_data["nodes"].items():
            await oracle.upsert_node(entity_id, node_data["properties"])
        
        # Populate with FULL test data - all edges - ONCE for all tests
        edge_count = 0
        if graph_data.get("edges"):
            print(f"📂 Populating baseline with {len(graph_data['edges'])} edges...")
            for edge in graph_data["edges"]:
                try:
                    start_node_id = edge.get("start_node_id")
                    end_node_id = edge.get("end_node_id")
                    
                    # Handle case where node IDs might be dictionaries
                    if isinstance(start_node_id, dict):
                        start_node_id = start_node_id.get("properties", {}).get("entity_id")
                    if isinstance(end_node_id, dict):
                        end_node_id = end_node_id.get("properties", {}).get("entity_id")
                    
                    if start_node_id and end_node_id:
                        # Oracle automatically handles both storages
                        await oracle.upsert_edge(start_node_id, end_node_id, edge.get("properties", {}))
                        edge_count += 1
                except Exception as e:
                    print(f"⚠️  Failed to insert edge: {e}")
        
        print(f"🎯 Neo4j Oracle storage ready with {len(graph_data['nodes'])} nodes and {edge_count} edges")
        print("🔄 This data will be shared across ALL tests in TestNeo4jStorage class")
        
        yield oracle, graph_data
        
    except Exception as e:
        print(f"❌ Error during storage initialization: {e}")
        raise
    finally:
        # Final cleanup - ONCE after all tests complete
        print("🧹 Starting final cleanup after all tests...")
        try:
            result = await oracle.drop()  # Clean up test data and drop database
            print(f"📦 Database drop result: {result}")
        except Exception as e:
            print(f"⚠️  Error during drop: {e}")
        finally:
            await oracle.finalize()
        print("✅ Oracle storage final cleanup completed")


@pytest.mark.asyncio
class TestNeo4jStorage(GraphStorageTestRunner):
    """
    Neo4j storage test class with Oracle verification.
    
    Features:
    - Uses FULL test dataset (all nodes and edges) 
    - Class-level fixture: data imported ONCE for all tests
    - Automatic result verification via Oracle
    - Tests Neo4j sync implementation
    - Inherits comprehensive test suite from GraphStorageTestRunner
    
    性能优化：
    - 数据只在类开始时导入一次
    - 所有测试共享同一份数据 
    - 类结束时统一清理和删除数据库
    """

    # Override all test methods to use neo4j_oracle_storage instead of oracle_storage
    
    async def test_has_node(self, neo4j_oracle_storage):
        """Test has_node function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_has_node(oracle, graph_data)

    async def test_get_node(self, neo4j_oracle_storage):
        """Test get_node function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_node(oracle, graph_data)

    async def test_get_nodes_batch(self, neo4j_oracle_storage):
        """Test get_nodes_batch function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_batch(oracle, graph_data)

    async def test_node_degree(self, neo4j_oracle_storage):
        """Test node_degree function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_node_degree(oracle, graph_data)

    async def test_node_degrees_batch(self, neo4j_oracle_storage):
        """Test node_degrees_batch function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_node_degrees_batch(oracle, graph_data)

    async def test_upsert_node(self, neo4j_oracle_storage):
        """Test upsert_node function via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_upsert_node(oracle)

    async def test_delete_node(self, neo4j_oracle_storage):
        """Test delete_node function via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_delete_node(oracle)

    async def test_remove_nodes(self, neo4j_oracle_storage):
        """Test remove_nodes function via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_remove_nodes(oracle)

    async def test_has_edge(self, neo4j_oracle_storage):
        """Test has_edge function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_has_edge(oracle, graph_data)

    async def test_get_edge(self, neo4j_oracle_storage):
        """Test get_edge function via storage"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_edge(oracle.storage, graph_data)

    async def test_get_edges_batch(self, neo4j_oracle_storage):
        """Test get_edges_batch function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_edges_batch(oracle, graph_data)

    async def test_get_node_edges(self, neo4j_oracle_storage):
        """Test get_node_edges function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_node_edges(oracle, graph_data)

    async def test_get_nodes_edges_batch(self, neo4j_oracle_storage):
        """Test get_nodes_edges_batch function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_edges_batch(oracle, graph_data)

    async def test_edge_degree(self, neo4j_oracle_storage):
        """Test edge_degree function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_edge_degree(oracle, graph_data)

    async def test_edge_degrees_batch(self, neo4j_oracle_storage):
        """Test edge_degrees_batch function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_edge_degrees_batch(oracle, graph_data)

    async def test_upsert_edge(self, neo4j_oracle_storage):
        """Test upsert_edge function via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_upsert_edge(oracle)

    async def test_remove_edges(self, neo4j_oracle_storage):
        """Test remove_edges function via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_remove_edges(oracle)

    async def test_large_batch_operations(self, neo4j_oracle_storage):
        """Test large batch operations via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_large_batch_operations(oracle)

    async def test_get_all_labels(self, neo4j_oracle_storage):
        """Test get_all_labels function via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_all_labels(oracle, graph_data)

    async def test_data_integrity(self, neo4j_oracle_storage):
        """Test data integrity via oracle"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_data_integrity(oracle, graph_data)

    async def test_data_consistency_after_operations(self, neo4j_oracle_storage):
        """Test data consistency after operations via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_data_consistency_after_operations(oracle)

    async def test_get_knowledge_graph(self, neo4j_oracle_storage):
        """Test get_knowledge_graph function via storage"""
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_knowledge_graph(oracle.storage, graph_data)

    async def test_oracle_summary(self, neo4j_oracle_storage):
        """Test oracle summary via oracle"""
        oracle, _ = neo4j_oracle_storage
        await GraphStorageTestSuite.test_interface_coverage_summary(oracle)
