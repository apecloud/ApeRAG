"""
Neo4j-specific E2E tests using the universal graph storage test suite with Oracle verification.
This file provides Neo4j storage instances and runs all tests from GraphStorageTestSuite.
"""

import os
import uuid

import dotenv
import pytest
import pytest_asyncio

from aperag.graph.lightrag.kg.neo4j_sync_impl import Neo4JSyncStorage
from tests.e2e_test.graphstorage.graph_storage_oracle import GraphStorageOracle
from tests.e2e_test.graphstorage.networkx_baseline_storage import NetworkXBaselineStorage
from tests.e2e_test.graphstorage.test_graph_storage import GraphStorageTestSuite, load_graph_data

dotenv.load_dotenv(".env")


def check_neo4j_environment() -> bool:
    """Check if Neo4j environment variables are properly configured."""
    required_vars = ["NEO4J_HOST", "NEO4J_PORT", "NEO4J_USERNAME", "NEO4J_PASSWORD"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        return False

    return True


# Skip all tests in this module if Neo4j environment is not configured
pytestmark = pytest.mark.skipif(
    not check_neo4j_environment(),
    reason="Neo4j environment variables not configured. Required: NEO4J_HOST, NEO4J_PORT, NEO4J_USERNAME, NEO4J_PASSWORD",
)


@pytest_asyncio.fixture(scope="class")
async def neo4j_oracle_storage():
    """Create Oracle storage with Neo4j storage and NetworkX baseline using full test data."""

    graph_data = load_graph_data()

    # Generate unique workspace for this test class run
    workspace = f"test_neo4j_oracle_{uuid.uuid4().hex[:8]}"

    # Initialize Neo4j storage
    neo4j_storage = Neo4JSyncStorage(
        namespace="test_neo4j_oracle",
        workspace=workspace,
    )

    # Initialize NetworkX baseline
    baseline_storage = NetworkXBaselineStorage(
        namespace="baseline_neo4j_test",
        workspace="baseline_neo4j_workspace",
    )

    # Create Oracle
    oracle = GraphStorageOracle(
        storage=neo4j_storage,
        baseline=baseline_storage,
        namespace="test_neo4j_oracle",
        workspace=workspace,
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
class TestNeo4jStorage:
    """Neo4j storage test class - directly calls GraphStorageTestSuite methods."""

    # ===== Node Operations =====
    async def test_has_node(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_has_node(oracle, graph_data)

    async def test_get_node(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_node(oracle, graph_data)

    async def test_get_nodes_batch(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_batch(oracle, graph_data)

    async def test_node_degree(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_node_degree(oracle, graph_data)

    async def test_node_degrees_batch(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_node_degrees_batch(oracle, graph_data)

    async def test_upsert_node(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_upsert_node(oracle)

    async def test_delete_node(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_delete_node(oracle)

    async def test_remove_nodes(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_remove_nodes(oracle)

    # ===== Edge Operations =====
    async def test_has_edge(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_has_edge(oracle, graph_data)

    async def test_get_edge(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_edge(oracle.storage, graph_data)

    async def test_get_edges_batch(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_edges_batch(oracle, graph_data)

    async def test_get_node_edges(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_node_edges(oracle, graph_data)

    async def test_get_nodes_edges_batch(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_edges_batch(oracle, graph_data)

    async def test_edge_degree(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_edge_degree(oracle, graph_data)

    async def test_edge_degrees_batch(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_edge_degrees_batch(oracle, graph_data)

    async def test_upsert_edge(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_upsert_edge(oracle)

    async def test_remove_edges(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_remove_edges(oracle)

    # ===== Complex Operations =====
    async def test_data_integrity(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_data_integrity(oracle, graph_data)

    async def test_large_batch_operations(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_large_batch_operations(oracle)

    async def test_data_consistency_after_operations(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_data_consistency_after_operations(oracle)

    async def test_get_all_labels(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_get_all_labels(oracle, graph_data)

    async def test_interface_coverage_summary(self, neo4j_oracle_storage):
        oracle, graph_data = neo4j_oracle_storage
        await GraphStorageTestSuite.test_interface_coverage_summary(oracle)
