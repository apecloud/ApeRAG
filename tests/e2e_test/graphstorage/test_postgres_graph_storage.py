"""
PostgreSQL Graph-specific E2E tests using the universal graph storage test suite with Oracle verification.
This file provides PostgreSQL Graph storage instances and runs all tests from GraphStorageTestSuite.
"""

import os
import uuid

import dotenv
import pytest
import pytest_asyncio

from aperag.graph.lightrag.kg.pg_ops_sync_graph_storage import PGOpsSyncGraphStorage
from tests.e2e_test.graphstorage.graph_storage_oracle import GraphStorageOracle
from tests.e2e_test.graphstorage.networkx_baseline_storage import NetworkXBaselineStorage
from tests.e2e_test.graphstorage.test_graph_storage import GraphStorageTestSuite, load_graph_data

dotenv.load_dotenv(".env")


def check_postgres_graph_environment() -> bool:
    """Check if PostgreSQL Graph environment variables are properly configured."""
    required_vars = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        return False

    return True


# Skip all tests in this module if PostgreSQL environment is not configured
pytestmark = pytest.mark.skipif(
    not check_postgres_graph_environment(),
    reason="PostgreSQL environment variables not configured. Required: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB",
)


@pytest_asyncio.fixture(scope="class")
async def postgres_graph_oracle_storage():
    """Create Oracle storage with PostgreSQL Graph storage and NetworkX baseline using full test data."""

    graph_data = load_graph_data()

    # Generate unique workspace for this test class run
    workspace = f"test_postgres_graph_oracle_{uuid.uuid4().hex[:8]}"

    # Initialize PostgreSQL Graph storage
    postgres_graph_storage = PGOpsSyncGraphStorage(
        namespace="test_postgres_graph_oracle",
        workspace=workspace,
    )

    # Initialize NetworkX baseline
    baseline_storage = NetworkXBaselineStorage(
        namespace="baseline_postgres_graph_test",
        workspace="baseline_postgres_graph_workspace",
    )

    # Create Oracle
    oracle = GraphStorageOracle(
        storage=postgres_graph_storage,
        baseline=baseline_storage,
        namespace="test_postgres_graph_oracle",
        workspace=workspace,
    )

    try:
        # Initialize all storages - ONCE for all tests
        await oracle.initialize()
        print(f"🔗 PostgreSQL Graph storage initialized with workspace: {workspace}")

        # Populate with FULL test data - all nodes - ONCE for all tests
        print(f"📂 Populating baseline with {len(graph_data['nodes'])} nodes...")
        node_count = 0
        for entity_id, node_data in graph_data["nodes"].items():
            await oracle.upsert_node(entity_id, node_data["properties"])
            node_count += 1

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

        print(f"🎯 PostgreSQL Graph Oracle storage ready with {len(graph_data['nodes'])} nodes and {edge_count} edges")
        print("🔄 This data will be shared across ALL tests in TestPostgresGraphStorage class")

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
class TestPostgresGraphStorage:
    """PostgreSQL Graph storage test class - directly calls GraphStorageTestSuite methods."""

    # ===== Node Operations =====
    async def test_has_node(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_has_node(oracle, graph_data)

    async def test_get_node(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_node(oracle, graph_data)

    async def test_get_nodes_batch(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_batch(oracle, graph_data)

    async def test_node_degree(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_node_degree(oracle, graph_data)

    async def test_node_degrees_batch(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_node_degrees_batch(oracle, graph_data)

    async def test_upsert_node(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_upsert_node(oracle)

    async def test_delete_node(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_delete_node(oracle)

    async def test_remove_nodes(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_remove_nodes(oracle)

    # ===== Edge Operations =====
    async def test_has_edge(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_has_edge(oracle, graph_data)

    async def test_get_edge(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_edge(oracle.storage, graph_data)

    async def test_get_edges_batch(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_edges_batch(oracle, graph_data)

    async def test_get_node_edges(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_node_edges(oracle, graph_data)

    async def test_get_nodes_edges_batch(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_edges_batch(oracle, graph_data)

    async def test_edge_degree(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_edge_degree(oracle, graph_data)

    async def test_edge_degrees_batch(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_edge_degrees_batch(oracle, graph_data)

    async def test_upsert_edge(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_upsert_edge(oracle)

    async def test_remove_edges(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_remove_edges(oracle)

    # ===== Complex Operations =====
    async def test_data_integrity(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_data_integrity(oracle, graph_data)

    async def test_large_batch_operations(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_large_batch_operations(oracle)

    async def test_data_consistency_after_operations(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_data_consistency_after_operations(oracle)

    async def test_get_all_labels(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_get_all_labels(oracle, graph_data)

    async def test_interface_coverage_summary(self, postgres_graph_oracle_storage):
        oracle, graph_data = postgres_graph_oracle_storage
        await GraphStorageTestSuite.test_interface_coverage_summary(oracle)
