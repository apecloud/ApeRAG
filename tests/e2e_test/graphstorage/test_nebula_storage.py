"""
Nebula-specific E2E tests using the universal graph storage test suite with Oracle verification.
This file provides Nebula storage instances and runs all tests from GraphStorageTestSuite.
"""

import os
import uuid

import dotenv
import pytest
import pytest_asyncio

from aperag.graph.lightrag.kg.nebula_sync_impl import NebulaSyncStorage
from tests.e2e_test.graphstorage.graph_storage_oracle import GraphStorageOracle
from tests.e2e_test.graphstorage.networkx_baseline_storage import NetworkXBaselineStorage
from tests.e2e_test.graphstorage.test_graph_storage import GraphStorageTestSuite, load_graph_data

dotenv.load_dotenv(".env")


def check_nebula_environment() -> bool:
    """Check if Nebula environment variables are properly configured."""
    required_vars = ["NEBULA_HOST", "NEBULA_PORT", "NEBULA_USER", "NEBULA_PASSWORD"]
    return all(os.getenv(var) for var in required_vars)


pytestmark = pytest.mark.skipif(
    not check_nebula_environment(),
    reason="Nebula environment variables not configured. Required: NEBULA_HOST, NEBULA_PORT, NEBULA_USER, NEBULA_PASSWORD",
)


@pytest_asyncio.fixture(scope="class")
async def nebula_oracle_storage():
    """Create Oracle storage with Nebula storage and NetworkX baseline using full test data."""
    graph_data = load_graph_data()
    workspace = f"test_nebula_oracle_{uuid.uuid4().hex[:8]}"

    nebula_storage = NebulaSyncStorage(
        namespace="test_nebula_oracle",
        workspace=workspace,
    )
    baseline_storage = NetworkXBaselineStorage(
        namespace="baseline_nebula_test",
        workspace="baseline_nebula_workspace",
    )
    oracle = GraphStorageOracle(
        storage=nebula_storage,
        baseline=baseline_storage,
        namespace="test_nebula_oracle",
        workspace=workspace,
    )

    try:
        await oracle.initialize()
        for entity_id, node_data in graph_data["nodes"].items():
            await oracle.upsert_node(entity_id, node_data["properties"])

        if graph_data.get("edges"):
            for edge in graph_data["edges"]:
                start_node_id = edge.get("start_node_id")
                end_node_id = edge.get("end_node_id")

                if isinstance(start_node_id, dict):
                    start_node_id = start_node_id.get("properties", {}).get("entity_id")
                if isinstance(end_node_id, dict):
                    end_node_id = end_node_id.get("properties", {}).get("entity_id")

                if start_node_id and end_node_id:
                    await oracle.upsert_edge(start_node_id, end_node_id, edge.get("properties", {}))

        yield oracle, graph_data
    finally:
        try:
            await oracle.drop()
        finally:
            await oracle.finalize()


@pytest.mark.asyncio
class TestNebulaStorage:
    """Nebula storage test class - directly calls GraphStorageTestSuite methods."""

    async def test_has_node(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_has_node(oracle, graph_data)

    async def test_get_node(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_node(oracle, graph_data)

    async def test_get_nodes_batch(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_batch(oracle, graph_data)

    async def test_node_degree(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_node_degree(oracle, graph_data)

    async def test_node_degrees_batch(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_node_degrees_batch(oracle, graph_data)

    async def test_upsert_node(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_upsert_node(oracle)

    async def test_delete_node(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_delete_node(oracle)

    async def test_remove_nodes(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_remove_nodes(oracle)

    async def test_has_edge(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_has_edge(oracle, graph_data)

    async def test_get_edge(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_edge(oracle.storage, graph_data)

    async def test_get_edges_batch(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_edges_batch(oracle, graph_data)

    async def test_get_node_edges(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_node_edges(oracle, graph_data)

    async def test_get_nodes_edges_batch(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_nodes_edges_batch(oracle, graph_data)

    async def test_edge_degree(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_edge_degree(oracle, graph_data)

    async def test_edge_degrees_batch(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_edge_degrees_batch(oracle, graph_data)

    async def test_upsert_edge(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_upsert_edge(oracle)

    async def test_remove_edges(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_remove_edges(oracle)

    async def test_data_integrity(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_data_integrity(oracle, graph_data)

    async def test_large_batch_operations(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_large_batch_operations(oracle)

    async def test_data_consistency_after_operations(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_data_consistency_after_operations(oracle)

    async def test_get_all_labels(self, nebula_oracle_storage):
        oracle, graph_data = nebula_oracle_storage
        await GraphStorageTestSuite.test_get_all_labels(oracle, graph_data)

    async def test_interface_coverage_summary(self, nebula_oracle_storage):
        oracle, _ = nebula_oracle_storage
        await GraphStorageTestSuite.test_interface_coverage_summary(oracle)
