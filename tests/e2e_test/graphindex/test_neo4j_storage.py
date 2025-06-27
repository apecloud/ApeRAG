"""
Neo4j-specific E2E tests using the universal graph storage test suite.
This file provides Neo4j storage instances and inherits all tests from GraphStorageTestRunner.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import List
import numpy as np

import dotenv
from aperag.graph.lightrag.kg.neo4j_sync_impl import Neo4JSyncStorage
from aperag.graph.lightrag.utils import EmbeddingFunc
from tests.e2e_test.graphindex.test_graph_storage import GraphStorageTestRunner, graph_data, mock_embedding_func

dotenv.load_dotenv(".env")


# Global cache for test data to avoid repeated initialization
_neo4j_test_storage_cache = {}

@pytest_asyncio.fixture(scope="function")
async def storage_with_data(mock_embedding_func, graph_data):
    """Initialize Neo4j storage and populate it with test data, with caching"""
    global _neo4j_test_storage_cache
    
    cache_key = "neo4j_test_data"
    
    # Check if we already have initialized storage
    if cache_key in _neo4j_test_storage_cache:
        storage, cached_graph_data = _neo4j_test_storage_cache[cache_key]
        # Verify storage is still valid
        try:
            await storage.has_node("三国演义")  # Quick connectivity test
            yield storage, cached_graph_data
            return
        except Exception:
            # Storage is invalid, remove from cache and reinitialize
            del _neo4j_test_storage_cache[cache_key]
    
    # Initialize new storage
    import uuid
    workspace = f"test_neo4j_persistent_{uuid.uuid4().hex[:8]}"

    storage = Neo4JSyncStorage(
        namespace="test_neo4j",
        workspace=workspace,
        embedding_func=mock_embedding_func,
    )
    
    try:
        await storage.initialize()
        print(f"Neo4j storage initialized with workspace: {workspace}")
        
        # Populate nodes first
        node_count = 0
        for entity_id, node_data in graph_data["nodes"].items():
            try:
                await storage.upsert_node(entity_id, node_data["properties"])
                node_count += 1
                if node_count % 500 == 0:
                    print(f"Inserted {node_count} nodes...")
            except Exception as e:
                print(f"Warning: Failed to insert node {entity_id}: {e}")
                
        print(f"Successfully inserted {node_count} nodes")
        
        # Then populate edges
        edge_count = 0
        for edge in graph_data["edges"]:
            try:
                # Verify both nodes exist before creating edge
                start_exists = await storage.has_node(edge["start_node_id"])
                end_exists = await storage.has_node(edge["end_node_id"])
                
                if start_exists and end_exists:
                    await storage.upsert_edge(
                        edge["start_node_id"], 
                        edge["end_node_id"], 
                        edge["properties"]
                    )
                    edge_count += 1
                    if edge_count % 500 == 0:
                        print(f"Inserted {edge_count} edges...")
            except Exception as e:
                print(f"Warning: Failed to insert edge {edge['start_node_id']}->{edge['end_node_id']}: {e}")
                
        print(f"Successfully inserted {edge_count} edges")
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Cache the initialized storage
        _neo4j_test_storage_cache[cache_key] = (storage, graph_data)
        
        yield storage, graph_data
        
    except Exception as e:
        print(f"Error during storage initialization: {e}")
        # Cleanup on error
        try:
            await storage.drop()
        except:
            pass
        finally:
            await storage.finalize()
        raise


@pytest.mark.asyncio
class TestNeo4jStorage(GraphStorageTestRunner):
    """
    Neo4j-specific test class that inherits all tests from GraphStorageTestRunner.
    
    This class automatically runs all BaseGraphStorage interface tests against
    the Neo4j storage implementation.
    """
    pass  # All tests are inherited from GraphStorageTestRunner 