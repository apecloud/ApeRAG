"""
Shared fixtures for graph storage tests.
"""

from typing import List

import numpy as np
import pytest_asyncio

from aperag.graph.lightrag.utils import EmbeddingFunc
from tests.e2e_test.graphindex.networkx_baseline_storage import NetworkXBaselineStorage


@pytest_asyncio.fixture(scope="function")
async def baseline_storage():
    """Create NetworkX baseline storage for comparison testing"""
    
    async def mock_embed(texts: List[str]) -> np.ndarray:
        return np.random.rand(len(texts), 128).astype(np.float32)

    mock_embedding_func = EmbeddingFunc(embedding_dim=128, max_token_size=8192, func=mock_embed)
    
    storage = NetworkXBaselineStorage(
        namespace="baseline_test",
        workspace="baseline_workspace",
        embedding_func=mock_embedding_func
    )
    await storage.initialize()
    yield storage
    await storage.finalize() 