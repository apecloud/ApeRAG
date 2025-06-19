# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LightRAG Module for ApeRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ApeRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

# Import all storage implementations with conditional handling
try:
    from .neo4j_impl import Neo4JStorage
except ImportError:
    Neo4JStorage = None

try:
    from .neo4j_sync_impl import Neo4JSyncStorage
except ImportError:
    Neo4JSyncStorage = None

try:
    from .redis_impl import RedisKVStorage
except ImportError:
    RedisKVStorage = None

try:
    from .postgres_impl import PGDocStatusStorage, PGKVStorage, PGVectorStorage
except ImportError:
    PGKVStorage = None
    PGVectorStorage = None
    PGDocStatusStorage = None

try:
    from .postgres_sync_impl import PGOpsSyncKVStorage, PGOpsSyncVectorStorage
except ImportError:
    PGOpsSyncDocStatusStorage = None
    PGOpsSyncKVStorage = None
    PGOpsSyncVectorStorage = None

try:
    from .qdrant_impl import QdrantVectorDBStorage
except ImportError:
    QdrantVectorDBStorage = None

STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "RedisKVStorage",
            "PGKVStorage",
            "PGOpsSyncKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "Neo4JStorage",
            "Neo4JSyncStorage",
            "Neo4JHybridStorage",
            "PGGraphStorage",
            "AGEStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "PGVectorStorage",
            "PGOpsSyncVectorStorage",
            "QdrantVectorDBStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "RedisKVStorage": ["REDIS_URI"],
    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    # Vector Storage Implementations
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "QdrantVectorDBStorage": ["QDRANT_URL"],  # QDRANT_API_KEY has default value None
    # Document Status Storage Implementations
    "PGDocStatusStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
}

# Storage implementation module mapping - build conditionally
STORAGES = {}

# Add Neo4J implementations
if Neo4JStorage is not None:
    STORAGES["Neo4JStorage"] = ".kg.neo4j_impl"

if Neo4JSyncStorage is not None:
    STORAGES["Neo4JSyncStorage"] = ".kg.neo4j_sync_impl"

# Add Redis implementations
if RedisKVStorage is not None:
    STORAGES["RedisKVStorage"] = ".kg.redis_impl"

# Add PostgreSQL async implementations
if PGKVStorage is not None:
    STORAGES["PGKVStorage"] = ".kg.postgres_impl"

if PGVectorStorage is not None:
    STORAGES["PGVectorStorage"] = ".kg.postgres_impl"

if PGDocStatusStorage is not None:
    STORAGES["PGDocStatusStorage"] = ".kg.postgres_impl"

if PGOpsSyncKVStorage is not None:
    STORAGES["PGOpsSyncKVStorage"] = ".kg.postgres_sync_impl"

if PGOpsSyncVectorStorage is not None:
    STORAGES["PGOpsSyncVectorStorage"] = ".kg.postgres_sync_impl"

# Add Qdrant implementations
if QdrantVectorDBStorage is not None:
    STORAGES["QdrantVectorDBStorage"] = ".kg.qdrant_impl"


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info["implementations"]:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
            f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
        )
