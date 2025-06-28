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
    from .neo4j_sync_impl import Neo4JSyncStorage
except ImportError:
    Neo4JSyncStorage = None

try:
    from .nebula_sync_impl import NebulaSyncStorage
except ImportError:
    NebulaSyncStorage = None

try:
    from .redis_impl import RedisKVStorage
except ImportError:
    RedisKVStorage = None


try:
    from .postgres_sync_impl import PGOpsSyncKVStorage, PGOpsSyncVectorStorage
except ImportError:
    PGOpsSyncDocStatusStorage = None
    PGOpsSyncKVStorage = None
    PGOpsSyncVectorStorage = None

STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "RedisKVStorage",
            "PGOpsSyncKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "Neo4JSyncStorage",
            "Neo4JHybridStorage",
            "NebulaSyncStorage",
            "PGGraphStorage",
            "AGEStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "PGOpsSyncVectorStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "RedisKVStorage": ["REDIS_URI"],
    # Graph Storage Implementations
    "NebulaSyncStorage": ["NEBULA_HOST", "NEBULA_PORT", "NEBULA_USER", "NEBULA_PASSWORD"],
}

# Storage implementation module mapping - build conditionally
STORAGES = {}

if Neo4JSyncStorage is not None:
    STORAGES["Neo4JSyncStorage"] = ".kg.neo4j_sync_impl"

# Add NebulaGraph implementations
if NebulaSyncStorage is not None:
    STORAGES["NebulaSyncStorage"] = ".kg.nebula_sync_impl"

# Add Redis implementations
if RedisKVStorage is not None:
    STORAGES["RedisKVStorage"] = ".kg.redis_impl"

if PGOpsSyncKVStorage is not None:
    STORAGES["PGOpsSyncKVStorage"] = ".kg.postgres_sync_impl"

if PGOpsSyncVectorStorage is not None:
    STORAGES["PGOpsSyncVectorStorage"] = ".kg.postgres_sync_impl"

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
