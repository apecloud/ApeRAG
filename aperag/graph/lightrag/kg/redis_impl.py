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

# aioredis is a depricated library, replaced with redis
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, final

from redis.asyncio import ConnectionPool, Redis  # type: ignore
from redis.exceptions import ConnectionError, RedisError  # type: ignore

from aperag.graph.lightrag.base import BaseKVStorage
from aperag.graph.lightrag.utils import logger

# Constants for Redis connection pool
MAX_CONNECTIONS = 50
SOCKET_TIMEOUT = 5.0
SOCKET_CONNECT_TIMEOUT = 3.0


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get("REDIS_URI", "redis://localhost:6379")
        # Create a connection pool with limits
        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=MAX_CONNECTIONS,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT,
            socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        )
        self._redis = Redis(connection_pool=self._pool)
        logger.info(
            f"Initialized Redis connection pool for {self.workspace}:{self.storage_type} with max {MAX_CONNECTIONS} connections"
        )

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        try:
            yield self._redis
        except ConnectionError as e:
            logger.error(f"Redis connection error in {self.workspace}:{self.storage_type}: {e}")
            raise
        except RedisError as e:
            logger.error(f"Redis operation error in {self.workspace}:{self.storage_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Redis operation for {self.workspace}:{self.storage_type}: {e}")
            raise

    async def close(self):
        """Close the Redis connection pool to prevent resource leaks."""
        if hasattr(self, "_redis") and self._redis:
            await self._redis.close()
            await self._pool.disconnect()
            logger.debug(f"Closed Redis connection pool for {self.workspace}:{self.storage_type}")

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self.close()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{self.workspace}:{self.storage_type}:{id}")
                return json.loads(data) if data else None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for id {id}: {e}")
                return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{self.workspace}:{self.storage_type}:{id}")
                results = await pipe.execute()
                return [json.loads(result) if result else None for result in results]
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in batch get: {e}")
                return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for key in keys:
                pipe.exists(f"{self.workspace}:{self.storage_type}:{key}")
            results = await pipe.execute()

            existing_ids = {list(keys)[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        logger.info(f"Inserting {len(data)} items to {self.workspace}:{self.storage_type}")
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{self.workspace}:{self.storage_type}:{k}", json.dumps(v))
                await pipe.execute()

                for k in data:
                    data[k]["_id"] = k
            except json.JSONEncodeError as e:
                logger.error(f"JSON encode error during upsert: {e}")
                raise

    async def delete(self, ids: list[str]) -> None:
        """Delete entries with specified IDs"""
        if not ids:
            return

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for id in ids:
                pipe.delete(f"{self.workspace}:{self.storage_type}:{id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(f"Deleted {deleted_count} of {len(ids)} entries from {self.workspace}:{self.storage_type}")

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all keys under the current workspace:storage_type.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        async with self._get_redis_connection() as redis:
            try:
                keys = await redis.keys(f"{self.workspace}:{self.storage_type}:*")

                if keys:
                    pipe = redis.pipeline()
                    for key in keys:
                        pipe.delete(key)
                    results = await pipe.execute()
                    deleted_count = sum(results)

                    logger.info(f"Dropped {deleted_count} keys from {self.workspace}:{self.storage_type}")
                    return {
                        "status": "success",
                        "message": f"{deleted_count} keys dropped",
                    }
                else:
                    logger.info(f"No keys found to drop in {self.workspace}:{self.storage_type}")
                    return {"status": "success", "message": "no keys to drop"}

            except Exception as e:
                logger.error(f"Error dropping keys from {self.workspace}:{self.storage_type}: {e}")
                return {"status": "error", "message": str(e)}
