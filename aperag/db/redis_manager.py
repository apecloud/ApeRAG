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
Redis connection manager for the application.

This module provides a simple and efficient Redis connection management system
using redis-py's built-in connection pooling capabilities.
"""

import logging
from typing import Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Simple Redis connection manager using redis-py's built-in connection pooling.

    This provides a shared connection pool for the entire application, avoiding
    the overhead of creating multiple connections for different Redis operations.

    Features:
    - Automatic connection pooling with redis-py
    - Configurable pool size and timeouts
    - Global shared instance for efficiency
    - Proper cleanup handling
    """

    _instance: Optional["RedisConnectionManager"] = None
    _client: Optional[redis.Redis] = None
    _pool: Optional[redis.ConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_client(cls, redis_url: str = None) -> redis.Redis:
        """
        Get Redis client with shared connection pool.

        Args:
            redis_url: Redis connection URL. If None, will use from config.

        Returns:
            Redis client instance with shared connection pool
        """
        if cls._client is None:
            await cls._initialize_client(redis_url)
        return cls._client

    @classmethod
    async def _initialize_client(cls, redis_url: str = None):
        """Initialize Redis client with connection pool."""
        if redis_url is None:
            # Import here to avoid circular imports
            from aperag.config import settings

            redis_url = settings.memory_redis_url

        logger.debug(f"Initializing Redis connection pool: {redis_url}")

        # Create connection pool with reasonable defaults
        cls._pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=20,  # Pool size
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Create client using the pool
        cls._client = redis.Redis(connection_pool=cls._pool)

        # Test connection
        try:
            await cls._client.ping()
            logger.debug("Redis connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Cannot connect to Redis: {e}")

    @classmethod
    async def close(cls):
        """Close Redis connection pool and clean up resources."""
        if cls._client:
            logger.debug("Closing Redis connection pool")
            await cls._client.close()
            cls._client = None
            cls._pool = None
            logger.debug("Redis connection pool closed")

    @classmethod
    def get_pool_info(cls) -> dict:
        """Get connection pool information for monitoring."""
        if cls._pool:
            return {
                "max_connections": cls._pool.max_connections,
                "created_connections": cls._pool.created_connections,
                "available_connections": len(cls._pool._available_connections),
                "in_use_connections": len(cls._pool._in_use_connections),
            }
        return {"status": "not_initialized"}


# Convenience functions for backward compatibility with history.py
async def get_async_redis_client() -> redis.Redis:
    """Get async Redis client - backward compatible with history.py"""
    return await RedisConnectionManager.get_client()


def get_redis_connection_manager() -> RedisConnectionManager:
    """Get the Redis connection manager instance."""
    return RedisConnectionManager()
