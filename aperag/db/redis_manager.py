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
using redis-py's built-in connection pooling capabilities for both sync and async operations.
"""

import logging
from typing import Optional

import redis
import redis.asyncio as async_redis

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Redis connection manager supporting both sync and async operations.

    This provides shared connection pools for the entire application, avoiding
    the overhead of creating multiple connections for different Redis operations.

    Features:
    - Automatic connection pooling with redis-py (both sync and async)
    - Configurable pool size and timeouts
    - Global shared instance for efficiency
    - Proper cleanup handling
    """

    _instance: Optional["RedisConnectionManager"] = None
    _async_client: Optional[async_redis.Redis] = None
    _sync_client: Optional[redis.Redis] = None
    _async_pool: Optional[async_redis.ConnectionPool] = None
    _sync_pool: Optional[redis.ConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_async_client(cls, redis_url: str = None) -> async_redis.Redis:
        """
        Get async Redis client with shared connection pool.

        Args:
            redis_url: Redis connection URL. If None, will use from config.

        Returns:
            Async Redis client instance with shared connection pool
        """
        if cls._async_client is None:
            await cls._initialize_async_client(redis_url)
        return cls._async_client

    @classmethod
    def get_sync_client(cls, redis_url: str = None) -> redis.Redis:
        """
        Get sync Redis client with shared connection pool.

        Args:
            redis_url: Redis connection URL. If None, will use from config.

        Returns:
            Sync Redis client instance with shared connection pool
        """
        if cls._sync_client is None:
            cls._initialize_sync_client(redis_url)
        return cls._sync_client

    @classmethod
    async def _initialize_async_client(cls, redis_url: str = None):
        """Initialize async Redis client with connection pool."""
        if redis_url is None:
            redis_url = cls._get_redis_url()

        logger.debug(f"Initializing async Redis connection pool: {redis_url}")

        # Create async connection pool
        cls._async_pool = async_redis.ConnectionPool.from_url(
            redis_url,
            max_connections=20,  # Pool size
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Create async client using the pool
        cls._async_client = async_redis.Redis(connection_pool=cls._async_pool)

        # Test connection
        try:
            await cls._async_client.ping()
            logger.debug("Async Redis connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to async Redis: {e}")
            raise ConnectionError(f"Cannot connect to async Redis: {e}")

    @classmethod
    def _initialize_sync_client(cls, redis_url: str = None):
        """Initialize sync Redis client with connection pool."""
        if redis_url is None:
            redis_url = cls._get_redis_url()

        logger.debug(f"Initializing sync Redis connection pool: {redis_url}")

        # Create sync connection pool
        cls._sync_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=20,  # Pool size
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Create sync client using the pool
        cls._sync_client = redis.Redis(connection_pool=cls._sync_pool)

        # Test connection
        try:
            cls._sync_client.ping()
            logger.debug("Sync Redis connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to sync Redis: {e}")
            raise ConnectionError(f"Cannot connect to sync Redis: {e}")

    @classmethod
    def _get_redis_url(cls) -> str:
        """Get Redis URL from configuration."""
        # Import here to avoid circular imports
        from aperag.config import settings

        return settings.MEMORY_REDIS_URL

    @classmethod
    async def close(cls):
        """Close Redis connection pools and clean up resources."""
        # Close async client
        if cls._async_client:
            logger.debug("Closing async Redis connection pool")
            await cls._async_client.close()
            cls._async_client = None
            cls._async_pool = None

        # Close sync client
        if cls._sync_client:
            logger.debug("Closing sync Redis connection pool")
            cls._sync_client.close()
            cls._sync_client = None
            cls._sync_pool = None

        logger.debug("All Redis connection pools closed")

    @classmethod
    def get_pool_info(cls) -> dict:
        """Get connection pool information for monitoring."""
        info = {}

        if cls._async_pool:
            info["async_pool"] = {
                "max_connections": cls._async_pool.max_connections,
                "created_connections": cls._async_pool.created_connections,
                "available_connections": len(cls._async_pool._available_connections),
                "in_use_connections": len(cls._async_pool._in_use_connections),
            }

        if cls._sync_pool:
            info["sync_pool"] = {
                "max_connections": cls._sync_pool.max_connections,
                "created_connections": cls._sync_pool.created_connections,
                "available_connections": len(cls._sync_pool._available_connections),
                "in_use_connections": len(cls._sync_pool._in_use_connections),
            }

        if not info:
            info["status"] = "not_initialized"

        return info


# Convenience functions for backward compatibility
async def get_async_redis_client() -> async_redis.Redis:
    """Get async Redis client - backward compatible with history.py"""
    return await RedisConnectionManager.get_async_client()


def get_sync_redis_client() -> redis.Redis:
    """Get sync Redis client for cache and other sync operations."""
    return RedisConnectionManager.get_sync_client()


def get_redis_connection_manager() -> RedisConnectionManager:
    """Get the Redis connection manager instance."""
    return RedisConnectionManager()


# Legacy compatibility - keep the old function name
async def get_client(redis_url: str = None) -> async_redis.Redis:
    """Legacy function name - use get_async_client instead."""
    return await RedisConnectionManager.get_async_client(redis_url)
