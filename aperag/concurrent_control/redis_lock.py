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
Redis-based distributed lock implementation.

This module contains the RedisLock implementation that uses Redis for
distributed locking across multiple processes, containers, or machines.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as async_redis

from .protocols import LockProtocol
from .utils import LockAcquisitionError

logger = logging.getLogger(__name__)


class RedisLock(LockProtocol):
    """
    Redis-based distributed lock implementation.

    This implementation uses Redis for distributed locking across
    multiple processes, containers, or machines using the SET NX EX pattern
    with Lua scripts for safe lock release.

    Features:
    - Works across multiple processes (celery --pool=prefork)
    - Works across multiple machines/containers
    - Works with any task queue (Celery, Prefect, etc.)
    - Automatic lock expiration to prevent deadlocks
    - Retry mechanisms for lock acquisition
    - Safe lock release using Lua scripts
    - Shared connection pool for efficiency

    Performance considerations:
    - Network round-trip overhead for each lock operation
    - Redis server becomes a critical dependency
    - Higher latency compared to in-process locks
    """

    # Lua script for safe lock release (atomic check-and-delete)
    RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    # Lua script for safe lock renewal (atomic check-and-expire)
    RENEW_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("expire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """

    def __init__(
        self,
        key: str,
        expire_time: int = 120,
        retry_times: int = 3,
        retry_delay: float = 0.1,
        name: str = None,
        redis_client: Optional[async_redis.Redis] = None,
    ):
        """
        Initialize the Redis lock.

        Args:
            key: Redis key for the lock (required)
            expire_time: Lock expiration time in seconds (prevents deadlocks)
            retry_times: Number of retry attempts for lock acquisition
            retry_delay: Delay between retry attempts in seconds
            name: Optional name for the lock (for compatibility with factory)
        """
        if not key:
            raise ValueError("Redis lock key is required")

        self._key = key
        self._name = name or f"redis_lock_{key}"
        self._expire_time = expire_time
        self._retry_times = retry_times
        self._retry_delay = retry_delay
        self._lock_value: Optional[str] = None
        self._is_locked = False
        self._redis_client = redis_client

    async def _get_redis_client(self):
        """Get Redis client from shared connection manager."""
        if self._redis_client:
            return self._redis_client

        from aperag.db.redis_manager import RedisConnectionManager

        self._redis_client = await RedisConnectionManager.get_async_client()
        return self._redis_client

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the distributed lock from Redis.

        Uses SET NX EX pattern for atomic lock acquisition with expiration.

        Args:
            timeout: Maximum time to wait for lock acquisition (seconds).
                    None means retry according to retry_times parameter.

        Returns:
            True if lock was acquired successfully, False if timeout/retry exhausted.
        """
        if self._is_locked:
            logger.warning(f"Redis lock '{self._key}' is already held by this instance")
            return True

        # Generate unique lock value (UUID) to ensure only holder can release
        lock_value = str(uuid.uuid4())
        redis_client = await self._get_redis_client()

        start_time = time.time()
        attempt = 0
        max_attempts = self._retry_times + 1

        while attempt < max_attempts:
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.debug(f"Redis lock '{self._key}' acquisition timed out after {elapsed:.3f}s")
                    return False

            try:
                # Attempt to acquire lock using SET NX EX
                result = await redis_client.set(
                    self._key,
                    lock_value,
                    nx=True,  # Only set if key doesn't exist
                    ex=self._expire_time,  # Set expiration time
                )

                if result:
                    # Lock acquired successfully
                    self._lock_value = lock_value
                    self._is_locked = True
                    elapsed = time.time() - start_time
                    logger.debug(
                        f"Redis lock '{self._key}' acquired after {elapsed:.3f}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    return True

                # Lock not available, wait before retry
                attempt += 1
                if attempt < max_attempts:
                    # Calculate remaining timeout for sleep
                    sleep_time = self._retry_delay
                    if timeout is not None:
                        remaining_timeout = timeout - (time.time() - start_time)
                        sleep_time = min(sleep_time, remaining_timeout)
                        if sleep_time <= 0:
                            break

                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error acquiring Redis lock '{self._key}' on attempt {attempt + 1}: {e}")
                attempt += 1
                if attempt < max_attempts:
                    await asyncio.sleep(self._retry_delay)

        elapsed = time.time() - start_time
        logger.debug(f"Redis lock '{self._key}' acquisition failed after {elapsed:.3f}s ({attempt} attempts)")
        return False

    async def release(self) -> None:
        """
        Release the distributed lock from Redis.

        Uses Lua script for atomic check-and-delete to ensure only
        the lock holder can release the lock.
        """
        if not self._is_locked:
            logger.warning(f"Redis lock '{self._key}' is not held by this instance")
            return

        if not self._lock_value:
            logger.error(f"Redis lock '{self._key}' has no lock value, cannot release safely")
            return

        try:
            redis_client = await self._get_redis_client()

            # Use Lua script for atomic release - simplified to always use eval
            result = await redis_client.eval(
                self.RELEASE_SCRIPT,
                1,  # Number of keys
                self._key,  # KEYS[1]
                self._lock_value,  # ARGV[1]
            )

            if result == 1:
                logger.debug(f"Redis lock '{self._key}' released successfully")
            else:
                logger.warning(f"Redis lock '{self._key}' was not released (may have expired or been released already)")

        except Exception as e:
            logger.error(f"Error releasing Redis lock '{self._key}': {e}")
        finally:
            # Clear local state regardless of Redis operation result
            self._lock_value = None
            self._is_locked = False

    def is_locked(self) -> bool:
        """
        Check if the lock is currently held by this instance.

        Note: This only checks local state. The actual Redis key might
        have expired. For distributed scenarios, consider this a hint only.
        """
        return self._is_locked

    def get_name(self) -> str:
        """Get the name/identifier of the lock."""
        return self._name

    async def __aenter__(self) -> "RedisLock":
        """Async context manager entry."""
        success = await self.acquire()
        if not success:
            raise LockAcquisitionError(f"Failed to acquire Redis lock '{self._key}'")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()

    async def close(self) -> None:
        """Close and clean up resources. Connection pool is managed globally."""
        if self._is_locked:
            await self.release()
        # Note: We don't close the Redis client here since it's shared via connection manager

    def __del__(self):
        """Destructor to ensure cleanup."""
        # Use getattr to safely check attributes that may not exist if __init__ failed
        if getattr(self, "_is_locked", False):
            key = getattr(self, "_key", "unknown")
            logger.warning(
                f"Redis lock '{key}' is being garbage collected while still held. "
                f"Make sure to call release() or use context manager."
            )


# NOTE: This implementation might have issues if renewal fails; ensure your use case can tolerate such problems.
@asynccontextmanager
async def redis_lock_with_renewal(lock: RedisLock, renewal_interval: int = 10):
    """
    A context manager specifically for RedisLock that adds watchdog renewal.
    It does not modify the LockProtocol.
    """
    if not isinstance(lock, RedisLock):
        raise TypeError("This context manager only works with RedisLock instances.")

    watchdog_task = None
    is_active = True

    async def watchdog():
        """Periodically renews the lock."""
        lock_key = lock._key
        lock_value = lock._lock_value
        expire_time = lock._expire_time
        redis_client = await lock._get_redis_client()

        while is_active:
            await asyncio.sleep(renewal_interval)
            if not is_active:
                break
            try:
                result = await redis_client.eval(
                    RedisLock.RENEW_SCRIPT,
                    1,
                    lock_key,
                    lock_value,
                    expire_time,
                )
                if result != 1:
                    logger.error(f"Lock '{lock_key}' lost during renewal. Watchdog stopping.")
                    lock._is_locked = False  # Mark lock as lost, for the main loop to detect
                    break
                else:
                    logger.debug(f"Lock '{lock_key}' renewed successfully.")
            except Exception as e:
                logger.error(f"Error renewing lock '{lock_key}': {e}")
                break

    try:
        if not await lock.acquire():
            raise LockAcquisitionError(f"Failed to acquire lock '{lock.get_name()}'")

        watchdog_task = asyncio.create_task(watchdog())
        yield lock
    finally:
        # Stop the watchdog
        is_active = False
        if watchdog_task:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass  # Expected behavior

        # Release the lock if it's still held by this instance
        if lock.is_locked():
            await lock.release()
