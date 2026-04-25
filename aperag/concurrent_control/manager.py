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
Lock manager and factory functions for the concurrent control system.

This module contains the LockManager class and factory functions for creating
and managing lock instances across the application. Production defaults are
Redis-backed so locks work across workers and processes.
"""

import os
import threading
from typing import Dict, Optional

import redis.asyncio as async_redis

from .protocols import LockProtocol
from .redis_lock import RedisLock
from .threading_lock import ThreadingLock

LOCK_TYPE_ENV = "APERAG_LOCK_TYPE"
DEFAULT_LOCK_TYPE = "redis"
LOCK_TYPE_REDIS = "redis"
LOCK_TYPE_THREADING = "threading"
SUPPORTED_LOCK_TYPES = {LOCK_TYPE_REDIS, LOCK_TYPE_THREADING}


def _resolve_lock_type(lock_type: Optional[str] = None) -> str:
    resolved = (lock_type or os.getenv(LOCK_TYPE_ENV, DEFAULT_LOCK_TYPE)).strip().lower()
    if resolved not in SUPPORTED_LOCK_TYPES:
        raise ValueError(
            f"Unknown lock type: {resolved}. Use '{LOCK_TYPE_REDIS}' or '{LOCK_TYPE_THREADING}'. "
            f"Set {LOCK_TYPE_ENV} only for process-wide default override."
        )
    return resolved


class LockManager:
    """
    Lock manager for creating and managing lock instances.

    This class provides a centralized way to create and manage different types
    of locks with consistent configuration and naming conventions.
    """

    def __init__(self):
        """Initialize the lock manager."""
        self._locks: Dict[str, LockProtocol] = {}
        self._lock = threading.Lock()  # Thread safety for _locks dict operations

    def create_threading_lock(self, name: str = None) -> ThreadingLock:
        """
        Create a threading lock for single-process scenarios.

        Args:
            name: Optional name for the lock

        Returns:
            ThreadingLock instance
        """
        return ThreadingLock(name=name)

    def create_redis_lock(
        self,
        key: str,
        expire_time: int = 120,
        retry_times: int = 3,
        retry_delay: float = 0.1,
        name: Optional[str] = None,
        redis_client: Optional[async_redis.Redis] = None,
    ) -> RedisLock:
        """
        Create a Redis lock for distributed scenarios.

        Args:
            key: Redis key for the lock (required)
            expire_time: Lock expiration time in seconds
            retry_times: Number of retry attempts
            retry_delay: Delay between retry attempts
            name: Optional lock name
            redis_client: Optional Redis client override for tests or explicit callers

        Returns:
            RedisLock instance
        """
        return RedisLock(
            key=key,
            expire_time=expire_time,
            retry_times=retry_times,
            retry_delay=retry_delay,
            name=name,
            redis_client=redis_client,
        )

    def create_distributed_lock(
        self,
        name: str,
        ttl: int = 120,
        redis_client: Optional[async_redis.Redis] = None,
        retry_times: int = 3,
        retry_delay: float = 0.1,
    ) -> RedisLock:
        """
        Create a Redis-backed production lock.

        Args:
            name: Stable distributed lock name. This becomes the Redis key.
            ttl: Lock expiration time in seconds.
            redis_client: Optional Redis client override.
            retry_times: Number of retry attempts.
            retry_delay: Delay between retry attempts.

        Returns:
            RedisLock instance.
        """
        return self.create_redis_lock(
            key=name,
            expire_time=ttl,
            retry_times=retry_times,
            retry_delay=retry_delay,
            name=name,
            redis_client=redis_client,
        )

    def get_or_create_lock(self, lock_id: str, lock_type: Optional[str] = None, **kwargs) -> LockProtocol:
        """
        Get an existing lock or create a new one.

        Args:
            lock_id: Unique identifier for the lock
            lock_type: Type of lock ('redis' or 'threading'). Defaults to APERAG_LOCK_TYPE or redis.
            **kwargs: Additional arguments for lock creation

        Returns:
            Lock instance
        """
        resolved_lock_type = _resolve_lock_type(lock_type)
        with self._lock:  # Thread-safe check-and-set operation
            # Check if lock already exists
            if lock_id in self._locks:
                return self._locks[lock_id]

            # Create new lock
            if resolved_lock_type == LOCK_TYPE_THREADING:
                lock = self.create_threading_lock(name=kwargs.get("name", lock_id))
            elif resolved_lock_type == LOCK_TYPE_REDIS:
                # For Redis locks, use lock_id as the key if no key is provided
                key = kwargs.get("key", lock_id)
                lock = self.create_redis_lock(key=key, **{k: v for k, v in kwargs.items() if k != "key"})
            else:
                raise ValueError(f"Unknown lock type: {resolved_lock_type}")

            # Store the new lock
            self._locks[lock_id] = lock
            return lock

    def remove_lock(self, lock_id: str) -> bool:
        """
        Remove a lock from the manager.

        Args:
            lock_id: Unique identifier for the lock

        Returns:
            True if lock was removed, False if not found
        """
        with self._lock:  # Thread-safe check-and-delete operation
            if lock_id in self._locks:
                del self._locks[lock_id]
                return True
            return False

    def list_locks(self) -> Dict[str, str]:
        """
        List all managed locks.

        Returns:
            Dict mapping lock_id to lock type
        """
        with self._lock:  # Thread-safe read operation
            return {lock_id: type(lock).__name__ for lock_id, lock in self._locks.items()}


# Default global lock manager instance for convenience
default_lock_manager = LockManager()


def create_distributed_lock(
    name: str,
    ttl: int = 120,
    redis_client: Optional[async_redis.Redis] = None,
    retry_times: int = 3,
    retry_delay: float = 0.1,
) -> RedisLock:
    """
    Create a Redis-backed production lock.

    This is the preferred public API for business code that needs mutual
    exclusion across API workers, Celery workers, or multiple containers.
    """
    return default_lock_manager.create_distributed_lock(
        name=name,
        ttl=ttl,
        redis_client=redis_client,
        retry_times=retry_times,
        retry_delay=retry_delay,
    )


def create_lock(lock_type: Optional[str] = None, **kwargs) -> LockProtocol:
    """
    Create a new lock instance.

    If a 'name' is provided, the lock will be automatically registered
    in the default lock manager for later retrieval.

    Args:
        lock_type: Type of lock to create ('redis' or 'threading'). Defaults to APERAG_LOCK_TYPE or redis.
        name: Optional lock name (if provided, auto-registered for retrieval)
        **kwargs: Additional arguments passed to lock constructor

    Returns:
        LockProtocol: Lock implementation instance

    Examples:
        # Create named Redis lock (automatically managed)
        managed_lock = create_lock("redis", key="my_app:lock", name="my_lock")
        same_lock = get_lock("my_lock")  # Returns same instance

        # Create a local single-process lock explicitly
        local_lock = create_lock("threading", name="test_lock")
    """
    resolved_lock_type = _resolve_lock_type(lock_type)
    if resolved_lock_type == LOCK_TYPE_REDIS and "key" not in kwargs and kwargs.get("name"):
        kwargs["key"] = kwargs["name"]

    if resolved_lock_type == LOCK_TYPE_THREADING:
        lock_instance = ThreadingLock(**kwargs)
    elif resolved_lock_type == LOCK_TYPE_REDIS:
        lock_instance = RedisLock(**kwargs)
    else:
        raise ValueError(f"Unknown lock type: {resolved_lock_type}. Use 'redis' or 'threading'.")

    # Auto-register named locks in default manager (thread-safe)
    lock_name = kwargs.get("name") or getattr(lock_instance, "_name", None)
    if lock_name and hasattr(lock_instance, "_name"):
        with default_lock_manager._lock:
            # Only register if not already exists (avoid overwriting existing locks)
            if lock_name not in default_lock_manager._locks:
                default_lock_manager._locks[lock_name] = lock_instance

    return lock_instance


def get_lock(name: str) -> Optional[LockProtocol]:
    """
    Get a lock from the default manager by name.

    Args:
        name: Name of the lock to retrieve

    Returns:
        The lock instance if found, None otherwise

    Examples:
        # Create a named lock
        create_lock("threading", name="my_operation")

        # Later retrieve it
        lock = get_lock("my_operation")
        if lock:
            async with lock:
                await work()
    """
    with default_lock_manager._lock:  # Thread-safe read operation
        return default_lock_manager._locks.get(name)


def get_or_create_lock(name: str, lock_type: Optional[str] = None, **kwargs) -> LockProtocol:
    """
    Get an existing lock by name or create a new one.

    This is a convenience function that combines get_lock and create_lock.

    Args:
        name: Name of the lock
        lock_type: Type of lock to create if not found. Defaults to APERAG_LOCK_TYPE or redis.
        **kwargs: Additional arguments for lock creation

    Returns:
        Lock instance (existing or newly created)

    Examples:
        # Get existing or create new Redis-backed production lock
        lock = get_or_create_lock("database_ops")

        # All subsequent calls return the same instance
        same_lock = get_or_create_lock("database_ops")
        assert lock is same_lock

        # Single-process local locks must opt in explicitly
        local_lock = get_or_create_lock("local_ops", "threading")
    """
    # Use the LockManager's thread-safe get_or_create_lock method
    # This ensures atomic check-and-create operation
    kwargs["name"] = name
    return default_lock_manager.get_or_create_lock(name, lock_type, **kwargs)


def get_default_lock_manager() -> LockManager:
    """
    Get the default global lock manager instance.

    Returns:
        LockManager: Default lock manager instance
    """
    return default_lock_manager
