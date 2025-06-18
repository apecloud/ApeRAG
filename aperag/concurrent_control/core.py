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
Core implementation of the universal concurrent control system.

This module contains the main lock implementations and management classes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Dict, Optional

logger = logging.getLogger(__name__)


class LockProtocol(ABC):
    """
    Abstract interface for concurrent locks.

    This protocol defines the common interface that all lock implementations
    must follow to ensure compatibility across different deployment scenarios.
    """

    @abstractmethod
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock asynchronously.

        Args:
            timeout: Maximum time to wait for the lock (seconds).
                    None means wait indefinitely.

        Returns:
            True if lock was acquired successfully, False if timeout occurred.
        """
        pass

    @abstractmethod
    async def release(self) -> None:
        """Release the lock asynchronously."""
        pass

    @abstractmethod
    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        pass

    @abstractmethod
    async def __aenter__(self) -> LockProtocol:
        """Async context manager entry."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass


class ThreadingLock(LockProtocol):
    """
    Threading-based lock implementation using asyncio.to_thread wrapper.

    This implementation uses a threading.Lock wrapped with asyncio.to_thread
    to provide async compatibility while supporting both coroutine and thread
    concurrency scenarios within a single process.

    Features:
    - Works in single-process multi-coroutine environments (celery --pool=solo)
    - Works in single-process multi-thread environments (celery --pool=threads)
    - Does NOT work across multiple processes (celery --pool=prefork)
    - Non-blocking for the event loop (uses background thread pool)

    Performance:
    - Higher overhead than asyncio.Lock but supports broader concurrency models
    - Lower overhead than distributed locks for single-process scenarios
    """

    def __init__(self, name: str = None):
        """
        Initialize the threading lock.

        Args:
            name: Descriptive name for the lock (used in logging).
                 If None, a UUID will be generated.
        """
        self._lock = threading.Lock()
        self._name = name or f"threading_lock_{uuid.uuid4().hex[:8]}"
        self._holder_info: Optional[str] = None

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock using asyncio.to_thread to avoid blocking the event loop.

        Args:
            timeout: Maximum time to wait for the lock (seconds).
                    None means wait indefinitely.

        Returns:
            True if lock was acquired, False if timeout occurred.
        """

        def _sync_acquire():
            # Use threading.Lock.acquire with timeout parameter
            if timeout is not None:
                acquired = self._lock.acquire(blocking=True, timeout=timeout)
            else:
                acquired = self._lock.acquire(blocking=True)

            if acquired:
                self._holder_info = f"Thread-{threading.get_ident()}"
                logger.debug(f"Lock '{self._name}' acquired by {self._holder_info}")

            return acquired

        try:
            # Use asyncio.to_thread to run the acquire in background thread
            result = await asyncio.to_thread(_sync_acquire)
            return result

        except Exception as e:
            logger.error(f"Error acquiring lock '{self._name}': {e}")
            return False

    async def release(self) -> None:
        """Release the lock using asyncio.to_thread."""

        def _sync_release():
            try:
                self._lock.release()
                logger.debug(f"Lock '{self._name}' released by {self._holder_info}")
                self._holder_info = None
                return True
            except Exception as e:
                logger.error(f"Error releasing lock '{self._name}': {e}")
                return False

        await asyncio.to_thread(_sync_release)

    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.locked()

    async def __aenter__(self) -> ThreadingLock:
        """Async context manager entry."""
        success = await self.acquire()
        if not success:
            raise RuntimeError(f"Failed to acquire lock '{self._name}'")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()


class RedisLock(LockProtocol):
    """
    Redis-based distributed lock implementation.

    This implementation will use Redis for distributed locking across
    multiple processes, containers, or machines.

    TODO: Implement Redis-based distributed locking

    Features (when implemented):
    - Works across multiple processes (celery --pool=prefork)
    - Works across multiple machines/containers
    - Works with any task queue (Celery, Prefect, etc.)
    - Automatic lock expiration to prevent deadlocks
    - Retry mechanisms for lock acquisition

    Performance considerations:
    - Network round-trip overhead for each lock operation
    - Redis server becomes a critical dependency
    - Higher latency compared to in-process locks
    """

    def __init__(
        self,
        key: str,
        redis_url: str = "redis://localhost:6379",
        expire_time: int = 30,
        retry_times: int = 3,
        retry_delay: float = 0.1,
    ):
        """
        Initialize the Redis lock.

        Args:
            key: Redis key for the lock (required)
            redis_url: Redis connection URL
            expire_time: Lock expiration time in seconds (prevents deadlocks)
            retry_times: Number of retry attempts for lock acquisition
            retry_delay: Delay between retry attempts in seconds
        """
        if not key:
            raise ValueError("Redis lock key is required")

        self._key = key
        self._redis_url = redis_url
        self._expire_time = expire_time
        self._retry_times = retry_times
        self._retry_delay = retry_delay
        self._redis_client = None
        self._lock_value: Optional[str] = None

        # TODO: Initialize Redis client
        logger.warning("RedisLock is not yet implemented. Use ThreadingLock instead.")

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the distributed lock from Redis.

        TODO: Implement Redis SET NX EX pattern for distributed locking
        """
        raise NotImplementedError("RedisLock is not yet implemented")

    async def release(self) -> None:
        """
        Release the distributed lock from Redis.

        TODO: Implement safe lock release using Lua script
        """
        raise NotImplementedError("RedisLock is not yet implemented")

    def is_locked(self) -> bool:
        """
        Check if the lock is currently held.

        TODO: Implement Redis EXISTS check
        """
        raise NotImplementedError("RedisLock is not yet implemented")

    async def __aenter__(self) -> RedisLock:
        """Async context manager entry."""
        success = await self.acquire()
        if not success:
            raise RuntimeError(f"Failed to acquire Redis lock '{self._key}'")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()


class LockManager:
    """
    Lock manager for creating and managing lock instances.

    This class provides a centralized way to create and manage different types
    of locks with consistent configuration and naming conventions.
    """

    def __init__(self, default_redis_url: str = "redis://localhost:6379"):
        """
        Initialize the lock manager.

        Args:
            default_redis_url: Default Redis URL for distributed locks
        """
        self.default_redis_url = default_redis_url
        self._locks: Dict[str, LockProtocol] = {}

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
        self, key: str, redis_url: str = None, expire_time: int = 30, retry_times: int = 3, retry_delay: float = 0.1
    ) -> RedisLock:
        """
        Create a Redis lock for distributed scenarios.

        Args:
            key: Redis key for the lock (required)
            redis_url: Redis connection URL (uses default if None)
            expire_time: Lock expiration time in seconds
            retry_times: Number of retry attempts
            retry_delay: Delay between retry attempts

        Returns:
            RedisLock instance
        """
        return RedisLock(
            key=key,
            redis_url=redis_url or self.default_redis_url,
            expire_time=expire_time,
            retry_times=retry_times,
            retry_delay=retry_delay,
        )

    def get_or_create_lock(self, lock_id: str, lock_type: str = "threading", **kwargs) -> LockProtocol:
        """
        Get an existing lock or create a new one.

        Args:
            lock_id: Unique identifier for the lock
            lock_type: Type of lock ('threading' or 'redis')
            **kwargs: Additional arguments for lock creation

        Returns:
            Lock instance
        """
        if lock_id in self._locks:
            return self._locks[lock_id]

        if lock_type == "threading":
            lock = self.create_threading_lock(name=kwargs.get("name", lock_id))
        elif lock_type == "redis":
            # For Redis locks, use lock_id as the key if no key is provided
            key = kwargs.get("key", lock_id)
            lock = self.create_redis_lock(key=key, **{k: v for k, v in kwargs.items() if k != "key"})
        else:
            raise ValueError(f"Unknown lock type: {lock_type}")

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
        return {lock_id: type(lock).__name__ for lock_id, lock in self._locks.items()}


def create_lock(lock_type: str = "threading", **kwargs) -> LockProtocol:
    """
    Create a new lock instance.

    If a 'name' is provided, the lock will be automatically registered
    in the default lock manager for later retrieval.

    Args:
        lock_type: Type of lock to create ('threading' or 'redis')
        name: Optional lock name (if provided, auto-registered for retrieval)
        **kwargs: Additional arguments passed to lock constructor

    Returns:
        LockProtocol: Lock implementation instance

    Examples:
        # Create anonymous lock (not managed)
        temp_lock = create_lock("threading")

        # Create named lock (automatically managed)
        managed_lock = create_lock("threading", name="my_lock")
        same_lock = get_lock("my_lock")  # Returns same instance

        # Create Redis lock
        redis_lock = create_lock("redis", key="my_app:lock", redis_url="redis://localhost:6379")
    """
    if lock_type == "threading":
        lock_instance = ThreadingLock(**kwargs)
    elif lock_type == "redis":
        lock_instance = RedisLock(**kwargs)
    else:
        raise ValueError(f"Unknown lock type: {lock_type}. Use 'threading' or 'redis'.")

    # Auto-register named locks in default manager
    lock_name = kwargs.get("name") or getattr(lock_instance, "_name", None)
    if lock_name and hasattr(lock_instance, "_name"):
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
    return default_lock_manager._locks.get(name)


def get_or_create_lock(name: str, lock_type: str = "threading", **kwargs) -> LockProtocol:
    """
    Get an existing lock by name or create a new one.

    This is a convenience function that combines get_lock and create_lock.

    Args:
        name: Name of the lock
        lock_type: Type of lock to create if not found
        **kwargs: Additional arguments for lock creation

    Returns:
        Lock instance (existing or newly created)

    Examples:
        # Get existing or create new
        lock = get_or_create_lock("database_ops", "threading")

        # All subsequent calls return the same instance
        same_lock = get_or_create_lock("database_ops", "threading")
        assert lock is same_lock
    """
    existing_lock = get_lock(name)
    if existing_lock:
        return existing_lock

    # Create new lock with the specified name
    kwargs["name"] = name
    return create_lock(lock_type, **kwargs)


@asynccontextmanager
async def lock_context(lock: LockProtocol, timeout: Optional[float] = None) -> AsyncContextManager[None]:
    """
    Convenient async context manager for operations that need locking.

    Usage:
        async with lock_context(my_lock):
            # Your protected operations here
            await some_critical_operation()

    Args:
        lock: Lock instance to use
        timeout: Maximum time to wait for lock acquisition
    """
    success = await lock.acquire(timeout=timeout)
    if not success:
        raise TimeoutError(f"Failed to acquire lock within {timeout} seconds")

    try:
        yield
    finally:
        await lock.release()


# Default global lock manager instance for convenience
default_lock_manager = LockManager()


def get_default_lock_manager() -> LockManager:
    """
    Get the default global lock manager instance.

    Returns:
        LockManager: Default lock manager instance
    """
    return default_lock_manager


class MultiLock:
    """
    Multi-lock manager for acquiring multiple locks in a sorted order to prevent deadlocks.

    This class ensures that multiple locks are always acquired in the same order
    (sorted by lock name) across all workers, preventing circular wait conditions
    that could lead to deadlocks.

    Example:
        locks = [
            get_or_create_lock("entity:Apple:workspace1"),
            get_or_create_lock("entity:Google:workspace1"),
            get_or_create_lock("entity:Microsoft:workspace1")
        ]

        async with MultiLock(locks):
            # All locks are held here
            await process_entities()
        # All locks are automatically released
    """

    # Class-level logger for MultiLock operations
    _logger = logging.getLogger(__name__ + ".MultiLock")

    def __init__(self, locks: list[LockProtocol]):
        """
        Initialize MultiLock with a list of locks.

        Args:
            locks: List of lock instances to manage
        """
        # Sort locks by name to ensure consistent acquisition order
        self._locks = sorted(locks, key=lambda lock: getattr(lock, "_name", str(id(lock))))
        self._acquired_locks: list[LockProtocol] = []
        self._lock_names = [getattr(lock, "_name", str(id(lock))) for lock in self._locks]

    async def acquire_all(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire all locks in sorted order.

        Args:
            timeout: Maximum time to wait for each lock

        Returns:
            True if all locks were acquired, False otherwise
        """
        import time

        # Log all locks that are about to be acquired
        self._logger.info(f"MultiLock attempting to acquire {len(self._locks)} locks: {self._lock_names}")

        # Record start time for measuring wait duration
        start_time = time.time()

        for i, lock in enumerate(self._locks):
            try:
                lock_name = self._lock_names[i]
                self._logger.debug(f"Acquiring lock {i + 1}/{len(self._locks)}: {lock_name}")

                success = await lock.acquire(timeout=timeout)
                if not success:
                    # Failed to acquire this lock, release all previously acquired
                    elapsed_time = time.time() - start_time
                    self._logger.warning(
                        f"MultiLock failed to acquire lock '{lock_name}' after {elapsed_time:.3f}s, "
                        f"releasing {len(self._acquired_locks)} already acquired locks"
                    )
                    await self._release_acquired()
                    return False

                self._acquired_locks.append(lock)
                self._logger.debug(f"Successfully acquired lock: {lock_name}")

            except Exception as e:
                elapsed_time = time.time() - start_time
                lock_name = self._lock_names[i] if i < len(self._lock_names) else "unknown"
                self._logger.error(f"Error acquiring lock '{lock_name}' in MultiLock after {elapsed_time:.3f}s: {e}")
                await self._release_acquired()
                return False

        # All locks acquired successfully
        elapsed_time = time.time() - start_time
        self._logger.info(
            f"MultiLock successfully acquired all {len(self._locks)} locks in {elapsed_time:.3f}s: {self._lock_names}"
        )
        return True

    async def release_all(self) -> None:
        """Release all locks in reverse order of acquisition."""
        if self._acquired_locks:
            self._logger.info(f"MultiLock releasing {len(self._acquired_locks)} locks")
        await self._release_acquired()

    async def _release_acquired(self) -> None:
        """Release all acquired locks in reverse order."""
        if not self._acquired_locks:
            return

        # Log the locks being released
        released_lock_names = []
        for lock in self._acquired_locks:
            released_lock_names.append(getattr(lock, "_name", str(id(lock))))

        self._logger.debug(f"Releasing locks in reverse order: {list(reversed(released_lock_names))}")

        # Release in reverse order to minimize contention
        for i, lock in enumerate(reversed(self._acquired_locks)):
            try:
                lock_name = getattr(lock, "_name", str(id(lock)))
                self._logger.debug(f"Releasing lock {i + 1}/{len(self._acquired_locks)}: {lock_name}")
                await lock.release()
            except Exception as e:
                lock_name = getattr(lock, "_name", str(id(lock)))
                self._logger.error(f"Error releasing lock '{lock_name}' in MultiLock: {e}")

        self._logger.info(f"MultiLock released all {len(self._acquired_locks)} locks successfully")
        self._acquired_locks.clear()

    async def __aenter__(self) -> "MultiLock":
        """Async context manager entry - acquire all locks."""
        success = await self.acquire_all()
        if not success:
            raise RuntimeError("Failed to acquire all locks in MultiLock")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - release all locks."""
        await self.release_all()
