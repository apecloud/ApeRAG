"""
Unit tests for LockManager implementation.

This module tests the LockManager functionality including lock creation,
management, and lifecycle operations.
"""

import pytest

from aperag.concurrent_control import LockManager, RedisLock, ThreadingLock, get_default_lock_manager


class TestLockManager:
    """Test suite for LockManager implementation."""

    def test_lock_manager_creation(self):
        """Test basic LockManager creation."""
        manager = LockManager()
        assert manager._locks == {}

    def test_create_threading_lock(self):
        """Test creating threading locks through manager."""
        manager = LockManager()

        # Create with name
        lock1 = manager.create_threading_lock(name="test_lock_1")
        assert isinstance(lock1, ThreadingLock)
        assert lock1._name == "test_lock_1"

        # Create without name
        lock2 = manager.create_threading_lock()
        assert isinstance(lock2, ThreadingLock)
        assert lock2._name.startswith("threading_lock_")

        # Locks should be different instances
        assert lock1 is not lock2

    def test_create_redis_lock(self):
        """Test creating Redis locks through manager."""
        manager = LockManager()

        # Create with all parameters
        lock1 = manager.create_redis_lock(key="test_key_1", expire_time=60, retry_times=5, retry_delay=0.2)
        assert isinstance(lock1, RedisLock)
        assert lock1._key == "test_key_1"
        assert lock1._expire_time == 60
        assert lock1._retry_times == 5
        assert lock1._retry_delay == 0.2

        # Create with defaults
        lock2 = manager.create_redis_lock(key="test_key_2")
        assert isinstance(lock2, RedisLock)
        assert lock2._key == "test_key_2"
        assert lock2._expire_time == 120  # Default
        assert lock2._retry_times == 3  # Default
        assert lock2._retry_delay == 0.1  # Default

    def test_get_or_create_lock_threading(self):
        """Test get_or_create_lock for threading locks."""
        manager = LockManager()

        # Create new lock
        lock1 = manager.get_or_create_lock("test_lock", "threading", name="custom_name")
        assert isinstance(lock1, ThreadingLock)
        assert lock1._name == "custom_name"

        # Get existing lock (should return same instance)
        lock2 = manager.get_or_create_lock("test_lock", "threading")
        assert lock1 is lock2

        # Different lock_id should create new lock
        lock3 = manager.get_or_create_lock("different_lock", "threading")
        assert lock3 is not lock1
        assert isinstance(lock3, ThreadingLock)

    def test_get_or_create_lock_redis(self):
        """Test get_or_create_lock for Redis locks."""
        manager = LockManager()

        # Create new Redis lock
        lock1 = manager.get_or_create_lock("redis_lock", "redis", key="custom_key", expire_time=120)
        assert isinstance(lock1, RedisLock)
        assert lock1._key == "custom_key"
        assert lock1._expire_time == 120

        # Get existing lock
        lock2 = manager.get_or_create_lock("redis_lock", "redis")
        assert lock1 is lock2

        # Different lock_id with default key
        lock3 = manager.get_or_create_lock("redis_lock_2", "redis")
        assert isinstance(lock3, RedisLock)
        assert lock3._key == "redis_lock_2"  # Uses lock_id as key

    def test_get_or_create_lock_invalid_type(self):
        """Test get_or_create_lock with invalid lock type."""
        manager = LockManager()

        with pytest.raises(ValueError, match="Unknown lock type: invalid"):
            manager.get_or_create_lock("test", "invalid")

    def test_remove_lock(self):
        """Test removing locks from manager."""
        manager = LockManager()

        # Create some locks
        manager.get_or_create_lock("lock1", "threading")
        manager.get_or_create_lock("lock2", "threading")

        assert len(manager._locks) == 2

        # Remove existing lock
        result = manager.remove_lock("lock1")
        assert result is True
        assert len(manager._locks) == 1
        assert "lock1" not in manager._locks
        assert "lock2" in manager._locks

        # Remove non-existing lock
        result = manager.remove_lock("non_existing")
        assert result is False
        assert len(manager._locks) == 1

    def test_list_locks(self):
        """Test listing managed locks."""
        manager = LockManager()

        # Initially empty
        locks_list = manager.list_locks()
        assert locks_list == {}

        # Add some locks
        manager.get_or_create_lock("threading_lock", "threading")
        manager.get_or_create_lock("redis_lock", "redis", key="test_key")

        locks_list = manager.list_locks()
        assert len(locks_list) == 2
        assert locks_list["threading_lock"] == "ThreadingLock"
        assert locks_list["redis_lock"] == "RedisLock"

    @pytest.mark.asyncio
    async def test_managed_locks_functionality(self):
        """Test that managed locks work correctly."""
        manager = LockManager()

        # Create and use a threading lock
        lock = manager.get_or_create_lock("functional_test", "threading")

        # Test basic functionality
        assert not lock.is_locked()

        async with lock:
            assert lock.is_locked()

        assert not lock.is_locked()

    def test_manager_isolation(self):
        """Test that different managers are isolated."""
        manager1 = LockManager()
        manager2 = LockManager()

        # Create locks with same ID in different managers
        lock1 = manager1.get_or_create_lock("same_id", "threading")
        lock2 = manager2.get_or_create_lock("same_id", "threading")

        # Should be different instances
        assert lock1 is not lock2
        assert len(manager1._locks) == 1
        assert len(manager2._locks) == 1


class TestGlobalLockManager:
    """Test suite for global lock manager functionality."""

    def test_get_default_lock_manager(self):
        """Test getting the default global lock manager."""
        manager1 = get_default_lock_manager()
        manager2 = get_default_lock_manager()

        # Should return the same instance
        assert manager1 is manager2
        assert isinstance(manager1, LockManager)

    def test_global_manager_persistence(self):
        """Test that global manager persists locks across calls."""
        manager1 = get_default_lock_manager()
        manager1.get_or_create_lock("global_test_lock", "threading")

        manager2 = get_default_lock_manager()
        locks_list = manager2.list_locks()

        assert "global_test_lock" in locks_list
        assert locks_list["global_test_lock"] == "ThreadingLock"

    @pytest.mark.asyncio
    async def test_global_manager_concurrent_access(self):
        """Test concurrent access to global manager."""
        manager = get_default_lock_manager()

        # Create a lock through global manager
        lock = manager.get_or_create_lock("concurrent_global_test", "threading")

        async def worker(worker_id: int):
            # Get the same lock from global manager
            worker_lock = manager.get_or_create_lock("concurrent_global_test", "threading")
            assert worker_lock is lock  # Should be same instance

            async with worker_lock:
                return f"worker_{worker_id}_completed"

        # Run multiple workers
        import asyncio

        results = await asyncio.gather(*[worker(i) for i in range(3)])

        assert len(results) == 3
        assert all("completed" in result for result in results)


class TestLockManagerEdgeCases:
    """Test edge cases and error conditions for LockManager."""

    def test_manager_with_empty_lock_id(self):
        """Test manager behavior with empty lock ID."""
        manager = LockManager()

        # Empty string as lock_id should work but not be practical
        lock = manager.get_or_create_lock("", "threading")
        assert isinstance(lock, ThreadingLock)
        assert "" in manager._locks

    def test_manager_with_special_characters_in_lock_id(self):
        """Test manager with special characters in lock IDs."""
        manager = LockManager()

        special_ids = [
            "lock:with:colons",
            "lock-with-dashes",
            "lock_with_underscores",
            "lock.with.dots",
            "lock with spaces",
            "lock/with/slashes",
        ]

        for lock_id in special_ids:
            lock = manager.get_or_create_lock(lock_id, "threading")
            assert isinstance(lock, ThreadingLock)
            assert lock_id in manager._locks

    def test_redis_lock_parameter_validation(self):
        """Test Redis lock parameter validation through manager."""
        manager = LockManager()

        # Missing key should raise error
        with pytest.raises(ValueError, match="Redis lock key is required"):
            manager.create_redis_lock(key="")

        with pytest.raises(ValueError, match="Redis lock key is required"):
            manager.create_redis_lock(key=None)

    def test_manager_memory_efficiency(self):
        """Test that manager doesn't leak memory with many locks."""
        manager = LockManager()

        # Create many locks
        num_locks = 100
        for i in range(num_locks):
            manager.get_or_create_lock(f"lock_{i}", "threading")

        assert len(manager._locks) == num_locks

        # Remove half of them
        for i in range(0, num_locks, 2):
            manager.remove_lock(f"lock_{i}")

        assert len(manager._locks) == num_locks // 2

        # Verify remaining locks
        remaining_locks = manager.list_locks()
        for i in range(1, num_locks, 2):
            assert f"lock_{i}" in remaining_locks

    @pytest.mark.asyncio
    async def test_manager_with_mixed_lock_types(self):
        """Test manager handling both threading and Redis locks."""
        manager = LockManager()

        # Create locks of different types
        threading_lock = manager.get_or_create_lock("threading_lock", "threading")
        redis_lock = manager.get_or_create_lock("redis_lock", "redis", key="redis_key")

        assert isinstance(threading_lock, ThreadingLock)
        assert isinstance(redis_lock, RedisLock)

        locks_list = manager.list_locks()
        assert len(locks_list) == 2
        assert locks_list["threading_lock"] == "ThreadingLock"
        assert locks_list["redis_lock"] == "RedisLock"

        # Test threading lock functionality
        async with threading_lock:
            assert threading_lock.is_locked()

        assert not threading_lock.is_locked()

    def test_manager_kwargs_handling(self):
        """Test proper handling of kwargs in get_or_create_lock."""
        manager = LockManager()

        # Test threading lock with custom name
        lock1 = manager.get_or_create_lock("test1", "threading", name="custom_threading_name")
        assert lock1._name == "custom_threading_name"

        # Test Redis lock with custom parameters
        lock2 = manager.get_or_create_lock("test2", "redis", key="custom_redis_key", expire_time=300, retry_times=10)
        assert lock2._key == "custom_redis_key"
        assert lock2._expire_time == 300
        assert lock2._retry_times == 10
