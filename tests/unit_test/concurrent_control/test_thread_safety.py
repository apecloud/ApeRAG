"""
Thread safety tests for concurrent_control module.

This module contains tests to verify that the lock manager and related
components are thread-safe and handle concurrent access correctly.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aperag.concurrent_control import (
    LockManager,
    create_lock,
    get_default_lock_manager,
    get_lock,
    get_or_create_lock,
)


class TestLockManagerThreadSafety:
    """Test thread safety of LockManager operations."""

    def test_concurrent_get_or_create_same_lock(self):
        """Test that concurrent get_or_create operations return same instance."""
        manager = LockManager()
        results = []

        def worker():
            lock = manager.get_or_create_lock("concurrent_test", "threading")
            results.append(id(lock))
            return lock

        # Run 10 threads concurrently trying to get/create the same lock
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            locks = [future.result() for future in futures]

        # All threads should get the same lock instance
        assert len(set(results)) == 1, "Multiple lock instances created for same lock_id"
        assert all(lock is locks[0] for lock in locks), "Not all threads got the same lock instance"

    def test_concurrent_get_or_create_different_locks(self):
        """Test concurrent creation of different locks."""
        manager = LockManager()
        results = {}
        lock = threading.Lock()

        def worker(lock_id):
            lock_instance = manager.get_or_create_lock(f"test_lock_{lock_id}", "threading")
            with lock:
                results[lock_id] = id(lock_instance)

        # Create 20 different locks concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            for future in futures:
                future.result()

        # All locks should be different instances
        lock_ids = list(results.values())
        assert len(set(lock_ids)) == 20, "Some locks were not created or duplicated"

    def test_concurrent_remove_lock(self):
        """Test concurrent lock removal."""
        manager = LockManager()

        # Pre-create some locks
        for i in range(10):
            manager.get_or_create_lock(f"remove_test_{i}", "threading")

        removal_results = []
        lock = threading.Lock()

        def worker(lock_id):
            result = manager.remove_lock(f"remove_test_{lock_id}")
            with lock:
                removal_results.append((lock_id, result))

        # Try to remove locks concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()

        # Each lock should be removed exactly once
        successful_removals = [lock_id for lock_id, result in removal_results if result]
        assert len(successful_removals) == 10, "Not all locks were removed successfully"
        assert len(set(successful_removals)) == 10, "Some locks were removed multiple times"

    def test_concurrent_list_locks(self):
        """Test that list_locks is thread-safe during concurrent modifications."""
        manager = LockManager()
        exceptions = []

        def creator_worker(worker_id):
            try:
                for i in range(5):
                    manager.get_or_create_lock(f"list_test_{worker_id}_{i}", "threading")
                    time.sleep(0.001)  # Small delay to interleave operations
            except Exception as e:
                exceptions.append(e)

        def lister_worker():
            try:
                for _ in range(10):
                    manager.list_locks()
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

        # Run creators and listers concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            creator_futures = [executor.submit(creator_worker, i) for i in range(3)]
            lister_futures = [executor.submit(lister_worker) for _ in range(3)]

            for future in creator_futures + lister_futures:
                future.result()

        # No exceptions should occur
        assert not exceptions, f"Exceptions occurred during concurrent operations: {exceptions}"

        # Verify final state
        final_locks = manager.list_locks()
        assert len(final_locks) == 15, "Incorrect number of locks created"


class TestGlobalManagerThreadSafety:
    """Test thread safety of global manager functions."""

    def test_concurrent_global_get_or_create(self):
        """Test concurrent access to global get_or_create_lock function."""
        results = []

        def worker():
            lock = get_or_create_lock("global_concurrent_test", "threading")
            results.append(id(lock))
            return lock

        # Clear any existing lock first
        manager = get_default_lock_manager()
        manager.remove_lock("global_concurrent_test")

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(worker) for _ in range(15)]
            locks = [future.result() for future in futures]

        # All should get the same instance
        assert len(set(results)) == 1, "Multiple instances created in global manager"
        assert all(lock is locks[0] for lock in locks), "Not all threads got same instance"

    def test_concurrent_create_and_get(self):
        """Test concurrent create_lock and get_lock operations."""
        # Clear any existing locks
        manager = get_default_lock_manager()
        manager.remove_lock("create_get_test")

        create_results = []
        get_results = []

        def creator():
            lock = create_lock("threading", name="create_get_test")
            create_results.append(id(lock))
            return lock

        def getter():
            time.sleep(0.001)  # Small delay to let create_lock run first
            lock = get_lock("create_get_test")
            if lock:
                get_results.append(id(lock))
            return lock

        # Run one creator and multiple getters
        with ThreadPoolExecutor(max_workers=10) as executor:
            creator_future = executor.submit(creator)
            getter_futures = [executor.submit(getter) for _ in range(9)]

            created_lock = creator_future.result()
            gotten_locks = [future.result() for future in getter_futures]

        # Creator should succeed
        assert created_lock is not None
        assert len(create_results) == 1

        # Getters should get the same instance (or None if they ran before creator)
        non_none_locks = [lock for lock in gotten_locks if lock is not None]
        if non_none_locks:  # Some getters succeeded
            assert all(id(lock) == create_results[0] for lock in non_none_locks)


class TestStressTest:
    """Stress tests for thread safety under high concurrency."""

    def test_high_concurrency_stress(self):
        """Stress test with many threads and operations."""
        manager = LockManager()
        operations_completed = []
        exceptions = []
        lock = threading.Lock()

        def worker(worker_id):
            try:
                for operation_id in range(10):
                    # Mix of different operations
                    if operation_id % 3 == 0:
                        # Create/get lock
                        manager.get_or_create_lock(f"stress_{worker_id}_{operation_id}", "threading")
                        with lock:
                            operations_completed.append(f"create_{worker_id}_{operation_id}")

                    elif operation_id % 3 == 1:
                        # List locks
                        manager.list_locks()
                        with lock:
                            operations_completed.append(f"list_{worker_id}_{operation_id}")

                    else:
                        # Try to remove a lock (may not exist)
                        manager.remove_lock(f"stress_{worker_id}_{operation_id - 1}")
                        with lock:
                            operations_completed.append(f"remove_{worker_id}_{operation_id}")

                    # Small random delay
                    time.sleep(0.0001)

            except Exception as e:
                with lock:
                    exceptions.append((worker_id, e))

        # Run 20 workers with 10 operations each
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            for future in futures:
                future.result()

        # Check results
        assert not exceptions, f"Exceptions occurred: {exceptions}"
        assert len(operations_completed) == 200, "Not all operations completed"

        # Verify manager state is consistent
        final_locks = manager.list_locks()
        assert isinstance(final_locks, dict), "list_locks returned invalid type"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
