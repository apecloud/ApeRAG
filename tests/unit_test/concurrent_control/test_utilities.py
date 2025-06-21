"""
Unit tests for utility functions in concurrent_control module.

This module tests the utility functions including create_lock factory,
lock_context context manager, and other helper functions.
"""

import asyncio
import time

import pytest

from aperag.concurrent_control import (
    RedisLock,
    ThreadingLock,
    create_lock,
    get_default_lock_manager,
    lock_context,
)


class TestCreateLockFactory:
    """Test suite for create_lock factory function."""

    def test_create_threading_lock(self):
        """Test creating threading locks via factory."""
        # Basic threading lock
        lock1 = create_lock("threading")
        assert isinstance(lock1, ThreadingLock)
        assert lock1._name.startswith("threading_lock_")

        # Threading lock with name
        lock2 = create_lock("threading", name="custom_name")
        assert isinstance(lock2, ThreadingLock)
        assert lock2._name == "custom_name"

        # Different instances should be created
        assert lock1 is not lock2

    def test_create_redis_lock(self):
        """Test creating Redis locks via factory."""
        # Basic Redis lock
        lock1 = create_lock("redis", key="test_key")
        assert isinstance(lock1, RedisLock)
        assert lock1._key == "test_key"
        assert lock1._expire_time == 120

        # Redis lock with custom parameters
        lock2 = create_lock("redis", key="custom_key", expire_time=60, retry_times=5, retry_delay=0.5)
        assert isinstance(lock2, RedisLock)
        assert lock2._key == "custom_key"
        assert lock2._expire_time == 60
        assert lock2._retry_times == 5
        assert lock2._retry_delay == 0.5

    def test_create_lock_invalid_type(self):
        """Test create_lock with invalid lock type."""
        with pytest.raises(ValueError, match="Unknown lock type: invalid"):
            create_lock("invalid")

    def test_create_lock_default_type(self):
        """Test create_lock with default type."""
        lock = create_lock()  # Should default to "threading"
        assert isinstance(lock, ThreadingLock)

    def test_create_redis_lock_missing_key(self):
        """Test creating Redis lock without required key."""
        with pytest.raises(TypeError):
            create_lock("redis")

        with pytest.raises(ValueError, match="Redis lock key is required"):
            create_lock("redis", key="")

        with pytest.raises(ValueError, match="Redis lock key is required"):
            create_lock("redis", key=None)


class TestLockContext:
    """Test suite for lock_context context manager."""

    @pytest.mark.asyncio
    async def test_basic_lock_context(self):
        """Test basic lock_context usage."""
        lock = create_lock("threading", name="context_test")

        assert not lock.is_locked()

        async with lock_context(lock):
            assert lock.is_locked()

        assert not lock.is_locked()

    @pytest.mark.asyncio
    async def test_lock_context_with_timeout_success(self):
        """Test lock_context with timeout - successful acquisition."""
        lock = create_lock("threading", name="timeout_success_test")

        async with lock_context(lock, timeout=1.0):
            assert lock.is_locked()
            await asyncio.sleep(0.1)  # Brief operation

        assert not lock.is_locked()

    @pytest.mark.asyncio
    async def test_lock_context_with_timeout_failure(self):
        """Test lock_context with timeout - timeout occurs."""
        lock = create_lock("threading", name="timeout_failure_test")

        async def blocking_task():
            async with lock:
                await asyncio.sleep(0.3)  # Hold lock for a while

        async def timeout_task():
            # This should timeout
            async with lock_context(lock, timeout=0.1):
                assert False, "Should not reach here"

        # Start blocking task first
        blocking_task_handle = asyncio.create_task(blocking_task())
        await asyncio.sleep(0.05)  # Ensure blocking task gets the lock

        # Try to acquire with timeout
        with pytest.raises(TimeoutError, match="Failed to acquire lock .* within 0.1 seconds"):
            await timeout_task()

        # Wait for blocking task to complete
        await blocking_task_handle
        assert not lock.is_locked()

    @pytest.mark.asyncio
    async def test_lock_context_exception_handling(self):
        """Test that lock_context releases lock on exception."""
        lock = create_lock("threading", name="exception_test")

        assert not lock.is_locked()

        try:
            async with lock_context(lock):
                assert lock.is_locked()
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Lock should be released even after exception
        assert not lock.is_locked()

    @pytest.mark.asyncio
    async def test_lock_context_nested_usage(self):
        """Test nested lock_context usage."""
        lock1 = create_lock("threading", name="nested_1")
        lock2 = create_lock("threading", name="nested_2")

        async with lock_context(lock1):
            assert lock1.is_locked()
            assert not lock2.is_locked()

            async with lock_context(lock2):
                assert lock1.is_locked()
                assert lock2.is_locked()

            assert lock1.is_locked()
            assert not lock2.is_locked()

        assert not lock1.is_locked()
        assert not lock2.is_locked()

    @pytest.mark.asyncio
    async def test_lock_context_concurrent_access(self):
        """Test lock_context with concurrent tasks."""
        lock = create_lock("threading", name="concurrent_context_test")
        results = []

        async def worker(worker_id: int):
            async with lock_context(lock):
                results.append(f"worker_{worker_id}_start")
                await asyncio.sleep(0.05)
                results.append(f"worker_{worker_id}_end")
                return worker_id

        # Run multiple workers
        task_results = await asyncio.gather(*[worker(i) for i in range(3)])

        # All workers should complete
        assert len(task_results) == 3
        assert set(task_results) == {0, 1, 2}

        # Should have proper start/end pairs (serialized execution)
        assert len(results) == 6
        for i in range(3):
            start_msg = f"worker_{i}_start"
            end_msg = f"worker_{i}_end"
            assert start_msg in results
            assert end_msg in results
            start_idx = results.index(start_msg)
            end_idx = results.index(end_msg)
            assert end_idx == start_idx + 1, f"Worker {i} execution was not atomic"

    @pytest.mark.asyncio
    async def test_lock_context_timeout_edge_cases(self):
        """Test lock_context timeout edge cases."""
        lock = create_lock("threading", name="timeout_edge_test")

        # Test timeout with competing task (not reentrant lock)
        async def blocking_task():
            async with lock:
                await asyncio.sleep(0.2)  # Hold lock for a while

        # Start blocking task
        blocking_task_handle = asyncio.create_task(blocking_task())
        await asyncio.sleep(0.05)  # Ensure blocking task gets the lock

        # Very small timeout should fail immediately
        with pytest.raises(TimeoutError):
            async with lock_context(lock, timeout=0.001):
                pass

        # Wait for blocking task to complete
        await blocking_task_handle

        # Test that timeout of 0 fails immediately when lock is held
        async def another_blocking_task():
            async with lock:
                await asyncio.sleep(0.1)

        blocking_task_handle2 = asyncio.create_task(another_blocking_task())
        await asyncio.sleep(0.02)  # Ensure task gets the lock

        with pytest.raises(TimeoutError):
            async with lock_context(lock, timeout=0):
                pass

        await blocking_task_handle2

    @pytest.mark.asyncio
    async def test_lock_context_without_timeout(self):
        """Test lock_context without timeout parameter."""
        lock = create_lock("threading", name="no_timeout_test")

        # Should work normally without timeout
        async with lock_context(lock):
            assert lock.is_locked()
            await asyncio.sleep(0.01)

        assert not lock.is_locked()


class TestIntegrationScenarios:
    """Integration tests combining different components."""

    @pytest.mark.asyncio
    async def test_factory_manager_context_integration(self):
        """Test integration of factory, manager, and context."""
        # Use factory to create lock
        lock = create_lock("threading", name="integration_test")

        # Use global manager to get the same type of lock
        manager = get_default_lock_manager()
        managed_lock = manager.get_or_create_lock("integration_managed", "threading")

        # Use both locks with context manager
        async with lock_context(lock):
            assert lock.is_locked()

            async with lock_context(managed_lock):
                assert managed_lock.is_locked()
                assert lock.is_locked()  # Should still be locked

        assert not lock.is_locked()
        assert not managed_lock.is_locked()

    @pytest.mark.asyncio
    async def test_multiple_lock_types_coordination(self):
        """Test coordination between different lock implementations."""
        threading_lock = create_lock("threading", name="threading_coord")
        redis_lock = create_lock("redis", key="redis_coord_key")

        # Both should be different types but same interface
        assert isinstance(threading_lock, ThreadingLock)
        assert isinstance(redis_lock, RedisLock)
        assert hasattr(threading_lock, "acquire")
        assert hasattr(redis_lock, "acquire")

        # Threading lock should work
        async with lock_context(threading_lock):
            assert threading_lock.is_locked()

        # Redis lock should also work (but may fail without Redis server)
        # We expect either success or connection error (not NotImplementedError)
        try:
            async with lock_context(redis_lock):
                assert redis_lock.is_locked()
        except (ConnectionError, ImportError):
            # Expected if Redis is not available or not running
            pass

    @pytest.mark.asyncio
    async def test_real_world_usage_pattern(self):
        """Test realistic usage patterns."""
        # Simulate different components using locks
        database_lock = create_lock("threading", name="database_operations")
        cache_lock = create_lock("threading", name="cache_operations")
        file_lock = create_lock("threading", name="file_operations")

        operations_completed = []

        async def database_operation():
            async with lock_context(database_lock, timeout=2.0):
                operations_completed.append("db_start")
                await asyncio.sleep(0.1)  # Simulate DB work
                operations_completed.append("db_end")

        async def cache_operation():
            async with lock_context(cache_lock, timeout=2.0):
                operations_completed.append("cache_start")
                await asyncio.sleep(0.05)  # Simulate cache work
                operations_completed.append("cache_end")

        async def file_operation():
            async with lock_context(file_lock, timeout=2.0):
                operations_completed.append("file_start")
                await asyncio.sleep(0.08)  # Simulate file work
                operations_completed.append("file_end")

        # Run operations concurrently - they should not interfere
        await asyncio.gather(database_operation(), cache_operation(), file_operation())

        # All operations should complete
        assert len(operations_completed) == 6

        # Each operation should have proper start/end pair
        assert "db_start" in operations_completed
        assert "db_end" in operations_completed
        assert "cache_start" in operations_completed
        assert "cache_end" in operations_completed
        assert "file_start" in operations_completed
        assert "file_end" in operations_completed

    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Test performance characteristics of different approaches."""
        # Compare direct lock usage vs context manager
        lock = create_lock("threading", name="performance_test")

        # Direct usage timing
        start_time = time.time()
        for _ in range(10):
            await lock.acquire()
            await asyncio.sleep(0.001)
            await lock.release()
        direct_time = time.time() - start_time

        # Context manager timing
        start_time = time.time()
        for _ in range(10):
            async with lock_context(lock):
                await asyncio.sleep(0.001)
        context_time = time.time() - start_time

        # Context manager should have minimal overhead
        overhead_ratio = context_time / direct_time
        assert overhead_ratio < 2.0, f"Context manager has too much overhead: {overhead_ratio}"

        # Both should complete in reasonable time
        assert direct_time < 1.0
        assert context_time < 1.0

    @pytest.mark.asyncio
    async def test_error_recovery_patterns(self):
        """Test error recovery and cleanup patterns."""
        lock = create_lock("threading", name="error_recovery_test")
        error_count = 0
        success_count = 0

        async def potentially_failing_operation(should_fail: bool):
            nonlocal error_count, success_count

            try:
                async with lock_context(lock, timeout=1.0):
                    if should_fail:
                        error_count += 1
                        raise RuntimeError("Simulated failure")
                    else:
                        success_count += 1
                        await asyncio.sleep(0.01)
            except RuntimeError:
                pass  # Expected for failing operations

        # Mix of successful and failing operations
        operations = [
            potentially_failing_operation(False),  # Success
            potentially_failing_operation(True),  # Fail
            potentially_failing_operation(False),  # Success
            potentially_failing_operation(True),  # Fail
            potentially_failing_operation(False),  # Success
        ]

        await asyncio.gather(*operations, return_exceptions=True)

        # Check that operations completed as expected
        assert success_count == 3
        assert error_count == 2

        # Lock should be available after all operations
        assert not lock.is_locked()

        # Should be able to use lock normally after errors
        async with lock_context(lock):
            assert lock.is_locked()

        assert not lock.is_locked()
