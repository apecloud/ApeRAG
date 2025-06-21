"""
Unit tests for RedisLock implementation.

This module provides comprehensive tests for the Redis-based distributed lock
implementation, including functionality tests, timeout handling, retry mechanisms,
distributed lock behaviors, and error conditions.

Test Coverage:
=============

1. Basic Lock Operations:
   - Lock creation and initialization
   - Basic acquire/release cycle
   - Context manager usage
   - Lock state tracking

2. Timeout and Retry Mechanisms:
   - Acquire timeout handling
   - Retry attempts with delays
   - Timeout vs retry interaction
   - Edge cases with zero timeout

3. Distributed Lock Features:
   - Automatic expiration
   - Safe release with Lua scripts
   - Unique lock values (UUID)
   - Cross-instance lock behavior

4. Error Handling:
   - Redis connection failures
   - Network timeouts
   - Invalid configurations
   - Cleanup on errors

5. Integration Tests:
   - Multiple lock instances
   - Concurrent access simulation
   - Resource cleanup
   - Connection management

Note: These tests use mocked Redis connections to avoid requiring
a real Redis server during testing.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from aperag.concurrent_control import RedisLock, create_lock


class TestRedisLockInitialization:
    """Test suite for RedisLock initialization and basic structure."""

    def test_redis_lock_creation(self):
        """Test basic RedisLock creation with valid parameters."""
        lock = RedisLock(key="test_key")
        assert lock._key == "test_key"
        assert lock._redis_url == "redis://localhost:6379"
        assert lock._expire_time == 30
        assert lock._retry_times == 3
        assert lock._retry_delay == 0.1
        assert lock._redis_client is None
        assert lock._lock_value is None

    def test_redis_lock_custom_parameters(self):
        """Test RedisLock creation with custom parameters."""
        lock = RedisLock(
            key="custom_key",
            redis_url="redis://custom-host:6380",
            expire_time=60,
            retry_times=5,
            retry_delay=0.2,
        )
        assert lock._key == "custom_key"
        assert lock._redis_url == "redis://custom-host:6380"
        assert lock._expire_time == 60
        assert lock._retry_times == 5
        assert lock._retry_delay == 0.2

    def test_redis_lock_empty_key(self):
        """Test RedisLock creation with empty key."""
        with pytest.raises(ValueError, match="Redis lock key is required"):
            RedisLock(key="")

        with pytest.raises(ValueError, match="Redis lock key is required"):
            RedisLock(key=None)

    def test_redis_lock_factory_creation(self):
        """Test RedisLock creation through factory function."""
        lock = create_lock("redis", key="factory_test")
        assert isinstance(lock, RedisLock)
        assert lock._key == "factory_test"


class TestRedisLockBasicOperations:
    """Test suite for basic RedisLock operations."""

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_successful_acquire_and_release(self, mock_redis_module):
        """Test successful lock acquisition and release."""
        # Setup mock Redis client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True  # Lock acquired
        mock_client.evalsha.return_value = 1  # Lock released
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_acquire_release")

        # Test acquisition
        assert not lock.is_locked()
        success = await lock.acquire()
        assert success is True
        assert lock.is_locked()
        assert lock._lock_value is not None
        assert len(lock._lock_value) == 36  # UUID length

        # Verify Redis calls
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "test_acquire_release"  # key
        assert call_args[1]["nx"] is True
        assert call_args[1]["ex"] == 30

        # Test release
        await lock.release()
        assert not lock.is_locked()
        assert lock._lock_value is None

        # Verify Lua script call
        mock_client.evalsha.assert_called_once()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_redis_module):
        """Test RedisLock as async context manager."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_context")

        assert not lock.is_locked()

        async with lock:
            assert lock.is_locked()

        assert not lock.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, mock_redis_module):
        """Test that lock is released even when exception occurs."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_exception")

        with pytest.raises(ValueError):
            async with lock:
                assert lock.is_locked()
                raise ValueError("Test exception")

        # Lock should be released even after exception
        assert not lock.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_acquire_already_held_lock(self, mock_redis_module):
        """Test acquiring lock that is already held by same instance."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_already_held")

        # First acquire
        success = await lock.acquire()
        assert success is True
        assert lock.is_locked()

        # Second acquire should return True immediately
        success = await lock.acquire()
        assert success is True
        assert lock.is_locked()

        # Redis set should only be called once
        assert mock_client.set.call_count == 1


class TestRedisLockTimeoutAndRetry:
    """Test suite for timeout and retry mechanisms."""

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_acquire_with_timeout_success(self, mock_redis_module):
        """Test successful acquisition within timeout."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_timeout_success")

        start_time = time.time()
        success = await lock.acquire(timeout=2.0)
        elapsed = time.time() - start_time

        assert success is True
        assert elapsed < 0.5  # Should be fast
        assert lock.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_acquire_with_timeout_failure(self, mock_redis_module):
        """Test timeout when lock cannot be acquired."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = False  # Lock always unavailable
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_timeout_failure", retry_delay=0.1)

        start_time = time.time()
        success = await lock.acquire(timeout=0.3)
        elapsed = time.time() - start_time

        assert success is False
        assert elapsed >= 0.25  # Should have tried for ~0.3 seconds
        assert elapsed <= 0.5  # But not much longer
        assert not lock.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_acquire_retry_mechanism(self, mock_redis_module):
        """Test retry mechanism without timeout."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        # Fail first 2 attempts, succeed on 3rd
        mock_client.set.side_effect = [False, False, True]
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_retry", retry_times=3, retry_delay=0.05)

        start_time = time.time()
        success = await lock.acquire()
        elapsed = time.time() - start_time

        assert success is True
        assert elapsed >= 0.1  # Should have waited for retries
        assert lock.is_locked()
        assert mock_client.set.call_count == 3

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_acquire_retry_exhaustion(self, mock_redis_module):
        """Test retry exhaustion without success."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = False  # Always fail
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_retry_exhaustion", retry_times=2, retry_delay=0.05)

        start_time = time.time()
        success = await lock.acquire()
        elapsed = time.time() - start_time

        assert success is False
        assert elapsed >= 0.1  # Should have waited for retries
        assert not lock.is_locked()
        assert mock_client.set.call_count == 3  # retry_times + 1


class TestRedisLockDistributedFeatures:
    """Test suite for distributed lock specific features."""

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_unique_lock_values(self, mock_redis_module):
        """Test that each lock instance uses unique UUID values."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_redis_module.from_url.return_value = mock_client

        lock1 = RedisLock(key="test_unique_1")
        lock2 = RedisLock(key="test_unique_2")

        await lock1.acquire()
        await lock2.acquire()

        # Each lock should have different UUID values
        assert lock1._lock_value != lock2._lock_value
        assert len(lock1._lock_value) == 36  # UUID length
        assert len(lock2._lock_value) == 36

        # Verify that SET was called with unique values
        assert mock_client.set.call_count == 2
        call1_value = mock_client.set.call_args_list[0][0][1]
        call2_value = mock_client.set.call_args_list[1][0][1]
        assert call1_value != call2_value

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_lua_script_safe_release(self, mock_redis_module):
        """Test that release uses Lua script for safety."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1  # Successful release
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_lua_release")

        await lock.acquire()
        # Save lock value before release (release sets it to None)
        lock_value_before_release = lock._lock_value
        await lock.release()

        # Should use evalsha with the loaded script
        mock_client.evalsha.assert_called_once()
        args = mock_client.evalsha.call_args
        assert args[0][0] == "sha123"  # Script SHA
        assert args[0][1] == 1  # Number of keys
        assert args[0][2] == "test_lua_release"  # Key
        assert args[0][3] == lock_value_before_release  # Lock value (UUID)

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_lua_script_fallback(self, mock_redis_module):
        """Test fallback to direct eval if script SHA is not available."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = None  # No SHA available
        mock_client.set.return_value = True
        mock_client.eval.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_lua_fallback")

        await lock.acquire()
        await lock.release()

        # Should use eval as fallback
        mock_client.eval.assert_called_once()
        args = mock_client.eval.call_args
        assert "redis.call" in args[0][0]  # Lua script content
        assert args[0][1] == 1  # Number of keys

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_expire_time_configuration(self, mock_redis_module):
        """Test that expiration time is properly configured."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_expire", expire_time=120)

        await lock.acquire()

        # Verify SET call includes correct expiration
        call_args = mock_client.set.call_args
        assert call_args[1]["ex"] == 120


class TestRedisLockErrorHandling:
    """Test suite for error handling and edge cases."""

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, mock_redis_module):
        """Test handling of Redis connection failures."""
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Connection failed")
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_connection_failure")

        with pytest.raises(ConnectionError, match="Cannot connect to Redis"):
            await lock.acquire()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_redis_operation_failure(self, mock_redis_module):
        """Test handling of Redis operation failures during acquire."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.side_effect = Exception("Redis operation failed")
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_operation_failure", retry_times=1, retry_delay=0.01)

        success = await lock.acquire()
        assert success is False
        assert not lock.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_release_without_acquire(self, mock_redis_module):
        """Test releasing lock that was never acquired."""
        mock_client = AsyncMock()
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_release_without_acquire")

        # Should handle gracefully without errors
        await lock.release()
        assert not lock.is_locked()

        # Redis operations should not be called
        mock_client.evalsha.assert_not_called()
        mock_client.eval.assert_not_called()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_release_failure_cleanup(self, mock_redis_module):
        """Test that local state is cleaned up even if release fails."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.side_effect = Exception("Release failed")
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_release_failure")

        await lock.acquire()
        assert lock.is_locked()

        # Release should clean up local state despite Redis error
        await lock.release()
        assert not lock.is_locked()
        assert lock._lock_value is None

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_redis_module):
        """Test proper cleanup when closing lock."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1
        mock_client.close.return_value = None
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="test_close")

        await lock.acquire()
        assert lock.is_locked()

        await lock.close()

        # Should release lock and close connection
        assert not lock.is_locked()
        mock_client.evalsha.assert_called_once()  # Release called
        mock_client.close.assert_called_once()  # Connection closed
        assert lock._redis_client is None


class TestRedisLockIntegration:
    """Integration tests for RedisLock with realistic scenarios."""

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_multiple_locks_same_key(self, mock_redis_module):
        """Test multiple lock instances competing for same key."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        # First lock succeeds, second fails
        mock_client.set.side_effect = [True, False, False, False]
        mock_client.evalsha.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        lock1 = RedisLock(key="shared_resource", retry_times=1, retry_delay=0.01)
        lock2 = RedisLock(key="shared_resource", retry_times=1, retry_delay=0.01)

        # First lock should succeed
        success1 = await lock1.acquire()
        assert success1 is True
        assert lock1.is_locked()

        # Second lock should fail (same key)
        success2 = await lock2.acquire()
        assert success2 is False
        assert not lock2.is_locked()

        # Release first lock
        await lock1.release()
        assert not lock1.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_concurrent_operations_simulation(self, mock_redis_module):
        """Test concurrent lock operations simulation."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        results = []

        async def worker(worker_id: int):
            """Simulate worker acquiring lock and doing work."""
            lock = RedisLock(key=f"worker_resource_{worker_id}")

            async with lock:
                results.append(f"worker_{worker_id}_start")
                await asyncio.sleep(0.01)  # Simulate work
                results.append(f"worker_{worker_id}_end")
                return worker_id

        # Run multiple workers concurrently
        worker_results = await asyncio.gather(*[worker(i) for i in range(3)])

        assert len(worker_results) == 3
        assert set(worker_results) == {0, 1, 2}
        assert len(results) == 6  # 3 start + 3 end

        # Each worker should have completed
        for i in range(3):
            assert f"worker_{i}_start" in results
            assert f"worker_{i}_end" in results

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_lock_with_factory_and_manager(self, mock_redis_module):
        """Test RedisLock integration with factory and manager."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        from aperag.concurrent_control import get_or_create_lock, lock_context

        # Create lock through get_or_create_lock
        lock = get_or_create_lock("redis_integration_test", "redis", key="integration_key")
        assert isinstance(lock, RedisLock)

        # Use with lock_context
        async with lock_context(lock, timeout=1.0):
            assert lock.is_locked()

        assert not lock.is_locked()

    @patch("aperag.concurrent_control.redis_lock.redis")
    @pytest.mark.asyncio
    async def test_performance_characteristics(self, mock_redis_module):
        """Test performance characteristics of RedisLock."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.script_load.return_value = "sha123"
        mock_client.set.return_value = True
        mock_client.evalsha.return_value = 1
        mock_redis_module.from_url.return_value = mock_client

        lock = RedisLock(key="performance_test")

        # Test multiple acquire/release cycles
        start_time = time.time()
        for _ in range(5):
            async with lock:
                await asyncio.sleep(0.001)  # Minimal work
        total_time = time.time() - start_time

        # Should complete reasonably quickly with mocked Redis
        assert total_time < 1.0

        # Verify Redis operations were called correctly
        assert mock_client.set.call_count == 5
        assert mock_client.evalsha.call_count == 5
