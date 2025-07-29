"""
Unit tests for RedisLock implementation with new connection manager.

This module provides tests for the Redis-based distributed lock
implementation using the new Redis connection manager architecture.
"""

from unittest.mock import AsyncMock, patch

import pytest

from aperag.concurrent_control.redis_lock import RedisLock


class TestRedisLockWithConnectionManager:
    """Test RedisLock using the new connection manager."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        client.set = AsyncMock()
        client.eval = AsyncMock()
        client.ping = AsyncMock()
        return client

    @pytest.fixture
    def mock_connection_manager(self, mock_redis_client):
        """Create a mock Redis connection manager."""
        with patch("aperag.db.redis_manager.RedisConnectionManager") as mock_manager:
            mock_manager.get_async_client = AsyncMock(return_value=mock_redis_client)
            yield mock_manager, mock_redis_client

    def test_redis_lock_creation(self):
        """Test RedisLock creation with new architecture."""
        lock = RedisLock(key="test_key")
        assert lock._key == "test_key"
        assert lock._name == "redis_lock_test_key"

        assert lock._expire_time == 120
        assert lock._retry_times == 3
        assert lock._retry_delay == 0.1
        assert not lock._is_locked
        assert lock._lock_value is None

    def test_redis_lock_with_custom_params(self):
        """Test RedisLock creation with custom parameters."""
        lock = RedisLock(key="custom_key", expire_time=60, retry_times=5, retry_delay=0.2, name="custom_lock")
        assert lock._key == "custom_key"
        assert lock._name == "custom_lock"
        assert lock._expire_time == 60
        assert lock._retry_times == 5
        assert lock._retry_delay == 0.2

    def test_get_name(self):
        """Test lock name retrieval."""
        lock = RedisLock(key="test_key", name="my_lock")
        assert lock.get_name() == "my_lock"

        lock2 = RedisLock(key="test_key2")
        assert lock2.get_name() == "redis_lock_test_key2"

    @pytest.mark.asyncio
    async def test_successful_acquire_and_release(self, mock_connection_manager):
        """Test successful lock acquisition and release."""
        mock_manager, mock_client = mock_connection_manager
        mock_client.set.return_value = True  # Lock acquired
        mock_client.eval.return_value = 1  # Lock released

        lock = RedisLock(key="test_acquire")

        # Test acquire
        success = await lock.acquire()
        assert success is True
        assert lock.is_locked() is True
        assert lock._lock_value is not None

        # Verify Redis SET was called with correct parameters
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "test_acquire"  # key
        assert call_args[1]["nx"] is True
        assert call_args[1]["ex"] == 120

        # Test release
        await lock.release()
        assert lock.is_locked() is False
        assert lock._lock_value is None

        # Verify Lua script was called
        mock_client.eval.assert_called_once()
        eval_args = mock_client.eval.call_args
        assert "test_acquire" in eval_args[0]

    @pytest.mark.asyncio
    async def test_acquire_failure(self, mock_connection_manager):
        """Test lock acquisition failure."""
        mock_manager, mock_client = mock_connection_manager
        mock_client.set.return_value = False  # Lock not acquired

        lock = RedisLock(key="test_fail", retry_times=1)

        success = await lock.acquire(timeout=0.5)
        assert success is False
        assert lock.is_locked() is False
        assert lock._lock_value is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_connection_manager):
        """Test using RedisLock as async context manager."""
        mock_manager, mock_client = mock_connection_manager
        mock_client.set.return_value = True
        mock_client.eval.return_value = 1

        lock = RedisLock(key="test_context")

        async with lock:
            assert lock.is_locked() is True

        assert lock.is_locked() is False

    @pytest.mark.asyncio
    async def test_context_manager_acquire_failure(self, mock_connection_manager):
        """Test context manager when acquire fails."""
        mock_manager, mock_client = mock_connection_manager
        mock_client.set.return_value = False

        lock = RedisLock(key="test_context_fail", retry_times=0)

        with pytest.raises(RuntimeError, match="Failed to acquire Redis lock"):
            async with lock:
                pass

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_connection_manager):
        """Test retry mechanism."""
        mock_manager, mock_client = mock_connection_manager
        # First two calls fail, third succeeds
        mock_client.set.side_effect = [False, False, True]

        lock = RedisLock(key="test_retry", retry_times=3, retry_delay=0.01)

        success = await lock.acquire()
        assert success is True
        assert mock_client.set.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_respected(self, mock_connection_manager):
        """Test that timeout is respected."""
        mock_manager, mock_client = mock_connection_manager
        mock_client.set.return_value = False

        lock = RedisLock(key="test_timeout", retry_times=10, retry_delay=0.1)

        import time

        start_time = time.time()
        success = await lock.acquire(timeout=0.2)
        elapsed = time.time() - start_time

        assert success is False
        assert elapsed < 0.5  # Should timeout quickly, not wait for all retries

    @pytest.mark.asyncio
    async def test_release_safety(self, mock_connection_manager):
        """Test release safety mechanisms."""
        mock_manager, mock_client = mock_connection_manager

        lock = RedisLock(key="test_safety")

        # Test release without acquire (should not crash)
        await lock.release()  # Should log warning but not crash

        # Test release with no lock value (should not crash)
        lock._is_locked = True
        lock._lock_value = None
        await lock.release()  # Should log error but not crash

    @pytest.mark.asyncio
    async def test_connection_manager_integration(self):
        """Test integration with Redis connection manager."""
        lock = RedisLock(key="test_integration")

        # Test that _get_redis_client calls the connection manager
        with patch("aperag.db.redis_manager.RedisConnectionManager.get_async_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = mock_client

            client = await lock._get_redis_client()
            assert client is mock_client
            mock_get.assert_called_once_with()  # Uses default settings

    @pytest.mark.asyncio
    async def test_close_method(self, mock_connection_manager):
        """Test close method behavior."""
        mock_manager, mock_client = mock_connection_manager
        mock_client.set.return_value = True
        mock_client.eval.return_value = 1

        lock = RedisLock(key="test_close")

        # Acquire lock then close
        await lock.acquire()
        assert lock.is_locked() is True

        await lock.close()
        assert lock.is_locked() is False
        # Connection manager handles connection, so no client.close() call

    def test_invalid_key(self):
        """Test creation with invalid key."""
        with pytest.raises(ValueError, match="Redis lock key is required"):
            RedisLock(key="")

        with pytest.raises(ValueError, match="Redis lock key is required"):
            RedisLock(key=None)


class TestRedisLockErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_redis_operation_error_during_acquire(self):
        """Test Redis operation errors during acquire."""
        with patch("aperag.db.redis_manager.RedisConnectionManager.get_async_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.set.side_effect = Exception("Redis error")
            mock_get.return_value = mock_client

            lock = RedisLock(key="test_error", retry_times=1, retry_delay=0.01)

            success = await lock.acquire()
            assert success is False
            assert not lock.is_locked()

    @pytest.mark.asyncio
    async def test_redis_operation_error_during_release(self):
        """Test Redis operation errors during release."""
        with patch("aperag.db.redis_manager.RedisConnectionManager.get_async_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.set.return_value = True
            mock_client.eval.side_effect = Exception("Redis error")
            mock_get.return_value = mock_client

            lock = RedisLock(key="test_release_error")

            # Acquire successfully
            await lock.acquire()
            assert lock.is_locked()

            # Release with error should still clean up local state
            await lock.release()
            assert not lock.is_locked()
            assert lock._lock_value is None


class TestRedisLockLuaScript:
    """Test Lua script execution."""

    @pytest.mark.asyncio
    async def test_lua_script_execution(self):
        """Test that Lua script is executed with correct parameters."""
        with patch("aperag.db.redis_manager.RedisConnectionManager.get_async_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.set.return_value = True
            mock_client.eval.return_value = 1
            mock_get.return_value = mock_client

            lock = RedisLock(key="test_lua")

            await lock.acquire()
            lock_value = lock._lock_value
            await lock.release()

            # Verify Lua script execution
            mock_client.eval.assert_called_once()
            call_args = mock_client.eval.call_args

            # Check script content
            script = call_args[0][0]
            assert "redis.call" in script
            assert "get" in script
            assert "del" in script

            # Check parameters
            assert call_args[0][1] == 1  # Number of keys
            assert call_args[0][2] == "test_lua"  # Key
            assert call_args[0][3] == lock_value  # Lock value
