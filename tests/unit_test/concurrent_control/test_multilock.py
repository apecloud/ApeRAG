"""
Unit tests for MultiLock implementation.

This module tests the MultiLock functionality including deadlock prevention,
concurrent access control, and entity-level locking for graph operations.

Test Coverage:
=============

1. Basic MultiLock Operations:
   - Creation with different lock configurations
   - Lock sorting for deadlock prevention
   - Acquire/release operations
   - Context manager usage

2. Error Handling:
   - Partial acquisition failures
   - Exception during acquisition
   - Error handling during release
   - Timeout scenarios

3. Deadlock Prevention:
   - Lock ordering validation
   - Concurrent workers with reverse lock order
   - Multiple entity combinations

4. Real-world Scenarios:
   - Document processing simulation
   - Entity extraction with overlapping entities
   - Workspace isolation
   - Performance with many entities

5. Edge Cases:
   - Empty lock lists
   - Single lock scenarios
   - Error recovery patterns

Usage Example:
=============
    from aperag.concurrent_control import MultiLock, get_or_create_lock

    # Create locks for entities in a document
    entity_locks = [
        get_or_create_lock(f"entity:{name}:{workspace}")
        for name in sorted(entity_names)
    ]

    # Use MultiLock to acquire all locks atomically
    async with MultiLock(entity_locks):
        # Perform graph merge operations
        await merge_entities_and_relations()
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from aperag.concurrent_control import MultiLock, ThreadingLock, get_or_create_lock


class TestMultiLock:
    """Test suite for MultiLock implementation."""

    def test_multilock_creation(self):
        """Test basic MultiLock creation."""
        # Create locks
        lock1 = ThreadingLock(name="lock_a")
        lock2 = ThreadingLock(name="lock_b")
        locks = [lock1, lock2]

        # Create MultiLock
        multi_lock = MultiLock(locks)

        # Should sort locks by name
        assert multi_lock._locks == [lock1, lock2]  # a comes before b
        assert len(multi_lock._acquired_locks) == 0

    def test_multilock_lock_sorting(self):
        """Test that locks are sorted by name to prevent deadlocks."""
        # Create locks in reverse alphabetical order
        lock_z = ThreadingLock(name="lock_z")
        lock_a = ThreadingLock(name="lock_a")
        lock_m = ThreadingLock(name="lock_m")
        locks = [lock_z, lock_a, lock_m]

        multi_lock = MultiLock(locks)

        # Should be sorted alphabetically
        expected_order = [lock_a, lock_m, lock_z]
        assert multi_lock._locks == expected_order

    def test_multilock_empty_locks(self):
        """Test MultiLock with empty lock list."""
        multi_lock = MultiLock([])
        assert multi_lock._locks == []
        assert len(multi_lock._acquired_locks) == 0

    @pytest.mark.asyncio
    async def test_basic_acquire_release_all(self):
        """Test basic acquire_all and release_all operations."""
        lock1 = ThreadingLock(name="test_lock_1")
        lock2 = ThreadingLock(name="test_lock_2")
        multi_lock = MultiLock([lock1, lock2])

        # Initially no locks should be held
        assert not lock1.is_locked()
        assert not lock2.is_locked()
        assert len(multi_lock._acquired_locks) == 0

        # Acquire all locks
        success = await multi_lock.acquire_all()
        assert success is True
        assert lock1.is_locked()
        assert lock2.is_locked()
        assert len(multi_lock._acquired_locks) == 2

        # Release all locks
        await multi_lock.release_all()
        assert not lock1.is_locked()
        assert not lock2.is_locked()
        assert len(multi_lock._acquired_locks) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test MultiLock as async context manager."""
        lock1 = ThreadingLock(name="ctx_lock_1")
        lock2 = ThreadingLock(name="ctx_lock_2")
        multi_lock = MultiLock([lock1, lock2])

        # Initially no locks held
        assert not lock1.is_locked()
        assert not lock2.is_locked()

        async with multi_lock:
            # Both locks should be held
            assert lock1.is_locked()
            assert lock2.is_locked()

        # Locks should be released
        assert not lock1.is_locked()
        assert not lock2.is_locked()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test that all locks are released even when exception occurs."""
        lock1 = ThreadingLock(name="exc_lock_1")
        lock2 = ThreadingLock(name="exc_lock_2")
        multi_lock = MultiLock([lock1, lock2])

        try:
            async with multi_lock:
                assert lock1.is_locked()
                assert lock2.is_locked()
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # All locks should be released even after exception
        assert not lock1.is_locked()
        assert not lock2.is_locked()
        assert len(multi_lock._acquired_locks) == 0

    @pytest.mark.asyncio
    async def test_deadlock_prevention_scenario(self):
        """Test that lock ordering prevents deadlocks."""
        # Create shared locks
        entity_apple = get_or_create_lock("entity:Apple:workspace1")
        entity_google = get_or_create_lock("entity:Google:workspace1")

        results = []

        async def worker1():
            """Worker that processes Apple then Google (reverse order)."""
            locks = [entity_google, entity_apple]  # Intentionally reverse order
            async with MultiLock(locks):
                results.append("worker1_start")
                await asyncio.sleep(0.1)
                results.append("worker1_end")

        async def worker2():
            """Worker that processes Google then Apple (forward order)."""
            locks = [entity_apple, entity_google]  # Forward order
            async with MultiLock(locks):
                results.append("worker2_start")
                await asyncio.sleep(0.1)
                results.append("worker2_end")

        # Run both workers concurrently
        await asyncio.gather(worker1(), worker2())

        # Both workers should complete (no deadlock)
        assert len(results) == 4
        assert "worker1_start" in results
        assert "worker1_end" in results
        assert "worker2_start" in results
        assert "worker2_end" in results

    @pytest.mark.asyncio
    async def test_empty_multilock(self):
        """Test MultiLock with no locks."""
        multi_lock = MultiLock([])

        # Should work fine with no locks
        success = await multi_lock.acquire_all()
        assert success is True

        async with multi_lock:
            # No locks to hold, should just work
            pass

        await multi_lock.release_all()  # Should be safe

    @pytest.mark.asyncio
    async def test_partial_acquire_failure_cleanup(self):
        """Test cleanup when partial lock acquisition fails."""
        # Create mock locks where second one fails
        lock1 = AsyncMock()
        lock1.acquire = AsyncMock(return_value=True)
        lock1.release = AsyncMock()
        lock1._name = "lock_1"

        lock2 = AsyncMock()
        lock2.acquire = AsyncMock(return_value=False)  # Fails
        lock2.release = AsyncMock()
        lock2._name = "lock_2"

        multi_lock = MultiLock([lock1, lock2])

        # Should fail and clean up
        success = await multi_lock.acquire_all()
        assert success is False

        # First lock should have been acquired then released
        lock1.acquire.assert_called_once()
        lock1.release.assert_called_once()

        # Second lock should have been attempted
        lock2.acquire.assert_called_once()
        # Second lock should not be released (never acquired)
        lock2.release.assert_not_called()

        # No locks should be in acquired list
        assert len(multi_lock._acquired_locks) == 0

    @pytest.mark.asyncio
    async def test_acquire_exception_cleanup(self):
        """Test cleanup when lock acquisition raises exception."""
        # Create mock locks where second one raises exception
        lock1 = AsyncMock()
        lock1.acquire = AsyncMock(return_value=True)
        lock1.release = AsyncMock()
        lock1._name = "lock_1"

        lock2 = AsyncMock()
        lock2.acquire = AsyncMock(side_effect=Exception("Acquire failed"))
        lock2.release = AsyncMock()
        lock2._name = "lock_2"

        multi_lock = MultiLock([lock1, lock2])

        # Should fail and clean up
        success = await multi_lock.acquire_all()
        assert success is False

        # First lock should have been released
        lock1.release.assert_called_once()
        assert len(multi_lock._acquired_locks) == 0

    @pytest.mark.asyncio
    async def test_concurrent_document_processing_simulation(self):
        """Test simulation of concurrent document processing with overlapping entities."""
        # Simulate entities from different documents
        apple_lock = get_or_create_lock("entity:Apple:test_workspace")
        google_lock = get_or_create_lock("entity:Google:test_workspace")
        microsoft_lock = get_or_create_lock("entity:Microsoft:test_workspace")
        amazon_lock = get_or_create_lock("entity:Amazon:test_workspace")

        processing_order = []

        async def process_document(doc_id: str, entity_locks: list):
            """Simulate processing a document with given entities."""
            async with MultiLock(entity_locks):
                processing_order.append(f"{doc_id}_start")
                await asyncio.sleep(0.05)  # Simulate processing time
                processing_order.append(f"{doc_id}_end")

        # Create documents with overlapping entities
        doc1_locks = [apple_lock, google_lock]
        doc2_locks = [microsoft_lock, amazon_lock]  # No overlap with doc1
        doc3_locks = [google_lock, amazon_lock]  # Overlaps with both doc1 and doc2

        # Process documents concurrently
        start_time = time.time()
        await asyncio.gather(
            process_document("doc1", doc1_locks),
            process_document("doc2", doc2_locks),
            process_document("doc3", doc3_locks),
        )
        total_time = time.time() - start_time

        # All documents should complete
        assert "doc1_start" in processing_order
        assert "doc1_end" in processing_order
        assert "doc2_start" in processing_order
        assert "doc2_end" in processing_order
        assert "doc3_start" in processing_order
        assert "doc3_end" in processing_order

        # doc1 and doc2 should be able to run in parallel (no overlapping entities)
        # doc3 should wait for both (overlaps with both)
        # Total time should be less than serial execution (3 * 0.05 = 0.15s)
        assert total_time < 0.14, f"Processing took too long: {total_time}s"
        assert total_time >= 0.09, f"Processing too fast, may not be properly synchronized: {total_time}s"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test MultiLock with timeout parameter."""
        lock1 = ThreadingLock(name="timeout_lock_1")
        lock2 = ThreadingLock(name="timeout_lock_2")

        # First, acquire one lock externally
        await lock1.acquire()

        try:
            multi_lock = MultiLock([lock1, lock2])

            # Should fail quickly with timeout
            start_time = time.time()
            success = await multi_lock.acquire_all(timeout=0.1)
            end_time = time.time()

            assert success is False
            assert end_time - start_time < 0.2  # Should fail quickly
            assert len(multi_lock._acquired_locks) == 0  # No locks should be held

        finally:
            # Clean up
            await lock1.release()

    @pytest.mark.asyncio
    async def test_release_error_handling(self):
        """Test error handling during lock release."""
        # Create a mock lock that fails during release
        failing_lock = AsyncMock()
        failing_lock.acquire = AsyncMock(return_value=True)
        failing_lock.release = AsyncMock(side_effect=Exception("Release failed"))
        failing_lock._name = "failing_lock"

        normal_lock = AsyncMock()
        normal_lock.acquire = AsyncMock(return_value=True)
        normal_lock.release = AsyncMock()
        normal_lock._name = "normal_lock"

        multi_lock = MultiLock([normal_lock, failing_lock])

        # Acquire all locks
        success = await multi_lock.acquire_all()
        assert success is True

        # Release should handle the error gracefully
        await multi_lock.release_all()

        # Both locks should have had release called
        normal_lock.release.assert_called_once()
        failing_lock.release.assert_called_once()

        # Acquired locks list should be cleared even with errors
        assert len(multi_lock._acquired_locks) == 0


class TestMultiLockIntegration:
    """Integration tests for MultiLock with real scenarios."""

    @pytest.mark.asyncio
    async def test_lightrag_entity_extraction_simulation(self):
        """Test MultiLock in a scenario similar to LightRAG entity extraction."""
        # Simulate chunk processing results with overlapping entities
        chunk_results = [
            ({"Apple": ["data1"], "Google": ["data2"]}, {}),  # Chunk 1
            ({"Microsoft": ["data3"], "Amazon": ["data4"]}, {}),  # Chunk 2
            ({"Google": ["data5"], "Amazon": ["data6"]}, {}),  # Chunk 3 - overlaps
        ]

        processing_results = []

        async def process_chunk_entities(chunk_id: int, entities: dict):
            """Simulate processing entities from a chunk."""
            # Collect entity names and create locks
            entity_names = list(entities.keys())
            entity_locks = [get_or_create_lock(f"entity:{name}:test") for name in sorted(entity_names)]

            async with MultiLock(entity_locks):
                processing_results.append(f"chunk_{chunk_id}_start")
                # Simulate merge operations
                await asyncio.sleep(0.03)
                processing_results.append(f"chunk_{chunk_id}_end")
                return f"chunk_{chunk_id}_processed"

        # Process chunks concurrently
        tasks = []
        for i, (entities, _) in enumerate(chunk_results):
            task = process_chunk_entities(i, entities)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All chunks should be processed
        assert len(results) == 3
        assert all("processed" in result for result in results)

        # All processing events should be recorded
        assert len(processing_results) == 6
        for i in range(3):
            assert f"chunk_{i}_start" in processing_results
            assert f"chunk_{i}_end" in processing_results

    @pytest.mark.asyncio
    async def test_workspace_isolation(self):
        """Test that different workspaces don't interfere with each other."""
        # Same entity names but different workspaces
        workspace1_locks = [
            get_or_create_lock("entity:Apple:workspace1"),
            get_or_create_lock("entity:Google:workspace1"),
        ]
        workspace2_locks = [
            get_or_create_lock("entity:Apple:workspace2"),
            get_or_create_lock("entity:Google:workspace2"),
        ]

        results = []

        async def process_workspace(workspace_id: str, locks: list):
            async with MultiLock(locks):
                results.append(f"{workspace_id}_start")
                await asyncio.sleep(0.05)
                results.append(f"{workspace_id}_end")

        # Both workspaces should be able to process simultaneously
        start_time = time.time()
        await asyncio.gather(
            process_workspace("workspace1", workspace1_locks), process_workspace("workspace2", workspace2_locks)
        )
        total_time = time.time() - start_time

        # Should complete in parallel (not serial)
        assert total_time < 0.08, f"Workspaces seem to have blocked each other: {total_time}s"
        assert len(results) == 4
        assert "workspace1_start" in results
        assert "workspace1_end" in results
        assert "workspace2_start" in results
        assert "workspace2_end" in results

    @pytest.mark.asyncio
    async def test_performance_with_many_entities(self):
        """Test MultiLock performance with many entities."""
        # Create many entity locks
        num_entities = 20
        entity_names = [f"Entity_{i:02d}" for i in range(num_entities)]
        entity_locks = [get_or_create_lock(f"entity:{name}:perf_test") for name in entity_names]

        async def process_large_document():
            async with MultiLock(entity_locks):
                await asyncio.sleep(0.01)  # Quick processing
                return "completed"

        # Time the operation
        start_time = time.time()
        result = await process_large_document()
        end_time = time.time()

        assert result == "completed"
        # Should complete reasonably quickly even with many locks
        assert end_time - start_time < 1.0, f"Processing {num_entities} entities took too long"
