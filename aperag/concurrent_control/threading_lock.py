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
Threading-based lock implementation.

This module contains the ThreadingLock implementation that uses threading.Lock
wrapped with asyncio.to_thread for async compatibility.
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Optional

from .protocols import LockProtocol
from .utils import LockAcquisitionError

logger = logging.getLogger(__name__)


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
        Acquire the lock using non-blocking polling to avoid blocking the event loop.

        Args:
            timeout: Maximum time to wait for the lock (seconds).
                    None means wait indefinitely.

        Returns:
            True if lock was acquired, False if timeout occurred.
        """
        start_time = time.time() if timeout is not None else None

        while True:
            try:
                # Try non-blocking acquire
                acquired = self._lock.acquire(blocking=False)

                if acquired:
                    self._holder_info = f"Thread-{threading.get_ident()}"
                    logger.debug(f"Lock '{self._name}' acquired by {self._holder_info}")
                    return True

                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.debug(f"Lock '{self._name}' acquisition timed out after {elapsed:.3f}s")
                        return False

                # Sleep briefly before retrying (non-blocking for event loop)
                await asyncio.sleep(0.001)  # 1ms polling interval

            except Exception as e:
                logger.error(f"Error acquiring lock '{self._name}': {e}")
                return False

    async def release(self) -> None:
        """Release the lock directly."""
        try:
            self._lock.release()
            logger.debug(f"Lock '{self._name}' released by {self._holder_info}")
            self._holder_info = None
        except Exception as e:
            logger.error(f"Error releasing lock '{self._name}': {e}")

    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.locked()

    def get_name(self) -> str:
        """Get the name/identifier of the lock."""
        return self._name

    async def __aenter__(self) -> "ThreadingLock":
        """Async context manager entry."""
        success = await self.acquire()
        if not success:
            raise LockAcquisitionError(f"Failed to acquire lock '{self._name}'")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()
