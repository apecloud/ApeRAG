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
Utility functions for the concurrent control system.

This module contains helper functions and context managers for working
with locks in a more convenient way.
"""

from contextlib import asynccontextmanager
from typing import AsyncContextManager, Optional

from .protocols import LockProtocol


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
        lock_name = lock.get_name()
        raise TimeoutError(f"Failed to acquire lock '{lock_name}' within {timeout} seconds")

    try:
        yield
    finally:
        await lock.release()
