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
Universal Concurrent Control Module

A flexible and reusable concurrent control system that provides unified locking
mechanisms for any Python application. Designed to handle different deployment
scenarios and task queue environments.

Features:
- Redis-backed production locks by default
- Auto-managed locks with default manager
- Flexible timeout support
- Universal applicability
- Easy extensibility
- Production ready with comprehensive error handling

Quick Start:
    from aperag.concurrent_control import create_distributed_lock, lock_context

    # Create a Redis-backed production lock
    my_lock = create_distributed_lock("database_operations")

    # Use with distributed behavior
    async with my_lock:
        await critical_work()

    # Use with timeout
    async with lock_context(my_lock, timeout=5.0):
        await critical_work()
"""

from .manager import (
    LockManager,  # noqa: F401  # Available for testing and advanced usage
    create_distributed_lock,  # Create Redis-backed production locks
    create_lock,  # Create new locks
    get_default_lock_manager,  # Access default manager for advanced operations
    get_lock,  # Retrieve existing locks
    get_or_create_lock,  # Get existing or create new
)
from .protocols import LockProtocol  # noqa: F401  # Available for testing and advanced usage
from .redis_lock import RedisLock  # noqa: F401  # Available for testing and advanced usage
from .threading_lock import ThreadingLock  # noqa: F401  # Available for testing and advanced usage
from .utils import lock_context  # ⭐ Timeout support for locks

__all__ = [
    # Main interface (recommended)
    "create_distributed_lock",  # ⭐ Primary production function - Redis-backed lock
    "get_or_create_lock",  # Get existing or create new (defaults to Redis)
    "get_lock",  # Get existing lock only
    "create_lock",  # Create new locks
    "lock_context",  # ⭐ Timeout support for locks
    # Advanced/internal (use sparingly)
    "get_default_lock_manager",  # Advanced lock management
]

# Note: ThreadingLock, RedisLock, LockProtocol, LockManager are available
# for testing and advanced usage but not in __all__ to keep public API simple

__version__ = "1.0.0"
