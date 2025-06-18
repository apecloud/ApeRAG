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
- Auto-managed locks with default manager
- Flexible timeout support
- Universal applicability
- Easy extensibility
- Production ready with comprehensive error handling

Quick Start:
    from aperag.concurrent_control import get_or_create_lock, lock_context

    # Create/get a managed lock (most common usage)
    my_lock = get_or_create_lock("database_operations")

    # Use with default behavior
    async with my_lock:
        await critical_work()

    # Use with timeout
    async with lock_context(my_lock, timeout=5.0):
        await critical_work()
"""

from .core import (
    LockManager,  # noqa: F401  # Available for testing and advanced usage
    LockProtocol,  # noqa: F401  # Available for testing and advanced usage
    MultiLock,  # Multi-lock manager for preventing deadlocks
    RedisLock,  # noqa: F401  # Available for testing and advanced usage
    # Internal classes (for testing and advanced usage only - not in __all__)
    ThreadingLock,  # noqa: F401  # Available for testing and advanced usage
    # Main factory functions
    create_lock,  # Create new locks
    get_default_lock_manager,  # Access default manager for advanced operations
    get_lock,  # Retrieve existing locks
    get_or_create_lock,  # Get existing or create new (recommended)
    # Utility functions
    lock_context,  # Async context manager with timeout support
)

__all__ = [
    # Main interface (recommended)
    "get_or_create_lock",  # ⭐ Primary function - get existing or create new
    "get_lock",  # Get existing lock only
    "create_lock",  # Create new locks
    "lock_context",  # ⭐ Timeout support for locks
    "MultiLock",  # Multi-lock manager for acquiring multiple locks
    # Advanced/internal (use sparingly)
    "get_default_lock_manager",  # Advanced lock management
]

# Note: ThreadingLock, RedisLock, LockProtocol, LockManager are available
# for testing and advanced usage but not in __all__ to keep public API simple

__version__ = "1.0.0"
