# Universal Concurrent Control Module

A flexible and reusable concurrent control system that provides unified locking mechanisms for any Python application. This module is designed to handle different deployment scenarios and task queue environments.

## Features

- **Simple Interface**: `get_or_create_lock()` function covers 90% of use cases
- **Auto-managed Locks**: Named locks are automatically registered for later retrieval
- **Flexible Timeout Support**: Use `lock_context()` for timeout control in async contexts
- **Universal Applicability**: Works with any Python application requiring concurrent control
- **Easy Extensibility**: Clear protocol interface for implementing new lock types
- **Production Ready**: Organized management and comprehensive error handling

## Supported Lock Types

### ThreadingLock
- **Best for**: Single-process environments (Celery `--pool=solo`, `--pool=threads`, `--pool=gevent`)
- **Implementation**: `threading.Lock` wrapped with `asyncio.to_thread`
- **Features**: 
  - Works with both coroutine and thread concurrency
  - Non-blocking for the event loop
  - Lower overhead than distributed locks
- **Limitations**: Does NOT work across multiple processes

### RedisLock
- **Best for**: Multi-process environments (Celery `--pool=prefork`, distributed deployment)
- **Implementation**: Redis-based distributed locking
- **Features**: 
  - Works across processes, containers, and machines
  - Automatic lock expiration to prevent deadlocks
  - Retry mechanisms for lock acquisition
  - Safe release using Lua scripts
- **Trade-offs**: Network overhead, Redis dependency

## Quick Start

### Basic Usage (90% of use cases)

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

# Create/get a managed lock (recommended for most cases)
my_lock = get_or_create_lock("database_operations")

# Use with default behavior
async def critical_operation():
    async with my_lock:
        # Your protected operations here
        await some_critical_work()

# Use with timeout support
async def operation_with_timeout():
    async with lock_context(my_lock, timeout=5.0):
        await some_critical_work()
```

### Advanced Usage (10% of use cases)

```python
from aperag.concurrent_control import create_lock, get_lock, get_or_create_lock

# Create a named lock (automatically managed)
create_lock("threading", name="database_ops")

# Later retrieve the same lock instance
lock = get_lock("database_ops")
if lock:
    async with lock:
        await execute_query()

# Or get existing/create new in one call (recommended)
lock = get_or_create_lock("cache_ops")
async with lock:
    await update_cache()
```

## Default Lock Manager

The module uses a **default global lock manager** to automatically manage named locks. This provides several benefits:

### Automatic Registration
```python
# When you create a named lock, it's automatically registered
lock1 = create_lock("threading", name="shared_resource")

# Later, anywhere in your application, get the same instance
lock2 = get_lock("shared_resource") 
assert lock1 is lock2  # True - same lock instance
```

### Cross-Module Consistency
```python
# In module A
from aperag.concurrent_control import get_or_create_lock
db_lock = get_or_create_lock("database_operations")

# In module B  
from aperag.concurrent_control import get_or_create_lock
same_db_lock = get_or_create_lock("database_operations")
# Gets the exact same lock instance - no coordination needed
```

### Workspace Isolation
```python
# Different workspaces automatically get different locks
workspace_a_lock = get_or_create_lock("operations_workspace_a")
workspace_b_lock = get_or_create_lock("operations_workspace_b")
# These are completely independent locks
```

### Lock Lifecycle
- **Creation**: Named locks are automatically registered when created
- **Retrieval**: Use `get_lock()` or `get_or_create_lock()` to access
- **Cleanup**: Locks persist for the lifetime of the application
- **Thread Safety**: The default manager is thread-safe

## API Reference

### Primary Interface (Recommended)

#### `get_or_create_lock(name: str, lock_type: str = "threading", **kwargs) -> LockProtocol`
**⭐ Main function for most use cases.** Get existing lock or create new one.

```python
# Simple usage (covers 90% of scenarios)
my_lock = get_or_create_lock("database_operations")

# With specific lock type  
redis_lock = get_or_create_lock("distributed_ops", "redis", key="my_key")
```

#### `lock_context(lock: LockProtocol, timeout: float = None)`
**⭐ Timeout support for async contexts.**

```python
# With timeout
async with lock_context(my_lock, timeout=10.0):
    await long_operation()
```

### Secondary Interface (Special Cases)

#### `get_lock(name: str) -> Optional[LockProtocol]`
Get existing lock only. Returns None if not found.

```python
# Check if lock exists
existing = get_lock("my_operation")
if existing:
    async with existing:
        await work()
```

#### `create_lock(lock_type: str = "threading", **kwargs) -> LockProtocol`
Create a new lock. If `name` is provided, it will be auto-registered.

```python
# Create anonymous lock (not managed)
temp_lock = create_lock("threading")

# Create named lock (automatically managed)
managed_lock = create_lock("threading", name="my_lock")
```

### Advanced Access

#### `get_default_lock_manager() -> LockManager`
Access the default lock manager for advanced operations.

```python
from aperag.concurrent_control import get_default_lock_manager

manager = get_default_lock_manager()
print("Managed locks:", manager.list_locks())
manager.remove_lock("some_lock_name")
```

## Deployment Recommendations

### Celery Deployment Modes

| Pool Type | Recommended Lock | Use Case |
|-----------|------------------|----------|
| `--pool=prefork` | `RedisLock` | Production multi-process |
| `--pool=threads` | `ThreadingLock` | Single-process multi-thread |
| `--pool=gevent` | `ThreadingLock` | Single-process async |
| `--pool=solo` | `ThreadingLock` | Development/testing |

## Usage Patterns

### Simple Operations (Most Common)

```python
from aperag.concurrent_control import get_or_create_lock

# One-liner for simple cases
my_lock = get_or_create_lock("file_operations")
async with my_lock:
    await process_file()
```

### With Timeout Control

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

migration_lock = get_or_create_lock("database_migration")

async def migration_with_timeout():
    try:
        async with lock_context(migration_lock, timeout=30.0):
            await run_migration()
    except TimeoutError:
        print("Migration timed out, will retry later")
```

### Multi-component Applications

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

# Different components use the same lock names
database_lock = get_or_create_lock("database_operations")
cache_lock = get_or_create_lock("cache_operations")

async def update_user_data(user_id):
    # Operations can run in parallel since they use different locks
    async with database_lock:
        await update_user_in_db(user_id)
    
    async with cache_lock:
        await invalidate_user_cache(user_id)

async def batch_operation():
    # But batch operations serialize properly
    async with lock_context(database_lock, timeout=60):
        await process_large_batch()
```

### Workspace/Tenant Isolation

```python
from aperag.concurrent_control import get_or_create_lock

class WorkspaceManager:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        # Each workspace gets its own lock automatically
        self.processing_lock = get_or_create_lock(f"processing_{workspace_id}")
    
    async def process_data(self, data):
        async with self.processing_lock:
            await self._process_data_for_workspace(data)

# Different workspaces get different locks automatically
manager_a = WorkspaceManager("tenant_a")
manager_b = WorkspaceManager("tenant_b")
# Their locks are completely independent
```

## Error Handling

The module provides comprehensive error handling:

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

my_lock = get_or_create_lock("critical_section")

# Automatic lock release on exceptions
async def operation_with_error():
    try:
        async with my_lock:
            await some_operation()
            raise ValueError("Something went wrong")
    except ValueError:
        # Lock is automatically released
        pass

# Timeout handling
async def operation_with_timeout():
    try:
        async with lock_context(my_lock, timeout=10.0):
            await long_operation()
    except TimeoutError:
        print("Operation timed out")
```

## Performance Considerations

### ThreadingLock
- **Pros**: Low overhead for single-process scenarios, no external dependencies
- **Cons**: Higher overhead than `asyncio.Lock`, process-local only

### RedisLock (Future)
- **Pros**: Works across processes and machines, built-in expiration
- **Cons**: Network overhead, Redis dependency, higher latency

## Migration Guide

### From asyncio.Lock

```python
# Before
import asyncio
old_lock = asyncio.Lock()

async with old_lock:
    # No timeout support
    await work()

# After  
from aperag.concurrent_control import get_or_create_lock, lock_context

new_lock = get_or_create_lock("my_operation")

# Default behavior
async with new_lock:
    await work()

# With timeout
async with lock_context(new_lock, timeout=5.0):
    await work()
```

## Why This Design?

### Main Interface Simplicity
```python
# 90% of use cases - clear and explicit function name
my_lock = get_or_create_lock("my_operation")
async with my_lock:
    await work()
```

### `lock_context()` Value
```python
# Without lock_context, timeout requires verbose code:
if await my_lock.acquire(timeout=5.0):
    try:
        await work()
    finally:
        await my_lock.release()

# With lock_context - clean and safe:
async with lock_context(my_lock, timeout=5.0):
    await work()
```

### Auto-managed Locks
```python
# Create once, use anywhere
create_lock("threading", name="database_ops")

# Later in any module
same_lock = get_lock("database_ops")  # Gets the same instance
```

### Single Source of Truth
- All named locks automatically managed by default manager
- No need to pass lock instances around
- Consistent naming across application
- Workspace isolation through naming conventions

## Testing

The module includes comprehensive unit tests. Run them with:

```bash
pytest tests/unit_test/concurrent_control/
```

## Future Enhancements

1. **Lock Monitoring**: Usage metrics and performance monitoring
2. **Advanced Features**: Lock priorities, deadlock detection
3. **Connection Pooling**: Optimize Redis connection management

## License

This module is part of the ApeRAG project and follows the same license terms.

# Usage Examples

## Basic RedisLock Usage

### Direct Usage

```python
import asyncio
from aperag.concurrent_control import RedisLock

async def example():
    # Create a Redis lock
    lock = RedisLock(
        key="my_resource",
        redis_url="redis://localhost:6379",
        expire_time=30,  # Lock expires in 30 seconds
        retry_times=3,
        retry_delay=0.1
    )
    
    # Acquire the lock
    if await lock.acquire(timeout=5.0):
        try:
            print("Lock acquired! Doing critical work...")
            await asyncio.sleep(2)  # Simulate work
        finally:
            await lock.release()
            print("Lock released!")
    else:
        print("Failed to acquire lock")
    
    # Clean up
    await lock.close()

# Run the example
asyncio.run(example())
```

### Context Manager Usage

```python
import asyncio
from aperag.concurrent_control import RedisLock

async def example_with_context():
    lock = RedisLock(key="my_resource", expire_time=60)
    
    # Use as context manager for automatic cleanup
    async with lock:
        print("Lock acquired automatically!")
        await asyncio.sleep(1)
        # Lock will be released automatically

asyncio.run(example_with_context())
```

### Factory and Manager Usage

```python
import asyncio
from aperag.concurrent_control import create_lock, get_default_lock_manager, lock_context

async def example_factory():
    # Create lock using factory
    redis_lock = create_lock(
        "redis",
        key="shared_resource",
        redis_url="redis://localhost:6379",
        expire_time=120
    )
    
    # Use with lock context for timeout support
    try:
        async with lock_context(redis_lock, timeout=10.0):
            print("Processing with distributed lock...")
            await asyncio.sleep(5)
    except TimeoutError:
        print("Could not acquire lock within timeout")
    
    # Use lock manager for reuse
    manager = get_default_lock_manager()
    managed_lock = manager.get_or_create_lock(
        "distributed_task", 
        "redis",
        key="task_queue",
        redis_url="redis://localhost:6379"
    )
    
    async with lock_context(managed_lock, timeout=5.0):
        print("Using managed Redis lock...")

asyncio.run(example_factory())
```

## ThreadingLock Usage

```python
import asyncio
from aperag.concurrent_control import ThreadingLock, lock_context

async def example_threading():
    # For single-process, multi-thread scenarios
    lock = ThreadingLock(name="local_resource")
    
    async with lock_context(lock, timeout=2.0):
        print("Processing with threading lock...")
        await asyncio.sleep(1)

asyncio.run(example_threading())
``` 