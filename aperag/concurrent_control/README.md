# Universal Concurrent Control Module

A flexible and reusable concurrent control system that provides unified locking mechanisms for Python applications. This module is designed for different deployment scenarios and task queue environments, offering a simple API with powerful features.

## Core Features

* **Minimal API**: `get_or_create_lock()` solves 90% of use cases with one function
* **Auto Management**: Named locks are automatically registered and reused without manual instance passing
* **Timeout Support**: Use `lock_context()` for flexible timeout control
* **Thread Safe**: Lock manager is completely thread-safe, supporting multi-threaded concurrent access
* **Production Ready**: Complete error handling, logging, and monitoring support
* **Zero Configuration**: Works out of the box with automatic Redis connection management

## Supported Lock Types

### ThreadingLock - Process-Local Lock
* **Use Cases**: Single-process environments (Celery `--pool=solo`, `--pool=threads`, `--pool=gevent`)
* **Implementation**: Based on `threading.Lock` with non-blocking polling to avoid event loop blocking
* **Performance**:
  - Low latency, no network overhead
  - Supports both coroutine and thread concurrency
  - Event loop friendly async implementation
* **Limitation**: Limited to single process only

### RedisLock - Distributed Lock
* **Use Cases**: Multi-process environments (Celery `--pool=prefork`, containerized deployment, distributed systems)
* **Implementation**: Based on Redis SET NX EX pattern with Lua scripts for atomicity
* **Advanced Features**:
  - Works across processes, containers, and machines
  - Auto-expiration to prevent deadlocks (default 120 seconds)
  - Retry mechanism with intelligent backoff
  - Uses shared connection pool for efficient resource utilization
* **Trade-offs**: Network latency, Redis service dependency

## Quick Start

### Basic Usage (90% of scenarios)

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

# Create/get lock (recommended approach)
my_lock = get_or_create_lock("database_operations")

# Simple usage
async def critical_operation():
    async with my_lock:
        # Your critical operations
        await process_data()

# With timeout protection
async def operation_with_timeout():
    try:
        async with lock_context(my_lock, timeout=30.0):
            await long_running_task()
    except TimeoutError:
        print("Operation timed out, will retry later")
```

### Distributed Scenarios

```python
# Distributed lock - cross-process/container coordination
distributed_lock = get_or_create_lock("global_migration", "redis", 
                                      key="migration:v2.0")

async def database_migration():
    async with lock_context(distributed_lock, timeout=300):  # 5 minutes timeout
        await run_migration_safely()
```

### Multi-Component Applications

```python
# Different components use different locks, can run in parallel
db_lock = get_or_create_lock("database_ops")
cache_lock = get_or_create_lock("cache_ops") 
file_lock = get_or_create_lock("file_ops")

async def update_user_data(user_id):
    # Operations can run in parallel since they use different locks
    async with db_lock:
        await update_user_in_database(user_id)
    
    async with cache_lock:
        await invalidate_user_cache(user_id)
```

## Architecture Design

### Core Components

```
aperag.concurrent_control/
├── protocols.py        # Abstract interface definitions
├── threading_lock.py   # Process-local lock implementation  
├── redis_lock.py      # Distributed lock implementation
├── manager.py         # Lock manager and factory functions
└── utils.py           # Utility functions (lock_context)
```

### Design Philosophy

1. **Single Entry Point**: `get_or_create_lock()` is the main interface, covering most use cases
2. **Auto Management**: Named locks are automatically registered to global manager, supporting cross-module reuse
3. **Type Transparency**: Unified `LockProtocol` interface, business code doesn't need to care about specific implementation
4. **Thread Safety**: All components are thread-safe, supporting multi-threaded environments

### Global Lock Manager

The module uses a **global lock manager** to automatically manage all named locks:

```python
# Create in module A
lock_a = get_or_create_lock("shared_resource")

# Get in module B - returns exactly the same lock instance
lock_b = get_or_create_lock("shared_resource")
assert lock_a is lock_b  # True
```

**Advantages**:
- No need to manually pass lock instances
- Cross-module consistency guarantee
- Automatic workspace isolation
- Memory efficiency optimization

## API Reference

### Primary Interface

#### `get_or_create_lock(name, lock_type="threading", **kwargs) -> LockProtocol`
⭐ **Core Function**: Get existing lock or create new one

```python
# Process-local lock (default)
local_lock = get_or_create_lock("local_operations")

# Distributed lock
distributed_lock = get_or_create_lock("distributed_ops", "redis", 
                                      key="app:critical_section")
```

#### `lock_context(lock, timeout=None)`
⭐ **Timeout Control**: Add timeout protection to any lock

```python
async with lock_context(my_lock, timeout=60.0):
    await critical_operation()
```

### Secondary Interface

#### `create_lock(lock_type="threading", **kwargs) -> LockProtocol`
Create lock instance, automatically registers if `name` is specified

#### `get_lock(name) -> Optional[LockProtocol]`
Only get existing lock, returns None if not found

#### `get_default_lock_manager() -> LockManager`
Get global lock manager (advanced usage)

## Deployment Guide

### Celery Deployment Recommendations

| Pool Type | Recommended Lock | Reason |
|-----------|------------------|---------|
| `--pool=prefork` | `RedisLock` | Multi-process needs distributed coordination |
| `--pool=threads` | `ThreadingLock` | Single-process multi-thread, no need for distributed |
| `--pool=gevent` | `ThreadingLock` | Single-process async, better performance |
| `--pool=solo` | `ThreadingLock` | Development/testing environment |

### Containerized Deployment

```python
# Kubernetes/Docker environment recommends Redis locks
k8s_lock = get_or_create_lock("pod_coordination", "redis",
                              key="namespace:app:resource")
```

### Microservices Architecture

```python
# Inter-service coordination uses Redis locks
service_lock = get_or_create_lock("payment_processing", "redis",
                                  key="payment:daily_settlement")
```

## Usage Patterns

### Database Migration

```python
migration_lock = get_or_create_lock("database_migration", "redis",
                                   key="migration:schema_v3")

async def safe_migration():
    try:
        async with lock_context(migration_lock, timeout=600):  # 10 minutes
            await run_database_migration()
    except TimeoutError:
        await notify_admin("Migration timed out, another instance may be running")
```

### Scheduled Task Coordination

```python
# Prevent duplicate execution of scheduled tasks
job_lock = get_or_create_lock("daily_report_job", "redis",
                              key="cron:daily_report")

async def daily_report_task():
    try:
        async with lock_context(job_lock, timeout=30):
            await generate_daily_report()
    except TimeoutError:
        logger.info("Report generation task already running on another node")
```

### Multi-Tenant Resource Isolation

```python
class TenantResourceManager:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        # Each tenant automatically gets independent locks
        self.processing_lock = get_or_create_lock(f"processing_{tenant_id}")
    
    async def process_tenant_data(self, data):
        async with lock_context(self.processing_lock, timeout=120):
            await self._process_data_safely(data)
```

### Cache Update Coordination

```python
cache_lock = get_or_create_lock("cache_refresh", "threading")

async def refresh_cache_safely():
    async with cache_lock:
        if await cache.is_stale():
            await cache.rebuild()
```

## Advanced Features

### Error Handling and Recovery

```python
async def robust_operation():
    lock = get_or_create_lock("critical_section")
    
    try:
        async with lock_context(lock, timeout=30):
            await risky_operation()
    except TimeoutError:
        await handle_timeout_scenario()
    except Exception as e:
        await handle_operation_error(e)
        # Lock will be automatically released
```

### Performance Monitoring

```python
import time
from aperag.concurrent_control import get_or_create_lock, lock_context

async def monitored_operation():
    lock = get_or_create_lock("monitored_resource")
    
    start_time = time.time()
    try:
        async with lock_context(lock, timeout=60):
            await critical_operation()
        
        duration = time.time() - start_time
        await record_metric("operation_duration", duration)
        
    except TimeoutError:
        await record_metric("operation_timeout", 1)
```

### Lock Status Query

```python
from aperag.concurrent_control import get_default_lock_manager

# View all managed locks
manager = get_default_lock_manager()
locks_info = manager.list_locks()
print(f"Currently managing {len(locks_info)} locks:")
for name, lock_type in locks_info.items():
    print(f"  {name}: {lock_type}")
```

## Technical Details

### Redis Connection Management

The module uses a unified Redis connection manager:
- Automatic connection pool management
- Configuration from `settings.memory_redis_url`
- No need for manual connection parameter configuration
- Automatic reconnection and error recovery

### ThreadingLock Optimization

Uses non-blocking polling to avoid event loop blocking:
```python
# Implementation that avoids blocking the event loop
while True:
    acquired = self._lock.acquire(blocking=False)
    if acquired:
        return True
    await asyncio.sleep(0.001)  # Yield control
```

### RedisLock Safety

Uses Lua scripts to guarantee atomic operations:
```lua
-- Lua script for safe lock release
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

## Best Practices

### 1. Lock Naming Convention

```python
# Good naming: clear, hierarchical, meaningful
get_or_create_lock("user_profile_update")
get_or_create_lock("payment_processing_daily")
get_or_create_lock("cache_rebuild_products")

# Avoid: vague, overly generic
get_or_create_lock("lock")  # Too vague
get_or_create_lock("process")  # Not specific enough
```

### 2. Timeout Setting Guidelines

```python
# Set reasonable timeouts based on operation type
async with lock_context(quick_lock, timeout=5):    # Quick operations
    await update_cache()

async with lock_context(medium_lock, timeout=60):   # Medium operations  
    await process_batch_data()

async with lock_context(long_lock, timeout=300):    # Long operations
    await database_migration()
```

### 3. Error Handling Strategy

```python
async def resilient_operation():
    lock = get_or_create_lock("resilient_task")
    
    for attempt in range(3):  # Retry mechanism
        try:
            async with lock_context(lock, timeout=30):
                await critical_operation()
            break  # Exit on success
        except TimeoutError:
            if attempt == 2:  # Last attempt
                raise
            await asyncio.sleep(10)  # Wait before retry
```

### 4. Workspace Isolation

```python
# Use prefixes for logical isolation
workspace_a_lock = get_or_create_lock("workspace_a:data_processing")
workspace_b_lock = get_or_create_lock("workspace_b:data_processing")

# Environment isolation
prod_lock = get_or_create_lock("prod:critical_operation")  
staging_lock = get_or_create_lock("staging:critical_operation")
```

## Performance Considerations

### ThreadingLock
- **Pros**: Sub-millisecond latency, no network overhead, no external dependencies
- **Cons**: Slight overhead compared to `asyncio.Lock`, process-local only
- **Use For**: High-frequency operations, single-process environments

### RedisLock  
- **Pros**: True distributed coordination, automatic expiration, horizontal scaling
- **Cons**: Network latency (1-10ms), Redis service dependency
- **Use For**: Distributed systems, cross-process coordination, high-availability scenarios

### Performance Benchmarks

```python
# Typical performance data (reference only)
ThreadingLock: ~0.1ms   per operation
RedisLock:     ~2-5ms   per operation (local Redis)
RedisLock:     ~10-20ms per operation (remote Redis)
```

## Testing

The module includes 76 comprehensive unit tests:

```bash
# Run all tests
pytest tests/unit_test/concurrent_control/ -v

# Run specific test categories
pytest tests/unit_test/concurrent_control/test_redis_lock.py
pytest tests/unit_test/concurrent_control/test_threading_lock.py
pytest tests/unit_test/concurrent_control/test_thread_safety.py
```

**Test Coverage**:
- ✅ Basic functionality (acquire, release, timeout)
- ✅ Concurrent safety and thread safety
- ✅ Error handling and exception recovery
- ✅ Redis connection manager integration
- ✅ Lock manager lifecycle
- ✅ High-concurrency stress testing

## Troubleshooting

### Common Issues

**1. ThreadingLock doesn't work in Celery prefork mode**
```python
# Solution: Use RedisLock
lock = get_or_create_lock("task_coordination", "redis", key="celery:task")
```

**2. Redis connection errors**
```python
# Check configuration
from aperag.config import settings
print(f"Redis URL: {settings.memory_redis_url}")

# Check connection
from aperag.db.redis_manager import RedisConnectionManager
client = await RedisConnectionManager.get_client()
await client.ping()
```

**3. Frequent lock timeouts**
```python
# Increase timeout or optimize operations
async with lock_context(lock, timeout=120):  # Increase timeout
    await optimized_operation()  # Optimize operation performance
```

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.getLogger('aperag.concurrent_control').setLevel(logging.DEBUG)

# View lock status
manager = get_default_lock_manager()
print("Active locks:", manager.list_locks())

# Check if lock is held
lock = get_lock("my_lock")
if lock:
    print(f"Lock status: {'Held' if lock.is_locked() else 'Available'}")
```

## Changelog

### v1.0.0
- ✨ Unified lock interface design
- ✨ Automatic lock manager  
- ✨ Redis connection manager integration
- ✨ Non-blocking ThreadingLock implementation
- ✨ Comprehensive error handling and logging
- ✨ 76 unit test coverage

## License

This module is part of the ApeRAG project and follows Apache License 2.0. 