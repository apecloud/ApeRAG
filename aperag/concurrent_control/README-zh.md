## 统一并发控制模块

一个灵活且可重用的并发控制系统，为任何 Python 应用程序提供统一的锁定机制。本模块旨在处理不同的部署场景和任务队列环境。

### 功能

* **简洁接口**：`get_or_create_lock()` 函数涵盖 90% 的用例
* **自动管理锁**：命名锁会自动注册，方便后续检索
* **灵活的超时支持**：在异步上下文中可使用 `lock_context()` 进行超时控制
* **通用适用性**：适用于任何需要并发控制的 Python 应用程序
* **易于扩展**：清晰的协议接口，可实现新的锁类型
* **生产就绪**：组织化管理和全面的错误处理

### 支持的锁类型

#### ThreadingLock

* **最适合**：单进程环境（Celery `--pool=solo`, `--pool=threads`, `--pool=gevent`）
* **实现**：`threading.Lock` 包装 `asyncio.to_thread`
* **特点**：
    * 支持协程和线程并发
    * 事件循环非阻塞
    * 开销低于分布式锁
* **限制**：**不支持**跨多个进程

#### RedisLock

* **最适合**：多进程环境（Celery `--pool=prefork`, 分布式部署）
* **实现**：基于 Redis 的分布式锁
* **特点**：
    * 支持跨进程、容器和机器
    * 自动锁过期，防止死锁
    * 锁获取重试机制
    * 使用Lua脚本安全释放锁
* **权衡**：网络开销，依赖 Redis

### 快速开始

#### 基本用法 (90% 的用例)

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

# 创建/获取一个受管理的锁（大多数情况推荐）
my_lock = get_or_create_lock("database_operations")

# 使用默认行为
async def critical_operation():
    async with my_lock:
        # 你的受保护操作在此
        await some_critical_work()

# 使用超时支持
async def operation_with_timeout():
    async with lock_context(my_lock, timeout=5.0):
        await some_critical_work()
```

#### 高级用法 (10% 的用例)

```python
from aperag.concurrent_control import create_lock, get_lock, get_or_create_lock

# 创建一个命名锁（自动管理）
create_lock("threading", name="database_ops")

# 稍后检索相同的锁实例
lock = get_lock("database_ops")
if lock:
    async with lock:
        await execute_query()

# 或者一次性获取现有/创建新的锁（推荐）
lock = get_or_create_lock("cache_ops")
async with lock:
    await update_cache()
```

### 默认锁管理器

该模块使用**默认全局锁管理器**自动管理命名锁。这提供了以下优点：

#### 自动注册

```python
# 创建命名锁时，它会自动注册
lock1 = create_lock("threading", name="shared_resource")

# 之后，在应用程序的任何地方，获取相同的实例
lock2 = get_lock("shared_resource") 
assert lock1 is lock2  # True - 相同的锁实例
```

#### 跨模块一致性

```python
# 在模块 A 中
from aperag.concurrent_control import get_or_create_lock
db_lock = get_or_create_lock("database_operations")

# 在模块 B 中
from aperag.concurrent_control import get_or_create_lock
same_db_lock = get_or_create_lock("database_operations")
# 获取完全相同的锁实例 - 无需协调
```

#### 工作区隔离

```python
# 不同的工作区会自动获得不同的锁
workspace_a_lock = get_or_create_lock("operations_workspace_a")
workspace_b_lock = get_or_create_lock("operations_workspace_b")
# 它们是完全独立的锁
```

#### 锁生命周期

* **创建**：命名锁在创建时自动注册
* **检索**：使用 `get_lock()` 或 `get_or_create_lock()` 访问
* **清理**：锁在应用程序的整个生命周期中持续存在
* **线程安全**：默认管理器是线程安全的

---
### API 参考

#### 主要接口 (推荐)

#### `get_or_create_lock(name: str, lock_type: str = "threading", **kwargs) -> LockProtocol`

**⭐ 大多数用例的主要函数。** 获取现有锁或创建新锁。

```python
# 简单用法（涵盖 90% 的场景）
my_lock = get_or_create_lock("database_operations")

# 带特定锁类型
redis_lock = get_or_create_lock("distributed_ops", "redis", key="my_key")
```

#### `lock_context(lock: LockProtocol, timeout: float = None)`

**⭐ 异步上下文的超时支持。**

```python
# 带超时
async with lock_context(my_lock, timeout=10.0):
    await long_operation()
```

#### 次要接口 (特殊情况)

#### `get_lock(name: str) -> Optional[LockProtocol]`

仅获取现有锁。如果未找到，则返回 None。

```python
# 检查锁是否存在
existing = get_lock("my_operation")
if existing:
    async with existing:
        await work()
```

#### `create_lock(lock_type: str = "threading", **kwargs) -> LockProtocol`

创建一个新锁。如果提供了 `name`，它将自动注册。

```python
# 创建匿名锁（不管理）
temp_lock = create_lock("threading")

# 创建命名锁（自动管理）
managed_lock = create_lock("threading", name="my_lock")
```

#### 高级访问

#### `get_default_lock_manager() -> LockManager`

访问默认锁管理器以进行高级操作。

```python
from aperag.concurrent_control import get_default_lock_manager

manager = get_default_lock_manager()
print("Managed locks:", manager.list_locks())
manager.remove_lock("some_lock_name")
```

---
### 部署建议

#### Celery 部署模式

| 池类型           | 推荐锁           | 用例                 |
| :--------------- | :--------------- | :------------------- |
| `--pool=prefork` | `RedisLock` | 生产多进程           |
| `--pool=threads` | `ThreadingLock`  | 单进程多线程         |
| `--pool=gevent`  | `ThreadingLock`  | 单进程异步           |
| `--pool=solo`    | `ThreadingLock`  | 开发/测试            |

### 使用模式

#### 简单操作 (最常见)

```python
from aperag.concurrent_control import get_or_create_lock

# 简单用例的一行代码
my_lock = get_or_create_lock("file_operations")
async with my_lock:
    await process_file()
```

#### 带超时控制

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

migration_lock = get_or_create_lock("database_migration")

async def migration_with_timeout():
    try:
        async with lock_context(migration_lock, timeout=30.0):
            await run_migration()
    except TimeoutError:
        print("迁移超时，稍后重试")
```

#### 多组件应用程序

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

# 不同的组件使用相同的锁名称
database_lock = get_or_create_lock("database_operations")
cache_lock = get_or_create_lock("cache_operations")

async def update_user_data(user_id):
    # 操作可以并行运行，因为它们使用不同的锁
    async with database_lock:
        await update_user_in_db(user_id)
    
    async with cache_lock:
        await invalidate_user_cache(user_id)

async def batch_operation():
    # 但批量操作会正确序列化
    async with lock_context(database_lock, timeout=60):
        await process_large_batch()
```

#### 工作区/租户隔离

```python
from aperag.concurrent_control import get_or_create_lock

class WorkspaceManager:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        # 每个工作区都会自动获取自己的锁
        self.processing_lock = get_or_create_lock(f"processing_{workspace_id}")
    
    async def process_data(self, data):
        async with self.processing_lock:
            await self._process_data_for_workspace(data)

# 不同的工作区会自动获取不同的锁
manager_a = WorkspaceManager("tenant_a")
manager_b = WorkspaceManager("tenant_b")
# 它们的锁是完全独立的
```

---
### 错误处理

该模块提供全面的错误处理：

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

my_lock = get_or_create_lock("critical_section")

# 异常时自动释放锁
async def operation_with_error():
    try:
        async with my_lock:
            await some_operation()
            raise ValueError("Something went wrong")
    except ValueError:
        # 锁会自动释放
        pass

# 超时处理
async def operation_with_timeout():
    try:
        async with lock_context(my_lock, timeout=10.0):
            await long_operation()
    except TimeoutError:
        print("操作超时")
```

### 性能考量

#### ThreadingLock

* **优点**：单进程场景开销低，无外部依赖
* **缺点**：开销高于 `asyncio.Lock`，仅限进程本地

#### RedisLock (未来功能)

* **优点**：跨进程和机器工作，内置过期机制
* **缺点**：网络开销，依赖 Redis，延迟更高

### 迁移指南

#### 从 asyncio.Lock

```python
# 之前
import asyncio
old_lock = asyncio.Lock()

async with old_lock:
    # 不支持超时
    await work()

# 之后
from aperag.concurrent_control import get_or_create_lock, lock_context

new_lock = get_or_create_lock("my_operation")

# 默认行为
async with new_lock:
    await work()

# 带超时
async with lock_context(new_lock, timeout=5.0):
    await work()
```

### 为什么选择这种设计？

#### 主接口简洁性

```python
# 90% 的用例 - 清晰明确的函数名
my_lock = get_or_create_lock("my_operation")
async with my_lock:
    await work()
```

#### `lock_context()` 的价值

```python
# 没有 lock_context，超时需要冗长的代码：
if await my_lock.acquire(timeout=5.0):
    try:
        await work()
    finally:
        await my_lock.release()

# 有了 lock_context - 简洁安全：
async with lock_context(my_lock, timeout=5.0):
    await work()
```

#### 自动管理锁

```python
# 创建一次，随处使用
create_lock("threading", name="database_ops")

# 之后在任何模块中
same_lock = get_lock("database_ops")  # 获取相同的实例
```

#### 单一事实来源

* 所有命名锁都由默认管理器自动管理
* 无需传递锁实例
* 应用程序中命名一致
* 通过命名约定实现工作区隔离

---
### 测试

该模块包含全面的单元测试。运行测试：

```bash
pytest tests/unit_test/concurrent_control/
```

### 未来增强

1.  **锁监控**：使用指标和性能监控
2.  **高级功能**：锁优先级，死锁检测
3.  **连接池优化**：优化Redis连接管理

### 许可证

本模块是 ApeRAG 项目的一部分，并遵循相同的许可证条款。