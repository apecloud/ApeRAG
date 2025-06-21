# 统一并发控制模块

一个灵活且可重用的并发控制系统，为 Python 应用程序提供统一的锁定机制。本模块专为不同的部署场景和任务队列环境而设计，提供简洁的 API 和强大的功能。

## 核心特性

* **极简 API**：`get_or_create_lock()` 一个函数解决 90% 的使用场景
* **自动管理**：命名锁自动注册和复用，无需手动传递锁实例
* **超时支持**：使用 `lock_context()` 提供灵活的超时控制
* **线程安全**：锁管理器完全线程安全，支持多线程并发访问
* **生产就绪**：完整的错误处理、日志记录和监控支持
* **零配置**：开箱即用，Redis 连接自动管理

## 支持的锁类型

### ThreadingLock - 进程内锁
* **适用场景**：单进程环境（Celery `--pool=solo`, `--pool=threads`, `--pool=gevent`）
* **技术实现**：基于 `threading.Lock`，使用非阻塞轮询避免事件循环阻塞
* **性能特点**：
  - 低延迟，无网络开销
  - 支持协程和线程并发
  - 事件循环友好的异步实现
* **限制**：仅限单进程内使用

### RedisLock - 分布式锁
* **适用场景**：多进程环境（Celery `--pool=prefork`，容器化部署，分布式系统）
* **技术实现**：基于 Redis SET NX EX 模式，使用 Lua 脚本保证原子性
* **高级特性**：
  - 跨进程、容器、机器工作
  - 自动过期防止死锁（默认 120 秒）
  - 重试机制和智能退避
  - 使用共享连接池，高效资源利用
* **权衡**：网络延迟，依赖 Redis 服务

## 快速开始

### 基础用法（90% 的场景）

```python
from aperag.concurrent_control import get_or_create_lock, lock_context

# 创建/获取锁（推荐方式）
my_lock = get_or_create_lock("database_operations")

# 简单使用
async def critical_operation():
    async with my_lock:
        # 你的关键操作
        await process_data()

# 带超时保护
async def operation_with_timeout():
    try:
        async with lock_context(my_lock, timeout=30.0):
            await long_running_task()
    except TimeoutError:
        print("操作超时，稍后重试")
```

### 分布式场景

```python
# 分布式锁 - 跨进程、容器协调
distributed_lock = get_or_create_lock("global_migration", "redis", 
                                      key="migration:v2.0")

async def database_migration():
    async with lock_context(distributed_lock, timeout=300):  # 5分钟超时
        await run_migration_safely()
```

### 多组件应用

```python
# 不同组件使用不同的锁，并行执行不冲突
db_lock = get_or_create_lock("database_ops")
cache_lock = get_or_create_lock("cache_ops") 
file_lock = get_or_create_lock("file_ops")

async def update_user_data(user_id):
    # 操作可以并行，因为使用不同的锁
    async with db_lock:
        await update_user_in_database(user_id)
    
    async with cache_lock:
        await invalidate_user_cache(user_id)
```

## 架构设计

### 核心组件

```
aperag.concurrent_control/
├── protocols.py        # 抽象接口定义
├── threading_lock.py   # 进程内锁实现  
├── redis_lock.py      # 分布式锁实现
├── manager.py         # 锁管理器和工厂函数
└── utils.py           # 工具函数（lock_context）
```

### 设计理念

1. **单一入口**：`get_or_create_lock()` 是主要接口，覆盖绝大多数使用场景
2. **自动管理**：命名锁自动注册到全局管理器，支持跨模块复用
3. **类型透明**：统一的 `LockProtocol` 接口，业务代码无需关心锁的具体实现
4. **线程安全**：所有组件都是线程安全的，支持多线程环境

### 全局锁管理器

模块使用**全局锁管理器**自动管理所有命名锁：

```python
# 在模块 A 中创建
lock_a = get_or_create_lock("shared_resource")

# 在模块 B 中获取 - 返回完全相同的锁实例
lock_b = get_or_create_lock("shared_resource")
assert lock_a is lock_b  # True
```

**优势**：
- 无需手动传递锁实例
- 跨模块一致性保证
- 自动工作区隔离
- 内存效率优化

## API 参考

### 主要接口

#### `get_or_create_lock(name, lock_type="threading", **kwargs) -> LockProtocol`
⭐ **核心函数**：获取现有锁或创建新锁

```python
# 进程内锁（默认）
local_lock = get_or_create_lock("local_operations")

# 分布式锁
distributed_lock = get_or_create_lock("distributed_ops", "redis", 
                                      key="app:critical_section")
```

#### `lock_context(lock, timeout=None)`
⭐ **超时控制**：为任何锁添加超时保护

```python
async with lock_context(my_lock, timeout=60.0):
    await critical_operation()
```

### 辅助接口

#### `create_lock(lock_type="threading", **kwargs) -> LockProtocol`
创建锁实例，如果指定 `name` 则自动注册

#### `get_lock(name) -> Optional[LockProtocol]`
仅获取已存在的锁，不存在时返回 None

#### `get_default_lock_manager() -> LockManager`
获取全局锁管理器（高级用法）

## 部署指南

### Celery 部署建议

| 池类型 | 推荐锁类型 | 原因 |
|--------|------------|------|
| `--pool=prefork` | `RedisLock` | 多进程需要分布式协调 |
| `--pool=threads` | `ThreadingLock` | 单进程多线程，无需分布式 |
| `--pool=gevent` | `ThreadingLock` | 单进程异步，性能更好 |
| `--pool=solo` | `ThreadingLock` | 开发测试环境 |

### 容器化部署

```python
# Kubernetes/Docker 环境推荐使用 Redis 锁
k8s_lock = get_or_create_lock("pod_coordination", "redis",
                              key="namespace:app:resource")
```

### 微服务架构

```python
# 服务间协调使用 Redis 锁
service_lock = get_or_create_lock("payment_processing", "redis",
                                  key="payment:daily_settlement")
```

## 使用模式

### 数据库迁移

```python
migration_lock = get_or_create_lock("database_migration", "redis",
                                   key="migration:schema_v3")

async def safe_migration():
    try:
        async with lock_context(migration_lock, timeout=600):  # 10分钟
            await run_database_migration()
    except TimeoutError:
        await notify_admin("迁移超时，可能有其他实例在运行")
```

### 定时任务协调

```python
# 防止定时任务重复执行
job_lock = get_or_create_lock("daily_report_job", "redis",
                              key="cron:daily_report")

async def daily_report_task():
    try:
        async with lock_context(job_lock, timeout=30):
            await generate_daily_report()
    except TimeoutError:
        logger.info("报告生成任务已在其他节点运行")
```

### 多租户资源隔离

```python
class TenantResourceManager:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        # 每个租户自动获得独立的锁
        self.processing_lock = get_or_create_lock(f"processing_{tenant_id}")
    
    async def process_tenant_data(self, data):
        async with lock_context(self.processing_lock, timeout=120):
            await self._process_data_safely(data)
```

### 缓存更新协调

```python
cache_lock = get_or_create_lock("cache_refresh", "threading")

async def refresh_cache_safely():
    async with cache_lock:
        if await cache.is_stale():
            await cache.rebuild()
```

## 高级特性

### 错误处理和恢复

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
        # 锁会自动释放
```

### 性能监控

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

### 锁状态查询

```python
from aperag.concurrent_control import get_default_lock_manager

# 查看所有管理的锁
manager = get_default_lock_manager()
locks_info = manager.list_locks()
print(f"当前管理 {len(locks_info)} 个锁:")
for name, lock_type in locks_info.items():
    print(f"  {name}: {lock_type}")
```

## 技术细节

### Redis 连接管理

模块使用统一的 Redis 连接管理器：
- 自动连接池管理
- 配置来自 `settings.memory_redis_url`
- 无需手动配置连接参数
- 自动重连和错误恢复

### ThreadingLock 优化

采用非阻塞轮询避免事件循环阻塞：
```python
# 避免阻塞事件循环的实现
while True:
    acquired = self._lock.acquire(blocking=False)
    if acquired:
        return True
    await asyncio.sleep(0.001)  # 让出控制权
```

### RedisLock 安全性

使用 Lua 脚本保证原子操作：
```lua
-- 安全释放锁的 Lua 脚本
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

## 最佳实践

### 1. 锁命名规范

```python
# 好的命名：清晰、分层、有意义
get_or_create_lock("user_profile_update")
get_or_create_lock("payment_processing_daily")
get_or_create_lock("cache_rebuild_products")

# 避免的命名：模糊、过于通用
get_or_create_lock("lock")  # 太模糊
get_or_create_lock("process")  # 不够具体
```

### 2. 超时设置指导

```python
# 根据操作类型设置合理超时
async with lock_context(quick_lock, timeout=5):    # 快速操作
    await update_cache()

async with lock_context(medium_lock, timeout=60):   # 中等操作  
    await process_batch_data()

async with lock_context(long_lock, timeout=300):    # 长时间操作
    await database_migration()
```

### 3. 错误处理策略

```python
async def resilient_operation():
    lock = get_or_create_lock("resilient_task")
    
    for attempt in range(3):  # 重试机制
        try:
            async with lock_context(lock, timeout=30):
                await critical_operation()
            break  # 成功则退出
        except TimeoutError:
            if attempt == 2:  # 最后一次尝试
                raise
            await asyncio.sleep(10)  # 等待后重试
```

### 4. 工作区隔离

```python
# 使用前缀实现逻辑隔离
workspace_a_lock = get_or_create_lock("workspace_a:data_processing")
workspace_b_lock = get_or_create_lock("workspace_b:data_processing")

# 环境隔离
prod_lock = get_or_create_lock("prod:critical_operation")  
staging_lock = get_or_create_lock("staging:critical_operation")
```

## 性能考量

### ThreadingLock
- **优势**：亚毫秒级延迟，无网络开销，无外部依赖
- **劣势**：相比 `asyncio.Lock` 有轻微开销，仅限进程内
- **适用**：高频操作，单进程环境

### RedisLock  
- **优势**：真正的分布式协调，自动过期机制，横向扩展
- **劣势**：网络延迟（1-10ms），依赖 Redis 可用性
- **适用**：分布式系统，跨进程协调，高可用场景

### 性能基准

```python
# 典型性能数据（仅供参考）
ThreadingLock: ~0.1ms   per operation
RedisLock:     ~2-5ms   per operation (本地 Redis)
RedisLock:     ~10-20ms per operation (远程 Redis)
```

## 测试

模块包含 76 个全面的单元测试：

```bash
# 运行所有测试
pytest tests/unit_test/concurrent_control/ -v

# 运行特定测试类别
pytest tests/unit_test/concurrent_control/test_redis_lock.py
pytest tests/unit_test/concurrent_control/test_threading_lock.py
pytest tests/unit_test/concurrent_control/test_thread_safety.py
```

**测试覆盖**：
- ✅ 基本功能（获取、释放、超时）
- ✅ 并发安全性和线程安全
- ✅ 错误处理和异常恢复
- ✅ Redis 连接管理集成
- ✅ 锁管理器生命周期
- ✅ 高并发压力测试

## 故障排除

### 常见问题

**1. ThreadingLock 在 Celery prefork 模式下不工作**
```python
# 解决方案：使用 RedisLock
lock = get_or_create_lock("task_coordination", "redis", key="celery:task")
```

**2. Redis 连接错误**
```python
# 检查配置
from aperag.config import settings
print(f"Redis URL: {settings.memory_redis_url}")

# 检查连接
from aperag.db.redis_manager import RedisConnectionManager
client = await RedisConnectionManager.get_client()
await client.ping()
```

**3. 锁超时频繁发生**
```python
# 增加超时时间或优化操作
async with lock_context(lock, timeout=120):  # 增加超时
    await optimized_operation()  # 优化操作性能
```

### 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger('aperag.concurrent_control').setLevel(logging.DEBUG)

# 查看锁状态
manager = get_default_lock_manager()
print("活跃锁列表:", manager.list_locks())

# 检查锁是否被持有
lock = get_lock("my_lock")
if lock:
    print(f"锁状态: {'已持有' if lock.is_locked() else '可用'}")
```

## 更新历史

### v1.0.0
- ✨ 统一的锁接口设计
- ✨ 自动锁管理器  
- ✨ Redis 连接管理器集成
- ✨ 非阻塞 ThreadingLock 实现
- ✨ 全面的错误处理和日志
- ✨ 76 个单元测试覆盖

## 许可证

本模块是 ApeRAG 项目的一部分，遵循 Apache License 2.0 许可证。