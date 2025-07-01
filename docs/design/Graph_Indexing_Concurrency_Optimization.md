# ApeRAG 图谱索引并发性能优化技术方案

## 1. 概述

本文档分析 ApeRAG 系统中图谱索引（Graph Indexing）的并发性能问题，并提出优化方案。

### 1.1 问题现象
- 提交 120 个文档，Celery 并发度为 48
- 实际表现：后期处理速度急剧下降，如同只有 1 个 worker 在工作
- 日志显示：单个文档的图谱索引耗时超过 1200 秒

### 1.2 根本原因
系统使用了一个 workspace 级别的全局锁 `_graph_db_lock`，该锁保护的代码块包含：
1. LLM 调用（生成实体描述摘要）
2. Embedding 计算（向量化）
3. 数据库读写操作

由于锁内包含耗时的网络 I/O 操作，导致锁持有时间极长，其他 worker 只能串行等待。

---

## 2. 问题深度分析

### 2.1 锁的保护范围

```python
# 当前实现
async with graph_db_lock:
    # 1. 读取已有实体数据
    already_node = await knowledge_graph_inst.get_node(entity_name)
    
    # 2. 合并描述，可能调用 LLM（耗时 5-30 秒）
    description = await _handle_entity_relation_summary(...)
    
    # 3. 更新图数据库
    await knowledge_graph_inst.upsert_node(...)
    
    # 4. 计算 embedding 并更新向量数据库（耗时 1-5 秒）
    await entity_vdb.upsert(data_for_vdb)
```

### 2.2 锁的必要性分析

锁的目的是保证数据一致性：
- **防止重复创建**：多个文档可能同时提到"Apple Inc."
- **防止更新丢失**：确保实体描述的"读-改-写"原子性
- **保证数据完整性**：实体的图数据和向量数据需要同步更新

### 2.3 性能瓶颈定量分析

假设每个实体的处理时间：
- 数据库读取：10ms
- LLM 摘要生成：10s（平均）
- Embedding 计算：2s
- 数据库写入：20ms

总锁持有时间 ≈ 12s，这意味着 48 个 worker 实际上以 1/48 的效率运行。

---

## 3. 优化方案

### 3.1 方案一：分离计算与存储（推荐）

**核心思想**：将耗时的计算操作移出锁的保护范围。

```python
async def aprocess_graph_indexing(self, chunks: dict[str, Any]):
    # 阶段1：无锁提取（可并行）
    chunk_results = await extract_entities(chunks, ...)
    
    # 阶段2：预处理（无锁，包含所有耗时操作）
    processed_entities = {}
    processed_relations = {}
    
    for entity_name, entity_data in all_entities.items():
        # 读取现有数据（使用快照隔离）
        existing = await self._read_entity_snapshot(entity_name)
        
        # 计算新描述（包括 LLM 调用）
        merged_description = await self._compute_description(
            existing, entity_data, use_llm=True
        )
        
        # 计算 embedding
        embedding = await self.embedding_func(merged_description)
        
        processed_entities[entity_name] = {
            "description": merged_description,
            "embedding": embedding,
            "version": existing.version + 1 if existing else 1
        }
    
    # 阶段3：原子更新（最小化锁范围）
    async with self._graph_db_lock:
        # 批量更新，只包含快速的数据库操作
        await self._batch_update_entities(processed_entities)
        await self._batch_update_relations(processed_relations)
```

**优势**：
- 锁持有时间从 12s 降至 30ms
- 并发度提升 400 倍
- 保持数据一致性

### 3.2 方案二：实体级细粒度锁

**核心思想**：每个实体使用独立的锁，不同实体可以并行处理。

```python
async def process_entity(self, entity_name: str, entity_data: list):
    # 获取实体专属锁
    entity_lock = get_or_create_lock(f"entity_{self.workspace}_{entity_name}")
    
    async with entity_lock:
        # 处理单个实体的完整流程
        existing = await self.knowledge_graph.get_node(entity_name)
        merged = await self._merge_and_summarize(existing, entity_data)
        await self.knowledge_graph.upsert_node(entity_name, merged)
        await self.entity_vdb.upsert({entity_name: merged})
```

**优势**：
- 不同实体完全并行
- 实现相对简单

**劣势**：
- 热门实体仍会成为瓶颈
- 需要管理大量锁对象

### 3.3 方案三：乐观并发控制

**核心思想**：使用版本号代替锁，冲突时重试。

```python
async def update_entity_optimistic(self, entity_name: str, new_data: dict):
    max_retries = 3
    for attempt in range(max_retries):
        # 1. 读取当前版本
        current = await self.knowledge_graph.get_node_with_version(entity_name)
        version = current["version"] if current else 0
        
        # 2. 计算新状态（无锁）
        merged = await self._merge_and_compute(current, new_data)
        
        # 3. 尝试更新
        success = await self.knowledge_graph.update_if_version_matches(
            entity_name, merged, expected_version=version
        )
        
        if success:
            return
        
        # 4. 版本冲突，等待后重试
        await asyncio.sleep(0.1 * (2 ** attempt))
    
    raise Exception(f"Failed to update {entity_name} after {max_retries} attempts")
```

---

## 4. 推荐实施方案

建议采用**方案一（分离计算与存储）**，原因如下：

1. **性能提升最大**：锁持有时间减少 99.7%
2. **实现复杂度适中**：主要是代码重构，不需要修改底层存储
3. **可靠性高**：保持了强一致性保证
4. **可扩展性好**：未来可以进一步优化为分布式处理

### 4.1 实施步骤

1. **重构 `merge_nodes_and_edges` 函数**
   - 分离数据读取、计算处理、数据写入三个阶段
   - 将 LLM 调用和 Embedding 计算移到锁外

2. **实现批量更新接口**
   - 在图存储和向量存储中实现高效的批量更新方法
   - 使用数据库事务保证原子性

3. **添加冲突检测机制**
   - 在锁内进行版本检查，确保数据未被其他进程修改
   - 发生冲突时的处理策略

4. **性能测试与调优**
   - 测试不同并发度下的性能表现
   - 调整批量大小等参数

---

## 5. 预期效果

实施优化后，预期效果：

- **并发处理能力**：从实质 1 个并发提升到接近 48 个并发
- **处理时间**：120 个文档的总处理时间从 ~6 小时降至 ~10 分钟
- **资源利用率**：CPU 和网络资源利用率提升 40 倍以上
- **可扩展性**：支持更大规模的文档批量处理

---

## 6. 风险与应对

### 6.1 数据一致性风险
- **风险**：计算过程中数据被修改
- **应对**：使用快照读取 + 版本校验

### 6.2 内存使用增加
- **风险**：预处理阶段需要缓存大量数据
- **应对**：实现分批处理机制

### 6.3 错误处理复杂度
- **风险**：部分更新失败的回滚处理
- **应对**：实现补偿事务机制

---

## 7. 总结

当前的全局锁机制严重限制了系统的并发处理能力。通过将耗时的计算操作（LLM 调用、Embedding 生成）移出锁的保护范围，可以将锁持有时间减少 99% 以上，从而实现真正的高并发处理。这种优化在保证数据一致性的同时，能够充分利用系统资源，大幅提升处理效率。

---

## 8. 最终推荐方案：Multi-Lock 实体级并发控制

经过深入分析，确定采用 **Multi-Lock 实体级并发控制方案**。这是在保证数据一致性前提下，实现最佳并发性能的方案。

### 8.1 方案核心思想

```python
# 核心思想：只锁定当前文档涉及的具体实体
async def aprocess_graph_indexing(self, chunks: dict[str, Any]):
    # 阶段1：提取实体和关系（完全并行，无锁）
    chunk_results = await extract_entities(chunks, ...)
    
    # 阶段2：收集所有涉及的实体名称
    all_entity_names = set()
    for maybe_nodes, maybe_edges in chunk_results:
        # 收集节点实体
        all_entity_names.update(maybe_nodes.keys())
        # 收集边的源实体和目标实体
        for edge_key in maybe_edges.keys():
            all_entity_names.update(edge_key)
    
    # 阶段3：为所有实体创建锁，并按名称排序（防止死锁）
    entity_locks = [
        get_or_create_lock(f"entity:{entity_name}:{self.workspace}")
        for entity_name in sorted(all_entity_names)
    ]
    
    # 阶段4：使用 MultiLock 获取所有相关锁，然后执行合并
    async with MultiLock(entity_locks):
        await merge_nodes_and_edges(
            chunk_results,
            self.knowledge_graph_inst,
            self.entity_vdb,
            self.relationships_vdb,
            # ... 其他参数
        )
```

### 8.2 方案优势

1. **精确并发控制**：
   - 只锁定当前文档涉及的具体实体
   - 不同文档如果涉及不同实体，可以完全并行处理
   - 相同实体的更新严格串行，保证数据一致性

2. **防止死锁**：
   - 通过实体名称排序，保证所有worker按相同顺序获取锁
   - 使用 MultiLock 统一管理多个锁的获取和释放

3. **性能可预期**：
   - 并发度 = min(worker数量, 不重叠实体组合数)
   - 对于实体重叠度低的文档集，接近完全并行
   - 对于实体重叠度高的文档集，仍有显著改善

### 8.3 实现步骤

#### 步骤1：实现 MultiLock 类

在 `aperag/concurrent_control/core.py` 中添加：

```python
class MultiLock:
    """多重锁管理器，按排序顺序获取多个锁以防止死锁"""
    
    def __init__(self, locks: list[LockProtocol]):
        # 按锁名称排序，确保获取顺序一致
        self._locks = sorted(locks, key=lambda lock: getattr(lock, "_name", str(id(lock))))
        self._acquired_locks = []
    
    async def __aenter__(self):
        """按顺序获取所有锁"""
        for lock in self._locks:
            await lock.acquire()
            self._acquired_locks.append(lock)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """按相反顺序释放所有锁"""
        for lock in reversed(self._acquired_locks):
            await lock.release()
        self._acquired_locks.clear()
```

#### 步骤2：修改 LightRAG 类

在 `aperag/graph/lightrag/lightrag.py` 中：

```python
from aperag.concurrent_control import MultiLock, get_or_create_lock

class LightRAG:
    def __post_init__(self):
        # 移除全局锁
        # self._graph_db_lock = get_graph_db_lock(self.workspace)
        pass
    
    async def aprocess_graph_indexing(self, chunks: dict[str, Any]):
        # 实体提取阶段（无锁，完全并行）
        chunk_results = await extract_entities(chunks, ...)
        
        # 收集所有涉及的实体
        all_entity_names = set()
        for maybe_nodes, maybe_edges in chunk_results:
            all_entity_names.update(maybe_nodes.keys())
            for edge_key in maybe_edges.keys():
                all_entity_names.update(edge_key)
        
        # 创建实体锁并排序
        entity_locks = [
            get_or_create_lock(f"entity:{name}:{self.workspace}")
            for name in sorted(all_entity_names)
        ]
        
        # 使用 MultiLock 进行合并操作
        async with MultiLock(entity_locks):
            await merge_nodes_and_edges(
                chunk_results,
                self.knowledge_graph_inst,
                self.entity_vdb,
                self.relationships_vdb,
                # ... 其他参数，不再传递 graph_db_lock
            )
```

#### 步骤3：清理 operate.py

移除 `merge_nodes_and_edges` 函数的 `graph_db_lock` 参数及相关逻辑。

### 8.4 性能分析

假设120个文档的实体重叠情况：

**场景1：低重叠度（如不同领域文档）**
- 实体重叠率：10%
- 预期并发度：接近48（几乎完全并行）
- 性能提升：45倍以上

**场景2：中等重叠度（如同领域不同公司）**
- 实体重叠率：50%
- 预期并发度：15-25
- 性能提升：15-25倍

**场景3：高重叠度（如同一公司的多篇报告）**
- 实体重叠率：80%
- 预期并发度：5-10
- 性能提升：5-10倍

即使在最坏情况下，仍有5倍以上的性能提升。

### 8.5 风险评估

1. **锁数量增加**：每个实体一个锁，但Redis锁管理开销很小
2. **内存占用**：MultiLock需要维护锁列表，但影响微乎其微
3. **实现复杂度**：相比分组方案，实现更简洁直观

### 8.6 监控指标

建议添加以下监控：
- 平均实体重叠度
- 平均锁等待时间
- 并发任务数分布
- 单个文档处理时间分布

---

## 9. 总结

Multi-Lock 实体级并发控制方案是解决当前性能瓶颈的最佳选择：

1. **精确性**：只锁定必要的实体，最小化锁竞争
2. **安全性**：通过排序防止死锁，保证数据一致性
3. **可预测性**：性能提升直接与实体重叠度相关
4. **简洁性**：实现简单，易于维护和调试

相比全局锁的"一刀切"和分组锁的"跨组冲突"，Multi-Lock实现了"精确制导"的并发控制，是在数据一致性和性能之间的最佳平衡点。

---

## 10. 实施状态

**✅ 已完成实施（2025-01-03）**

Multi-Lock 实体级并发控制方案已成功实施，主要改动包括：

### 10.1 核心实现

1. **MultiLock 类实现**（`aperag/concurrent_control/core.py`）
   - 实现了按锁名称排序的多重锁管理器
   - 支持批量获取和释放锁
   - 内置死锁预防机制

2. **LightRAG 类改造**（`aperag/graph/lightrag/lightrag.py`）
   - 移除了全局锁 `_graph_db_lock`
   - 在 `aprocess_graph_indexing` 中实现了实体级锁收集和排序
   - 使用 MultiLock 保护合并操作

3. **operate.py 清理**（`aperag/graph/lightrag/operate.py`）
   - 移除了 `merge_nodes_and_edges` 函数的 `graph_db_lock` 参数
   - 简化了函数签名和实现

### 10.2 预期收益

- **并发度提升**：从 1 个全局锁提升到 N 个实体级锁
- **性能改善**：根据实体重叠度，性能提升 5-45 倍
- **死锁预防**：通过排序机制完全避免死锁
- **维护性增强**：代码更清晰，锁的粒度更明确

### 10.3 后续监控

建议在生产环境中监控以下指标：
- 平均锁等待时间
- 实体重叠度分布
- 并发任务执行效率
- 系统整体吞吐量
