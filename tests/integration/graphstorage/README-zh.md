# 图存储集成测试套件

本目录包含 ApeRAG 图存储实现的后端集成测试。这些测试直接验证存储行为，不属于产品层面的 HTTP E2E。

## 🌟 特色亮点

### Test Oracle 测试模式 ⚖️

本测试套件采用了**Test Oracle 模式**，这是一种优雅且强大的测试方法论：

- **双重验证**: 每个图存储实现都与经过验证的 NetworkX 基线实现进行比较
- **自动同步**: Oracle 自动将所有写操作同步到真实存储和基线存储
- **即时验证**: 每次查询操作都会实时比较两个存储的结果，确保完全一致
- **零误差容忍**: 任何不一致都会立即抛出异常，确保实现的严格正确性
- **灵活比较**: 支持浮点数容差、字段顺序无关性、边方向规范化等智能比较策略

这种方法确保了每个图存储实现都经过**严格的行为验证**，避免了单纯依赖预期结果可能遗漏的边界情况。

## 📁 文件结构

```
tests/integration/graphstorage/
├── conftest.py                     # pytest 配置
├── graph_storage_oracle.py         # Test Oracle 实现 ⚖️
├── networkx_baseline_storage.py    # NetworkX 基线实现
├── test_graph_storage.py           # 通用测试套件
├── test_neo4j_storage.py           # Neo4j 存储测试
└── graph_storage_test_data.json    # 测试数据
```

## 🎯 测试用例覆盖

### 节点操作测试
- `test_has_node` - 节点存在性检查
- `test_get_node` - 单节点数据获取
- `test_get_nodes_batch` - 批量节点获取
- `test_node_degree` - 节点度数计算
- `test_node_degrees_batch` - 批量节点度数计算
- `test_upsert_node` - 节点创建/更新
- `test_delete_node` - 单节点删除
- `test_remove_nodes` - 批量节点删除

### 边操作测试
- `test_has_edge` - 边存在性检查
- `test_get_edge` - 单边数据获取
- `test_get_edges_batch` - 批量边获取
- `test_get_node_edges` - 节点关联边获取
- `test_get_nodes_edges_batch` - 批量节点关联边获取
- `test_edge_degree` - 边度数计算
- `test_edge_degrees_batch` - 批量边度数计算
- `test_upsert_edge` - 边创建/更新
- `test_remove_edges` - 批量边删除

### 复杂操作测试
- `test_data_integrity` - 数据完整性验证
- `test_large_batch_operations` - 大批量操作性能
- `test_data_consistency_after_operations` - 操作后一致性检查
- `test_get_all_labels` - 所有标签获取
- `test_interface_coverage_summary` - 接口覆盖率总结

## 🛠️ 环境配置

### 必需环境变量

#### Neo4j 配置
```bash
NEO4J_HOST=127.0.0.1
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 环境变量配置方法

1. **使用 .env 文件** (推荐)
   ```bash
   # 在项目根目录创建 .env 文件
   cp envs/env.template .env
   # 编辑 .env 文件，添加上述配置
   ```

2. **使用环境变量**
   ```bash
   export NEO4J_HOST=127.0.0.1
   export NEO4J_PORT=7687
   # ... 其他变量
   ```

## 🚀 运行测试

### 运行所有图存储集成测试
```bash
# 在项目根目录执行
uv run pytest tests/integration/graphstorage/ -v
```

### 运行特定数据库测试

#### Neo4j 存储测试
```bash
uv run pytest tests/integration/graphstorage/test_neo4j_storage.py::TestNeo4jStorage -v
```

### 运行特定测试用例
```bash
# 测试 Neo4j 节点操作
uv run pytest tests/integration/graphstorage/test_neo4j_storage.py::TestNeo4jStorage::test_has_node -v
```

## 📊 测试数据

- **数据文件**: `graph_storage_test_data.json`
- **数据格式**: 每行一个 JSON 对象（节点或边）
- **数据规模**: 包含大量真实图数据，确保测试的全面性
- **数据来源**: 从实际 LightRAG 运行中导出的图结构

## ⚠️ 注意事项

### 环境依赖
- 如果缺少对应数据库的环境变量配置，相关测试将**自动跳过**
- 测试会自动检测数据库连接可用性
- 建议在隔离环境中运行，避免影响生产数据

### 测试数据管理
- 每个测试类使用唯一的工作空间（workspace），避免冲突
- 测试结束后会自动清理测试数据
- 使用 `DROP SPACE/DATABASE` 确保彻底清理

### 性能考虑
- 完整测试套件需要较长时间（涉及大量数据导入）
- 建议使用 SSD 存储提升 I/O 性能
- 可以通过 `-k` 参数运行部分测试

## 🔧 故障排除

### 常见问题

1. **数据库连接失败**
   ```bash
   # 检查数据库服务状态
   docker ps | grep neo4j
   ```

2. **环境变量未设置**
   ```bash
   # 验证环境变量
   echo $NEO4J_HOST
   ```

3. **测试数据文件缺失**
   ```bash
   # 检查测试数据文件
   ls -la tests/integration/graphstorage/graph_storage_test_data.json
   ```

### 调试模式
```bash
# 启用详细日志
uv run pytest tests/integration/graphstorage/ -v -s --log-cli-level=DEBUG
```

## 🎉 扩展新的存储实现

要为新的图存储实现添加测试：

1. **实现存储接口**: 继承 `BaseGraphStorage`
2. **创建测试文件**: 参考 `test_neo4j_storage.py` 的结构
3. **配置 Oracle**: 使用 `GraphStorageOracle` 包装你的实现
4. **调用测试套件**: 直接使用 `GraphStorageTestSuite` 的静态方法

这种设计确保了**所有存储实现都使用相同的测试标准**，保证了 API 的一致性和可靠性。

---

**Test Oracle 模式让我们对图存储实现的正确性充满信心！** ⚖️✨ 
