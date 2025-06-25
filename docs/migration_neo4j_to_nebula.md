# 从 Neo4j 迁移到 NebulaGraph 指南

## 背景

由于 Neo4j 社区版不支持 `CREATE DATABASE` 命令，这对 ApeRAG 的多租户需求造成了限制。NebulaGraph 提供了更好的多租户支持，通过 `SPACE` 概念可以很好地隔离不同的 workspace 数据。

## 主要改动

### 1. 新增文件

#### `aperag/db/nebula_sync_manager.py`
- Worker 级别的 NebulaGraph 连接管理器
- 使用同步驱动避免事件循环问题
- 提供连接池复用功能
- 自动创建和管理 SPACE（对应 workspace）

#### `aperag/graph/lightrag/kg/nebula_sync_impl.py`
- 实现 `BaseGraphStorage` 接口
- 将所有 Cypher 查询翻译成 nGQL
- 支持 LightRAG 所需的所有图操作

### 2. 配置更新

#### 环境变量 (`envs/env.template`)
```bash
# NebulaGraph
NEBULA_HOST=127.0.0.1
NEBULA_PORT=9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula
NEBULA_MAX_CONNECTION_POOL_SIZE=10
NEBULA_TIMEOUT=60000

# 切换到 NebulaGraph
LIGHTRAG_GRAPH_STORAGE=NebulaSyncStorage
```

#### Celery 配置 (`config/celery.py`)
- 添加了 NebulaGraph 的 worker 生命周期信号处理器
- 确保连接池在 worker 启动时初始化，关闭时清理

#### Storage 注册 (`aperag/graph/lightrag/kg/__init__.py`)
- 注册 `NebulaSyncStorage` 为可用的图存储实现
- 添加环境变量要求检查

### 3. Schema 设计

NebulaGraph 是强 Schema 的数据库，需要预先定义 TAG 和 EDGE TYPE：

#### TAGs（节点类型）
- `base`: 所有节点的基础 TAG，包含 `entity_id` 和 `entity_type`
- `entity`: 实体节点，包含额外的属性如 `entity_name`、`description` 等

#### EDGE TYPEs（边类型）
- `DIRECTED`: 有向边，包含 `weight`、`source_id`、`description`、`keywords` 等属性

#### 索引
- `base_entity_id_idx`: 在 `base.entity_id` 上的索引，加速查询
- `entity_name_idx`: 在 `entity.entity_name` 上的索引

### 4. 查询翻译对照

| 操作 | Neo4j (Cypher) | NebulaGraph (nGQL) |
|------|----------------|-------------------|
| 创建节点 | `MERGE (n:base {entity_id: $id})` | `UPSERT VERTEX ON base "id" SET ...` |
| 创建边 | `MERGE (a)-[r:DIRECTED]-(b)` | `UPSERT EDGE ON DIRECTED "a" -> "b" SET ...` |
| 查询节点 | `MATCH (n:base {entity_id: $id})` | `FETCH PROP ON base "id"` |
| 查询边 | `MATCH (a)-[r]-(b)` | `FETCH PROP ON DIRECTED "a" -> "b"` |
| 删除节点 | `DETACH DELETE n` | `DELETE VERTEX "id" WITH EDGE` |
| 子图遍历 | `MATCH path = (n)-[*..3]-(m)` | `GO 3 STEPS FROM "id" OVER * BIDIRECT` |

### 5. 迁移步骤

1. **安装 NebulaGraph Python 客户端**
   ```bash
   pip install nebula3-python
   ```

2. **启动 NebulaGraph 服务**
   ```bash
   docker-compose -f nebula-docker-compose.yml up -d
   ```

3. **更新环境变量**
   ```bash
   # 编辑 .env 文件
   LIGHTRAG_GRAPH_STORAGE=NebulaSyncStorage
   NEBULA_HOST=localhost
   NEBULA_PORT=9669
   NEBULA_USER=root
   NEBULA_PASSWORD=nebula
   ```

4. **重启 Celery Worker**
   ```bash
   make run-celery
   ```

### 6. 注意事项

1. **VID 类型**: NebulaGraph 使用 `FIXED_STRING(256)` 作为顶点 ID 类型，足够存储实体 ID
2. **字符串转义**: 需要特别注意转义引号和反斜杠
3. **强 Schema**: 必须先创建 TAG 和 EDGE TYPE 才能插入数据
4. **双向边**: LightRAG 将边视为无向的，但 NebulaGraph 的边是有向的，需要在查询时处理
5. **索引**: 创建索引后需要等待几秒钟才能生效

### 7. 性能优化建议

1. **连接池大小**: 根据并发需求调整 `NEBULA_MAX_CONNECTION_POOL_SIZE`
2. **批量操作**: 尽可能使用批量 UPSERT 操作
3. **索引使用**: 确保常用查询字段都有索引
4. **SPACE 分区**: 对于大数据量，可以调整 `partition_num` 和 `replica_factor`

### 8. 故障排查

1. **连接失败**: 检查 NebulaGraph 服务是否启动，端口是否正确
2. **Schema 错误**: 确保 SPACE 和 Schema 已正确创建
3. **查询错误**: 检查 nGQL 语法，特别是引号使用
4. **性能问题**: 使用 `PROFILE` 或 `EXPLAIN` 分析查询计划

## 总结

这个迁移方案保持了与现有系统的兼容性，只需要修改环境变量即可切换图数据库后端。NebulaGraph 的多租户支持和高性能特性使其成为 Neo4j 社区版的理想替代品。 