# Nebula Test Spaces Cleanup

该脚本用于清理Nebula Graph数据库中所有以"test"开头的space（数据库）。

## 前置条件

1. **安装依赖**：
   ```bash
   pip install nebula3-python
   ```

2. **配置环境变量**：
   确保在`.env`文件中配置了Nebula连接信息：
   ```bash
   NEBULA_HOST=127.0.0.1
   NEBULA_PORT=9669
   NEBULA_USER=root
   NEBULA_PASSWORD=nebula
   ```

## 使用方法

### 方法1：直接运行脚本
```bash
# 在项目根目录下运行
python scripts/cleanup_test_nebula_spaces.py
```

### 方法2：使用可执行权限
```bash
# 确保脚本有可执行权限
chmod +x scripts/cleanup_test_nebula_spaces.py

# 直接执行
./scripts/cleanup_test_nebula_spaces.py
```

### 方法3：通过uv运行（推荐）
```bash
# 激活uv环境并运行
uv run python scripts/cleanup_test_nebula_spaces.py
```

## 脚本功能

1. **连接检查**：自动检查Nebula连接配置
2. **空间列举**：获取所有现有的spaces
3. **过滤测试空间**：筛选出以"test"开头的spaces
4. **确认删除**：交互式确认是否删除
5. **批量删除**：安全删除所有匹配的test spaces
6. **结果汇总**：显示删除操作的统计结果

## 安全特性

- ✅ 只删除以"test"开头的spaces
- ✅ 执行前需要用户确认
- ✅ 详细的日志记录
- ✅ 错误处理和回滚
- ✅ 连接资源自动清理

## 示例输出

```
2025-01-15 10:30:00 - INFO - 🚀 Starting Nebula test space cleanup...
2025-01-15 10:30:00 - INFO - Connecting to Nebula at 127.0.0.1:9669
2025-01-15 10:30:01 - INFO - Found 5 spaces: ['test_collection_1', 'test_collection_2', 'production_space', 'test_graph', 'main']
2025-01-15 10:30:01 - INFO - 🎯 Found 3 test spaces to delete: ['test_collection_1', 'test_collection_2', 'test_graph']
❓ Do you want to delete 3 test spaces? (yes/no): yes
2025-01-15 10:30:05 - INFO - Dropping space: test_collection_1
2025-01-15 10:30:05 - INFO - ✅ Successfully dropped space: test_collection_1
2025-01-15 10:30:05 - INFO - Dropping space: test_collection_2
2025-01-15 10:30:06 - INFO - ✅ Successfully dropped space: test_collection_2
2025-01-15 10:30:06 - INFO - Dropping space: test_graph
2025-01-15 10:30:06 - INFO - ✅ Successfully dropped space: test_graph
==================================================
📊 CLEANUP SUMMARY
✅ Successfully deleted: 3 spaces
❌ Failed to delete: 0 spaces
📊 Total processed: 3 spaces
🎉 All test spaces cleaned up successfully!
```

## 故障排除

### 连接失败
```bash
# 检查Nebula服务是否运行
nebula-console -addr 127.0.0.1 -port 9669 -u root -p nebula

# 检查环境变量
echo $NEBULA_HOST $NEBULA_PORT $NEBULA_USER $NEBULA_PASSWORD
```

### 权限问题
```bash
# 确保用户有删除space的权限
# 通常root用户有所有权限
```

### 依赖问题
```bash
# 重新安装nebula3-python
pip uninstall nebula3-python
pip install nebula3-python
``` 