---
title: 开发指南
description: ApeRAG 开发环境设置和工作流程
---

# 🛠️ 开发指南

本指南重点介绍如何为 ApeRAG 设置开发环境和开发工作流程。这是为希望为 ApeRAG 做贡献或在本地运行它进行开发的开发人员设计的。

## 🚀 开发环境设置

按照以下步骤从源代码设置 ApeRAG 进行开发：

### 1. 📂 克隆仓库并设置环境

首先，获取源代码并配置环境变量：

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
```

如果需要，编辑 `.env` 文件以配置您的 AI 服务设置。默认设置适用于下一步启动的本地数据库服务。

### 2. 📋 系统前提条件

在开始之前，请确保您的系统具备：

*   **Node.js**：推荐版本 20 或更高版本用于前端开发。[下载 Node.js](https://nodejs.org/)
*   **Docker & Docker Compose**：本地运行数据库服务所需。[下载 Docker](https://docs.docker.com/get-docker/)

**注意**：需要 Python 3.11，但将在下一步中通过 `uv` 自动管理。

### 3. 🗄️ 启动数据库服务

使用 Docker Compose 启动必要的数据库服务：

```bash
# 启动核心数据库：PostgreSQL、Redis、Qdrant、Elasticsearch
make infra-up
```

这将在后台启动所有必需的数据库服务。您的 `.env` 文件中的默认连接设置已预配置为与这些服务一起工作。

<details>
<summary><strong>高级数据库选项</strong></summary>

```bash
# 使用 Neo4j 而不是 PostgreSQL 进行图存储
make infra-up WITH_NEO4J=1
```

</details>

### 4. ⚙️ 设置开发环境

创建 Python 虚拟环境并设置开发工具：

```bash
make env-dev
```

此命令将：
*   如果尚未可用，则安装 `uv`
*   创建 Python 3.11 虚拟环境（位于 `.venv/` 中）
*   安装后端依赖和仓库 git hooks
*   为代码质量安装 pre-commit hooks
*   安装 addlicense 工具进行许可证管理

**激活虚拟环境：**
```bash
source .venv/bin/activate
```

当您在终端提示符中看到 `(.venv)` 时，您就知道它是活动的。

### 5. 📦 安装依赖项

安装所有后端和前端依赖项：

```bash
make env-install
```

此命令将：
*   将 `pyproject.toml` 中的所有 Python 后端依赖项安装到虚拟环境中
*   使用 `yarn` 安装前端 Node.js 依赖项

### 6. 🔄 应用数据库迁移

设置数据库架构：

```bash
make db-migrate
```

### 7. ▶️ 启动开发服务

现在您可以启动开发服务。为每个服务打开单独的终端窗口/选项卡：

**终端 1 - 后端 API 服务器：**
```bash
make serve-api
```
这将在 `http://localhost:8000` 启动 FastAPI 开发服务器，代码更改时自动重新加载。

**终端 2 - Celery Worker：**
```bash
make serve-worker
```
这将启动 Celery worker 以处理异步后台任务。

**终端 3 - 前端（可选）：**
```bash
make serve-web
```
这将在 `http://localhost:3000` 启动前端开发服务器，支持热重载。

### 8. 🌐 访问 ApeRAG

服务运行后，您可以访问：
*   **前端 UI**：http://localhost:3000 (如果已启动)
*   **后端 API**：http://localhost:8000
*   **API 文档**：http://localhost:8000/docs

### 9. ⏹️ 停止服务

要停止开发环境：

**停止数据库服务：**
```bash
# 停止数据库服务（保留数据）
make stack-down

# 停止服务并移除所有数据卷
make stack-down REMOVE_VOLUMES=1
```

**停止开发服务：**
- 后端 API 服务器：在运行 `make serve-api` 的终端中按 `Ctrl+C`
- Celery Worker：在运行 `make serve-worker` 的终端中按 `Ctrl+C`
- 前端服务器：在运行 `make serve-web` 的终端中按 `Ctrl+C`

**数据管理：**
- `make stack-down` - 停止服务但保留所有数据（PostgreSQL、Redis、Qdrant 等）
- `make stack-down REMOVE_VOLUMES=1` - 停止服务并**⚠️ 永久删除所有数据**
- 即使已经运行过 `make stack-down`，您也可以运行 `make stack-down REMOVE_VOLUMES=1`

**验证数据移除：**
```bash
# 检查卷是否仍然存在
docker volume ls | grep aperag

# REMOVE_VOLUMES=1 后应该不返回结果
```

现在您已经从源代码本地运行 ApeRAG，准备好进行开发！🎉

## ❓ 常见开发任务

### Q: 🔧 如何添加或修改 REST API 端点？

**完整工作流程：**
1. 使用 Python / Pydantic 定义 request / response model。
2. 实现后端视图：`aperag/views/[module].py`
3. 导出 code-first OpenAPI specs：
   ```bash
   make openapi-generate  # 写入 openapi.full.json 和 openapi.public.json
   ```
4. 验证导出链路：
   ```bash
   make openapi-check
   ```
5. 前端 typed client 更新统一通过 FE v2 adapter 层推进。
6. 测试 API：
   ```bash
   make test-all
   # ✅ 检查实时文档：http://localhost:8000/docs
   ```

### Q: 🗃️ 如何修改数据库模型/架构？

**数据库迁移工作流程：**
1. 编辑 `aperag/db/models.py` 中的 SQLModel 类
2. 生成迁移文件：
   ```bash
   make db-revision  # 在 migration/versions/ 中创建新迁移
   ```
3. 将迁移应用到数据库：
   ```bash
   make db-migrate  # 更新数据库架构
   ```
4. 更新相关代码（`aperag/db/repositories/` 中的仓库，`aperag/service/` 中的服务）
5. 验证更改：
   ```bash
   make test-all  # ✅ 确保一切正常工作
   ```

### Q: ⚡ 如何添加具有后台处理的新功能？

**功能实现工作流程：**
1. 实现功能组件：
   - 后端逻辑：`aperag/[module]/`
   - 异步任务：`aperag/tasks/`
   - 数据库模型：`aperag/db/models.py`
2. 更新 API 并验证 code-first contract：
   ```bash
   make db-revision      # 生成迁移文件
   make db-migrate           # 应用数据库更改
   make openapi-check         # 验证 FastAPI/Pydantic OpenAPI 导出
   ```
3. 质量保证：
   ```bash
   make format && make lint && make test-all
   ```

### Q: 🧪 如何运行单元测试和 e2e 测试？

**单元测试（快速，无外部依赖）：**
```bash
# 运行所有单元测试
make test-unit

# 运行特定测试文件
uv run pytest tests/unit_test/test_model_service.py -v

# 运行特定测试类或函数
uv run pytest tests/unit_test/test_model_service.py::TestModelService::test_get_models -v

# 运行带覆盖率的测试
make test-unit
```

**E2E 测试（需要运行服务）：**
```bash
# 设置：首先启动所需服务
make infra-up      # 🗄️ 启动数据库
make serve-api       # 🚀 启动 API 服务器（单独终端）

# 运行剩余的 pytest 版产品级 e2e
make test-e2e

# 对运行中的服务执行 HTTP 黑盒 smoke
make test-http-smoke

# 运行后端集成测试
make test-integration

# 运行特定 pytest e2e 模块
uv run pytest tests/e2e_pytest/test_chat.py -v

# 运行特定集成测试模块
uv run pytest tests/integration/graphstorage/ -v

# 运行带详细输出且不捕获的测试
uv run pytest tests/e2e_pytest/test_chat.py -v -s

# 性能基准测试（带计时）
make test-e2e-perf
```

**完整测试套件：**
```bash
# 运行所有内容（单元 + e2e）
make test-all

# 使用不同配置进行测试
make infra-up WITH_NEO4J=1  # 使用 Neo4j 而不是 PostgreSQL 进行测试
make test-all
```

### Q: 🐛 如何调试失败的测试？

**调试工作流程：**
1. 单独运行失败的测试：
   ```bash
   # 带完整输出的单个测试
   uv run pytest tests/unit_test/test_failing.py::test_specific_function -v -s
   
   # 在第一次失败时停止
   uv run pytest tests/unit_test/ -x --tb=short
   ```
2. 对于 e2e 测试失败，确保服务正在运行：
   ```bash
   make infra-up       # 数据库服务
   make serve-api         # API 服务器
   make serve-worker         # 后台 workers（如果测试异步任务）
   ```
3. 使用调试工具：
   ```bash
   # 使用 pdb 调试器运行
   uv run pytest tests/unit_test/test_failing.py --pdb
   
   # 在测试期间捕获日志
   uv run pytest tests/e2e_pytest/test_chat.py --log-cli-level=DEBUG
   ```
4. 修复并重新测试：
   ```bash
   make format              # 自动修复样式问题
   make lint               # 检查剩余问题
   uv run pytest tests/path/to/fixed_test.py -v  # 验证修复
   ```

### Q: 📦 如何安全地更新依赖项？

**Python 依赖项：**
1. 编辑 `pyproject.toml`（添加/更新包）
2. 更新虚拟环境：
   ```bash
   make env-install            # 使用 uv 同步所有组和额外内容
   make test-all              # 验证兼容性
   ```

**前端依赖项：**
1. 编辑 `frontend/package.json`
2. 更新并测试：
   ```bash
   cd frontend && yarn install
   make serve-web      # 测试前端编译
   ```

### Q: 🚀 如何准备代码进行生产部署？

**部署前检查清单：**
1. 代码质量验证：
   ```bash
   make format            # 自动修复所有样式问题
   make lint             # 验证无样式违规
   make static-check     # MyPy 类型检查
   ```
2. 全面测试：
   ```bash
   make test-all             # 所有单元 + e2e 测试
   make test-e2e-perf  # 性能基准测试
   ```
3. API 一致性：
   ```bash
   make openapi-check       # 确保 code-first OpenAPI 可以导出
   ```
4. 数据库迁移：
   ```bash
   make db-revision    # 生成任何待处理的迁移
   ```
5. 全栈集成测试：
   ```bash
   make stack-up WITH_NEO4J=1 WITH_DOCRAY=1  # 类似生产的设置
   # 在 http://localhost:3000/web/ 手动测试
   make stack-down
   ```

### Q: 🔄 如何完全重置我的开发环境？

**核选项重置（销毁所有数据）：**
```bash
make stack-down REMOVE_VOLUMES=1  # ⚠️ 停止服务 + 删除所有数据
make env-clean                         # 🧹 清理临时文件

# 重新开始
make infra-up                 # 🗄️ 新的数据库
make db-migrate                      # 🔄 应用所有迁移
make serve-api                  # 🚀 启动 API 服务器
make serve-worker                   # ⚡ 启动后台 workers
```

**软重置（保留数据）：**
```bash
make stack-down                 # ⏹️ 停止服务，保留数据
make infra-up               # 🗄️ 重启数据库
make db-migrate                    # 🔄 应用任何新迁移
```

**仅重置 Python 环境：**
```bash
rm -rf .venv/                   # 🗑️ 移除虚拟环境
make env-dev                       # ⚙️ 重新创建所有内容
source .venv/bin/activate      # ✅ 重新激活
``` 
