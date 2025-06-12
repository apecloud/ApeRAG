# ApeRAG

ApeRAG 是一个全面的 RAG（Retrieval-Augmented Generation，检索增强生成）平台，专为构建先进的企业级人工智能应用而设计。它集成了 **[LightRAG](https://github.com/HKUDS/LightRAG)** 以增强其功能，特别是实现了强大的**基于图的查询和知识检索**。主要特性包括：

*   **多样化的文档处理**：高效解析各种文档类型。
*   **灵活的数据管理**：后端采用 FastAPI 和 Celery 进行异步任务处理，支持 PostgreSQL、Qdrant、Neo4j 和 Elasticsearch 等多种数据库。
*   **动态前端**：基于 React 和 TypeScript (UmiJS) 构建的现代化用户界面。
*   **先进的 RAG 工作流**：支持 Embedding 生成、混合搜索（向量、关键词和图搜索）以及针对复杂 RAG 场景的精密工作流自动化。
*   **大语言模型集成**：与大语言模型无缝连接。

## 目录

- [快速上手](#快速上手)
  - [通过 Kubernetes 快速上手](#通过-kubernetes-快速上手)
  - [通过源码快速上手](#通过源码快速上手)
  - [通过 Docker Compose 快速上手](#通过-docker-compose-快速上手)
- [开发](#开发)
  - [开发环境](#开发环境)
  - [主要的 `make` 开发命令](#主要的-make-开发命令)
  - [典型开发工作流](#典型开发工作流)
- [构建 Docker 镜像](#构建-docker-镜像)
  - [构建容器镜像](#构建容器镜像)
  - [部署](#部署)
- [项目结构概览](#项目结构概览)
- [许可证](#许可证)

## 快速上手

本节将指导您使用不同方法搭建 ApeRAG。

### 通过 Kubernetes 快速上手

本指南介绍了如何使用提供的 Helm chart 将 ApeRAG 部署到 Kubernetes。主要包括两个阶段：设置数据库（如果您已有数据库则可选）和部署 ApeRAG 应用。

**阶段一：使用 KubeBlocks 部署数据库 (可选)**

ApeRAG 需要 PostgreSQL、Redis、Qdrant 和 Elasticsearch。如果您没有这些数据库，可以使用 `deploy/databases/` 目录下的 KubeBlocks 脚本进行部署。

*如果您的 Kubernetes 集群中已有这些数据库，请跳过此阶段。*

1.  **先决条件**:
    *   Kubernetes 集群。
    *   已配置 `kubectl`。
    *   Helm v3+。

2.  **数据库配置 (`deploy/databases/00-config.sh`)**:
    此脚本控制数据库部署（默认：在 `default` 命名空间中部署 PostgreSQL、Redis、Qdrant、Elasticsearch）。**通常默认配置即可，标准设置无需更改。** 仅在高级情况下（例如，更改命名空间、启用 Neo4j 等可选数据库）才需编辑。

3.  **运行数据库部署脚本**:
    ```bash
    cd deploy/databases/
    bash ./01-prepare.sh          # 准备 KubeBlocks 环境。
    bash ./02-install-database.sh # 部署数据库集群。
    cd ../.. # 返回项目根目录。
    ```
    监控 `default` 命名空间（或您自定义的命名空间）中的 Pods，直到它们变为 Ready 状态：
    ```bash
    kubectl get pods -n default
    ```

**阶段二：部署 ApeRAG 应用**

数据库运行后：

1.  **Helm Chart 配置 (`deploy/aperag/values.yaml`)**:
    *   **使用 KubeBlocks (阶段一，在 `default` 命名空间)？** `values.yaml` 中的数据库连接信息可能已预先配置好。**通常无需更改。**
    *   **使用您自己的数据库？** 您必须更新 `values.yaml` 文件，填入您的数据库连接详细信息。
    *   默认情况下，此 Helm chart 会部署用于文档解析的 [`doc-ray`](https://github.com/apecloud/doc-ray) 服务，该服务至少需要 4 个 CPU 核心和 8GB 内存。如果您的 Kubernetes 集群资源不足，可以通过将 `docray.enabled` 设置为 `false` 来禁用 `doc-ray` 的部署。禁用后，系统将使用一个基础的文档解析器。
    *   （可选）检查其他设置（如镜像、资源、Ingress 等）。

2.  **使用 Helm 部署 ApeRAG**:
    这将把 ApeRAG 安装到 `default` 命名空间：
    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    ```
    监控 ApeRAG Pods 直到状态变为 `Running`：
    ```bash
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```

3.  **访问 ApeRAG UI**:
    使用 `kubectl port-forward` 进行快速访问：
    ```bash
    kubectl port-forward deploy/aperag-frontend 3000:3000 -n default
    ```
    在浏览器中打开 `http://localhost:3000`。

关于 KubeBlocks 的详细信息（凭证、卸载），请参阅 `deploy/databases/README.md`。

### 通过源码快速上手

本指南适用于希望为 ApeRAG 贡献代码或在本地运行以进行开发的开发人员。请按照以下步骤从源代码运行 ApeRAG：

**1. 克隆仓库**

首先，获取源代码：
```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
```

**2. 系统先决条件**

开始之前，请确保您的系统已安装：

*   **Python 3.11**: 项目使用 Python 3.11。如果它不是您系统的默认版本，`uv`（见下文）在创建虚拟环境时会尝试使用它（如果可用）。
*   **Node.js**: 推荐使用 20 或更高版本进行前端开发。
*   **`uv`**: 这是一个快速的 Python 包安装器和虚拟环境管理器。
    *   如果您没有 `uv`，`make install` 命令（步骤3）会尝试通过 `pip` 安装它。
*   **Docker**: （推荐用于本地数据库）如果您计划在本地运行 PostgreSQL、Redis 等依赖服务，Docker 是最简单的方法。`make run-db` 命令使用 Docker Compose。

**3. 安装依赖并设置虚拟环境**

这个关键的 `make` 命令会自动执行多项设置任务：

```bash
make install
```

该命令将：
*   验证或安装 `uv`。
*   使用 `uv` 创建一个 Python 3.11 虚拟环境（位于 `.venv/`）。
*   从 `pyproject.toml` 将所有 Python 后端依赖（包括开发工具）安装到虚拟环境中。
*   使用 `yarn` 安装前端 Node.js 依赖。

**4. 配置环境变量**

ApeRAG 使用 `.env` 文件进行配置。

*   **后端 (`.env`)**: 复制模板并根据您的设置进行自定义。
    ```bash
    cp envs/env.template .env
    ```
    然后，编辑新创建的 `.env` 文件。

    **注意**：如果您使用 `make run-db` 命令（参见步骤5）启动所需的数据库服务，则从 `envs/env.template` 复制的 `.env` 文件中的默认连接设置已预先配置为可与这些服务配合使用，通常无需更改。只有当您连接到外部管理的数据库或具有自定义配置时，才需要修改这些设置。

*   **前端 (`frontend/.env`)** (可选 - 如果您正在开发前端)：
    ```bash
    cp frontend/deploy/env.local.template frontend/.env
    ```
    如果您需要更改前端特定设置（例如后端 API URL），请编辑 `frontend/.env`（尽管默认设置通常适用于本地开发）。

**5. 启动数据库并应用迁移**

*   **启动数据库服务**:
    如果您使用 Docker 进行本地数据库部署，`Makefile` 提供了一个方便的命令：
    ```bash
    make run-db
    ```

*   **应用数据库迁移**:
    一旦您的数据库正在运行并在 `.env` 中配置完毕，请设置数据库模式：
    ```bash
    make migrate
    ```

**6. 运行 ApeRAG 后端服务**

这些服务通常应在单独的终端窗口/标签页中运行。`make` 命令将自动使用正确的 Python 虚拟环境。

*   **FastAPI 开发服务器**:
    ```bash
    make run-backend
    ```
    这将启动主后端应用程序。它通常可在 `http://localhost:8000` 访问，并具有代码更改时自动重新加载的功能。

*   **Celery Worker & Beat**:
    ```bash
    make run-celery
    ```
    这将启动 Celery worker 以处理异步后台任务。

**7. 运行前端开发服务器 (可选)**

如果您需要处理或查看前端：
```bash
make run-frontend
```
这将启动前端开发服务器，通常可在 `http://localhost:3000` 访问。它配置为将 API 请求代理到在端口 8000 上运行的后端。

**8. 访问 ApeRAG**

在后端（以及可选的前端）服务运行后：
*   在 `http://localhost:3000` 访问**前端 UI**（如果已启动）。
*   **后端 API** 可在 `http://localhost:8000` 访问。

现在您已在本地从源代码运行 ApeRAG，可以进行开发或测试了！

### 通过 Docker Compose 快速上手

要使用 Docker Compose 快速上手 ApeRAG，请按照以下步骤操作：

1.  **先决条件**:
    *   Docker & Docker Compose
    *   Git

2.  **环境设置**:
    通过复制模板文件来配置环境变量：
    ```bash
    cp envs/env.template .env
    cp frontend/deploy/env.local.template frontend/.env
    ```
    然后，**编辑 `.env` 文件**，根据您的需求配置 AI 服务设置和其他必要的配置。

3.  **启动服务**:
    您可以使用以下 `make` 命令启动所有 ApeRAG 服务：
    ```bash
    # 可选：如果在中国，请使用阿里云镜像仓库
    # export REGISTRY=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com

    # 启动 ApeRAG 服务
    make compose-up
    ```
    如果您需要使用 `doc-ray` 服务进行高级文档解析（推荐用于复杂文档、表格或公式），可以将其与其他服务一起启动：
    ```bash
    make compose-up WITH_DOCRAY=1
    ```
    如果您的环境有 GPU，可以为 `doc-ray` 启用 GPU 支持以获得更好的性能：
    ```bash
    make compose-up WITH_DOCRAY=1 WITH_GPU=1
    ```
    > **关于 doc-ray 解析服务**
    >
    > ApeRAG 包含一个基本的内置解析器，用于从 PDF 和 DOCX 等文档中提取文本以进行 RAG 索引。但是，此解析器可能无法最佳处理复杂的文档结构、表格或公式。
    >
    > 为了增强文档解析能力并获得更准确的内容提取，我们建议部署 [doc-ray](https://github.com/apecloud/doc-ray) 服务。`doc-ray` 利用 **MinerU** 进行高级文档分析。
    >
    > *   未指定 `WITH_GPU=1` 时，`doc-ray` 将仅使用 CPU 运行。在这种情况下，建议为其分配至少 4 个 CPU 核心和 8GB+ 的内存。
    > *   指定 `WITH_GPU=1` 时，`doc-ray` 将使用 GPU 运行。它大约需要 6GB 的显存，以及 2 个 CPU 核心和 8GB 的内存。

4.  **访问 ApeRAG**:
    服务启动并运行后，打开浏览器并导航至：http://localhost:3000/web/

## 开发

本节重点介绍 ApeRAG 的开发工作流和提供的工具。

### 开发环境

建议使用"通过源码快速上手"的方法来设置开发环境。确保已满足所有先决条件并使用 `make install` 安装了依赖项。

### 主要的 `make` 开发命令

项目根目录下的 `Makefile` 文件提供了几个有用的命令来简化开发：

*   **环境与依赖**:
    *   `make install`: 安装所有必需的后端 (Python) 和前端 (Node.js) 依赖。它使用 `uv` 设置 Python 3.11 虚拟环境。
    *   `make dev`: 安装诸如 pre-commit hooks 之类的开发工具，以确保提交前的代码质量。

*   **运行服务**:
    *   `make run-db`: (使用 Docker Compose) 启动 `docker-compose.yml` 中定义的所有必需数据库服务（PostgreSQL、Redis、Qdrant 等）。如果您没有在其他地方运行这些服务，则此命令非常有用。
    *   `make run-backend`: 启动 FastAPI 开发服务器。
    *   `make run-frontend`: 启动 UmiJS 前端开发服务器。
    *   `make run-celery`: 启动 Celery worker 以处理后台任务（包括 Celery Beat）。
    *   `make run-celery-beat`: (注意: `make run-celery` 通常由于 `-B` 标志而包含 Beat。此目标可能是多余的，或用于特定场景。如果明确需要与 worker 分开，请检查 Makefile)。

*   **代码质量与测试**:
    *   `make format`: 使用 Ruff 格式化 Python 代码，使用 Prettier 格式化前端代码。
    *   `make lint`: 使用 Ruff 对 Python 代码进行 Lint 检查，并对前端代码进行 Lint 检查。
    *   `make static-check`: 使用 Mypy 对 Python 代码进行静态类型检查（如果已配置）。
    *   `make test`: 运行所有自动化测试（Python 单元测试、集成测试）。

*   **数据库管理**:
    *   `make makemigration`: 根据 SQLAlchemy模型的更改创建新的数据库迁移文件。
    *   `make migrate`: 将待处理的数据库迁移应用到您连接的数据库。
    *   `make connect-metadb`: 提供连接到主 PostgreSQL 数据库的命令（通常用于检查，如果通过 `make run-db` 运行）。

*   **生成器**:
    *   `make generate-models`: 从 OpenAPI schema 生成 Pydantic 模型。
    *   `make generate-frontend-sdk`: 从 OpenAPI 规范生成前端 API 客户端/SDK。**每当后端 API 定义发生更改时，请运行此命令。**

*   **Docker Compose (用于本地全栈测试)**:
    *   `make compose-up`: 使用 Docker Compose 启动所有服务（后端、前端、数据库、Celery）。
    *   `make compose-down`: 停止使用 `make compose-up` 启动的所有服务。
    *   `make compose-logs`: 查看 Docker Compose 下运行的所有服务的日志。

*   **清理**:
    *   `make clean`: 从开发环境中删除临时文件、构建产物和缓存。

### 典型开发工作流

为 ApeRAG 做贡献涉及以下典型工作流程。在开始重要工作之前，最好先创建一个 issue 与维护者讨论您提议的更改。

1.  **Fork 和创建分支**:
    *   将官方 ApeRAG 仓库 Fork到您的 GitHub 帐户。
    *   从 `main` 分支为您的功能或错误修复创建一个新分支。使用描述性的分支名称（例如，`feat/add-new-parser` 或 `fix/login-bug`）。

2.  **环境设置**: 确保您的开发环境已按照 [开发环境](#开发环境) 和 [通过源码快速上手](#通过源码快速上手) 中的说明进行设置（已安装依赖项，数据库正在运行/可访问）。

3.  **代码实现**:
    *   在后端 (`aperag/`) 或前端 (`frontend/src/`) 目录中进行代码更改。
    *   **遵循代码风格**: Python 代码遵循 PEP 8，TypeScript/React 代码遵循标准实践。所有代码、注释和文档均使用英文。
    *   定期使用 `make format` 和 `make lint` 来确保代码的一致性和质量。

4.  **处理 API 和模型更改**:
    *   如果您更改了后端 API 端点（添加、删除、修改参数/响应）：更新 OpenAPI 规范（通常在 `aperag/api/openapi.yaml` 中），然后运行 `make generate-frontend-sdk` 以更新前端客户端。如果 schema 组件受到影响，还需运行 `make generate-models`。
    *   如果您更改了 SQLAlchemy模型：运行 `make makemigration` 创建迁移文件，然后运行 `make migrate` 将更改应用到您的开发数据库。

5.  **测试**: 为新的后端逻辑添加单元测试，为 API 更改添加集成测试。通过运行 `make test` 确保所有现有测试通过。

6.  **文档**: 如果您的更改影响 API 规范、用户指南或部署过程，请更新相关文档（例如，OpenAPI 规范、此 README、`docs/` 中的文件）。

7.  **提交和推送**:
    *   编写清晰简洁的提交消息。
    *   将您的分支推送到您在 GitHub 上的 Fork。

8.  **提交拉取请求 (PR)**:
    *   从您的分支向官方 ApeRAG 仓库的 `main` 分支提交 PR。
    *   在 PR 中提供更改的清晰描述，并链接任何相关 issue。

9.  **代码审查**: 您的 PR 将由维护者审查。准备好处理反馈并在必要时进行进一步更改。

## 构建 Docker 镜像

本节介绍如何构建 ApeRAG 容器镜像。主要适用于需要创建自己的构建或部署到"快速上手"部分未涵盖的环境的用户。

### 构建容器镜像

项目使用 Docker 和 `make` 命令来构建容器镜像。

*   **本地平台构建**:
    这些命令为您的当前计算机架构构建镜像。
    ```bash
    # 为本地平台构建所有必需的镜像
    make build-local

    # 仅为本地平台构建后端镜像
    make build-aperag-local

    # 仅为本地平台构建前端镜像
    make build-aperag-frontend-local
    ```

*   **多平台构建**:
    这些命令为多种架构（例如 amd64、arm64）构建镜像。这需要设置和配置 Docker Buildx。
    ```bash
    # 为多个平台构建所有必需的镜像
    make build

    # 仅为多个平台构建后端镜像
    make build-aperag

    # 仅为多个平台构建前端镜像
    make build-aperag-frontend
    ```
    您可以使用 `PLATFORMS` 变量指定目标平台，例如：
    ```bash
    make build PLATFORMS=linux/amd64,linux/arm64
    ```

### 部署

有关常见的部署方法，请参阅"快速上手"部分：
*   [通过 Kubernetes 快速上手](#通过-kubernetes-快速上手)
*   [通过 Docker Compose 快速上手](#通过-docker-compose-快速上手)

对于自定义部署，您需要调整这些方法或将构建的容器镜像与您选择的编排平台一起使用。确保所有必需的服务（数据库、后端、前端、Celery worker）都已正确配置并且可以相互通信。


## 致谢

ApeRAG 集成并基于多个优秀的开源项目构建：

### LightRAG
ApeRAG 中的基于图的知识检索功能由深度修改版的 [LightRAG](https://github.com/HKUDS/LightRAG) 提供支持：
- **论文**："LightRAG: Simple and Fast Retrieval-Augmented Generation"（[arXiv:2410.05779](https://arxiv.org/abs/2410.05779)）
- **作者**：Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- **许可证**：MIT License

我们对 LightRAG 进行了大量修改，以支持生产级并发处理、分布式任务队列（Celery/Prefect）和无状态操作。详细信息请参阅我们的 [LightRAG 修改日志](./aperag/graph/changelog.md)。

## 许可证

ApeRAG 根据 Apache License 2.0 获得许可。有关详细信息，请参阅 [LICENSE](./LICENSE) 文件。 