# ApeRAG

![HarryPotterKG.png](docs%2Fimages%2FHarryPotterKG.png)

ApeRAG 是一个生产级 RAG（检索增强生成）平台，结合了图 RAG、向量搜索和全文搜索。构建具有混合检索、多模态文档处理和企业级管理功能的复杂 AI 应用程序。

ApeRAG是你构建自己的Knowledge Graph、做Context Engineering的最好选择。

[Read English Documentation](README.md)

- [快速开始](#快速开始)
- [核心特性](#核心特性)
- [Kubernetes 部署（推荐生产环境）](#kubernetes-部署推荐生产环境)
- [开发指南](./docs/development-guide-zh.md)
- [构建 Docker 镜像](./docs/build-docker-image-zh.md)
- [致谢](#致谢)
- [许可证](#许可证)

## 快速开始

> 在安装 ApeRAG 之前，请确保您的机器满足以下最低系统要求：
>
> - CPU >= 2 核心
> - RAM >= 4 GiB
> - Docker & Docker Compose

启动 ApeRAG 最简单的方法是通过 Docker Compose。在运行以下命令之前，请确保您的机器上已安装 [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)：

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
cp frontend/deploy/env.local.template frontend/.env
docker-compose up -d --pull always
```

运行后，您可以在浏览器中访问 ApeRAG：
- **Web 界面**: http://localhost:3000/web/
- **API 文档**: http://localhost:8000/docs

#### MCP（模型上下文协议）支持

ApeRAG 支持 [MCP（模型上下文协议）](https://modelcontextprotocol.io/) 集成，允许 AI 助手直接与您的知识库交互。启动服务后，使用以下配置您的 MCP 客户端：

```json
{
  "mcpServers": {
    "aperag-mcp": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

**重要提示**：将 `http://localhost:8000` 替换为您实际的 ApeRAG API 地址，将 `your-api-key-here` 替换为 ApeRAG 设置中的有效 API 密钥。

MCP 服务器提供：
- **集合浏览**：列出和探索您的知识集合
- **混合搜索**：使用向量、全文和图搜索方法
- **智能查询**：对您的文档提出自然语言问题

#### 增强文档解析

为了获得增强的文档解析能力，ApeRAG 支持由 MinerU 驱动的**高级文档解析服务**，可为复杂文档、表格和公式提供优异的解析能力。

<details>
<summary><strong>增强文档解析命令</strong></summary>

```bash
# 启用高级文档解析服务
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# 启用带 GPU 加速的高级解析
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

或使用 Makefile 快捷方式（需要 [GNU Make](https://www.gnu.org/software/make/)）：
```bash
# 启用高级文档解析服务
make compose-up WITH_DOCRAY=1

# 启用带 GPU 加速的高级解析（推荐）
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

</details>

#### 开发与贡献

对于有兴趣进行源代码开发、高级配置或为 ApeRAG 做贡献的开发人员，请参考我们的[开发指南](./docs/development-guide-zh.md)获取详细的设置说明。

## 核心特性

**1. 混合检索引擎**：
结合图 RAG、向量搜索和全文搜索，实现全面的文档理解和检索。

**2. 基于 LightRAG 的图 RAG**：
增强版的 LightRAG，用于高级图形知识提取，支持深度关系和上下文查询。

**3. MinerU 集成**：
由 MinerU 技术驱动的高级文档解析服务，为复杂文档、表格、公式和科学内容提供优异的解析能力，可选 GPU 加速。

**4. 生产级部署**：
完整的 Kubernetes 支持，配有 Helm charts 和 KubeBlocks 集成，简化生产级数据库（PostgreSQL、Redis、Qdrant、Elasticsearch、Neo4j）的部署。

**5. 多模态文档处理**：
支持各种文档格式（PDF、DOCX 等），具有智能内容提取和结构识别功能。

**6. 企业管理**：
内置审计日志、LLM 模型管理、图形可视化和全面的文档管理界面。

**7. MCP 集成**：
完全支持模型上下文协议（MCP），实现与 AI 助手和工具的无缝集成，支持直接访问知识库和智能查询。

**8. 开发者友好**：
FastAPI 后端、React 前端、使用 Celery 的异步任务处理、广泛的测试以及全面的开发指南，便于贡献和定制。

## Kubernetes 部署（推荐生产环境）

> **具有高可用性和可扩展性的企业级部署**

使用我们提供的 Helm chart 将 ApeRAG 部署到 Kubernetes。这种方法提供高可用性、可扩展性和生产级管理能力。

### 前提条件

*   [Kubernetes 集群](https://kubernetes.io/docs/setup/)（v1.20+）
*   [`kubectl`](https://kubernetes.io/docs/tasks/tools/) 已配置并连接到您的集群
*   [Helm v3+](https://helm.sh/docs/intro/install/) 已安装

### 克隆仓库

首先，克隆 ApeRAG 仓库以获取部署文件：

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
```

### 步骤 1：部署数据库服务

ApeRAG 需要 PostgreSQL、Redis、Qdrant 和 Elasticsearch。您有两个选择：

**选项 A：使用现有数据库** - 如果您的集群中已经运行这些数据库，请编辑 `deploy/aperag/values.yaml` 配置您的数据库连接详情，然后跳到步骤 2。

**选项 B：使用 KubeBlocks 部署数据库** - 使用我们的自动化数据库部署（数据库连接已预配置）：

```bash
# 进入数据库部署脚本目录
cd deploy/databases/

# （可选）查看配置 - 默认设置适用于大多数情况
# edit 00-config.sh

# 安装 KubeBlocks 并部署数据库
bash ./01-prepare.sh          # 安装 KubeBlocks
bash ./02-install-database.sh # 部署 PostgreSQL、Redis、Qdrant、Elasticsearch

# 监控数据库部署
kubectl get pods -n default

# 返回项目根目录进行步骤 2
cd ../../
```

等待所有数据库 pod 状态变为 `Running` 后再继续。

### 步骤 2：部署 ApeRAG 应用

```bash
# 如果您在步骤 1 中使用 KubeBlocks 部署了数据库，数据库连接已预配置
# 如果您使用现有数据库，请使用您的连接详情编辑 deploy/aperag/values.yaml

# 部署 ApeRAG
helm install aperag ./deploy/aperag --namespace default --create-namespace

# 监控 ApeRAG 部署
kubectl get pods -n default -l app.kubernetes.io/instance=aperag
```

### 配置选项

**资源要求**：默认包含 [`doc-ray`](https://github.com/apecloud/doc-ray) 服务（需要 4+ CPU 核心，8GB+ RAM）。要禁用：在 `values.yaml` 中设置 `docray.enabled: false`。

**高级设置**：查看 `values.yaml` 了解额外的配置选项，包括镜像、资源和 Ingress 设置。

### 访问您的部署

部署完成后，使用端口转发访问 ApeRAG：

```bash
# 转发端口以便快速访问
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default

# 在浏览器中访问
# Web 界面: http://localhost:3000
# API 文档: http://localhost:8000/docs
```

对于生产环境，请在 `values.yaml` 中配置 Ingress 以获得外部访问。

### 故障排除

**数据库问题**：查看 `deploy/databases/README.md` 了解 KubeBlocks 管理、凭据和卸载程序。

**Pod 状态**：检查 pod 日志以查看任何部署问题：
```bash
kubectl logs -f deployment/aperag-api -n default
kubectl logs -f deployment/aperag-frontend -n default
```

## 致谢

ApeRAG 集成并构建在几个优秀的开源项目之上：

### LightRAG
ApeRAG 中基于图的知识检索能力由深度修改的 [LightRAG](https://github.com/HKUDS/LightRAG) 版本提供支持：
- **论文**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
- **作者**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- **许可证**: MIT License

我们对 LightRAG 进行了广泛修改，以支持生产级并发处理、分布式任务队列（Celery/Prefect）和无状态操作。详情请参见我们的 [LightRAG 修改更新日志](./aperag/graph/changelog.md)。

## 许可证

ApeRAG 采用 Apache License 2.0 许可。详情请参见 [LICENSE](./LICENSE) 文件。 