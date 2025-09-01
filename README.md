# ApeRAG
[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

![HarryPotterKG.png](docs%2Fimages%2FHarryPotterKG.png)

ApeRAG is a production-ready RAG (Retrieval-Augmented Generation) platform that combines Graph RAG, vector search, and full-text search with advanced AI agents. Build sophisticated AI applications with hybrid retrieval, multimodal document processing, intelligent agents, and enterprise-grade management features.

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the full platform capabilities with our hosted demo

ApeRAG is the best choice for building your own Knowledge Graph, Context Engineering, and deploying intelligent AI agents that can autonomously search and reason across your knowledge base.

[阅读中文文档](README-zh.md)

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Kubernetes Deployment (Recommended for Production)](#kubernetes-deployment-recommended-for-production)
- [Development](./docs/development-guide.md)
- [Build Docker Image](./docs/build-docker-image.md)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Quick Start

> Before installing ApeRAG, make sure your machine meets the following minimum system requirements:
>
> - CPU >= 2 Core
> - RAM >= 4 GiB
> - Docker & Docker Compose

The easiest way to start ApeRAG is through Docker Compose. Before running the following commands, make sure that [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) are installed on your machine:

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
cp frontend/deploy/env.local.template frontend/.env
docker-compose up -d --pull always
```

After running, you can access ApeRAG in your browser at:
- **Web Interface**: http://localhost:3000/web/
- **API Documentation**: http://localhost:8000/docs

#### MCP (Model Context Protocol) Support

ApeRAG supports [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) integration, allowing AI assistants to interact with your knowledge base directly. After starting the services, configure your MCP client with:

```json
{
  "mcpServers": {
    "aperag-mcp": {
      "url": "https://rag.apecloud.com/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

**Important**: Replace `http://localhost:8000` with your actual ApeRAG API URL and `your-api-key-here` with a valid API key from your ApeRAG settings.

The MCP server provides:
- **Collection browsing**: List and explore your knowledge collections
- **Hybrid search**: Search using vector, full-text, and graph methods
- **Intelligent querying**: Ask natural language questions about your documents

#### Enhanced Document Parsing

For enhanced document parsing capabilities, ApeRAG supports an **advanced document parsing service** powered by MinerU, which provides superior parsing for complex documents, tables, and formulas. 

<details>
<summary><strong>Enhanced Document Parsing Commands</strong></summary>

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration 
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

Or use the Makefile shortcuts (requires [GNU Make](https://www.gnu.org/software/make/)):
```bash
# Enable advanced document parsing service
make compose-up WITH_DOCRAY=1

# Enable advanced parsing with GPU acceleration (recommended)
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

</details>

#### Development & Contributing

For developers interested in source code development, advanced configurations, or contributing to ApeRAG, please refer to our [Development Guide](./docs/development-guide.md) for detailed setup instructions.

## Key Features

**1. Advanced Index Types**:
Five comprehensive index types for optimal retrieval: **Vector**, **Full-text**, **Graph**, **Summary**, and **Vision** - providing multi-dimensional document understanding and search capabilities.

**2. Intelligent AI Agents**:
Built-in AI agents with MCP (Model Context Protocol) tool support that can automatically identify relevant collections, search content intelligently, and provide web search capabilities for comprehensive question answering.

**3. Enhanced Graph RAG with Entity Normalization**:
Deeply modified LightRAG implementation with advanced entity normalization (entity merging) for cleaner knowledge graphs and improved relational understanding.

**4. Multimodal Processing & Vision Support**:
Complete multimodal document processing including vision capabilities for images, charts, and visual content analysis alongside traditional text processing.

**5. Hybrid Retrieval Engine**:
Sophisticated retrieval system combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive document understanding.

**6. MinerU Integration**:
Advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.

**7. Production-Grade Deployment**:
Full Kubernetes support with Helm charts and KubeBlocks integration for simplified deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).

**8. Enterprise Management**:
Built-in audit logging, LLM model management, graph visualization, comprehensive document management interface, and agent workflow management.

**9. MCP Integration**:
Full support for Model Context Protocol (MCP), enabling seamless integration with AI assistants and tools for direct knowledge base access and intelligent querying.

**10. Developer Friendly**:
FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and agent development framework for easy contribution and customization.

## Kubernetes Deployment (Recommended for Production)

> **Enterprise-grade deployment with high availability and scalability**

Deploy ApeRAG to Kubernetes using our provided Helm chart. This approach offers high availability, scalability, and production-grade management capabilities.

### Prerequisites

*   [Kubernetes cluster](https://kubernetes.io/docs/setup/) (v1.20+)
*   [`kubectl`](https://kubernetes.io/docs/tasks/tools/) configured and connected to your cluster
*   [Helm v3+](https://helm.sh/docs/intro/install/) installed

### Clone the Repository

First, clone the ApeRAG repository to get the deployment files:

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
```

### Step 1: Deploy Database Services

ApeRAG requires PostgreSQL, Redis, Qdrant, and Elasticsearch. You have two options:

**Option A: Use existing databases** - If you already have these databases running in your cluster, edit `deploy/aperag/values.yaml` to configure your database connection details, then skip to Step 2.

**Option B: Deploy databases with KubeBlocks** - Use our automated database deployment (database connections are pre-configured):

```bash
# Navigate to database deployment scripts
cd deploy/databases/

# (Optional) Review configuration - defaults work for most cases
# edit 00-config.sh

# Install KubeBlocks and deploy databases
bash ./01-prepare.sh          # Installs KubeBlocks
bash ./02-install-database.sh # Deploys PostgreSQL, Redis, Qdrant, Elasticsearch

# Monitor database deployment
kubectl get pods -n default

# Return to project root for Step 2
cd ../../
```

Wait for all database pods to be in `Running` status before proceeding.

### Step 2: Deploy ApeRAG Application

```bash
# If you deployed databases with KubeBlocks in Step 1, database connections are pre-configured
# If you're using existing databases, edit deploy/aperag/values.yaml with your connection details

# Deploy ApeRAG
helm install aperag ./deploy/aperag --namespace default --create-namespace

# Monitor ApeRAG deployment
kubectl get pods -n default -l app.kubernetes.io/instance=aperag
```

### Configuration Options

**Resource Requirements**: By default, includes [`doc-ray`](https://github.com/apecloud/doc-ray) service (requires 4+ CPU cores, 8GB+ RAM). To disable: set `docray.enabled: false` in `values.yaml`.

**Advanced Settings**: Review `values.yaml` for additional configuration options including images, resources, and Ingress settings.

### Access Your Deployment

Once deployed, access ApeRAG using port forwarding:

```bash
# Forward ports for quick access
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default

# Access in browser
# Web Interface: http://localhost:3000
# API Documentation: http://localhost:8000/docs
```

For production environments, configure Ingress in `values.yaml` for external access.

### Troubleshooting

**Database Issues**: See `deploy/databases/README.md` for KubeBlocks management, credentials, and uninstall procedures.

**Pod Status**: Check pod logs for any deployment issues:
```bash
kubectl logs -f deployment/aperag-api -n default
kubectl logs -f deployment/aperag-frontend -n default
```

## Acknowledgments

ApeRAG integrates and builds upon several excellent open-source projects:

### LightRAG
The graph-based knowledge retrieval capabilities in ApeRAG are powered by a deeply modified version of [LightRAG](https://github.com/HKUDS/LightRAG):
- **Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
- **Authors**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- **License**: MIT License

We have extensively modified LightRAG to support production-grade concurrent processing, distributed task queues (Celery/Prefect), and stateless operations. See our [LightRAG modifications changelog](./aperag/graph/changelog.md) for details.

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.