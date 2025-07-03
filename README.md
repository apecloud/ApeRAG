# ApeRAG

[阅读中文文档](./README_zh.md)

![collection-page.png](docs%2Fimages%2Fcollection-page.png)

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Advanced Setup](#advanced-setup)
  - [Getting Started with Kubernetes (Recommend for Production)](#getting-started-with-kubernetes-recommend-for-production)
- [Development](./docs/development-guide.md)
- [Build Docker Image](./docs/build-docker-image.md)
- [Acknowledgments](#acknowledgments)
- [License](#license)

ApeRAG is a production-ready RAG (Retrieval-Augmented Generation) platform that combines vector search, full-text search, and graph knowledge extraction powered by **[LightRAG](https://github.com/HKUDS/LightRAG)**. Build sophisticated AI applications with hybrid retrieval, multimodal document processing, and enterprise-grade management features.

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
make compose-up
```

After running, you can access ApeRAG in your browser at:
- **Web Interface**: http://localhost:3000/web/
- **API Documentation**: http://localhost:8000/docs

#### Seeking help

Please refer to our [Development Guide](./docs/development-guide.md) for advanced configurations, development setup, and troubleshooting. Reach out to [the community](#acknowledgments) if you are still having issues.

> If you'd like to contribute to ApeRAG or do additional development, refer to our [Development Guide](./docs/development-guide.md)

## Key Features

**1. Hybrid Retrieval Engine**:
Combines vector search, full-text search, and graph knowledge extraction for comprehensive document understanding and retrieval.

**2. LightRAG Integration**:
Enhanced version of LightRAG for advanced graph-based knowledge extraction, enabling deep relational and contextual queries.

**3. Multimodal Document Processing**:
Supports various document formats (PDF, DOCX, etc.) with advanced parsing capabilities for text, tables, and images.

**4. Enterprise Management**:
Built-in audit logging, LLM model management, graph visualization, and comprehensive document management interface.

**5. Production Ready**:
Kubernetes support, Docker deployment, async task processing with Celery, and comprehensive API documentation.

**6. Developer Friendly**:
FastAPI backend, React frontend, extensive testing, and detailed development guides for easy contribution and customization.

## Advanced Setup

### Getting Started with Kubernetes (Recommend for Production)

This guide covers deploying ApeRAG to Kubernetes using the provided Helm chart. It involves two main phases: setting up databases (optional if you have them) and deploying the ApeRAG application.

**Phase 1: Deploy Databases with KubeBlocks (Optional)**

ApeRAG needs PostgreSQL, Redis, Qdrant, and Elasticsearch. If you don't have these, use the KubeBlocks scripts in `deploy/databases/`.

*Skip this phase if your databases are already available in your Kubernetes cluster.*

1.  **Prerequisites**:
    *   Kubernetes cluster.
    *   `kubectl` configured.
    *   Helm v3+.

2.  **Database Configuration (`deploy/databases/00-config.sh`)**:
    This script controls database deployment (defaults: PostgreSQL, Redis, Qdrant, Elasticsearch in the `default` namespace). **Defaults are usually fine; no changes needed for a standard setup.** Edit only for advanced cases (e.g., changing namespace, enabling optional databases like Neo4j).

3.  **Run Database Deployment Scripts**:
    ```bash
    cd deploy/databases/
    bash ./01-prepare.sh          # Prepares KubeBlocks environment.
    bash ./02-install-database.sh # Deploys database clusters.
    cd ../.. # Back to project root.
    ```
    Monitor pods in the `default` namespace (or your custom one) until ready:
    ```bash
    kubectl get pods -n default
    ```

**Phase 2: Deploy ApeRAG Application**

With databases running:

1.  **Helm Chart Configuration (`deploy/aperag/values.yaml`)**:
    *   **Using KubeBlocks (Phase 1 in `default` namespace)?** Database connections in `values.yaml` are likely pre-configured. **No changes usually needed.**
    *   **Using your own databases?** You MUST update `values.yaml` with your database connection details.
    *   By default, this Helm chart deploys the [`doc-ray`](https://github.com/apecloud/doc-ray) service for advanced document parsing, which requires at least 4 CPU cores and 8GB of memory. If your Kubernetes cluster has insufficient resources, you can disable the `doc-ray` deployment by setting `docray.enabled` to `false`. In this case, a basic document parser will be used.
    *   Optionally, review other settings (images, resources, Ingress, etc.).

2.  **Deploy ApeRAG with Helm**:
    This installs ApeRAG to the `default` namespace:
    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    ```
    Monitor ApeRAG pods until `Running`:
    ```bash
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```

3.  **Access ApeRAG UI**:
    Use `kubectl port-forward` for quick access:
    ```bash
    kubectl port-forward svc/aperag-frontend 3000:3000 -n default
    ```
    Open `http://localhost:3000` in your browser.

For KubeBlocks details (credentials, uninstall), see `deploy/databases/README.md`.

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