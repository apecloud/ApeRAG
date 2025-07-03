# ApeRAG

[阅读中文文档](./README_zh.md)

![collection-page.png](docs%2Fimages%2Fcollection-page.png)

## Table of Contents

- [Getting Started](#getting-started)
  - [Getting Started with Docker Compose](#getting-started-with-docker-compose)
  - [Getting Started with Kubernetes (Recommend for Production)](#getting-started-with-kubernetes)
- [Development](./docs/development-guide.md)
- [Build Docker Image](./docs/build-docker-image.md)
- [Acknowledgments](#acknowledgments)
- [License](#license)

ApeRAG is a production-ready, comprehensive RAG (Retrieval-Augmented Generation) platform designed for building advanced, enterprise-grade AI applications. It empowers developers to create sophisticated **Agentic RAG** systems with a powerful, hybrid retrieval engine.

Key features include:

*   **Advanced Hybrid Retrieval**: Go beyond simple vector search. ApeRAG integrates three powerful indexing strategies:
    *   **Vector Index**: For semantic similarity search.
    *   **Full-Text Index**: For precise keyword-based retrieval.
    *   **Graph Knowledge Index**: Powered by an integrated and enhanced version of **[LightRAG](https://github.com/HKUDS/LightRAG)**, enabling deep relational and contextual queries.

*   **Multimodal Document Processing**: Ingest and understand a wide array of document formats, extracting not just text but also tables, images, and complex structures from files like PDFs and DOCX.

*   **Enterprise-Grade Management**: ApeRAG is built for production environments with a suite of essential features:
    *   **Audit Logging**: Track all critical system and user activities.
    *   **LLM Model Management**: Easily configure and switch between various Large Language Models.
    *   **Graph Visualization**: Visually explore and understand the knowledge graph.
    *   **Comprehensive Document Management**: A user-friendly interface to manage document collections, track processing status, and inspect content.

## Getting Started

This section will guide you through setting up ApeRAG using different methods.

### Getting Started with Docker Compose

Docker Compose provides the fastest way to get ApeRAG running. We support two deployment modes and flexible service combinations to meet different needs.

#### Prerequisites
*   Docker & Docker Compose
*   Git

#### Environment Setup
Configure environment variables by copying the template files:
```bash
cp envs/env.template .env
cp frontend/deploy/env.local.template frontend/.env
```
Then, **edit the `.env` file** to configure your AI service settings and other configurations according to your needs.

#### Deployment Modes

**🏗️ Infrastructure Mode (Recommended for Development)**

Start only the essential database services. Perfect for developers who want to run the application code locally for debugging and development.

```bash
# Core databases: PostgreSQL, Redis, Qdrant, Elasticsearch
make compose-infra

# Add graph database capabilities  
make compose-infra WITH_NEO4J=1
```

After starting infrastructure, run your app locally:
```bash
make run-backend   # Start API server at localhost:8000
make run-frontend  # Start frontend at localhost:3000 (optional)
```

**🚀 Full Application Mode (Production Ready)**

Launch the complete ApeRAG platform with all services containerized.

```bash
# Complete system (API + Frontend + Background workers + Databases)
make compose-up

# Add graph knowledge capabilities with Neo4j
make compose-up WITH_NEO4J=1

# Add advanced document parsing with DocRay
make compose-up WITH_DOCRAY=1

# Combine multiple optional services
make compose-up WITH_NEO4J=1 WITH_DOCRAY=1

# Full-featured deployment with GPU acceleration
make compose-up WITH_NEO4J=1 WITH_DOCRAY=1 WITH_GPU=1
```

#### Optional Services

**Neo4j Graph Database** (`WITH_NEO4J=1`)
- Enables graph-based knowledge extraction and querying
- Powers advanced relational search capabilities
- Accessible at http://localhost:7474 (Web UI)

**DocRay Advanced Document Parsing** (`WITH_DOCRAY=1`)
- Enhanced parsing for complex documents, tables, and formulas
- Powered by [MinerU](https://github.com/opendatalab/MinerU) technology
- CPU mode: Requires 4+ CPU cores, 8GB+ RAM
- GPU mode (`WITH_GPU=1`): Requires ~6GB VRAM, 2 CPU cores, 8GB RAM
- Service endpoint: http://localhost:8639

#### Access Your Deployment

Once services are running:
- **Web Interface**: http://localhost:3000/web/
- **API Documentation**: http://localhost:8000/docs
- **Flower (Task Monitor)**: http://localhost:5555
- **Neo4j Browser**: http://localhost:7474 (if enabled)

#### Service Management

```bash
# View running services
docker-compose ps

# View logs
make compose-logs

# Stop all services
make compose-down

# Stop services and remove data volumes
make compose-down REMOVE_VOLUMES=1
```

#### Example Workflows

**For Open Source Users (Quick Start)**:
```bash
make compose-up
# Access http://localhost:3000/web/ and start exploring!
```

**For Developers**:
```bash
make compose-infra WITH_NEO4J=1  # Start databases
make run-backend                 # Run API in development mode
# Code with hot reload and debugging
```

**For Production Deployment**:
```bash
make compose-up WITH_NEO4J=1 WITH_DOCRAY=1 WITH_GPU=1
# Full-featured deployment with all capabilities
```

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