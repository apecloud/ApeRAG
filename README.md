# ApeRAG

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Development](./docs/development-guide.md)
- [Build Docker Image](./docs/build-docker-image.md)
- [Acknowledgments](#acknowledgments)
- [License](#license)

ApeRAG is a production-ready RAG (Retrieval-Augmented Generation) platform that combines vector search, full-text search, and graph knowledge extraction inspired by **[LightRAG](https://github.com/HKUDS/LightRAG)**. Build sophisticated AI applications with hybrid retrieval, multimodal document processing, and enterprise-grade management features.

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

## Kubernetes Deployment

> **Recommended for Production Environment**

Deploy ApeRAG to Kubernetes using our provided Helm chart. This approach offers high availability, scalability, and production-grade management capabilities.

### Prerequisites

*   Kubernetes cluster (v1.20+)
*   `kubectl` configured and connected to your cluster
*   Helm v3+ installed

### Option 1: Quick Deployment (Recommended)

If you already have PostgreSQL, Redis, Qdrant, and Elasticsearch running in your cluster, you can deploy ApeRAG directly:

```bash
# Clone the repository
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG

# Update database connections in values.yaml for your existing databases
# Then deploy ApeRAG
helm install aperag ./deploy/aperag --namespace aperag --create-namespace
```

### Option 2: Full Deployment with KubeBlocks

If you need to deploy databases as well, use our KubeBlocks integration:

#### Step 1: Deploy Database Services

```bash
cd deploy/databases/

# (Optional) Review configuration - defaults work for most cases
# edit 00-config.sh

# Install KubeBlocks and deploy databases
bash ./01-prepare.sh          # Installs KubeBlocks
bash ./02-install-database.sh # Deploys PostgreSQL, Redis, Qdrant, Elasticsearch

# Monitor database deployment
kubectl get pods -n default
```

Wait for all database pods to be in `Running` status before proceeding.

#### Step 2: Deploy ApeRAG Application

```bash
cd ../../  # Back to project root

# Deploy ApeRAG (database connections pre-configured for KubeBlocks)
helm install aperag ./deploy/aperag --namespace default --create-namespace

# Monitor ApeRAG deployment
kubectl get pods -n default -l app.kubernetes.io/instance=aperag
```

### Configuration Options

**Database Connections**: Edit `deploy/aperag/values.yaml` to configure database connections for your environment.

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