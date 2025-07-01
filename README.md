# ApeRAG

[阅读中文文档](./README_zh.md)

![collection-page.png](docs%2Fimages%2Fcollection-page.png)

## Table of Contents

- [Getting Started](#getting-started)
  - [Getting Started with Kubernetes (Recommend for Production)](#getting-started-with-kubernetes)
  - [Getting Started with Source Code](#getting-started-with-source-code)
  - [Getting Started with Docker Compose](#getting-started-with-docker-compose)
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

### Getting Started with Kubernetes

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

### Getting Started with Source Code

This guide is for developers looking to contribute to ApeRAG or run it locally for development. Follow these steps to get ApeRAG running from the source code:

**1. Clone the Repository**

First, get the source code:
```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
```

**2. System Prerequisites**

Before you begin, ensure your system has:

*   **Python 3.11**: The project uses Python 3.11. If it's not your system default, `uv` (see below) will attempt to use it when creating the virtual environment if available.
*   **Node.js**: Version 20 or higher is recommended for frontend development.
*   **`uv`**: This is a fast Python package installer and virtual environment manager.
    *   If you don't have `uv`, the `make install` command (Step 3) will try to install it via `pip`.
*   **Docker**: (Recommended for local databases) If you plan to run dependent services like PostgreSQL, Redis, etc., locally, Docker is the easiest way. The `make run-db` command uses Docker Compose.

**3. Install Dependencies & Setup Virtual Environment**

This crucial `make` command automates several setup tasks:

```bash
make install
```

This command will:
*   Verify or install `uv`.
*   Create a Python 3.11 virtual environment (located in `.venv/`) using `uv`.
*   Install all Python backend dependencies (including development tools) from `pyproject.toml` into the virtual environment.
*   Install frontend Node.js dependencies using `yarn`.

**4. Configure Environment Variables**

ApeRAG uses `.env` files for configuration.

*   **Backend (`.env`)**: Copy the template and customize it for your setup.
    ```bash
    cp envs/env.template .env
    ```
    Then, edit the newly created `.env` file.

    **Note**: If you start the required database services using the `make run-db` command (see Step 5), the default connection settings in the `.env` file (copied from `envs/env.template`) are pre-configured to work with these services, and you typically won't need to change them. You would only need to modify these if you are connecting to externally managed databases or have custom configurations.

*   **Frontend (`frontend/.env`)** (Optional - if you are developing the frontend):
    ```bash
    cp frontend/deploy/env.local.template frontend/.env
    ```
    Edit `frontend/.env` if you need to change frontend-specific settings, such as the backend API URL (though defaults usually work for local development).

**5. Start Databases & Apply Migrations**

*   **Start Database Services**:
    If you're using Docker for local databases, the `Makefile` provides a convenient command:
    ```bash
    make run-db
    ```

*   **Apply Database Migrations**:
    Once your databases are running and configured in `.env`, set up the database schema:
    ```bash
    make migrate
    ```

**6. Run ApeRAG Backend Services**

These should typically be run in separate terminal windows/tabs. The `make` commands will automatically use the correct Python virtual environment.

*   **FastAPI Development Server**:
    ```bash
    make run-backend
    ```
    This starts the main backend application. It will typically be accessible at `http://localhost:8000` and features auto-reload on code changes.

*   **Celery Worker & Beat**:
    ```bash
    make run-celery
    ```
    This starts the Celery worker for processing asynchronous background tasks.

**7. Run Frontend Development Server (Optional)**

If you need to work on or view the frontend:
```bash
make run-frontend
```
This will start the frontend development server, usually available at `http://localhost:3000`. It's configured to proxy API requests to the backend running on port 8000.

**8. Access ApeRAG**

With the backend (and optionally frontend) services running:
*   Access the **Frontend UI** at `http://localhost:3000` (if started).
*   The **Backend API** is available at `http://localhost:8000`.

Now you have ApeRAG running locally from the source code, ready for development or testing!

For detailed development workflows, see the [Development Guide](./docs/DEVELOPMENT.md).

### Getting Started with Docker Compose

To get started with ApeRAG using Docker Compose, follow these steps:

1.  **Prerequisites**:
    *   Docker & Docker Compose
    *   Git

2.  **Environment Setup**:
    Configure environment variables by copying the template files:
    ```bash
    cp envs/env.template .env
    cp frontend/deploy/env.local.template frontend/.env
    ```
    Then, **edit the `.env` file** to configure your AI service settings and other necessary configurations according to your needs.

3.  **Start Services**:
    You can start all ApeRAG services using the following `make` command:
    ```bash
    # Optional: Use Aliyun registry if in China
    # export REGISTRY=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com

    # Start ApeRAG services
    make compose-up
    ```
    If you need to use the `doc-ray` service for advanced document parsing (recommended for complex documents, tables, or formulas), you can start it along with other services:
    ```bash
    make compose-up WITH_DOCRAY=1
    ```
    If your environment has GPUs, you can enable GPU support for `doc-ray` for better performance:
    ```bash
    make compose-up WITH_DOCRAY=1 WITH_GPU=1
    ```
    > **About the doc-ray parsing service**
    >
    > ApeRAG includes a basic built-in parser for extracting text from documents like PDFs and DOCX files for RAG indexing. However, this parser may not optimally handle complex document structures, tables, or formulas.
    >
    > For enhanced document parsing capabilities and more accurate content extraction, we recommend deploying the [doc-ray](https://github.com/apecloud/doc-ray) service. `doc-ray` leverages **MinerU** for advanced document analysis.
    >
    > * When `WITH_GPU=1` is not specified, `doc-ray` will run using only the CPU. In this case, it is recommended to allocate at least 4 CPU cores and 8GB+ of RAM for it.
    > * When `WITH_GPU=1` is specified, `doc-ray` will run using the GPU. It requires approximately 6GB of VRAM, along with 2 CPU cores and 8GB of RAM.

4.  **Access ApeRAG**:
    Once the services are up and running, open your browser and navigate to: http://localhost:3000/web/

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