# ApeRAG

[阅读中文文档](./README_zh.md)

ApeRAG is a comprehensive RAG (Retrieval-Augmented Generation) platform designed for building advanced, enterprise-grade AI applications. It integrates **LightRAG** to enhance its capabilities, notably enabling powerful **graph-based query and knowledge retrieval**. Key features include:

*   **Versatile Document Processing**: Efficiently parses various document types.
*   **Flexible Data Management**: Utilizes Django backend with Celery for asynchronous tasks, supporting databases like PostgreSQL, Qdrant, Neo4j, and Elasticsearch.
*   **Dynamic Frontend**: Modern user interface built with React and TypeScript (UmiJS).
*   **Advanced RAG Pipelines**: Supports embedding generation, hybrid search (vector, keyword, and graph), and sophisticated workflow automation for complex RAG scenarios.
*   **LLM Integration**: Seamlessly connects with Large Language Models.

## Table of Contents

- [Getting Started](#getting-started)
  - [Getting Started with Kubernetes](#getting-started-with-kubernetes)
  - [Getting Started with Source Code](#getting-started-with-source-code)
  - [Getting Started with Docker Compose](#getting-started-with-docker-compose)
- [Development](#development)
  - [Development Environment](#development-environment)
  - [Key `make` Commands for Development](#key-make-commands-for-development)
  - [Typical Development Workflow](#typical-development-workflow)
- [Build Docker Image](#build-docker-image)
  - [Building Container Images](#building-container-images)
  - [Deployment](#deployment)
- [Project Structure Overview](#project-structure-overview)
- [License](#license)

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

*   **Django Development Server**:
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

## Development

This section focuses on the development workflow and tools provided for ApeRAG.

### Development Environment

It's recommended to use the "Getting Started with Source Code" approach for setting up a development environment. Ensure all prerequisites are met and dependencies are installed using `make install`.

### Key `make` Commands for Development

The `Makefile` at the root of the project provides several helpful commands to streamline development:

*   **Environment & Dependencies**:
    *   `make install`: Installs all necessary backend (Python) and frontend (Node.js) dependencies. It sets up a Python 3.11 virtual environment using `uv`.
    *   `make dev`: Installs development tools like pre-commit hooks to ensure code quality before commits.

*   **Running Services**:
    *   `make run-db`: (Uses Docker Compose) Starts all required database services (PostgreSQL, Redis, Qdrant, etc.) as defined in `docker-compose.yml`. Useful if you don't have these services running elsewhere.
    *   `make run-backend`: Starts the Django development server.
    *   `make run-frontend`: Starts the UmiJS frontend development server.
    *   `make run-celery`: Starts a Celery worker for processing background tasks (includes Celery Beat).
    *   `make run-celery-beat`: (Note: `make run-celery` usually includes Beat due to the `-B` flag. This target might be redundant or for specific scenarios. Check Makefile if explicitly needed separate from worker).

*   **Code Quality & Testing**:
    *   `make format`: Formats Python code using Ruff and frontend code using Prettier.
    *   `make lint`: Lints Python code with Ruff and frontend code.
    *   `make static-check`: Performs static type checking for Python code using Mypy (if configured).
    *   `make test`: Runs all automated tests (Python unit tests, integration tests).

*   **Database Management**:
    *   `make makemigration`: Creates new database migration files based on changes to Django models.
    *   `make migrate`: Applies pending database migrations to your connected database.
    *   `make connect-metadb`: Provides a command to connect to the primary PostgreSQL database (usually for inspection, if run via `make run-db`).
    *   `make diff`: Shows differences in Django settings (useful for debugging configurations).

*   **Generators**:
    *   `make generate-models`: Generates Pydantic models from the OpenAPI schema.
    *   `make generate-frontend-sdk`: Generates the frontend API client/SDK from the OpenAPI specification. **Run this command whenever backend API definitions change.**

*   **Docker Compose (for local full-stack testing)**:
    *   `make compose-up`: Starts all services (backend, frontend, databases, Celery) using Docker Compose.
    *   `make compose-down`: Stops all services started with `make compose-up`.
    *   `make compose-logs`: Tails the logs from all services running under Docker Compose.

*   **Cleanup**:
    *   `make clean`: Removes temporary files, build artifacts, and caches from the development environment.

### Typical Development Workflow

Contributing to ApeRAG involves the following typical workflow. Before starting significant work, it's a good idea to open an issue to discuss your proposed changes with the maintainers.

1.  **Fork and Branch**:
    *   Fork the official ApeRAG repository to your GitHub account.
    *   Create a new branch from `main` for your feature or bug fix. Use a descriptive branch name (e.g., `feat/add-new-parser` or `fix/login-bug`).

2.  **Environment Setup**: Ensure your development environment is set up as described in [Development Environment](#development-environment) and [Getting Started with Source Code](#getting-started-with-source-code) (dependencies installed, databases running/accessible).

3.  **Code Implementation**:
    *   Make your code changes in the backend (`aperag/`) or frontend (`frontend/src/`) directories.
    *   **Follow Code Style**: Adhere to PEP 8 for Python and standard practices for TypeScript/React. Use English for all code, comments, and documentation.
    *   Regularly use `make format` and `make lint` to ensure code consistency and quality.

4.  **Handle API and Model Changes**:
    *   If you change backend API endpoints (add, remove, modify parameters/responses): Update the OpenAPI specification (usually in `aperag/api/openapi.yaml`) and then run `make generate-frontend-sdk` to update the frontend client. Also, run `make generate-models` if schema components are affected.
    *   If you change Django models: Run `make makemigration` to create migration files, and then `make migrate` to apply changes to your development database.

5.  **Testing**: Add unit tests for new backend logic and integration tests for API changes. Ensure all existing tests pass by running `make test`.

6.  **Documentation**: If your changes affect API specifications, user guides, or deployment processes, update the relevant documentation (e.g., OpenAPI specs, this README, files in `docs/`).

7.  **Commit and Push**:
    *   Make clear and concise commit messages.
    *   Push your branch to your fork on GitHub.

8.  **Submit a Pull Request (PR)**:
    *   Submit a PR from your branch to the `main` branch of the official ApeRAG repository.
    *   Provide a clear description of your changes in the PR and link any relevant issues.

9.  **Code Review**: Your PR will be reviewed by maintainers. Be prepared to address feedback and make further changes if necessary.

## Build Docker Image

This section covers how to build ApeRAG container images. It's primarily for users who need to create their own builds or deploy to environments other than the ones covered in "Getting Started".

### Building Container Images

The project uses Docker and `make` commands to build container images.

*   **Local Platform Builds**:
    These commands build images for your current machine's architecture.
    ```bash
    # Build all necessary images for local platform
    make build-local

    # Build only the backend image for local platform
    make build-aperag-local

    # Build only the frontend image for local platform
    make build-aperag-frontend-local
    ```

*   **Multi-platform Builds**:
    These commands build images for multiple architectures (e.g., amd64, arm64). This requires Docker Buildx to be set up and configured.
    ```bash
    # Build all necessary images for multiple platforms
    make build

    # Build only the backend image for multiple platforms
    make build-aperag

    # Build only the frontend image for multiple platforms
    make build-aperag-frontend
    ```
    You can specify the target platforms using the `PLATFORMS` variable, for example:
    ```bash
    make build PLATFORMS=linux/amd64,linux/arm64
    ```

### Deployment

Refer to the "Getting Started" section for common deployment methods:
*   [Getting Started with Kubernetes](#getting-started-with-kubernetes)
*   [Getting Started with Docker Compose](#getting-started-with-docker-compose)

For custom deployments, you will need to adapt these methods or use the built container images with your chosen orchestration platform. Ensure all required services (databases, backend, frontend, Celery workers) are correctly configured and can communicate with each other.


## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.