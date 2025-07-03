# Development Guide

This guide focuses on setting up a development environment and the development workflow for ApeRAG. This is designed for developers looking to contribute to ApeRAG or run it locally for development purposes.

## Development Environment Setup

Follow these steps to set up ApeRAG from source code for development:

### 1. Clone the Repository

First, get the source code:
```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
```

### 2. System Prerequisites

Before you begin, ensure your system has:

*   **Python 3.11**: The project uses Python 3.11. If it's not your system default, `uv` (see below) will attempt to use it when creating the virtual environment if available.
*   **Node.js**: Version 20 or higher is recommended for frontend development.
*   **`uv`**: This is a fast Python package installer and virtual environment manager.
    *   If you don't have `uv`, the `make install` command (Step 3) will try to install it via `pip`.
*   **Docker**: (Recommended for local databases) If you plan to run dependent services like PostgreSQL, Redis, etc., locally, Docker is the easiest way. The `make run-db` command uses Docker Compose.

### 3. Install Dependencies & Setup Virtual Environment

This crucial `make` command automates several setup tasks:

```bash
make install
```

This command will:
*   Verify or install `uv`.
*   Create a Python 3.11 virtual environment (located in `.venv/`) using `uv`.
*   Install all Python backend dependencies (including development tools) from `pyproject.toml` into the virtual environment.
*   Install frontend Node.js dependencies using `yarn`.

### 4. Configure Environment Variables

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

### 5. Start Databases & Apply Migrations

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

### 6. Run ApeRAG Backend Services

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

### 7. Run Frontend Development Server (Optional)

If you need to work on or view the frontend:
```bash
make run-frontend
```
This will start the frontend development server, usually available at `http://localhost:3000`. It's configured to proxy API requests to the backend running on port 8000.

### 8. Access ApeRAG

With the backend (and optionally frontend) services running:
*   Access the **Frontend UI** at `http://localhost:3000` (if started).
*   The **Backend API** is available at `http://localhost:8000`.

Now you have ApeRAG running locally from the source code, ready for development or testing!

## Docker Compose Development Options

For developers who prefer containerized development, Docker Compose offers flexible deployment modes:

### Infrastructure Mode (Recommended for Development)

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

### Full Application Mode

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

### Optional Services

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

### Service Management

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

### Example Development Workflows

**For Quick Testing**:
```bash
make compose-up
# Access http://localhost:3000/web/ and start exploring!
```

**For Active Development**:
```bash
make compose-infra WITH_NEO4J=1  # Start databases
make run-backend                 # Run API in development mode
# Code with hot reload and debugging
```

**For Production Testing**:
```bash
make compose-up WITH_NEO4J=1 WITH_DOCRAY=1 WITH_GPU=1
# Full-featured deployment with all capabilities
```

## Key `make` Commands for Development

The `Makefile` at the root of the project provides several helpful commands to streamline development:

*   **Environment & Dependencies**:
    *   `make install`: Installs all necessary backend (Python) and frontend (Node.js) dependencies. It sets up a Python 3.11 virtual environment using `uv`.
    *   `make dev`: Installs development tools like pre-commit hooks to ensure code quality before commits.

*   **Running Services**:
    *   `make run-db`: (Uses Docker Compose) Starts all required database services (PostgreSQL, Redis, Qdrant, etc.) as defined in `docker-compose.yml`. Useful if you don't have these services running elsewhere.
    *   `make run-backend`: Starts the FastAPI development server.
    *   `make run-frontend`: Starts the UmiJS frontend development server.
    *   `make run-celery`: Starts a Celery worker for processing background tasks (includes Celery Beat).
    *   `make run-celery-beat`: (Note: `make run-celery` usually includes Beat due to the `-B` flag. This target might be redundant or for specific scenarios. Check Makefile if explicitly needed separate from worker).

*   **Code Quality & Testing**:
    *   `make format`: Formats Python code using Ruff and frontend code using Prettier.
    *   `make lint`: Lints Python code with Ruff and frontend code.
    *   `make static-check`: Performs static type checking for Python code using Mypy (if configured).
    *   `make test`: Runs all automated tests (Python unit tests, integration tests).

*   **Database Management**:
    *   `make makemigration`: Creates new database migration files based on changes to SQLAlchemy models.
    *   `make migrate`: Applies pending database migrations to your connected database.
    *   `make connect-metadb`: Provides a command to connect to the primary PostgreSQL database (usually for inspection, if run via `make run-db`).

*   **Generators**:
    *   `make generate-models`: Generates Pydantic models from the OpenAPI schema.
    *   `make generate-frontend-sdk`: Generates the frontend API client/SDK from the OpenAPI specification. **Run this command whenever backend API definitions change.**

*   **Docker Compose (for local full-stack testing)**:
    *   `make compose-up`: Starts all services (backend, frontend, databases, Celery) using Docker Compose.
    *   `make compose-down`: Stops all services started with `make compose-up`.
    *   `make compose-logs`: Tails the logs from all services running under Docker Compose.

*   **Cleanup**:
    *   `make clean`: Removes temporary files, build artifacts, and caches from the development environment.

## Typical Development Workflow

Contributing to ApeRAG involves the following typical workflow. Before starting significant work, it's a good idea to open an issue to discuss your proposed changes with the maintainers.

1.  **Fork and Branch**:
    *   Fork the official ApeRAG repository to your GitHub account.
    *   Create a new branch from `main` for your feature or bug fix. Use a descriptive branch name (e.g., `feat/add-new-parser` or `fix/login-bug`).

2.  **Environment Setup**: Ensure your development environment is set up as described in [Development Environment Setup](#development-environment-setup) (dependencies installed, databases running/accessible).

3.  **Code Implementation**:
    *   Make your code changes in the backend (`aperag/`) or frontend (`frontend/src/`) directories.
    *   **Follow Code Style**: Adhere to PEP 8 for Python and standard practices for TypeScript/React. Use English for all code, comments, and documentation.
    *   Regularly use `make format` and `make lint` to ensure code consistency and quality.

4.  **Handle API and Model Changes**:
    *   If you change backend API endpoints (add, remove, modify parameters/responses): Update the OpenAPI specification (usually in `aperag/api/openapi.yaml`) and then run `make generate-frontend-sdk` to update the frontend client. Also, run `make generate-models` if schema components are affected.
    *   If you change SQLAlchemy models: Run `make makemigration` to create migration files, and then `make migrate` to apply changes to your development database.

5.  **Testing**: Add unit tests for new backend logic and integration tests for API changes. Ensure all existing tests pass by running `make test`.

6.  **Documentation**: If your changes affect API specifications, user guides, or deployment processes, update the relevant documentation (e.g., OpenAPI specs, this README, files in `docs/`).

7.  **Commit and Push**:
    *   Make clear and concise commit messages.
    *   Push your branch to your fork on GitHub.

8.  **Submit a Pull Request (PR)**:
    *   Submit a PR from your branch to the `main` branch of the official ApeRAG repository.
    *   Provide a clear description of your changes in the PR and link any relevant issues.

9.  **Code Review**: Your PR will be reviewed by maintainers. Be prepared to address feedback and make further changes if necessary. 