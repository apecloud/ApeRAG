# Development Guide

This guide focuses on setting up a development environment and the development workflow for ApeRAG. This is designed for developers looking to contribute to ApeRAG or run it locally for development purposes.

## Development Environment Setup

Follow these steps to set up ApeRAG from source code for development:

### 1. Clone the Repository and Setup Environment

First, get the source code and configure environment variables:

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
cp frontend/deploy/env.local.template frontend/.env
```

Edit the `.env` file to configure your AI service settings if needed. The default settings work with the local database services started in the next step.

### 2. System Prerequisites

Before you begin, ensure your system has:

*   **Node.js**: Version 20 or higher is recommended for frontend development.
*   **Docker & Docker Compose**: Required for running database services locally.

**Note**: Python 3.11 is required but will be automatically managed by `uv` in the next steps.

### 3. Start Database Services

Use Docker Compose to start the essential database services:

```bash
# Start core databases: PostgreSQL, Redis, Qdrant, Elasticsearch
make compose-infra

# Optional: Use Neo4j instead of PostgreSQL for graph storage
# make compose-infra WITH_NEO4J=1
```

This will start all required database services in the background. The default connection settings in your `.env` file are pre-configured to work with these services.

### 4. Setup Development Environment

Create Python virtual environment and setup development tools:

```bash
make dev
```

This command will:
*   Install `uv` if not already available
*   Create a Python 3.11 virtual environment (located in `.venv/`)
*   Install development tools (redocly, openapi-generator-cli, etc.)
*   Install pre-commit hooks for code quality
*   Install addlicense tool for license management

**Activate the virtual environment:**
```bash
source .venv/bin/activate
```

You'll know it's active when you see `(.venv)` in your terminal prompt.

### 5. Install Dependencies

Install all backend and frontend dependencies:

```bash
make install
```

This command will:
*   Install all Python backend dependencies from `pyproject.toml` into the virtual environment
*   Install frontend Node.js dependencies using `yarn`

### 6. Apply Database Migrations

Setup the database schema:

```bash
make migrate
```

### 7. Start Development Services

Now you can start the development services. Open separate terminal windows/tabs for each service:

**Terminal 1 - Backend API Server:**
```bash
make run-backend
```
This starts the FastAPI development server at `http://localhost:8000` with auto-reload on code changes.

**Terminal 2 - Celery Worker:**
```bash
make run-celery
```
This starts the Celery worker for processing asynchronous background tasks.

**Terminal 3 - Frontend (Optional):**
```bash
make run-frontend
```
This starts the frontend development server at `http://localhost:3000` with hot reload.

### 8. Access ApeRAG

With the services running, you can access:
*   **Frontend UI**: http://localhost:3000 (if started)
*   **Backend API**: http://localhost:8000
*   **API Documentation**: http://localhost:8000/docs

### 9. Development Workflow

Your typical development cycle:

1. **Code Changes**: Edit files in `aperag/` (backend) or `frontend/src/` (frontend)
2. **Auto-reload**: Backend and frontend automatically reload on file changes
3. **Testing**: Run `make test` to execute tests
4. **Code Quality**: Run `make format` and `make lint` before committing
5. **Database Changes**: If you modify models, run `make makemigration` then `make migrate`

### 10. Stopping Services

To stop the development environment:

```bash
# Stop database services
make compose-down

# Stop development services (Ctrl+C in each terminal)
```

Now you have ApeRAG running locally from source code, ready for development!

## Docker Compose Development Options

For developers who prefer containerized development, Docker Compose offers flexible deployment modes:

### Infrastructure Mode (Recommended for Development)

Start only the essential database services. Perfect for developers who want to run the application code locally for debugging and development.

```bash
# Core databases: PostgreSQL, Redis, Qdrant, Elasticsearch
make compose-infra

# Use Neo4j instead of PostgreSQL for graph storage
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

# Use Neo4j instead of PostgreSQL for graph storage
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
- Uses Neo4j instead of PostgreSQL for graph storage backend
- Provides native graph database capabilities for better performance
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
make compose-infra WITH_NEO4J=1  # Start databases (with Neo4j for graph storage)
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
    *   `make dev`: Sets up the development environment by creating Python virtual environment, installing `uv`, and setting up pre-commit hooks.
    *   `make install`: Installs all necessary backend (Python) and frontend (Node.js) dependencies into the virtual environment.

*   **Database Services**:
    *   `make compose-infra`: Starts essential database services (PostgreSQL, Redis, Qdrant, Elasticsearch) using Docker Compose.
    *   `make compose-infra WITH_NEO4J=1`: Starts database services using Neo4j instead of PostgreSQL for graph storage.
    *   `make compose-down`: Stops all database services.

*   **Development Services**:
    *   `make run-backend`: Starts the FastAPI development server with auto-reload.
    *   `make run-frontend`: Starts the UmiJS frontend development server with hot reload.
    *   `make run-celery`: Starts a Celery worker for processing background tasks.

*   **Code Quality & Testing**:
    *   `make format`: Formats Python code using Ruff and frontend code using Prettier.
    *   `make lint`: Lints Python code with Ruff and frontend code.
    *   `make static-check`: Performs static type checking for Python code using Mypy (if configured).
    *   `make test`: Runs all automated tests (Python unit tests, integration tests).

*   **Database Management**:
    *   `make migrate`: Applies pending database migrations to your connected database.
    *   `make makemigration`: Creates new database migration files based on changes to SQLAlchemy models.

*   **Generators**:
    *   `make generate-models`: Generates Pydantic models from the OpenAPI schema.
    *   `make generate-frontend-sdk`: Generates the frontend API client/SDK from the OpenAPI specification. **Run this command whenever backend API definitions change.**

*   **Docker Compose (for full application testing)**:
    *   `make compose-up`: Starts all services (backend, frontend, databases, Celery) using Docker Compose.
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