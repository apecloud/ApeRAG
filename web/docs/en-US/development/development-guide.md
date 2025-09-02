---
title: Development Guide
description:
keywords:
---

# 🛠️ Development Guide

This guide focuses on setting up a development environment and the development workflow for ApeRAG. This is designed for developers looking to contribute to ApeRAG or run it locally for development purposes.

## 🚀 Development Environment Setup

Follow these steps to set up ApeRAG from source code for development:

### 1. 📂 Clone the Repository and Setup Environment

First, get the source code and configure environment variables:

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
```

Edit the `.env` file to configure your AI service settings if needed. The default settings work with the local database services started in the next step.

### 2. 📋 System Prerequisites

Before you begin, ensure your system has:

- **Node.js**: Version 20 or higher is recommended for frontend development. [Download Node.js](https://nodejs.org/)
- **Docker & Docker Compose**: Required for running database services locally. [Download Docker](https://docs.docker.com/get-docker/)

**Note**: Python 3.11 is required but will be automatically managed by `uv` in the next steps.

### 3. 🗄️ Start Database Services

Use Docker Compose to start the essential database services:

```bash
# Start core databases: PostgreSQL, Redis, Qdrant, Elasticsearch
make compose-infra
```

This will start all required database services in the background. The default connection settings in your `.env` file are pre-configured to work with these services.

<details>
<summary><strong>Advanced Database Options</strong></summary>

```bash
# Use Neo4j instead of PostgreSQL for graph storage
make compose-infra WITH_NEO4J=1

# Add advanced document parsing service (DocRay)
make compose-infra WITH_DOCRAY=1

# Combine multiple options
make compose-infra WITH_NEO4J=1 WITH_DOCRAY=1

# GPU-accelerated document parsing (requires ~6GB VRAM)
make compose-infra WITH_DOCRAY=1 WITH_GPU=1
```

**Note**: DocRay provides enhanced document parsing for complex PDFs, tables, and formulas. CPU mode requires 4+ cores and 8GB+ RAM.

</details>

### 4. ⚙️ Setup Development Environment

Create Python virtual environment and setup development tools:

```bash
make dev
```

This command will:

- Install `uv` if not already available
- Create a Python 3.11 virtual environment (located in `.venv/`)
- Install development tools (redocly, openapi-generator-cli, etc.)
- Install pre-commit hooks for code quality
- Install addlicense tool for license management

**Activate the virtual environment:**

```bash
source .venv/bin/activate
```

You'll know it's active when you see `(.venv)` in your terminal prompt.

### 5. 📦 Install Dependencies

Install all backend and frontend dependencies:

```bash
make install
```

This command will:

- Install all Python backend dependencies from `pyproject.toml` into the virtual environment
- Install frontend Node.js dependencies using `yarn`

### 6. 🔄 Apply Database Migrations

Setup the database schema:

```bash
make migrate
```

### 7. ▶️ Start Development Services

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

### 8. 🌐 Access ApeRAG

With the services running, you can access:

- **Frontend UI**: http://localhost:3000 (if started)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 9. ⏹️ Stopping Services

To stop the development environment:

**Stop Database Services:**

```bash
# Stop database services (data preserved)
make compose-down

# Stop services and remove all data volumes
make compose-down REMOVE_VOLUMES=1
```

**Stop Development Services:**

- Backend API Server: Press `Ctrl+C` in the terminal running `make run-backend`
- Celery Worker: Press `Ctrl+C` in the terminal running `make run-celery`
- Frontend Server: Press `Ctrl+C` in the terminal running `make run-frontend`

**Data Management:**

- `make compose-down` - Stops services but preserves all data (PostgreSQL, Redis, Qdrant, etc.)
- `make compose-down REMOVE_VOLUMES=1` - Stops services and **⚠️ permanently deletes all data**
- You can run `make compose-down REMOVE_VOLUMES=1` even after already running `make compose-down`

**Verify Data Removal:**

```bash
# Check if volumes still exist
docker volume ls | grep aperag

# Should return no results after REMOVE_VOLUMES=1
```

Now you have ApeRAG running locally from source code, ready for development! 🎉

## ❓ Common Development Tasks

### Q: 🔧 How do I add or modify a REST API endpoint?

**Complete workflow:**

1. Edit OpenAPI specification: `aperag/api/paths/[endpoint-name].yaml`
2. Regenerate backend models:
   ```bash
   make generate-models  # This runs merge-openapi internally
   ```
3. Implement backend view: `aperag/views/[module].py`
4. Generate frontend TypeScript client:
   ```bash
   make generate-frontend-sdk  # Updates frontend/src/api/
   ```
5. Test the API:
   ```bash
   make test
   # ✅ Check live docs: http://localhost:8000/docs
   ```

### Q: 🗃️ How do I modify database models/schema?

**Database migration workflow:**

1. Edit SQLModel classes in `aperag/db/models.py`
2. Generate migration file:
   ```bash
   make makemigration  # Creates new migration in migration/versions/
   ```
3. Apply migration to database:
   ```bash
   make migrate  # Updates database schema
   ```
4. Update related code (repositories in `aperag/db/repositories/`, services in `aperag/service/`)
5. Verify changes:
   ```bash
   make test  # ✅ Ensure everything works
   ```

### Q: ⚡ How do I add a new feature with background processing?

**Feature implementation workflow:**

1. Implement feature components:
   - Backend logic: `aperag/[module]/`
   - Async tasks: `aperag/tasks/`
   - Database models: `aperag/db/models.py`
2. Update API and generate code:
   ```bash
   make makemigration      # Generate migration files
   make migrate           # Apply database changes
   make generate-models   # Update Pydantic models
   make generate-frontend-sdk  # Update TypeScript client
   ```
3. Quality assurance:
   ```bash
   make format && make lint && make test
   ```

### Q: 🧪 How do I run unit tests and e2e tests?

**Unit Tests (Fast, No External Dependencies):**

```bash
# Run all unit tests
make unit-test

# Run specific test file
uv run pytest tests/unit_test/test_model_service.py -v

# Run specific test class or function
uv run pytest tests/unit_test/test_model_service.py::TestModelService::test_get_models -v

# Run tests with coverage
uv run pytest tests/unit_test/ --cov=aperag --cov-report=html
```

**E2E Tests (Require Running Services):**

```bash
# Setup: Start required services first
make compose-infra      # 🗄️ Start databases
make run-backend       # 🚀 Start API server (separate terminal)

# Run all e2e tests
make e2e-test

# Run specific e2e test modules
uv run pytest tests/e2e_test/test_chat/ -v
uv run pytest tests/e2e_test/graphstorage/ -v

# Run with detailed output and no capture
uv run pytest tests/e2e_test/test_specific.py -v -s

# Performance benchmarks (with timing)
make e2e-performance-test
```

**Complete Test Suite:**

```bash
# Run everything (unit + e2e)
make test

# Test with different configurations
make compose-infra WITH_NEO4J=1  # Test with Neo4j instead of PostgreSQL
make test
```

### Q: 🐛 How do I debug failing tests?

**Debugging workflow:**

1. Run failing test in isolation:

   ```bash
   # Single test with full output
   uv run pytest tests/unit_test/test_failing.py::test_specific_function -v -s

   # Stop on first failure
   uv run pytest tests/unit_test/ -x --tb=short
   ```

2. For e2e test failures, ensure services are running:
   ```bash
   make compose-infra       # Database services
   make run-backend         # API server
   make run-celery         # Background workers (if testing async tasks)
   ```
3. Use debugging tools:

   ```bash
   # Run with pdb debugger
   uv run pytest tests/unit_test/test_failing.py --pdb

   # Capture logs during test
   uv run pytest tests/e2e_test/test_failing.py --log-cli-level=DEBUG
   ```

4. Fix and retest:
   ```bash
   make format              # Auto-fix style issues
   make lint               # Check remaining issues
   uv run pytest tests/path/to/fixed_test.py -v  # Verify fix
   ```

### Q: 📊 How do I run RAG evaluation and analysis?

**Evaluation workflow:**

```bash
# Ensure environment is ready
make compose-infra WITH_NEO4J=1  # Use Neo4j for better graph performance
make run-backend
make run-celery

# Run comprehensive RAG evaluation
make evaluate               # 📊 Runs aperag.evaluation.run module

# 📈 Check evaluation reports in tests/report/
```

### Q: 📦 How do I update dependencies safely?

**Python dependencies:**

1. Edit `pyproject.toml` (add/update packages)
2. Update virtual environment:
   ```bash
   make install            # Syncs all groups and extras with uv
   make test              # Verify compatibility
   ```

**Frontend dependencies:**

1. Edit `frontend/package.json`
2. Update and test:
   ```bash
   cd frontend && yarn install
   make run-frontend      # Test frontend compilation
   make generate-frontend-sdk  # Ensure API client still works
   ```

### Q: 🚀 How do I prepare code for production deployment?

**Pre-deployment checklist:**

1. Code quality validation:
   ```bash
   make format            # Auto-fix all style issues
   make lint             # Verify no style violations
   make static-check     # MyPy type checking
   ```
2. Comprehensive testing:
   ```bash
   make test             # All unit + e2e tests
   make e2e-performance-test  # Performance benchmarks
   ```
3. API consistency:
   ```bash
   make generate-models         # Ensure models match OpenAPI spec
   make generate-frontend-sdk   # Update frontend client
   ```
4. Database migrations:
   ```bash
   make makemigration    # Generate any pending migrations
   ```
5. Full-stack integration test:
   ```bash
   make compose-up WITH_NEO4J=1 WITH_DOCRAY=1  # Production-like setup
   # Manual testing at http://localhost:3000/web/
   make compose-down
   ```

### Q: 🔄 How do I completely reset my development environment?

**Nuclear reset (destroys all data):**

```bash
make compose-down REMOVE_VOLUMES=1  # ⚠️ Stop services + delete ALL data
make clean                         # 🧹 Clean temporary files

# Restart fresh
make compose-infra                 # 🗄️ Fresh databases
make migrate                      # 🔄 Apply all migrations
make run-backend                  # 🚀 Start API server
make run-celery                   # ⚡ Start background workers
```

**Soft reset (preserve data):**

```bash
make compose-down                 # ⏹️ Stop services, keep data
make compose-infra               # 🗄️ Restart databases
make migrate                    # 🔄 Apply any new migrations
```

**Reset just Python environment:**

```bash
rm -rf .venv/                   # 🗑️ Remove virtual environment
make dev                       # ⚙️ Recreate everything
source .venv/bin/activate      # ✅ Reactivate
```
