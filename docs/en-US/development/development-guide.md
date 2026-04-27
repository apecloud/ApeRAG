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

*   **Node.js**: Version 20 or higher is recommended for frontend development. [Download Node.js](https://nodejs.org/)
*   **Docker & Docker Compose**: Required for running database services locally. [Download Docker](https://docs.docker.com/get-docker/)

**Note**: Python 3.11 is required but will be automatically managed by `uv` in the next steps.

### 3. 🗄️ Start Database Services

Use Docker Compose to start the essential database services:

```bash
# Start core databases: PostgreSQL, Redis, Qdrant, Elasticsearch
make infra-up
```

This will start all required database services in the background. The default connection settings in your `.env` file are pre-configured to work with these services.

<details>
<summary><strong>Advanced Database Options</strong></summary>

```bash
# Use Neo4j instead of PostgreSQL for graph storage
make infra-up WITH_NEO4J=1
```

</details>

### 4. ⚙️ Setup Development Environment

Create Python virtual environment and setup development tools:

```bash
make env-dev
```

This command will:
*   Install `uv` if not already available
*   Create a Python 3.11 virtual environment (located in `.venv/`)
*   Install backend dependencies and repository git hooks
*   Install pre-commit hooks for code quality
*   Install addlicense tool for license management

**Activate the virtual environment:**
```bash
source .venv/bin/activate
```

You'll know it's active when you see `(.venv)` in your terminal prompt.

### 5. 📦 Install Dependencies

Install all backend and frontend dependencies:

```bash
make env-install
```

This command will:
*   Install all Python backend dependencies from `pyproject.toml` into the virtual environment
*   Install frontend Node.js dependencies using `yarn`

### 6. 🔄 Apply Database Migrations

Setup the database schema:

```bash
make db-migrate
```

### 7. ▶️ Start Development Services

Now you can start the development services. Open separate terminal windows/tabs for each service:

**Terminal 1 - Backend API Server:**
```bash
make serve-api
```
This starts the FastAPI development server at `http://localhost:8000` with auto-reload on code changes.

**Terminal 2 - Celery Worker:**
```bash
make serve-worker
```
This starts the Celery worker for processing asynchronous background tasks.

**Terminal 3 - Frontend (Optional):**
```bash
make serve-web
```
This starts the frontend development server at `http://localhost:3000` with hot reload.

### 8. 🌐 Access ApeRAG

With the services running, you can access:
*   **Frontend UI**: http://localhost:3000 (if started)
*   **Backend API**: http://localhost:8000
*   **API Documentation**: http://localhost:8000/docs

### 9. ⏹️ Stopping Services

To stop the development environment:

**Stop Database Services:**
```bash
# Stop database services (data preserved)
make stack-down

# Stop services and remove all data volumes
make stack-down REMOVE_VOLUMES=1
```

**Stop Development Services:**
- Backend API Server: Press `Ctrl+C` in the terminal running `make serve-api`
- Celery Worker: Press `Ctrl+C` in the terminal running `make serve-worker`
- Frontend Server: Press `Ctrl+C` in the terminal running `make serve-web`

**Data Management:**
- `make stack-down` - Stops services but preserves all data (PostgreSQL, Redis, Qdrant, etc.)
- `make stack-down REMOVE_VOLUMES=1` - Stops services and **⚠️ permanently deletes all data**
- You can run `make stack-down REMOVE_VOLUMES=1` even after already running `make stack-down`

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
1. Define request/response models in Python using Pydantic models.
2. Implement backend view: `aperag/views/[module].py`
3. Export the code-first OpenAPI specs:
   ```bash
   make openapi-generate  # Writes openapi.full.json and openapi.public.json
   ```
4. Verify the exported specs:
   ```bash
   make openapi-check
   ```
5. Coordinate frontend typed client updates through the FE v2 adapter layer.
6. Test the API:
   ```bash
   make test-all
   # ✅ Check live docs: http://localhost:8000/docs
   ```

### Q: 🗃️ How do I modify database models/schema?

**Database migration workflow:**
1. Edit SQLModel classes in `aperag/db/models.py`
2. Generate migration file:
   ```bash
   make db-revision  # Creates new migration in migration/versions/
   ```
3. Apply migration to database:
   ```bash
   make db-migrate  # Updates database schema
   ```
4. Update related code (repositories in `aperag/db/repositories/`, services in `aperag/service/`)
5. Verify changes:
   ```bash
   make test-all  # ✅ Ensure everything works
   ```

### Q: ⚡ How do I add a new feature with background processing?

**Feature implementation workflow:**
1. Implement feature components:
   - Backend logic: `aperag/[module]/`
   - Async tasks: `aperag/tasks/`
   - Database models: `aperag/db/models.py`
2. Update API and verify the code-first contract:
   ```bash
   make db-revision      # Generate migration files
   make db-migrate           # Apply database changes
   make openapi-check         # Verify FastAPI/Pydantic OpenAPI export
   ```
3. Quality assurance:
   ```bash
   make format && make lint && make test-all
   ```

### Q: 🧪 How do I run unit tests and e2e tests?

**Unit Tests (Fast, No External Dependencies):**
```bash
# Run all unit tests
make test-unit

# Run specific test file
uv run pytest tests/unit_test/test_model_service.py -v

# Run specific test class or function
uv run pytest tests/unit_test/test_model_service.py::TestModelService::test_get_models -v

# Run tests with coverage
make test-unit
```

**E2E Tests (Require Running Services):**
```bash
# Setup: Start required services first
make infra-up      # 🗄️ Start databases
make serve-api       # 🚀 Start API server (separate terminal)

# Run the remaining pytest-based product e2e tests
make test-e2e

# Run HTTP black-box smoke against a running service
make test-http-smoke

# Run backend integration tests
make test-integration

# Run specific pytest e2e modules
uv run pytest tests/e2e_pytest/test_chat.py -v

# Run specific integration modules
uv run pytest tests/integration/graphstorage/ -v

# Run with detailed output and no capture
uv run pytest tests/e2e_pytest/test_chat.py -v -s

# Performance benchmarks (with timing)
make test-e2e-perf
```

**Complete Test Suite:**
```bash
# Run everything (unit + e2e)
make test-all

# Test with different configurations
make infra-up WITH_NEO4J=1  # Test with Neo4j instead of PostgreSQL
make test-all
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
   make infra-up       # Database services
   make serve-api         # API server
   make serve-worker         # Background workers (if testing async tasks)
   ```
3. Use debugging tools:
   ```bash
   # Run with pdb debugger
   uv run pytest tests/unit_test/test_failing.py --pdb
   
   # Capture logs during test
   uv run pytest tests/e2e_pytest/test_chat.py --log-cli-level=DEBUG
   ```
4. Fix and retest:
   ```bash
   make format              # Auto-fix style issues
   make lint               # Check remaining issues
   uv run pytest tests/path/to/fixed_test.py -v  # Verify fix
   ```

### Q: 📦 How do I update dependencies safely?

**Python dependencies:**
1. Edit `pyproject.toml` (add/update packages)
2. Update virtual environment:
   ```bash
   make env-install            # Syncs all groups and extras with uv
   make test-all              # Verify compatibility
   ```

**Frontend dependencies:**
1. Edit `frontend/package.json`
2. Update and test:
   ```bash
   cd frontend && yarn install
   make serve-web      # Test frontend compilation
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
   make test-all             # All unit + e2e tests
   make test-e2e-perf  # Performance benchmarks
   ```
3. API consistency:
   ```bash
   make openapi-check       # Ensure code-first OpenAPI export works
   ```
4. Database migrations:
   ```bash
   make db-revision    # Generate any pending migrations
   ```
5. Full-stack integration test:
   ```bash
   make stack-up WITH_NEO4J=1 WITH_DOCRAY=1  # Production-like setup
   # Manual testing at http://localhost:3000/web/
   make stack-down
   ```

### Q: 🔄 How do I completely reset my development environment?

**Nuclear reset (destroys all data):**
```bash
make stack-down REMOVE_VOLUMES=1  # ⚠️ Stop services + delete ALL data
make env-clean                         # 🧹 Clean temporary files

# Restart fresh
make infra-up                 # 🗄️ Fresh databases
make db-migrate                      # 🔄 Apply all migrations
make serve-api                  # 🚀 Start API server
make serve-worker                   # ⚡ Start background workers
```

**Soft reset (preserve data):**
```bash
make stack-down                 # ⏹️ Stop services, keep data
make infra-up               # 🗄️ Restart databases
make db-migrate                    # 🔄 Apply any new migrations
```

**Reset just Python environment:**
```bash
rm -rf .venv/                   # 🗑️ Remove virtual environment
make env-dev                       # ⚙️ Recreate everything
source .venv/bin/activate      # ✅ Reactivate
``` 
