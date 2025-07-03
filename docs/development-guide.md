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

*   **Node.js**: Version 20 or higher is recommended for frontend development. [Download Node.js](https://nodejs.org/)
*   **Docker & Docker Compose**: Required for running database services locally. [Download Docker](https://docs.docker.com/get-docker/)

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

### 9. Stopping Services

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
- `make compose-down REMOVE_VOLUMES=1` - Stops services and **permanently deletes all data**
- You can run `make compose-down REMOVE_VOLUMES=1` even after already running `make compose-down`

**Verify Data Removal:**
```bash
# Check if volumes still exist
docker volume ls | grep aperag

# Should return no results after REMOVE_VOLUMES=1
```

Now you have ApeRAG running locally from source code, ready for development!



## Key `make` Commands for Development

Based on the Makefile structure, here are the essential commands organized by workflow:

### 🚀 Environment Setup
```bash
make dev                    # Complete development setup (virtual env, tools, git hooks)
make install               # Install all Python and Node.js dependencies  
make venv                  # Create Python virtual environment only
make install-uv            # Install uv package manager
```

### 🗄️ Database & Infrastructure
```bash
# Start databases only (recommended for development)
make compose-infra                     # PostgreSQL + Redis + Qdrant + Elasticsearch
make compose-infra WITH_NEO4J=1        # Add Neo4j for graph storage

# Database schema management
make migrate                           # Apply pending migrations
make makemigration                     # Generate new migration files

# Stop services
make compose-down                      # Stop services (keep data)
make compose-down REMOVE_VOLUMES=1     # Stop services and DELETE ALL DATA
```

### ⚡ Development Services
```bash
# Backend services (run in separate terminals)
make run-backend           # FastAPI server with auto-reload (includes migration)
make run-celery           # Celery worker + beat scheduler (--pool=threads --concurrency=16)
make run-beat             # Celery beat scheduler only
make run-flower           # Celery monitoring web UI

# Frontend service
make run-frontend         # React development server (copies env template)
```

### 🐳 Docker Compose (Full Stack)
```bash
# Complete application testing
make compose-up                           # Full stack (API + Frontend + Workers + Databases)
make compose-up WITH_NEO4J=1              # Add Neo4j 
make compose-up WITH_DOCRAY=1             # Add advanced document parsing (CPU)
make compose-up WITH_DOCRAY=1 WITH_GPU=1  # Add GPU-accelerated document parsing
make compose-logs                         # View all service logs
```

### 🧪 Testing & Quality
```bash
# Testing
make test                  # All tests (unit + e2e)
make unit-test            # Unit tests only
make e2e-test             # End-to-end tests (--benchmark-disable)
make e2e-performance-test # Performance benchmarks

# Code quality
make format               # Auto-fix code style (Ruff for Python)
make lint                 # Check code style (no auto-fix)
make static-check         # Type checking with MyPy
```

### 🔧 Code Generation & API
```bash
# When you modify APIs
make generate-models           # Regenerate Pydantic models from OpenAPI spec
make generate-frontend-sdk     # Regenerate frontend TypeScript API client

# OpenAPI workflow
make merge-openapi            # Bundle OpenAPI spec files

# LLM configuration
make llm_provider             # Generate LLM provider configurations
```

### 📊 Evaluation & Analysis  
```bash
make evaluate             # Run RAG evaluation suite
make clean               # Clean temporary files
```

## Common Development Tasks

### Q: How do I add or modify a REST API endpoint?

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
   # Check live docs: http://localhost:8000/docs
   ```

### Q: How do I modify database models/schema?

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
   make test
   ```

### Q: How do I add a new feature with background processing?

**Full-stack feature development:**
1. Setup development environment:
   ```bash
   make dev                    # One-time setup
   source .venv/bin/activate   # Activate environment
   make compose-infra          # Start databases
   ```
2. Start development services (3 terminals):
   ```bash
   # Terminal 1: Backend API
   make run-backend           # Includes auto-migration
   
   # Terminal 2: Background tasks  
   make run-celery           # Worker + scheduler (threads=16)
   
   # Terminal 3: Task monitoring (optional)
   make run-flower           # Web UI at http://localhost:5555
   ```
3. Implement feature:
   - Backend logic: `aperag/[module]/`
   - Async tasks: `aperag/tasks/`
   - Database models: `aperag/db/models.py`
4. Update API and generate code:
   ```bash
   make makemigration
   make migrate
   make generate-models
   make generate-frontend-sdk
   ```
5. Quality assurance:
   ```bash
   make format && make lint && make test
   ```

### Q: How do I test with different database backends?

**Testing with PostgreSQL (default):**
```bash
make compose-infra         # PostgreSQL + Redis + Qdrant + Elasticsearch
make run-backend
```

**Testing with Neo4j for graph storage:**
```bash
make compose-infra WITH_NEO4J=1  # Neo4j replaces PostgreSQL for graphs
make run-backend
# Access Neo4j UI: http://localhost:7474
```

**Testing complete stack with advanced parsing:**
```bash
# CPU-based document parsing
make compose-up WITH_DOCRAY=1

# GPU-accelerated parsing (requires ~6GB VRAM)
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

### Q: How do I debug failing tests?

**Test debugging workflow:**
1. Run specific test categories:
   ```bash
   make unit-test            # Fast unit tests only
   make e2e-test            # Integration tests (--benchmark-disable)
   
   # Specific test file with verbose output
   uv run pytest tests/unit_test/test_specific.py -v -s
   ```
2. Debug with services running:
   ```bash
   make compose-infra       # Ensure databases are up
   make run-backend         # API server for integration tests
   ```
3. Performance testing:
   ```bash
   make e2e-performance-test  # Benchmarks with --benchmark-enable
   ```
4. Fix and verify:
   ```bash
   make format              # Auto-fix style issues
   make lint               # Check remaining issues
   make test               # Full test suite
   ```

### Q: How do I run RAG evaluation and analysis?

**Evaluation workflow:**
```bash
# Ensure environment is ready
make compose-infra WITH_NEO4J=1  # Use Neo4j for better graph performance
make run-backend
make run-celery

# Run comprehensive RAG evaluation
make evaluate               # Runs aperag.evaluation.run module

# Check evaluation reports in tests/report/
```

### Q: How do I update dependencies safely?

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

### Q: How do I prepare code for production deployment?

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

### Q: How do I completely reset my development environment?

**Nuclear reset (destroys all data):**
```bash
make compose-down REMOVE_VOLUMES=1  # Stop services + delete ALL data
make clean                         # Clean temporary files

# Restart fresh
make compose-infra                 # Fresh databases
make migrate                      # Apply all migrations
make run-backend                  # Start API server
make run-celery                   # Start background workers
```

**Soft reset (preserve data):**
```bash
make compose-down                 # Stop services, keep data
make compose-infra               # Restart databases
make migrate                    # Apply any new migrations
```

**Reset just Python environment:**
```bash
rm -rf .venv/                   # Remove virtual environment
make dev                       # Recreate everything
source .venv/bin/activate      # Reactivate
``` 