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

The `Makefile` provides essential commands for development workflow:

### Environment Setup
*   `make dev` - Sets up complete development environment (creates virtual environment, installs development tools, git hooks)
*   `make install` - Installs all Python and Node.js dependencies

### Database Services  
*   `make compose-infra` - Start core databases (PostgreSQL, Redis, Qdrant, Elasticsearch)
*   `make compose-infra WITH_NEO4J=1` - Start databases with Neo4j for graph storage
*   `make compose-down` - Stop database services (preserves data)
*   `make compose-down REMOVE_VOLUMES=1` - Stop services and delete all data

### Development Services
*   `make run-backend` - Start FastAPI development server with auto-reload
*   `make run-celery` - Start Celery worker for background tasks  
*   `make run-frontend` - Start React frontend development server

### Database Management
*   `make migrate` - Apply database migrations
*   `make makemigration` - Create new migration files for model changes

### Code Quality
*   `make format` - Format code (Python with Ruff, frontend with Prettier)
*   `make lint` - Lint code and check style
*   `make test` - Run all tests (unit + integration)

### Code Generation
*   `make generate-models` - Generate Pydantic models from OpenAPI schema
*   `make generate-frontend-sdk` - Generate frontend API client (**run after API changes**)

### Docker Compose Testing
*   `make compose-up` - Start complete application stack for testing
*   `make compose-logs` - View logs from all services

## Common Development Tasks

### Q: How do I add or modify a REST API endpoint?

**Steps:**
1. Edit the OpenAPI specification: `aperag/api/paths/[endpoint-name].yaml`
2. Update backend view: `aperag/views/[module].py`
3. Generate models and frontend SDK:
   ```bash
   make generate-models
   make generate-frontend-sdk
   ```
4. Test the API:
   ```bash
   make test
   # Check API docs at http://localhost:8000/docs
   ```

### Q: How do I modify database models/schema?

**Steps:**
1. Edit SQLModel classes in `aperag/db/models.py`
2. Generate and apply migration:
   ```bash
   make makemigration
   make migrate
   ```
3. Update related code (repositories, services)
4. Test database changes:
   ```bash
   make test
   ```

### Q: How do I add a new feature from scratch?

**Complete workflow:**
1. Create feature branch: `git checkout -b feat/my-feature`
2. Start development environment:
   ```bash
   make compose-infra
   make run-backend
   make run-celery
   ```
3. Implement backend logic in `aperag/[module]/`
4. Add database models and migrate:
   ```bash
   make makemigration
   make migrate
   ```
5. Create API endpoints and update OpenAPI spec
6. Generate frontend code:
   ```bash
   make generate-models
   make generate-frontend-sdk
   ```
7. Implement frontend (optional): `frontend/src/`
8. Code quality check:
   ```bash
   make format
   make lint
   make test
   ```

### Q: How do I debug a failing test?

**Debugging workflow:**
1. Run specific test:
   ```bash
   # Unit tests
   make unit-test
   
   # Specific test file
   uv run pytest tests/unit_test/test_specific.py -v
   
   # Integration tests  
   uv run pytest tests/e2e_test/test_specific.py -v
   ```
2. Check services are running:
   ```bash
   make compose-infra
   make run-backend
   ```
3. Check logs and debug with breakpoints
4. Fix code and retest:
   ```bash
   make format
   make test
   ```

### Q: How do I update dependencies?

**For Python dependencies:**
1. Edit `pyproject.toml`
2. Update environment:
   ```bash
   make install
   make test
   ```

**For frontend dependencies:**
1. Edit `frontend/package.json`
2. Update and test:
   ```bash
   cd frontend && yarn install
   make run-frontend
   ```

### Q: How do I prepare code for pull request?

**Before submitting:**
1. Ensure all tests pass:
   ```bash
   make test
   ```
2. Code quality checks:
   ```bash
   make format
   make lint
   ```
3. If API changed, update frontend SDK:
   ```bash
   make generate-frontend-sdk
   ```
4. If models changed, ensure migrations are created:
   ```bash
   make makemigration
   ```
5. Test complete application:
   ```bash
   make compose-up
   # Test at http://localhost:3000/web/
   make compose-down
   ```

### Q: How do I reset my development environment?

**Complete reset:**
```bash
# Stop all services and delete data
make compose-down REMOVE_VOLUMES=1

# Restart fresh environment
make compose-infra
make migrate
make run-backend
make run-celery
```

**Partial reset (keep data):**
```bash
# Just restart services
make compose-down
make compose-infra
``` 