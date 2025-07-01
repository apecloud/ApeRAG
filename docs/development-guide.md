# Development Guide

This section focuses on the development workflow and tools provided for ApeRAG.

## Development Environment

It's recommended to use the "Getting Started with Source Code" approach for setting up a development environment. Ensure all prerequisites are met and dependencies are installed using `make install`.

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

2.  **Environment Setup**: Ensure your development environment is set up as described in [Development Environment](#development-environment) and [Getting Started with Source Code](../README.md#getting-started-with-source-code) (dependencies installed, databases running/accessible).

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