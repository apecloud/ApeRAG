# Configuration variables
VERSION ?= nightly
VERSION_FILE ?= aperag/version/__init__.py
BUILDX_PLATFORM ?= linux/amd64,linux/arm64
BUILDX_ARGS ?= --sbom=false --provenance=false
REGISTRY ?=  apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com

# Image names
APERAG_IMAGE = apecloud/aperag
APERAG_FRONTEND_IMG = apecloud/aperag-frontend

# Detect host architecture
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    LOCAL_PLATFORM = linux/amd64
else ifeq ($(UNAME_M),aarch64)
    LOCAL_PLATFORM = linux/arm64
else ifeq ($(UNAME_M),arm64)
    LOCAL_PLATFORM = linux/arm64
else
    LOCAL_PLATFORM = linux/amd64
endif

.PHONY: help
help:
	@printf "\nApeRAG Make Targets\n\n"
	@printf "Recommended commands:\n\n"
	@printf "Environment\n"
	@printf "  make env-install          Install Python dependencies into .venv\n"
	@printf "  make env-dev              Prepare the local development environment\n"
	@printf "  make env-clean            Clean local development state\n\n"
	@printf "Database / Infra\n"
	@printf "  make db-migrate           Apply database migrations\n"
	@printf "  make db-check             Verify schema matches SQLAlchemy models (no pending diff)\n"
	@printf "  make db-revision          Create a new alembic migration\n"
	@printf "  make infra-up             Start infra dependencies only\n"
	@printf "  make stack-up             Start the full local stack\n"
	@printf "  make stack-down           Stop the local stack\n"
	@printf "  make stack-logs           Tail stack logs\n\n"
	@printf "Services\n"
	@printf "  make serve-api            Run backend API locally\n"
	@printf "  make serve-worker         Run celery worker locally\n"
	@printf "  make serve-beat           Run celery beat locally\n"
	@printf "  make serve-flower         Run flower locally\n"
	@printf "  make serve-web            Run frontend locally\n\n"
	@printf "Tests\n"
	@printf "  make test-all             Run unit + integration + pytest E2E suites\n"
	@printf "  make test-unit            Run unit tests\n"
	@printf "  make test-integration     Run integration tests\n"
	@printf "  make test-e2e             Run pytest-based residual E2E tests\n"
	@printf "  make test-e2e-perf        Run pytest-based E2E performance tests\n"
	@printf "  make test-http-bootstrap  Prepare HTTP E2E bootstrap state\n"
	@printf "  make test-http-smoke      Run HTTP smoke suite against an existing target\n"
	@printf "  make test-http-full       Run full HTTP suite against an existing target\n"
	@printf "  make test-http-up-compose / test-http-down-compose\n"
	@printf "  make test-http-smoke-compose / test-http-full-compose\n"
	@printf "  make test-http-up-k8s / test-http-down-k8s\n"
	@printf "  make test-http-smoke-k8s / test-http-full-k8s\n\n"
	@printf "Build / API\n"
	@printf "  make openapi-generate     Export code-first OpenAPI specs\n"
	@printf "  make openapi-check        Verify code-first OpenAPI export\n"
	@printf "  make build                Build production images\n"
	@printf "  make release-version      Generate version metadata\n\n"

##################################################
# Environment & Dependencies
##################################################

# Python environment setup
.PHONY: install-uv venv env-install env-dev env-clean
install-uv:
	@if [ -z "$$(which uv)" ]; then \
		echo "Installing uv..."; \
		pip install uv; \
	fi

venv: install-uv
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv -p 3.11.12; \
	fi

env-install: venv
	@echo "Installing Python dependencies..."
	uv sync --all-groups --all-extras

# Development environment setup
env-dev: env-install
	@echo "Installing development tools..."
	@echo ""
	@echo "✅ Development environment ready!"
	@echo "📝 Next steps:"
	@echo "   1. Activate virtual environment: source .venv/bin/activate"
	@echo "   2. Start databases: make infra-up"
	@echo "   3. Apply migrations: make db-migrate"
	@echo "   4. Run services: make serve-api, make serve-worker"

# Environment cleanup
env-clean:
	@echo "Cleaning development environment..."
	@rm -f db.sqlite3
	@$(MAKE) stack-down REMOVE_VOLUMES=1

##################################################
# Database & Infrastructure
##################################################

# Database schema management
.PHONY: db-revision db-migrate db-check
db-revision:
	@uv run alembic -c aperag/alembic.ini revision --autogenerate

db-migrate:
	@uv run alembic -c aperag/alembic.ini upgrade head

db-check:
	@uv run alembic -c aperag/alembic.ini check

# Docker Compose infrastructure

# Variables for compose command based on environment flags
# Usage examples:
#   make stack-up                                # Full application
#   make stack-up WITH_NEO4J=1                   # Full application + Neo4j
#   make stack-up WITH_NEBULA=1                  # Full application + Nebula Graph
#   make infra-up                                # Infrastructure only (databases)
#   make infra-up WITH_NEO4J=1                   # Infrastructure + Neo4j
#   make stack-down                              # Stop all services
#   make stack-down REMOVE_VOLUMES=1             # Stop and remove volumes
_PROFILES_TO_ACTIVATE :=
_EXTRA_ENVS :=
_COMPOSE_DOWN_FLAGS :=

# Determine which additional profiles to activate
ifeq ($(WITH_NEO4J),1)
    _PROFILES_TO_ACTIVATE += --profile neo4j
endif

ifeq ($(WITH_NEBULA),1)
    _PROFILES_TO_ACTIVATE += --profile nebula
endif

# Determine flags for 'compose-down'
ifeq ($(REMOVE_VOLUMES),1)
    _COMPOSE_DOWN_FLAGS += -v
endif

.PHONY: stack-up stack-down stack-logs infra-up
# Full application startup
stack-up:
	$(_EXTRA_ENVS) docker-compose $(_PROFILES_TO_ACTIVATE) -f docker-compose.yml up -d

# Infrastructure only (databases + supporting services)
# Optional services like Neo4j and Nebula will ONLY start if explicitly enabled:
#   make infra-up WITH_NEO4J=1    # adds Neo4j
#   make infra-up WITH_NEBULA=1   # adds Nebula Graph
infra-up:
	docker-compose $(_PROFILES_TO_ACTIVATE) -f docker-compose.yml up -d \
		postgres redis qdrant es \
		$(if $(filter 1,$(WITH_NEO4J)),neo4j,) \
		$(if $(filter 1,$(WITH_NEBULA)),nebula-metad nebula-storaged nebula-graphd nebula-storage-activator,)

stack-down:
	docker-compose --profile neo4j --profile nebula -f docker-compose.yml down $(_COMPOSE_DOWN_FLAGS)

stack-logs:
	docker-compose -f docker-compose.yml logs -f

##################################################
# Development Services
##################################################

# Local development services
.PHONY: serve-api serve-web serve-worker serve-flower serve-beat
serve-api: db-migrate
	uvicorn aperag.app:app --host 0.0.0.0 --log-config scripts/uvicorn-log-config.yaml

serve-worker:
	celery -A config.celery worker -B -l INFO --pool=threads --concurrency=16

serve-beat:
	celery -A config.celery beat -l INFO

serve-flower:
	celery -A config.celery flower --conf/flowerconfig.py

serve-web:
	cd ./web && yarn dev

##################################################
# Code Quality & Testing
##################################################

# Code quality checks
.PHONY: format lint static-check
format:
	uvx ruff check --fix ./aperag ./tests
	uvx ruff format ./aperag ./tests

lint:
	uvx ruff check --no-fix ./aperag ./tests
	uvx ruff format --check ./aperag ./tests

static-check:
	uvx mypy ./aperag

# Testing suite
.PHONY: test-all test-unit test-integration test-e2e test-e2e-perf \
	test-http-bootstrap test-http-smoke test-http-full \
	test-http-up-compose test-http-down-compose test-http-smoke-compose test-http-full-compose \
	test-http-up-k8s test-http-down-k8s test-http-smoke-k8s test-http-full-k8s
test-all: test-unit test-integration test-e2e

# Cross-backend compatibility tests.
# Require running databases; use `make infra-up WITH_NEO4J=1 WITH_NEBULA=1`
# to start all backends, then run these targets.
#
# PG is always tested (uses the default compose postgres).
# Neo4j / Nebula are opt-in via their env vars.
.PHONY: test-compat-graph test-compat-vector test-compat-all
test-compat-graph:
	COMPAT_PG_URL=$${COMPAT_PG_URL:-postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres} \
	COMPAT_NEO4J_URI=$${COMPAT_NEO4J_URI:-} \
	COMPAT_NEO4J_USER=$${COMPAT_NEO4J_USER:-neo4j} \
	COMPAT_NEO4J_PASS=$${COMPAT_NEO4J_PASS:-password} \
	COMPAT_NEBULA_HOSTS=$${COMPAT_NEBULA_HOSTS:-} \
	uv run pytest tests/integration/compat/test_graph_compat.py -v

test-compat-vector:
	COMPAT_QDRANT_URL=$${COMPAT_QDRANT_URL:-http://127.0.0.1:6333} \
	COMPAT_PGVECTOR_URL=$${COMPAT_PGVECTOR_URL:-postgresql://postgres:postgres@127.0.0.1:5432/postgres} \
	uv run pytest tests/integration/compat/test_vector_compat.py -v

test-compat-all: test-compat-graph test-compat-vector

test-unit:
	@mkdir -p tests/report
	uv run pytest tests/unit_test/ -v \
		--cov=aperag \
		--cov-report=term-missing:skip-covered \
		--cov-report=xml:tests/report/unit-coverage.xml \
		--cov-report=json:tests/report/unit-coverage.json

test-integration:
	uv run pytest tests/integration/ -v

test-e2e:
	uv run pytest --benchmark-disable tests/e2e_pytest/ -v

test-e2e-perf:
	@echo "Running E2E performance test..."
	@uv run pytest -v \
		--benchmark-enable \
		--benchmark-max-time=10 \
		--benchmark-min-rounds=100 \
		--benchmark-save-data \
		--benchmark-storage=tests/report \
		--benchmark-save=benchmark-result-$$(date +%Y%m%d%H%M%S) \
		tests/e2e_pytest/

test-http-bootstrap:
	@./tests/e2e_http/bootstrap/bootstrap.sh

test-http-smoke:
	@./tests/e2e_http/scripts/run_smoke.sh

test-http-full:
	@./tests/e2e_http/scripts/run_full.sh

test-http-up-compose:
	@./tests/e2e_http/runners/compose/up.sh

test-http-down-compose:
	@./tests/e2e_http/runners/compose/down.sh

test-http-smoke-compose:
	@./tests/e2e_http/scripts/run_compose_smoke.sh

test-http-full-compose:
	@./tests/e2e_http/scripts/run_compose_full.sh

test-http-up-k8s:
	@./tests/e2e_http/runners/k8s/up.sh

test-http-down-k8s:
	@./tests/e2e_http/runners/k8s/down.sh

test-http-smoke-k8s:
	@./tests/e2e_http/scripts/run_k8s_smoke.sh

test-http-full-k8s:
	@./tests/e2e_http/scripts/run_k8s_full.sh

##################################################
# Code Generation & API
##################################################

# OpenAPI and model generation
.PHONY: openapi-generate openapi-check
openapi-generate:
	@uv run python scripts/export_openapi.py

openapi-check:
	@uv run python scripts/export_openapi.py --check

# LLM configuration generation
.PHONY: llm_provider
llm_provider:
	python ./models/generate_model_configs.py

# Version management
.PHONY: release-version
release-version:
	@git rev-parse HEAD | cut -c1-7 > commit_id.txt
	@echo "VERSION = \"$(VERSION)\"" > $(VERSION_FILE)
	@echo "GIT_COMMIT_ID = \"$$(cat commit_id.txt)\"" >> $(VERSION_FILE)
	@rm commit_id.txt

##################################################
# Build & Deploy
##################################################

# Docker builder setup
.PHONY: setup-builder clean-builder
setup-builder:
	@if ! docker buildx inspect multi-platform >/dev/null 2>&1; then \
		docker buildx create --name multi-platform --use --driver docker-container --bootstrap; \
	else \
		docker buildx use multi-platform; \
	fi

clean-builder:
	@if docker buildx inspect multi-platform >/dev/null 2>&1; then \
		docker buildx rm multi-platform; \
	fi

build-aperag-frontend-assets:
	cd web && yarn install && yarn build

# Production builds (multi-platform with registry push)
.PHONY: build build-aperag build-aperag-frontend
build: build-aperag build-aperag-frontend

build-aperag: setup-builder release-version
	docker buildx build -t $(REGISTRY)/$(APERAG_IMAGE):$(VERSION) \
		--platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push \
		-f ./Dockerfile .

build-aperag-frontend: setup-builder build-aperag-frontend-assets
	cd web && docker buildx build \
		--platform=$(BUILDX_PLATFORM) -f Dockerfile --push \
		-t $(REGISTRY)/$(APERAG_FRONTEND_IMG):$(VERSION) .

# Local builds (single platform for testing)
.PHONY: build-local build-aperag-local build-aperag-frontend-local
build-local: build-aperag-local build-aperag-frontend-local

build-aperag-local: setup-builder release-version
	docker buildx build -t $(APERAG_IMAGE):$(VERSION) \
		--platform $(LOCAL_PLATFORM) $(BUILDX_ARGS) --load \
		-f ./Dockerfile .

build-aperag-frontend-local: setup-builder build-aperag-frontend-assets
	cd web && docker buildx build \
		--platform=$(LOCAL_PLATFORM) -f Dockerfile --load \
		-t $(APERAG_FRONTEND_IMG):$(VERSION) .

# Kubernetes deployment helpers
.PHONY: load-images-to-minikube load-images-to-kind
load-images-to-minikube:
	@echo "Start To Load Image To Minikube"
	docker save $(APERAG_IMAGE):$(VERSION) -o aperag.tar
	minikube image load aperag.tar
	rm aperag.tar
	docker save $(APERAG_FRONTEND_IMG):$(VERSION) -o aperag-frontend.tar
	minikube image load aperag-frontend.tar
	rm aperag-frontend.tar
	@echo "Already Load Image To Minikube"

load-images-to-kind:
	@echo "Start To Load Image To KinD"
	kind load docker-image $(APERAG_IMAGE):$(VERSION) --name $(KIND_CLUSTER_NAME)
	kind load docker-image $(APERAG_FRONTEND_IMG):$(VERSION) --name $(KIND_CLUSTER_NAME)
	@echo "Already Load Image To KinD"

##################################################
# Utilities & Tools
##################################################

# Documentation sync
.PHONY: docs
docs:
	@echo "Syncing documentation from docs/ to web/docs/"
	@/usr/bin/python3 scripts/sync-docs.py

# System information
.PHONY: info
info:
	@echo "VERSION: $(VERSION)"
	@echo "BUILDX_PLATFORM: $(BUILDX_PLATFORM)"
	@echo "LOCAL_PLATFORM: $(LOCAL_PLATFORM)"
	@echo "REGISTRY: $(REGISTRY)"
	@echo "HOST ARCH: $(UNAME_M)"

