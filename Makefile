# Configuration variables
VERSION ?= nightly
VERSION_FILE ?= aperag/version/__init__.py
BUILDX_PLATFORM ?= linux/amd64,linux/arm64
BUILDX_ARGS ?= --sbom=false --provenance=false
REGISTRY ?= registry.cn-hangzhou.aliyuncs.com

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

##################################################
# Users - Local Development and Deployment
##################################################

# Environment setup
.PHONY: install-uv venv install
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

install: venv
	@echo "Installing Python dependencies..."
	uv sync --all-groups --all-extras

# Database management
.PHONY: makemigration migrate
makemigration:
	@alembic -c aperag/alembic.ini revision --autogenerate

migrate:
	@alembic -c aperag/alembic.ini upgrade head

# Local services
.PHONY: run-backend run-frontend run-celery run-flower
run-backend: migrate
	uvicorn aperag.app:app --host 0.0.0.0 --reload --log-config scripts/uvicorn-log-config.yaml

run-celery:
	celery -A config.celery worker -B -l INFO --pool=threads --concurrency=16

run-beat:
	celery -A config.celery beat -l INFO

run-flower:
	celery -A config.celery flower --conf/flowerconfig.py

run-frontend:
	cp ./frontend/deploy/env.local.template ./frontend/.env
	cd ./frontend && yarn dev

# Docker Compose deployment

# Variables for compose command based on environment flags
# Usage examples:
#   make compose-up                              # Full application
#   make compose-up WITH_NEO4J=1                 # Full application + Neo4j
#   make compose-up WITH_DOCRAY=1                # Full application + DocRay
#   make compose-up WITH_NEO4J=1 WITH_DOCRAY=1   # Full application + Neo4j + DocRay
#   make compose-up WITH_NEO4J=1 WITH_DOCRAY=1 WITH_GPU=1  # All features
#   make compose-infra                           # Infrastructure only (databases)
#   make compose-infra WITH_NEO4J=1              # Infrastructure + Neo4j
#   make compose-down                            # Stop all services
#   make compose-down REMOVE_VOLUMES=1           # Stop and remove volumes
_PROFILES_TO_ACTIVATE :=
_EXTRA_ENVS :=
_COMPOSE_DOWN_FLAGS :=

# Determine which additional profiles to activate
ifeq ($(WITH_NEO4J),1)
    _PROFILES_TO_ACTIVATE += --profile neo4j
endif

ifeq ($(WITH_DOCRAY),1)
    ifeq ($(WITH_GPU),1)
        _PROFILES_TO_ACTIVATE += --profile docray-gpu
		_EXTRA_ENVS += DOCRAY_HOST=http://aperag-docray-gpu:8639
    else
        _PROFILES_TO_ACTIVATE += --profile docray
		_EXTRA_ENVS += DOCRAY_HOST=http://aperag-docray:8639
    endif
endif

# Determine flags for 'compose-down'
ifeq ($(REMOVE_VOLUMES),1)
    _COMPOSE_DOWN_FLAGS += -v
endif

.PHONY: compose-up compose-down compose-logs compose-infra
# Full application startup 
compose-up:
	VERSION=v0.5.0-alpha.30 DOCRAY_VERSION=v0.1.1 $(_EXTRA_ENVS) docker-compose --profile app $(_PROFILES_TO_ACTIVATE) -f docker-compose.yml up -d

# Infrastructure only (databases + supporting services)
compose-infra:
	VERSION=v0.5.0-alpha.30 DOCRAY_VERSION=v0.1.1 docker-compose $(_PROFILES_TO_ACTIVATE) -f docker-compose.yml up -d

compose-down:
	VERSION=v0.5.0-alpha.30 DOCRAY_VERSION=v0.1.1 docker-compose --profile app --profile docray --profile docray-gpu --profile neo4j -f docker-compose.yml down $(_COMPOSE_DOWN_FLAGS)

compose-logs:
	VERSION=v0.5.0-alpha.30 DOCRAY_VERSION=v0.1.1 docker-compose -f docker-compose.yml logs -f

# Environment cleanup
.PHONY: clean
clean:
	@echo "Cleaning development environment..."
	@rm -f db.sqlite3
	@echo "Use 'make compose-down REMOVE_VOLUMES=1' to clean Docker Compose services and data"

##################################################
# Developers - Code Quality and Tools
##################################################

# Development tools installation
.PHONY: dev install-hooks
dev: install-uv venv install-addlicense install-hooks
	@echo "Installing development tools..."
	@command -v redocly >/dev/null || npm install @redocly/cli -g
	@command -v openapi-generator-cli >/dev/null || npm install @openapitools/openapi-generator-cli -g
	@command -v datamodel-codegen >/dev/null || uv tool install datamodel-code-generator
	@echo ""
	@echo "✅ Development environment ready!"
	@echo "📝 Next steps:"
	@echo "   1. Activate virtual environment: source .venv/bin/activate"
	@echo "   2. Install dependencies: make install"
	@echo "   3. Start databases: make compose-infra"
	@echo "   4. Apply migrations: make migrate"
	@echo "   5. Run services: make run-backend, make run-celery"

# Code quality checks
.PHONY: format lint static-check test unit-test e2e-test
format:
	uvx ruff check --fix ./aperag ./tests
	uvx ruff format ./aperag ./tests

lint:
	uvx ruff check --no-fix ./aperag
	uvx ruff format --check ./aperag

static-check:
	uvx mypy ./aperag

test:
	uv run pytest tests/ -v

unit-test:
	uv run pytest tests/unit_test/ -v

e2e-test:
	uv run pytest --benchmark-disable tests/e2e_test/ -v

e2e-performance-test:
	@echo "Running E2E performance test..."
	@uv run pytest -v \
		--benchmark-enable \
		--benchmark-max-time=10 \
		--benchmark-min-rounds=100 \
		--benchmark-save-data \
		--benchmark-storage=tests/report \
		--benchmark-save=benchmark-result-$$(date +%Y%m%d%H%M%S) \
		tests/e2e_test/

# Evaluation
.PHONY: evaluate
evaluate:
	@echo "Running RAG evaluation..."
	@python -m aperag.evaluation.run

# Code generation
.PHONY: merge-openapi generate-models generate-frontend-sdk llm_provider
merge-openapi:
	@cd aperag && redocly bundle ./api/openapi.yaml > ./api/openapi.merged.yaml

generate-models: merge-openapi
	@datamodel-codegen \
		--input aperag/api/openapi.merged.yaml \
		--input-file-type openapi \
		--output aperag/schema/view_models.py \
		--output-model-type pydantic.BaseModel \
		--target-python-version 3.11 \
		--use-standard-collections \
		--use-schema-description \
		--enum-field-as-literal all
	@rm aperag/api/openapi.merged.yaml

generate-frontend-sdk:
	cd ./frontend && yarn api:build

llm_provider:
	python ./models/generate_model_configs.py

# Version management and licensing
.PHONY: version
version:
	@git rev-parse HEAD | cut -c1-7 > commit_id.txt
	@echo "VERSION = \"$(VERSION)\"" > $(VERSION_FILE)
	@echo "GIT_COMMIT_ID = \"$$(cat commit_id.txt)\"" >> $(VERSION_FILE)
	@rm commit_id.txt

.PHONY: add-license
add-license: install-addlicense
	./downloads/addlicense -c "ApeCloud, Inc." -y 2025 -l apache \
		-ignore "aperag/readers/**" \
		-ignore "aperag/vectorstore/**" \
		aperag/**/*.py

.PHONY: check-license
check-license: install-addlicense
	./downloads/addlicense -check \
		-c "ApeCloud, Inc." -y 2025 -l apache \
		-ignore "aperag/readers/**" \
		-ignore "aperag/vectorstore/**" \
		aperag/**/*.py

.PHONY: install-addlicense
install-addlicense:
	@mkdir -p ./downloads
	@if [ ! -f ./downloads/addlicense ]; then \
		echo "Installing addlicense..."; \
		OS=$$(uname -s); \
		ARCH=$$(uname -m); \
		case $$OS in \
			Darwin) OS=macOS ;; \
			Linux) OS=Linux ;; \
			MINGW*|CYGWIN*) OS=Windows ;; \
		esac; \
		case $$ARCH in \
			x86_64) ARCH=x86_64 ;; \
			aarch64) ARCH=arm64 ;; \
			arm64) ARCH=arm64 ;; \
		esac; \
		echo "Detected platform: $$OS/$$ARCH"; \
		if [ "$$OS" = "Windows" ]; then \
			curl -L https://github.com/google/addlicense/releases/download/v1.1.1/addlicense_1.1.1_$${OS}_$${ARCH}.zip -o /tmp/addlicense.zip; \
			unzip -j /tmp/addlicense.zip -d ./downloads; \
			rm /tmp/addlicense.zip; \
		else \
			curl -L https://github.com/google/addlicense/releases/download/v1.1.1/addlicense_1.1.1_$${OS}_$${ARCH}.tar.gz | tar -xz -C ./downloads; \
		fi; \
		chmod +x ./downloads/addlicense; \
		echo "addlicense installed to ./downloads/addlicense"; \
	fi

install-hooks:
	@echo "Installing git hooks..."
	@./scripts/install-hooks.sh

##################################################
# Build and CI/CD
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

# Image builds - multi-platform
.PHONY: build build-aperag build-aperag-frontend
build: build-aperag build-aperag-frontend

build-aperag: setup-builder version
	docker buildx build -t $(REGISTRY)/$(APERAG_IMAGE):$(VERSION) \
		--platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push \
		-f ./Dockerfile .

build-aperag-frontend: setup-builder
	cd frontend && BASE_PATH=/web/ yarn build
	cd frontend && docker buildx build \
		--platform=$(BUILDX_PLATFORM) -f Dockerfile.prod --push \
		-t $(REGISTRY)/$(APERAG_FRONTEND_IMG):$(VERSION) .

# Image builds - local platform
.PHONY: build-local build-aperag-local build-aperag-frontend-local
build-local: build-aperag-local build-aperag-frontend-local

build-aperag-local: setup-builder version
	docker buildx build -t $(APERAG_IMAGE):$(VERSION) \
		--platform $(LOCAL_PLATFORM) $(BUILDX_ARGS) --load \
		-f ./Dockerfile .

build-aperag-frontend-local: setup-builder
	cd frontend && BASE_PATH=/web/ yarn build
	cd frontend && docker buildx build \
		--platform=$(LOCAL_PLATFORM) -f Dockerfile.prod --load \
		-t $(APERAG_FRONTEND_IMG):$(VERSION) .

##################################################
# Utilities and Information
##################################################

# Configuration info
.PHONY: info
info:
	@echo "VERSION: $(VERSION)"
	@echo "BUILDX_PLATFORM: $(BUILDX_PLATFORM)"
	@echo "LOCAL_PLATFORM: $(LOCAL_PLATFORM)"
	@echo "REGISTRY: $(REGISTRY)"
	@echo "HOST ARCH: $(UNAME_M)"




.PHONY: load-images-to-minikube
load-images-to-minikube:
	@echo "Start To Load Image To Minikube"
	docker save $(APERAG_IMAGE):$(VERSION) -o aperag.tar
	minikube image load aperag.tar
	rm aperag.tar
	docker save $(APERAG_FRONTEND_IMG):$(VERSION) -o aperag-frontend.tar
	minikube image load aperag-frontend.tar
	rm aperag-frontend.tar
	@echo "Already Load Image To Minikube"

.PHONY: load-images-to-kind
load-images-to-kind:
	@echo "Start To Load Image To KinD"
	kind load docker-image $(APERAG_IMAGE):$(VERSION) --name $(KIND_CLUSTER_NAME)
	kind load docker-image $(APERAG_FRONTEND_IMG):$(VERSION) --name $(KIND_CLUSTER_NAME)
	@echo "Already Load Image To KinD"
