# Configuration variables
VERSION ?= v0.5.0-alpha.29
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
.PHONY: run-backend run-frontend run-db run-celery run-flower
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

run-db:
	@echo "Starting all database services..."
	@$(MAKE) run-redis run-postgres run-qdrant run-es run-minio

# Docker Compose deployment

# Variables for compose command based on environment flags
# Usage examples:
#   make compose-up
#   make compose-up WITH_DOCRAY=1
#   make compose-up WITH_DOCRAY=1 WITH_GPU=1
#   make compose-down
#   make compose-down REMOVE_VOLUMES=1
_PROFILES_TO_ACTIVATE :=
_EXTRA_ENVS :=
_COMPOSE_DOWN_FLAGS :=

# Determine which docray profile to use for 'compose-up'
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

.PHONY: compose-up compose-down compose-logs
compose-up:
	$(_EXTRA_ENVS) docker-compose $(_PROFILES_TO_ACTIVATE) -f docker-compose.yml up -d

compose-down:
	docker-compose --profile docray --profile docray-gpu -f docker-compose.yml down $(_COMPOSE_DOWN_FLAGS)

compose-logs:
	docker-compose -f docker-compose.yml logs -f

# Environment cleanup
.PHONY: clean
clean:
	@echo "Cleaning development environment..."
	@rm -f db.sqlite3
	@docker rm -fv aperag-postgres-dev aperag-redis-dev aperag-qdrant-dev aperag-es-dev aperag-minio-dev aperag-neo4j-dev 2>/dev/null || true
	@if [ -f "nebula-docker-compose.yml" ]; then \
		echo "Stopping NebulaGraph containers..."; \
		docker-compose -f nebula-docker-compose.yml down 2>/dev/null || true; \
	fi

##################################################
# Developers - Code Quality and Tools
##################################################

# Development tools installation
.PHONY: dev install-hooks
dev: install-uv install-addlicense install-hooks
	@echo "Installing development tools..."
	@command -v redocly >/dev/null || npm install @redocly/cli -g
	@command -v openapi-generator-cli >/dev/null || npm install @openapitools/openapi-generator-cli -g
	@command -v datamodel-codegen >/dev/null || uv tool install datamodel-code-generator

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

# Database connection tools
.PHONY: connect-metadb
connect-metadb:
	@docker exec -it aperag-postgres-dev psql -p 5432 -U postgres

# Individual service startup (for advanced users)
.PHONY: run-redis run-postgres run-qdrant run-es run-minio run-neo4j run-nebula stop-nebula
run-redis:
	@docker inspect aperag-redis-dev >/dev/null 2>&1 || docker run -d --name aperag-redis-dev -p 6379:6379 redis:latest
	@docker start aperag-redis-dev

run-postgres:
	@docker inspect aperag-postgres-dev >/dev/null 2>&1 || \
		docker run -d --name aperag-postgres-dev -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
	@docker start aperag-postgres-dev
	@sleep 3
	@docker exec aperag-postgres-dev psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

run-qdrant:
	@docker inspect aperag-qdrant-dev >/dev/null 2>&1 || docker run -d --name aperag-qdrant-dev -p 6333:6333 qdrant/qdrant
	@docker start aperag-qdrant-dev

run-es:
	@echo "Starting Elasticsearch (development mode)"
	@docker inspect aperag-es-dev > /dev/null 2>&1 || \
	docker run -d \
		--name aperag-es-dev \
		-p 9200:9200 \
		-e discovery.type=single-node \
		-e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
		-e xpack.security.enabled=false \
		-v esdata:/usr/share/elasticsearch/data \
		apecloud/elasticsearch:8.8.2
	@docker start aperag-es-dev || true
	@echo "Checking if IK Analyzer is installed..."
	@docker exec aperag-es-dev bash -c \
		"if [ ! -d plugins/analysis-ik ]; then \
			echo 'Installing IK Analyzer from get.infini.cloud...'; \
			curl -L --output /tmp/analysis-ik.zip https://get.infini.cloud/elasticsearch/analysis-ik/8.8.2; \
			echo 'y' | bin/elasticsearch-plugin install file:///tmp/analysis-ik.zip; \
			echo 'Restarting Elasticsearch to apply changes...'; \
		else \
			echo 'IK Analyzer is already installed.'; \
		fi"
	@docker restart aperag-es-dev > /dev/null
	@echo "Elasticsearch is ready with IK Analyzer!"

run-minio:
	@docker inspect aperag-minio-dev >/dev/null 2>&1 || \
		docker run -d --name aperag-minio-dev -p 9000:9000 -p 9001:9001 \
		quay.io/minio/minio server /data --console-address ":9001"
	@docker start aperag-minio-dev

run-neo4j:
	@docker inspect aperag-neo4j-dev >/dev/null 2>&1 || \
		docker run -d --name aperag-neo4j-dev -p 7474:7474 -p 7687:7687 \
		-e NEO4J_AUTH=neo4j/password \
		-e NEO4J_PLUGINS=\[\"apoc\"\] \
		-e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
		-e NEO4J_apoc_export_file_enabled=true \
		neo4j:5.26.5-enterprise
	@docker start aperag-neo4j-dev

run-nebula:
	@echo "Setting up NebulaGraph with docker-compose (no persistence)..."
	@TZ=UTC docker-compose -f nebula-docker-compose.yml up -d
	@echo "NebulaGraph is starting up..."
	@echo ""
	@echo "✅ Graph service available at: localhost:9669"
	@echo ""
	@echo "🌐 Studio Web UI: http://localhost:7001"
	@echo "   📝 Connection Info:"
	@echo "   • Graphd IP address: graphd"
	@echo "   • Port: 9669"
	@echo "   • Username: root"
	@echo "   • Password: nebula (or any password)"
	@echo ""
	@echo "💻 Console: docker run --rm -ti --network host vesoft/nebula-console:nightly -addr 127.0.0.1 -port 9669 -u root -p nebula"
	@echo ""
	@echo "🔍 Check status: docker-compose -f nebula-docker-compose.yml ps"
	@echo "🛑 Stop: make stop-nebula"

stop-nebula:
	@echo "Stopping NebulaGraph..."
	@docker-compose -f nebula-docker-compose.yml down
	@echo "NebulaGraph stopped."


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

# Compatibility aliases
.PHONY: image celery flower
image: build
celery: run-celery
flower: run-flower
