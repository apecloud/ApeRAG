# Configuration variables
VERSION ?= dev-latest
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
		uv venv -p 3.11; \
	fi

install: venv
	@echo "Installing Python dependencies..."
	uv sync --all-groups --all-extras

# Database management
.PHONY: makemigration migrate diff
makemigration:
	@python manage.py makemigrations

migrate:
	@python manage.py migrate aperag
	@python manage.py migrate

diff:
	@python manage.py diffsettings

# Local services
.PHONY: run-backend run-frontend run-db run-celery run-flower
run-backend: migrate
	python manage.py collectstatic --noinput
	uvicorn config.asgi:application --host 0.0.0.0 --reload --reload-include '*.html'

run-celery:
	celery -A config.celery worker -B -l INFO

run-flower:
	celery -A config.celery flower --conf/flowerconfig.py

run-frontend:
	cp ./frontend/deploy/env.local.template ./frontend/.env
	cd ./frontend && yarn dev

run-db:
	@echo "Starting all database services..."
	@$(MAKE) run-redis run-postgres run-qdrant run-es run-minio

# Docker Compose deployment
.PHONY: compose-up compose-down compose-logs
compose-up: migrate
	docker-compose -f compose.yml up -d

compose-down:
	docker-compose -f compose.yml down

compose-logs:
	docker-compose -f compose.yml logs -f

# Environment cleanup
.PHONY: clean
clean:
	@echo "Cleaning development environment..."
	@rm -f db.sqlite3
	@docker rm -fv aperag-postgres-dev aperag-redis-dev aperag-qdrant-dev aperag-es-dev aperag-minio-dev 2>/dev/null || true

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

performance-test:
	uv run pytest --benchmark-enable --benchmark-max-time 10 --benchmark-min-rounds 100 tests/e2e_test/ -v

# Code generation
.PHONY: merge-openapi generate-models generate-frontend-sdk generate_model_configs
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

generate_model_configs:
	python ./scripts/generate_model_configs.py

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
.PHONY: run-redis run-postgres run-qdrant run-es run-minio
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