VERSION ?= v0.1.2
VERSION_FILE ?= aperag/version/__init__.py
LLMSERVER_VERSION ?= v0.1.1
BUILDX_PLATFORM ?= linux/amd64
BUILDX_ARGS ?= --sbom=false --provenance=false
REGISTRY ?= registry.cn-hangzhou.aliyuncs.com

.PHONY: version
version:
	@git rev-parse HEAD | cut -c1-7 > commit_id.txt
	@echo "VERSION = '$(VERSION)'" > $(VERSION_FILE)
	@echo "GIT_COMMIT_ID = '$$(cat commit_id.txt)'" >> $(VERSION_FILE)
	@rm commit_id.txt

.PHONY: image

# Create a new builder instance for multi-platform builds
setup-builder:
	@if ! docker buildx inspect multi-platform >/dev/null 2>&1; then \
		docker buildx create --name multi-platform --use --driver docker-container; \
	else \
		docker buildx use multi-platform; \
	fi

# Build and push multi-platform image
image: setup-builder
	docker buildx build -t $(REGISTRY)/apecloud/aperag:$(VERSION) --platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push -f ./Dockerfile .
	cd web && docker buildx build -t $(REGISTRY)/apecloud/aperag-frontend:$(VERSION) --platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push -f ./Dockerfile .

# Clean up builder instance
clean-builder:
	docker buildx rm multi-platform

diff:
	@python manage.py diffsettings

migrate:
	@python manage.py makemigrations
	@python manage.py migrate

run-redis:
	@echo "Starting redis"
	@docker inspect aperag-redis > /dev/null 2>&1 || docker run -d --name aperag-redis -p 6379:6379 redis:latest
	@docker start aperag-redis

run-postgres:
	@echo "Starting postgres"
	@docker inspect aperag-postgres > /dev/null 2>&1 || docker run -d --name aperag-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres
	@docker start aperag-postgres

run-qdrant:
	@echo "Starting qdrant"
	@docker inspect aperag-qdrant > /dev/null 2>&1 || docker run -d --name aperag-qdrant -p 6333:6333 qdrant/qdrant
	@docker start aperag-qdrant

run-es:
	@echo "Starting Elasticsearch (development mode)"
	@docker inspect aperag-es > /dev/null 2>&1 || \
	docker run -d \
		--name aperag-es \
		-p 9200:9200 \
		-e discovery.type=single-node \
		-e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
		-e xpack.security.enabled=false \
		-v esdata:/usr/share/elasticsearch/data \
		apecloud/elasticsearch:8.8.2
	@docker start aperag-es || true
	@echo "Checking if IK Analyzer is installed..."
	@docker exec aperag-es bash -c \
		"if [ ! -d plugins/analysis-ik ]; then \
			echo 'Installing IK Analyzer from get.infini.cloud...'; \
			bin/elasticsearch-plugin install -b https://get.infini.cloud/elasticsearch/analysis-ik/8.8.2; \
			echo 'Restarting Elasticsearch to apply changes...'; \
		else \
			echo 'IK Analyzer is already installed.'; \
		fi"
	@docker restart aperag-es > /dev/null
	@echo "Elasticsearch is ready with IK Analyzer!"

run-db: run-redis run-postgres run-qdrant run-es

connect-metadb:
	@docker exec -it aperag-postgres psql -p 5432 -U postgres

clean:
	@echo "Removing db.sqlite3"
	@/bin/rm -f db.sqlite3
	@echo "Removing container aperag-postgres"
	@docker rm -fv aperag-postgres > /dev/null 2>&1 || true
	@echo "Removing container aperag-redis"
	@docker rm -fv aperag-redis > /dev/null 2>&1 || true
	@echo "Removing container aperag-qdrant"
	@docker rm -fv aperag-qdrant > /dev/null 2>&1 || true
	@echo "Removing container aperag-es"
	@docker rm -fv aperag-es > /dev/null 2>&1 || true

run-frontend:
	cp ./web/deploy/env.local.template ./web/.env
	cd ./web && yarn install && yarn dev

run-backend: migrate
	python manage.py collectstatic --noinput
	uvicorn config.asgi:application --host 0.0.0.0 --reload --reload-include '*.html'

compose-up: migrate
	docker-compose -f compose.yml up -d

compose-down:
	docker-compose -f compose.yml down

compose-logs:
	docker-compose -f compose.yml logs -f

format:
	poetry run ruff --preview --fix .

lint:
	poetry run ruff --preview .

static-check:
	poetry run mypy .

test:
	echo "mock"

celery:
	celery -A config.celery worker -B -l INFO

flower:
	celery -A config.celery flower --conf/flowerconfig.py

poetry-lock-no-update:
	poetry lock --no-update

addlicense:
	@echo "Adding license headers..."
	addlicense -c "ApeCloud, Inc." -y 2025 -l apache \
	  -ignore "**/*.md" \
	  -ignore "**/*.sh" \
	  -ignore "**/*.yml" \
	  -ignore "**/*.yaml" \
	  -ignore "**/*.toml" \
	  -ignore "**/Makefile" \
	  -ignore "**/Dockerfile" \
	  -ignore "**/compose.yml" \
	  -ignore "**/__pycache__/**" \
	  -ignore "**/migrations/**" \
	  -ignore "**/templates/**" \
	  -ignore "aperag/readers/**" \
	  -ignore "aperag/vectorstore/**" \
	  .