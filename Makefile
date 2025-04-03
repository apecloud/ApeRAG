VERSION ?= v0.1.2
VERSION_FILE ?= aperag/version/__init__.py
LLMSERVER_VERSION ?= v0.1.1
BUILDX_PLATFORM ?= linux/amd64
BUILDX_ARGS ?= --sbom=false --provenance=false
REGISTRY ?= registry.cn-hangzhou.aliyuncs.com
FRONTEND_BRANCH ?= main
FRONTEND_REPO_DIR ?= /tmp/ApeRAG-FrontEnd

init:
	cp envs/.env.template .env
	cp envs/.env.frontend.template .env.frontend

.PHONY: version
version:
	@git rev-parse HEAD | cut -c1-7 > commit_id.txt
	@echo "VERSION = '$(VERSION)'" > $(VERSION_FILE)
	@echo "GIT_COMMIT_ID = '$$(cat commit_id.txt)'" >> $(VERSION_FILE)
	@rm commit_id.txt

build-requirements:
	sh scripts/export-requirements.sh

image: build-requirements version
	docker buildx build -t $(REGISTRY)/apecloud/aperag:$(VERSION) --platform $(BUILDX_PLATFORM) $(BUILDX_ARGS) --push -f ./Dockerfile  .

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
	@echo "Starting elasticsearch"
	@docker inspect aperag-es > /dev/null 2>&1 || docker run -d --name aperag-es -p 9200:9200 apecloud/elasticsearch:8.8.2
	@docker start aperag-es

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

frontend.%:
	if cd $(FRONTEND_REPO_DIR) > /dev/null 2>&1; then \
		cd $(FRONTEND_REPO_DIR) && rm .env && git pull origin $(FRONTEND_BRANCH) && git checkout $(FRONTEND_BRANCH); \
	else \
		git clone https://github.com/apecloud/ApeRAG-FrontEnd.git $(FRONTEND_REPO_DIR); \
	fi
	cp .env.$@ $(FRONTEND_REPO_DIR)/.env
	mkdir -p /tmp/aperag-static-files/
	mkdir -p static/web/
	cd $(FRONTEND_REPO_DIR) && yarn install && yarn build
	cp -r /tmp/aperag-static-files/* static/web/
	if [ -f "static/web/index.html" ]; then \
  		cp static/web/index.html aperag/templates/404.html; \
 	fi

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
