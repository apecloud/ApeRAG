# ApeRAG

ApeRAG is a powerful RAG system that deeply analyzes documents and multimedia content while building vector indexes in parallel and measuring retrieval quality. It streamlines workflows with integrated LLM management, data source handling, automatic syncing, and seamless office tool compatibility.

# Development Guide

You should install Python 3.11 first.

* install poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry self add poetry-plugin-shell
```

* install dependencies

```bash
poetry env use 3.11
poetry lock
poetry install
```

* prepare configs

```bash
cp envs/.env.template .env
```

* prepare postgres/redis/qdrant/elasticsearch

```bash
make run-db
```

* run the django service

```bash
poetry shell
```

```
make run-backend
```

* run the celery service

```bash
poetry shell
```

```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python celery -A config.celery worker -l INFO --pool threads
```

* run the frontend

```
make run-frontend
```


then open the aperag frontend console: http://localhost:8001
