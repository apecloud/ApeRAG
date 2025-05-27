# ApeRAG

ApeRAG is a powerful RAG system that deeply analyzes documents and multimedia content while building vector indexes in parallel and measuring retrieval quality. It streamlines workflows with integrated LLM management, data source handling, automatic syncing, and seamless office tool compatibility.

## User Guide

### Prerequisites

- Docker
- Docker Compose
- Git

### Quick Start

1. Configure environment variables:
   ```bash
   cp envs/env.template .env
   cp frontend/deploy/env.local.template frontend/.env
   ```

   **Then edit the `.env` file to configure your AI service settings.**

2. (Optional) Set up [doc-ray](https://github.com/apecloud/doc-ray) for improved (but slower) parsing of PDF and MS Office documents.

   * Refer to the deployment instructions within the doc-ray project. **Using a GPU is highly recommended for optimal performance.**
   * In your `.env` file, update the `DOCRAY_HOST` variable to your doc-ray service endpoint, for example: `DOCRAY_HOST=http://<your-doc-ray-server-host>:8639`.

3. (Optional) Use Aliyun image registry if you are in China:
   ```bash
   export REGISTRY=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com
   ```

4. Start the services:
   ```bash
   docker compose up --build -d
   ```

5. Access the services: http://localhost:3000/web/

## License

[Your License Here]

# Development Guide

* install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

* install dependencies

```bash
uv venv -p 3.11
uv sync --all-groups --all-extras
```

* prepare configs

```bash
cp envs/env.template .env
```

* prepare postgres/redis/qdrant/elasticsearch

```bash
make run-db
```

* run the django service

```bash
source .venv/bin/activate
```

```
make run-backend
```

To debug django service, see [HOW-TO-DEBUG.md](docs%2FHOW-TO-DEBUG.md)

* run the celery service

```bash
source .venv/bin/activate
```

```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python celery -A config.celery worker -l INFO --pool threads
```

To debug celery service, see [HOW-TO-DEBUG.md](docs%2FHOW-TO-DEBUG.md)

* run the frontend

```
# The node engine version should be ">=20"
make run-frontend
```

then open the aperag frontend console: http://localhost:3000
