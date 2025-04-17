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
   cp envs/docker.env.template .docker.env
   cp web/deploy/env.local.template web/.env
   ```

2. (Optional) use aliyun image registry if you are in China:
   ```bash
   export REGISTRY=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com
   ```

3. Start the services:
   ```bash
   docker compose up -d
   ```

4. Access the services: http://localhost:80

## License

[Your License Here]

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
poetry install --with model
```

* prepare configs

```bash
cp envs/env.template .env
```

* prepare postgres/redis/qdrant/elasticsearch

```bash
make run-db
```

* optional: enable MinerU

   ApeRAG supports parsing PDF and Microsoft Office documents using MinerU. However, this feature requires pre-downloaded models for document layout detection, as well as a magic_pdf.json configuration file. Therefore, it is disabled by default.

   To enable MinerU, run the following script. It will automatically download the necessary models and generate the configuration file for you:

   ```bash
   pip install huggingface_hub
   python ./scripts/prepare_for_mineru.py
   ```

   You can further customize the `magic_pdf.json` file to configure LLM-related settings under the `llm-aided-config` section. Be sure to set the `enable` field to `true`. When enabled, MinerU will leverage an LLM to enhance parsing capabilities, for example, correcting OCR errors in text and formulas, and adjusting title levels (by default, all titles are set to level 1).

* run the django service

```bash
poetry shell
```

```
make run-backend
```

To debug django service, see [HOW-TO-DEBUG.md](docs%2FHOW-TO-DEBUG.md)

* run the celery service

```bash
poetry shell
```

```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python celery -A config.celery worker -l INFO --pool threads
```

To debug celery service, see [HOW-TO-DEBUG.md](docs%2FHOW-TO-DEBUG.md)

* run the frontend

```
make run-frontend
```

then open the aperag frontend console: http://localhost:8001
