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

2. (Optional) Enable MinerU for parsing PDF & MS-Office documents.

   MinerU is an excellent tool for parsing documents. However, using it requires downloading specific models beforehand. Additionally, it demands more computing resources and has a longer parsing time. You can decide whether to enable MinerU based on your specific requirements.

   To enable it, run the following command:
   ```bash
   pip install huggingface_hub
   python ./scripts/prepare_for_mineru.py
   ```

3. (Optional) use aliyun image registry if you are in China:
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
