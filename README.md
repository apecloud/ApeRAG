# ApeRAG

ApeRAG is a powerful RAG system that deeply analyzes documents and multimedia content while building vector indexes in parallel and measuring retrieval quality. It streamlines workflows with integrated LLM management, data source handling, automatic syncing, and seamless office tool compatibility.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### 1. Environment Setup

Configure environment variables:
```bash
cp envs/env.template .env
cp frontend/deploy/env.local.template frontend/.env
```

**Edit the `.env` file to configure your AI service settings.**

### 2. Start Services

```bash
# Optional: Use Aliyun registry if in China
export REGISTRY=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com

# Start ApeRAG services
make compose-up

# Alternatively, start ApeRAG with the doc-ray parsing service
make compose-up WITH_DOCRAY=1

# Additionally, if your environment has GPUs, you can also enable GPU support by adding `WITH_GPU=1` to the command:
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

> **About the doc-ray parsing service**
>
> ApeRAG includes a basic built-in parser for extracting text from documents like PDFs and DOCX files for RAG indexing. However, this parser may not optimally handle complex document structures, tables, or formulas.
>
> For enhanced document parsing capabilities and more accurate content extraction, we recommend deploying the [doc-ray](https://github.com/apecloud/doc-ray) service. `doc-ray` leverages **MinerU** for advanced document analysis.
>
> * When `WITH_GPU=1` is not specified, `doc-ray` will run using only the CPU. In this case, it is recommended to allocate at least 4 CPU cores and 8GB+ of RAM for it.
> * When `WITH_GPU=1` is specified, `doc-ray` will run using the GPU. It requires approximately 6GB of VRAM, along with 2 CPU cores and 8GB of RAM.

### 3. Access

Open your browser: http://localhost:3000/web/

## Development

### Environment Setup

Install dependencies:
```bash
make install
```

Configure environment:
```bash
cp envs/env.template .env
```

### Database Services

Start all required databases:
```bash
make run-db
```

### Backend Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Django backend
make run-backend

# In another terminal, run Celery worker
make run-celery
```

### Frontend Development

```bash
# Node.js >= 20 required
make run-frontend
```

Access frontend at: http://localhost:3000

### Development Tools

```bash
# Install development tools
make dev

# Code formatting
make format

# Code linting
make lint

# Type checking
make static-check

# Run tests
make test

# Generate LLM models configs
make generate-models

# Generate frontend SDK
make generate-frontend-sdk
```

### Database Management

```bash
# Create migrations
make makemigration

# Apply migrations
make migrate

# Connect to database
make connect-metadb

# View Django settings differences
make diff
```

### Docker Compose Operations

```bash
# Start services
make compose-up

# Stop services
make compose-down

# View logs
make compose-logs
```

### Cleanup

```bash
# Clean development environment
make clean
```

## Building and Deployment

### Local Builds

```bash
# Build for local platform
make build-local

# Build specific components
make build-aperag-local
make build-aperag-frontend-local
```

### Multi-platform Builds

```bash
# Build for multiple platforms (requires Docker Buildx)
make build

# Build specific components
make build-aperag
make build-aperag-frontend
```