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

### 2. Optional: Enhanced Document Parsing

For improved PDF and MS Office document parsing, set up [doc-ray](https://github.com/apecloud/doc-ray):

```bash
# Deploy doc-ray (GPU recommended for optimal performance)
docker run -d -p 8639:8639 -p 8265:8265 --gpus=all --name doc-ray apecloud/doc-ray
```

Then update your `.env` file:
```bash
DOCRAY_HOST=http://localhost:8639
```

### 3. Start Services

```bash
# Optional: Use Aliyun registry if in China
export REGISTRY=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com

# Start all services
make compose-up
```

### 4. Access

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