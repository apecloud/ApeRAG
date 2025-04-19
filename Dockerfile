# Build stage for dependencies
FROM python:3.11.1-slim AS builder

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential git curl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
RUN /root/.local/bin/uv venv --python 3.11 && \
    /root/.local/bin/uv sync

# Final stage
FROM python:3.11.1-slim

# Define MinerU dependencies
ARG MINERU_DEPS="libglib2.0-0 libgl1"

# Install minimal system dependencies
RUN apt update && \
    apt install --no-install-recommends -y curl ${MINERU_DEPS} && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary files from builder
COPY --from=builder /app/.venv/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/.venv/bin /usr/local/bin

# Copy application code
COPY . /app

WORKDIR /app

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
