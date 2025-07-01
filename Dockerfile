# Build stage for dependencies
FROM python:3.11.13-slim AS builder

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install system dependencies and uv in one layer, then create venv and install deps
RUN apt update && \
    apt install --no-install-recommends -y build-essential git curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv /opt/venv --python 3.11 && \
    . /opt/venv/bin/activate && \
    /root/.local/bin/uv sync --active && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

# Final stage
FROM python:3.11.13-slim

# Install minimal system dependencies in one layer and clean up
RUN apt update && \
    apt install --no-install-recommends -y curl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*

# Copy the entire virtual environment
COPY --from=builder /opt/venv /opt/venv

# Make sure we use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . /app

WORKDIR /app

# Install only the package structure without dependencies (dependencies already installed by uv)
RUN . /opt/venv/bin/activate && pip install --no-deps -e . && \
    rm -rf /root/.cache /tmp/*

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
