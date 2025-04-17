# Build stage for dlptool
FROM curlimages/curl:8.4.0 AS downloader
RUN curl -sL http://kubeblocks.oss-cn-hangzhou.aliyuncs.com/dlptool -o /tmp/dlptool

# Build stage for dependencies
FROM python:3.11.1-slim AS builder

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential git curl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /root/.local/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

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
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=downloader /tmp/dlptool /bin/dlptool
RUN chmod +x /bin/dlptool

# Copy application code
COPY . /app

WORKDIR /app

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
