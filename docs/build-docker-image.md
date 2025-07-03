# Build Guide

[阅读中文文档](build-docker-image-zh.md)

This section covers how to build ApeRAG container images. It's primarily for users who need to create their own builds or deploy to environments other than the ones covered in "Getting Started".

## Building Container Images

The project uses Docker and `make` commands to build container images.

*   **Local Platform Builds**:
    These commands build images for your current machine's architecture.
    ```bash
    # Build all necessary images for local platform
    make build-local

    # Build only the backend image for local platform
    make build-aperag-local

    # Build only the frontend image for local platform
    make build-aperag-frontend-local
    ```

*   **Multi-platform Builds**:
    These commands build images for multiple architectures (e.g., amd64, arm64). This requires Docker Buildx to be set up and configured.
    ```bash
    # Build all necessary images for multiple platforms
    make build

    # Build only the backend image for multiple platforms
    make build-aperag

    # Build only the frontend image for multiple platforms
    make build-aperag-frontend
    ```
    You can specify the target platforms using the `PLATFORMS` variable, for example:
    ```bash
    make build PLATFORMS=linux/amd64,linux/arm64
    ```

## Deployment

Refer to the "Getting Started" section in the main README for common deployment methods:
*   [Getting Started with Kubernetes](../README.md#getting-started-with-kubernetes)
*   [Getting Started with Docker Compose](../README.md#getting-started-with-docker-compose)

For custom deployments, you will need to adapt these methods or use the built container images with your chosen orchestration platform. Ensure all required services (databases, backend, frontend, Celery workers) are correctly configured and can communicate with each other. 