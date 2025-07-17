# ApeRAG Helm Chart

This Helm chart deploys ApeRAG application on Kubernetes.

## Default Configuration

By default, this chart uses images from Docker Container Registry:

- Backend: `docker.io/apecloud/aperag:latest`
- Frontend: `docker.io/apecloud/aperag-frontend:latest`

## Installation

```bash
# Install the chart
helm install aperag ./deploy/aperag

# Or with custom values
helm install aperag ./deploy/aperag \
  --set image.tag=v0.0.0-nightly \
  --set frontend.image.tag=v0.0.0-nightly
```

## Environment Variables

All environment variables are managed through the `aperag-env` Secret. See `aperag-secret.yaml` template for configuration options.
