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

## Graph Backend Values

The default graph backend is PostgreSQL (`api.env.GRAPH_DB_TYPE=postgresql`).
For external graph stores, set the deployment default and enable the matching
first-class dependency block:

```bash
# Neo4j
helm upgrade -i aperag ./deploy/aperag \
  --set api.env.GRAPH_DB_TYPE=neo4j \
  --set neo4j.enabled=true \
  --set neo4j.NEO4J_URI=bolt://neo4j-cluster-neo4j:7687 \
  --set neo4j.NEO4J_CREDENTIALS_SECRET_NAME=neo4j-cluster-neo4j-account-neo4j

# NebulaGraph
helm upgrade -i aperag ./deploy/aperag \
  --set api.env.GRAPH_DB_TYPE=nebula \
  --set nebula.enabled=true \
  --set nebula.NEBULA_HOSTS=nebula-cluster-graphd:9669 \
  --set nebula.NEBULA_CREDENTIALS_SECRET_NAME=nebula-cluster-account-root
```

When `*.CREDENTIALS_SECRET_NAME` is set, the API and indexing-worker
Deployments read usernames and passwords from Kubernetes Secret keys
`username` and `password`. The same graph backend values are injected into both
Deployments so read paths and indexing write paths use the same backend.
