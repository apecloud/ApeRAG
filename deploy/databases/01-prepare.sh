#!/bin/bash

set -euo pipefail

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

check_dependencies

# Check if KubeBlocks is already installed, install it if it is not.
source "$DATABASE_SCRIPT_DIR/install-kubeblocks.sh"

# Create namespaces
print "Creating namespaces..."
kubectl create namespace $NAMESPACE 2>/dev/null || true

# Install database addons
print "Installing KubeBlocks database addons..."

# Add and update Helm repository
print "Adding and updating KubeBlocks Helm repository..."
helm repo add kubeblocks $HELM_REPO
helm repo update
# Install database addons based on configuration
if [ "$ENABLE_POSTGRESQL" = true ]; then
    print "Installing PostgreSQL addon..."
    helm upgrade --install kb-addon-postgresql kubeblocks/postgresql --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
fi

if [ "$ENABLE_REDIS" = true ]; then
    print "Installing Redis addon..."
    helm upgrade --install kb-addon-redis kubeblocks/redis --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
fi

if [ "$ENABLE_ELASTICSEARCH" = true ]; then
    print "Installing Elasticsearch addon..."
    helm upgrade --install kb-addon-elasticsearch kubeblocks/elasticsearch --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
fi

if [ "$ENABLE_QDRANT" = true ]; then
    print "Installing Qdrant addon..."
    helm upgrade --install kb-addon-qdrant kubeblocks/qdrant --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
fi

if [ "$ENABLE_MONGODB" = true ]; then
    print "Installing MongoDB addon..."
    helm upgrade --install kb-addon-mongodb kubeblocks/mongodb --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
fi

if [ "$ENABLE_NEO4J" = true ]; then
    print "Installing Neo4j addon..."
    helm upgrade --install kb-addon-neo4j kubeblocks/neo4j --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
fi

print_success "KubeBlocks database addons installation completed!"
print "Now you can run 02-install-database.sh to install database clusters"
