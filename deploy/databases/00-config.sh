#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$DATABASE_SCRIPT_DIR/scripts/common.sh"

# Namespace configuration
NAMESPACE="default"
# version
KB_VERSION="1.0.0"
ADDON_CLUSTER_CHART_VERSION="1.0.0"
# Helm repository
HELM_REPO="https://apecloud.github.io/helm-charts"

# Set to true to enable the database, false to disable
ENABLE_POSTGRESQL=true
ENABLE_REDIS=true
ENABLE_QDRANT=true
ENABLE_NEO4J=false
ENABLE_ELASTICSEARCH=true
ENABLE_MONGODB=false
