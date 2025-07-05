#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$DATABASE_SCRIPT_DIR/scripts/common.sh"

# Namespace configuration
NAMESPACE="default"
# version
KB_VERSION="0.9.4"

# Helm repository
KUBEBLOCKS_ADDONS_HELM_REPO="https://jihulab.com/api/v4/projects/150246/packages/helm/stable"

# Set to true to enable the database, false to disable
ENABLE_POSTGRESQL=true
ENABLE_REDIS=true
ENABLE_QDRANT=true
ENABLE_ELASTICSEARCH=true
ENABLE_MONGODB=false
