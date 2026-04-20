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

install_addon_release() {
    local addon_name="$1"
    local release_name="$2"
    local chart_name="$3"
    local attempt_output=""

    print "Installing ${addon_name} addon..."

    if helm status "$release_name" --namespace kb-system >/dev/null 2>&1; then
        print_success "${addon_name} addon Helm release already exists."
        return 0
    fi

    if attempt_output="$(helm upgrade --install "$release_name" "kubeblocks/${chart_name}" --namespace kb-system --version "$ADDON_CLUSTER_CHART_VERSION" 2>&1)"; then
        echo "$attempt_output"
        return 0
    fi

    echo "$attempt_output" >&2

    if grep -q "release: already exists" <<<"$attempt_output"; then
        print_warning "${addon_name} addon release appeared during install; waiting for Helm state to settle..."
        for _ in {1..12}; do
            if helm status "$release_name" --namespace kb-system >/dev/null 2>&1; then
                print_success "${addon_name} addon Helm release is available after the concurrent install race."
                return 0
            fi
            sleep 5
        done
    fi

    return 1
}

wait_for_addon_enabled() {
    local addon_resource="$1"
    local addon_name="$2"
    local phase=""

    print "Waiting for ${addon_name} addon to reach Enabled phase..."
    for _ in {1..36}; do
        phase="$(kubectl get addons.extensions.kubeblocks.io "$addon_resource" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
        if [ "$phase" = "Enabled" ]; then
            print_success "${addon_name} addon is Enabled."
            return 0
        fi
        sleep 5
    done

    print_error "${addon_name} addon did not reach Enabled phase. Last observed phase: ${phase:-<missing>}"
    kubectl get addons.extensions.kubeblocks.io -n "$NAMESPACE" || true
    return 1
}

# Install database addons based on configuration
if [ "$ENABLE_POSTGRESQL" = true ]; then
    install_addon_release "PostgreSQL" "kb-addon-postgresql" "postgresql"
    wait_for_addon_enabled "postgresql" "PostgreSQL"
fi

if [ "$ENABLE_REDIS" = true ]; then
    install_addon_release "Redis" "kb-addon-redis" "redis"
    wait_for_addon_enabled "redis" "Redis"
fi

if [ "$ENABLE_ELASTICSEARCH" = true ]; then
    install_addon_release "Elasticsearch" "kb-addon-elasticsearch" "elasticsearch"
    wait_for_addon_enabled "elasticsearch" "Elasticsearch"
fi

if [ "$ENABLE_QDRANT" = true ]; then
    install_addon_release "Qdrant" "kb-addon-qdrant" "qdrant"
    wait_for_addon_enabled "qdrant" "Qdrant"
fi

if [ "$ENABLE_MONGODB" = true ]; then
    install_addon_release "MongoDB" "kb-addon-mongodb" "mongodb"
    wait_for_addon_enabled "mongodb" "MongoDB"
fi

if [ "$ENABLE_NEO4J" = true ]; then
    install_addon_release "Neo4j" "kb-addon-neo4j" "neo4j"
    wait_for_addon_enabled "neo4j" "Neo4j"
fi

print_success "KubeBlocks database addons installation completed!"
print "Now you can run 02-install-database.sh to install database clusters"
