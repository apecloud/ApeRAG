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

helm_release_status() {
    local release_name="$1"

    helm status "$release_name" --namespace kb-system 2>/dev/null \
        | awk '/^STATUS: / {print $2; exit}' || true
}

run_addon_release_install() {
    local release_name="$1"
    local chart_name="$2"

    helm upgrade --install "$release_name" "kubeblocks/${chart_name}" \
        --namespace kb-system \
        --version "$ADDON_CLUSTER_CHART_VERSION"
}

install_addon_release() {
    local addon_name="$1"
    local release_name="$2"
    local chart_name="$3"
    local addon_resource="$4"
    local attempt_output=""
    local retry_output=""
    local release_status=""

    print "Installing ${addon_name} addon..."

    release_status="$(helm_release_status "$release_name")"
    if [ "$release_status" = "deployed" ]; then
        print_success "${addon_name} addon Helm release already exists."
        return 0
    fi
    if [ -n "$release_status" ]; then
        print_warning "${addon_name} addon Helm release exists in ${release_status} state; reconciling it."
    fi

    if attempt_output="$(run_addon_release_install "$release_name" "$chart_name" 2>&1)"; then
        echo "$attempt_output"
        return 0
    fi

    echo "$attempt_output" >&2

    if grep -q "already exists" <<<"$attempt_output"; then
        print_warning "${addon_name} addon install hit an already-exists race; waiting for addon state to settle..."
        for _ in {1..12}; do
            release_status="$(helm_release_status "$release_name")"
            if [ "$release_status" = "deployed" ]; then
                print_success "${addon_name} addon Helm release recovered to deployed after the concurrent install race."
                return 0
            fi
            if [ -n "$release_status" ] || \
               kubectl get addons.extensions.kubeblocks.io "$addon_resource" -n "$NAMESPACE" >/dev/null 2>&1; then
                print_warning "${addon_name} addon state is available after the concurrent install race (release=${release_status:-<missing>}); reconciling once."
                if retry_output="$(run_addon_release_install "$release_name" "$chart_name" 2>&1)"; then
                    echo "$retry_output"
                    return 0
                fi
                echo "$retry_output" >&2
                break
            fi
            sleep 5
        done
    fi

    return 1
}

wait_for_addon_enabled() {
    local addon_resource="$1"
    local addon_name="$2"
    local release_name="$3"
    local phase=""
    local release_status=""

    print "Waiting for ${addon_name} addon to reach Enabled phase..."
    for _ in {1..36}; do
        phase="$(kubectl get addons.extensions.kubeblocks.io "$addon_resource" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
        if [ "$phase" = "Enabled" ]; then
            print_success "${addon_name} addon is Enabled."
            return 0
        fi
        if [ "$phase" = "Failed" ]; then
            release_status="$(helm_release_status "$release_name")"
            if [ "$release_status" = "deployed" ]; then
                print_warning "${addon_name} addon phase stayed Failed after an install race, but Helm release ${release_name} is deployed; continuing."
                return 0
            fi
        fi
        if [ -z "$phase" ]; then
            release_status="$(helm_release_status "$release_name")"
            if [ "$release_status" = "failed" ]; then
                print_error "${addon_name} addon Helm release ${release_name} is failed and addon resource is still missing."
                helm status "$release_name" --namespace kb-system || true
                return 1
            fi
        fi
        sleep 5
    done

    print_error "${addon_name} addon did not reach Enabled phase. Last observed phase: ${phase:-<missing>}"
    kubectl get addons.extensions.kubeblocks.io -n "$NAMESPACE" || true
    return 1
}

# Install database addons based on configuration
if [ "$ENABLE_POSTGRESQL" = true ]; then
    install_addon_release "PostgreSQL" "kb-addon-postgresql" "postgresql" "postgresql"
    wait_for_addon_enabled "postgresql" "PostgreSQL" "kb-addon-postgresql"
fi

if [ "$ENABLE_REDIS" = true ]; then
    install_addon_release "Redis" "kb-addon-redis" "redis" "redis"
    wait_for_addon_enabled "redis" "Redis" "kb-addon-redis"
fi

if [ "$ENABLE_ELASTICSEARCH" = true ]; then
    install_addon_release "Elasticsearch" "kb-addon-elasticsearch" "elasticsearch" "elasticsearch"
    wait_for_addon_enabled "elasticsearch" "Elasticsearch" "kb-addon-elasticsearch"
fi

if [ "$ENABLE_QDRANT" = true ]; then
    install_addon_release "Qdrant" "kb-addon-qdrant" "qdrant" "qdrant"
    wait_for_addon_enabled "qdrant" "Qdrant" "kb-addon-qdrant"
fi

if [ "$ENABLE_MONGODB" = true ]; then
    install_addon_release "MongoDB" "kb-addon-mongodb" "mongodb" "mongodb"
    wait_for_addon_enabled "mongodb" "MongoDB" "kb-addon-mongodb"
fi

if [ "$ENABLE_NEO4J" = true ]; then
    install_addon_release "Neo4j" "kb-addon-neo4j" "neo4j" "neo4j"
    wait_for_addon_enabled "neo4j" "Neo4j" "kb-addon-neo4j"
fi

print_success "KubeBlocks database addons installation completed!"
print "Now you can run 02-install-database.sh to install database clusters"
