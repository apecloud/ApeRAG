#!/bin/bash

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
helm repo add kubeblocks-addons $KUBEBLOCKS_ADDONS_HELM_REPO
helm repo update kubeblocks-addons

function disable_addons() {
    for addon in "$@"; do
        kbcli addon disable $addon
    done
    while true; do
        count=0
        for addon in "$@"; do
            if kubectl get addon $addon -n kb-system -o jsonpath='{.status.phase}' | grep -q "Disabled"; then
                count=$((count + 1))
                continue
            fi
        done
        if [ "$count" = $# ]; then
            break
        fi
        sleep 1
    done
}

# enable multi addon with different versions at once, the parameter is a map of addon name and version like "postgresql=0.9.5 redis=0.9.1 mongodb=0.9.1 qdrant=0.9.1"
function enable_addons() {
    local addons=()
    for item in "$@"; do
        addon=$(echo $item | cut -d '=' -f 1)
        version=$(echo $item | cut -d '=' -f 2)
        kbcli addon upgrade $addon --version $version --force
        kbcli addon enable $addon
        addons+=($addon)
    done

    while true; do
        count=0
        for addon in "${addons[@]}"; do
            if kubectl get addon $addon -n kb-system -o jsonpath='{.status.phase}' | grep -q "Enabled"; then
                count=$((count + 1))
            fi
        done
        if [ "$count" = $# ]; then
            break
        fi
    done
}

# elasticsearch is not installed by default in kubeblocks-addons, install it first and then enable it
helm upgrade --install kb-addon-elasticsearch kubeblocks-addons/elasticsearch --namespace kb-system --version 0.9.1

# postgresql/redis/mongodb are enabled by default, disable them to bypass the immutable check when upgrade addons.
disable_addons postgresql redis mongodb qdrant

# enable addons
enable_addons postgresql=0.9.5 redis=0.9.1 mongodb=0.9.1 qdrant=0.9.1


print_success "KubeBlocks database addons installation completed!"
print "Now you can run 02-install-database.sh to install database clusters"
