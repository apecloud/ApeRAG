#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

source "$DATABASE_SCRIPT_DIR/uninstall-kubeblocks.sh"

kubectl delete namespace kb-system

print_success "KubeBlocks uninstallation completed!"
