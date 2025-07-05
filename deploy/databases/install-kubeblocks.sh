#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

# Check dependencies
check_dependencies

# Function for installing KubeBlocks
install_kubeblocks() {
    print "Ready to install kbcli..."
    if which kbcli &>/dev/null; then
        print_success "kbcli is already installed, skipping installation."
    else
        curl -fsSL https://kubeblocks.io/installer/install_cli.sh | bash -s 0.9.4
        print_success "kbcli installation complete!"
    fi

    print "Ready to install KubeBlocks."
    kbcli kubeblocks install --version=${KB_VERSION} --namespace=kb-system --create-namespace --set dataProtection.enabled=false
    print_success "KubeBlocks installation complete!"
}

# Check if KubeBlocks is already installed
print "Checking if KubeBlocks is already installed in kb-system namespace..."
if kubectl get namespace kb-system &>/dev/null && kubectl get deployment kubeblocks -n kb-system &>/dev/null; then
    print_success "KubeBlocks is already installed in kb-system namespace."
else
    # Call the function to install KubeBlocks
    install_kubeblocks
fi
