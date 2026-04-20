#!/bin/bash

set -euo pipefail

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

# Check dependencies
check_dependencies

# Function for installing KubeBlocks
install_kubeblocks() {
    print "Ready to install KubeBlocks."

    # Install CSI Snapshotter CRDs
    kubectl create -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshotclasses.yaml
    kubectl create -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshots.yaml
    kubectl create -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshotcontents.yaml

    # Add and update Piraeus repository
    helm repo add piraeus-charts https://piraeus.io/helm-charts/
    helm repo update

    # Install snapshot controller
    helm install snapshot-controller piraeus-charts/snapshot-controller -n kb-system --create-namespace
    kubectl wait --for=condition=ready pods -l app.kubernetes.io/name=snapshot-controller -n kb-system --timeout=60s
    print_success "snapshot-controller installation complete!"

    # Install KubeBlocks CRDs
    kubectl create -f https://github.com/apecloud/kubeblocks/releases/download/${KB_VERSION}/kubeblocks_crds.yaml

    # Add and update KubeBlocks repository
    helm repo add kubeblocks $HELM_REPO
    helm repo update

    # Install KubeBlocks
    helm install kubeblocks kubeblocks/kubeblocks --namespace kb-system --create-namespace --version=${KB_VERSION}

    # Verify installation
    print "Waiting for KubeBlocks to be ready..."
    kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=kubeblocks -n kb-system --timeout=120s
    print_success "KubeBlocks installation complete!"
}

wait_for_kubeblocks_crds() {
    local crds=(
        "clusters.apps.kubeblocks.io"
        "clusterdefinitions.apps.kubeblocks.io"
        "componentdefinitions.apps.kubeblocks.io"
        "componentversions.apps.kubeblocks.io"
        "actionsets.dataprotection.kubeblocks.io"
        "backuppolicytemplates.dataprotection.kubeblocks.io"
        "paramconfigrenderers.parameters.kubeblocks.io"
        "parametersdefinitions.parameters.kubeblocks.io"
    )

    print "Waiting for KubeBlocks CRDs to become established..."
    for crd in "${crds[@]}"; do
        kubectl wait --for=condition=Established "crd/${crd}" --timeout=180s
    done
    print_success "KubeBlocks CRDs are established."
}

# Check if KubeBlocks is already installed
print "Checking if KubeBlocks is already installed in kb-system namespace..."
if kubectl get namespace kb-system &>/dev/null && kubectl get deployment kubeblocks -n kb-system &>/dev/null; then
    print_success "KubeBlocks is already installed in kb-system namespace."
else
    # Call the function to install KubeBlocks
    install_kubeblocks
fi

wait_for_kubeblocks_crds
