#!/bin/bash

set -euo pipefail

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

KB_RELEASE_TAG="v${KB_VERSION#v}"
KB_PUBLIC_IMAGE_REGISTRY="docker.io"
KB_PUBLIC_IMAGE_NAMESPACE="apecloud"

# Check dependencies
check_dependencies

# Function for installing KubeBlocks
install_kubeblocks() {
    print "Ready to install KubeBlocks."

    # Install CSI Snapshotter CRDs
    kubectl create -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshotclasses.yaml
    kubectl create -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshots.yaml
    kubectl create -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshotcontents.yaml

    kubectl create namespace kb-system --dry-run=client -o yaml | kubectl apply -f -

    # Install snapshot controller from the upstream manifests to avoid Helm chart
    # template failures in CI.
    curl -fsSL https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/deploy/kubernetes/snapshot-controller/rbac-snapshot-controller.yaml \
        | sed 's/namespace: kube-system/namespace: kb-system/g' \
        | kubectl apply -f -
    curl -fsSL https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/deploy/kubernetes/snapshot-controller/setup-snapshot-controller.yaml \
        | sed 's/namespace: kube-system/namespace: kb-system/g' \
        | kubectl apply -f -
    kubectl wait --for=condition=ready pods -l app.kubernetes.io/name=snapshot-controller -n kb-system --timeout=120s
    print_success "snapshot-controller installation complete!"

    # Install KubeBlocks CRDs
    kubectl create -f https://github.com/apecloud/kubeblocks/releases/download/${KB_RELEASE_TAG}/kubeblocks_crds.yaml

    # Add and update KubeBlocks repository
    helm repo add kubeblocks $HELM_REPO
    helm repo update

    # Install KubeBlocks
    helm install kubeblocks kubeblocks/kubeblocks \
        --namespace kb-system \
        --create-namespace \
        --version="${KB_VERSION}" \
        --set image.registry="${KB_PUBLIC_IMAGE_REGISTRY}" \
        --set dataProtection.image.registry="${KB_PUBLIC_IMAGE_REGISTRY}" \
        --set addonChartsImage.registry="${KB_PUBLIC_IMAGE_REGISTRY}" \
        --set registryConfig.defaultRegistry="${KB_PUBLIC_IMAGE_REGISTRY}" \
        --set registryConfig.defaultNamespace="${KB_PUBLIC_IMAGE_NAMESPACE}"

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
