#!/bin/bash

set -euo pipefail

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

CLUSTER_READY_TIMEOUT="${CLUSTER_READY_TIMEOUT:-600}"
DATABASE_PROFILE="${DATABASE_PROFILE:-}"

resolve_values_file() {
  local values_dir="$1"
  local default_file="${values_dir}/values.yaml"

  if [ -n "$DATABASE_PROFILE" ]; then
    local profile_file="${values_dir}/values.${DATABASE_PROFILE}.yaml"
    if [ -f "$profile_file" ]; then
      echo "$profile_file"
      return 0
    fi
  fi

  echo "$default_file"
}

cluster_phase() {
  local cluster_name="$1"

  kubectl get clusters.apps.kubeblocks.io "$cluster_name" -n "$NAMESPACE" \
    -o jsonpath='{.status.phase}' 2>/dev/null || true
}

cluster_exists() {
  local cluster_name="$1"

  kubectl get clusters.apps.kubeblocks.io "$cluster_name" -n "$NAMESPACE" >/dev/null 2>&1
}

dump_cluster_diagnostics() {
  local cluster_name="$1"
  local release_name="$2"

  print_warning "Diagnostics for cluster ${cluster_name} (release=${release_name})"
  kubectl get clusters.apps.kubeblocks.io "$cluster_name" -n "$NAMESPACE" -o wide || true
  kubectl describe clusters.apps.kubeblocks.io "$cluster_name" -n "$NAMESPACE" || true
  kubectl get components.apps.kubeblocks.io -n "$NAMESPACE" || true
  kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=${release_name}" -o wide || true
  kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp | tail -n 40 || true
}

wait_for_cluster_ready() {
  local cluster_name="$1"
  local release_name="$2"
  local display_name="$3"
  local phase=""
  local elapsed=0

  print "Waiting for ${display_name} cluster (${cluster_name}) to become ready..."
  while [ "$elapsed" -lt "$CLUSTER_READY_TIMEOUT" ]; do
    if ! cluster_exists "$cluster_name"; then
      print_warning "${display_name} cluster resource ${cluster_name} is not visible yet (${elapsed}s elapsed)."
      sleep 10
      elapsed=$((elapsed + 10))
      continue
    fi

    phase="$(cluster_phase "$cluster_name")"
    if kubectl wait --for=condition=ready pods -l "app.kubernetes.io/instance=${release_name}" \
      -n "$NAMESPACE" --timeout=10s >/dev/null 2>&1; then
      print_success "${display_name} cluster is ready (phase=${phase:-<missing>})."
      return 0
    fi

    case "$phase" in
      Failed|Error|Abnormal)
        print_error "${display_name} cluster entered terminal phase ${phase}."
        dump_cluster_diagnostics "$cluster_name" "$release_name"
        return 1
        ;;
    esac

    kubectl get clusters.apps.kubeblocks.io "$cluster_name" -n "$NAMESPACE" || true
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=${release_name}" -o wide || true
    print "Waiting for ${display_name} pods to be ready (${elapsed}s elapsed, phase=${phase:-<missing>})..."
    sleep 10
    elapsed=$((elapsed + 10))
  done

  print_error "Timeout waiting for ${display_name} cluster ${cluster_name} to be ready after ${CLUSTER_READY_TIMEOUT}s."
  dump_cluster_diagnostics "$cluster_name" "$release_name"
  return 1
}

install_cluster() {
  local display_name="$1"
  local release_name="$2"
  local chart_name="$3"
  local values_dir="$4"
  local cluster_name="$5"
  local values_file=""

  values_file="$(resolve_values_file "$values_dir")"

  print "Installing ${display_name} cluster with values file ${values_file}..."
  helm upgrade --install "$release_name" "kubeblocks/${chart_name}" \
    -f "$values_file" \
    --namespace "$NAMESPACE" \
    --version "$ADDON_CLUSTER_CHART_VERSION"

  wait_for_cluster_ready "$cluster_name" "$release_name" "$display_name"
}

print "Installing database clusters..."

if [ "$ENABLE_POSTGRESQL" = true ]; then
  install_cluster \
    "PostgreSQL" \
    "pg-cluster" \
    "postgresql-cluster" \
    "$DATABASE_SCRIPT_DIR/postgresql" \
    "pg-cluster"
fi

if [ "$ENABLE_REDIS" = true ]; then
  install_cluster \
    "Redis" \
    "redis-cluster" \
    "redis-cluster" \
    "$DATABASE_SCRIPT_DIR/redis" \
    "redis-cluster"
fi

if [ "$ENABLE_ELASTICSEARCH" = true ]; then
  install_cluster \
    "Elasticsearch" \
    "es-cluster" \
    "elasticsearch-cluster" \
    "$DATABASE_SCRIPT_DIR/elasticsearch" \
    "es-cluster"
fi

if [ "$ENABLE_QDRANT" = true ]; then
  install_cluster \
    "Qdrant" \
    "qdrant-cluster" \
    "qdrant-cluster" \
    "$DATABASE_SCRIPT_DIR/qdrant" \
    "qdrant-cluster"
fi

if [ "$ENABLE_MONGODB" = true ]; then
  install_cluster \
    "MongoDB" \
    "mongodb-cluster" \
    "mongodb-cluster" \
    "$DATABASE_SCRIPT_DIR/mongodb" \
    "mongodb-cluster"
fi

if [ "$ENABLE_NEO4J" = true ]; then
  install_cluster \
    "Neo4j" \
    "neo4j-cluster" \
    "neo4j-cluster" \
    "$DATABASE_SCRIPT_DIR/neo4j" \
    "neo4j-cluster"
fi

print_success "Database clusters installation completed!"
print "Use the following command to check the status of installed clusters:"
print "kubectl get clusters -n $NAMESPACE"
