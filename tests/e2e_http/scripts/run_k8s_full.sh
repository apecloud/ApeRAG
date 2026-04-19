#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

cleanup() {
  if [[ "${E2E_KEEP_UP:-0}" != "1" ]]; then
    "${ROOT_DIR}/tests/e2e_http/runners/k8s/down.sh"
  fi
}

trap cleanup EXIT

export E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:${E2E_K8S_LOCAL_PORT:-18080}}"

"${ROOT_DIR}/tests/e2e_http/runners/k8s/up.sh"
"${ROOT_DIR}/tests/e2e_http/bootstrap/bootstrap.sh"
"${ROOT_DIR}/tests/e2e_http/scripts/run_full.sh"
