#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

STATE_DIR="${ROOT_DIR}/tests/e2e_http/runners/k8s/.generated"
PID_FILE="${STATE_DIR}/port-forward.pid"
LOG_FILE="${STATE_DIR}/port-forward.log"

mkdir -p "${STATE_DIR}"

E2E_K8S_NAMESPACE="${E2E_K8S_NAMESPACE:-default}"
E2E_K8S_SERVICE="${E2E_K8S_SERVICE:-aperag-api}"
E2E_K8S_REMOTE_PORT="${E2E_K8S_REMOTE_PORT:-8000}"
E2E_K8S_LOCAL_PORT="${E2E_K8S_LOCAL_PORT:-18080}"
E2E_HEALTH_ATTEMPTS="${E2E_HEALTH_ATTEMPTS:-90}"
E2E_HEALTH_SLEEP_SECONDS="${E2E_HEALTH_SLEEP_SECONDS:-2}"
E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:${E2E_K8S_LOCAL_PORT}}"

if [[ "${E2E_K8S_USE_PORT_FORWARD:-1}" == "1" ]]; then
  if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(cat "${PID_FILE}")"
    if kill -0 "${existing_pid}" >/dev/null 2>&1; then
      echo "K8s runner already has an active port-forward (pid ${existing_pid})." >&2
      exit 1
    fi
    rm -f "${PID_FILE}"
  fi

  kubectl -n "${E2E_K8S_NAMESPACE}" port-forward "svc/${E2E_K8S_SERVICE}" \
    "${E2E_K8S_LOCAL_PORT}:${E2E_K8S_REMOTE_PORT}" >"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
fi

for ((i = 1; i <= E2E_HEALTH_ATTEMPTS; i++)); do
  if curl --silent --show-error --fail "${E2E_BASE_URL}/health/ready" >/dev/null; then
    echo "K8s runner ready at ${E2E_BASE_URL}"
    exit 0
  fi
  sleep "${E2E_HEALTH_SLEEP_SECONDS}"
done

echo "K8s runner failed to reach healthy state at ${E2E_BASE_URL}" >&2
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 50 "${LOG_FILE}" >&2 || true
fi
exit 1
