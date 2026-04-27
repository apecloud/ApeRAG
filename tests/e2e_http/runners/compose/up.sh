#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:8000}"
E2E_COMPOSE_SERVICES="${E2E_COMPOSE_SERVICES:-postgres redis qdrant es api}"
E2E_HEALTH_ATTEMPTS="${E2E_HEALTH_ATTEMPTS:-90}"
E2E_HEALTH_SLEEP_SECONDS="${E2E_HEALTH_SLEEP_SECONDS:-2}"

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/envs/env.template" "${ROOT_DIR}/.env"
fi

docker compose -f docker-compose.yml up -d --build ${E2E_COMPOSE_SERVICES}

for ((i = 1; i <= E2E_HEALTH_ATTEMPTS; i++)); do
  if curl --silent --show-error --fail "${E2E_BASE_URL}/health" >/dev/null; then
    echo "Compose runner ready at ${E2E_BASE_URL}"
    exit 0
  fi
  sleep "${E2E_HEALTH_SLEEP_SECONDS}"
done

echo "Compose runner failed to reach healthy state at ${E2E_BASE_URL}" >&2
exit 1
