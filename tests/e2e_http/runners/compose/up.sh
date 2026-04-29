#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:8000}"
E2E_HEALTH_ATTEMPTS="${E2E_HEALTH_ATTEMPTS:-90}"
E2E_HEALTH_SLEEP_SECONDS="${E2E_HEALTH_SLEEP_SECONDS:-2}"

# SHAPE is the canonical knob — it sources tests/e2e_http/shapes/<shape>.env
# which declares the (vector backend, graph backend, service set, compose
# profiles) combination as a single named deployment form. Adding a new
# shape is one new file under tests/e2e_http/shapes/.
#
# For backward compatibility, callers may instead set VECTOR_DB_TYPE
# and/or GRAPH_DB_TYPE directly (PR #1740 entry point); the service list
# and profiles are then derived from those choices.
SHAPE="${SHAPE:-}"

if [[ -n "${SHAPE}" ]]; then
  SHAPE_FILE="${ROOT_DIR}/tests/e2e_http/shapes/${SHAPE}.env"
  if [[ ! -f "${SHAPE_FILE}" ]]; then
    echo "Unknown SHAPE='${SHAPE}': ${SHAPE_FILE} not found." >&2
    echo "Available shapes:" >&2
    ls "${ROOT_DIR}/tests/e2e_http/shapes/" 2>/dev/null \
      | sed 's/\.env$//' | sed 's/^/  - /' >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${SHAPE_FILE}"
  VECTOR_DB_TYPE="${SHAPE_VECTOR_DB_TYPE}"
  GRAPH_DB_TYPE="${SHAPE_GRAPH_DB_TYPE}"
  E2E_COMPOSE_SERVICES="${E2E_COMPOSE_SERVICES:-${SHAPE_COMPOSE_SERVICES}}"
  E2E_COMPOSE_PROFILES="${E2E_COMPOSE_PROFILES:-${SHAPE_COMPOSE_PROFILES}}"
else
  VECTOR_DB_TYPE="${VECTOR_DB_TYPE:-qdrant}"
  GRAPH_DB_TYPE="${GRAPH_DB_TYPE:-postgresql}"

  case "${VECTOR_DB_TYPE}" in
    qdrant)   default_services="postgres redis es qdrant api indexing-worker" ;;
    pgvector) default_services="postgres redis es api indexing-worker" ;;
    *)
      echo "VECTOR_DB_TYPE must be one of: qdrant, pgvector (got '${VECTOR_DB_TYPE}')" >&2
      exit 2
      ;;
  esac

  case "${GRAPH_DB_TYPE}" in
    postgresql) default_profiles="" ;;
    neo4j)      default_profiles="--profile neo4j" ;;
    nebula)     default_profiles="--profile nebula" ;;
    *)
      echo "GRAPH_DB_TYPE must be one of: postgresql, neo4j, nebula (got '${GRAPH_DB_TYPE}')" >&2
      exit 2
      ;;
  esac

  E2E_COMPOSE_SERVICES="${E2E_COMPOSE_SERVICES:-${default_services}}"
  E2E_COMPOSE_PROFILES="${E2E_COMPOSE_PROFILES:-${default_profiles}}"
fi

# Final validation (catches bad values inside shape files too).
case "${VECTOR_DB_TYPE}" in
  qdrant|pgvector) ;;
  *)
    echo "VECTOR_DB_TYPE must be one of: qdrant, pgvector (got '${VECTOR_DB_TYPE}')" >&2
    exit 2
    ;;
esac
case "${GRAPH_DB_TYPE}" in
  postgresql|neo4j|nebula) ;;
  *)
    echo "GRAPH_DB_TYPE must be one of: postgresql, neo4j, nebula (got '${GRAPH_DB_TYPE}')" >&2
    exit 2
    ;;
esac

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/envs/env.template" "${ROOT_DIR}/.env"
fi

# Idempotently set or append a KEY=VALUE in .env so the api container's
# env_file pickup reflects the selected backends.
update_env_var() {
  local key="$1"
  local value="$2"
  local file="${ROOT_DIR}/.env"
  if grep -qE "^${key}=" "${file}"; then
    awk -v k="${key}" -v v="${value}" 'BEGIN{FS=OFS="="} $1==k{print k"="v; next} {print}' \
      "${file}" > "${file}.tmp"
    mv "${file}.tmp" "${file}"
  else
    printf '\n%s=%s\n' "${key}" "${value}" >> "${file}"
  fi
}

update_env_var VECTOR_DB_TYPE "${VECTOR_DB_TYPE}"
update_env_var GRAPH_DB_TYPE "${GRAPH_DB_TYPE}"

label="${SHAPE:-vector=${VECTOR_DB_TYPE},graph=${GRAPH_DB_TYPE}}"
echo "Compose runner starting (shape=${label}, services='${E2E_COMPOSE_SERVICES}', profiles='${E2E_COMPOSE_PROFILES}')"

# shellcheck disable=SC2086
docker compose ${E2E_COMPOSE_PROFILES} -f docker-compose.yml up -d --build ${E2E_COMPOSE_SERVICES}

for ((i = 1; i <= E2E_HEALTH_ATTEMPTS; i++)); do
  if curl --silent --show-error --fail "${E2E_BASE_URL}/health/ready" >/dev/null; then
    echo "Compose runner ready at ${E2E_BASE_URL} (shape=${label})"
    exit 0
  fi
  sleep "${E2E_HEALTH_SLEEP_SECONDS}"
done

echo "Compose runner failed to reach healthy state at ${E2E_BASE_URL}" >&2
exit 1
