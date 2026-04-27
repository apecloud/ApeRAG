#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:8000}"
E2E_HEALTH_ATTEMPTS="${E2E_HEALTH_ATTEMPTS:-90}"
E2E_HEALTH_SLEEP_SECONDS="${E2E_HEALTH_SLEEP_SECONDS:-2}"

# Backend selection. Defaults preserve historical behavior (Qdrant + PG-graph).
VECTOR_DB_TYPE="${VECTOR_DB_TYPE:-qdrant}"
GRAPH_DB_TYPE="${GRAPH_DB_TYPE:-postgresql}"

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

# Always include qdrant in the service set: the api container's depends_on
# requires it healthy regardless of which vector backend is selected. Leaving
# qdrant idle in pgvector mode is cheap and avoids docker-compose surgery.
DEFAULT_SERVICES="postgres redis qdrant es api"
E2E_COMPOSE_SERVICES="${E2E_COMPOSE_SERVICES:-${DEFAULT_SERVICES}}"

profile_flags=()
case "${GRAPH_DB_TYPE}" in
  neo4j)  profile_flags=(--profile neo4j) ;;
  nebula) profile_flags=(--profile nebula) ;;
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

echo "Compose runner starting (vector=${VECTOR_DB_TYPE}, graph=${GRAPH_DB_TYPE}, profiles='${profile_flags[*]:-}')"

# `${array[@]+"${array[@]}"}` is the set -u-safe expansion for an empty array.
docker compose ${profile_flags[@]+"${profile_flags[@]}"} -f docker-compose.yml up -d --build ${E2E_COMPOSE_SERVICES}

for ((i = 1; i <= E2E_HEALTH_ATTEMPTS; i++)); do
  if curl --silent --show-error --fail "${E2E_BASE_URL}/health" >/dev/null; then
    echo "Compose runner ready at ${E2E_BASE_URL} (vector=${VECTOR_DB_TYPE}, graph=${GRAPH_DB_TYPE})"
    exit 0
  fi
  sleep "${E2E_HEALTH_SLEEP_SECONDS}"
done

echo "Compose runner failed to reach healthy state at ${E2E_BASE_URL}" >&2
exit 1
