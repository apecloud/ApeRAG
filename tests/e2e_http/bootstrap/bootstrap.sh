#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GENERATED_DIR="${ROOT_DIR}/tests/e2e_http/bootstrap/.generated"
mkdir -p "${GENERATED_DIR}"

E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:8000}"
E2E_BOOTSTRAP_MODE="${E2E_BOOTSTRAP_MODE:-public-register}"
E2E_RUN_ID="${E2E_RUN_ID:-$(date +%Y%m%d%H%M%S)}"
E2E_USERNAME="${E2E_USERNAME:-e2e_${E2E_RUN_ID}}"
E2E_PASSWORD="${E2E_PASSWORD:-E2E_${E2E_RUN_ID}_pass}"
E2E_EMAIL="${E2E_EMAIL:-${E2E_USERNAME}@example.com}"
E2E_ENV_FILE="${E2E_ENV_FILE:-${GENERATED_DIR}/e2e.env}"
E2E_ARTIFACTS_DIR="${E2E_ARTIFACTS_DIR:-${GENERATED_DIR}/artifacts}"
E2E_TESTDATA_DIR="${E2E_TESTDATA_DIR:-${ROOT_DIR}/tests/e2e_http/testdata}"

mkdir -p "${E2E_ARTIFACTS_DIR}"

wait_for_health() {
  local attempts="${1:-60}"
  local sleep_seconds="${2:-2}"
  local i

  for ((i = 1; i <= attempts; i++)); do
    if curl --silent --show-error --fail "${E2E_BASE_URL}/health/ready" >/dev/null; then
      return 0
    fi
    sleep "${sleep_seconds}"
  done

  echo "Timed out waiting for ${E2E_BASE_URL}/health/ready" >&2
  return 1
}

register_via_public_api() {
  local payload response body status
  payload="$(cat <<EOF
{"username":"${E2E_USERNAME}","email":"${E2E_EMAIL}","password":"${E2E_PASSWORD}"}
EOF
)"

  response="$(
    curl --silent --show-error \
      --write-out $'\n%{http_code}' \
      --header 'Content-Type: application/json' \
      --request POST \
      --data "${payload}" \
      "${E2E_BASE_URL}/api/v2/auth/register"
  )"
  status="$(printf '%s' "${response}" | tail -n 1)"
  body="$(printf '%s' "${response}" | sed '$d')"

  if [[ "${status}" == "200" ]]; then
    echo "Registered ${E2E_USERNAME} via public API"
    return 0
  fi

  if [[ "${status}" == "400" && "${body}" == *"already exists"* ]]; then
    echo "User ${E2E_USERNAME} already exists, continuing with login"
    return 0
  fi

  echo "Public bootstrap failed with status ${status}: ${body}" >&2
  return 1
}

verify_login() {
  local payload cookie_file
  payload="$(cat <<EOF
{"username":"${E2E_USERNAME}","password":"${E2E_PASSWORD}"}
EOF
)"
  cookie_file="${GENERATED_DIR}/bootstrap.cookies"

  curl --silent --show-error --fail \
    --cookie-jar "${cookie_file}" \
    --header 'Content-Type: application/json' \
    --request POST \
    --data "${payload}" \
    "${E2E_BASE_URL}/api/v2/auth/login" >/dev/null
}

write_env_file() {
  cat >"${E2E_ENV_FILE}" <<EOF
export E2E_BASE_URL='${E2E_BASE_URL}'
export E2E_BOOTSTRAP_MODE='${E2E_BOOTSTRAP_MODE}'
export E2E_RUN_ID='${E2E_RUN_ID}'
export E2E_USERNAME='${E2E_USERNAME}'
export E2E_PASSWORD='${E2E_PASSWORD}'
export E2E_EMAIL='${E2E_EMAIL}'
export E2E_ARTIFACTS_DIR='${E2E_ARTIFACTS_DIR}'
export E2E_TESTDATA_DIR='${E2E_TESTDATA_DIR}'
EOF
}

main() {
  wait_for_health

  case "${E2E_BOOTSTRAP_MODE}" in
    public-register)
      register_via_public_api
      ;;
    *)
      echo "Unsupported E2E_BOOTSTRAP_MODE=${E2E_BOOTSTRAP_MODE} (only 'public-register' is supported after Phase 8 task #65)" >&2
      exit 1
      ;;
  esac

  verify_login
  write_env_file

  echo "Bootstrap complete. Env file written to ${E2E_ENV_FILE}"
}

main "$@"
