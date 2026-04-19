#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

E2E_ENV_FILE="${E2E_ENV_FILE:-${ROOT_DIR}/tests/e2e_http/bootstrap/.generated/e2e.env}"

if [[ ! -f "${E2E_ENV_FILE}" ]]; then
  echo "Missing bootstrap env file: ${E2E_ENV_FILE}" >&2
  echo "Run ./tests/e2e_http/bootstrap/bootstrap.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${E2E_ENV_FILE}"

SMOKE_DIR="${ROOT_DIR}/tests/e2e_http/hurl/smoke"

for file in "${SMOKE_DIR}"/*.hurl; do
  echo "Running ${file}"
  hurl --test \
    --file-root "${ROOT_DIR}" \
    --variable base_url="${E2E_BASE_URL}" \
    --variable username="${E2E_USERNAME}" \
    --variable password="${E2E_PASSWORD}" \
    --variable run_id="${E2E_RUN_ID}" \
    --variable testdata_dir="${E2E_TESTDATA_DIR}" \
    "${file}"
done
