#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

cleanup() {
  if [[ "${E2E_KEEP_UP:-0}" != "1" ]]; then
    "${ROOT_DIR}/tests/e2e_http/runners/compose/down.sh"
  fi
}

trap cleanup EXIT

"${ROOT_DIR}/tests/e2e_http/runners/compose/up.sh"
"${ROOT_DIR}/tests/e2e_http/bootstrap/bootstrap.sh"
"${ROOT_DIR}/tests/e2e_http/scripts/run_smoke.sh"
