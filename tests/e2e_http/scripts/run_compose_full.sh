#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

cleanup() {
  # Capture diagnostics BEFORE ``down.sh`` wipes the containers. Without this
  # order the ``if: failure()`` GitHub step that runs ``docker compose logs``
  # finds no containers (because the trap already tore them down) and
  # provider-aware failures — e.g. ``POST /api/v1/embeddings`` returning 500 —
  # stay opaque. See ``#模块化重构`` task #11.
  local exit_code=$?
  if [[ "${exit_code}" -ne 0 && "${E2E_DIAGNOSTIC_ON_FAILURE:-1}" == "1" ]]; then
    echo "[run_compose_full] capturing diagnostics (exit=${exit_code})"
    "${ROOT_DIR}/tests/e2e_http/scripts/provider_diagnostic.sh" || true
  fi
  if [[ "${E2E_KEEP_UP:-0}" != "1" ]]; then
    "${ROOT_DIR}/tests/e2e_http/runners/compose/down.sh"
  fi
  return "${exit_code}"
}

trap cleanup EXIT

"${ROOT_DIR}/tests/e2e_http/runners/compose/up.sh"
"${ROOT_DIR}/tests/e2e_http/bootstrap/bootstrap.sh"
"${ROOT_DIR}/tests/e2e_http/scripts/run_full.sh"
