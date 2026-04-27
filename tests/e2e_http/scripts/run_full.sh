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

required_vars=(
  E2E_ALIBABACLOUD_API_KEY
  E2E_OPENROUTER_API_KEY
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: ${var_name}" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
source "${E2E_ENV_FILE}"

FULL_DIR="${ROOT_DIR}/tests/e2e_http/hurl/full"

for file in "${FULL_DIR}"/*.hurl; do
  echo "Running ${file}"
  # ``--error-format long`` prints the HTTP response body when an assertion
  # fails, so provider-aware failures like ``POST /api/v1/embeddings`` → 500
  # surface their FastAPI ``detail`` / LiteLLM error in the job log instead
  # of leaving reviewers with only the status code. See task #11.
  hurl --test \
    --error-format long \
    --file-root "${ROOT_DIR}" \
    --variable base_url="${E2E_BASE_URL}" \
    --variable username="${E2E_USERNAME}" \
    --variable password="${E2E_PASSWORD}" \
    --variable run_id="${E2E_RUN_ID}" \
    --variable testdata_dir="${E2E_TESTDATA_DIR}" \
    --secret alibabacloud_api_key="${E2E_ALIBABACLOUD_API_KEY}" \
    --secret openrouter_api_key="${E2E_OPENROUTER_API_KEY}" \
    "${file}"
done

echo "Running scripted business-flow checks"
"${ROOT_DIR}/tests/e2e_http/scripts/run_chat_collection_flow.sh"
# Wave 3 release gates the graph modality (architect msg=c79e9a3f):
# the §D.3.6 Nebula/Postgres LineageGraphStore adapter + LLM
# extractor are Wave 4 scope, so ProductionWorkerFactory raises
# WorkerFactoryError for any graph dispatch and the row lands at
# FAILED — run_graph_index_flow.sh then times out waiting for
# ACTIVE. Re-enable this script the moment Wave 4 wires the real
# backend; until then it would only verify the placeholder.
echo "Skipping run_graph_index_flow.sh until Wave 4 (graph modality gated; see docs/private-deployment.md)"
