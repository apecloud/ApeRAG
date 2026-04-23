#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

required_vars=(
  E2E_BASE_URL
  E2E_USERNAME
  E2E_PASSWORD
  E2E_RUN_ID
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: ${var_name}" >&2
    exit 1
  fi
done

for cmd in curl jq; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
done

BASE_URL="${E2E_BASE_URL%/}"
COOKIE_JAR="$(mktemp)"
TMP_FILES=("${COOKIE_JAR}")

collection_id=""
document_id=""

cleanup() {
  if [[ -n "${document_id}" && -n "${collection_id}" ]]; then
    curl -sS -o /dev/null -b "${COOKIE_JAR}" -X DELETE \
      "${BASE_URL}/api/v2/collections/${collection_id}/documents/${document_id}" || true
  fi
  if [[ -n "${collection_id}" ]]; then
    curl -sS -o /dev/null -b "${COOKIE_JAR}" -X DELETE \
      "${BASE_URL}/api/v2/collections/${collection_id}" || true
  fi
  rm -f "${TMP_FILES[@]}"
}
trap cleanup EXIT

request_json() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local response_file
  response_file="$(mktemp)"
  TMP_FILES+=("${response_file}")

  local http_code
  if [[ -n "${body}" ]]; then
    http_code="$(
      curl -sS \
        -o "${response_file}" \
        -w '%{http_code}' \
        -b "${COOKIE_JAR}" \
        -c "${COOKIE_JAR}" \
        -X "${method}" \
        -H 'Content-Type: application/json' \
        -d "${body}" \
        "${BASE_URL}${path}"
    )"
  else
    http_code="$(
      curl -sS \
        -o "${response_file}" \
        -w '%{http_code}' \
        -b "${COOKIE_JAR}" \
        -c "${COOKIE_JAR}" \
        -X "${method}" \
        "${BASE_URL}${path}"
    )"
  fi

  if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
    echo "Request failed: ${method} ${path} -> ${http_code}" >&2
    cat "${response_file}" >&2
    exit 1
  fi

  cat "${response_file}"
}

request_file_upload() {
  local path="$1"
  local file_path="$2"
  local response_file
  response_file="$(mktemp)"
  TMP_FILES+=("${response_file}")

  local http_code
  http_code="$(
    curl -sS \
      -o "${response_file}" \
      -w '%{http_code}' \
      -b "${COOKIE_JAR}" \
      -c "${COOKIE_JAR}" \
      -X POST \
      -F "file=@${file_path};type=text/plain" \
      "${BASE_URL}${path}"
  )"

  if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
    echo "Upload failed: POST ${path} -> ${http_code}" >&2
    cat "${response_file}" >&2
    exit 1
  fi

  cat "${response_file}"
}

wait_for_graph_index() {
  local collection_id="$1"
  local document_id="$2"
  local max_attempts=120
  local attempt=1

  while (( attempt <= max_attempts )); do
    local body
    body="$(request_json GET "/api/v2/collections/${collection_id}/documents/${document_id}")"
    local graph_status
    graph_status="$(jq -r '.graph_index_status // empty' <<<"${body}")"

    if [[ "${graph_status}" == "ACTIVE" ]]; then
      return 0
    fi

    if [[ "${graph_status}" == "FAILED" || "${graph_status}" == "SKIPPED" ]]; then
      echo "Graph index did not complete successfully: graph=${graph_status}" >&2
      jq . <<<"${body}" >&2
      exit 1
    fi

    sleep 3
    (( attempt += 1 ))
  done

  echo "Timed out waiting for graph index to become ACTIVE" >&2
  exit 1
}

echo "Running graph index business flow"

request_json POST "/api/v1/login" "$(jq -nc \
  --arg username "${E2E_USERNAME}" \
  --arg password "${E2E_PASSWORD}" \
  '{username: $username, password: $password}')"

collection_body="$(request_json POST "/api/v2/collections" "$(jq -nc \
  --arg run_id "${E2E_RUN_ID}" \
  '{
    title: ("Graph Index Flow Script " + $run_id),
    description: "Business-flow graph validation with real graph index completion",
    type: "document",
    config: {
      source: "system",
      enable_vector: false,
      enable_fulltext: false,
      enable_knowledge_graph: true,
      enable_summary: false,
      enable_vision: false,
      embedding: {
        model: "text-embedding-v3",
        model_service_provider: "alibabacloud",
        custom_llm_provider: "openai"
      },
      completion: {
        model: "google/gemini-2.5-flash",
        model_service_provider: "openrouter",
        custom_llm_provider: "openrouter"
      }
    }
  }')")"
collection_id="$(jq -r '.id' <<<"${collection_body}")"

upload_body="$(request_file_upload \
  "/api/v2/collections/${collection_id}/documents/upload" \
  "${ROOT_DIR}/tests/e2e_http/testdata/graph-document.txt")"
document_id="$(jq -r '.document_id' <<<"${upload_body}")"

request_json POST "/api/v2/collections/${collection_id}/documents/confirm" "$(jq -nc \
  --arg document_id "${document_id}" \
  '{document_ids: [$document_id]}')"

wait_for_graph_index "${collection_id}" "${document_id}"

labels_body="$(request_json GET "/api/v2/collections/${collection_id}/graphs/labels")"
graph_body="$(request_json GET "/api/v2/collections/${collection_id}/graphs?label=*&max_nodes=50&max_depth=2")"

jq -e '(.labels // []) | length > 0' <<<"${labels_body}" >/dev/null
jq -e '(.nodes // []) | length > 0' <<<"${graph_body}" >/dev/null
jq -e '(.edges // []) | length > 0' <<<"${graph_body}" >/dev/null

echo "Graph index business flow passed"
