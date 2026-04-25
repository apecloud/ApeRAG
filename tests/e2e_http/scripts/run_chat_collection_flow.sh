#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

required_vars=(
  E2E_BASE_URL
  E2E_USERNAME
  E2E_PASSWORD
  E2E_RUN_ID
  E2E_ALIBABACLOUD_API_KEY
  E2E_OPENROUTER_API_KEY
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
bot_id=""
chat_id=""

cleanup() {
  if [[ -n "${chat_id}" && -n "${bot_id}" ]]; then
    curl -sS -o /dev/null -b "${COOKIE_JAR}" -X DELETE \
      "${BASE_URL}/api/v2/bots/${bot_id}/chats/${chat_id}" || true
  fi
  if [[ -n "${bot_id}" ]]; then
    curl -sS -o /dev/null -b "${COOKIE_JAR}" -X DELETE \
      "${BASE_URL}/api/v2/bots/${bot_id}" || true
  fi
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

wait_for_document_indexes() {
  local collection_id="$1"
  local document_id="$2"
  local max_attempts=90
  local attempt=1

  while (( attempt <= max_attempts )); do
    local body
    body="$(request_json GET "/api/v2/collections/${collection_id}/documents/${document_id}")"
    local vector_status
    local fulltext_status
    vector_status="$(jq -r '.vector_index_status // empty' <<<"${body}")"
    fulltext_status="$(jq -r '.fulltext_index_status // empty' <<<"${body}")"

    if [[ "${vector_status}" == "ACTIVE" && "${fulltext_status}" == "ACTIVE" ]]; then
      return 0
    fi

    if [[ "${vector_status}" == "FAILED" || "${fulltext_status}" == "FAILED" ]]; then
      echo "Document indexes failed: vector=${vector_status}, fulltext=${fulltext_status}" >&2
      jq . <<<"${body}" >&2
      exit 1
    fi

    sleep 2
    (( attempt += 1 ))
  done

  echo "Timed out waiting for document indexes to become ACTIVE" >&2
  exit 1
}

wait_for_turn_completion() {
  local chat_id="$1"
  local turn_id="$2"
  local max_attempts=90
  local attempt=1
  local last_body=""

  while (( attempt <= max_attempts )); do
    local body
    body="$(request_json GET "/api/v2/agent/chats/${chat_id}/turns/${turn_id}")"
    last_body="${body}"
    local turn_status
    local answer_artifact_id
    local reference_artifact_id
    turn_status="$(jq -r '.turn.status // empty' <<<"${body}")"
    answer_artifact_id="$(jq -r '.turn.answer_artifact_id // empty' <<<"${body}")"
    reference_artifact_id="$(jq -r '.turn.reference_bundle_artifact_id // empty' <<<"${body}")"

    if [[ "${turn_status}" == "COMPLETED" && -n "${answer_artifact_id}" ]]; then
      printf '%s\n%s\n' "${answer_artifact_id}" "${reference_artifact_id}"
      return 0
    fi

    if [[ "${turn_status}" == "FAILED" || "${turn_status}" == "CANCELLED" ]]; then
      echo "Turn finished unsuccessfully: status=${turn_status}" >&2
      jq . <<<"${body}" >&2
      exit 1
    fi

    sleep 2
    (( attempt += 1 ))
  done

  echo "Timed out waiting for turn completion artifacts" >&2
  if [[ -n "${last_body}" ]]; then
    jq . <<<"${last_body}" >&2 || true
  fi
  exit 1
}

echo "Running business chat collection flow"

request_json POST "/api/v2/auth/login" "$(jq -nc \
  --arg username "${E2E_USERNAME}" \
  --arg password "${E2E_PASSWORD}" \
  '{username: $username, password: $password}')"

# v3 model-platform setup — ``ModelSpec`` only accepts ``model_id`` now,
# so the collection / bot configs below reference freshly-seeded models.
dashscope_account_body="$(request_json POST "/api/v3/model-accounts" "$(jq -nc \
  --arg run_id "${E2E_RUN_ID}" \
  --arg api_key "${E2E_ALIBABACLOUD_API_KEY}" \
  '{
    provider_type: "dashscope",
    name: ("alibabacloud-chat-flow-" + $run_id),
    display_name: ("DashScope (E2E chat flow " + $run_id + ")"),
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key: $api_key
  }')")"
dashscope_account_id="$(jq -r '.id' <<<"${dashscope_account_body}")"

openrouter_account_body="$(request_json POST "/api/v3/model-accounts" "$(jq -nc \
  --arg run_id "${E2E_RUN_ID}" \
  --arg api_key "${E2E_OPENROUTER_API_KEY}" \
  '{
    provider_type: "openai_compatible",
    name: ("openrouter-chat-flow-" + $run_id),
    display_name: ("OpenRouter (E2E chat flow " + $run_id + ")"),
    base_url: "https://openrouter.ai/api/v1",
    api_key: $api_key
  }')")"
openrouter_account_id="$(jq -r '.id' <<<"${openrouter_account_body}")"

embedding_model_body="$(request_json POST "/api/v3/models" "$(jq -nc \
  --arg account_id "${dashscope_account_id}" \
  '{
    account_id: $account_id,
    provider_model_id: "text-embedding-v3",
    display_name: "DashScope text-embedding-v3",
    capability: "embedding",
    embedding_dimensions: 1024
  }')")"
embedding_model_id="$(jq -r '.id' <<<"${embedding_model_body}")"

completion_model_body="$(request_json POST "/api/v3/models" "$(jq -nc \
  --arg account_id "${openrouter_account_id}" \
  '{
    account_id: $account_id,
    provider_model_id: "google/gemini-2.5-flash",
    display_name: "OpenRouter Gemini 2.5 Flash",
    capability: "chat",
    context_window: 8192
  }')")"
completion_model_id="$(jq -r '.id' <<<"${completion_model_body}")"

collection_body="$(request_json POST "/api/v2/collections" "$(jq -nc \
  --arg run_id "${E2E_RUN_ID}" \
  --arg embedding_model_id "${embedding_model_id}" \
  --arg completion_model_id "${completion_model_id}" \
  '{
    title: ("Chat Collection Flow Script " + $run_id),
    description: "Business-flow chat validation with real index completion",
    type: "document",
    config: {
      source: "system",
      enable_vector: true,
      enable_fulltext: true,
      enable_knowledge_graph: false,
      enable_summary: false,
      enable_vision: false,
      embedding: {
        model_id: $embedding_model_id
      },
      completion: {
        model_id: $completion_model_id
      }
    }
  }')")"
collection_id="$(jq -r '.id' <<<"${collection_body}")"

upload_body="$(request_file_upload \
  "/api/v2/collections/${collection_id}/documents/upload" \
  "${ROOT_DIR}/tests/e2e_http/testdata/full-document.txt")"
document_id="$(jq -r '.document_id' <<<"${upload_body}")"

request_json POST "/api/v2/collections/${collection_id}/documents/confirm" "$(jq -nc \
  --arg document_id "${document_id}" \
  '{document_ids: [$document_id]}')"

wait_for_document_indexes "${collection_id}" "${document_id}"

search_body="$(request_json POST "/api/v2/collections/${collection_id}/searches" "$(jq -nc '{
  query: "Hurl",
  fulltext_search: {
    topk: 3
  },
  save_to_history: false
}')")"
jq -e '(.items // []) | length > 0' <<<"${search_body}" >/dev/null

bot_body="$(request_json POST "/api/v2/bots" "$(jq -nc \
  --arg run_id "${E2E_RUN_ID}" \
  --arg collection_id "${collection_id}" \
  --arg completion_model_id "${completion_model_id}" \
  '{
    title: ("Collection Chat Script " + $run_id),
    description: "Business-flow chat validation bot",
    type: "agent",
    config: {
      agent: {
        completion: {
          model_id: $completion_model_id,
          temperature: 0.2
        },
        system_prompt_template: "Answer with the collection context when available.",
        query_prompt_template: "{query}",
        collections: [{id: $collection_id}]
      }
    }
  }')")"
bot_id="$(jq -r '.id' <<<"${bot_body}")"

chat_body="$(request_json POST "/api/v2/bots/${bot_id}/chats")"
chat_id="$(jq -r '.id' <<<"${chat_body}")"

turn_body="$(request_json POST "/api/v2/agent/chats/${chat_id}/turns" "$(jq -nc \
  --arg run_id "${E2E_RUN_ID}" \
  '{
    query: "What does the uploaded document say about Hurl?",
    language: "en-US",
    client_idempotency_key: ("collection-chat-script-" + $run_id)
  }')")"
turn_id="$(jq -r '.turn.turn_id' <<<"${turn_body}")"

mapfile -t artifact_ids < <(wait_for_turn_completion "${chat_id}" "${turn_id}")
answer_artifact_id="${artifact_ids[0]}"
reference_artifact_id="${artifact_ids[1]:-}"

answer_body="$(request_json GET "/api/v2/agent/artifacts/${answer_artifact_id}")"

jq -e '.artifact_type == "answer"' <<<"${answer_body}" >/dev/null
jq -e '(.payload.text // "") | length > 0' <<<"${answer_body}" >/dev/null

if [[ -n "${reference_artifact_id}" ]]; then
  reference_body="$(request_json GET "/api/v2/agent/artifacts/${reference_artifact_id}")"
  jq -e '.artifact_type == "reference_bundle"' <<<"${reference_body}" >/dev/null
  jq -e '(.payload.items // []) | length > 0' <<<"${reference_body}" >/dev/null
fi

echo "Business chat collection flow passed"
