# Provider-aware HTTP smoke for the new model-platform v3 surface.
#
# Replaces the legacy ``/api/v2/providers/...`` flow that was removed in the
# model-platform refactor (provider/dialect → ModelProvider/ModelAccount/Model
# /ModelUse/ModelRunner). What this file exercises against a live compose
# stack with real provider keys:
#
#   1. ``GET /api/v3/model-providers`` returns the seeded built-in providers
#      (``openai``, ``dashscope``, ``jina``, ``openai_compatible``).
#   2. The current user can ``POST /api/v3/model-accounts`` to create
#      DashScope-OpenAI-compat (``alibabacloud``) and OpenRouter (via
#      ``openai_compatible``) accounts using the secrets from
#      ``E2E_ALIBABACLOUD_API_KEY`` / ``E2E_OPENROUTER_API_KEY``.
#   3. Per-account models are created via ``POST /api/v3/models`` for the
#      embedding / rerank / chat capabilities the rest of the suite needs.
#   4. ``PUT /api/v3/model-uses/{scenario}`` wires the canonical
#      collection-completion / collection-embedding / retrieval-rerank /
#      background-task scenarios.
#   5. The OpenAI-compat ``POST /api/v1/embeddings`` and ``POST /api/v1/rerank``
#      endpoints accept ``{model_id}`` and round-trip a real provider call.

POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

GET {{base_url}}/api/v3/model-providers
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.items" count >= 4

POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "dashscope",
  "name": "alibabacloud-{{run_id}}",
  "display_name": "DashScope (E2E {{run_id}})",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "{{alibabacloud_api_key}}"
}
HTTP 200
[Captures]
dashscope_account_id: jsonpath "$.id"
[Asserts]
jsonpath "$.provider_type" == "dashscope"
jsonpath "$.id" matches "^ma[A-Za-z0-9]+$"

POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "openai_compatible",
  "name": "openrouter-{{run_id}}",
  "display_name": "OpenRouter (E2E {{run_id}})",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Captures]
openrouter_account_id: jsonpath "$.id"
[Asserts]
jsonpath "$.provider_type" == "openai_compatible"

POST {{base_url}}/api/v3/models
Content-Type: application/json
{
  "account_id": "{{dashscope_account_id}}",
  "provider_model_id": "text-embedding-v3",
  "display_name": "DashScope text-embedding-v3",
  "capability": "embedding",
  "embedding_dimensions": 1024
}
HTTP 200
[Captures]
dashscope_embedding_model_id: jsonpath "$.id"
[Asserts]
jsonpath "$.capability" == "embedding"
jsonpath "$.runner_type" == "openai_compatible"

POST {{base_url}}/api/v3/models
Content-Type: application/json
{
  "account_id": "{{dashscope_account_id}}",
  "provider_model_id": "gte-rerank-v2",
  "display_name": "DashScope gte-rerank-v2",
  "capability": "rerank"
}
HTTP 200
[Captures]
dashscope_rerank_model_id: jsonpath "$.id"
[Asserts]
jsonpath "$.capability" == "rerank"
jsonpath "$.runner_type" == "dashscope_rerank"

POST {{base_url}}/api/v3/models
Content-Type: application/json
{
  "account_id": "{{openrouter_account_id}}",
  "provider_model_id": "google/gemini-2.5-flash",
  "display_name": "OpenRouter Gemini 2.5 Flash",
  "capability": "chat",
  "context_window": 8192
}
HTTP 200
[Captures]
openrouter_chat_model_id: jsonpath "$.id"
[Asserts]
jsonpath "$.capability" == "chat"
jsonpath "$.runner_type" == "openai_compatible"

GET {{base_url}}/api/v3/models
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.items" count >= 3

GET {{base_url}}/api/v3/model-accounts/{{dashscope_account_id}}/models
HTTP 200
[Asserts]
jsonpath "$.items" count >= 2

PUT {{base_url}}/api/v3/model-uses/collection_completion
Content-Type: application/json
{
  "primary_model_id": "{{openrouter_chat_model_id}}",
  "fallback_model_ids": [],
  "strategy": "single",
  "enabled": true
}
HTTP 200
[Asserts]
jsonpath "$.scenario" == "collection_completion"
jsonpath "$.primary_model_id" == "{{openrouter_chat_model_id}}"

PUT {{base_url}}/api/v3/model-uses/collection_embedding
Content-Type: application/json
{
  "primary_model_id": "{{dashscope_embedding_model_id}}",
  "fallback_model_ids": [],
  "strategy": "single",
  "enabled": true
}
HTTP 200
[Asserts]
jsonpath "$.primary_model_id" == "{{dashscope_embedding_model_id}}"

PUT {{base_url}}/api/v3/model-uses/retrieval_rerank
Content-Type: application/json
{
  "primary_model_id": "{{dashscope_rerank_model_id}}",
  "fallback_model_ids": [],
  "strategy": "single",
  "enabled": true
}
HTTP 200
[Asserts]
jsonpath "$.primary_model_id" == "{{dashscope_rerank_model_id}}"

PUT {{base_url}}/api/v3/model-uses/background_task
Content-Type: application/json
{
  "primary_model_id": "{{openrouter_chat_model_id}}",
  "fallback_model_ids": [],
  "strategy": "single",
  "enabled": true
}
HTTP 200

GET {{base_url}}/api/v3/model-uses
HTTP 200
[Asserts]
jsonpath "$.items" count >= 4

POST {{base_url}}/api/v1/embeddings
Content-Type: application/json
{
  "model_id": "{{dashscope_embedding_model_id}}",
  "input": [
    "ApeRAG is a retrieval augmented generation platform.",
    "Embeddings convert text into vectors."
  ]
}
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.model" == "{{dashscope_embedding_model_id}}"
jsonpath "$.data[0].index" == 0
jsonpath "$.data[1].index" == 1
jsonpath "$.usage.total_tokens" > 0

POST {{base_url}}/api/v1/rerank
Content-Type: application/json
{
  "model_id": "{{dashscope_rerank_model_id}}",
  "query": "capital of France",
  "documents": [
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan.",
    "Pasta is often served with tomato sauce."
  ],
  "top_k": 2,
  "return_documents": true
}
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.model" == "{{dashscope_rerank_model_id}}"
jsonpath "$.data[0].index" == 0
jsonpath "$.data[0].document.text" == "Paris is the capital of France."
jsonpath "$.usage.total_tokens" > 0
