POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# v3 model-platform setup — every file is run independently and
# ``ModelSpec`` only accepts ``model_id`` now, so the collection /
# bot configs below reference freshly-seeded models.
POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "dashscope",
  "name": "alibabacloud-flow-{{run_id}}",
  "display_name": "DashScope (E2E flow {{run_id}})",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "{{alibabacloud_api_key}}"
}
HTTP 200
[Captures]
dashscope_account_id: jsonpath "$.id"

POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "openai_compatible",
  "name": "openrouter-flow-{{run_id}}",
  "display_name": "OpenRouter (E2E flow {{run_id}})",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Captures]
openrouter_account_id: jsonpath "$.id"

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
embedding_model_id: jsonpath "$.id"

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
completion_model_id: jsonpath "$.id"

POST {{base_url}}/api/v2/collections
Content-Type: application/json
{
  "title": "Chat Collection Flow {{run_id}}",
  "description": "HTTP business flow coverage for collection -> document -> chat",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": true,
    "enable_fulltext": true,
    "enable_knowledge_graph": false,
    "enable_summary": false,
    "enable_vision": false,
    "embedding": {
      "model_id": "{{embedding_model_id}}"
    },
    "completion": {
      "model_id": "{{completion_model_id}}"
    }
  }
}
HTTP 200
[Captures]
collection_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{collection_id}}"
jsonpath "$.config.enable_vector" == true
jsonpath "$.config.enable_fulltext" == true

POST {{base_url}}/api/v2/collections/{{collection_id}}/documents/upload
[MultipartFormData]
file: file,tests/e2e_http/testdata/full-document.txt; type=text/plain
HTTP 200
[Captures]
document_id: jsonpath "$.document_id"
[Asserts]
jsonpath "$.document_id" == "{{document_id}}"
jsonpath "$.status" == "UPLOADED"

POST {{base_url}}/api/v2/collections/{{collection_id}}/documents/confirm
Content-Type: application/json
{
  "document_ids": ["{{document_id}}"]
}
HTTP 200
[Asserts]
jsonpath "$.confirmed_count" == 1
jsonpath "$.failed_count" == 0

POST {{base_url}}/api/v2/bots
Content-Type: application/json
{
  "title": "Collection Chat Flow {{run_id}}",
  "description": "HTTP business flow chat coverage",
  "type": "agent",
  "config": {
    "agent": {
      "completion": {
        "model_id": "{{completion_model_id}}",
        "temperature": 0.2
      },
      "system_prompt_template": "Answer with the collection context when available.",
      "query_prompt_template": "{query}",
      "collections": [
        {
          "id": "{{collection_id}}"
        }
      ]
    }
  }
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.config.agent.collections[0].id" == "{{collection_id}}"

POST {{base_url}}/api/v2/bots/{{bot_id}}/chats
HTTP 200
[Captures]
chat_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.bot_id" == "{{bot_id}}"

POST {{base_url}}/api/v2/agent/chats/{{chat_id}}/turns
Content-Type: application/json
{
  "query": "What does the uploaded document say about Hurl?",
  "language": "en-US",
  "client_idempotency_key": "collection-chat-flow-{{run_id}}"
}
HTTP 200
[Captures]
turn_id: jsonpath "$.turn.turn_id"
[Asserts]
jsonpath "$.turn.turn_id" == "{{turn_id}}"
jsonpath "$.turn.chat_id" == "{{chat_id}}"
jsonpath "$.turn.bot_id" == "{{bot_id}}"
jsonpath "$.turn.client_idempotency_key" == "collection-chat-flow-{{run_id}}"
jsonpath "$.turn.input_text" == "What does the uploaded document say about Hurl?"
jsonpath "$.turn.schema_version" exists
jsonpath "$.stream_url" contains "/api/v2/agent/chats/{{chat_id}}/turns/{{turn_id}}/events"

GET {{base_url}}/api/v2/agent/chats/{{chat_id}}/turns/{{turn_id}}
HTTP 200
[Asserts]
# Phase 8 D8.4d (#90): snapshot endpoint returns canonical UIMessage parts.
# Legacy {turn, timeline, artifacts} keys are gone; bot_id is no longer in
# the snapshot envelope (it lives on the turn-create response which is
# already asserted above).
jsonpath "$.turn_id" == "{{turn_id}}"
jsonpath "$.chat_id" == "{{chat_id}}"
jsonpath "$.role" == "assistant"
jsonpath "$.schema_version" exists
jsonpath "$.status" exists
jsonpath "$.parts" exists

DELETE {{base_url}}/api/v2/bots/{{bot_id}}/chats/{{chat_id}}
HTTP 204

DELETE {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
