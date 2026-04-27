POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

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
      "model": "text-embedding-v3",
      "model_service_provider": "alibabacloud",
      "custom_llm_provider": "openai"
    },
    "completion": {
      "model": "google/gemini-2.5-flash",
      "model_service_provider": "openrouter",
      "custom_llm_provider": "openrouter"
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
        "model_service_provider": "openrouter",
        "model": "google/gemini-2.5-flash",
        "custom_llm_provider": "openrouter",
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
jsonpath "$.turn.turn_id" == "{{turn_id}}"
jsonpath "$.turn.chat_id" == "{{chat_id}}"
jsonpath "$.turn.bot_id" == "{{bot_id}}"
jsonpath "$.timeline" exists
jsonpath "$.artifacts" exists

DELETE {{base_url}}/api/v2/bots/{{bot_id}}/chats/{{chat_id}}
HTTP 204

DELETE {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
