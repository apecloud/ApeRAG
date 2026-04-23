POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/api/v2/bots
Content-Type: application/json
{
  "title": "Agent Runtime V3 {{run_id}}",
  "description": "HTTP contract coverage for v2 agent runtime",
  "type": "agent",
  "config": {
    "agent": {
      "completion": {
        "model_service_provider": "openrouter",
        "model": "google/gemini-2.5-flash",
        "custom_llm_provider": "openrouter",
        "temperature": 0.2
      },
      "system_prompt_template": "You are Agent Runtime V3 {{run_id}}.",
      "query_prompt_template": "{query}"
    }
  }
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"

POST {{base_url}}/api/v2/bots/{{bot_id}}/chats
HTTP 200
[Captures]
chat_id: jsonpath "$.id"

POST {{base_url}}/api/v2/agent/chats/{{chat_id}}/turns
Content-Type: application/json
{
  "query": "Summarize the new agent runtime contract.",
  "language": "en-US",
  "client_idempotency_key": "agent-runtime-v3-{{run_id}}"
}
HTTP 200
[Captures]
turn_id: jsonpath "$.turn.turn_id"
[Asserts]
jsonpath "$.turn.schema_version" exists
jsonpath "$.turn.chat_id" == "{{chat_id}}"
jsonpath "$.turn.bot_id" == "{{bot_id}}"
jsonpath "$.turn.client_idempotency_key" == "agent-runtime-v3-{{run_id}}"
jsonpath "$.turn.input_text" == "Summarize the new agent runtime contract."
jsonpath "$.stream_url" contains "/api/v2/agent/chats/{{chat_id}}/turns/{{turn_id}}/events"

POST {{base_url}}/api/v2/agent/chats/{{chat_id}}/turns
Content-Type: application/json
{
  "query": "Summarize the new agent runtime contract.",
  "language": "en-US",
  "client_idempotency_key": "agent-runtime-v3-{{run_id}}"
}
HTTP 200
[Asserts]
jsonpath "$.turn.turn_id" == "{{turn_id}}"
jsonpath "$.turn.client_idempotency_key" == "agent-runtime-v3-{{run_id}}"

GET {{base_url}}/api/v2/agent/chats/{{chat_id}}/turns/{{turn_id}}
HTTP 200
[Asserts]
jsonpath "$.turn.turn_id" == "{{turn_id}}"
jsonpath "$.turn.chat_id" == "{{chat_id}}"
jsonpath "$.turn.schema_version" exists
jsonpath "$.timeline" exists
jsonpath "$.artifacts" exists

POST {{base_url}}/api/v2/agent/chats/{{chat_id}}/turns/{{turn_id}}/cancel
HTTP 200
[Asserts]
jsonpath "$.turn_id" == "{{turn_id}}"
jsonpath "$.status" == "CANCELLED"

GET {{base_url}}/api/v2/agent/artifacts/not-found-{{run_id}}
HTTP 404

DELETE {{base_url}}/api/v2/bots/{{bot_id}}/chats/{{chat_id}}
HTTP 204

DELETE {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 204
