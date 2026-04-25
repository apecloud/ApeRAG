POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# v3 model-platform setup — ``ModelSpec`` only accepts ``model_id`` now,
# so the bot config below references a freshly-seeded chat model.
POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "openai_compatible",
  "name": "openrouter-agent-{{run_id}}",
  "display_name": "OpenRouter (E2E agent {{run_id}})",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Captures]
openrouter_account_id: jsonpath "$.id"

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

POST {{base_url}}/api/v2/bots
Content-Type: application/json
{
  "title": "Agent Runtime V3 {{run_id}}",
  "description": "HTTP contract coverage for v2 agent runtime",
  "type": "agent",
  "config": {
    "agent": {
      "completion": {
        "model_id": "{{completion_model_id}}",
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
# Phase 8 D8.4d (#90): snapshot endpoint returns canonical UIMessage parts.
# Legacy {turn, timeline, artifacts} keys are gone per D8 §2 wire/at-rest byte-equal.
jsonpath "$.turn_id" == "{{turn_id}}"
jsonpath "$.chat_id" == "{{chat_id}}"
jsonpath "$.role" == "assistant"
jsonpath "$.schema_version" exists
jsonpath "$.status" exists
jsonpath "$.parts" exists

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
