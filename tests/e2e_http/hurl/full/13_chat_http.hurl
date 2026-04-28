POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# v2 model-platform setup — ``ModelSpec`` only accepts ``model_id``
# now, so the bot config below references a freshly-seeded chat model.
POST {{base_url}}/api/v2/model-accounts
Content-Type: application/json
{
  "provider_type": "openai_compatible",
  "name": "openrouter-chat-{{run_id}}",
  "display_name": "OpenRouter (E2E chat {{run_id}})",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Captures]
openrouter_account_id: jsonpath "$.id"

POST {{base_url}}/api/v2/models
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
  "title": "Chat Bot {{run_id}}",
  "description": "HTTP full chat coverage",
  "type": "agent",
  "config": {
    "agent": {
      "completion": {
        "model_id": "{{completion_model_id}}",
        "temperature": 0.2
      },
      "system_prompt_template": "You are Chat Bot {{run_id}}.",
      "query_prompt_template": "{query}"
    }
  }
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.config.agent.completion.model_id" == "{{completion_model_id}}"

POST {{base_url}}/api/v2/bots/{{bot_id}}/chats
HTTP 200
[Captures]
chat_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.bot_id" == "{{bot_id}}"

GET {{base_url}}/api/v2/bots/{{bot_id}}/chats?page=1&page_size=20
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{chat_id}}"

PUT {{base_url}}/api/v2/bots/{{bot_id}}/chats/{{chat_id}}
Content-Type: application/json
{
  "title": "Chat Title {{run_id}}"
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.title" == "Chat Title {{run_id}}"

GET {{base_url}}/api/v2/bots/{{bot_id}}/chats/{{chat_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.title" == "Chat Title {{run_id}}"

POST {{base_url}}/v1/chat/completions?bot_id={{bot_id}}&chat_id={{chat_id}}
Content-Type: application/json
{
  "model": "aperag",
  "messages": [
    {
      "role": "user",
      "content": "Hello from HTTP E2E."
    }
  ],
  "stream": false
}
HTTP 200
[Asserts]
jsonpath "$.object" == "chat.completion"
jsonpath "$.model" == "aperag"
jsonpath "$.choices[0].message.role" == "assistant"
jsonpath "$.choices[0].message.content" exists

DELETE {{base_url}}/api/v2/bots/{{bot_id}}/chats/{{chat_id}}
HTTP 204

DELETE {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 204
