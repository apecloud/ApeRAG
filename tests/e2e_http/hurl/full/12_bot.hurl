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
  "name": "openrouter-bot-{{run_id}}",
  "display_name": "OpenRouter (E2E bot {{run_id}})",
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
  "title": "Bot Full {{run_id}}",
  "description": "HTTP full bot coverage",
  "type": "agent"
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}}"
jsonpath "$.type" == "agent"

GET {{base_url}}/api/v2/bots
HTTP 200
[Asserts]
body contains "\"id\":\"{{bot_id}}\""

GET {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.description" == "HTTP full bot coverage"

POST {{base_url}}/api/v2/collections
Content-Type: application/json
{
  "title": "Bot Collection {{run_id}}",
  "description": "HTTP bot config coverage",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": false,
    "enable_fulltext": false,
    "enable_knowledge_graph": false,
    "enable_summary": false,
    "enable_vision": false
  }
}
HTTP 200
[Captures]
collection_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{collection_id}}"
jsonpath "$.config.source" == "system"

PUT {{base_url}}/api/v2/bots/{{bot_id}}
Content-Type: application/json
{
  "title": "Bot Full {{run_id}} Updated",
  "description": "HTTP full bot coverage updated",
  "config": {
    "agent": {
      "completion": {
        "model_id": "{{completion_model_id}}",
        "temperature": 0.2
      },
      "system_prompt_template": "You are Bot Full {{run_id}}.",
      "query_prompt_template": "Answer the request: {query}",
      "collections": [
        {
          "id": "{{collection_id}}"
        }
      ]
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}} Updated"
jsonpath "$.description" == "HTTP full bot coverage updated"
jsonpath "$.config.agent.completion.model_id" == "{{completion_model_id}}"
jsonpath "$.config.agent.system_prompt_template" == "You are Bot Full {{run_id}}."
jsonpath "$.config.agent.query_prompt_template" == "Answer the request: {query}"
jsonpath "$.config.agent.collections[0].id" == "{{collection_id}}"

GET {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}} Updated"
jsonpath "$.config.agent.completion.model_id" == "{{completion_model_id}}"
jsonpath "$.config.agent.collections[0].id" == "{{collection_id}}"

DELETE {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
