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
        "model_service_provider": "openrouter",
        "model": "google/gemini-2.5-flash",
        "custom_llm_provider": "openrouter",
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
jsonpath "$.config.agent.completion.model_service_provider" == "openrouter"
jsonpath "$.config.agent.completion.model" == "google/gemini-2.5-flash"
jsonpath "$.config.agent.completion.custom_llm_provider" == "openrouter"
jsonpath "$.config.agent.system_prompt_template" == "You are Bot Full {{run_id}}."
jsonpath "$.config.agent.query_prompt_template" == "Answer the request: {query}"
jsonpath "$.config.agent.collections[0].id" == "{{collection_id}}"

GET {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}} Updated"
jsonpath "$.config.agent.completion.model" == "google/gemini-2.5-flash"
jsonpath "$.config.agent.collections[0].id" == "{{collection_id}}"

DELETE {{base_url}}/api/v2/bots/{{bot_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
