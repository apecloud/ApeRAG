POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/api/v1/bots
Content-Type: application/json
{
  "title": "Bot Full {{run_id}}",
  "description": "HTTP full bot coverage",
  "type": "agent",
  "config": {
    "agent": {
      "completion": {
        "model": "google/gemini-2.5-flash",
        "model_service_provider": "openrouter",
        "custom_llm_provider": "openrouter",
        "temperature": 0.2
      }
    }
  }
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}}"
jsonpath "$.type" == "agent"
jsonpath "$.config.agent.completion.model" == "google/gemini-2.5-flash"

GET {{base_url}}/api/v1/bots
HTTP 200
[Asserts]
body contains "\"id\":\"{{bot_id}}\""

GET {{base_url}}/api/v1/bots/{{bot_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.description" == "HTTP full bot coverage"
jsonpath "$.type" == "agent"

PUT {{base_url}}/api/v1/bots/{{bot_id}}
Content-Type: application/json
{
  "title": "Bot Full {{run_id}} Updated",
  "description": "HTTP full bot coverage updated",
  "config": {
    "agent": {
      "completion": {
        "model": "google/gemini-2.5-flash",
        "model_service_provider": "openrouter",
        "custom_llm_provider": "openrouter",
        "temperature": 0.3
      },
      "system_prompt_template": "You are a concise assistant."
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}} Updated"
jsonpath "$.description" == "HTTP full bot coverage updated"
jsonpath "$.config.agent.completion.temperature" == 0.3
jsonpath "$.config.agent.system_prompt_template" == "You are a concise assistant."
