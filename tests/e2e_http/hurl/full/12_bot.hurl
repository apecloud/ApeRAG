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
  "type": "common"
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}}"
jsonpath "$.type" == "common"

GET {{base_url}}/api/v1/bots
HTTP 200
[Asserts]
body contains "\"id\":\"{{bot_id}}\""

GET {{base_url}}/api/v1/bots/{{bot_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.description" == "HTTP full bot coverage"

PUT {{base_url}}/api/v1/bots/{{bot_id}}
Content-Type: application/json
{
  "title": "Bot Full {{run_id}} Updated",
  "description": "HTTP full bot coverage updated"
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.title" == "Bot Full {{run_id}} Updated"
jsonpath "$.description" == "HTTP full bot coverage updated"

PUT {{base_url}}/api/v1/bots/{{bot_id}}/flow
Content-Type: application/json
{
  "name": "basic_flow",
  "title": "Basic Flow",
  "description": "Minimal non-streaming flow for HTTP E2E",
  "version": "1.0.0",
  "nodes": [
    {
      "id": "start",
      "type": "start",
      "title": "Start",
      "data": {
        "input": {
          "schema": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string"
              }
            },
            "required": ["query"]
          },
          "values": {
            "query": "hello"
          }
        },
        "output": {
          "schema": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string"
              }
            },
            "required": ["query"]
          }
        }
      }
    },
    {
      "id": "llm_1",
      "type": "llm",
      "title": "LLM",
      "data": {
        "input": {
          "schema": {
            "type": "object",
            "properties": {
              "model_service_provider": {
                "type": "string"
              },
              "model_name": {
                "type": "string"
              },
              "custom_llm_provider": {
                "type": "string"
              },
              "prompt_template": {
                "type": "string"
              },
              "temperature": {
                "type": "number"
              },
              "query": {
                "type": "string"
              }
            },
            "required": [
              "model_service_provider",
              "model_name",
              "custom_llm_provider",
              "prompt_template",
              "temperature",
              "query"
            ]
          },
          "values": {
            "model_service_provider": "openrouter",
            "model_name": "google/gemini-2.5-flash",
            "custom_llm_provider": "openrouter",
            "prompt_template": "{query}",
            "temperature": 0.2,
            "query": "hello"
          }
        },
        "output": {
          "schema": {
            "type": "object",
            "properties": {
              "text": {
                "type": "string"
              }
            }
          }
        }
      }
    }
  ],
  "edges": [
    {
      "source": "start",
      "target": "llm_1"
    }
  ]
}
HTTP 200
[Asserts]
jsonpath "$.name" == "basic_flow"
jsonpath "$.nodes[0].type" == "start"
jsonpath "$.nodes[1].type" == "llm"

GET {{base_url}}/api/v1/bots/{{bot_id}}/flow
HTTP 200
[Asserts]
jsonpath "$.name" == "basic_flow"
jsonpath "$.version" == "1.0.0"
