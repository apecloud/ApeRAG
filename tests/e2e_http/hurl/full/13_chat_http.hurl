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
  "title": "Chat Bot {{run_id}}",
  "description": "HTTP full chat coverage",
  "type": "common"
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"

PUT {{base_url}}/api/v1/bots/{{bot_id}}/flow
Content-Type: application/json
{
  "name": "basic_flow",
  "title": "Basic Flow",
  "description": "Minimal non-streaming flow for chat HTTP E2E",
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
            "model_service_provider": "siliconflow",
            "model_name": "Qwen/Qwen3-8B",
            "custom_llm_provider": "openai",
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

POST {{base_url}}/api/v1/bots/{{bot_id}}/chats
HTTP 200
[Captures]
chat_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.bot_id" == "{{bot_id}}"

GET {{base_url}}/api/v1/bots/{{bot_id}}/chats?page=1&page_size=20
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{chat_id}}"

PUT {{base_url}}/api/v1/bots/{{bot_id}}/chats/{{chat_id}}
Content-Type: application/json
{
  "title": "Chat Title {{run_id}}"
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.title" == "Chat Title {{run_id}}"

GET {{base_url}}/api/v1/bots/{{bot_id}}/chats/{{chat_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{chat_id}}"
jsonpath "$.title" == "Chat Title {{run_id}}"
