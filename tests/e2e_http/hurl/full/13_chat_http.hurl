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
  "type": "agent",
  "config": {
    "agent": {
      "completion": {
        "model": "google/gemini-2.5-flash",
        "model_service_provider": "openrouter",
        "custom_llm_provider": "openrouter",
        "temperature": 0.2
      },
      "system_prompt_template": "You are a concise assistant."
    }
  }
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.type" == "agent"
jsonpath "$.config.agent.completion.model" == "google/gemini-2.5-flash"

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

POST {{base_url}}/api/v1/chat/completions/frontend?stream=false&bot_id={{bot_id}}&chat_id={{chat_id}}
Content-Type: text/plain
msg_id: chat_http_{{run_id}}
`Summarize ApeRAG in one sentence.`
HTTP 200
[Asserts]
jsonpath "$.type" exists
jsonpath "$.id" exists
jsonpath "$.data" exists
jsonpath "$.timestamp" exists
