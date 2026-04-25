POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# v3 model-platform setup — every file is run independently and
# ``ModelSpec`` only accepts ``model_id`` now, so the collection config
# below references freshly-seeded embedding / completion models.
POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "dashscope",
  "name": "alibabacloud-graph-{{run_id}}",
  "display_name": "DashScope (E2E graph {{run_id}})",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "{{alibabacloud_api_key}}"
}
HTTP 200
[Captures]
dashscope_account_id: jsonpath "$.id"

POST {{base_url}}/api/v3/model-accounts
Content-Type: application/json
{
  "provider_type": "openai_compatible",
  "name": "openrouter-graph-{{run_id}}",
  "display_name": "OpenRouter (E2E graph {{run_id}})",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Captures]
openrouter_account_id: jsonpath "$.id"

POST {{base_url}}/api/v3/models
Content-Type: application/json
{
  "account_id": "{{dashscope_account_id}}",
  "provider_model_id": "text-embedding-v3",
  "display_name": "DashScope text-embedding-v3",
  "capability": "embedding",
  "embedding_dimensions": 1024
}
HTTP 200
[Captures]
embedding_model_id: jsonpath "$.id"

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

POST {{base_url}}/api/v2/collections
Content-Type: application/json
{
  "title": "Graph Full {{run_id}}",
  "description": "HTTP graph coverage",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": false,
    "enable_fulltext": false,
    "enable_knowledge_graph": true,
    "enable_summary": false,
    "enable_vision": false,
    "embedding": {
      "model_id": "{{embedding_model_id}}"
    },
    "completion": {
      "model_id": "{{completion_model_id}}"
    }
  }
}
HTTP 200
[Captures]
collection_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{collection_id}}"
jsonpath "$.config.enable_knowledge_graph" == true

GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs/labels
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.labels" exists

GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs?label=*&max_nodes=50&max_depth=2
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.nodes" exists
jsonpath "$.edges" exists
jsonpath "$.is_truncated" == false

GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs?label=*&max_nodes=0&max_depth=2
HTTP 400
[Asserts]
header "content-type" contains "application/json"
body contains "max_nodes must be between 1 and 10000"

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
