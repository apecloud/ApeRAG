POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

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
      "model": "text-embedding-v3",
      "model_service_provider": "alibabacloud",
      "custom_llm_provider": "openai"
    },
    "completion": {
      "model": "google/gemini-2.5-flash",
      "model_service_provider": "openrouter",
      "custom_llm_provider": "openrouter"
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
