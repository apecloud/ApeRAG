POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

PUT {{base_url}}/api/v1/llm_providers/alibabacloud
Content-Type: application/json
{
  "api_key": "{{alibabacloud_api_key}}"
}
HTTP 200
[Asserts]
jsonpath "$.name" == "alibabacloud"

PUT {{base_url}}/api/v1/llm_providers/openrouter
Content-Type: application/json
{
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Asserts]
jsonpath "$.name" == "openrouter"

PUT {{base_url}}/api/v1/default_models
Content-Type: application/json
{
  "defaults": [
    {
      "scenario": "default_for_collection_completion",
      "provider_name": "openrouter",
      "model": "google/gemini-2.5-flash",
      "custom_llm_provider": "openrouter"
    },
    {
      "scenario": "default_for_embedding",
      "provider_name": "alibabacloud",
      "model": "text-embedding-v3",
      "custom_llm_provider": "openai"
    },
    {
      "scenario": "default_for_rerank",
      "provider_name": "alibabacloud",
      "model": "gte-rerank-v2",
      "custom_llm_provider": "alibabacloud"
    }
  ]
}
HTTP 200
[Asserts]
jsonpath "$.items[0].scenario" == "default_for_collection_completion"
jsonpath "$.items[0].provider_name" == "openrouter"
jsonpath "$.items[2].scenario" == "default_for_embedding"
jsonpath "$.items[2].provider_name" == "alibabacloud"
jsonpath "$.items[3].scenario" == "default_for_rerank"
jsonpath "$.items[3].provider_name" == "alibabacloud"

POST {{base_url}}/api/v1/embeddings
Content-Type: application/json
{
  "provider": "alibabacloud",
  "model": "text-embedding-v3",
  "input": [
    "ApeRAG is a retrieval augmented generation platform.",
    "Embeddings convert text into vectors."
  ]
}
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.model" == "text-embedding-v3"
jsonpath "$.data[0].index" == 0
jsonpath "$.data[1].index" == 1
jsonpath "$.usage.total_tokens" > 0

POST {{base_url}}/api/v1/rerank
Content-Type: application/json
{
  "provider": "alibabacloud",
  "model": "gte-rerank-v2",
  "query": "capital of France",
  "documents": [
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan.",
    "Pasta is often served with tomato sauce."
  ],
  "top_k": 2,
  "return_documents": true
}
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.model" == "gte-rerank-v2"
jsonpath "$.data[0].index" == 0
jsonpath "$.data[0].document.text" == "Paris is the capital of France."
jsonpath "$.usage.total_tokens" > 0
