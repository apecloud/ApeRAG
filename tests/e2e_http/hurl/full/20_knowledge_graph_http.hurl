# Deterministic knowledge-graph contract coverage (v2 domain hard-cut).
#
# Exercises the read paths + curation-run list on a collection with
# ``enable_knowledge_graph=true`` but without any indexed content, so
# every response is the typed empty shape (``{ "labels": [] }``,
# ``{ "nodes": [], "edges": [], "is_truncated": false }``, ``{ "run":
# null, "suggestions": [] }``). The full curation write flow (starting
# a run + asserting suggestions) depends on an LLM provider and is
# deferred to Phase 3 provider/full coverage.

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
  "name": "alibabacloud-kg-{{run_id}}",
  "display_name": "DashScope (E2E kg {{run_id}})",
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
  "name": "openrouter-kg-{{run_id}}",
  "display_name": "OpenRouter (E2E kg {{run_id}})",
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
  "title": "Knowledge Graph Full {{run_id}}",
  "description": "HTTP knowledge-graph coverage",
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

# 1. v2 labels read — empty list when no content is indexed (lesson 9a).
GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs/labels
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.labels" exists

# 2. v2 subgraph read — overview mode, default window.
GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs?label=*&max_nodes=50&max_depth=2
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.nodes" exists
jsonpath "$.edges" exists
jsonpath "$.is_truncated" == false

# 3. v2 subgraph read — out-of-range max_nodes -> 400 (route-level
# validation, before the graph backend is touched).
GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs?label=*&max_nodes=0&max_depth=2
HTTP 400
[Asserts]
body contains "max_nodes must be between 1 and 10000"

# 4. v2 subgraph read — out-of-range max_depth -> 400.
GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs?label=*&max_nodes=10&max_depth=99
HTTP 400
[Asserts]
body contains "max_depth must be between 1 and 10"

# 5. v2 merge-suggestions GET when no run has been scheduled — typed
# empty shape.
GET {{base_url}}/api/v2/collections/{{collection_id}}/graphs/merge-suggestions
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.suggestions" exists

# 6. v2 nodes merge with too few entity_ids — 400.
POST {{base_url}}/api/v2/collections/{{collection_id}}/graphs/nodes/merge
Content-Type: application/json
{
  "entity_ids": ["only_one"]
}
HTTP 400
[Asserts]
body contains "entity_ids"

# 7. v2 suggestion action on a missing suggestion id — 404.
POST {{base_url}}/api/v2/collections/{{collection_id}}/graphs/merge-suggestions/suggestion_does_not_exist/action
Content-Type: application/json
{
  "action": "reject"
}
HTTP 404

# 8. v2 labels read on an unknown collection — 404.
GET {{base_url}}/api/v2/collections/col_does_not_exist/graphs/labels
HTTP 404

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204

POST {{base_url}}/api/v2/auth/logout
HTTP 200
