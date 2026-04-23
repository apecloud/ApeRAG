# Deterministic knowledge-graph contract coverage (v2 domain hard-cut).
#
# Exercises the read paths + curation-run list on a collection with
# ``enable_knowledge_graph=true`` but without any indexed content, so
# every response is the typed empty shape (``{ "labels": [] }``,
# ``{ "nodes": [], "edges": [], "is_truncated": false }``, ``{ "run":
# null, "suggestions": [] }``). The full curation write flow (starting
# a run + asserting suggestions) depends on an LLM provider and is
# deferred to Phase 3 provider/full coverage.

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

POST {{base_url}}/api/v1/logout
HTTP 200
