# Deterministic retrieval contract coverage.
#
# Scoped to request-validation + empty-shape assertions so the suite is
# safe to run in `e2e-http-smoke`. Provider-dependent live search coverage
# (vector / fulltext / graph recall against a real embedding provider)
# lives elsewhere; this file must never depend on external LLM provider
# availability.

POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# Create a bare collection so the retrieval endpoints have a valid
# target. All indexing is disabled — the search request below is
# expected to fail at the "no strategy enabled" gate before any recall
# backend is touched, which keeps the suite deterministic.
POST {{base_url}}/api/v2/collections
Content-Type: application/json
{
  "title": "Retrieval Full {{run_id}}",
  "description": "HTTP retrieval coverage",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": false,
    "enable_fulltext": false,
    "enable_knowledge_graph": false,
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

# 1. GET /api/v2/collections/{id}/searches with no history — typed
# empty shape, never null / {} (lesson 9a).
GET {{base_url}}/api/v2/collections/{{collection_id}}/searches
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.items" exists

# 2. POST /api/v2/collections/{id}/searches with empty body — pydantic
# validation rejects the request (422) before it reaches the pipeline.
POST {{base_url}}/api/v2/collections/{{collection_id}}/searches
Content-Type: application/json
```
not json
```
HTTP 422

# 3. POST with a structurally valid body but zero enabled search
# strategies — the pipeline rejects with 400 (ValidationException
# surfaced by the exception handler).
POST {{base_url}}/api/v2/collections/{{collection_id}}/searches
Content-Type: application/json
{
  "query": "hello",
  "save_to_history": false,
  "rerank": false
}
HTTP 400

# 4. POST with missing query — pipeline requires a non-empty query.
POST {{base_url}}/api/v2/collections/{{collection_id}}/searches
Content-Type: application/json
{
  "query": "",
  "save_to_history": false,
  "rerank": false,
  "vector_search": { "topk": 1, "similarity": 0.5 }
}
HTTP 400

# 5. DELETE a non-existent search id on an existing collection — the
# route is idempotent so the response is 200 with a falsy body.
DELETE {{base_url}}/api/v2/collections/{{collection_id}}/searches/search_does_not_exist
HTTP 200

# 6. GET on an unknown collection — 404.
GET {{base_url}}/api/v2/collections/col_does_not_exist/searches
HTTP 404

# Clean up the collection so repeated runs stay hermetic.
DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204

# 7. Unauthenticated POST — fresh hurl run without cookie should 401.
# We assert this by explicitly logging out and re-firing.
POST {{base_url}}/api/v1/logout
HTTP 200

GET {{base_url}}/api/v2/collections/{{collection_id}}/searches
HTTP 401
