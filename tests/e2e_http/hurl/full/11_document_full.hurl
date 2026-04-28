POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# v2 model-platform setup — every file is run independently, so the
# embedding / completion ``model_id`` references in the collection
# config must be resolved per file. ``ModelSpec`` no longer accepts
# ``model`` / ``model_service_provider`` / ``custom_llm_provider``
# — Pydantic silently drops extras, so any old-shape blob would parse
# as ``model_id=None`` and the indexing pipeline would break silently.
POST {{base_url}}/api/v2/model-accounts
Content-Type: application/json
{
  "provider_type": "dashscope",
  "name": "alibabacloud-doc-{{run_id}}",
  "display_name": "DashScope (E2E doc {{run_id}})",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "{{alibabacloud_api_key}}"
}
HTTP 200
[Captures]
dashscope_account_id: jsonpath "$.id"

POST {{base_url}}/api/v2/model-accounts
Content-Type: application/json
{
  "provider_type": "openai_compatible",
  "name": "openrouter-doc-{{run_id}}",
  "display_name": "OpenRouter (E2E doc {{run_id}})",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "{{openrouter_api_key}}"
}
HTTP 200
[Captures]
openrouter_account_id: jsonpath "$.id"

POST {{base_url}}/api/v2/models
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

POST {{base_url}}/api/v2/models
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
  "title": "Document Full {{run_id}}",
  "description": "HTTP full document lifecycle coverage",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": true,
    "enable_fulltext": true,
    "enable_knowledge_graph": false,
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
jsonpath "$.config.enable_vector" == true
jsonpath "$.config.enable_fulltext" == true

POST {{base_url}}/api/v2/collections/{{collection_id}}/documents/upload
[MultipartFormData]
file: file,tests/e2e_http/testdata/full-document.txt; type=text/plain
HTTP 200
[Captures]
document_id: jsonpath "$.document_id"
[Asserts]
jsonpath "$.document_id" == "{{document_id}}"
jsonpath "$.status" == "UPLOADED"
jsonpath "$.filename" == "tests/e2e_http/testdata/full-document.txt"

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents/staged
HTTP 200
[Asserts]
jsonpath "$.documents[0].document_id" == "{{document_id}}"
jsonpath "$.documents[0].status" == "UPLOADED"
jsonpath "$.total" == 1

POST {{base_url}}/api/v2/collections/{{collection_id}}/documents/confirm
Content-Type: application/json
{
  "document_ids": ["{{document_id}}"]
}
HTTP 200
[Asserts]
jsonpath "$.confirmed_count" == 1
jsonpath "$.failed_count" == 0

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{document_id}}"
jsonpath "$.items[0].name" == "tests/e2e_http/testdata/full-document.txt"
jsonpath "$.items[0].status" exists
jsonpath "$.items[0].fulltext_index_status" exists
jsonpath "$.total" == 1

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{document_id}}"
jsonpath "$.name" == "tests/e2e_http/testdata/full-document.txt"
jsonpath "$.status" exists
jsonpath "$.vector_index_status" exists
jsonpath "$.fulltext_index_status" exists
jsonpath "$.graph_index_status" == null
jsonpath "$.summary_index_status" == null
jsonpath "$.vision_index_status" == null

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}/preview
HTTP 200
[Asserts]
jsonpath "$.doc_object_path" == "original.txt"
jsonpath "$.doc_filename" == "tests/e2e_http/testdata/full-document.txt"
jsonpath "$.converted_pdf_object_path" == null
jsonpath "$.markdown_content" exists
jsonpath "$.chunks" exists
jsonpath "$.vision_chunks" exists

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents?search=full-document&sort_by=status&sort_order=asc
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{document_id}}"
jsonpath "$.items[0].name" == "tests/e2e_http/testdata/full-document.txt"
jsonpath "$.items[0].status" exists
jsonpath "$.items[0].fulltext_index_status" exists
jsonpath "$.total" == 1

POST {{base_url}}/api/v2/collections/{{collection_id}}/searches
Content-Type: application/json
{
  "query": "Hurl full document coverage",
  "fulltext_search": {
    "topk": 3
  },
  "save_to_history": true
}
HTTP 200
[Captures]
search_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{search_id}}"
jsonpath "$.query" == "Hurl full document coverage"
jsonpath "$.fulltext_search.topk" == 3
jsonpath "$.items" exists
jsonpath "$.created" exists

GET {{base_url}}/api/v2/collections/{{collection_id}}/searches
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{search_id}}"
jsonpath "$.items[0].query" == "Hurl full document coverage"
jsonpath "$.items[0].items" exists

DELETE {{base_url}}/api/v2/collections/{{collection_id}}/searches/{{search_id}}
HTTP 200

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}/download
HTTP 200
[Asserts]
header "content-disposition" contains "attachment"
header "content-disposition" contains "full-document.txt"

POST {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}/rebuild_indexes
Content-Type: application/json
{
  "index_types": ["FULLTEXT"]
}
HTTP 200

DELETE {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
