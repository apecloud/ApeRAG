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
  "title": "Document Smoke {{run_id}}",
  "description": "HTTP document smoke collection",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": false,
    "enable_fulltext": false,
    "enable_knowledge_graph": false,
    "enable_summary": false,
    "enable_vision": false
  }
}
HTTP 200
[Captures]
collection_id: jsonpath "$.id"

POST {{base_url}}/api/v2/collections/{{collection_id}}/documents
[MultipartFormData]
files: file,tests/e2e_http/testdata/minimal-document.txt; type=text/plain
HTTP 200
[Captures]
document_id: jsonpath "$.items[0].id"
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.items[0].id" == "{{document_id}}"
jsonpath "$.items[0].name" == "tests/e2e_http/testdata/minimal-document.txt"
jsonpath "$.items[0].graph_index_status" == "SKIPPED"

GET {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{document_id}}"
jsonpath "$.name" == "tests/e2e_http/testdata/minimal-document.txt"
jsonpath "$.graph_index_status" == "SKIPPED"
jsonpath "$.summary_index_status" == "SKIPPED"
jsonpath "$.vision_index_status" == "SKIPPED"

DELETE {{base_url}}/api/v2/collections/{{collection_id}}/documents/{{document_id}}
HTTP 204

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204
