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
  "title": "Smoke Collection {{run_id}}",
  "description": "HTTP smoke collection",
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
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.type" == "document"
jsonpath "$.status" == "ACTIVE"
jsonpath "$.config.source" == "system"
jsonpath "$.config.enable_knowledge_graph" == false

GET {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{collection_id}}"
jsonpath "$.type" == "document"

GET {{base_url}}/api/v2/collections?page=1&page_size=50
HTTP 200
[Asserts]
header "content-type" contains "application/json"

PUT {{base_url}}/api/v2/collections/{{collection_id}}
Content-Type: application/json
{
  "title": "Smoke Collection {{run_id}} Updated",
  "description": "HTTP smoke collection updated",
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
[Asserts]
jsonpath "$.id" == "{{collection_id}}"
jsonpath "$.title" == "Smoke Collection {{run_id}} Updated"

DELETE {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 204

GET {{base_url}}/api/v2/collections/{{collection_id}}
HTTP 404
