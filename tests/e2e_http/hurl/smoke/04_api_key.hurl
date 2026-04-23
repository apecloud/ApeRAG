POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/api/v1/apikeys
Content-Type: application/json
{
  "description": "smoke key {{run_id}}"
}
HTTP 200
[Captures]
api_key_id: jsonpath "$.id"
api_key_value: jsonpath "$.key"
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.id" == "{{api_key_id}}"
jsonpath "$.key" == "{{api_key_value}}"
jsonpath "$.description" == "smoke key {{run_id}}"

GET {{base_url}}/api/v1/apikeys
HTTP 200

POST {{base_url}}/api/v1/logout
HTTP 200

GET {{base_url}}/api/v2/collections
Authorization: Bearer {{api_key_value}}
HTTP 200

POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

PUT {{base_url}}/api/v1/apikeys/{{api_key_id}}
Content-Type: application/json
{
  "description": "smoke key {{run_id}} updated"
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{api_key_id}}"
jsonpath "$.description" == "smoke key {{run_id}} updated"

DELETE {{base_url}}/api/v1/apikeys/{{api_key_id}}
HTTP 200

POST {{base_url}}/api/v1/logout
HTTP 200

GET {{base_url}}/api/v2/collections
Authorization: Bearer {{api_key_value}}
HTTP 401
