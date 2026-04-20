POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/api/v1/available_models
Content-Type: application/json
{}
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.items" exists

POST {{base_url}}/api/v1/available_models
Content-Type: application/json
{
  "tag_filters": []
}
HTTP 200
[Asserts]
jsonpath "$.items" exists

POST {{base_url}}/api/v1/available_models
Content-Type: application/json
{
  "tag_filters": [
    {
      "operation": "INVALID",
      "tags": ["recommend"]
    }
  ]
}
HTTP 422
[Asserts]
jsonpath "$.detail" exists
