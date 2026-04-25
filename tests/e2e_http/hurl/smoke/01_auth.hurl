POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.username" == "{{username}}"
jsonpath "$.is_active" == true

GET {{base_url}}/api/v2/auth/user
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.username" == "{{username}}"
jsonpath "$.is_active" == true

POST {{base_url}}/api/v2/auth/logout
HTTP 200
