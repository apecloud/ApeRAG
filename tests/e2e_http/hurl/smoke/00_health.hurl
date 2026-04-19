GET {{base_url}}/health
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.status" == "healthy"
jsonpath "$.service" == "aperag-api"
