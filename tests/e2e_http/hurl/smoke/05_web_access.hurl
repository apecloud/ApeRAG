# Soft-fail aware contract coverage for the web_access domain (search +
# read endpoints). Both endpoints are graceful by design: external
# provider failures (no JINA key, DuckDuckGo unreachable, target URL
# unfetchable, etc.) are absorbed and surface as a 200 response with
# diagnostic ``meta`` / ``failed`` counters rather than 5xx. That makes
# them safe to invoke from the smoke lane: assertions verify schema +
# 200 + status-enum membership, never that a particular provider
# returned non-empty results. Provider-flavored live-fetch coverage
# (asserting JINA returned >0 results) belongs in the provider lane,
# not here.
#
# Negative-only validation coverage already lives in
# ``tests/e2e_http/hurl/full/18_web_access_http.hurl`` (4 cases for
# 400/422 boundaries). This file is the positive counterpart.

POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# 1. /web/search with a query — graceful 200 even if external providers
# are unreachable / return empty / are disabled. Asserts the response
# envelope is well-formed and ``meta.search_status`` is one of the
# Literal values the schema declares (``ok`` / ``empty`` /
# ``unavailable`` / ``disabled``).
POST {{base_url}}/api/v2/web/search
Content-Type: application/json
{
  "query": "smoke web search {{run_id}}",
  "max_results": 3
}
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.query" == "smoke web search {{run_id}}"
jsonpath "$.results" isCollection
jsonpath "$.total_results" isInteger
jsonpath "$.search_time" isFloat
jsonpath "$.meta.search_status" matches "^(ok|empty|unavailable|disabled)$"
jsonpath "$.meta.provider_used" isCollection
jsonpath "$.meta.backend_used" isCollection
jsonpath "$.meta.fallback_used" isBoolean

# 2. /web/search with a ``source`` (site browse) instead of a free-text
# query. The route normalises the response ``query`` field to
# ``site:<source>`` per ``_build_search_response_query``.
POST {{base_url}}/api/v2/web/search
Content-Type: application/json
{
  "source": "example.com",
  "max_results": 3
}
HTTP 200
[Asserts]
jsonpath "$.query" == "site:example.com"
jsonpath "$.results" isCollection
jsonpath "$.meta.search_status" matches "^(ok|empty|unavailable|disabled)$"

# 3. /web/read with a single URL — ``total_urls`` must reflect the
# input length and ``successful + failed`` must equal it. The
# Trafilatura fallback path runs when no per-user JINA key is
# configured, so the test does not require any provider key to pass.
POST {{base_url}}/api/v2/web/read
Content-Type: application/json
{
  "url_list": ["https://example.com"],
  "timeout": 10
}
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.results" isCollection
jsonpath "$.total_urls" == 1
jsonpath "$.successful" isInteger
jsonpath "$.failed" isInteger
jsonpath "$.processing_time" isFloat
jsonpath "$.results[0].url" == "https://example.com"
jsonpath "$.results[0].status" matches "^(success|error)$"

POST {{base_url}}/api/v2/auth/logout
HTTP 200
