# Deterministic web_access contract coverage.
#
# Scoped to input-validation only. Every case is provider-independent so the
# suite is safe to run in `e2e-http-smoke`; provider-dependent live-fetch
# coverage (JINA / DuckDuckGo success paths) must live elsewhere and is out
# of scope here.

POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# 1. POST /api/v2/web/search with empty body: the route refuses when neither
# `query` nor `source` is supplied (HTTP 400, not pydantic 422).
POST {{base_url}}/api/v2/web/search
Content-Type: application/json
{}
HTTP 400

# 2. POST /api/v2/web/read with empty body: `url_list` is required.
POST {{base_url}}/api/v2/web/read
Content-Type: application/json
{}
HTTP 422

# 3. POST /api/v2/web/read with a `url` key instead of `url_list`:
# still missing the required `url_list` → pydantic 422.
POST {{base_url}}/api/v2/web/read
Content-Type: application/json
{
  "url": "not-a-url"
}
HTTP 422

# 4. POST /api/v2/web/search with malformed JSON body.
POST {{base_url}}/api/v2/web/search
Content-Type: application/json
```
not json
```
HTTP 422

POST {{base_url}}/api/v1/logout
HTTP 200
