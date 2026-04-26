# Phase 9 D10.h (#100) — D10.e §C cursor pagination wire contract.
#
# Per ``docs/modularization/d10-design-pack.md`` §C the paginated
# read primitives (list_collections / list_documents) accept a
# ``cursor`` parameter and return ``next_cursor`` / ``total_count``
# alongside ``items``. The cursor codec lives in
# ``aperag/mcp/cursor`` (#1712) and the wiring helper is in
# ``aperag/service/pagination`` (#1715); this hurl pins their shape
# on the MCP wire.
#
# Body is JSON-RPC framed inside SSE, so we use substring assertions
# (mirrors 21_d10_capabilities and 22_d10_read_primitives).

POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/mcp/
Content-Type: application/json
Accept: application/json, text/event-stream
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "d10-pagination-hurl",
      "version": "1.0.0"
    }
  }
}
HTTP 200

# tools/list — assert the cursor wiring is exposed on every
# paginated tool's inputSchema, and the result envelope is
# discoverable via outputSchema (FastMCP includes ``returns`` /
# ``outputSchema`` on registered tools).
POST {{base_url}}/mcp/
Content-Type: application/json
Accept: application/json, text/event-stream
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
HTTP 200

# §C.5 PaginationParams input markers — cursor + limit on every
# paginated read primitive.
body contains "\"list_collections\""
body contains "\"list_documents\""
body contains "\"cursor\""
body contains "\"limit\""

# §C.5 PaginationResult output markers — items / next_cursor /
# total_count on every paginated tool's outputSchema. FastMCP emits
# the Pydantic model JSON schema, so the field names round-trip.
body contains "\"items\""
body contains "\"next_cursor\""
body contains "\"total_count\""
