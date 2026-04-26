# Phase 9 D10.h (#100) — D10.c §A read primitive surface contract.
#
# Per ``docs/modularization/d10-design-pack.md`` §A, the canonical
# read primitive tool names + parameter names form the wire contract
# every external Agent (Claude Code / Codex / Cursor) must see on
# ``tools/list``. This hurl pins the parameter shape so a future
# rename / removal / reorder fails CI loudly before merge.
#
# The wire is JSON-RPC framed inside an SSE event, so the assertions
# use substring (``contains``) rather than jsonpath (which would skip
# the SSE envelope). The unit suite covers full Pydantic round-trip;
# this file is the cutover-time wire pin.

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
      "name": "d10-read-primitives-hurl",
      "version": "1.0.0"
    }
  }
}
HTTP 200

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

# §A.1 list_collections inputSchema markers — paginated read primitive.
body contains "\"list_collections\""
body contains "\"cursor\""
body contains "\"limit\""
body contains "\"sort_by\""
body contains "\"sort_order\""
body contains "\"title_filter\""

# §A.2 list_documents inputSchema markers — collection-scoped paginated.
body contains "\"list_documents\""
body contains "\"collection_id\""
body contains "\"type_filter\""
body contains "\"indexed_only\""

# §A.5 read_document range parameters — best-effort byte range, optional.
body contains "\"read_document\""
body contains "\"range_start\""
body contains "\"range_end\""

# §A.7 read_document_section path/anchor handles — stable handles per §A.9.
body contains "\"read_document_section\""

# §A.8 read_document_chunk — chunk_id stable handle per §A.9 + §E.6.
body contains "\"read_document_chunk\""
body contains "\"chunk_id\""

# §A.6 outline + §A.3/§A.4 metadata helpers all on the wire.
body contains "\"read_document_outline\""
body contains "\"get_collection_metadata\""
body contains "\"get_document_metadata\""
