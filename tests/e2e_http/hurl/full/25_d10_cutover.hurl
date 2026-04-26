# Phase 9 D10.h (#100) — legacy search tool cutover wire-shape contract.
#
# Per ``docs/modularization/d10-design-pack.md`` §H + amendment-#2
# Drift #2/#3, D10.h hard-cuts the omnibus
# ``search_collection`` / ``search_chat_files`` tools and keeps only
# the canonical D10.c read + D10.d split-search surface. This hurl
# pins the cutover on the wire so a future regression that
# accidentally re-registers either legacy tool fails CI loudly.
#
# We use substring assertions on the SSE-framed JSON-RPC body for
# the same reason 21_d10_capabilities.hurl does — the wire is
# ``event: message\ndata: {...}`` and jsonpath against the unframed
# JSON would skip the SSE envelope entirely.

POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# 1. MCP initialize handshake — needed before tools/list will emit
# any annotation metadata, even on a stateless mount.
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
      "name": "d10-cutover-hurl",
      "version": "1.0.0"
    }
  }
}
HTTP 200

# 2. tools/list — the cutover assertion battery. Each canonical D10
# tool name MUST still be present; the two legacy omnibus tools
# MUST NOT be present.
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
[Asserts]

# Canonical surface still on the wire — these are the names a
# post-cutover external Agent should discover.
body contains "\"list_collections\""
body contains "\"list_documents\""
body contains "\"get_collection_metadata\""
body contains "\"get_document_metadata\""
body contains "\"read_document\""
body contains "\"read_document_outline\""
body contains "\"read_document_section\""
body contains "\"read_document_chunk\""
body contains "\"vector_search\""
body contains "\"graph_search\""
body contains "\"fulltext_search\""
body contains "\"web_search\""
body contains "\"web_read\""

# Hard-cut guard — both legacy omnibus tools were removed in #100.
# A future regression that accidentally re-registers either tool
# (e.g., a copy/paste from history, a revert PR) will fail this
# substring assertion and surface the issue in CI before merge.
body not contains "\"search_collection\""
body not contains "\"search_chat_files\""
