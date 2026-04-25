# Phase 9 D10.f (#98) — capability annotation wire-shape contract.
#
# Per ``docs/modularization/d10-design-pack.md`` §D Option A canonical:
# every D10 tool registered through ``aperag/mcp/tools/_annotations.py``
# must carry its ``ToolAnnotation`` payload (``requires`` /
# ``capabilities`` / ``deprecated`` / ``deprecated_until`` /
# ``fallback_to``) onto the MCP wire so external Agents (Claude Code /
# Codex / Cursor) can do client-side filtering without server-side
# session state.
#
# This hurl exercises the FastMCP HTTP mount at ``/mcp/`` end-to-end:
#
#   1. ``initialize`` handshake (stateless_http=True still requires the
#      MCP-Protocol-Version negotiation so the server emits annotation
#      metadata for the matching protocol revision).
#   2. ``tools/list`` returns the full surface — assert each D10 tool
#      name appears alongside the §D.1 annotation field markers.
#
# We deliberately use substring (``contains``) assertions on the body
# rather than ``jsonpath`` because the wire is JSON-RPC framed inside an
# SSE event (``event: message\ndata: {...}``), so a raw text contains is
# the most robust assertion that survives transport framing changes.

POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

# 1. MCP initialize handshake. FastMCP's ``stateless_http=True`` mount
# still requires this so it can negotiate protocol version + capability
# flags before serving ``tools/list``. We don't need to keep the
# session id around for stateless mode but the Mcp-Session-Id may be
# echoed back on the response — hurl is fine with either shape.
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
      "name": "d10-capabilities-hurl",
      "version": "1.0.0"
    }
  }
}
HTTP 200

# 2. tools/list — the assertion battery for §D.1 schema appearance on
# the wire. Body is JSON-RPC inside an SSE frame, hence the substring
# checks (rather than jsonpath against an unframed JSON document).
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
# §D.1 annotation envelope keys appear at least once.
body contains "\"requires\""
body contains "\"capabilities\""
body contains "\"deprecated\""
body contains "\"deprecated_until\""
body contains "\"fallback_to\""

# Every D10.c read primitive name lands on the wire.
body contains "\"list_collections\""
body contains "\"list_documents\""
body contains "\"get_collection_metadata\""
body contains "\"get_document_metadata\""
body contains "\"read_document\""
body contains "\"read_document_outline\""
body contains "\"read_document_section\""
body contains "\"read_document_chunk\""

# Every D10.d split search primitive name lands on the wire.
body contains "\"vector_search\""
body contains "\"graph_search\""
body contains "\"fulltext_search\""
body contains "\"web_search\""

# §D.1 capability discriminators surface (graph_search needs graph
# index, fulltext_search needs fulltext index, web_search needs web
# access). Hurl substring assertions can't bind these to the right
# tool; the unit suite (test_capability_negotiation.py) does that
# binding. Here we just confirm the keys appear so an external Agent's
# client-side filter has something to read.
body contains "\"graph_index\""
body contains "\"fulltext_index\""
body contains "\"web_access\""
body contains "\"long_context\""

# §D.1 ``requires`` value markers (collection_access for collection-
# scoped tools, web_access for the web search tool).
body contains "\"collection_access\""

# §D.1 ``deprecated_until`` is null on every D10 tool today (none are
# on the deprecation timeline yet). The presence assertion above plus
# this null check confirms the field is present and shaped correctly.
body contains "\"deprecated_until\":null"
