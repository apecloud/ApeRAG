# Phase 9 D10.h (#100) — D10.d §B split search surface contract.
#
# Per ``docs/modularization/d10-design-pack.md`` §B the omnibus
# ``search_collection`` / ``search_chat_files`` tools were retired in
# this lane; only the split surface (``vector_search`` /
# ``graph_search`` / ``fulltext_search`` / ``web_search``) remains
# on the MCP wire. ``web_search`` carries the §B.4 canonical
# parameter names (``top_k`` / kw-only / ``source: str | None``)
# applied in #100; the legacy ``max_results`` parameter must not
# appear.
#
# Body is JSON-RPC framed inside SSE, so we use substring
# assertions (mirrors 21 / 22 / 23).

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
      "name": "d10-search-split-hurl",
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
[Asserts]

# §B.1 / §B.2 / §B.3 collection-scoped split search inputSchema
# markers — every tool keeps the per-tool ``query`` + collection
# scoping. Search tools do not expose cursor until real search
# pagination exists.
body contains "\"vector_search\""
body contains "\"graph_search\""
body contains "\"fulltext_search\""
body contains "\"query\""
body contains "\"collection_id\""

# §B.4 web_search canonical parameter set — D10.h #100 cutover
# replaces ``max_results`` with ``top_k`` and switches ``source`` to
# ``str | None``. The wire MUST carry ``top_k`` and MUST NOT carry
# the legacy ``max_results``.
body contains "\"web_search\""
body contains "\"top_k\""
body not contains "\"max_results\""

# Drift #1 (amendment-#2): SearchResultItem.metadata is meant to
# expose the three stable handles ``chunk_id`` / ``section_path`` /
# ``heading_anchor``. The MCP wire from FastMCP cannot prove that
# here — the split-search tools return ``Dict[str, Any]`` so their
# outputSchema is just ``additionalProperties: true`` and a substring
# match against ``tools/list`` would be satisfied by unrelated read
# primitive input schemas (Weston msg=8a691444 false-positive
# warning). The Pydantic-level pin lives in the new
# ``tests/unit_test/domains/retrieval/test_search_result_metadata.py``
# alongside the schema change, which is the right level for this
# guarantee.
