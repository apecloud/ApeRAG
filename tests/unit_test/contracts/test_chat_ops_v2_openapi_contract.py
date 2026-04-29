"""Phase 8 #53 G4d-BE contract: conversation chat ops mounted at /api/v2.

Locks the architect canonical (Option A, msg≈2026-04-25 16:18 in
#模块化重构): chat_router prefix migrated from ``/api/v1`` to
``/api/v2`` while keeping chat-rooted nesting unchanged.
"""

from __future__ import annotations

from aperag.app import app
from aperag.openapi_spec import build_full_openapi_spec

# Per-method canonical chat-rooted paths owned by ``chat_router`` (tags=["chats"]).
# bot-rooted chat CRUD lives in ``bots_router`` (tags=["bots-v2"]) and is out of scope.
CHAT_OPS_V2_PATHS_BY_METHOD = {
    "get": {
        "/api/v2/chats/{chat_id}/feedback",
        "/api/v2/chats/{chat_id}/documents/{document_id}",
    },
    "post": {
        "/api/v2/chats/{chat_id}/turns/{turn_id}/feedback",
        "/api/v2/chats/{chat_id}/search",
        "/api/v2/chats/{chat_id}/documents",
    },
    "delete": {
        "/api/v2/chats/{chat_id}/turns/{turn_id}/feedback",
    },
}


def _chat_router_paths_by_method(spec: dict) -> dict[str, set[str]]:
    """Collect paths from the spec keyed by method, restricted to the ``chats`` tag."""
    out: dict[str, set[str]] = {"get": set(), "post": set(), "delete": set()}
    for path, operations in (spec.get("paths") or {}).items():
        if not isinstance(operations, dict):
            continue
        for method in ("get", "post", "delete"):
            op = operations.get(method)
            if not isinstance(op, dict):
                continue
            tags = op.get("tags") or []
            if "chats" in tags:
                out[method].add(path)
    return out


def test_chat_ops_mounted_under_api_v2():
    """Every chat ops route must surface at /api/v2/chats/... (chat-rooted)."""
    spec = build_full_openapi_spec(app)
    by_method = _chat_router_paths_by_method(spec)
    for method, expected_paths in CHAT_OPS_V2_PATHS_BY_METHOD.items():
        actual_paths = by_method[method]
        missing = expected_paths - actual_paths
        assert not missing, (
            f"chat ops {method.upper()} route(s) missing from /api/v2: {sorted(missing)}. "
            "Did the chat_router mount prefix get reverted to /api/v1?"
        )


def test_chat_ops_no_v1_ghost():
    """No chat_router route may remain under /api/v1 after G4d hard-cut."""
    spec = build_full_openapi_spec(app)
    by_method = _chat_router_paths_by_method(spec)
    all_paths = set().union(*by_method.values())
    v1_chat_ops = {p for p in all_paths if p.startswith("/api/v1/chats/")}
    assert not v1_chat_ops, f"v1 chat_router ghost paths still present after G4d hard-cut: {sorted(v1_chat_ops)}"


def test_chat_ops_chat_rooted_nesting_preserved():
    """G4d Option A: chat-rooted (/chats/{chat_id}/...), NOT bot-rooted."""
    spec = build_full_openapi_spec(app)
    by_method = _chat_router_paths_by_method(spec)
    all_paths = set().union(*by_method.values())
    bot_rooted = {p for p in all_paths if "/bots/" in p}
    assert not bot_rooted, f"chat ops were re-nested under /bots/ — Option A canonical violation: {sorted(bot_rooted)}"
