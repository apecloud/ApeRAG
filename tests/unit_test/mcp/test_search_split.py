# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 9 D10.d (#96) — search primitives split contract tests.

Locks the public surface of the ``aperag.mcp.tools.search_*`` modules
introduced in D10.d:

- 4 split tools (``vector_search`` / ``graph_search`` / ``fulltext_search``
  / ``web_search``) exist with the §B canonical signatures.
- They are registered under ``mcp_server`` via the ``@mcp_server.tool``
  decorator chain at module import time (server.py re-export gate).
- ``search_collection`` and ``search_chat_files`` carry the
  ``[DEPRECATED]`` banner in their docstrings per §B.5 / §H.1 / §H.2 —
  but their implementation body is untouched (Forbidden per §G D10.d).
- ``search_collection`` keeps hitting ``/api/v2/collections/{id}/searches``
  during the deprecation window (no body changes).

Caller-migration assertion semantics (§G hard gate #4): backward
compatibility for existing ``mcp_server.web_search`` attribute access
(legacy test path) is preserved by re-exporting ``web_search`` at the
server module level.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

# Importing ``aperag.mcp.server`` triggers the bottom-of-module
# registration block which loads ``aperag.mcp.tools.search_*`` and
# triggers ``@mcp_server.tool`` decorators. The tool functions are
# re-exported at the server module level for backward-compat attribute
# access (see ``aperag/mcp/server.py`` re-export comment); we read the
# symbols off the server module to avoid the partial-load cycle that
# would otherwise occur if the search modules were imported via
# ``aperag.mcp.tools`` package's ``__init__``.
from aperag.mcp import server as mcp_server_module
from aperag.mcp.server import (
    fulltext_search,
    graph_search,
    vector_search,
    web_search,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MCP_SERVER_PATH = REPO_ROOT / "aperag" / "mcp" / "server.py"


# --- 1. Existence / import surface ----------------------------------


def test_split_search_tools_exist_in_tools_package():
    """All 4 D10.d split tools are importable from ``aperag.mcp.tools``."""

    for tool in (vector_search, graph_search, fulltext_search, web_search):
        assert callable(tool), f"{tool.__name__} must be callable"
        assert inspect.iscoroutinefunction(tool), f"{tool.__name__} must be ``async def`` (MCP tool surface)."


def test_split_search_tools_re_exported_at_server_module_level():
    """Server module re-exports split tools for backward compat with
    legacy callers reading ``aperag.mcp.server.web_search`` etc.
    """

    for name in ("vector_search", "graph_search", "fulltext_search", "web_search"):
        assert hasattr(mcp_server_module, name), (
            f"aperag.mcp.server must re-export {name} for backward compatibility with legacy attribute access."
        )


# --- 2. §B signature shape lock --------------------------------------


def _params(func) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(func).parameters)


def _kw_only_params(func) -> set[str]:
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def test_vector_search_signature_matches_b1_lock():
    """``vector_search`` signature follows §B.1: ``collection_id`` +
    ``query`` positional, then a ``*,`` kw-only barrier with
    ``top_k`` / ``similarity_threshold`` / ``rerank`` / ``cursor``.
    The kw-only enforcement aligns with the D10.c precedent (per
    `[D10 spec amendment]` msg=b9b7072a Drift #5 resolution).
    """
    params = _params(vector_search)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only_params(vector_search) == {
        "top_k",
        "similarity_threshold",
        "rerank",
        "cursor",
    }, "§B.1 requires kw-only `top_k / similarity_threshold / rerank / cursor`."
    assert params["top_k"].default == 5
    # similarity_threshold defaults to None to mean "use collection default".
    assert params["similarity_threshold"].default is None
    assert params["rerank"].default is True
    # cursor placeholder per amendment msg=b9b7072a Drift #4 (c).
    assert params["cursor"].default is None


def test_graph_search_signature_matches_b2_lock():
    """``graph_search`` signature follows §B.2 + amendment msg=b9b7072a:
    collection_id + query positional then kw-only top_k + cursor.
    The §B.2 spec mentions ``depth`` / ``entity_types`` as
    forward-looking knobs; first-cut keeps them out until the backend
    surface for graph traversal is wired (deferred to D10.d follow-up).
    """
    params = _params(graph_search)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only_params(graph_search) == {"top_k", "cursor"}, "§B.2 requires kw-only `top_k / cursor`."
    assert params["top_k"].default == 5
    assert params["cursor"].default is None


def test_fulltext_search_signature_matches_b3_lock():
    """``fulltext_search`` signature follows §B.3 + amendment
    msg=b9b7072a: collection_id + query positional then kw-only
    top_k + keywords + rerank + cursor.
    """
    params = _params(fulltext_search)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only_params(fulltext_search) == {
        "top_k",
        "keywords",
        "rerank",
        "cursor",
    }, "§B.3 requires kw-only `top_k / keywords / rerank / cursor`."
    assert params["top_k"].default == 5
    assert params["keywords"].default is None
    assert params["rerank"].default is True
    assert params["cursor"].default is None


def test_web_search_signature_matches_b4_canonical():
    """``web_search`` carries the §B.4 canonical signature applied in
    D10.h #100: ``query`` is positional + required; every other
    parameter is keyword-only; the result-count limit is named
    ``top_k``; ``source`` is ``str | None``.
    """
    import inspect

    sig = inspect.signature(web_search)
    params = sig.parameters

    assert params["query"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["query"].default is inspect.Parameter.empty, "query must be required"
    for kw_name in ("top_k", "timeout", "locale", "source"):
        assert kw_name in params, f"{kw_name!r} missing"
        assert params[kw_name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{kw_name!r} must be keyword-only after the §B.4 cutover"
        )

    assert params["top_k"].default == 5
    assert params["timeout"].default == 30
    assert params["locale"].default == "en-US"
    assert params["source"].default is None
    assert "max_results" not in params, "legacy `max_results` parameter must be gone"


# --- 2b. Cursor placeholder explicit-not-silent (§B / amendment Drift #4) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [vector_search, graph_search, fulltext_search],
    ids=["vector_search", "graph_search", "fulltext_search"],
)
@pytest.mark.parametrize(
    "bad_cursor",
    ["garbage", "AAAA", "{}", "0"],
    ids=["garbage", "base64_no_payload", "empty_json", "single_zero"],
)
async def test_collection_scoped_split_tools_reject_non_empty_cursor(tool, bad_cursor):
    """Per `[D10 spec amendment]` msg=b9b7072a Drift #4 (c) +
    architect sign-off msg=ebfcdabe + Weston blocker msg=177a1dd8:
    any non-empty cursor must loud-fail with ``NotImplementedError``
    bearing a "search pagination is not yet implemented" message.

    The sign-off explicitly forbids reusing
    ``CursorError("cursor_invalid", ...)`` for this case — the canonical
    cursor codes describe wire-level malformed / expired / drifted
    cursors, and the "feature not implemented yet" semantic must not
    be camouflaged as a malformed-cursor error.

    ``cursor=None`` and ``cursor=""`` are covered by the
    happy-path-passthrough test below; this case only exercises the
    truthy-cursor loud-fail path.
    """

    with pytest.raises(NotImplementedError) as excinfo:
        await tool("collection-id", "query", cursor=bad_cursor)

    message = str(excinfo.value)
    assert "search pagination is not yet implemented" in message, (
        f"{tool.__name__} loud-fail message must clearly state that "
        f"search pagination is not implemented (got {message!r})."
    )
    assert tool.__name__ in message, (
        "Loud-fail message must identify the originating tool so logs can attribute the rejection."
    )
    assert "search_not_paginated" in message, (
        "Loud-fail message must surface the deferred-pagination reason so clients understand they should not retry."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [vector_search, graph_search, fulltext_search],
    ids=["vector_search", "graph_search", "fulltext_search"],
)
@pytest.mark.parametrize(
    "noop_cursor",
    [None, ""],
    ids=["none", "empty_string"],
)
async def test_collection_scoped_split_tools_treat_none_and_empty_cursor_as_first_page(tool, noop_cursor, monkeypatch):
    """Weston blocker msg=177a1dd8 lock: ``None`` *and* ``""`` cursor
    must both preserve the existing single-page ``top_k`` behavior
    (``""`` is **not** a malformed cursor; it is the canonical
    "no pagination requested" wire value).

    We stub ``get_api_key()`` and the outbound httpx call so the test
    only validates that the cursor guard does **not** raise — it does
    not exercise the backend search path itself.
    """

    import aperag.mcp.tools.search_fulltext as sft
    import aperag.mcp.tools.search_graph as sg
    import aperag.mcp.tools.search_vector as sv

    monkeypatch.setattr(sv, "get_api_key", lambda: "test-token")
    monkeypatch.setattr(sg, "get_api_key", lambda: "test-token")
    monkeypatch.setattr(sft, "get_api_key", lambda: "test-token")

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"items": []}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            return _FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    # No exception expected — the cursor guard must not raise on
    # ``None`` or ``""``.
    result = await tool("collection-id", "query", cursor=noop_cursor)
    assert isinstance(result, dict), (
        f"{tool.__name__} must return the SearchResult dict for None/empty cursor (got {type(result).__name__})."
    )


@pytest.mark.asyncio
async def test_web_search_does_not_carry_cursor_param():
    """``web_search`` is intentionally cursor-less — its provider
    chain (Jina / DDG) is provider-bounded, not paginated, per §B.4.
    The amendment thread (msg=b9b7072a Drift #4) only added cursor
    placeholders to the 3 collection-scoped split tools.
    """

    params = _params(web_search)
    assert "cursor" not in params, (
        "web_search must not carry a cursor parameter — its scope is provider-bounded per §B.4 (no pagination)."
    )


# --- 3. Deprecation banner on omnibus aliases ------------------------


def test_search_collection_carries_deprecation_banner():
    """``search_collection`` docstring is prefixed with
    ``[DEPRECATED]`` and references the canonical D10.b doc (§B.5 /
    §H.1) so reviewers and tool consumers see the migration signal.
    """
    doc = mcp_server_module.search_collection.__doc__ or ""
    assert "[DEPRECATED]" in doc, "search_collection must carry a [DEPRECATED] banner per §B.5 / §H.1."
    # Banner must point migrators at the new split tools; cite the
    # spec source for traceability.
    assert "vector_search" in doc and "graph_search" in doc and "fulltext_search" in doc, (
        "Deprecation banner must enumerate the split tools so callers know the migration target."
    )


def test_search_chat_files_carries_deprecation_banner():
    """``search_chat_files`` shares the deprecation timeline of
    ``search_collection`` per §H.2.
    """
    doc = mcp_server_module.search_chat_files.__doc__ or ""
    assert "[DEPRECATED]" in doc, "search_chat_files must carry a [DEPRECATED] banner per §H.2."


# --- 4. Forbidden boundary (§G D10.d): impl body untouched -----------


def _async_def_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"async def {name!r} not found in source")


def test_search_collection_body_still_targets_v2_collections_path():
    """§G D10.d Forbidden: ``search_collection`` implementation must be
    untouched — only the docstring banner changed. Re-asserts the same
    invariant as ``test_mcp_contract.test_search_collection_targets_v2_path``
    so the D10.d split lane cannot accidentally drift the body.
    """
    source = MCP_SERVER_PATH.read_text()
    body = _async_def_source(source, "search_collection")
    assert "/api/v2/collections/" in body, "search_collection body must still hit /api/v2/collections/."
    assert "/api/v1/collections/" not in body
    # The 5 mode flags accepted by the omnibus alias must remain so
    # existing callers keep working during the deprecation window.
    for mode in (
        "use_vector_index",
        "use_fulltext_index",
        "use_graph_index",
        "use_summary_index",
        "use_vision_index",
    ):
        assert mode in body, (
            f"search_collection must still accept `{mode}` for backward "
            "compatibility (Forbidden: implementation untouched until D10.h)."
        )


def test_search_chat_files_body_still_targets_v2_chats_path():
    """§G D10.d Forbidden: ``search_chat_files`` implementation must be
    untouched — only the docstring banner changed.
    """
    source = MCP_SERVER_PATH.read_text()
    body = _async_def_source(source, "search_chat_files")
    assert "/api/v2/chats/" in body, "search_chat_files body must still hit /api/v2/chats/."


# --- 5. Server module no longer defines old web_search inline --------


def test_old_web_search_def_removed_from_server_module():
    """``web_search`` was relocated to ``aperag.mcp.tools.search_web``.
    The server module must no longer carry an inline ``async def
    web_search`` so we don't double-register the tool.
    """
    source = MCP_SERVER_PATH.read_text()
    tree = ast.parse(source)
    inline_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "web_search"
    ]
    assert inline_defs == [], (
        "aperag.mcp.server must not define `async def web_search` "
        "anymore — implementation moved to "
        "`aperag/mcp/tools/search_web.py`. The re-export at server "
        "module bottom is sufficient for backward attribute access."
    )


def test_web_search_module_targets_v2_web_path():
    """The relocated ``web_search`` body must still target the existing
    ``/api/v2/web/search`` backend path so external MCP clients keep
    receiving identical wire results.
    """
    web_search_path = REPO_ROOT / "aperag" / "mcp" / "tools" / "search_web.py"
    source = web_search_path.read_text()
    body = _async_def_source(source, "web_search")
    assert "/api/v2/web/search" in body, "web_search body must still hit /api/v2/web/search after the D10.d relocation."
