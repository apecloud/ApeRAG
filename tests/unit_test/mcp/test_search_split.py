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

"""Search primitive contract tests.

Locks the public surface of the ``aperag.mcp.tools.search_*`` modules
after the omnibus search cutover:

- 4 split tools (``vector_search`` / ``graph_search`` / ``fulltext_search``
  / ``web_search``) exist with the §B canonical signatures.
- They are registered under ``mcp_server`` via the ``@mcp_server.tool``
  decorator chain at module import time (server.py re-export gate).
- ``search_collection`` and ``search_chat_files`` stay removed.
- Collection-scoped search tools do not expose pagination cursor
  placeholders until real search pagination exists.

Caller-migration assertion semantics (§G hard gate #4): backward
compatibility for existing ``mcp_server.web_search`` attribute access
(legacy test path) is preserved by re-exporting ``web_search`` at the
server module level.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

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
    ``top_k`` / ``similarity_threshold`` / ``rerank``.
    """
    params = _params(vector_search)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only_params(vector_search) == {
        "top_k",
        "similarity_threshold",
        "rerank",
    }, "§B.1 requires kw-only `top_k / similarity_threshold / rerank`."
    assert params["top_k"].default == 5
    # similarity_threshold defaults to None to mean "use collection default".
    assert params["similarity_threshold"].default is None
    assert params["rerank"].default is True


def test_graph_search_signature_matches_b2_lock():
    """``graph_search`` signature follows §B.2:
    collection_id + query positional then kw-only top_k.
    The §B.2 spec mentions ``depth`` / ``entity_types`` as
    forward-looking knobs; first-cut keeps them out until the backend
    surface for graph traversal is wired (deferred to D10.d follow-up).
    """
    params = _params(graph_search)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only_params(graph_search) == {"top_k"}, "§B.2 requires kw-only `top_k`."
    assert params["top_k"].default == 5


def test_fulltext_search_signature_matches_b3_lock():
    """``fulltext_search`` signature follows §B.3: collection_id +
    query positional then kw-only top_k + keywords + rerank.
    """
    params = _params(fulltext_search)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only_params(fulltext_search) == {
        "top_k",
        "keywords",
        "rerank",
    }, "§B.3 requires kw-only `top_k / keywords / rerank`."
    assert params["top_k"].default == 5
    assert params["keywords"].default is None
    assert params["rerank"].default is True


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


def test_search_tools_do_not_carry_cursor_param():
    """Search pagination is not implemented, so search tools must not
    advertise a cursor parameter in their MCP schemas.
    """

    for tool in (vector_search, graph_search, fulltext_search, web_search):
        assert "cursor" not in _params(tool), f"{tool.__name__} must not expose a cursor placeholder"


# --- 3. Cutover removal of omnibus aliases (D10.h #100) --------------
#
# D10.d (#1713 / #1717) preserved ``search_collection`` and
# ``search_chat_files`` behind a ``[DEPRECATED]`` banner so external
# callers had a migration window. D10.h (#100) hard-cuts both — the
# banners + bodies are now gone, and the only thing the test surface
# pins is that they stay gone, since ``earayu2 msg=9730bb6b`` made
# the project's hard-cut philosophy explicit (no users / no data).


def test_search_collection_legacy_omnibus_removed_from_module():
    """``search_collection`` was deleted in D10.h #100 — neither the
    runtime attribute nor an ``async def`` in the source survives.
    """
    assert not hasattr(mcp_server_module, "search_collection"), (
        "search_collection must be removed from aperag.mcp.server in the D10.h cutover."
    )

    source = MCP_SERVER_PATH.read_text()
    tree = ast.parse(source)
    inline_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "search_collection"
    ]
    assert inline_defs == [], "aperag.mcp.server must not define `async def search_collection` after the D10.h cutover."


def test_search_chat_files_legacy_omnibus_removed_from_module():
    """``search_chat_files`` shared the deprecation timeline and is
    cut in the same lane.
    """
    assert not hasattr(mcp_server_module, "search_chat_files"), (
        "search_chat_files must be removed from aperag.mcp.server in the D10.h cutover."
    )

    source = MCP_SERVER_PATH.read_text()
    tree = ast.parse(source)
    inline_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "search_chat_files"
    ]
    assert inline_defs == [], "aperag.mcp.server must not define `async def search_chat_files` after the D10.h cutover."


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
    tree = ast.parse(source)
    body: str = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "web_search":
            body = ast.get_source_segment(source, node) or ""
            break
    assert body, "async def web_search not found in aperag/mcp/tools/search_web.py"
    assert "/api/v2/web/search" in body, "web_search body must still hit /api/v2/web/search after the D10.d relocation."
