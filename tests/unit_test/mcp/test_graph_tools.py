# Copyright 2026 ApeCloud, Inc.
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

"""Wave 7 §K.12.6 / §K.12.7 — graph MCP tool contract tests (task #7).

Mirror of ``test_search_split.py`` for the three Wave 7 tools. Locks:

1. The three tools exist and are registered as ``async`` MCP tools.
2. Their signatures match the spec (collection_id positional, kw-only
   knobs).
3. The httpx call shape — verb / URL / params / body — matches the
   internal-only REST endpoints implemented in
   ``aperag/domains/knowledge_graph/api/routes.py``.
4. Success / 404 / 5xx response paths return the right shape.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.mcp import server as mcp_server_module
from aperag.mcp.tools.graph_tools import (
    expand_graph_subgraph,
    get_entity_detail,
    query_graph_entities,
)

API_BASE = "http://localhost:8000"


# --- 1. Registration / surface ---------------------------------------


def test_graph_tools_are_async_callables():
    for tool in (query_graph_entities, expand_graph_subgraph, get_entity_detail):
        assert callable(tool)
        assert inspect.iscoroutinefunction(tool), f"{tool.__name__} must be async (MCP tool surface)."


def test_graph_tools_are_re_exported_on_server_module():
    """Importing ``aperag.mcp.server`` must register the 3 tools (the
    decorators run at module import time)."""
    for name in ("query_graph_entities", "expand_graph_subgraph", "get_entity_detail"):
        assert hasattr(mcp_server_module, name), (
            f"aperag.mcp.server must re-export {name} (decorator registration)."
        )


# --- 2. Signature locks ---------------------------------------------


def _params(func) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(func).parameters)


def _kw_only(func) -> set[str]:
    return {
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }


def test_query_graph_entities_signature():
    params = _params(query_graph_entities)
    assert list(params)[:2] == ["collection_id", "query"]
    assert _kw_only(query_graph_entities) == {"top_k"}
    assert params["top_k"].default == 10


def test_expand_graph_subgraph_signature():
    params = _params(expand_graph_subgraph)
    assert list(params)[:2] == ["collection_id", "entity_names"]
    assert _kw_only(expand_graph_subgraph) == {"hops"}
    assert params["hops"].default == 1


def test_get_entity_detail_signature():
    params = _params(get_entity_detail)
    assert list(params) == ["collection_id", "name"]


# --- 3. httpx call shape ---------------------------------------------


def _make_async_client(response_mock: Any) -> Any:
    """Return a mock that mimics ``httpx.AsyncClient(...)`` async ctx."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=response_mock)
    client.post = AsyncMock(return_value=response_mock)
    return client


def _ok_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=payload)
    return resp


def _err_response(status: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


@pytest.mark.asyncio
async def test_query_graph_entities_httpx_call_shape():
    payload = {"entities": [{"name": "X", "entity_type": "organization", "description": "d", "source_chunk_count": 2}]}
    response = _ok_response(payload)
    client = _make_async_client(response)

    with (
        patch("aperag.mcp.tools.graph_tools.httpx.AsyncClient", return_value=client),
        patch("aperag.mcp.tools.graph_tools.get_api_key", return_value="kkey"),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await query_graph_entities("col_abc", "what is X", top_k=5)

    client.get.assert_awaited_once()
    call = client.get.call_args
    assert call.args[0] == f"{API_BASE}/api/v2/collections/col_abc/graphs/entities/search"
    assert call.kwargs["params"] == {"q": "what is X", "top_k": 5}
    assert call.kwargs["headers"]["Authorization"] == "Bearer kkey"
    assert result == payload


@pytest.mark.asyncio
async def test_expand_graph_subgraph_httpx_call_shape():
    payload = {"entities": [], "relations": []}
    response = _ok_response(payload)
    client = _make_async_client(response)

    with (
        patch("aperag.mcp.tools.graph_tools.httpx.AsyncClient", return_value=client),
        patch("aperag.mcp.tools.graph_tools.get_api_key", return_value="kkey"),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await expand_graph_subgraph("col_abc", ["A", "B"], hops=2)

    client.post.assert_awaited_once()
    call = client.post.call_args
    assert call.args[0] == f"{API_BASE}/api/v2/collections/col_abc/graphs/subgraph/expand"
    assert call.kwargs["json"] == {"entity_names": ["A", "B"], "hops": 2}
    assert result == payload


@pytest.mark.asyncio
async def test_get_entity_detail_httpx_call_shape_url_encodes_name():
    payload = {"name": "墨香居", "entity_type": "organization", "description": "...", "source_chunk_count": 3}
    response = _ok_response(payload)
    client = _make_async_client(response)

    with (
        patch("aperag.mcp.tools.graph_tools.httpx.AsyncClient", return_value=client),
        patch("aperag.mcp.tools.graph_tools.get_api_key", return_value="kkey"),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await get_entity_detail("col_abc", "墨香居")

    client.get.assert_awaited_once()
    url = client.get.call_args.args[0]
    # Unicode entity name must be URL-encoded — no raw chinese in URL.
    assert url.startswith(f"{API_BASE}/api/v2/collections/col_abc/graphs/entities/")
    assert "%E5%A2%A8%E9%A6%99%E5%B1%85" in url
    assert result == payload


# --- 4. Error path response shapes -----------------------------------


@pytest.mark.asyncio
async def test_query_graph_entities_404_returns_error_dict():
    response = _err_response(404, "not found")
    client = _make_async_client(response)
    with (
        patch("aperag.mcp.tools.graph_tools.httpx.AsyncClient", return_value=client),
        patch("aperag.mcp.tools.graph_tools.get_api_key", return_value="kkey"),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await query_graph_entities("col_abc", "q")
    assert result == {"error": "Collection not found"}


@pytest.mark.asyncio
async def test_get_entity_detail_404_distinguishes_entity_missing():
    """``get_entity_detail`` 404 → 'Entity not found' (vs collection-level
    'Collection not found' the other tools emit). The route already
    distinguishes the two with a different ``HTTPException(detail=...)``;
    this lock just ensures the tool surface preserves it."""
    response = _err_response(404, "entity gone")
    client = _make_async_client(response)
    with (
        patch("aperag.mcp.tools.graph_tools.httpx.AsyncClient", return_value=client),
        patch("aperag.mcp.tools.graph_tools.get_api_key", return_value="kkey"),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await get_entity_detail("col_abc", "missing")
    assert result == {"error": "Entity not found"}


@pytest.mark.asyncio
async def test_expand_graph_subgraph_5xx_returns_error_dict_with_details():
    response = _err_response(500, "boom")
    client = _make_async_client(response)
    with (
        patch("aperag.mcp.tools.graph_tools.httpx.AsyncClient", return_value=client),
        patch("aperag.mcp.tools.graph_tools.get_api_key", return_value="kkey"),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await expand_graph_subgraph("col_abc", ["A"], hops=1)
    assert result["error"].startswith("expand_graph_subgraph failed:")
    assert result["details"] == "boom"


@pytest.mark.asyncio
async def test_query_graph_entities_missing_api_key_returns_error():
    """When ``get_api_key`` raises ``ValueError``, the tool surfaces
    the message as ``{"error": ...}`` instead of propagating the
    exception (mirror ``aperag/mcp/tools/search_graph.py`` convention).
    """
    with (
        patch("aperag.mcp.tools.graph_tools.get_api_key", side_effect=ValueError("missing key")),
        patch("aperag.mcp.tools.graph_tools.API_BASE_URL", API_BASE),
    ):
        result = await query_graph_entities("col_abc", "q")
    assert result == {"error": "missing key"}
