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

"""Graph search MCP tools — Wave 7 §K.12.6 / §K.12.7 (task #7).

Three discrete graph-recall primitives that let agents access the
lineage graph through semantic vector search, neighbour expansion,
and single-entity detail lookup. Mirrors the existing
``aperag/mcp/tools/search_graph.py`` pattern: each tool calls an
internal-only REST endpoint via httpx with ``API_BASE_URL`` so the
MCP server can run colocated with — or detached from — the API
service.

Wire shape:

* :func:`query_graph_entities` — vector recall over entity names /
  descriptions; returns the top-K matching entities with
  compacted-or-fallback descriptions plus bounded evidence refs for
  follow-up ``read_document_chunk`` calls.
* :func:`expand_graph_subgraph` — n-hop expansion from anchor entity
  names via :meth:`LineageGraphStore.expand_neighbors_n_hops`; both
  returned entities and relations carry bounded evidence refs.
* :func:`get_entity_detail` — single entity lookup by canonical name.

Per spec §K.12 invariant #11 (D-3 candidate detection writes only)
none of these tools mutate state — they are read-only. Per invariant
#12 the tool names are grep-zero clean for legacy naming.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from aperag.mcp.capabilities import ToolAnnotation
from aperag.mcp.server import API_BASE_URL, get_api_key, mcp_server
from aperag.mcp.tools._annotations import register as _register_tool_annotation

logger = logging.getLogger(__name__)


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "query_graph_entities",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def query_graph_entities(
    collection_id: str,
    query: str,
    *,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Semantic search over knowledge-graph entities (§K.12.6).

    Use this when:
    - The query benefits from semantic similarity over entity names
      and descriptions (e.g., "find entities related to renewable
      energy companies").
    - You want a ranked list of candidate anchor entities to feed
      into ``expand_graph_subgraph``.

    Do not use this when:
    - You already know the exact entity name; use ``get_entity_detail``.
    - You need free-text content rather than entities; use
      ``vector_search`` or ``fulltext_search``.

    What success means:
    - You retrieved up to ``top_k`` entities ranked by descending
      vector similarity.

    What an empty result means:
    - The collection has no graph index built yet.
    - The query did not match any indexed entity within the score
      threshold.

    Args:
        collection_id: The ID of the collection to search.
        query: The natural-language search query.
        top_k: Maximum number of entities to return (default 10, max 100).

    Returns:
        ``{"entities": [{"name", "entity_type", "description",
        "source_chunk_count", "evidence_refs"}, ...]}``, where each
        evidence ref includes ``document_id`` + ``chunk_id`` for direct
        ``read_document_chunk(collection_id, document_id, chunk_id)``
        follow-up.
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v2/collections/{collection_id}/graphs/entities/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                params={"q": query, "top_k": top_k},
            )
            return _parse_response(response, "query_graph_entities")
    except ValueError as exc:
        return {"error": str(exc)}


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "expand_graph_subgraph",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def expand_graph_subgraph(
    collection_id: str,
    entity_names: List[str],
    *,
    hops: int = 1,
) -> Dict[str, Any]:
    """Expand neighbours from anchor entities (§K.12.6).

    Use this when:
    - You have anchor entity names (e.g., from
      ``query_graph_entities``) and want their direct or n-hop
      neighbours plus the connecting relations.

    Do not use this when:
    - You only need to look up a single entity; use
      ``get_entity_detail``.
    - You want ranked chunk-level recall; use ``graph_search``,
      ``vector_search`` or ``fulltext_search`` against the original
      chunks. This tool returns bounded source refs for returned graph
      elements, not ranked chunk search results.

    Args:
        collection_id: The ID of the collection to traverse.
        entity_names: Anchor entity names to expand from (must be
            non-empty).
        hops: Number of hops to expand (default 1, max 3). Backend
            implementations cap the result size — callers MUST not
            assume an unbounded subgraph.

    Returns:
        ``{"entities": [...], "relations": [...]}`` where ``relations``
        carries ``source`` / ``target`` entity names, a description,
        ``source_chunk_count`` and bounded ``evidence_refs``.
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v2/collections/{collection_id}/graphs/subgraph/expand",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"entity_names": list(entity_names), "hops": hops},
            )
            return _parse_response(response, "expand_graph_subgraph")
    except ValueError as exc:
        return {"error": str(exc)}


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "get_entity_detail",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def get_entity_detail(
    collection_id: str,
    name: str,
) -> Dict[str, Any]:
    """Return the full record for a single entity (§K.12.6).

    Use this when:
    - You already know the exact entity name and want its full
      description and source-chunk reference count.

    Do not use this when:
    - You're not sure of the exact name; use ``query_graph_entities``
      first.

    Args:
        collection_id: The ID of the collection.
        name: The canonical entity name to look up.

    Returns:
        ``{"name", "entity_type", "description",
        "source_chunk_count", "evidence_refs"}`` on success,
        ``{"error": "Entity not found"}`` when the entity is missing
        from the lineage store.
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Use ``urllib.parse.quote`` so entity names containing
            # ``/`` or unicode characters survive path encoding.
            from urllib.parse import quote

            encoded = quote(name, safe="")
            response = await client.get(
                f"{API_BASE_URL}/api/v2/collections/{collection_id}/graphs/entities/{encoded}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            return _parse_response(response, "get_entity_detail")
    except ValueError as exc:
        return {"error": str(exc)}


def _parse_response(response: httpx.Response, tool_name: str) -> Dict[str, Any]:
    """Decode the API response into the tool's return shape.

    Mirrors the convention in ``aperag/mcp/tools/search_graph.py``:
    success → parsed JSON dict; non-2xx → ``{"error", "details"}``.
    """
    if response.status_code in (200, 201):
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001 — defensive, surface message
            logger.error("Failed to parse %s response: %s", tool_name, exc)
            return {
                "error": f"Failed to parse {tool_name} response",
                "details": str(exc),
            }
    if response.status_code == 404:
        return {"error": "Entity not found" if tool_name == "get_entity_detail" else "Collection not found"}
    return {
        "error": f"{tool_name} failed: {response.status_code}",
        "details": response.text,
    }
