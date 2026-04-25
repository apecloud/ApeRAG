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

"""Phase 9 D10.d (#96) — graph_search split MCP tool (§B.2).

Replaces ``search_collection use_graph_index=True`` with a discrete tool
per ``docs/modularization/d10-design-pack.md`` §B.2 (Lock #5 split).
``search_collection`` itself remains as a deprecated alias (§B.5 / §H.1)
until the D10.h cutover.

Wire shape: returns the existing ``SearchResult`` dict shape with
``recall_type = "graph_search"`` for produced items. §B canonical shape
follow-up gated on the chunk_id propagation amendment thread — see
``search_vector.py`` module docstring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from aperag.domains.retrieval.schemas import SearchResult
from aperag.mcp.server import API_BASE_URL, get_api_key, mcp_server

logger = logging.getLogger(__name__)


@mcp_server.tool
async def graph_search(
    collection_id: str,
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Knowledge-graph search within a collection (§B.2).

    Use this when:
    - The query benefits from entity / relation traversal rather than
      similarity match (e.g., "who collaborated with X on Y?").
    - You need to navigate from a known entity to related ones via
      indexed relations.

    Do not use this when:
    - The query is best served by semantic similarity; use ``vector_search``.
    - The query is best served by keyword match; use ``fulltext_search``.

    What success means:
    - You retrieved candidate evidence ranked by graph relevance.

    What an empty result means:
    - The collection did not return strong graph hits for the query.
    - The collection may not have a graph index built; consider another
      recall mode.

    What failure may mean:
    - auth / permission: the current user cannot access this collection.
    - network / timeout: the graph path did not complete (graph search can
      be time-consuming on large indexes).
    - bad request: the collection ID is invalid or graph index missing.

    Args:
        collection_id: The ID of the collection to search.
        query: The natural-language search query.
        top_k: Maximum number of results to return (default: 5).

    Returns:
        Search results with ``items`` carrying ``recall_type =
        "graph_search"``. Graph-specific fields (entity / relation / path)
        are surfaced via ``items[*].metadata``.
    """
    try:
        api_key = get_api_key()

        search_data: Dict[str, Any] = {
            "query": query,
            "graph_search": {"topk": top_k},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v2/collections/{collection_id}/searches",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=search_data,
            )
            if response.status_code in (200, 201):
                try:
                    search_result = SearchResult.model_validate(response.json())
                    if search_result.items and len(search_result.items) > top_k:
                        search_result.items = search_result.items[:top_k]
                        for i, item in enumerate(search_result.items):
                            if item.rank is not None:
                                item.rank = i + 1
                    return search_result.model_dump()
                except Exception as exc:
                    logger.error("Failed to parse graph_search response: %s", exc)
                    return {
                        "error": "Failed to parse graph_search response",
                        "details": str(exc),
                    }
            return {
                "error": f"graph_search failed: {response.status_code}",
                "details": response.text,
            }
    except ValueError as exc:
        return {"error": str(exc)}
