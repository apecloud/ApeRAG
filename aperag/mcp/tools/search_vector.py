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

"""Phase 9 D10.d (#96) — vector_search split MCP tool (§B.1).

Replaces ``search_collection use_vector_index=True`` with a discrete tool
per ``docs/modularization/d10-design-pack.md`` §B.1 (Lock #5 split).
``search_collection`` itself remains as a deprecated alias (§B.5 / §H.1)
until the D10.h cutover.

Wire shape: returns the existing ``SearchResult`` dict shape from the
``aperag/api/v2/collections/{id}/searches`` backend (``recall_type =
"vector_search"`` for produced items). The §B canonical
``SearchResult`` / ``SearchResultItem`` with ``chunk_id`` /
``section_path`` / ``heading_anchor`` is deferred to a D10.d follow-up
PR after the chunk_id propagation question is resolved via a
``[D10 spec amendment]`` thread (current backend does not surface
``chunk_id`` in the public response shape — see PR description).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from aperag.domains.retrieval.schemas import SearchResult
from aperag.mcp.server import API_BASE_URL, get_api_key, mcp_server

logger = logging.getLogger(__name__)


@mcp_server.tool
async def vector_search(
    collection_id: str,
    query: str,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    rerank: bool = True,
) -> Dict[str, Any]:
    """Vector similarity search within a collection (§B.1).

    Use this when:
    - You need semantically similar passages to a natural-language query.
    - The user asks a knowledge question that benefits from embedding-based
      retrieval rather than keyword match.

    Do not use this when:
    - You need exact keyword/phrase matches; use ``fulltext_search`` instead.
    - You need entity / relation traversal; use ``graph_search`` instead.
    - You need information from outside the collections; use ``web_search``.

    What success means:
    - You retrieved candidate evidence ranked by vector similarity.

    What an empty result means:
    - The collection did not return strong vector matches for the query.
    - It does not prove the answer is false; consider trying ``graph_search``
      or ``fulltext_search`` for a different recall mode.

    What failure may mean:
    - auth / permission: the current user cannot access this collection.
    - network / timeout: the search path did not complete.
    - bad request: the collection ID is invalid or the embedding index is
      not built for this collection.

    Args:
        collection_id: The ID of the collection to search.
        query: The natural-language search query.
        top_k: Maximum number of results to return (default: 5).
        similarity_threshold: Minimum similarity score [0, 1]; ``None``
            uses the collection's default threshold.
        rerank: Whether to apply reranker on returned candidates (default: True).

    Returns:
        Search results with ``items`` ranked by vector similarity. Each
        item carries ``recall_type = "vector_search"``.
    """
    try:
        api_key = get_api_key()

        vector_payload: Dict[str, Any] = {"topk": top_k}
        if similarity_threshold is not None:
            vector_payload["similarity"] = similarity_threshold

        search_data: Dict[str, Any] = {
            "query": query,
            "rerank": rerank,
            "vector_search": vector_payload,
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
                    logger.error("Failed to parse vector_search response: %s", exc)
                    return {
                        "error": "Failed to parse vector_search response",
                        "details": str(exc),
                    }
            return {
                "error": f"vector_search failed: {response.status_code}",
                "details": response.text,
            }
    except ValueError as exc:
        return {"error": str(exc)}
