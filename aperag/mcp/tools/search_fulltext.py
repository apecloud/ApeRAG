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

"""Phase 9 D10.d (#96) — fulltext_search split MCP tool (§B.3).

Replaces ``search_collection use_fulltext_index=True`` with a discrete
tool per ``docs/modularization/d10-design-pack.md`` §B.3 (Lock #5
split). ``search_collection`` itself remains as a deprecated alias
(§B.5 / §H.1) until the D10.h cutover.

Wire shape: returns the existing ``SearchResult`` dict shape with
``recall_type = "fulltext_search"`` for produced items. §B canonical
shape follow-up settled by the ``[D10 spec amendment]`` thread
(msg=b9b7072a) — defer canonical SearchResultItem to D10.h cutover.

The ``cursor`` parameter is a placeholder per the same thread (Drift
#4 (c)): signature lands now, body raises ``NotImplementedError``
on any non-empty value until real search pagination ships.
``None`` and ``""`` both preserve single-page ``top_k`` behavior per
the Weston blocker review (msg=177a1dd8).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from aperag.domains.retrieval.schemas import SearchResult
from aperag.mcp.capabilities import ToolAnnotation
from aperag.mcp.server import API_BASE_URL, get_api_key, mcp_server
from aperag.mcp.tools._annotations import register as _register_tool_annotation

logger = logging.getLogger(__name__)


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "fulltext_search",
        ToolAnnotation(
            requires=("collection_access",),
            # fulltext_search needs the inverted index — explicit-not-silent
            # per §D.3 if the collection has no fulltext index.
            capabilities={"long_context": False, "fulltext_index": True},
        ),
    ),
)
async def fulltext_search(
    collection_id: str,
    query: str,
    *,
    top_k: int = 5,
    keywords: list[str] | None = None,
    rerank: bool = True,
    cursor: str | None = None,
) -> Dict[str, Any]:
    """Full-text keyword search within a collection (§B.3).

    Use this when:
    - You need exact keyword / phrase matches.
    - The query has identifiable proper nouns or domain-specific terms
      that benefit from inverted-index lookup over similarity.

    Do not use this when:
    - You need semantic similarity; use ``vector_search``.
    - You need entity / relation traversal; use ``graph_search``.

    What success means:
    - You retrieved candidate evidence ranked by full-text relevance
      (Elasticsearch / PostgreSQL FTS depending on collection backend).

    What an empty result means:
    - No documents matched the keywords.
    - Consider relaxing keyword constraints, or trying ``vector_search``
      for fuzzier semantic match.

    What failure may mean:
    - auth / permission: the current user cannot access this collection.
    - network / timeout: the full-text path did not complete.
    - bad request: the collection ID is invalid or full-text index missing.

    Args:
        collection_id: The ID of the collection to search.
        query: The natural-language search query.
        top_k: Maximum number of results to return (default: 5).
        keywords: Optional explicit keyword list overriding the
            auto-extracted keywords from the query.
        rerank: Whether to apply reranker on returned candidates
            (default: True).
        cursor: Pagination cursor placeholder (§B.3 / amendment
            msg=b9b7072a Drift #4 (c)). ``None`` and ``""`` return
            first page; any non-empty value raises
            ``NotImplementedError`` with a clear "not implemented"
            message until real search pagination ships.

    Returns:
        Search results with ``items`` carrying ``recall_type =
        "fulltext_search"``. Highlight snippets / matched terms are
        surfaced via ``items[*].metadata``.
    """
    if cursor:
        raise NotImplementedError(
            "search pagination is not yet implemented (tool=fulltext_search, reason=search_not_paginated)"
        )
    try:
        api_key = get_api_key()

        fulltext_payload: Dict[str, Any] = {"topk": top_k}
        if keywords is not None:
            fulltext_payload["keywords"] = keywords

        search_data: Dict[str, Any] = {
            "query": query,
            "rerank": rerank,
            "fulltext_search": fulltext_payload,
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
                    logger.error("Failed to parse fulltext_search response: %s", exc)
                    return {
                        "error": "Failed to parse fulltext_search response",
                        "details": str(exc),
                    }
            return {
                "error": f"fulltext_search failed: {response.status_code}",
                "details": response.text,
            }
    except ValueError as exc:
        return {"error": str(exc)}
