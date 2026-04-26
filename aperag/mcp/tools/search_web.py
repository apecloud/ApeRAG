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

"""Phase 9 D10.d (#96) + D10.h (#100) — web_search split MCP tool (§B.4).

Per ``docs/modularization/d10-design-pack.md`` §B.4: ``web_search``
remains an independent tool (it does not need ``collection_id`` and
its tenancy gate is per-user provider key, not collection access). This
module relocates the existing ``web_search`` implementation from
``aperag/mcp/server.py`` into the ``aperag/mcp/tools/`` subpackage so
all D10 search tools live in one place.

D10.h (#100) applies the §B.4 parameter canonicalization that #96
deferred — ``query`` is required, every other parameter is keyword-
only, the result-count limit is named ``top_k`` (not
``max_results``), and ``source`` is ``str | None`` so a missing
domain filter is null rather than the empty string. The internal
``aperag/domains/web_access`` backend still names its payload field
``max_results``; the tool wrapper translates ``top_k`` to that field
on the wire to keep the backend rename out of this lane.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from aperag.domains.web_access.schemas import WebSearchResponse
from aperag.mcp.capabilities import ToolAnnotation
from aperag.mcp.server import API_BASE_URL, get_api_key, mcp_server
from aperag.mcp.tools._annotations import register as _register_tool_annotation

logger = logging.getLogger(__name__)


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "web_search",
        ToolAnnotation(
            requires=("web_access",),
            # web_search reaches the public internet — explicit-not-silent
            # per §D.3 if the caller's environment denies web egress.
            capabilities={"long_context": False, "web_access": True},
        ),
    ),
)
async def web_search(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 30,
    locale: str = "en-US",
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Search the web for current or missing information (§B.4).

    Use this when:
    - The current turn allows web access.
    - You need current information, external verification, or
      gap-filling beyond ApeRAG collections.

    Do not use this when:
    - The current turn disables web access.
    - Collection or chat-file evidence is already sufficient for the
      requested step.

    What success means:
    - You received candidate web results with titles, snippets, and URLs.

    What an empty result means:
    - No strong web results were found for this query and scope.
    - Use ``meta.search_status`` to distinguish a genuine empty result
      from ``unavailable`` or ``disabled``.

    What failure may mean:
    - network / timeout: external search could not complete.
    - upstream search provider issue: the search backend could not
      return usable results.

    How to explain this step to the user:
    - While running: "Searching the web for current or missing information."
    - After completion: "Checked web sources for supporting information."

    Args:
        query: Search query for web search (required).
        top_k: Maximum number of results to return (default: 5).
        timeout: Request timeout in seconds (default: 30).
        locale: Browser locale (default: ``"en-US"``).
        source: Optional domain or URL for site-specific filtering;
            ``None`` means no domain restriction (default: ``None``).

    Returns:
        Web search results with URLs, titles, snippets, and metadata.

    Note:
        Uses JINA first when configured, otherwise falls back to
        DuckDuckGo. Search failures are soft-failed into empty result
        sets with lightweight ``meta`` diagnostics so downstream
        workflows stay stable while still distinguishing ``ok`` /
        ``empty`` / ``unavailable`` / ``disabled``.
    """
    query = query.strip()
    if not query:
        raise ValueError("web_search requires a non-empty query")
    source_value = source.strip() if source else None
    try:
        api_key = get_api_key()
        logger.info(
            "MCP web_search request query=%s source=%s top_k=%s timeout=%s locale=%s",
            query,
            source_value or "",
            top_k,
            timeout,
            locale,
        )

        search_data: Dict[str, Any] = {
            "query": query,
            "max_results": top_k,
            "timeout": timeout,
            "locale": locale,
        }

        if source_value:
            search_data["source"] = source_value

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v2/web/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=search_data,
            )
            if response.status_code == 200:
                try:
                    search_response = WebSearchResponse.model_validate(response.json())
                    logger.info(
                        "MCP web_search completed query=%s source=%s status=%s results=%s providers=%s backends=%s fallback=%s",
                        query.strip() if query else "",
                        source.strip() if source else "",
                        search_response.meta.search_status if search_response.meta else "unknown",
                        len(search_response.results),
                        search_response.meta.provider_used if search_response.meta else [],
                        search_response.meta.backend_used if search_response.meta else [],
                        search_response.meta.fallback_used if search_response.meta else False,
                    )
                    return search_response.model_dump()
                except Exception as exc:
                    logger.error("Failed to parse web_search response: %s", exc)
                    return {
                        "error": "Failed to parse web_search response",
                        "details": str(exc),
                    }
            logger.warning(
                "MCP web_search failed status=%s query=%s source=%s body=%s",
                response.status_code,
                query.strip() if query else "",
                source.strip() if source else "",
                response.text,
            )
            return {
                "error": f"web_search failed: {response.status_code}",
                "details": response.text,
            }
    except ValueError as exc:
        return {"error": str(exc)}
