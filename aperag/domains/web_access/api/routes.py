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

import copy
import logging
import time
from typing import List, Protocol

from fastapi import APIRouter, Depends, HTTPException

from aperag.db.ops import async_db_ops
from aperag.domains.web_access.reader.reader_service import ReaderService, read_with_jina_fallback
from aperag.domains.web_access.schemas import (
    WebReadRequest,
    WebReadResponse,
    WebSearchMeta,
    WebSearchRequest,
    WebSearchResponse,
)
from aperag.domains.web_access.search.search_service import SearchService
from aperag.views.auth import required_user


class AuthenticatedUser(Protocol):
    """Minimal auth-context contract the `web_access` domain depends on.

    ``web_access`` only reads the authenticated user's ``id`` (to look up
    the per-user JINA API key configuration). Expressing the dependency
    as a narrow ``Protocol`` keeps the module boundary honest: the
    domain does not import ``aperag.db.models.User`` (forbidden by the
    Phase 0 strict ban) and also does not collapse to ``Any`` (which
    would silently erase the auth contract). Phase 4 identity hard-cut
    will promote this local ``Protocol`` to a canonical auth context
    owned by ``aperag/domains/identity``; until then the concrete
    SQLAlchemy ``User`` structurally satisfies this shape, so
    ``Depends(required_user)`` still plugs in without change.
    """

    id: object


logger = logging.getLogger(__name__)

router = APIRouter()


class WebReadError(Exception):
    """Custom exception for web read failures."""

    pass


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """Deduplicate strings while preserving their original order."""
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _merge_search_meta(base: WebSearchMeta, current: WebSearchMeta, *, fallback_used: bool) -> WebSearchMeta:
    """Merge provider diagnostics while preserving attempt order."""
    return WebSearchMeta(
        search_status=current.search_status,
        provider_used=_dedupe_preserve_order(base.provider_used + current.provider_used),
        backend_used=_dedupe_preserve_order(base.backend_used + current.backend_used),
        fallback_used=fallback_used or base.fallback_used or current.fallback_used,
        error_code=current.error_code,
    )


@router.post("/web/search", response_model=WebSearchResponse, tags=["websearch"])
async def web_search_endpoint(
    request: WebSearchRequest, user: AuthenticatedUser = Depends(required_user)
) -> WebSearchResponse:
    """
    Perform best-effort web search with domain targeting.

    Logic:
    - If a JINA API key is configured for the current user (or public provider), use JINA first
    - Otherwise use DuckDuckGo directly
    - DuckDuckGo internally retries a small backend fallback chain for better zero-config reliability
    - External provider failures are soft-failed: the endpoint returns an empty result set plus lightweight
      diagnostics instead of 500 so callers can distinguish empty, unavailable, and fallback states
    """
    # Record start time for tracking search duration
    search_start_time = time.time()

    has_query = bool(request.query and request.query.strip())
    has_source = bool(request.source and request.source.strip())

    if not has_query and not has_source:
        raise HTTPException(
            status_code=400,
            detail="At least one search input is required: provide 'query' for web search or 'source' for site browsing.",
        )

    logger.info(
        "Starting web search query=%s source=%s",
        request.query.strip() if request.query else "",
        request.source.strip() if request.source else "",
    )
    regular_result = await _search_with_jina_fallback(request, user)
    merged_results = _merge_and_rank_results(regular_result.results, request.max_results)

    # Calculate total search time
    total_search_time = time.time() - search_start_time

    overall_meta = regular_result.meta or WebSearchMeta(
        search_status="ok" if merged_results else "empty",
        provider_used=[],
        backend_used=[],
        fallback_used=False,
        error_code=None,
    )
    if merged_results and overall_meta.search_status != "ok":
        overall_meta = overall_meta.model_copy(update={"search_status": "ok", "error_code": None})

    logger.info(
        "Web search completed query=%s source=%s status=%s results=%s fallback=%s providers=%s time=%.2fs",
        request.query.strip() if request.query else "",
        request.source.strip() if request.source else "",
        overall_meta.search_status,
        len(merged_results),
        overall_meta.fallback_used,
        overall_meta.provider_used,
        total_search_time,
    )

    return WebSearchResponse(
        query=_build_search_response_query(request),
        results=merged_results,
        total_results=len(merged_results),
        search_time=total_search_time,
        meta=overall_meta,
    )


def _merge_and_rank_results(all_results: List, max_results: int) -> List:
    """
    Merge results from multiple sources and re-rank them.

    Creates new result objects instead of modifying originals.

    Strategy:
    1. Remove duplicates by URL
    2. Sort by rank (lower is better)
    3. Create new results with sequential ranks
    4. Limit to max_results

    Args:
        all_results: List of search result items from different sources
        max_results: Maximum number of results to return

    Returns:
        List of merged and ranked results
    """
    if not all_results:
        return []

    # Remove duplicates by URL, keeping the first occurrence
    seen_urls = set()
    unique_results = []

    for result in all_results:
        if hasattr(result, "url") and result.url not in seen_urls:
            seen_urls.add(result.url)
            unique_results.append(result)

    # Sort by existing rank (assume lower rank = higher relevance)
    sorted_results = sorted(unique_results, key=lambda r: getattr(r, "rank", 999))

    # Create new result objects with updated ranks instead of modifying originals
    final_results = []
    for i, result in enumerate(sorted_results[:max_results]):
        # Create a deep copy to avoid modifying the original
        new_result = copy.deepcopy(result)
        # Update rank on the copy
        if hasattr(new_result, "rank"):
            new_result.rank = i + 1
        final_results.append(new_result)

    return final_results


@router.post("/web/read", response_model=WebReadResponse, tags=["websearch"])
async def web_read_endpoint(
    request: WebReadRequest, user: AuthenticatedUser = Depends(required_user)
) -> WebReadResponse:
    """
    Read and extract content from web pages.

    Supports:
    - Single URL or multiple URLs (use url_list array)
    - Serial processing for multiple URLs
    - Configurable timeout and locale settings
    - Multiple reader providers (JINA priority, Trafilatura fallback)

    Logic:
    - Try to get JINA API key from user's provider settings
    - If JINA API key available, try JINA first, fallback to Trafilatura on failure
    - If no JINA API key, use Trafilatura only
    """
    # Record start time for tracking processing duration
    processing_start_time = time.time()

    try:
        # Validate url_list parameter
        if not request.url_list or len(request.url_list) == 0:
            raise HTTPException(
                status_code=400, detail="url_list parameter is required and must contain at least one URL"
            )

        logger.info(f"Starting web read for {len(request.url_list)} URLs")

        # Try to get JINA API key for current user
        jina_api_key = await _get_user_jina_api_key(user)

        if jina_api_key:
            logger.info(f"JINA API key found for user {user.id}, using JINA with Trafilatura fallback")
            result = await _read_with_jina_fallback(request, jina_api_key)
        else:
            logger.info(f"No JINA API key found for user {user.id}, using Trafilatura only")
            result = await _read_with_trafilatura_only(request)

        # Calculate total processing time and update result
        total_processing_time = time.time() - processing_start_time

        # Update the processing_time in the result
        if hasattr(result, "processing_time"):
            result.processing_time = total_processing_time

        logger.info(
            f"Web read completed: {result.successful}/{result.total_urls} URLs successful "
            f"in {total_processing_time:.2f}s"
        )

        return result

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Web read endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Web read failed: {str(e)}")


async def _get_user_jina_api_key(user: AuthenticatedUser) -> str | None:
    """
    Get JINA API key for the current user.

    Args:
        user: Current user object

    Returns:
        JINA API key if found, None otherwise
    """
    try:
        jina_api_key = await async_db_ops.query_provider_api_key("jina", user_id=str(user.id), need_public=True)
        logger.debug(f"JINA API key query result for user {user.id}: {'found' if jina_api_key else 'not found'}")
        return jina_api_key
    except Exception as e:
        logger.debug(f"Could not query JINA API key for user {user.id}: {e}")
        return None


async def _read_with_jina_fallback(request: WebReadRequest, jina_api_key: str) -> WebReadResponse:
    """
    Read with JINA priority and Trafilatura fallback.

    Args:
        request: Web read request
        jina_api_key: JINA API key for authentication

    Returns:
        Web read response from JINA if successful, otherwise from Trafilatura

    Raises:
        WebReadError: If both JINA and Trafilatura fail
    """
    try:
        return await read_with_jina_fallback(request, jina_api_key, log_context="web_read_endpoint")
    except Exception as e:
        raise WebReadError(f"Both JINA and Trafilatura reading failed. Last error: {str(e)}") from e


async def _read_with_trafilatura_only(request: WebReadRequest) -> WebReadResponse:
    """
    Read using Trafilatura only.

    Args:
        request: Web read request

    Returns:
        Web read response from Trafilatura

    Raises:
        WebReadError: If Trafilatura fails
    """
    try:
        async with ReaderService(provider_name="trafilatura") as trafilatura_service:
            return await trafilatura_service.read(request)
    except Exception as e:
        raise WebReadError(f"Trafilatura reading failed: {str(e)}") from e


async def _search_with_jina_fallback(request: WebSearchRequest, user: AuthenticatedUser) -> WebSearchResponse:
    """
    Search with JINA priority and DuckDuckGo fallback.

    Args:
        request: Web search request
        user: Current user for API key lookup

    Returns:
        Search results from JINA if successful, otherwise from DuckDuckGo
    """
    # Try to get JINA API key for current user
    jina_api_key = await _get_user_jina_api_key(user)
    jina_attempt_meta = None

    # Try JINA first if API key is available
    if jina_api_key:
        logger.info("Web search using JINA first query=%s source=%s", request.query, request.source)
        async with SearchService(provider_name="jina", provider_config={"api_key": jina_api_key}) as jina_service:
            jina_result = await jina_service.search(request)

            # Check if JINA was successful
            if jina_result and hasattr(jina_result, "results") and jina_result.results:
                logger.info(f"JINA search succeeded: {len(jina_result.results)} results")
                return jina_result

            logger.info("JINA search completed but no results returned; falling back to DuckDuckGo")
            jina_attempt_meta = jina_result.meta.model_copy(deep=True) if jina_result.meta else None
            if jina_attempt_meta and jina_attempt_meta.search_status == "empty":
                jina_attempt_meta.error_code = None

    # Fallback to DuckDuckGo
    logger.info("Web search using DuckDuckGo fallback query=%s source=%s", request.query, request.source)
    async with SearchService(provider_name="duckduckgo") as duckduckgo_service:
        duckduckgo_result = await duckduckgo_service.search(request)
        if jina_attempt_meta and duckduckgo_result.meta:
            duckduckgo_result.meta = _merge_search_meta(
                jina_attempt_meta,
                duckduckgo_result.meta,
                fallback_used=True,
            )
        return duckduckgo_result


def _build_search_response_query(request: WebSearchRequest) -> str:
    """Build a stable response query string for regular and source-only searches."""
    if request.query and request.query.strip():
        return request.query.strip()
    if request.source and request.source.strip():
        return f"site:{request.source.strip()}"
    return ""
