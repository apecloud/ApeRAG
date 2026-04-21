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
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from aperag.db.models import User
from aperag.db.ops import async_db_ops
from aperag.schema.view_models import (
    WebReadRequest,
    WebReadResponse,
    WebSearchMeta,
    WebSearchRequest,
    WebSearchResponse,
)
from aperag.views.auth import required_user
from aperag.websearch.reader.reader_service import ReaderService
from aperag.websearch.search.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSearchError(Exception):
    """Custom exception for web search failures."""

    pass


class WebReadError(Exception):
    """Custom exception for web read failures."""

    pass


def _build_empty_web_search_response(query: str, meta: WebSearchMeta) -> WebSearchResponse:
    """Create an empty web search response with lightweight diagnostics."""
    return WebSearchResponse(
        query=query,
        results=[],
        total_results=0,
        search_time=0,
        meta=meta,
    )


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """Deduplicate strings while preserving order."""
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _merge_search_meta(base: WebSearchMeta, current: WebSearchMeta, *, fallback_used: bool) -> WebSearchMeta:
    """Merge two search meta objects while preserving diagnostic order."""
    return WebSearchMeta(
        search_status=current.search_status,
        provider_used=_dedupe_preserve_order(base.provider_used + current.provider_used),
        backend_used=_dedupe_preserve_order(base.backend_used + current.backend_used),
        fallback_used=fallback_used or base.fallback_used or current.fallback_used,
        error_code=current.error_code,
    )


def _build_search_response_query(request: WebSearchRequest) -> str:
    """Build a stable response query string for regular and LLM.txt discovery searches."""
    query_parts = []
    if request.query and request.query.strip():
        query_parts.append(request.query.strip())
    if request.search_llms_txt and request.search_llms_txt.strip():
        query_parts.append(f"LLM.txt:{request.search_llms_txt.strip()}")
    return " + ".join(query_parts)


def _build_overall_search_meta(search_responses: List[WebSearchResponse], merged_results: List) -> WebSearchMeta:
    """Aggregate search diagnostics across requested search paths."""
    provider_used: List[str] = []
    backend_used: List[str] = []
    fallback_used = False
    any_unavailable = False
    error_code = None

    for response in search_responses:
        if not response.meta:
            continue
        provider_used.extend(response.meta.provider_used)
        backend_used.extend(response.meta.backend_used)
        fallback_used = fallback_used or response.meta.fallback_used
        if response.meta.search_status == "unavailable":
            any_unavailable = True
            error_code = error_code or response.meta.error_code

    if merged_results:
        search_status = "ok"
        error_code = None
    elif any_unavailable:
        search_status = "unavailable"
        error_code = error_code or "search_provider_unavailable"
    else:
        search_status = "empty"
        error_code = None

    return WebSearchMeta(
        search_status=search_status,
        provider_used=_dedupe_preserve_order(provider_used),
        backend_used=_dedupe_preserve_order(backend_used),
        fallback_used=fallback_used,
        error_code=error_code,
    )


@router.post("/web/search", response_model=WebSearchResponse, tags=["websearch"])
async def web_search_endpoint(request: WebSearchRequest, user: User = Depends(required_user)) -> WebSearchResponse:
    """
    Perform web search using various search engines with advanced domain targeting.

    Supports serial execution of:
    - Regular web search with JINA priority and DuckDuckGo fallback (query + source)
    - LLM.txt discovery search (search_llms_txt)

    Logic:
    - query + source = site-specific regular search with JINA priority (AND relationship)
    - search_llms_txt = independent LLM.txt discovery (OR relationship)
    - Results are merged and ranked

    Results are merged and ranked automatically.
    """
    # Record start time for tracking search duration
    search_start_time = time.time()

    try:
        # Validate that at least one search type is requested
        has_regular_search = bool(request.query and request.query.strip())
        has_llm_txt_search = bool(request.search_llms_txt and request.search_llms_txt.strip())

        if not has_regular_search and not has_llm_txt_search:
            raise HTTPException(
                status_code=400,
                detail="At least one search type is required: provide 'query' for regular search or 'search_llms_txt' for LLM.txt discovery.",
            )

        # Collect search results
        all_results = []
        search_responses: List[WebSearchResponse] = []

        # Execute regular search if requested
        if has_regular_search:
            logger.info(
                f"Starting regular search: '{request.query}'" + (f" on {request.source}" if request.source else "")
            )
            try:
                regular_result = await _search_with_jina_fallback(request, user)
            except Exception as e:
                logger.warning("Regular search branch failed query=%s error=%s", request.query, e)
                regular_result = _build_empty_web_search_response(
                    request.query.strip(),
                    WebSearchMeta(
                        search_status="unavailable",
                        provider_used=["regular_search"],
                        backend_used=["regular_search"],
                        fallback_used=False,
                        error_code="regular_search_unavailable",
                    ),
                )
            search_responses.append(regular_result)
            if regular_result.results:
                all_results.extend(regular_result.results)
                logger.info(f"Regular search succeeded: {len(regular_result.results)} results")
            else:
                logger.info(
                    "Regular search completed without results status=%s error_code=%s",
                    regular_result.meta.search_status if regular_result.meta else "unknown",
                    regular_result.meta.error_code if regular_result.meta else None,
                )

        # Execute LLM.txt discovery search if requested
        if has_llm_txt_search:
            logger.info(f"Starting LLM.txt discovery search: {request.search_llms_txt}")
            try:
                llm_txt_result = await _search_llm_txt_discovery(request)
            except Exception as e:
                logger.warning("LLM.txt discovery branch failed source=%s error=%s", request.search_llms_txt, e)
                llm_txt_result = _build_empty_web_search_response(
                    f"LLM.txt:{request.search_llms_txt.strip()}",
                    WebSearchMeta(
                        search_status="unavailable",
                        provider_used=["llm_txt"],
                        backend_used=["llm_txt"],
                        fallback_used=False,
                        error_code="llm_txt_unavailable",
                    ),
                )
            search_responses.append(llm_txt_result)
            if llm_txt_result.results:
                all_results.extend(llm_txt_result.results)
                logger.info(f"LLM.txt discovery succeeded: {len(llm_txt_result.results)} results")
            else:
                logger.info(
                    "LLM.txt discovery completed without results status=%s error_code=%s",
                    llm_txt_result.meta.search_status if llm_txt_result.meta else "unknown",
                    llm_txt_result.meta.error_code if llm_txt_result.meta else None,
                )

        # Merge and rank results
        merged_results = _merge_and_rank_results(all_results, request.max_results)

        # Calculate total search time
        total_search_time = time.time() - search_start_time

        overall_meta = _build_overall_search_meta(search_responses, merged_results)

        logger.info(
            "Search completed query=%s status=%s results=%s fallback=%s providers=%s time=%.2fs",
            _build_search_response_query(request),
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

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Web search endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")


async def _search_llm_txt_discovery(request: WebSearchRequest) -> WebSearchResponse:
    """
    Perform LLM.txt discovery search using dedicated service.

    Args:
        request: Original search request with search_llms_txt parameter

    Returns:
        Search response from LLM.txt discovery

    Raises:
        WebSearchError: If LLM.txt search fails
    """
    try:
        async with SearchService(provider_name="llm_txt") as llm_txt_service:
            llm_txt_request = WebSearchRequest(
                query="",  # LLM.txt discovery doesn't use query
                max_results=request.max_results,
                timeout=request.timeout,
                locale=request.locale,
                source=request.search_llms_txt.strip(),  # LLM.txt domain
                search_llms_txt=None,  # Not used in this provider
            )

            return await llm_txt_service.search(llm_txt_request)

    except Exception as e:
        logger.warning("LLM.txt discovery failed source=%s error=%s", request.search_llms_txt, e)
        return _build_empty_web_search_response(
            f"LLM.txt:{request.search_llms_txt.strip()}",
            WebSearchMeta(
                search_status="unavailable",
                provider_used=["llm_txt"],
                backend_used=["llm_txt"],
                fallback_used=False,
                error_code="llm_txt_unavailable",
            ),
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
async def web_read_endpoint(request: WebReadRequest, user: User = Depends(required_user)) -> WebReadResponse:
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


async def _get_user_jina_api_key(user: User) -> str | None:
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
    # Try JINA first
    try:
        logger.info("Attempting to read with JINA")
        async with ReaderService(provider_name="jina", provider_config={"api_key": jina_api_key}) as jina_service:
            jina_result = await jina_service.read(request)

            # Check if JINA was successful
            if jina_result and hasattr(jina_result, "results"):
                successful_count = sum(1 for r in jina_result.results if r.status == "success")
                if successful_count > 0:
                    logger.info(f"JINA succeeded: {successful_count}/{jina_result.total_urls} URLs")
                    return jina_result
                else:
                    logger.info("JINA completed but no URLs were successfully processed")
            else:
                logger.info("JINA returned empty or invalid result")

    except Exception as e:
        logger.info(f"JINA failed: {e}")

    # Fallback to Trafilatura
    logger.info("Falling back to Trafilatura")
    try:
        return await _read_with_trafilatura_only(request)
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


async def _search_with_jina_fallback(request: WebSearchRequest, user: User) -> WebSearchResponse:
    """
    Search with JINA priority and DuckDuckGo fallback.

    Args:
        request: Web search request
        user: Current user for API key lookup

    Returns:
        Search results from JINA if successful, otherwise from DuckDuckGo

    Raises:
        WebSearchError: If both JINA and DuckDuckGo fail
    """
    query = request.query.strip() if request.query else ""

    # Try to get JINA API key for current user
    jina_api_key = await _get_user_jina_api_key(user)
    jina_attempt_meta = None

    # Try JINA first if API key is available
    if jina_api_key:
        try:
            logger.info("Attempting to search with JINA")
            async with SearchService(provider_name="jina", provider_config={"api_key": jina_api_key}) as jina_service:
                jina_result = await jina_service.search(request)

                # Check if JINA was successful
                if jina_result and hasattr(jina_result, "results") and jina_result.results:
                    logger.info(f"JINA search succeeded: {len(jina_result.results)} results")
                    return jina_result
                else:
                    logger.info("JINA search completed but no results returned")
                    jina_attempt_meta = jina_result.meta
                    if jina_attempt_meta:
                        jina_attempt_meta.error_code = None

        except Exception as e:
            logger.info(f"JINA search failed: {e}")
            jina_attempt_meta = WebSearchMeta(
                search_status="unavailable",
                provider_used=["jina"],
                backend_used=["jina"],
                fallback_used=False,
                error_code="jina_unavailable",
            )

    # Fallback to DuckDuckGo
    logger.info("Using DuckDuckGo search")
    try:
        async with SearchService(provider_name="duckduckgo") as duckduckgo_service:
            duckduckgo_result = await duckduckgo_service.search(request)
            if jina_attempt_meta and duckduckgo_result.meta:
                duckduckgo_result.meta = _merge_search_meta(
                    jina_attempt_meta,
                    duckduckgo_result.meta,
                    fallback_used=True,
                )
            return duckduckgo_result
    except Exception as e:
        logger.warning("DuckDuckGo search failed query=%s error=%s", query, e)
        if jina_attempt_meta:
            return _build_empty_web_search_response(
                query,
                _merge_search_meta(
                    jina_attempt_meta,
                    WebSearchMeta(
                        search_status="unavailable",
                        provider_used=["duckduckgo"],
                        backend_used=["duckduckgo"],
                        fallback_used=False,
                        error_code="duckduckgo_unavailable",
                    ),
                    fallback_used=True,
                ),
            )
        return _build_empty_web_search_response(
            query,
            WebSearchMeta(
                search_status="unavailable",
                provider_used=["duckduckgo"],
                backend_used=["duckduckgo"],
                fallback_used=False,
                error_code="duckduckgo_unavailable",
            ),
        )
