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

import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from aperag.db.models import User
from aperag.db.ops import async_db_ops
from aperag.schema.view_models import WebReadRequest, WebReadResponse, WebSearchRequest, WebSearchResponse
from aperag.views.auth import current_user
from aperag.websearch.reader.reader_service import ReaderService
from aperag.websearch.search.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


async def web_search_view(request: WebSearchRequest) -> WebSearchResponse:
    """
    Enhanced web search with parallel regular and LLM.txt discovery.

    Logic:
    - query + source = site-specific regular search (AND relationship)
    - search_llms_txt = independent LLM.txt discovery (OR relationship)
    - Results are merged and ranked
    """
    # Validate that at least one search type is requested
    has_regular_search = bool(request.query and request.query.strip())
    has_llm_txt_search = bool(request.search_llms_txt and request.search_llms_txt.strip())

    if not has_regular_search and not has_llm_txt_search:
        raise HTTPException(
            status_code=400,
            detail="At least one search type is required: provide 'query' for regular search or 'search_llms_txt' for LLM.txt discovery.",
        )

    # Prepare search tasks
    search_tasks = []
    search_descriptions = []

    # Regular search (query + optional source filtering)
    if has_regular_search:
        regular_service = SearchService(provider_name="duckduckgo")

        regular_request = WebSearchRequest(
            query=request.query.strip(),
            max_results=request.max_results,
            search_engine=request.search_engine,
            timeout=request.timeout,
            locale=request.locale,
            source=request.source,  # Optional site filtering
            search_llms_txt=None,  # Not used for regular search
        )

        search_tasks.append(regular_service.search(regular_request))
        search_descriptions.append(
            f"Regular search: '{request.query}'" + (f" on {request.source}" if request.source else "")
        )

    # LLM.txt discovery search (independent)
    if has_llm_txt_search:
        llm_txt_service = SearchService(provider_name="llm_txt")

        llm_txt_request = WebSearchRequest(
            query="",  # LLM.txt discovery doesn't use query
            max_results=request.max_results,
            search_engine="llm_txt",
            timeout=request.timeout,
            locale=request.locale,
            source=request.search_llms_txt.strip(),  # LLM.txt domain
            search_llms_txt=None,  # Not used in this provider
        )

        search_tasks.append(llm_txt_service.search(llm_txt_request))
        search_descriptions.append(f"LLM.txt discovery: {request.search_llms_txt}")

    # Execute searches in parallel
    try:
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")

    # Process results and handle errors
    all_results = []
    successful_searches = []
    failed_searches = []

    for i, result in enumerate(search_results):
        description = search_descriptions[i]

        if isinstance(result, Exception):
            failed_searches.append(f"{description}: {str(result)}")
            continue

        if hasattr(result, "results") and result.results:
            all_results.extend(result.results)
            successful_searches.append(description)
        else:
            failed_searches.append(f"{description}: No results returned")

    # If all searches failed, return error
    if not all_results and failed_searches:
        raise HTTPException(status_code=500, detail=f"All searches failed: {'; '.join(failed_searches)}")

    # Merge and rank results
    merged_results = _merge_and_rank_results(all_results, request.max_results)

    # Determine the query description for response
    query_parts = []
    if has_regular_search:
        query_parts.append(request.query.strip())
    if has_llm_txt_search:
        query_parts.append(f"LLM.txt:{request.search_llms_txt.strip()}")

    response_query = " + ".join(query_parts)

    return WebSearchResponse(
        query=response_query,
        results=merged_results,
        search_engine=f"parallel({len(successful_searches)} sources)",
        total_results=len(merged_results),
        search_time=0.0,  # TODO: Track actual search time
    )


def _merge_and_rank_results(all_results: List, max_results: int) -> List:
    """
    Merge results from multiple sources and re-rank them.

    Simple strategy:
    1. Remove duplicates by URL
    2. Sort by rank (lower is better)
    3. Re-assign sequential ranks
    4. Limit to max_results
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

    # Re-assign sequential ranks and limit results
    final_results = []
    for i, result in enumerate(sorted_results[:max_results]):
        # Create a new result item with updated rank
        if hasattr(result, "rank"):
            result.rank = i + 1
        final_results.append(result)

    return final_results


@router.post("/web/search", response_model=WebSearchResponse, tags=["websearch"])
async def web_search_endpoint(request: WebSearchRequest) -> WebSearchResponse:
    """
    Perform web search using various search engines with advanced domain targeting.

    Supports parallel execution of:
    - Regular web search with optional site filtering (query + source)
    - LLM.txt discovery search (search_llms_txt)

    Results are merged and ranked automatically.
    """
    return await web_search_view(request)


@router.post("/web/read", response_model=WebReadResponse, tags=["websearch"])
async def web_read_endpoint(request: WebReadRequest, user: User = Depends(current_user)) -> WebReadResponse:
    """
    Read and extract content from web pages.

    Supports:
    - Single URL or multiple URLs (use url_list array)
    - Concurrent processing for multiple URLs
    - Configurable timeout and locale settings
    - Multiple reader providers (JINA priority, Trafilatura fallback)

    Logic:
    - Try to get JINA API key from user's provider settings
    - If JINA API key available, run both JINA and Trafilatura concurrently
    - Return JINA result if successful, otherwise fallback to Trafilatura
    - If no JINA API key, use Trafilatura only
    """
    try:
        # Validate url_list parameter
        if not request.url_list or len(request.url_list) == 0:
            raise HTTPException(
                status_code=400, detail="url_list parameter is required and must contain at least one URL"
            )

        # Try to get JINA API key for current user
        jina_api_key = None
        try:
            jina_api_key = await async_db_ops.query_provider_api_key("jina", user_id=str(user.id), need_public=True)
            logger.debug(f"JINA API key query result for user {user.id}: {'found' if jina_api_key else 'not found'}")
        except Exception as e:
            logger.debug(f"Could not query JINA API key for user {user.id}: {e}")

        if jina_api_key:
            logger.info(f"JINA API key found for user {user.id}, using JINA with Trafilatura fallback")
            return await _read_with_jina_fallback(request, jina_api_key)
        else:
            logger.info(f"No JINA API key found for user {user.id}, using Trafilatura only")
            return await _read_with_trafilatura_only(request)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Web read endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Web read failed: {str(e)}")


async def _read_with_jina_fallback(request: WebReadRequest, jina_api_key: str) -> WebReadResponse:
    """
    Read with JINA priority and Trafilatura fallback - simple and reliable approach.

    Args:
        request: Web read request
        jina_api_key: JINA API key for authentication

    Returns:
        Web read response from JINA if successful, otherwise from Trafilatura
    """
    jina_service = ReaderService(provider_name="jina", provider_config={"api_key": jina_api_key})

    try:
        # Try JINA first
        try:
            logger.info("Attempting to read with JINA")
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

        # Fallback to Trafilatura using the dedicated function
        logger.info("Falling back to Trafilatura")
        return await _read_with_trafilatura_only(request)

    finally:
        await jina_service.close()


async def _read_with_trafilatura_only(request: WebReadRequest) -> WebReadResponse:
    """
    Read using Trafilatura only.

    Args:
        request: Web read request

    Returns:
        Web read response from Trafilatura
    """
    trafilatura_service = ReaderService(provider_name="trafilatura")
    try:
        return await trafilatura_service.read(request)
    finally:
        await trafilatura_service.close()
