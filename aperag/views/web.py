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

from fastapi import APIRouter, Depends, HTTPException, Request

from aperag.db.models import User
from aperag.schema.view_models import (
    WebReadRequest,
    WebReadResponse,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResultItem,
)
from aperag.utils.audit_decorator import audit
from aperag.views.auth import current_user
from aperag.websearch import ReaderService, SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


def deduplicate_search_results(
    llm_txt_results: List[WebSearchResultItem], 
    regular_results: List[WebSearchResultItem]
) -> List[WebSearchResultItem]:
    """
    Deduplicate search results with priority rules:
    1. LLM.txt results are placed first
    2. For URL conflicts, regular search results take priority (better descriptions)
    3. Maintain original relative order within each category
    
    Args:
        llm_txt_results: Results from LLM.txt search
        regular_results: Results from regular search
        
    Returns:
        Deduplicated and ordered list of search results
    """
    if not llm_txt_results and not regular_results:
        return []
    
    if not llm_txt_results:
        return regular_results
    
    if not regular_results:
        return llm_txt_results
    
    # Create URL index for regular results (higher priority for conflicts)
    regular_url_map = {result.url: result for result in regular_results}
    
    # Start with LLM.txt results, but replace any that conflict with regular results
    final_results = []
    used_urls = set()
    
    # Process LLM.txt results first
    for llm_result in llm_txt_results:
        if llm_result.url in regular_url_map:
            # URL conflict: use regular result instead (better description)
            regular_result = regular_url_map[llm_result.url]
            if regular_result.url not in used_urls:
                final_results.append(regular_result)
                used_urls.add(regular_result.url)
        else:
            # No conflict: use LLM.txt result
            if llm_result.url not in used_urls:
                final_results.append(llm_result)
                used_urls.add(llm_result.url)
    
    # Add remaining regular results that weren't used for conflicts
    for regular_result in regular_results:
        if regular_result.url not in used_urls:
            final_results.append(regular_result)
            used_urls.add(regular_result.url)
    
    return final_results


@router.post("/web/search", response_model=WebSearchResponse, tags=["websearch"])
@audit(resource_type="search", api_name="WebSearch")
async def web_search(http_request: Request, request: WebSearchRequest, user: User = Depends(current_user)):
    """
    Perform web search to find relevant information on the internet.

    Supports multiple search engines including DuckDuckGo and JINA AI.
    Results are returned in a structured format with ranking and metadata.
    """
    try:
        # Validate request parameters
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Validate parameter ranges
        if request.max_results <= 0:
            raise HTTPException(status_code=400, detail="max_results must be positive")
        if request.max_results > 50:
            raise HTTPException(status_code=400, detail="max_results cannot exceed 50")
        if request.timeout <= 0:
            raise HTTPException(status_code=400, detail="timeout must be positive")
        if request.timeout > 300:
            raise HTTPException(status_code=400, detail="timeout cannot exceed 300 seconds")
        
        # Sanitize query string
        query = request.query.strip()
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query cannot exceed 1000 characters")

        # Log the search request
        logger.info(f"Web search request from user {user.id}: query='{query}', engine={request.search_engine}, "
                   f"llm_txt={request.search_llms_txt or False}, source={request.source}")

        # Determine what searches to perform
        need_llm_txt = request.search_llms_txt and request.source
        need_regular = query or request.source
        
        # If neither search is needed, return empty results
        if not need_llm_txt and not need_regular:
            logger.info(f"No search needed for user {user.id} - no query or sources provided")
            return WebSearchResponse(
                query=query,
                results=[],
                search_engine=request.search_engine,
                total_results=0,
                search_time=0.0,
            )

        # Create concurrent search tasks
        async def llm_txt_search():
            async with SearchService.create_with_provider("llm_txt") as service:
                llm_request = WebSearchRequest(
                    query=query,
                    max_results=request.max_results,
                    search_engine="llm_txt",
                    timeout=request.timeout,
                    locale=request.locale,
                    source=request.source,
                    use_source_domain_only=request.use_source_domain_only,
                )
                return await service.search(llm_request)

        async def regular_search():
            async with SearchService() as service:
                regular_request = WebSearchRequest(
                    query=query,
                    max_results=request.max_results,
                    search_engine=request.search_engine,
                    timeout=request.timeout,
                    locale=request.locale,
                    source=request.source,
                    use_source_domain_only=request.use_source_domain_only,
                )
                return await service.search(regular_request)

        # Execute searches concurrently
        tasks = []
        if need_llm_txt:
            logger.info(f"Adding LLM.txt search for user {user.id}")
            tasks.append(llm_txt_search())
        if need_regular:
            logger.info(f"Adding regular search for user {user.id}")
            tasks.append(regular_search())

        try:
            search_responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Search execution failed for user {user.id}: {e}")
            raise e

        # Process search results with deduplication
        llm_txt_results = []
        regular_results = []
        total_search_time = 0.0
        effective_search_engine = request.search_engine

        for i, response in enumerate(search_responses):
            if isinstance(response, Exception):
                search_type = "LLM.txt" if (i == 0 and need_llm_txt) else "regular"
                logger.warning(f"{search_type} search failed for user {user.id}: {response}")
                continue
            
            total_search_time += response.search_time
            
            # Categorize results by search type
            if i == 0 and need_llm_txt:
                # First task is LLM.txt search
                llm_txt_results = response.results
            elif (i == 0 and not need_llm_txt) or (i == 1 and need_llm_txt):
                # Regular search results
                regular_results = response.results
                effective_search_engine = response.search_engine

        # Deduplicate and merge results
        deduplicated_results = deduplicate_search_results(llm_txt_results, regular_results)
        
        # Re-rank and limit results
        for i, result in enumerate(deduplicated_results):
            result.rank = i + 1
        final_results = deduplicated_results[:request.max_results]
        
        # Log deduplication results
        total_before = len(llm_txt_results) + len(regular_results)
        total_after = len(deduplicated_results)
        if total_before > total_after:
            logger.info(f"Deduplicated results for user {user.id}: {total_before} -> {total_after} (-{total_before - total_after} duplicates)")

        # Create response
        response = WebSearchResponse(
            query=query,
            results=final_results,
            search_engine=effective_search_engine,
            total_results=len(final_results),
            search_time=total_search_time,
        )

        # Log successful search
        logger.info(
            f"Web search completed for user {user.id}: {len(response.results)} results in {response.search_time:.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Web search failed for user {user.id}: {e}")

        # Handle specific errors
        if "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Search request timed out")
        elif "cannot be empty" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))
        elif "api key" in str(e).lower():
            raise HTTPException(status_code=401, detail="API key required or invalid")
        else:
            raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")


@router.post("/web/read", response_model=WebReadResponse, tags=["websearch"])
@audit(resource_type="search", api_name="WebRead")
async def web_read(http_request: Request, request: WebReadRequest, user: User = Depends(current_user)):
    """
    Read and extract content from web pages.

    Supports reading single or multiple URLs concurrently.
    Content is extracted in Markdown format with metadata.
    """
    try:
        # Validate request
        if not request.urls:
            raise HTTPException(status_code=400, detail="URLs cannot be empty")
        
        # Validate parameter ranges
        if request.timeout <= 0:
            raise HTTPException(status_code=400, detail="timeout must be positive")
        if request.timeout > 300:
            raise HTTPException(status_code=400, detail="timeout cannot exceed 300 seconds")

        # Normalize URLs to list for logging and validation
        if isinstance(request.urls, str):
            url_list = [request.urls]
        else:
            url_list = request.urls
            
        # Limit number of URLs
        if len(url_list) > 10:
            raise HTTPException(status_code=400, detail="Cannot process more than 10 URLs at once")
            
        # Basic URL validation
        from aperag.websearch.utils.url_validator import URLValidator
        for url in url_list:
            if not URLValidator.is_valid_url(url):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")

        # Log the read request
        logger.info(f"Web read request from user {user.id}: {len(url_list)} URLs, timeout={request.timeout}s")

        # Create reader service and ensure proper cleanup
        async with ReaderService() as reader_service:
            # Perform reading
            response = await reader_service.read(request)

            # Log successful read
            logger.info(
                f"Web read completed for user {user.id}: {response.successful}/{response.total_urls} successful in {response.processing_time:.2f}s"
            )

            return response

    except Exception as e:
        logger.error(f"Web read failed for user {user.id}: {e}")

        # Handle specific errors
        if "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Read request timed out")
        elif "urls" in str(e).lower() and "empty" in str(e).lower():
            raise HTTPException(status_code=400, detail="URLs list cannot be empty")
        elif "invalid url" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=f"Web read failed: {str(e)}")
