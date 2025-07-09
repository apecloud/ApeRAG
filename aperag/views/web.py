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

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from aperag.db.models import User
from aperag.schema.view_models import WebReadRequest, WebReadResponse, WebSearchRequest, WebSearchResponse
from aperag.utils.audit_decorator import audit
from aperag.views.auth import current_user
from aperag.websearch import ReaderService, SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/web/search", response_model=WebSearchResponse, tags=["websearch"])
@audit(resource_type="search", api_name="WebSearch")
async def web_search(http_request: Request, request: WebSearchRequest, user: User = Depends(current_user)):
    """
    Perform web search to find relevant information on the internet.

    Supports multiple search engines including DuckDuckGo and JINA AI.
    Results are returned in a structured format with ranking and metadata.
    """
    try:
        # Validate request
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")

        # Log the search request
        logger.info(f"Web search request from user {user.id}: query='{request.query}', engine={request.search_engine}")

        # Create search service and ensure proper cleanup
        async with SearchService() as search_service:
            # Perform search
            response = await search_service.search(request)

            # Log successful search
            logger.info(
                f"Web search completed for user {user.id}: {len(response.results)} results in {response.search_time:.2f}s"
            )

            return response

    except Exception as e:
        logger.error(f"Web search failed for user {user.id}: {e}")

        # Handle specific provider errors
        if "SearchProviderError" in str(type(e)):
            raise HTTPException(status_code=500, detail=f"Search provider error: {str(e)}")
        elif "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Search request timed out")
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

        # Normalize URLs to list for logging
        if isinstance(request.urls, str):
            url_list = [request.urls]
        else:
            url_list = request.urls

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

        # Handle specific provider errors
        if "ReaderProviderError" in str(type(e)):
            raise HTTPException(status_code=500, detail=f"Reader provider error: {str(e)}")
        elif "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Read request timed out")
        elif "urls list cannot be empty" in str(e).lower():
            raise HTTPException(status_code=400, detail="URLs list cannot be empty")
        else:
            raise HTTPException(status_code=500, detail=f"Web read failed: {str(e)}")
