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
import os
from typing import Any, Dict

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

# Import view models for type safety
from aperag.schema.view_models import CollectionList, SearchResult

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp_server = FastMCP("ApeRAG")

# Base URL for internal API calls
API_BASE_URL = "http://localhost:8000"


@mcp_server.tool
async def list_collections() -> Dict[str, Any]:
    """List all collections available to the user.

    Returns:
        List of collections with their metadata (CollectionList format)

    Note:
        Uses CollectionList view model for type-safe response parsing
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/collections", headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                try:
                    # Parse response using view model for type safety
                    collection_list = CollectionList.model_validate(response.json())
                    return collection_list.model_dump()
                except Exception as e:
                    logger.error(f"Failed to parse collections response: {e}")
                    return {"error": "Failed to parse collections response", "details": str(e)}
            else:
                return {"error": f"Failed to fetch collections: {response.status_code}", "details": response.text}
    except ValueError as e:
        return {"error": str(e)}


@mcp_server.tool
async def search_collection(
    collection_id: str,
    query: str,
    use_vector_index: bool = True,
    use_fulltext_index: bool = True,
    use_graph_index: bool = True,
    topk: int = 5,
) -> Dict[str, Any]:
    """Search for knowledge in a specific collection using vector, full-text, and/or graph search.

    Args:
        collection_id: The ID of the collection to search in
        query: The search query
        use_vector_index: Whether to use vector/semantic search (default: True)
        use_fulltext_index: Whether to use full-text keyword search (default: True)
        use_graph_index: Whether to use knowledge graph search (default: True)
        topk: Maximum number of results to return per search type (default: 10)

    Returns:
        Search results with relevant documents and metadata (SearchResult format)

    Note:
        Uses SearchResult view model for type-safe response parsing and validation
    """
    try:
        api_key = get_api_key()

        # Build search request based on enabled search types
        search_data = {"query": query}

        # Add search configurations for enabled types
        if use_vector_index:
            search_data["vector_search"] = {"topk": topk, "similarity": 0.7}

        if use_fulltext_index:
            search_data["fulltext_search"] = {"topk": topk}

        if use_graph_index:
            search_data["graph_search"] = {"topk": topk}

        # Ensure at least one search type is enabled
        if not any([use_vector_index, use_fulltext_index, use_graph_index]):
            return {"error": "At least one search type must be enabled"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/collections/{collection_id}/searches",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=search_data,
            )
            if response.status_code == 200 or response.status_code == 201:
                try:
                    # Parse response using view model for type safety
                    search_result = SearchResult.model_validate(response.json())

                    # Ensure returned results don't exceed topk limit
                    # This provides additional protection in case HTTP API doesn't apply global limit
                    if search_result.items and len(search_result.items) > topk:
                        search_result.items = search_result.items[:topk]
                        # Update ranks if they exist
                        for i, item in enumerate(search_result.items):
                            if item.rank is not None:
                                item.rank = i + 1

                    return search_result.model_dump()
                except Exception as e:
                    logger.error(f"Failed to parse search response: {e}")
                    return {"error": "Failed to parse search response", "details": str(e)}
            else:
                return {"error": f"Search failed: {response.status_code}", "details": response.text}
    except ValueError as e:
        return {"error": str(e)}


# Add a resource for ApeRAG usage information
@mcp_server.resource("aperag://usage-guide")
async def aperag_usage_guide() -> str:
    """Resource providing usage guide for ApeRAG search."""
    return """
# ApeRAG Search Guide

ApeRAG provides powerful knowledge search capabilities across your collections.

## Available Operations:
1. **list_collections**: Get all available collections with complete details
2. **search_collection**: Search within collections using multiple search methods

## Authentication:
API authentication is handled automatically through one of these methods:
1. **HTTP Authorization header**: `Authorization: Bearer your-api-key` (when using HTTP transport)
2. **Environment variable**: `APERAG_API_KEY=your-api-key` (fallback method)

The server will automatically try both methods in order of preference.

## Quick Start:
1. First, get available collections with complete details: `list_collections()`
2. Choose a collection from the list
3. Search the collection: `search_collection(collection_id="abc123", query="your question")`
   (By default, all search types are enabled for comprehensive results)

## Search Types:
You can enable/disable any combination of search methods:
- **Vector search** (use_vector_index): Semantic similarity search using embeddings
- **Full-text search** (use_fulltext_index): Traditional keyword-based text search
- **Graph search** (use_graph_index): Knowledge graph-based search

By default, all three search types are enabled for comprehensive results (hybrid search).

## Example Workflow:
```
# Step 1: Get collections with complete details
collections = list_collections()

# Step 2: Choose a collection from the list
# (collections.items contains all collection details)
collection_id = collections.items[0].id

# Step 3: Search with all methods (hybrid search)
results = search_collection(
    collection_id=collection_id,
    query="How to deploy applications?",
    use_vector_index=True,
    use_fulltext_index=True,
    use_graph_index=True,
    topk=5
)

# Or search with only specific methods
vector_only = search_collection(
    collection_id=collection_id,
    query="deployment strategies",
    use_vector_index=True,
    use_fulltext_index=False,
    use_graph_index=False,
    topk=10
)
```

Your search results will include relevant documents with context, similarity scores, and metadata.
"""


# Add a prompt for search assistance
@mcp_server.prompt
async def search_assistant() -> str:
    """Help prompt for effective ApeRAG searching."""
    return """
# ApeRAG Search Assistant

I can help you search your knowledge base effectively using ApeRAG.

## How to use me:
1. **Tell me what you're looking for** - I'll help you search across your collections
2. **Ask specific questions** - I can find relevant documents and provide context
3. **Explore collections** - I can show you what collections are available

## What I can do:
- 🔍 **Search your knowledge base** using multiple search methods
- 📚 **Browse your collections** to understand what data you have (with complete details)
- 🎯 **Find specific information** with precise queries
- 💡 **Suggest search strategies** for complex queries

## Search Tips:
- Use **specific terms** for better results
- **Combine different search methods** by enabling/disabling vector, fulltext, and graph indexes
- **Combine keywords** with natural language questions
- **Adjust topk values** based on your needs (number of results per search type)
- Enable **all search types** for comprehensive results, or **specific types** for focused searches

## Authentication:
API authentication is handled automatically through:
1. **HTTP Authorization header**: `Authorization: Bearer your-api-key` (preferred for HTTP transport)
2. **Environment variable**: `APERAG_API_KEY=your-api-key` (fallback method)

Make sure at least one authentication method is properly configured in your MCP client.

Ready to help you find the information you need!
"""


def get_api_key() -> str:
    """Get API key from HTTP headers or environment variable.

    Priority order:
    1. Authorization header from HTTP request (using FastMCP dependency)
    2. APERAG_API_KEY environment variable

    Returns:
        API key string

    Raises:
        ValueError: If API key is not found
    """
    # Try to get API key from HTTP headers first
    try:
        # Use FastMCP's dependency function to get HTTP headers
        headers = get_http_headers()

        if headers:
            # Try to extract Authorization header
            auth_header = headers.get("Authorization") or headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove 'Bearer ' prefix
                logger.info(f"API key found in Authorization header, length: {len(api_key)}")
                return api_key

    except Exception as e:
        # get_http_headers() might fail if not in HTTP request context
        logger.debug(f"Could not extract API key from headers: {e}")

    # Fallback to environment variable
    api_key = os.getenv("APERAG_API_KEY")

    if api_key:
        logger.info(f"API key found in environment variable, length: {len(api_key)}")
        return api_key

    raise ValueError(
        "API key not found. Please provide API key via:\n"
        "1. Authorization: Bearer <token> HTTP header, or\n"
        "2. APERAG_API_KEY environment variable"
    )


# Export the server instance
__all__ = ["mcp_server"]
