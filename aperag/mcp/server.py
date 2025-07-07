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

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp_server = FastMCP("ApeRAG")

# Base URL for internal API calls
API_BASE_URL = "http://localhost:8000"


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


@mcp_server.tool
async def list_collections() -> Dict[str, Any]:
    """List all collections available to the user.

    Returns:
        List of collections with their metadata
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/collections", headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to fetch collections: {response.status_code}", "details": response.text}
    except ValueError as e:
        return {"error": str(e)}


@mcp_server.tool
async def get_collection(collection_id: str) -> Dict[str, Any]:
    """Get details of a specific collection.

    Args:
        collection_id: The ID of the collection to retrieve

    Returns:
        Collection details including metadata and configuration
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/collections/{collection_id}", headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to fetch collection: {response.status_code}", "details": response.text}
    except ValueError as e:
        return {"error": str(e)}


@mcp_server.tool
async def search_collection(
    collection_id: str, query: str, search_type: str = "hybrid", limit: int = 10
) -> Dict[str, Any]:
    """Search for knowledge in a specific collection using vector, full-text, or graph search.

    Args:
        collection_id: The ID of the collection to search in
        query: The search query
        search_type: Type of search - "vector", "fulltext", "graph", or "hybrid" (default)
        limit: Maximum number of results to return (default: 10)

    Returns:
        Search results with relevant documents and metadata
    """
    try:
        api_key = get_api_key()

        # Build proper search request based on search_type
        search_data = {"query": query}

        # Configure search parameters based on search type
        if search_type == "vector":
            search_data["vector_search"] = {"topk": limit, "similarity": 0.7}
        elif search_type == "fulltext":
            search_data["fulltext_search"] = {"topk": limit}
        elif search_type == "graph":
            search_data["graph_search"] = {"topk": limit}
        elif search_type == "hybrid":
            # For hybrid search, include all search types
            search_data["vector_search"] = {"topk": limit, "similarity": 0.7}
            search_data["fulltext_search"] = {"topk": limit}
            search_data["graph_search"] = {"topk": limit}
        else:
            # Default to vector search if unknown type
            search_data["vector_search"] = {"topk": limit, "similarity": 0.7}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/collections/{collection_id}/searches",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=search_data,
            )
            if response.status_code == 200 or response.status_code == 201:
                return response.json()
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
1. **list_collections**: Get all available collections
2. **get_collection**: Get details about a specific collection  
3. **search_collection**: Search within collections using multiple search methods

## Authentication:
API authentication is handled automatically through one of these methods:
1. **HTTP Authorization header**: `Authorization: Bearer your-api-key` (when using HTTP transport)
2. **Environment variable**: `APERAG_API_KEY=your-api-key` (fallback method)

The server will automatically try both methods in order of preference.

## Quick Start:
1. First, get available collections: `list_collections()`
2. Choose a collection and get its details: `get_collection(collection_id="abc123")`
3. Search the collection: `search_collection(collection_id="abc123", query="your question")`

## Search Types:
- **hybrid** (recommended): Combines vector, full-text, and graph search
- **vector**: Semantic similarity search using embeddings
- **fulltext**: Traditional keyword-based text search
- **graph**: Knowledge graph-based search

## Example Workflow:
```
# Step 1: Get collections
collections = list_collections()

# Step 2: Choose a collection
collection = get_collection(collection_id="collection-123")

# Step 3: Search
results = search_collection(
    collection_id="collection-123",
    query="How to deploy applications?", 
    search_type="hybrid",
    limit=5
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
- 📚 **Browse your collections** to understand what data you have
- 🎯 **Find specific information** with precise queries
- 💡 **Suggest search strategies** for complex queries

## Search Tips:
- Use **specific terms** for better results
- Try **different search types** (hybrid, vector, fulltext, graph)
- **Combine keywords** with natural language questions
- **Adjust result limits** based on your needs

## Authentication:
API authentication is handled automatically through:
1. **HTTP Authorization header**: `Authorization: Bearer your-api-key` (preferred for HTTP transport)
2. **Environment variable**: `APERAG_API_KEY=your-api-key` (fallback method)

Make sure at least one authentication method is properly configured in your MCP client.

Ready to help you find the information you need!
"""


# Export the server instance
__all__ = ["mcp_server"]
