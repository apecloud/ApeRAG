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
from aperag.schema.view_models import CollectionList, SearchResult, WebReadResponse, WebSearchResponse

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp_server = FastMCP("ApeRAG")

# Base URL for internal API calls
API_BASE_URL = "http://localhost:8000"


@mcp_server.tool
async def list_collections() -> Dict[str, Any]:
    """List all collections available to the user.

    Returns:
        List of collections with only essential information (id, title, description)
        for security and optimized LLM search.

    Note:
        Uses CollectionList view model for type-safe response parsing but filters
        sensitive and unnecessary information.
    """
    try:
        api_key = get_api_key()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/collections", headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                try:
                    # Parse response using view model for type safety
                    collection_list = CollectionList.model_validate(response.json())

                    # Filter collection data to only include essential information
                    # by directly modifying the CollectionList object and setting unneeded fields to None
                    if collection_list.items:
                        for collection in collection_list.items:
                            # Keep essential fields and set sensitive fields to None
                            # This preserves the original object structure for better maintainability
                            collection.config = None
                            collection.created = None
                            collection.updated = None
                            collection.source = None
                            # Type and status are kept for compatibility and filtering

                    # Return the modified object using model_dump()
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
    use_fulltext_index: bool = False,
    use_graph_index: bool = True,
    topk: int = 5,
    query_keywords: list[str] = None,
) -> Dict[str, Any]:
    """Search for knowledge in a specific collection using vector, full-text, and/or graph search.

    Args:
        collection_id: The ID of the collection to search in
        query: The search query
        query_keywords: The keywords extracted from query to use for fulltext search (optional), only effective when use_fulltext_index is True.
        use_vector_index: Whether to use vector/semantic search (default: True)
        use_fulltext_index: Whether to use full-text keyword search (default: False)
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
            search_data["vector_search"] = {"topk": topk, "similarity": 0.2}

        if use_fulltext_index:
            search_data["fulltext_search"] = {"topk": topk, "keywords": query_keywords}

        if use_graph_index:
            search_data["graph_search"] = {"topk": topk}

        # Ensure at least one search type is enabled
        if not any([use_vector_index, use_fulltext_index, use_graph_index]):
            return {"error": "At least one search type must be enabled"}

        # Use longer timeout for search operations (graph search can be time-consuming)
        async with httpx.AsyncClient(timeout=120.0) as client:
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


@mcp_server.tool
async def web_search(
    query: str = "",
    max_results: int = 5,
    timeout: int = 30,
    locale: str = "en-US",
    source: str = "",
    search_llms_txt: str = "",
) -> Dict[str, Any]:
    """Perform web search using various search engines with advanced domain targeting.

    Args:
        query: Search query for regular web search. Optional if only using LLM.txt discovery.
        max_results: Maximum number of results to return (default: 5)
        timeout: Request timeout in seconds (default: 30)
        locale: Browser locale (default: en-US)
        source: Optional domain or URL for site-specific filtering. When provided with query,
                limits search results to this domain (e.g., 'site:vercel.com query').
        search_llms_txt: Domain for LLM.txt discovery search. When provided, performs additional
                        LLM-optimized content discovery from the specified domain, independent
                        of the main search. Results are merged with regular search results.

    Returns:
        Web search results with URLs, titles, snippets, and metadata

    Note:
        Supports parallel execution of regular search and LLM.txt discovery.
        Results are automatically merged and ranked.
    """
    try:
        api_key = get_api_key()

        # Build search request
        search_data = {
            "max_results": max_results,
            "timeout": timeout,
            "locale": locale,
        }

        # Only include non-empty optional parameters
        if query and query.strip():
            search_data["query"] = query.strip()

        if source and source.strip():
            search_data["source"] = source.strip()

        if search_llms_txt and search_llms_txt.strip():
            search_data["search_llms_txt"] = search_llms_txt.strip()

        # Use longer timeout for web search operations
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/web/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=search_data,
            )
            if response.status_code == 200:
                try:
                    # Parse response using view model for type safety
                    search_response = WebSearchResponse.model_validate(response.json())
                    return search_response.model_dump()
                except Exception as e:
                    logger.error(f"Failed to parse web search response: {e}")
                    return {"error": "Failed to parse web search response", "details": str(e)}
            else:
                return {"error": f"Web search failed: {response.status_code}", "details": response.text}
    except ValueError as e:
        return {"error": str(e)}


@mcp_server.tool
async def web_read(
    url_list: list[str],
    timeout: int = 30,
    locale: str = "en-US",
    max_concurrent: int = 5,
) -> Dict[str, Any]:
    """Read and extract content from web pages.

    Args:
        url_list: List of URLs to read content from (for single URL, use array with one element)
        timeout: Request timeout in seconds (default: 30)
        locale: Browser locale (default: en-US)
        max_concurrent: Maximum concurrent requests for multiple URLs (default: 5)

    Returns:
        Web content reading results with extracted text, titles, word counts, and metadata

    Note:
        Uses WebReadResponse view model for type-safe response parsing
    """
    try:
        api_key = get_api_key()

        # Validate url_list parameter
        if not url_list or len(url_list) == 0:
            return {"error": "url_list parameter is required and must contain at least one URL"}

        # Build read request using the correct WebReadRequest model
        read_data = {
            "url_list": url_list,
            "timeout": timeout,
            "locale": locale,
            "max_concurrent": max_concurrent,
        }

        # Use longer timeout for web content reading operations
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/web/read",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=read_data,
            )
            if response.status_code == 200:
                try:
                    # Parse response using view model for type safety
                    read_response = WebReadResponse.model_validate(response.json())
                    return read_response.model_dump()
                except Exception as e:
                    logger.error(f"Failed to parse web read response: {e}")
                    return {"error": "Failed to parse web read response", "details": str(e)}
            else:
                return {"error": f"Web read failed: {response.status_code}", "details": response.text}
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
1. **list_collections**: Get all available collections with essential information (ID, title, description)
2. **search_collection**: Search within collections using multiple search methods
3. **web_search**: Perform web search using various search engines (Google, DuckDuckGo, Bing)
4. **web_read**: Read and extract content from web pages

## Authentication:
API authentication is handled automatically through one of these methods:
1. **HTTP Authorization header**: `Authorization: Bearer your-api-key` (when using HTTP transport)
2. **Environment variable**: `APERAG_API_KEY=your-api-key` (fallback method)

The server will automatically try both methods in order of preference.

## Quick Start:
1. First, get available collections with essential information: `list_collections()`
2. Choose a collection from the list
3. Search the collection: `search_collection(collection_id="abc123", query="your question")`
   (By default, vector and graph search are enabled for optimal performance)

## Search Types:
You can enable/disable any combination of search methods:
- **Vector search** (use_vector_index): Semantic similarity search using embeddings (default: True)
- **Full-text search** (use_fulltext_index): Traditional keyword-based text search (default: False)
- **Graph search** (use_graph_index): Knowledge graph-based search (default: True)

⚠️ **Important**: Full-text search can return large amounts of text content which may cause context window overflow with smaller LLM models. Use with caution and consider reducing topk when enabling fulltext search.

By default, vector and graph search are enabled for optimal balance of quality and context size.

## Example Workflow:
```
# Step 1: Get collections with essential information
collections = list_collections()

# Step 2: Choose a collection from the list
# (collections.items contains collection ID, title, and description)
collection_id = collections.items[0].id

# Step 3: Search with default methods (vector + graph)
results = search_collection(
    collection_id=collection_id,
    query="How to deploy applications?",
    use_vector_index=True,
    use_fulltext_index=False,
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

# Enable fulltext search with caution (may cause context overflow)
fulltext_search = search_collection(
    collection_id=collection_id,
    query="specific keywords",
    use_vector_index=True,
    use_fulltext_index=True,  # Enable with caution
    use_graph_index=True,
    topk=3  # Use smaller topk to manage context size
)
```

Your search results will include relevant documents with context, similarity scores, and metadata.

## Web Search and Content Reading:
You can also search the web and extract content from web pages:

### Web Search Example:
```
# Basic web search
web_results = web_search(
    query="ApeRAG RAG system 2025",
    max_results=5,
    locale="zh-CN"
)

# Site-specific regular search
site_results = web_search(
    query="deployment documentation",
    source="vercel.com",  # limit search to vercel.com domain
    max_results=10
)

# LLM.txt discovery search (independent)
llms_txt_results = web_search(
    search_llms_txt="anthropic.com",  # discover LLM.txt content from anthropic.com
    max_results=5
)

# Combined search: regular + LLM.txt discovery
combined_results = web_search(
    query="machine learning tutorials",
    source="docs.python.org",  # regular search limited to Python docs
    search_llms_txt="openai.com",  # plus LLM.txt discovery from OpenAI
    max_results=8
)

# Search results include URLs, titles, snippets, and domains
for result in web_results.results:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Snippet: {result.snippet}")
    print(f"Domain: {result.domain}")
```

### Web Content Reading Example:
```
# Read content from web pages (single URL - use array with one element)
content = web_read(
    url_list=["https://example.com/article"],  # single URL in array
    timeout=30
)

# Read from multiple URLs
content = web_read(
    url_list=["https://example.com/page1", "https://example.com/page2"],  # multiple URLs
    max_concurrent=2
)

# Content includes extracted text, titles, word counts
for result in content.results:
    if result.status == "success":
        print(f"Title: {result.title}")
        print(f"Content: {result.content}")
        print(f"Word Count: {result.word_count}")
```

### Combined Workflow Example:
```
# 1. Search web for recent information with LLM.txt discovery
web_results = web_search(
    query="latest AI developments 2025", 
    source="anthropic.com",  # limit regular search to Anthropic's content
    search_llms_txt="anthropic.com",  # discover LLM-optimized content from Anthropic
    max_results=3
)

# 2. Extract URLs from search results
urls = [result.url for result in web_results.results]

# 3. Read full content from those pages
web_content = web_read(url_list=urls, max_concurrent=2)

# 4. Search your internal knowledge base for related information
collections = list_collections()
if collections.items:
    internal_results = search_collection(
        collection_id=collections.items[0].id,
        query="AI developments",
        topk=5
    )

# 5. Combine results for comprehensive analysis
print("=== Web Results ===")
for result in web_results.results:
    print(f"[{result.domain}] {result.title}: {result.url}")

print("\n=== Web Content ===")
for content in web_content.results:
    if content.status == "success":
        print(f"📄 {content.title} ({content.word_count} words)")

print("\n=== Internal Knowledge ===")
for item in internal_results.items:
    print(f"🔍 {item.content[:100]}...")

# Now you have both web and internal knowledge base results!
```
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
- 📚 **Browse your collections** to understand what data you have (with essential details)
- 🎯 **Find specific information** with precise queries
- 💡 **Suggest search strategies** for complex queries
- 🌐 **Search the web** for latest information using multiple search engines
- 📄 **Read web content** and extract clean text from any web page
- 🔗 **Combine web and internal search** for comprehensive results
- 🤖 **LLM.txt discovery** for AI-optimized content from any domain
- 🎯 **Domain-targeted search** with flexible result filtering
- 🏢 **Site-specific search** to focus on specific websites or domains

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
