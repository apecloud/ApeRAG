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
from typing import Any, Dict, Optional

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

# Import view models for type safety
from aperag.domains.retrieval.schemas import SearchResult
from aperag.domains.web_access.schemas import WebReadResponse
from aperag.mcp.capabilities import ToolAnnotation
from aperag.mcp.tools import (
    ByteRange,
)
from aperag.mcp.tools import (
    get_collection_metadata as _d10c_get_collection_metadata,
)
from aperag.mcp.tools import (
    get_document_metadata as _d10c_get_document_metadata,
)
from aperag.mcp.tools import (
    list_collections as _d10c_list_collections,
)
from aperag.mcp.tools import (
    list_documents as _d10c_list_documents,
)
from aperag.mcp.tools import (
    read_document as _d10c_read_document,
)
from aperag.mcp.tools import (
    read_document_chunk as _d10c_read_document_chunk,
)
from aperag.mcp.tools import (
    read_document_outline as _d10c_read_document_outline,
)
from aperag.mcp.tools import (
    read_document_section as _d10c_read_document_section,
)
from aperag.mcp.tools._annotations import register as _register_tool_annotation

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp_server = FastMCP("ApeRAG")

# Base URL for internal API calls
API_BASE_URL = "http://localhost:8000"


# === D10.c read primitives ===
#
# Per docs/modularization/d10-design-pack.md §A — 8 read primitives that
# replace the legacy HTTP-delegated list_collections + add 7 net-new
# tools (list_documents / get_collection_metadata / get_document_metadata
# / read_document / read_document_outline / read_document_section /
# read_document_chunk).
#
# Each primitive enforces (in order, never cache-shortcut per §E.7):
#   1. tenancy gate (D9 base canonical SoT — db_ops.query_collection)
#   2. 3-level authorization (D9 §2 — tools/authorization.py)
#   3. parse_version computation (only the 4 parse-version-keyed primitives)
#   4. authoritative fetch (un-cached; D10.g #99 wires cache around this)
#
# chenyexuan's D10.d (#96) split-search registrations land adjacent to
# this block — append below the marker, no merge churn expected.


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "list_collections",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def list_collections(
    cursor: Optional[str] = None,
    limit: int = 50,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    title_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover which knowledge bases the current user can access.

    Use this when:
    - You need to find available knowledge bases before choosing where to search.
    - The user asks what collections or knowledge bases are available.

    Do not use this when:
    - The user already specified a target collection and you can search it directly.
    - You need to search temporary files uploaded in the current chat.

    What success means:
    - You received the collections currently accessible to the user.

    What an empty result means:
    - The current user has no accessible collections in this environment.
    - It does not automatically mean the system is broken.

    What failure may mean:
    - auth / permission: the request is missing a valid ApeRAG credential or access right.
    - network / timeout: the MCP or backend path did not complete in time.

    How to explain this step to the user:
    - While running: "Checking which knowledge bases are available."
    - After completion: "Checked which knowledge bases are available."

    Returns:
        Paginated CollectionList envelope per D10 §A.1 (items + next_cursor + total_count).
    """
    result = await _d10c_list_collections(
        cursor=cursor,
        limit=limit,
        sort_by=sort_by,  # type: ignore[arg-type]
        sort_order=sort_order,  # type: ignore[arg-type]
        title_filter=title_filter,
    )
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "list_documents",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def list_documents(
    collection_id: str,
    cursor: Optional[str] = None,
    limit: int = 50,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    title_filter: Optional[str] = None,
    type_filter: Optional[list[str]] = None,
    indexed_only: bool = False,
) -> Dict[str, Any]:
    """List documents within a collection. D10 §A.2."""
    result = await _d10c_list_documents(
        collection_id,
        cursor=cursor,
        limit=limit,
        sort_by=sort_by,  # type: ignore[arg-type]
        sort_order=sort_order,  # type: ignore[arg-type]
        title_filter=title_filter,
        type_filter=type_filter,
        indexed_only=indexed_only,
    )
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "get_collection_metadata",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def get_collection_metadata(collection_id: str) -> Dict[str, Any]:
    """Get full metadata for a specific collection. D10 §A.4."""
    result = await _d10c_get_collection_metadata(collection_id)
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "get_document_metadata",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def get_document_metadata(collection_id: str, document_id: str) -> Dict[str, Any]:
    """Get metadata for a specific document. D10 §A.3."""
    result = await _d10c_get_document_metadata(collection_id, document_id)
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "read_document",
        ToolAnnotation(
            requires=("collection_access",),
            # read_document streams the full parsed markdown body —
            # callers without long-context tolerance should prefer
            # read_document_section / read_document_chunk.
            capabilities={"long_context": True},
        ),
    ),
)
async def read_document(
    collection_id: str,
    document_id: str,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
) -> Dict[str, Any]:
    """Read parsed markdown content of a document. D10 §A.5.

    Optional byte range is best-effort and NOT stable across re-parse.
    """
    byte_range: Optional[ByteRange] = None
    if range_start is not None and range_end is not None:
        byte_range = ByteRange(start=range_start, end=range_end)
    result = await _d10c_read_document(collection_id, document_id, range=byte_range)
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "read_document_outline",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def read_document_outline(
    collection_id: str,
    document_id: str,
    max_depth: int = 6,
) -> Dict[str, Any]:
    """Read the heading tree (table of contents) of a document. D10 §A.6."""
    result = await _d10c_read_document_outline(collection_id, document_id, max_depth=max_depth)
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "read_document_section",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def read_document_section(
    collection_id: str,
    document_id: str,
    section_path: Optional[str] = None,
    heading_anchor: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a section by section_path (preferred) or heading_anchor. D10 §A.7."""
    result = await _d10c_read_document_section(
        collection_id,
        document_id,
        section_path=section_path,
        heading_anchor=heading_anchor,
    )
    return result.model_dump()


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "read_document_chunk",
        ToolAnnotation(
            requires=("collection_access",),
            capabilities={"long_context": False},
        ),
    ),
)
async def read_document_chunk(
    collection_id: str,
    document_id: str,
    chunk_id: str,
) -> Dict[str, Any]:
    """Read a chunk by stable chunk_id. D10 §A.8."""
    result = await _d10c_read_document_chunk(collection_id, document_id, chunk_id)
    return result.model_dump()


# === end D10.c read primitives ===


@mcp_server.tool
async def search_collection(
    collection_id: str,
    query: str,
    use_vector_index: bool = True,
    use_fulltext_index: bool = True,
    use_graph_index: bool = True,
    use_summary_index: bool = True,
    use_vision_index: bool = True,
    rerank: bool = True,
    topk: int = 5,
    query_keywords: list[str] = None,
) -> Dict[str, Any]:
    """[DEPRECATED] Search a persistent knowledge base for evidence relevant to the current request.

    [DEPRECATED] Phase 9 D10.d (#96, ``docs/modularization/d10-design-pack.md``
    §B.5 / §H.1): use the discrete split tools instead —
    ``vector_search`` / ``graph_search`` / ``fulltext_search``. This
    omnibus tool is preserved as a deprecated alias for backward
    compatibility during the D10 migration window and will be removed in
    D11 once telemetry confirms no remaining external callers (D10.h
    cutover lane). Implementation is intentionally untouched.

    Use this when:
    - You already know which collection should be searched.
    - The user asks a knowledge question that should be answered from indexed documents.

    Do not use this when:
    - The target files were uploaded only in the current chat; use search_chat_files instead.
    - No collection has been selected or discovered yet.

    What success means:
    - You retrieved candidate evidence from the chosen collection.

    What an empty result means:
    - This collection did not return strong evidence for the current query.
    - It does not prove the answer is false; it means this source did not support the step.

    What failure may mean:
    - auth / permission: the current user cannot access this collection.
    - network / timeout: the search path did not complete.
    - bad request: the collection ID or search config is invalid.

    How to explain this step to the user:
    - While running: "Searching the selected knowledge base for evidence about the request."
    - After completion: "Reviewed results from the selected knowledge base."

    Args:
        collection_id: The ID of the collection to search in
        query: The search query
        query_keywords: The keywords extracted from query to use for fulltext search (optional), only effective when use_fulltext_index is True.
        use_vector_index: Whether to use vector/semantic search (default: True)
        use_fulltext_index: Whether to use full-text keyword search (default: True)
        use_graph_index: Whether to use knowledge graph search (default: True)
        use_summary_index: Whether to use summary search (default: True)
        use_vision_index: Whether to use vision search (default: True)
        rerank: Whether to enable reranking of search results for better relevance (default: True)
        topk: Maximum number of results to return per search type (default: 5)

    Returns:
        Search results with relevant documents and metadata (SearchResult format)

    Note:
        Uses SearchResult view model for type-safe response parsing and validation.

        ```
        class SearchResultItem(BaseModel):
            rank: Optional[int] = Field(None, description='Result rank')
            score: Optional[float] = Field(None, description='Result score')
            content: Optional[str] = Field(None, description='Result content')
            source: Optional[str] = Field(None, description='Source document or metadata')
            recall_type: Optional[
                Literal['vector_search', 'graph_search', 'fulltext_search', 'summary_search']
            ] = Field(None, description='Recall type')
            metadata: Optional[dict[str, Any]] = Field(
                None, description='Metadata of the result'
            )


        class SearchResult(BaseModel):
            id: Optional[str] = Field(None, description='The id of the search result')
            query: Optional[str] = None
            vector_search: Optional[VectorSearchParams] = None
            fulltext_search: Optional[FulltextSearchParams] = None
            graph_search: Optional[GraphSearchParams] = None
            summary_search: Optional[SummarySearchParams] = None
            vision_search: Optional[VisionSearchParams] = None
            items: Optional[list[SearchResultItem]] = None
            created: Optional[datetime] = Field(
                None, description='The creation time of the search result'
            )
        ```

        The `result.items[x].metadata["page_idx"]` field indicates that the item's content is from page `page_idx` of the document (`metadata["source"]`). Note that `page_idx` is 0-indexed.

        Vector search results may include images. Images are indexed in two ways:
        1.  A multimodal embedding model converts the image into a vector. Since text and images share the same vector space, you can use text for semantic search.
        2.  A Vision LLM generates a text description of the image, which is then converted into a vector by a text embedding model. This also enables retrieval based on vector similarity.

        If `result.items[x].metadata["indexer"]` is "vision", the item is an image.
        - If `item.content` is empty, the image was retrieved via multimodal embedding.
        - If `item.content` is not empty, it contains a visual description of the image.

        Although the LLM's Tool message interface doesn't support direct image input (meaning you can't "see" the images, even as a vision model), you can use `item.content` to understand the image and answer questions.
        If you reference an image in your response, include its URL so the user can see it and understand your reasoning.

        If your final output is in Markdown, you can display the image using an image block, like `![](<asset_url>)`. Here's how to construct the `asset_url` in Python pseudo-code:

        ```python
        m = result.items[0].metadata
        if m.get("asset_id") and m.get("document_id") and m.get("collection_id") and m.get("mimetype"):
            asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
        ```

        The `asset_url` uses a special `asset://` scheme instead of `http/https`. This helps the front-end parse and handle it. It uses `asset_id` as the path and passes `document_id`, `collection_id`, and `mimetype` as query parameters. Note that `asset_id`, `document_id`, and `collection_id` are required to display the image and must not be omitted.
    """
    try:
        api_key = get_api_key()

        # Build search request based on enabled search types
        search_data = {"query": query, "rerank": rerank}

        # Add search configurations for enabled types
        if use_vector_index:
            search_data["vector_search"] = {"topk": topk, "similarity": 0.2}

        if use_fulltext_index:
            search_data["fulltext_search"] = {"topk": topk, "keywords": query_keywords}

        if use_graph_index:
            search_data["graph_search"] = {"topk": topk}

        if use_summary_index:
            search_data["summary_search"] = {"topk": topk, "similarity": 0.2}

        if use_vision_index:
            search_data["vision_search"] = {"topk": topk, "similarity": 0.2}

        # Ensure at least one search type is enabled
        if not any([use_vector_index, use_fulltext_index, use_graph_index, use_summary_index]):
            return {"error": "At least one search type must be enabled"}

        # Use longer timeout for search operations (graph search can be time-consuming)
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v2/collections/{collection_id}/searches",
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
async def search_chat_files(
    chat_id: str,
    query: str,
    use_vector_index: bool = True,
    use_fulltext_index: bool = True,
    rerank: bool = True,
    topk: int = 5,
) -> Dict[str, Any]:
    """[DEPRECATED] Search files uploaded in the current chat for evidence relevant to this turn.

    [DEPRECATED] Phase 9 D10.d (#96, ``docs/modularization/d10-design-pack.md``
    §H.2): chat-scoped omnibus search shares the deprecation timeline of
    ``search_collection``. The split tool surface (``vector_search`` /
    ``graph_search`` / ``fulltext_search``) is collection-scoped today;
    a chat-scoped equivalent will be sequenced in the D10.h cutover
    lane. Implementation is intentionally untouched.

    Use this when:
    - The user refers to files shared in this chat session.
    - You need evidence from temporary, chat-scoped documents.

    Do not use this when:
    - You are searching a persistent knowledge base; use search_collection instead.
    - No files were uploaded in the current chat.

    What success means:
    - You found candidate evidence inside the current chat's uploaded files.

    What an empty result means:
    - The uploaded files did not return useful evidence for this query.
    - It does not automatically mean the files are unreadable or the system failed.

    What failure may mean:
    - auth / permission: the current request cannot access this chat's files.
    - network / timeout: the search path did not complete.

    How to explain this step to the user:
    - While running: "Searching files uploaded in this chat."
    - After completion: "Reviewed results from files uploaded in this chat."

    Args:
        chat_id: The ID of the chat to search files in
        query: The search query
        use_vector_index: Whether to use vector/semantic search (default: True)
        use_fulltext_index: Whether to use full-text keyword search (default: True)
        rerank: Whether to enable reranking of search results for better relevance (default: True)
        topk: Maximum number of results to return per search type (default: 5)

    Returns:
        Search results with relevant documents and metadata (SearchResult format)

    Note:
        Uses SearchResult view model for type-safe response parsing and validation.

        SCOPE: This tool ONLY searches temporary files uploaded in the current chat.
        It does NOT search permanent knowledge collections.

        Return format follows the same structure as search_collection:
        - rank: Result rank
        - score: Result score
        - content: Result content
        - source: Source document or metadata
        - recall_type: Type of search that found this result
        - metadata: Additional metadata including page_idx, asset_id, etc.

        Images are handled the same way as in collection search:
        - metadata["indexer"] == "vision" indicates an image
        - Use asset:// URLs for displaying images in markdown
    """
    try:
        api_key = get_api_key()

        # Build search request based on enabled search types
        search_data = {"query": query, "rerank": rerank}

        # Add search configurations for enabled types
        if use_vector_index:
            search_data["vector_search"] = {"topk": topk, "similarity": 0.2}

        if use_fulltext_index:
            search_data["fulltext_search"] = {"topk": topk}

        # Ensure at least one search type is enabled
        if not any([use_vector_index, use_fulltext_index]):
            return {"error": "At least one search type must be enabled"}

        # Use longer timeout for search operations
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v2/chats/{chat_id}/search",
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
                    logger.error(f"Failed to parse chat search response: {e}")
                    return {"error": "Failed to parse chat search response", "details": str(e)}
            else:
                return {"error": f"Chat search failed: {response.status_code}", "details": response.text}
    except ValueError as e:
        return {"error": str(e)}


# NOTE(D10.d #96 §B.4): the ``web_search`` tool implementation moved to
# ``aperag.mcp.tools.search_web`` so all D10 search tools live in the
# ``aperag/mcp/tools/`` subpackage. Wire signature is preserved (no
# breaking change for external MCP callers); §B.4 spec parameter
# canonicalization (``top_k`` / kw-only / ``source: str | None``) is
# deferred to the D10.h cutover lane.


@mcp_server.tool
async def web_read(
    url_list: list[str],
    timeout: int = 30,
    locale: str = "en-US",
    max_concurrent: int = 5,
) -> Dict[str, Any]:
    """Read web pages and extract the content needed for the current request.

    Use this when:
    - You already have one or more URLs that need to be inspected.
    - The next step requires reading source content, not just search snippets.

    Do not use this when:
    - You still need to discover candidate URLs first; use web_search instead.
    - Web access is disabled for the current turn.

    What success means:
    - You extracted readable content from the requested URLs.

    What an empty result means:
    - The pages did not yield usable readable content for this step.

    What failure may mean:
    - network / timeout: the reader could not fetch or finish processing the URLs.
    - page access issue: the target page blocked access or could not be parsed successfully.

    How to explain this step to the user:
    - While running: "Reading content from web sources."
    - After completion: "Reviewed content from the selected web pages."

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
        logger.info(
            "MCP web_read request urls=%s timeout=%s locale=%s max_concurrent=%s",
            len(url_list or []),
            timeout,
            locale,
            max_concurrent,
        )

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
                f"{API_BASE_URL}/api/v2/web/read",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=read_data,
            )
            if response.status_code == 200:
                try:
                    # Parse response using view model for type safety
                    read_response = WebReadResponse.model_validate(response.json())
                    logger.info(
                        "MCP web_read completed urls=%s successful=%s failed=%s",
                        read_response.total_urls,
                        read_response.successful,
                        read_response.failed,
                    )
                    return read_response.model_dump()
                except Exception as e:
                    logger.error(f"Failed to parse web read response: {e}")
                    return {"error": "Failed to parse web read response", "details": str(e)}
            else:
                logger.warning(
                    "MCP web_read failed status=%s urls=%s body=%s",
                    response.status_code,
                    len(url_list or []),
                    response.text,
                )
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
   (By default, vector search, graph search, and reranking are enabled for optimal performance)

## Search Types:
You can enable/disable any combination of search methods:
- **Vector search** (use_vector_index): Semantic similarity search using embeddings (default: True)
- **Full-text search** (use_fulltext_index): Traditional keyword-based text search (default: True)
- **Graph search** (use_graph_index): Knowledge graph-based search (default: True)
- **Summary search** (use_summary_index): Search through document summaries (default: True)
- **Reranking** (rerank): AI-powered reranking for improved result relevance (default: True)

⚠️ **Important**: Full-text search can return large amounts of text content which may cause context window overflow with smaller LLM models. Use with caution and consider reducing topk when enabling fulltext search.

By default, vector search, full-text search, graph search, summary search, and reranking are enabled for comprehensive search coverage.

## Example Workflow:
```
# Step 1: Get collections with essential information
collections = list_collections()

# Step 2: Choose a collection from the list
# (collections.items contains collection ID, title, and description)
collection_id = collections.items[0].id

# Step 3: Search with default methods (vector + fulltext + graph + summary + rerank)
results = search_collection(
    collection_id=collection_id,
    query="How to deploy applications?",
    use_vector_index=True,
    use_fulltext_index=True,
    use_graph_index=True,
    use_summary_index=True,
    rerank=True,
    topk=5
)

# Or search with only specific methods
vector_only = search_collection(
    collection_id=collection_id,
    query="deployment strategies",
    use_vector_index=True,
    use_fulltext_index=False,
    use_graph_index=False,
    rerank=True,  # Rerank still enabled for better results
    topk=10
)

# Enable summary search for high-level document overviews
summary_search = search_collection(
    collection_id=collection_id,
    query="project overview",
    use_vector_index=True,
    use_fulltext_index=True,
    use_graph_index=True,
    use_summary_index=True,  # Enable summary search
    rerank=True,
    topk=5
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
# 1. Search web for recent information
web_results = web_search(
    query="latest AI developments 2025",
    source="anthropic.com",  # limit regular search to Anthropic's content
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
        rerank=True,  # Default rerank for better results
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
- 🌐 **Search the web** for latest information with domain targeting and best-effort fallback
- 📄 **Read web content** and extract clean text from any web page
- 🔗 **Combine web and internal search** for comprehensive results
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
        headers = get_http_headers(include={"authorization"})

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


# Phase 9 D10.d (#96, ``docs/modularization/d10-design-pack.md`` §B):
# import the split search tool functions so their ``@mcp_server.tool``
# decorators register the new surface (``vector_search`` /
# ``graph_search`` / ``fulltext_search`` / ``web_search``). The imports
# happen at the bottom of this module — after ``mcp_server``,
# ``API_BASE_URL``, and ``get_api_key`` are defined — to break the
# circular import cycle (``aperag.mcp.tools.search_*`` import from
# ``aperag.mcp.server``).
#
# Re-exporting the function symbols at module level preserves the
# existing ``aperag.mcp.server.web_search`` access path for backward
# compatibility with callers (e.g. ``tests/unit_test/test_mcp_server.py``)
# that read attributes off the server module directly.
from aperag.mcp.tools.search_fulltext import fulltext_search  # noqa: E402, F401
from aperag.mcp.tools.search_graph import graph_search  # noqa: E402, F401
from aperag.mcp.tools.search_vector import vector_search  # noqa: E402, F401
from aperag.mcp.tools.search_web import web_search  # noqa: E402, F401

# Export the server instance
__all__ = ["mcp_server"]
