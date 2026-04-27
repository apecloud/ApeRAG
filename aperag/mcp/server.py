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

# Base URL for internal API calls. Deployments can point the MCP server
# at a colocated API service without changing the public tool surface.
API_BASE_URL = os.getenv("APERAG_API_BASE_URL", "http://localhost:8000").rstrip("/")


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
# D10 search tools (vector_search / graph_search / fulltext_search /
# web_search) live in ``aperag.mcp.tools.search_*`` per D10.d #96. The
# legacy ``search_collection`` / ``search_chat_files`` omnibus tools
# were removed in D10.h #100 — callers must compose the split tools
# directly. ``web_search`` parameter canonicalization (`query`
# required, kw-only, ``top_k``, ``source: str | None``) was applied
# in the same cutover.


@mcp_server.tool(
    annotations=_register_tool_annotation(
        "web_read",
        ToolAnnotation(
            requires=("web_access",),
            capabilities={"long_context": False, "web_access": True},
        ),
    ),
)
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

ApeRAG exposes the canonical D10 MCP tool surface (per
``docs/modularization/d10-design-pack.md``):

## Read primitives (D10.c §A)
- ``list_collections`` / ``list_documents`` — paginated metadata
- ``get_collection_metadata`` / ``get_document_metadata``
- ``read_document`` / ``read_document_outline``
- ``read_document_section`` / ``read_document_chunk``

## Search primitives (D10.d §B)
- ``vector_search`` — semantic similarity over collection embeddings
- ``graph_search`` — knowledge-graph-derived evidence
- ``fulltext_search`` — keyword / full-text hits
- ``web_search`` — public-internet search (independent of collections)

The legacy ``search_collection`` / ``search_chat_files`` omnibus
tools were removed in D10.h #100; callers compose the split tools
directly. Pagination is opaque cursor (D10.e §C); see ``next_cursor``
on the paginated read primitives.

## Authentication
- HTTP transport: ``Authorization: Bearer <api-key>``
- stdio fallback: ``APERAG_API_KEY`` environment variable

## Quick start
```
collections = list_collections()
collection_id = collections.items[0].id

# Pick the right primitive for your question:
hits = vector_search(collection_id=collection_id, query="deploy app")
graph = graph_search(collection_id=collection_id, query="deploy app")
fts = fulltext_search(collection_id=collection_id, query="deploy app")

# Pull the raw evidence for the top hit:
chunk = read_document_chunk(
    collection_id=hits.items[0].metadata.collection_id,
    document_id=hits.items[0].metadata.document_id,
    chunk_id=hits.items[0].metadata.chunk_id,
)
```

## Web search + content read
```
results = web_search(query="ApeRAG 2025", top_k=5, locale="zh-CN")
content = web_read(url_list=[r.url for r in results.results])
```

Each ``SearchResultItem`` carries ``chunk_id`` / ``section_path`` /
``heading_anchor`` (D10.h Drift #1) so the caller can navigate
straight from a search hit into the read primitives without an
extra resolution step.
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
# Wave 7 §K.12.6 — graph entity search / subgraph expand / detail.
# Importing the module is what registers the three ``@mcp_server.tool``
# decorators with the FastMCP instance above.
from aperag.mcp.tools.graph_tools import (  # noqa: E402, F401
    expand_graph_subgraph,
    get_entity_detail,
    query_graph_entities,
)
from aperag.mcp.tools.search_fulltext import fulltext_search  # noqa: E402, F401
from aperag.mcp.tools.search_graph import graph_search  # noqa: E402, F401
from aperag.mcp.tools.search_vector import vector_search  # noqa: E402, F401
from aperag.mcp.tools.search_web import web_search  # noqa: E402, F401

# Export the server instance
__all__ = ["mcp_server"]
