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

"""Pydantic view models owned by the ``web_access`` canonical domain.

These classes were previously defined in ``aperag.schema.view_models``; they
were relocated here by the Phase 2a hard-cut so the ``web_access`` domain no
longer reaches back into the legacy aggregate module. OpenAPI component
names and shapes are intentionally preserved byte-for-byte.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class WebSearchRequest(BaseModel):
    """
    Web search request
    """

    query: Optional[str] = Field(
        None,
        description="Search query for web search. Optional when using source-only site browsing.",
        examples=["ApeRAG 2025年最新发展"],
    )
    max_results: Optional[int] = Field(5, description="Maximum number of results to return", examples=[5])
    timeout: Optional[int] = Field(30, description="Request timeout in seconds", examples=[30])
    locale: Optional[str] = Field("en-US", description="Browser locale", examples=["en-US"])
    source: Optional[str] = Field(
        None,
        description="Domain or URL for site-specific filtering. When provided with query, limits search results to this domain (e.g., 'site:vercel.com query').",
        examples=["vercel.com"],
    )


class WebSearchResultItem(BaseModel):
    """
    Individual web search result
    """

    rank: int = Field(..., description="Result rank", examples=[1])
    title: str = Field(..., description="Page title", examples=["ApeRAG 2025年技术路线图"])
    url: str = Field(
        ...,
        description="Page URL",
        examples=["https://example.com/aperag-2025-roadmap"],
    )
    snippet: str = Field(..., description="Page snippet", examples=["ApeRAG在2025年将重点发展..."])
    domain: str = Field(..., description="Domain name", examples=["example.com"])
    timestamp: Optional[datetime] = Field(None, description="Result timestamp", examples=["2025-01-01T00:00:00Z"])


class WebSearchMeta(BaseModel):
    """
    Lightweight execution diagnostics for web search.
    """

    search_status: Literal["ok", "empty", "unavailable", "disabled"] = Field(
        ...,
        description="Overall search outcome: successful, empty, unavailable, or disabled.",
        examples=["ok"],
    )
    provider_used: list[str] = Field(
        default_factory=list,
        description="Search providers attempted during this request.",
        examples=[["jina", "duckduckgo"]],
    )
    backend_used: list[str] = Field(
        default_factory=list,
        description="Concrete backend path used for this request when known.",
        examples=[["duckduckgo:auto"]],
    )
    fallback_used: bool = Field(
        False,
        description="Whether the request had to fall back from its preferred path.",
        examples=[True],
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code when the search path is unavailable.",
        examples=["search_provider_unavailable"],
    )


class WebSearchResponse(BaseModel):
    """
    Web search response
    """

    query: str = Field(..., description="Original search query")
    results: list[WebSearchResultItem] = Field(..., description="List of search results")
    total_results: Optional[int] = Field(None, description="Total number of results found")
    search_time: Optional[float] = Field(None, description="Search time in seconds")
    meta: Optional[WebSearchMeta] = Field(
        None,
        description="Lightweight execution diagnostics for status, provider selection, and fallback behavior.",
    )


class WebReadRequest(BaseModel):
    """
    Web content reading request
    """

    url_list: list[str] = Field(
        ...,
        description="List of URLs to read (for single URL, use array with one element)",
        examples=[["https://example.com/article"]],
    )
    timeout: Optional[int] = Field(30, description="Request timeout in seconds", examples=[30])
    locale: Optional[str] = Field("en-US", description="Browser locale", examples=["en-US"])
    max_concurrent: Optional[int] = Field(3, description="Maximum concurrent requests for multiple URLs", examples=[3])


class WebReadResultItem(BaseModel):
    """
    Individual web content reading result
    """

    url: str = Field(..., description="Requested URL")
    status: Literal["success", "error"] = Field(..., description="Processing status")
    title: Optional[str] = Field(None, description="Page title")
    content: Optional[str] = Field(None, description="Extracted content in Markdown format")
    extracted_at: Optional[datetime] = Field(None, description="Content extraction timestamp")
    word_count: Optional[int] = Field(None, description="Word count of content")
    token_count: Optional[int] = Field(None, description="Estimated token count")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")


class WebReadResponse(BaseModel):
    """
    Web content reading response
    """

    results: list[WebReadResultItem] = Field(..., description="List of reading results")
    total_urls: int = Field(..., description="Total number of URLs processed")
    successful: int = Field(..., description="Number of successful extractions")
    failed: int = Field(..., description="Number of failed extractions")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
