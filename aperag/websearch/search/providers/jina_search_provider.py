"""
JINA Search Provider

Web search implementation using JINA Search API.
Provides LLM-friendly search results using JINA's s.jina.ai service.
"""

import logging
from datetime import datetime
from typing import List

import aiohttp

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


class SearchProviderError(Exception):
    """Exception raised by search providers."""

    pass


class JinaSearchProvider(BaseSearchProvider):
    """
    JINA search provider implementation.

    Uses JINA's s.jina.ai API to perform web searches with LLM-friendly results.
    Get your JINA AI API key for free: https://jina.ai/?sui=apikey
    """

    def __init__(self, config: dict = None):
        """
        Initialize JINA search provider.

        Args:
            config: Provider configuration containing api_key and other settings
        """
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None

        self.base_url = "https://s.jina.ai/"
        self.supported_engines = ["jina", "google", "bing"]  # JINA supports multiple search engines

        # Configure session headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "google",
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using JINA Search API.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_engine: Search engine to use (google, bing, etc.)
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            List of search result items

        Raises:
            SearchProviderError: If search fails
        """
        if not query or not query.strip():
            raise SearchProviderError("Query cannot be empty")

        if not self.api_key:
            raise SearchProviderError("JINA API key is required. Pass api_key in provider_config.")

        if not self.validate_search_engine(search_engine):
            raise SearchProviderError(
                f"Unsupported search engine: {search_engine}. Supported engines: {self.get_supported_engines()}"
            )

        try:
            # Prepare request payload
            payload = {
                "query": query,
                "count": max_results,
                "search_engine": search_engine if search_engine != "jina" else "google",
                "include_citations": True,
                "include_images": False,
                "include_image_descriptions": False,
            }

            # Add locale-specific parameters
            if locale.startswith("zh"):
                payload["safe_search"] = "moderate"
                payload["country"] = "CN"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(self.base_url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"JINA search API error {response.status}: {error_text}")
                        raise SearchProviderError(f"JINA API returned status {response.status}: {error_text}")

                    data = await response.json()
                    return self._parse_search_results(data, query)

        except aiohttp.ClientError as e:
            logger.error(f"JINA search request failed: {e}")
            raise SearchProviderError(f"Network request failed: {str(e)}")
        except Exception as e:
            logger.error(f"JINA search failed: {e}")
            raise SearchProviderError(f"Search failed: {str(e)}")

    def _parse_search_results(self, data: dict, query: str) -> List[WebSearchResultItem]:
        """
        Parse JINA search response into WebSearchResultItem objects.

        Args:
            data: Raw response data from JINA API
            query: Original search query

        Returns:
            List of parsed search result items
        """
        results = []

        # JINA s.jina.ai returns results in 'data' field with 'content' containing structured info
        content = data.get("data", {}).get("content", "")
        if not content:
            logger.warning("No content found in JINA search response")
            return results

        # Extract URLs from citations if available
        citations = data.get("data", {}).get("citations", [])

        # Parse content to extract search results
        # JINA typically returns markdown-formatted content with citations
        try:
            # Try to extract structured information from content
            # This is a simplified parser - JINA's format may vary
            lines = content.split("\n")
            rank = 1

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for URL patterns in citations
                for citation in citations:
                    if isinstance(citation, dict) and "url" in citation:
                        url = citation["url"]
                        if not URLValidator.is_valid_url(url):
                            continue

                        # Extract title and snippet from citation
                        title = citation.get("title", "")
                        snippet = citation.get("snippet", "")

                        if not title:
                            # Try to extract title from content
                            title = f"Search result {rank}"

                        if not snippet:
                            # Use part of the content as snippet
                            snippet = content[:200] + "..." if len(content) > 200 else content

                        results.append(
                            WebSearchResultItem(
                                rank=rank,
                                title=title,
                                url=url,
                                snippet=snippet,
                                domain=URLValidator.extract_domain(url),
                                timestamp=datetime.now(),
                            )
                        )
                        rank += 1

                        if len(results) >= 10:  # Reasonable limit
                            break

            # If no citations found, create a generic result
            if not results and content:
                results.append(
                    WebSearchResultItem(
                        rank=1,
                        title=f"Search results for: {query}",
                        url="https://jina.ai/",
                        snippet=content[:300] + "..." if len(content) > 300 else content,
                        domain="jina.ai",
                        timestamp=datetime.now(),
                    )
                )

        except Exception as e:
            logger.error(f"Error parsing JINA search results: {e}")
            # Fallback: create a single result with the content
            if content:
                results.append(
                    WebSearchResultItem(
                        rank=1,
                        title=f"Search results for: {query}",
                        url="https://jina.ai/",
                        snippet=content[:300] + "..." if len(content) > 300 else content,
                        domain="jina.ai",
                        timestamp=datetime.now(),
                    )
                )

        return results if results else []

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return self.supported_engines.copy()

    async def close(self):
        """
        Close and cleanup resources.
        """
        # JINA provider doesn't maintain persistent connections
        # No resources to close
        pass
