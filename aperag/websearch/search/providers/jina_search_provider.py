"""
JINA Search Provider

Web search provider using JINA's s.jina.ai API.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


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
        self.supported_engines = ["jina"]  # Jina only supports its own search

        # Configure session headers according to Jina API documentation
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ApeRAG-WebSearch/1.0",
            "X-Respond-With": "no-content",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using Jina Search API.

        According to Jina docs: GET https://s.jina.ai/{query}
        Supports additional headers for customization.

        Args:
            query: Search query (can be empty for site-specific browsing)
            max_results: Maximum number of results to return (applied via local filtering)
            timeout: Request timeout in seconds
            locale: Browser locale (converted to Accept-Language header)
            source: Domain or URL for site-specific search. When provided, search will be limited to this domain.

        Returns:
            List of search result items
        """
        # Validate parameters
        has_query = query and query.strip()
        has_source = source and source.strip()

        # Either query or source must be provided
        if not has_query and not has_source:
            raise ValueError("Either query or source must be provided")

        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_results > 100:
            raise ValueError("max_results cannot exceed 100")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        # Prepare search query and domain filtering
        final_query = query or ""
        target_domain = None

        if source:
            target_domain = URLValidator.extract_domain_from_source(source)
            if target_domain and has_query:
                # Add site restriction to query
                final_query = f"site:{target_domain} {query}"
            elif target_domain and not has_query:
                # Site browsing without specific query
                final_query = f"site:{target_domain}"
            elif not target_domain and not has_query:
                raise ValueError("Invalid source domain and no query provided")

        if not final_query.strip():
            raise ValueError("Search query cannot be empty")

        try:
            # Prepare headers with locale and additional options based on Jina docs
            request_headers = self.headers.copy()
            if locale:
                # Convert locale to Accept-Language header format
                accept_language = locale.replace("_", "-")
                request_headers["Accept-Language"] = accept_language

            # Add Jina-specific headers for better control
            # X-Return-Format controls response format
            request_headers["X-Return-Format"] = "json"
            # X-Target-Selector for better content extraction (if supported)
            if target_domain:
                request_headers["X-Target-Domain"] = target_domain

            # Make request to Jina Search API using correct URL format
            # According to Jina docs: GET https://s.jina.ai/?q={search_query}
            # URL encode the query using + for spaces (standard for query parameters)
            from urllib.parse import quote_plus

            encoded_query = quote_plus(final_query)
            search_url = f"{self.base_url}?q={encoded_query}"

            logger.info(f"Jina search request: {search_url}")
            logger.debug(f"Request headers: {request_headers}")

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(search_url, headers=request_headers) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"Jina Search API returned status {response.status}: {response_text}")
                        return []

                    logger.debug(f"Jina search response type: {response.content_type}")

                    # Parse response as JSON (Jina API should return JSON format)
                    try:
                        response_data = await response.json()
                        return self._parse_jina_json_response(response_data, target_domain, max_results)
                    except Exception as e:
                        logger.error(f"Failed to parse Jina JSON response: {e}")
                        return []

        except asyncio.TimeoutError:
            logger.error(f"Jina search timed out after {timeout} seconds")
            return []
        except Exception as e:
            logger.error(f"Error in Jina search: {e}")
            return []

    def _parse_jina_json_response(
        self, response_data: Dict[str, Any], target_domain: Optional[str] = None, max_results: int = 5
    ) -> List[WebSearchResultItem]:
        """Parse Jina API JSON response into standardized result items."""
        results = []

        # Handle different response formats that Jina might return
        items = []
        if isinstance(response_data, dict):
            items = response_data.get("data", []) or response_data.get("results", []) or response_data.get("items", [])
        elif isinstance(response_data, list):
            items = response_data

        logger.info(f"Parsing {len(items)} items from Jina JSON response")

        for i, item in enumerate(items):
            if len(results) >= max_results:
                break

            try:
                # Handle different item formats
                if isinstance(item, str):
                    # Simple string format - try to extract URL
                    continue
                elif isinstance(item, dict):
                    url = item.get("url", "") or item.get("href", "") or item.get("link", "")
                    if not url:
                        continue

                    # Apply domain filtering if specified
                    if target_domain:
                        result_domain = URLValidator.extract_domain(url)
                        if not result_domain or result_domain.lower() != target_domain.lower():
                            continue

                    # Extract title and description with fallbacks
                    title = (
                        item.get("title", "") or item.get("name", "") or item.get("heading", "") or "No Title"
                    ).strip()

                    snippet = (
                        item.get("description", "")
                        or item.get("snippet", "")
                        or item.get("content", "")
                        or item.get("summary", "")
                        or "No description available"
                    ).strip()

                    result = WebSearchResultItem(
                        url=url,
                        title=title,
                        snippet=snippet,
                        rank=len(results) + 1,  # Use filtered rank
                        domain=URLValidator.extract_domain(url) or "",
                        timestamp=datetime.now(),
                    )
                    results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse Jina result item {i}: {e}")
                continue

        logger.info(
            f"Jina search JSON parsing completed: {len(results)} results"
            + (f" from domain {target_domain}" if target_domain else "")
        )
        return results

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return self.supported_engines.copy()
