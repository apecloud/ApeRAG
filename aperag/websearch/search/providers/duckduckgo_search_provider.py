"""
DuckDuckGo Search Provider

Web search implementation using DuckDuckGo search engine.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from duckduckgo_search import DDGS

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


class SearchProviderError(Exception):
    """Exception raised by search providers."""

    pass


class DuckDuckGoProvider(BaseSearchProvider):
    """
    DuckDuckGo search provider implementation.

    Uses the duckduckgo-search library to perform web searches.
    """

    def __init__(self, config: dict = None):
        """
        Initialize DuckDuckGo provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.supported_engines = ["duckduckgo", "ddg"]

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "duckduckgo",
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_engine: Search engine to use (must be supported)
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            List of search result items

        Raises:
            SearchProviderError: If search fails
        """
        if not query or not query.strip():
            raise SearchProviderError("Query cannot be empty")

        if not self.validate_search_engine(search_engine):
            raise SearchProviderError(
                f"Unsupported search engine: {search_engine}. Supported engines: {self.get_supported_engines()}"
            )

        try:
            # Run the synchronous search in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._search_sync, query, max_results, timeout, locale)
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise SearchProviderError(f"Search failed: {str(e)}")

    def _search_sync(self, query: str, max_results: int, timeout: int, locale: str) -> List[WebSearchResultItem]:
        """
        Synchronous search implementation.

        Args:
            query: Search query
            max_results: Maximum number of results
            timeout: Request timeout
            locale: Browser locale

        Returns:
            List of search result items
        """
        try:
            # Configure DuckDuckGo search
            region = "cn-zh" if locale.startswith("zh") else "wt-wt"

            # Perform search
            with DDGS() as ddgs:
                search_results = list(
                    ddgs.text(
                        query,
                        region=region,
                        safesearch="moderate",
                        timelimit=None,
                        max_results=max_results,
                    )
                )

            # Convert results to our format
            results = []
            for i, result in enumerate(search_results):
                # Validate URL
                url = result.get("href", "")
                if not URLValidator.is_valid_url(url):
                    continue

                results.append(
                    WebSearchResultItem(
                        rank=i + 1,
                        title=result.get("title", ""),
                        url=url,
                        snippet=result.get("body", ""),
                        domain=URLValidator.extract_domain(url),
                        timestamp=datetime.now(),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo sync search failed: {e}")
            raise SearchProviderError(f"Search execution failed: {str(e)}")

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return self.supported_engines.copy()

    def get_provider_info(self) -> dict:
        """
        Get provider information.

        Returns:
            Provider information dictionary
        """
        return {
            "name": "DuckDuckGo",
            "description": "Privacy-focused search engine with no tracking",
            "supported_engines": self.get_supported_engines(),
            "free": True,
            "requires_api_key": False,
            "rate_limit": "None (but may be throttled by DuckDuckGo)",
        }

    async def close(self):
        """
        Close and cleanup resources.
        """
        # DuckDuckGo provider doesn't maintain persistent connections
        # No resources to close
        pass
