"""
Search Service

Main service class for web search functionality with provider abstraction.
"""

import logging
from typing import Dict, List

from aperag.schema.view_models import WebSearchRequest, WebSearchResponse, WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.search.providers.duckduckgo_search_provider import DuckDuckGoProvider
from aperag.websearch.search.providers.jina_search_provider import JinaSearchProvider
from aperag.websearch.search.providers.llm_txt_search_provider import LLMTxtSearchProvider

logger = logging.getLogger(__name__)


class SearchService:
    """
    Web search service with provider abstraction.

    Supports multiple search providers and provides a unified interface
    for web search functionality.
    """

    def __init__(
        self,
        provider_name: str = None,
        provider_config: Dict = None,
    ):
        """
        Initialize search service.

        Args:
            provider_name: Name of the search provider to use
            provider_config: Provider-specific configuration
        """
        self.provider_name = provider_name or "duckduckgo"
        self.provider_config = provider_config or {}
        self.provider = self._create_provider()

    def _create_provider(self) -> BaseSearchProvider:
        """
        Create search provider instance.

        Returns:
            Search provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_registry = {
            "duckduckgo": DuckDuckGoProvider,
            "ddg": DuckDuckGoProvider,
            "jina": JinaSearchProvider,
            "jina_search": JinaSearchProvider,
            "llm_txt": LLMTxtSearchProvider,
        }

        provider_class = provider_registry.get(self.provider_name.lower())
        if not provider_class:
            raise ValueError(
                f"Unsupported search provider: {self.provider_name}. "
                f"Supported providers: {list(provider_registry.keys())}"
            )

        return provider_class(self.provider_config)

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """
        Perform web search.

        Args:
            request: Search request

        Returns:
            Search response
        """
        # Allow empty query if source is provided (site-specific search)
        # Only require query when neither query nor source is provided
        has_query = request.query and request.query.strip()
        has_source = request.source and request.source.strip()

        if not has_query and not has_source:
            raise ValueError("Either 'query' or 'source' parameter is required for search")

        if request.max_results <= 0:
            raise ValueError("max_results must be positive")
        if request.max_results > 100:
            raise ValueError("max_results cannot exceed 100")
        if request.timeout <= 0:
            raise ValueError("timeout must be positive")
        if request.timeout > 300:
            raise ValueError("timeout cannot exceed 300 seconds")

        start_time = self._get_current_time()

        # Call the provider's search method
        results = await self.provider.search(
            query=request.query,
            max_results=request.max_results,
            search_engine=request.search_engine,
            timeout=request.timeout,
            locale=request.locale,
            source=request.source,
        )

        search_time = self._get_current_time() - start_time

        # Create response
        return WebSearchResponse(
            query=request.query or "",
            results=results,
            search_engine=request.search_engine,
            total_results=len(results),
            search_time=search_time,
        )

    async def search_simple(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "google",
        timeout: int = 30,
        locale: str = "en-US",
    ) -> List[WebSearchResultItem]:
        """
        Simplified search interface that returns only results.

        Args:
            query: Search query
            max_results: Maximum number of results
            search_engine: Search engine to use
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            List of search result items
        """
        request = WebSearchRequest(
            query=query,
            max_results=max_results,
            search_engine=search_engine,
            timeout=timeout,
            locale=locale,
        )

        response = await self.search(request)
        return response.results

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines for current provider.

        Returns:
            List of supported search engine names
        """
        return self.provider.get_supported_engines()

    @staticmethod
    def _get_current_time() -> float:
        """Get current time in seconds."""
        import time

        return time.time()

    @classmethod
    def create_default(cls) -> "SearchService":
        """
        Create search service with default configuration.

        Returns:
            SearchService instance with default settings
        """
        return cls()

    @classmethod
    def create_with_provider(cls, provider_name: str, provider_config: Dict = None) -> "SearchService":
        """
        Create search service with specific provider.

        Args:
            provider_name: Name of the search provider
            provider_config: Provider-specific configuration

        Returns:
            SearchService instance with specified provider
        """
        return cls(provider_name=provider_name, provider_config=provider_config)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close and cleanup resources."""
        if hasattr(self.provider, "close"):
            await self.provider.close()
