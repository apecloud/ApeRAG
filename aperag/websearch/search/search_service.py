"""
Search Service

Main service class for web search functionality with provider abstraction.
"""

import logging
from typing import Dict, List

from aperag.schema.view_models import WebSearchRequest, WebSearchResponse, WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.search.providers.duckduckgo_search_provider import DuckDuckGoProvider, SearchProviderError
from aperag.websearch.search.providers.jina_search_provider import JinaSearchProvider

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
        self.provider_name = provider_name or self._get_default_provider()
        self.provider_config = provider_config or {}
        self.provider = self._create_provider()

    def _get_default_provider(self) -> str:
        """
        Get default search provider.

        Returns:
            Default provider name
        """
        return "duckduckgo"

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

        Raises:
            SearchProviderError: If search fails
        """
        try:
            # Validate request
            if not request.query or not request.query.strip():
                raise SearchProviderError("Search query cannot be empty")

            # Validate search engine and determine effective engine to use
            effective_search_engine = request.search_engine
            if not self.provider.validate_search_engine(request.search_engine):
                logger.warning(
                    f"Unsupported search engine '{request.search_engine}' for provider '{self.provider_name}', "
                    f"using default supported engine"
                )
                # Use first supported engine as fallback
                effective_search_engine = self.provider.get_supported_engines()[0]

            # Perform search
            start_time = self._get_current_time()

            results = await self.provider.search(
                query=request.query,
                max_results=request.max_results,
                search_engine=effective_search_engine,
                timeout=request.timeout,
                locale=request.locale,
            )

            search_time = self._get_current_time() - start_time

            # Create response
            return WebSearchResponse(
                query=request.query,
                results=results,
                search_engine=effective_search_engine,
                total_results=len(results),  # For now, just return actual count
                search_time=search_time,
            )

        except SearchProviderError:
            # Re-raise provider errors
            raise
        except Exception as e:
            logger.error(f"Search service failed: {e}")
            raise SearchProviderError(f"Search service error: {str(e)}")

    async def search_simple(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "google",
        timeout: int = 30,
        locale: str = "zh-CN",
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

        Raises:
            SearchProviderError: If search fails
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
    def create_with_provider(cls, provider_name: str, **config) -> "SearchService":
        """
        Create search service with specific provider.

        Args:
            provider_name: Name of the search provider
            **config: Provider-specific configuration

        Returns:
            SearchService instance
        """
        return cls(provider_name=provider_name, provider_config=config)

    async def close(self):
        """
        Close provider and cleanup resources.
        """
        if hasattr(self.provider, "close"):
            await self.provider.close()

    async def cleanup(self):
        """
        Cleanup resources (alias for close).
        """
        await self.close()

    async def __aenter__(self):
        """
        Async context manager entry.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        """
        await self.close()
