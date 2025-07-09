"""
Base Search Provider

Abstract base class for web search providers.
"""

from abc import ABC, abstractmethod
from typing import List

from aperag.schema.view_models import WebSearchResultItem


class BaseSearchProvider(ABC):
    """
    Abstract base class for web search providers.

    All search providers must implement the search method and get_supported_engines method.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the search provider.

        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "google",
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> List[WebSearchResultItem]:
        """
        Perform web search.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_engine: Search engine to use
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            List of search result items

        Raises:
            SearchProviderError: If search fails
        """
        pass

    @abstractmethod
    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        pass

    def validate_search_engine(self, search_engine: str) -> bool:
        """
        Validate if search engine is supported.

        Args:
            search_engine: Search engine name to validate

        Returns:
            True if supported, False otherwise
        """
        return search_engine in self.get_supported_engines()

    async def close(self):
        """
        Close and cleanup resources.

        This is a base implementation that does nothing.
        Subclasses should override if they need to cleanup resources.
        """
        pass
