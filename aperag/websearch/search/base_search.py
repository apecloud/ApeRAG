"""
Base Search Provider

Abstract base class for web search providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

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
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
            locale: Browser locale
            source: Domain or URL for site-specific search

        Returns:
            List of search result items
        """
        raise NotImplementedError("Subclasses must implement search method")

    @abstractmethod
    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        raise NotImplementedError("Subclasses must implement get_supported_engines method")

    async def close(self):
        """
        Close and cleanup resources.

        This is a base implementation that does nothing.
        Subclasses should override if they need to cleanup resources.
        """
        pass
