"""
Base Reader Provider

Abstract base class for web content reading providers.
"""

from abc import ABC, abstractmethod
from typing import List

from aperag.schema.view_models import WebReadResultItem


class BaseReaderProvider(ABC):
    """
    Abstract base class for web content reading providers.

    All reader providers must implement the read and read_batch methods.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the reader provider.

        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    async def read(
        self,
        url: str,
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> WebReadResultItem:
        """
        Read content from a single URL.

        Args:
            url: URL to read content from
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            Web read result item

        Raises:
            ReaderProviderError: If reading fails
        """
        pass

    @abstractmethod
    async def read_batch(
        self,
        urls: List[str],
        timeout: int = 30,
        locale: str = "zh-CN",
        max_concurrent: int = 3,
    ) -> List[WebReadResultItem]:
        """
        Read content from multiple URLs concurrently.

        Args:
            urls: List of URLs to read content from
            timeout: Request timeout in seconds
            locale: Browser locale
            max_concurrent: Maximum concurrent requests

        Returns:
            List of web read result items

        Raises:
            ReaderProviderError: If reading fails
        """
        pass

    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is valid and supported.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic URL validation - providers can override for more specific validation
        return url.startswith(("http://", "https://"))

    async def close(self):
        """
        Close and cleanup resources.

        This is a base implementation that does nothing.
        Subclasses should override if they need to cleanup resources.
        """
        pass
