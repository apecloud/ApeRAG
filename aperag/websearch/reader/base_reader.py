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
        css_selector: str = None,
        wait_for_selector: str = None,
        exclude_selector: str = None,
        bypass_cache: bool = False,
        locale: str = "zh-CN",
    ) -> WebReadResultItem:
        """
        Read content from a single URL.

        Args:
            url: URL to read content from
            timeout: Request timeout in seconds
            css_selector: CSS selector for content extraction
            wait_for_selector: CSS selector to wait for (SPA pages)
            exclude_selector: CSS selector to exclude (ads, etc.)
            bypass_cache: Bypass cache for fresh content
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
        css_selector: str = None,
        wait_for_selector: str = None,
        exclude_selector: str = None,
        bypass_cache: bool = False,
        locale: str = "zh-CN",
        max_concurrent: int = 3,
    ) -> List[WebReadResultItem]:
        """
        Read content from multiple URLs concurrently.

        Args:
            urls: List of URLs to read content from
            timeout: Request timeout in seconds
            css_selector: CSS selector for content extraction
            wait_for_selector: CSS selector to wait for (SPA pages)
            exclude_selector: CSS selector to exclude (ads, etc.)
            bypass_cache: Bypass cache for fresh content
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
