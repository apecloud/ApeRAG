"""
Reader Service

Main service class for web content reading functionality with provider abstraction.
"""

import logging
from typing import Dict, List

from aperag.schema.view_models import WebReadRequest, WebReadResponse, WebReadResultItem
from aperag.websearch.reader.base_reader import BaseReaderProvider
from aperag.websearch.reader.providers.jina_read_provider import JinaReaderProvider
from aperag.websearch.reader.providers.trafilatura_read_provider import ReaderProviderError, TrafilaturaProvider

logger = logging.getLogger(__name__)


class ReaderService:
    """
    Web content reading service with provider abstraction.

    Supports multiple content reading providers and provides a unified interface
    for web content extraction functionality.
    """

    def __init__(
        self,
        provider_name: str = None,
        provider_config: Dict = None,
    ):
        """
        Initialize reader service.

        Args:
            provider_name: Name of the reader provider to use
            provider_config: Provider-specific configuration
        """
        self.provider_name = provider_name or self._get_default_provider()
        self.provider_config = provider_config or {}
        self.provider = self._create_provider()

    def _get_default_provider(self) -> str:
        """
        Get default reader provider.

        Returns:
            Default provider name
        """
        return "trafilatura"

    def _create_provider(self) -> BaseReaderProvider:
        """
        Create reader provider instance.

        Returns:
            Reader provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_registry = {
            "trafilatura": TrafilaturaProvider,
            "jina": JinaReaderProvider,
            "jina_reader": JinaReaderProvider,
        }

        provider_class = provider_registry.get(self.provider_name.lower())
        if not provider_class:
            raise ValueError(
                f"Unsupported reader provider: {self.provider_name}. "
                f"Supported providers: {list(provider_registry.keys())}"
            )

        return provider_class(self.provider_config)

    async def read(self, request: WebReadRequest) -> WebReadResponse:
        """
        Read content from URLs.

        Args:
            request: Read request

        Returns:
            Read response

        Raises:
            ReaderProviderError: If reading fails
        """
        try:
            # Normalize URLs input from the new url_list attribute
            if hasattr(request, "url_list") and request.url_list:
                urls = request.url_list
            elif hasattr(request, "urls"):
                # Backward compatibility for old urls attribute
                if isinstance(request.urls, str):
                    urls = [request.urls]
                else:
                    urls = request.urls
            else:
                raise ReaderProviderError("No URLs provided in request")

            if not urls:
                raise ReaderProviderError("URLs list cannot be empty")

            # Track timing
            start_time = self._get_current_time()

            # Read content based on number of URLs
            if len(urls) == 1:
                # Single URL - use read method
                result = await self.provider.read(
                    url=urls[0],
                    timeout=request.timeout,
                    locale=request.locale,
                )
                results = [result]
            else:
                # Multiple URLs - use read_batch method
                results = await self.provider.read_batch(
                    urls=urls,
                    timeout=request.timeout,
                    locale=request.locale,
                    max_concurrent=request.max_concurrent,
                )

            processing_time = self._get_current_time() - start_time

            # Calculate statistics
            successful = sum(1 for r in results if r.status == "success")
            failed = len(results) - successful

            # Create response
            return WebReadResponse(
                results=results,
                total_urls=len(urls),
                successful=successful,
                failed=failed,
                processing_time=processing_time,
            )

        except ReaderProviderError:
            # Re-raise provider errors
            raise
        except Exception as e:
            logger.error(f"Reader service failed: {e}")
            raise ReaderProviderError(f"Reader service error: {str(e)}")

    async def read_simple(
        self,
        url: str,
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> WebReadResultItem:
        """
        Simplified single URL reading interface.

        Args:
            url: URL to read content from
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            Web read result item

        Raises:
            ReaderProviderError: If reading fails
        """
        request = WebReadRequest(
            url_list=[url],  # Use url_list with single URL
            timeout=timeout,
            locale=locale,
        )

        response = await self.read(request)
        return response.results[0]

    async def read_batch_simple(
        self,
        urls: List[str],
        timeout: int = 30,
        locale: str = "zh-CN",
        max_concurrent: int = 3,
    ) -> List[WebReadResultItem]:
        """
        Simplified batch reading interface that returns only results.

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
        request = WebReadRequest(
            url_list=urls,  # Use url_list directly
            timeout=timeout,
            locale=locale,
            max_concurrent=max_concurrent,
        )

        response = await self.read(request)
        return response.results

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

    @staticmethod
    def _get_current_time() -> float:
        """Get current time in seconds."""
        import time

        return time.time()

    @classmethod
    def create_default(cls) -> "ReaderService":
        """
        Create reader service with default configuration.

        Returns:
            ReaderService instance with default settings
        """
        return cls()

    @classmethod
    def create_with_provider(cls, provider_name: str, **config) -> "ReaderService":
        """
        Create reader service with specific provider.

        Args:
            provider_name: Name of the reader provider
            **config: Provider-specific configuration

        Returns:
            ReaderService instance
        """
        return cls(provider_name=provider_name, provider_config=config)
