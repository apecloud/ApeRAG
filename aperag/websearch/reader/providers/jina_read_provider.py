"""
JINA Reader Provider

Web content reading implementation using JINA Reader API.
Provides LLM-friendly content extraction using JINA's r.jina.ai service.
"""

import asyncio
import logging
from datetime import datetime
from typing import List
from urllib.parse import urlparse

import aiohttp

from aperag.schema.view_models import WebReadResultItem
from aperag.websearch.reader.base_reader import BaseReaderProvider

logger = logging.getLogger(__name__)


class ReaderProviderError(Exception):
    """Exception raised by reader providers."""

    pass


class JinaReaderProvider(BaseReaderProvider):
    """
    JINA reader provider implementation.

    Uses JINA's r.jina.ai API to extract LLM-friendly content from web pages.
    Get your JINA AI API key for free: https://jina.ai/?sui=apikey
    """

    def __init__(self, config: dict = None):
        """
        Initialize JINA reader provider.

        Args:
            config: Provider configuration containing api_key and other settings
        """
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None

        self.base_url = "https://r.jina.ai/"

        # Configure session headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def read(
        self,
        url: str,
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> WebReadResultItem:
        """
        Read content from a single URL using JINA Reader API.

        Args:
            url: URL to read content from
            timeout: Request timeout in seconds
            locale: Browser locale

        Returns:
            Web read result item

        Raises:
            ReaderProviderError: If reading fails
        """
        if not url or not url.strip():
            return WebReadResultItem(url=url, status="error", error="URL cannot be empty", error_code="INVALID_URL")

        if not self.validate_url(url):
            return WebReadResultItem(
                url=url, status="error", error="Invalid URL format", error_code="INVALID_URL_FORMAT"
            )

        if not self.api_key:
            return WebReadResultItem(
                url=url,
                status="error",
                error="JINA API key is required. Pass api_key in provider_config.",
                error_code="MISSING_API_KEY",
            )

        try:
            # Prepare request payload
            payload = {
                "url": url,
                "return_markdown": True,
                "return_links": True,
                "return_images": False,
                "timeout": timeout,
            }

            # Add locale-specific headers
            request_headers = self.headers.copy()
            if locale:
                request_headers["Accept-Language"] = locale

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(self.base_url, json=payload, headers=request_headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"JINA reader API error {response.status}: {error_text}")
                        return WebReadResultItem(
                            url=url,
                            status="error",
                            error=f"JINA API returned status {response.status}: {error_text}",
                            error_code=f"API_ERROR_{response.status}",
                        )

                    data = await response.json()
                    return self._parse_read_result(url, data)

        except aiohttp.ClientError as e:
            logger.error(f"JINA reader request failed for {url}: {e}")
            return WebReadResultItem(
                url=url, status="error", error=f"Network request failed: {str(e)}", error_code="NETWORK_ERROR"
            )
        except Exception as e:
            logger.error(f"JINA reader failed for {url}: {e}")
            return WebReadResultItem(
                url=url, status="error", error=f"Reader failed: {str(e)}", error_code="READER_ERROR"
            )

    async def read_batch(
        self,
        urls: List[str],
        timeout: int = 30,
        locale: str = "zh-CN",
        max_concurrent: int = 3,
    ) -> List[WebReadResultItem]:
        """
        Read content from multiple URLs concurrently using JINA Reader API.

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
        if not urls:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_single(url: str) -> WebReadResultItem:
            async with semaphore:
                return await self.read(
                    url=url,
                    timeout=timeout,
                    locale=locale,
                )

        # Execute all requests concurrently
        tasks = [read_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions and convert to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception reading URL {urls[i]}: {result}")
                processed_results.append(
                    WebReadResultItem(
                        url=urls[i], status="error", error=f"Exception occurred: {str(result)}", error_code="EXCEPTION"
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _parse_read_result(self, url: str, data: dict) -> WebReadResultItem:
        """
        Parse JINA reader response into WebReadResultItem object.

        Args:
            url: Original URL
            data: Raw response data from JINA API

        Returns:
            Parsed web read result item
        """
        try:
            # JINA r.jina.ai returns data with structured content
            content_data = data.get("data", {})

            # Extract main content
            content = content_data.get("content", "")
            title = content_data.get("title", "")

            # Calculate word and token counts
            word_count = len(content.split()) if content else 0
            # Rough token estimation: ~4 characters per token for English, ~2 for Chinese
            token_count = len(content) // 3 if content else 0

            # Extract additional metadata (not used but available for future extensions)
            # url_info = content_data.get("url", url)
            # description = content_data.get("description", "")

            if not content:
                logger.warning(f"No content extracted from URL: {url}")
                return WebReadResultItem(
                    url=url,
                    status="error",
                    error="No content could be extracted from the page",
                    error_code="NO_CONTENT",
                )

            return WebReadResultItem(
                url=url,
                status="success",
                title=title or self._extract_title_from_url(url),
                content=content,
                extracted_at=datetime.now(),
                word_count=word_count,
                token_count=token_count,
            )

        except Exception as e:
            logger.error(f"Error parsing JINA reader result for {url}: {e}")
            return WebReadResultItem(
                url=url, status="error", error=f"Failed to parse response: {str(e)}", error_code="PARSE_ERROR"
            )

    def _extract_title_from_url(self, url: str) -> str:
        """
        Extract a reasonable title from URL if no title is provided.

        Args:
            url: URL to extract title from

        Returns:
            Extracted title string
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path.strip("/")

            if path:
                # Use the last part of the path as title
                title_part = path.split("/")[-1]
                # Clean up the title
                title_part = title_part.replace("-", " ").replace("_", " ")
                return f"{title_part} - {domain}".title()
            else:
                return domain.title()
        except Exception:
            return "Web Page"

    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is valid and supported.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        if not super().validate_url(url):
            return False

        try:
            parsed = urlparse(url)
            # Additional validation for JINA reader
            if not parsed.netloc:
                return False

            # Block some URLs that are known to cause issues
            blocked_domains = ["localhost", "127.0.0.1", "0.0.0.0"]
            if any(domain in parsed.netloc.lower() for domain in blocked_domains):
                return False

            return True
        except Exception:
            return False

    async def close(self):
        """
        Close and cleanup resources.
        """
        # JINA provider doesn't maintain persistent connections
        # No resources to close
        pass
