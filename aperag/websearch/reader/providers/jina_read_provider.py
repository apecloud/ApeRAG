"""
JINA Reader Provider

Web content reading implementation using JINA Reader API.
Provides LLM-friendly content extraction using JINA's r.jina.ai service.
"""

import asyncio
import logging
from datetime import datetime
from typing import List
from urllib.parse import quote, urlparse

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

        # Configure session headers according to Jina API documentation
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ApeCloud-WebReader/1.0",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def read(
        self,
        url: str,
        timeout: int = 30,
        locale: str = "en-US",
    ) -> WebReadResultItem:
        """
        Read content from a single URL using JINA Reader API.

        According to Jina docs: GET https://r.jina.ai/{url}
        Supports additional headers for content customization.

        Args:
            url: URL to read content from
            timeout: Request timeout in seconds
            locale: Browser locale (converted to Accept-Language header)

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

        # Validate API key is present before making requests
        if not self.api_key:
            return WebReadResultItem(
                url=url,
                status="error",
                error="JINA API key is required. Please configure your API key to use JINA Reader.",
                error_code="MISSING_API_KEY",
            )

        try:
            # Prepare headers with locale and Jina-specific options
            request_headers = self.headers.copy()
            if locale:
                # Convert locale to Accept-Language header format
                accept_language = locale.replace("_", "-")
                request_headers["Accept-Language"] = accept_language
                # Set X-Locale for browser rendering (important for region-specific content)
                request_headers["X-Locale"] = locale

            # Add Jina Reader specific headers based on documentation
            # X-Return-Format controls response format (json, text, markdown)
            request_headers["X-Return-Format"] = "json"
            # X-Retain-Images set to none to remove all images from response
            request_headers["X-Retain-Images"] = "none"
            # X-With-Links-Summary for link extraction
            request_headers["X-With-Links-Summary"] = "true"
            # Image processing is disabled (no X-With-Images-Summary or X-With-Generated-Alt)
            # X-Proxy for better access in China/Hong Kong regions
            request_headers["X-Proxy"] = "auto"
            # X-Proxy-URL for handling difficult-to-access content
            # request_headers["X-Proxy-URL"] = "true"  # Enable if needed

            # Make request to Jina Reader API using correct URL format
            # According to Jina docs: GET https://r.jina.ai/{url}
            # URL needs to be properly encoded
            encoded_url = quote(url, safe=":/?#[]@!$&'()*+,;=")
            reader_url = f"{self.base_url}{encoded_url}"

            logger.info(f"Jina reader request: {reader_url}")
            logger.debug(f"Request headers: {request_headers}")

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(reader_url, headers=request_headers) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"JINA reader API error {response.status}: {response_text}")
                        return WebReadResultItem(
                            url=url,
                            status="error",
                            error=f"JINA API returned status {response.status}: {response_text}",
                            error_code=f"API_ERROR_{response.status}",
                        )

                    logger.debug(f"Jina reader response type: {response.content_type}")

                    # Parse response as JSON (Jina API should return JSON format)
                    try:
                        data = await response.json()
                        return self._parse_json_result(url, data)
                    except Exception as e:
                        logger.error(f"Failed to parse Jina JSON response: {e}")
                        return WebReadResultItem(
                            url=url,
                            status="error",
                            error=f"Failed to parse JSON response: {str(e)}",
                            error_code="PARSE_ERROR",
                        )

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
        locale: str = "en-US",
        max_concurrent: int = 3,
    ) -> List[WebReadResultItem]:
        """
        Read content from multiple URLs concurrently using JINA Reader API.

        Args:
            urls: List of URLs to read content from
            timeout: Request timeout in seconds
            locale: Browser locale (converted to Accept-Language header)
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

    def _parse_json_result(self, url: str, data: dict) -> WebReadResultItem:
        """
        Parse JINA reader JSON response into WebReadResultItem object.

        Args:
            url: Original URL
            data: Raw response data from JINA API

        Returns:
            Parsed web read result item
        """
        try:
            # Handle different JSON response structures from Jina
            content_data = data
            if isinstance(data, dict):
                # Check for nested data structure
                content_data = data.get("data", data)

            # Extract main content with multiple fallback keys
            content = (
                content_data.get("content", "")
                or content_data.get("text", "")
                or content_data.get("markdown", "")
                or content_data.get("body", "")
                or ""
            )

            title = (
                content_data.get("title", "") or content_data.get("name", "") or content_data.get("heading", "") or ""
            )

            # Extract additional metadata if available
            description = content_data.get("description", "")
            author = content_data.get("author", "")
            published_date = content_data.get("published_date", "")

            # Extract links provided by X-With-Links-Summary (images are disabled)
            links = content_data.get("links", [])

            # Add metadata to content if available
            if description and description not in content:
                content = f"Description: {description}\n\n{content}"

            if author:
                content = f"Author: {author}\n{content}"

            if published_date:
                content = f"Published: {published_date}\n{content}"

            # Append links section to content if available
            if links:
                links_section = "\n\n## Links Found on Page\n"
                if isinstance(links, dict):
                    # Format: {"link_text": "url"}
                    for link_text, link_url in links.items():
                        links_section += f"- [{link_text}]({link_url})\n"
                elif isinstance(links, list):
                    # Format: [{"text": "...", "url": "..."}, ...] or ["url1", "url2", ...]
                    for i, link in enumerate(links, 1):
                        if isinstance(link, dict):
                            link_text = link.get("text", f"Link {i}")
                            link_url = link.get("url", "")
                            if link_url:
                                links_section += f"- [{link_text}]({link_url})\n"
                        elif isinstance(link, str):
                            links_section += f"- {link}\n"
                content += links_section

            # Calculate word and token counts
            word_count = len(content.split()) if content else 0
            # Improved token estimation: ~4 characters per token for English, ~2 for Chinese/Japanese
            avg_chars_per_token = 3.5  # More accurate estimation
            token_count = int(len(content) / avg_chars_per_token) if content else 0

            if not content:
                logger.warning(f"No content extracted from URL: {url}")
                return WebReadResultItem(
                    url=url,
                    status="error",
                    error="No content could be extracted from the page",
                    error_code="NO_CONTENT",
                )

            logger.info(f"Jina reader JSON parsing successful: {word_count} words, {token_count} tokens")

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
            logger.error(f"Error parsing JINA reader JSON result for {url}: {e}")
            return WebReadResultItem(
                url=url, status="error", error=f"Failed to parse JSON response: {str(e)}", error_code="PARSE_ERROR"
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
                title_part = title_part.replace("-", " ").replace("_", " ").replace(".html", "")
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
