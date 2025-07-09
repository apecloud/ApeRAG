"""
Trafilatura Reader Provider

Lightweight web content reading implementation using Trafilatura.
No browser dependencies - perfect for lightweight deployments.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

import aiohttp

try:
    import markdownify
    import trafilatura

    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

from aperag.schema.view_models import WebReadResultItem
from aperag.websearch.reader.base_reader import BaseReaderProvider
from aperag.websearch.utils.content_processor import ContentProcessor
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


class ReaderProviderError(Exception):
    """Exception raised by reader providers."""

    pass


class TrafilaturaProvider(BaseReaderProvider):
    """
    Trafilatura reader provider implementation.

    Uses Trafilatura for content extraction - no browser required.
    Lightweight, fast, but limited to static content.
    """

    def __init__(self, config: dict = None):
        """
        Initialize Trafilatura provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        if not HAS_TRAFILATURA:
            raise ReaderProviderError("Trafilatura is not installed. Run: uv add trafilatura markdownify")

    async def read(
        self,
        url: str,
        timeout: int = 30,
        locale: str = "zh-CN",
    ) -> WebReadResultItem:
        """
        Read content from a single URL using Trafilatura.

        Args:
            url: URL to read content from
            timeout: Request timeout in seconds
            locale: Browser locale (used for User-Agent)

        Returns:
            Web read result item

        Raises:
            ReaderProviderError: If reading fails
        """
        if not url or not url.strip():
            raise ReaderProviderError("URL cannot be empty")

        # Normalize and validate URL
        url = URLValidator.normalize_url(url.strip())
        if not URLValidator.is_valid_url(url):
            return WebReadResultItem(
                url=url,
                status="error",
                error="Invalid URL format",
                error_code="INVALID_URL",
            )

        try:
            # Fetch HTML content
            html_content = await self._fetch_html(url, timeout, locale)
            if not html_content:
                return WebReadResultItem(
                    url=url,
                    status="error",
                    error="Failed to fetch HTML content",
                    error_code="FETCH_ERROR",
                )

            # Extract main content using Trafilatura
            extracted_text = trafilatura.extract(
                html_content,
                output_format="xml",  # Get structured output
                include_comments=False,
                include_tables=True,
                include_links=True,
                deduplicate=True,
                favor_precision=True,  # Prefer quality over quantity
                no_fallback=False,  # Use fallback extraction if needed
            )

            if not extracted_text:
                # Fallback to simple text extraction
                extracted_text = trafilatura.extract(
                    html_content,
                    output_format="txt",
                    no_fallback=True,
                    favor_recall=True,
                )

                if not extracted_text:
                    return WebReadResultItem(
                        url=url,
                        status="error",
                        error="Failed to extract content",
                        error_code="EXTRACTION_ERROR",
                    )

            # Convert to Markdown
            content = self._to_markdown(extracted_text)

            # Process content
            content = ContentProcessor.sanitize_markdown(content)
            title = ContentProcessor.extract_title_from_content(content)

            # Try to get title from metadata if not found in content
            if not title:
                metadata = trafilatura.extract_metadata(html_content)
                if metadata and metadata.title:
                    title = metadata.title
                else:
                    title = "Untitled"

            return WebReadResultItem(
                url=url,
                status="success",
                title=title,
                content=content,
                extracted_at=datetime.now(),
                word_count=ContentProcessor.count_words(content),
                token_count=ContentProcessor.estimate_tokens(content),
            )

        except Exception as e:
            logger.error(f"Trafilatura read failed for {url}: {e}")
            return WebReadResultItem(
                url=url,
                status="error",
                error=f"Read failed: {str(e)}",
                error_code="READ_ERROR",
            )

    async def _fetch_html(self, url: str, timeout: int, locale: str) -> str:
        """
        Fetch HTML content from URL.

        Args:
            url: URL to fetch
            timeout: Request timeout
            locale: Locale for User-Agent

        Returns:
            HTML content string
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ApeRAG/1.0; +https://aperag.ai)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": f"{locale.replace('-', '_')},en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return ""
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""

    def _to_markdown(self, extracted_content: str) -> str:
        """
        Convert extracted content to Markdown.

        Args:
            extracted_content: Content from Trafilatura

        Returns:
            Markdown content
        """
        try:
            # If content is XML format, convert to HTML first then to Markdown
            if extracted_content.strip().startswith("<"):
                # Convert XML/HTML to Markdown
                markdown_content = markdownify.markdownify(
                    extracted_content,
                    heading_style="ATX",  # Use # for headings
                    bullets="-",  # Use - for lists
                    escape_asterisks=False,
                    escape_underscores=False,
                )
                return markdown_content
            else:
                # Already plain text, just return
                return extracted_content
        except Exception as e:
            logger.warning(f"Markdown conversion failed: {e}")
            return extracted_content

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
        """
        if not urls:
            return []

        # Normalize URLs
        normalized_urls = [URLValidator.normalize_url(url.strip()) for url in urls]

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_single(url: str) -> WebReadResultItem:
            async with semaphore:
                return await self.read(
                    url=url,
                    timeout=timeout,
                    locale=locale,
                )

        # Execute all reads concurrently
        try:
            results = await asyncio.gather(*[read_single(url) for url in normalized_urls], return_exceptions=True)

            # Handle exceptions in results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch read failed for {normalized_urls[i]}: {result}")
                    final_results.append(
                        WebReadResultItem(
                            url=normalized_urls[i],
                            status="error",
                            error=f"Batch read failed: {str(result)}",
                            error_code="BATCH_ERROR",
                        )
                    )
                else:
                    final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Batch read failed: {e}")
            raise ReaderProviderError(f"Batch read failed: {str(e)}")

    async def close(self):
        """Close and cleanup resources."""
        # No resources to close
        pass

    def get_provider_info(self) -> dict:
        """
        Get provider information.

        Returns:
            Provider information dictionary
        """
        return {
            "name": "Trafilatura",
            "description": "Lightweight content extraction without browser dependencies",
            "supports_javascript": False,
            "supports_spa": False,
            "output_format": "markdown",
            "free": True,
            "requires_api_key": False,
            "lightweight": True,
        }
