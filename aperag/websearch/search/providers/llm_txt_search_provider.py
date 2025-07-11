"""
LLM.txt Search Provider

Specialized search provider for discovering LLM.txt files from domains.
This provider focuses exclusively on finding LLM-optimized content indexes.
"""

import logging
import re
from typing import List, Optional

from aperag.schema.view_models import WebReadRequest, WebSearchResultItem
from aperag.websearch.reader.reader_service import ReaderService
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


class LLMTxtSearchProvider(BaseSearchProvider):
    """
    LLM.txt search provider implementation.

    This provider specializes in discovering LLM.txt files from specified domains.
    It does not perform traditional web search, but instead looks for LLM-optimized
    content indexes that websites provide for AI applications.
    """

    # LLM.txt file patterns to try (in priority order)
    # Simplified to most commonly used patterns for better performance
    LLM_TXT_PATTERNS = [
        # Standard root paths (most common)
        "/llms.txt",
        "/llms-full.txt",
        # RFC 5785 compliant paths (recommended standard)
        "/.well-known/llms.txt",
        "/.well-known/llms-full.txt",
        # Common documentation paths
        "/docs/llms.txt",
        "/docs/llms-full.txt",
        # API reference paths
        "/api/llms.txt",
        "/reference/llms.txt",
    ]

    def __init__(self, config: dict = None):
        """
        Initialize LLM.txt search provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.supported_engines = ["llm_txt"]
        self.reader_service = None  # Lazy initialization

    def _get_reader_service(self) -> ReaderService:
        """
        Get reader service instance (lazy initialization).

        Returns:
            ReaderService instance
        """
        if self.reader_service is None:
            self.reader_service = ReaderService.create_default()
        return self.reader_service

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "llm_txt",
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform LLM.txt discovery search.

        This provider specifically searches for llms.txt files from a given source domain.
        It ignores the query parameter as it's designed for LLM-optimized content discovery.

        Args:
            query: Search query (ignored by this provider)
            max_results: Maximum number of results to return
            search_engine: Search engine to use
            timeout: Request timeout in seconds
            locale: Browser locale
            source: Domain or URL to search for llms.txt files (required)

        Returns:
            List of search result items
        """
        # Validate parameters
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_results > 100:
            raise ValueError("max_results cannot exceed 100")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if not source:
            logger.info("No source provided for LLM.txt search, returning empty results")
            return []

        source = source.strip()

        # Check if source is already a direct LLM.txt URL
        if self._is_llms_txt_url(source):
            logger.info(f"Source appears to be a direct LLM.txt URL: {source}")
            result = await self._try_read_llms_txt_url(source, timeout)
            if result:
                result.rank = 1
                logger.info("LLM.txt search completed: 1 result found from direct URL")
                return [result]

        # Extract domain from source for pattern-based discovery
        domain = URLValidator.extract_domain_from_source(source)

        if not domain:
            logger.warning(f"No valid domain found in source '{source}' for LLM.txt search")
            return []

        logger.info(f"Starting pattern-based LLM.txt search for domain: {domain}")

        # Discover LLM.txt files using patterns
        result = await self._discover_llms_txt_for_domain(domain, timeout)

        if result:
            result.rank = 1
            results = [result]
        else:
            results = []

        # Limit results to max_results
        limited_results = results[:max_results]

        # Re-rank results
        for i, result in enumerate(limited_results):
            result.rank = i + 1

        logger.info(f"LLM.txt search completed: {len(limited_results)} results found")
        return limited_results

    async def _discover_llms_txt_for_domain(self, domain: str, timeout: int = 30) -> Optional[WebSearchResultItem]:
        """
        Discover LLM.txt files for a specific domain.

        Args:
            domain: Domain name to discover LLM.txt files for
            timeout: Request timeout in seconds

        Returns:
            WebSearchResultItem if LLM.txt found, None otherwise
        """
        reader_service = self._get_reader_service()

        for pattern in self.LLM_TXT_PATTERNS:
            url = f"https://{domain}{pattern}"

            try:
                logger.debug(f"Trying LLM.txt URL: {url}")

                # Try to read the LLM.txt file
                read_request = WebReadRequest(urls=url, timeout=timeout)
                read_response = await reader_service.read(read_request)

                if (
                    read_response.results
                    and read_response.results[0].status == "success"
                    and read_response.results[0].content
                ):
                    result_item = read_response.results[0]

                    # Convert read result to search result format
                    search_result = WebSearchResultItem(
                        rank=1,
                        title=result_item.title or f"LLM.txt from {domain}",
                        url=url,
                        snippet=self._create_snippet_from_content(result_item.content),
                        domain=domain,
                        timestamp=result_item.extracted_at,
                    )

                    logger.info(f"Successfully discovered LLM.txt: {url}")
                    return search_result

            except Exception as e:
                logger.debug(f"Failed to read LLM.txt from {url}: {e}")
                continue

        return None

    def _is_llms_txt_url(self, url: str) -> bool:
        """
        Check if the URL appears to be a direct LLM.txt file URL.

        Args:
            url: URL to check

        Returns:
            True if URL looks like an LLM.txt file URL
        """
        if not url:
            return False

        url_lower = url.lower()

        # Check if URL starts with http/https
        if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
            return False

        # Check if URL ends with llms.txt or llms-full.txt
        return url_lower.endswith("llms.txt") or url_lower.endswith("llms-full.txt")

    async def _try_read_llms_txt_url(self, url: str, timeout: int = 30) -> Optional[WebSearchResultItem]:
        """
        Try to read LLM.txt content from a direct URL.

        Args:
            url: Direct URL to LLM.txt file
            timeout: Request timeout in seconds

        Returns:
            WebSearchResultItem if successful, None otherwise
        """
        reader_service = self._get_reader_service()

        try:
            logger.debug(f"Trying direct LLM.txt URL: {url}")

            # Try to read the LLM.txt file
            read_request = WebReadRequest(urls=url, timeout=timeout)
            read_response = await reader_service.read(read_request)

            if (
                read_response.results
                and read_response.results[0].status == "success"
                and read_response.results[0].content
            ):
                result_item = read_response.results[0]
                domain = URLValidator.extract_domain(url)

                # Convert read result to search result format
                search_result = WebSearchResultItem(
                    rank=1,
                    title=result_item.title or f"LLM.txt from {domain}",
                    url=url,
                    snippet=self._create_snippet_from_content(result_item.content),
                    domain=domain,
                    timestamp=result_item.extracted_at,
                )

                logger.info(f"Successfully read direct LLM.txt URL: {url}")
                return search_result

        except Exception as e:
            logger.debug(f"Failed to read direct LLM.txt URL {url}: {e}")

        return None

    async def _discover_llms_txt_for_domains(self, domains: List[str], timeout: int = 30) -> List[WebSearchResultItem]:
        """
        Discover LLM.txt files for multiple domains concurrently.

        Args:
            domains: List of domain names
            timeout: Request timeout in seconds

        Returns:
            List of WebSearchResultItem from successful discoveries
        """
        import asyncio

        tasks = [self._discover_llms_txt_for_domain(domain, timeout) for domain in domains]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        llm_txt_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"LLM.txt discovery failed for {domains[i]}: {result}")
            elif result is not None:
                llm_txt_results.append(result)

        return llm_txt_results

    def _create_snippet_from_content(self, content: str, max_length: int = 200) -> str:
        """
        Create a snippet from content for search results.

        Args:
            content: Full content text
            max_length: Maximum snippet length

        Returns:
            Content snippet
        """
        if not content:
            return ""

        # Remove markdown formatting and extra whitespace
        text = re.sub(r"[#*`\[\]()]", "", content)
        text = " ".join(text.split())

        if len(text) <= max_length:
            return text

        # Truncate and add ellipsis
        return text[:max_length].rstrip() + "..."

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return self.supported_engines.copy()

    async def close(self):
        """
        Close and cleanup resources.
        """
        if self.reader_service and hasattr(self.reader_service, "close"):
            await self.reader_service.close()
