"""
LLM.txt Search Provider

Specialized search provider for discovering LLM.txt files from domains.
This provider focuses exclusively on finding LLM-optimized content indexes.
"""

import logging
import re
from typing import List, Optional

from aperag.schema.view_models import WebSearchResultItem
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

    async def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform LLM.txt discovery search on specified domain.

        Args:
            query: Search query (can be empty for domain discovery)
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
            locale: Browser locale (not used for LLM.txt)
            source: Domain for LLM.txt discovery (required)

        Returns:
            List of search result items containing LLM.txt content
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
            results = await self._try_read_llms_txt_url(source, timeout, max_results)
            if results:
                logger.info(f"LLM.txt search completed: {len(results)} results found from direct URL")
                return results

        # Extract domain from source for pattern-based discovery
        domain = URLValidator.extract_domain_from_source(source)

        if not domain:
            logger.warning(f"No valid domain found in source '{source}' for LLM.txt search")
            return []

        logger.info(f"Starting pattern-based LLM.txt search for domain: {domain}")

        # Discover LLM.txt files using patterns
        results = await self._discover_llms_txt_for_domain(domain, timeout, max_results)

        # Limit results to max_results and re-rank
        limited_results = results[:max_results]
        for i, result in enumerate(limited_results):
            result.rank = i + 1

        logger.info(f"LLM.txt search completed: {len(limited_results)} results found")
        return limited_results

    async def _discover_llms_txt_for_domain(
        self, domain: str, timeout: int = 30, max_results: int = 5
    ) -> List[WebSearchResultItem]:
        """
        Discover LLM.txt files for a specific domain and parse URLs from them.

        Args:
            domain: Domain name to discover LLM.txt files for
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return

        Returns:
            List of WebSearchResultItem from parsed LLM.txt URLs
        """
        for pattern in self.LLM_TXT_PATTERNS:
            url = f"https://{domain}{pattern}"

            try:
                # Try to fetch the LLM.txt file content directly
                content = await self._fetch_llm_txt_content_directly(url, timeout)

                if not content:
                    continue

                # Parse URLs from LLM.txt content
                url_data_list = self._parse_urls_from_llm_txt(content)

                if not url_data_list:
                    continue

                # Create search results for each URL (limited by max_results)
                results = []
                for i, url_data in enumerate(url_data_list[:max_results]):
                    parsed_url = url_data["url"]
                    url_domain = URLValidator.extract_domain(parsed_url)

                    # Generate title and snippet from line content
                    title, snippet = self._generate_title_and_snippet_from_line(url_data)

                    search_result = WebSearchResultItem(
                        rank=i + 1,
                        title=title,
                        url=parsed_url,
                        snippet=snippet,
                        domain=url_domain,
                        timestamp=None,  # No timestamp available from direct fetch
                    )
                    results.append(search_result)

                return results

            except Exception as e:
                logger.warning(f"Failed to process LLM.txt from {url}: {e}")
                continue

        return []

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

    async def _try_read_llms_txt_url(
        self, url: str, timeout: int = 30, max_results: int = 5
    ) -> List[WebSearchResultItem]:
        """
        Try to read LLM.txt content from a direct URL and parse URLs from it.

        Args:
            url: Direct URL to LLM.txt file
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return

        Returns:
            List of WebSearchResultItem from parsed URLs
        """
        try:
            # Try to read the LLM.txt file directly using HTTP request
            # This bypasses the ReaderService which may have issues with plain text files
            content = await self._fetch_llm_txt_content_directly(url, timeout)

            if not content:
                logger.warning(f"Failed to fetch content from LLM.txt URL: {url}")
                return []

            # Parse URLs from LLM.txt content
            url_data_list = self._parse_urls_from_llm_txt(content)

            if not url_data_list:
                return []

            # Create search results for each URL (limited by max_results)
            results = []
            for i, url_data in enumerate(url_data_list[:max_results]):
                parsed_url = url_data["url"]
                domain = URLValidator.extract_domain(parsed_url)

                # Generate title and snippet from line content
                title, snippet = self._generate_title_and_snippet_from_line(url_data)

                search_result = WebSearchResultItem(
                    rank=i + 1,
                    title=title,
                    url=parsed_url,
                    snippet=snippet,
                    domain=domain,
                    timestamp=None,  # No timestamp available from direct fetch
                )
                results.append(search_result)

            return results

        except Exception as e:
            logger.error(f"Failed to read direct LLM.txt URL {url}: {e}")

        return []

    def _parse_urls_from_llm_txt(self, content: str) -> List[dict]:
        """
        Parse URLs from LLM.txt file content with associated line content.

        Args:
            content: LLM.txt file content

        Returns:
            List of dicts with 'url', 'line_content', and 'title' keys
        """
        if not content:
            return []

        url_data = []
        lines = content.strip().split("\n")

        for line in lines:
            original_line = line
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            # Extract URL and title from line
            url, title = self._extract_url_and_title_from_line(line)

            if url and URLValidator.is_valid_url(url):
                url_data.append({"url": url, "line_content": original_line.strip(), "title": title})

        return url_data

    def _extract_url_and_title_from_line(self, line: str) -> tuple:
        """
        Extract URL and title from a single line.

        Args:
            line: Line content to parse

        Returns:
            Tuple of (url, title)
        """
        # Handle markdown format: [Title](URL)
        markdown_match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", line)
        if markdown_match:
            title = markdown_match.group(1).strip()
            url = markdown_match.group(2).strip()
            # Clean up any trailing punctuation from URL
            url = re.sub(r"[.,;!?]+$", "", url)
            return url, title

        # Handle plain URL with optional description: URL - Description
        parts = re.split(r"\s+[-–—]\s+|\s+", line)
        url = parts[0].strip()

        # Clean up any trailing punctuation from URL
        url = re.sub(r"[.,;!?]+$", "", url)

        if url.startswith(("http://", "https://")):
            title = " ".join(parts[1:]).strip() if len(parts) > 1 else None
            return url, title

        return None, None

    def _generate_title_and_snippet_from_line(self, url_data: dict) -> tuple:
        """
        Generate title and snippet from LLM.txt line content.

        Args:
            url_data: Dict with 'url', 'line_content', and 'title' keys

        Returns:
            Tuple of (title, snippet)
        """
        url = url_data["url"]
        line_content = url_data["line_content"]
        extracted_title = url_data.get("title")

        # Generate title: use extracted title if available, otherwise derive from URL
        if extracted_title and len(extracted_title.strip()) > 2:
            title = extracted_title.strip()
        else:
            title = self._generate_title_from_url(url)

        # Generate snippet: clean up and use the original line content
        snippet = self._clean_line_content_for_snippet(line_content)

        return title, snippet

    def _clean_line_content_for_snippet(self, line_content: str) -> str:
        """
        Clean line content to create a readable snippet.

        Args:
            line_content: Original line content from LLM.txt

        Returns:
            Cleaned snippet string
        """
        if not line_content:
            return "LLM-optimized content"

        # Remove leading list markers and whitespace
        content = re.sub(r"^[-*+]\s*", "", line_content.strip())

        # For markdown format: [Description](URL): Additional info
        markdown_match = re.search(r"\[([^\]]+)\]\([^)]+\)(?::\s*(.+))?", content)
        if markdown_match:
            description = markdown_match.group(1).strip()
            additional_info = markdown_match.group(2)
            snippet = f"{description}: {additional_info.strip()}" if additional_info else description
        else:
            snippet = content

        # Clean whitespace and truncate if needed
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet[:197] + "..." if len(snippet) > 200 else snippet or "LLM-optimized content"

    def _generate_title_from_url(self, url: str) -> str:
        """
        Generate a readable title from URL path.

        Args:
            url: URL to generate title from

        Returns:
            Generated title string
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            path_parts = [part for part in parsed.path.split("/") if part]

            if path_parts:
                # Use the last part of the path as title
                title_part = path_parts[-1]

                # Remove file extensions
                if "." in title_part:
                    title_part = title_part.rsplit(".", 1)[0]

                # Convert separators to spaces and title case
                title = title_part.replace("-", " ").replace("_", " ").title()

                # If title is too short or generic, include more context
                if len(title) < 3 or title.lower() in ["index", "home", "main"]:
                    if len(path_parts) > 1:
                        parent_part = path_parts[-2].replace("-", " ").replace("_", " ").title()
                        title = f"{parent_part} - {title}"
                    else:
                        title = f"{parsed.netloc.title()} - {title}"

                return title
            else:
                # Root URL, use domain name
                domain = parsed.netloc.replace("www.", "")
                return domain.replace(".", " ").title()

        except Exception:
            # Fallback to simple domain extraction
            domain = URLValidator.extract_domain(url) or "Unknown"
            return domain.replace(".", " ").title()

    async def _fetch_llm_txt_content_directly(self, url: str, timeout: int = 30) -> str:
        """
        Fetch LLM.txt content directly using HTTP request.

        This bypasses the ReaderService which may have issues with plain text files.

        Args:
            url: URL to fetch content from
            timeout: Request timeout in seconds

        Returns:
            Content string if successful, empty string otherwise
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} when fetching {url}")
                        return ""
        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return ""

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
        # Close reader service if it exists and has a close method
        if hasattr(self, "reader_service") and self.reader_service and hasattr(self.reader_service, "close"):
            await self.reader_service.close()
