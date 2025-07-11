"""
JINA Search Provider

Web search provider using JINA's s.jina.ai API.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

import aiohttp

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


class JinaSearchProvider(BaseSearchProvider):
    """
    JINA search provider implementation.

    Uses JINA's s.jina.ai API to perform web searches with LLM-friendly results.
    Get your JINA AI API key for free: https://jina.ai/?sui=apikey
    """

    def __init__(self, config: dict = None):
        """
        Initialize JINA search provider.

        Args:
            config: Provider configuration containing api_key and other settings
        """
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None

        self.base_url = "https://s.jina.ai/"
        self.supported_engines = ["jina", "google", "bing"]

        # Configure session headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "google",
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
        use_source_domain_only: bool = False,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using JINA Search API.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_engine: Search engine to use (google, bing, etc.)
            timeout: Request timeout in seconds
            locale: Browser locale
            source: Domain or URL for site-specific search
            use_source_domain_only: If True, only return results from specified source

        Returns:
            List of search result items
        """
        # Validate parameters
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_results > 100:
            raise ValueError("max_results cannot exceed 100")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if not self.api_key:
            raise ValueError("JINA API key is required. Pass api_key in provider_config.")

        # Construct query based on source restrictions
        final_query = query
        target_domain = None
        
        if source and use_source_domain_only:
            # Extract domain from source for site-specific search
            target_domain = URLValidator.extract_domain_from_source(source)
            if target_domain:
                final_query = f"site:{target_domain} {query}"
                logger.info(f"Using JINA site-specific search for domain: {target_domain}")
            else:
                logger.warning(f"No valid domain found in source '{source}', using regular search")

        # Perform search
        results = await self._jina_search(final_query, max_results, search_engine, timeout, locale)
        
        # Additional filtering if needed for site-specific search
        if target_domain:
            filtered_results = []
            for result in results:
                result_domain = URLValidator.extract_domain(result.url)
                if result_domain and result_domain.lower() == target_domain.lower():
                    filtered_results.append(result)

            # Re-rank filtered results
            for i, result in enumerate(filtered_results):
                result.rank = i + 1

            logger.info(f"JINA site-specific search completed: {len(filtered_results)} results from {target_domain}")
            return filtered_results
        
        return results

    async def _jina_search(
        self,
        query: str,
        max_results: int,
        search_engine: str,
        timeout: int,
        locale: str,
    ) -> List[WebSearchResultItem]:
        """
        Perform JINA search request.

        Args:
            query: Search query
            max_results: Maximum number of results
            search_engine: Search engine to use
            timeout: Request timeout
            locale: Browser locale

        Returns:
            List of search result items
        """
        # Prepare request URL
        request_url = f"{self.base_url}{query}"

        # Prepare query parameters
        params = {
            "engine": search_engine,
            "no-cache": "false",
            "gather": "title,snippet,url",
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(request_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
        except asyncio.TimeoutError:
            raise ValueError(f"JINA API request timed out after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise ValueError(f"JINA API request failed: {e}")
        except Exception as e:
            raise ValueError(f"JINA API error: {e}")

        # Parse results
        results = self._parse_search_results(data, query)
        return results[:max_results]

    def _parse_search_results(self, data: dict, query: str) -> List[WebSearchResultItem]:
        """
        Parse JINA search response into WebSearchResultItem objects.

        Args:
            data: Raw response data from JINA API
            query: Original search query

        Returns:
            List of parsed search result items
        """
        results = []

        # JINA s.jina.ai returns results in 'data' field with 'content' containing structured info
        try:
            data_section = data.get("data", {})
            content = data_section.get("content", "")
            citations = data_section.get("citations", [])
        except (AttributeError, TypeError) as e:
            logger.error(f"Invalid JINA API response format: {e}")
            return results
            
        if not content:
            logger.warning("No content found in JINA search response")
            return results
        
        if citations:
            for i, citation in enumerate(citations):
                if isinstance(citation, dict) and "url" in citation:
                    url = citation["url"]
                    title = citation.get("title", f"Result {i+1}")
                    
                    # Create snippet from description or content
                    snippet = citation.get("description", citation.get("snippet", ""))
                    if not snippet and content:
                        # Use part of the content as snippet
                        snippet = content[:200] + "..." if len(content) > 200 else content
                    
                    # Validate URL
                    if URLValidator.is_valid_url(url):
                        results.append(
                            WebSearchResultItem(
                                rank=i + 1,
                                title=title,
                                url=url,
                                snippet=snippet,
                                domain=URLValidator.extract_domain(url),
                                timestamp=datetime.now(),
                            )
                        )

        # If no citations found, create a generic result
        if not results and content:
            results.append(
                WebSearchResultItem(
                    rank=1,
                    title=f"Search results for: {query}",
                    url="https://jina.ai/",
                    snippet=content[:300] + "..." if len(content) > 300 else content,
                    domain="jina.ai",
                    timestamp=datetime.now(),
                )
            )

        return results

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return self.supported_engines.copy()
