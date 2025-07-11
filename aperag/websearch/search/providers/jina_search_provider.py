"""
JINA Search Provider

Web search provider using JINA's s.jina.ai API.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

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
        search_engine: str = "jina",
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using Jina Search API.

        Args:
            query: Search query (can be empty for site-specific browsing)
            max_results: Maximum number of results to return
            search_engine: Search engine to use
            timeout: Request timeout in seconds
            locale: Browser locale
            source: Domain or URL for site-specific search. When provided, search will be limited to this domain.

        Returns:
            List of search result items
        """
        # Validate parameters
        has_query = query and query.strip()
        has_source = source and source.strip()

        # Either query or source must be provided
        if not has_query and not has_source:
            raise ValueError("Either query or source must be provided")

        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_results > 100:
            raise ValueError("max_results cannot exceed 100")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        # Prepare search query and domain filtering
        final_query = query or ""
        target_domain = None

        if source:
            target_domain = URLValidator.extract_domain_from_source(source)
            if target_domain and has_query:
                # Add site restriction to query
                final_query = f"site:{target_domain} {query}"
            elif target_domain and not has_query:
                # Site browsing without specific query
                final_query = f"site:{target_domain}"
            elif not target_domain and not has_query:
                raise ValueError("Invalid source domain and no query provided")

        # Build request data
        request_data = {
            "q": final_query,
            "count": min(max_results, 20),  # Jina API typically supports up to 20 results
        }

        try:
            # Make request to Jina Search API
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(
                    "https://s.jina.ai/",
                    params=request_data,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": f"Mozilla/5.0 ({locale}) AppleWebKit/537.36",
                    },
                ) as response:
                    if response.status != 200:
                        logger.error(f"Jina Search API returned status {response.status}")
                        return []

                    response_data = await response.json()
                    return self._parse_jina_response(response_data, target_domain)

        except asyncio.TimeoutError:
            logger.error(f"Jina search timed out after {timeout} seconds")
            return []
        except Exception as e:
            logger.error(f"Error in Jina search: {e}")
            return []

    def _parse_jina_response(
        self, response_data: Dict[str, Any], target_domain: Optional[str] = None
    ) -> List[WebSearchResultItem]:
        """Parse Jina API response into standardized result items."""
        results = []

        # Handle different response formats
        items = response_data.get("data", []) or response_data.get("results", [])

        for i, item in enumerate(items):
            try:
                url = item.get("url", "")
                if not url:
                    continue

                # Apply domain filtering if specified
                if target_domain:
                    result_domain = URLValidator.extract_domain(url)
                    if not result_domain or result_domain.lower() != target_domain.lower():
                        continue

                result = WebSearchResultItem(
                    url=url,
                    title=item.get("title", "").strip() or "No Title",
                    snippet=item.get("description", "").strip()
                    or item.get("snippet", "").strip()
                    or "No description available",
                    rank=len(results) + 1,  # Use filtered rank
                    search_engine="jina",
                    metadata={
                        "score": item.get("score"),
                        "published_date": item.get("published_date"),
                        "source_domain": URLValidator.extract_domain(url),
                    },
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse Jina result item: {e}")
                continue

        logger.info(
            f"Jina search completed: {len(results)} results"
            + (f" from domain {target_domain}" if target_domain else "")
        )
        return results

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return self.supported_engines.copy()
