"""
DuckDuckGo Search Provider

Web search provider using DuckDuckGo search engine.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
except ImportError:
    logger.error("duckduckgo_search package is required. Install with: pip install duckduckgo-search")
    raise


class DuckDuckGoProvider(BaseSearchProvider):
    """
    DuckDuckGo search provider implementation.

    Uses the duckduckgo-search library to perform web searches.
    """

    def __init__(self, config: dict = None):
        """
        Initialize DuckDuckGo provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.supported_engines = ["duckduckgo", "ddg"]

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_engine: str = "duckduckgo",
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using DuckDuckGo.

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

        # Construct query based on source restrictions
        final_query = query or ""
        target_domain = None

        if source:
            # Extract domain from source for site-specific search
            target_domain = URLValidator.extract_domain_from_source(source)

            if target_domain:
                if has_query:
                    # Query + site restriction
                    final_query = f"site:{target_domain} {query}"
                    logger.info(f"Using site-specific search with query for domain: {target_domain}")
                else:
                    # Site browsing without specific query
                    final_query = f"site:{target_domain}"
                    logger.info(f"Using site browsing for domain: {target_domain}")
            else:
                logger.warning(f"No valid domain found in source '{source}', using regular search")
                if not has_query:
                    raise ValueError("Invalid source domain and no query provided")

        # Perform search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._search_sync, final_query, max_results, timeout, locale)

        # Filter results to target domain when source is provided
        if target_domain:
            filtered_results = []
            for result in results:
                result_domain = URLValidator.extract_domain(result.url)
                if result_domain and result_domain.lower() == target_domain.lower():
                    filtered_results.append(result)

            # Re-rank filtered results
            for i, result in enumerate(filtered_results):
                result.rank = i + 1

            logger.info(f"Site-specific search completed: {len(filtered_results)} results from {target_domain}")
            return filtered_results

        return results

    def _search_sync(self, query: str, max_results: int, timeout: int, locale: str) -> List[WebSearchResultItem]:
        """
        Synchronous search implementation.

        Args:
            query: Search query
            max_results: Maximum number of results
            timeout: Request timeout
            locale: Browser locale

        Returns:
            List of search result items
        """
        # Configure DuckDuckGo search
        region = "cn-zh" if locale.startswith("zh") else "wt-wt"

        # Perform search
        with DDGS() as ddgs:
            search_results = list(
                ddgs.text(
                    query,
                    region=region,
                    safesearch="moderate",
                    timelimit=None,
                    max_results=max_results,
                )
            )

        # Convert results to our format
        results = []
        for i, result in enumerate(search_results):
            # Validate URL
            url = result.get("href", "")
            if not URLValidator.is_valid_url(url):
                continue

            results.append(
                WebSearchResultItem(
                    rank=i + 1,
                    title=result.get("title", ""),
                    url=url,
                    snippet=result.get("body", ""),
                    domain=URLValidator.extract_domain(url),
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
