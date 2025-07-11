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
        use_source_domain_only: bool = False,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_engine: Search engine to use
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

        # Construct query based on source restrictions
        final_query = query
        target_domain = None
        
        if source and use_source_domain_only:
            # Extract domain from source for site-specific search
            target_domain = URLValidator.extract_domain_from_source(source)
            if target_domain:
                final_query = f"site:{target_domain} {query}"
                logger.info(f"Using site-specific search for domain: {target_domain}")
            else:
                logger.warning(f"No valid domain found in source '{source}', using regular search")

        # Perform search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._search_sync, final_query, max_results, timeout, locale)
        
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
