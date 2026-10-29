"""
DuckDuckGo Search Provider

Web search provider using DuckDuckGo search engine.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)

try:
    # ``ddgs`` is the renamed and actively maintained successor of
    # ``duckduckgo-search``. The 8.x line pins its metasearch to a single bing
    # backend and raises ``DuckDuckGoSearchException("<url> return None")`` as
    # soon as that backend returns an empty page, which broke the DuckDuckGo
    # fallback for users without a JINA key (see issue #1722). The old package
    # is kept only as a last-resort fallback for environments that still ship
    # it.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - depends on the installed packages
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        logger.error("ddgs package is required. Install with: pip install ddgs")
        raise


class DuckDuckGoSearchError(Exception):
    """Raised when the DuckDuckGo backends keep failing for a query."""


class DuckDuckGoProvider(BaseSearchProvider):
    """
    DuckDuckGo search provider implementation.

    Uses the ``ddgs`` metasearch library to perform web searches.
    """

    # The metasearch backends are flaky (empty pages, rate limiting, backend
    # rotation): retry a few times with exponential backoff before giving up
    # on a query.
    max_attempts = 3
    retry_backoff = 0.5

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
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using DuckDuckGo.

        Args:
            query: Search query (can be empty for site-specific browsing)
            max_results: Maximum number of results to return
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

        # Perform search, retrying transient backend failures
        search_results: List[dict] = []
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
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
            except Exception as e:
                last_error = e
                search_results = []
                logger.warning(
                    "DuckDuckGo search attempt %d/%d failed for query %r: %s",
                    attempt,
                    self.max_attempts,
                    query,
                    e,
                )
            else:
                last_error = None
                if search_results:
                    break
                # An empty result set is a normal answer (rare query), not a
                # backend failure: do not retry it.
                logger.info("DuckDuckGo returned no results for query %r", query)
                break

            if attempt < self.max_attempts:
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

        if not search_results and last_error is not None:
            # Surface the real provider error instead of letting the caller
            # report a misleading "both providers failed" message.
            raise DuckDuckGoSearchError(
                f"DuckDuckGo search failed after {self.max_attempts} attempts: {last_error}"
            ) from last_error

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
