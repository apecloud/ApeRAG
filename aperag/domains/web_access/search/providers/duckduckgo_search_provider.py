"""
DuckDuckGo Search Provider

Web search provider using DuckDuckGo search engine.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from aperag.domains.web_access.schemas import WebSearchResultItem
from aperag.domains.web_access.search.base_search import BaseSearchProvider
from aperag.domains.web_access.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException, TimeoutException
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
        self.backend_fallback_order = self._normalize_backend_order(
            (config or {}).get("backend_fallback_order", ["auto", "html", "lite"])
        )

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
        self.last_search_meta = {
            "search_status": "empty",
            "provider_used": ["duckduckgo"],
            "backend_used": [],
            "fallback_used": False,
            "error_code": None,
        }

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
        try:
            results = await loop.run_in_executor(None, self._search_sync, final_query, max_results, timeout, locale)
        except Exception as exc:
            logger.warning("DuckDuckGo search failed query=%s error=%s", final_query, exc)
            self.last_search_meta["search_status"] = "unavailable"
            self.last_search_meta["error_code"] = "duckduckgo_unavailable"
            return []

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
            if self.last_search_meta["search_status"] != "unavailable":
                self.last_search_meta["search_status"] = "ok" if filtered_results else "empty"
            return filtered_results

        if self.last_search_meta["search_status"] != "unavailable":
            self.last_search_meta["search_status"] = "ok" if results else "empty"
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

        search_results = []
        last_error: Exception | None = None
        attempted_backends: list[str] = []

        for backend in self.backend_fallback_order:
            attempted_backends.append(backend)
            logger.info("DuckDuckGo search attempt backend=%s query=%s", backend, query)
            try:
                with DDGS() as ddgs:
                    search_results = list(
                        ddgs.text(
                            query,
                            region=region,
                            safesearch="moderate",
                            timelimit=None,
                            backend=backend,
                            max_results=max_results,
                        )
                    )

                logger.info("DuckDuckGo search backend=%s completed results=%s", backend, len(search_results))
                self.last_search_meta["backend_used"] = [f"duckduckgo:{backend}"]
                self.last_search_meta["fallback_used"] = len(attempted_backends) > 1

                # Respect an empty response as a valid search result instead of issuing more external requests.
                break
            except (RatelimitException, TimeoutException, DuckDuckGoSearchException) as exc:
                last_error = exc
                logger.warning(
                    "DuckDuckGo search backend=%s failed query=%s error=%s",
                    backend,
                    query,
                    exc,
                )
                continue

        if search_results == [] and last_error is not None:
            logger.warning(
                "DuckDuckGo search exhausted all backends query=%s backends=%s last_error=%s",
                query,
                self.backend_fallback_order,
                last_error,
            )
            self.last_search_meta["search_status"] = "unavailable"
            self.last_search_meta["backend_used"] = [f"duckduckgo:{backend}" for backend in attempted_backends]
            self.last_search_meta["fallback_used"] = len(attempted_backends) > 1
            self.last_search_meta["error_code"] = "duckduckgo_unavailable"
            return []

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

    @staticmethod
    def _normalize_backend_order(backends: List[str]) -> List[str]:
        """Normalize DuckDuckGo backend fallback order while preserving the first occurrence."""
        valid_backends = {"auto", "html", "lite"}
        normalized = []

        for backend in backends or []:
            backend_name = str(backend).strip().lower()
            if backend_name in valid_backends and backend_name not in normalized:
                normalized.append(backend_name)

        return normalized or ["auto", "html", "lite"]
