"""
SerpBase Search Provider

Web search provider using SerpBase Google Search Results API.
Get an API key at https://serpbase.dev — 100 free searches, no credit card required.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)

API_URL = "https://api.serpbase.dev/google/search"


class SerpBaseSearchProvider(BaseSearchProvider):
    """
    SerpBase Google search provider implementation.

    Uses SerpBase's REST API to return structured Google search results
    with zero scraping maintenance. Requires a free API key from serpbase.dev.
    """

    def __init__(self, config: dict = None):
        """
        Initialize SerpBase search provider.

        Args:
            config: Provider configuration containing api_key and other settings
        """
        super().__init__(config)
        self.api_key = (config.get("api_key") if config else None) or os.environ.get(
            "SERPBASE_API_KEY", ""
        )
        self.supported_engines = ["google", "serpbase"]

    async def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using SerpBase Google Search API.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
            locale: Browser locale (maps to gl parameter)
            source: Domain or URL for site-specific search

        Returns:
            List of search result items
        """
        # Graceful skip when API key is not configured
        if not self.api_key:
            logger.warning(
                "SERPBASE_API_KEY not set — skipping SerpBase search. "
                "Get a free key at https://serpbase.dev"
            )
            return []

        # Validate parameters
        has_query = query and query.strip()
        has_source = source and source.strip()

        if not has_query and not has_source:
            raise ValueError("Either query or source must be provided")

        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_results > 100:
            raise ValueError("max_results cannot exceed 100")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        # Build query — site: search when source is provided
        final_query = query or ""
        target_domain = None

        if source:
            target_domain = URLValidator.extract_domain_from_source(source)
            if target_domain and has_query:
                final_query = f"site:{target_domain} {query}"
            elif target_domain and not has_query:
                final_query = f"site:{target_domain}"
            elif not target_domain and not has_query:
                raise ValueError("Invalid source domain and no query provided")

        if not final_query.strip():
            raise ValueError("Search query cannot be empty")

        # Build API request parameters
        params: Dict[str, Any] = {
            "q": final_query,
            "api_key": self.api_key,
            "num": min(max_results, 100),
        }

        # Map locale to gl (country) parameter
        gl_map = {
            "zh-CN": "cn",
            "zh-TW": "tw",
            "ja-JP": "jp",
            "ko-KR": "kr",
            "de-DE": "de",
            "fr-FR": "fr",
            "es-ES": "es",
        }
        if locale in gl_map:
            params["gl"] = gl_map[locale]

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(API_URL, params=params) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(
                            f"SerpBase API returned status {response.status}: {response_text}"
                        )
                        return []

                    data = await response.json()
                    return self._parse_response(data, target_domain, max_results)

        except asyncio.TimeoutError:
            logger.error(f"SerpBase search timed out after {timeout} seconds")
            return []
        except Exception as e:
            logger.error(f"Error in SerpBase search: {e}")
            return []

    def _parse_response(
        self,
        data: Dict[str, Any],
        target_domain: Optional[str] = None,
        max_results: int = 5,
    ) -> List[WebSearchResultItem]:
        """Parse SerpBase API response into standardized result items."""
        results: List[WebSearchResultItem] = []
        organic = data.get("organic_results", [])

        for item in organic:
            if len(results) >= max_results:
                break

            url = item.get("link", "")
            if not url or not URLValidator.is_valid_url(url):
                continue

            # Domain filtering
            if target_domain:
                result_domain = URLValidator.extract_domain(url)
                if not result_domain or result_domain.lower() != target_domain.lower():
                    continue

            rank = item.get("position", len(results) + 1)

            results.append(
                WebSearchResultItem(
                    rank=rank,
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("snippet", ""),
                    domain=URLValidator.extract_domain(url) or "",
                    timestamp=datetime.now(),
                )
            )

        logger.info(
            f"SerpBase search completed: {len(results)} results"
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
