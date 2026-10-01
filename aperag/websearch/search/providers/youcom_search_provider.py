"""
You.com Search Provider

Web search provider using You.com's Search API.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from aperag.schema.view_models import WebSearchResultItem
from aperag.websearch.search.base_search import BaseSearchProvider
from aperag.websearch.utils.url_validator import URLValidator

logger = logging.getLogger(__name__)


class YouComSearchProvider(BaseSearchProvider):
    """
    You.com search provider implementation.

    Uses You.com's Search API to perform web searches.
    Get your You.com API key: https://you.com/platform/api-keys
    """

    def __init__(self, config: dict = None):
        """
        Initialize You.com search provider.

        Args:
            config: Provider configuration containing api_key and other settings
        """
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None

        self.base_url = "https://ydc-index.io/v1/search"
        self.supported_engines = ["youcom", "you_com"]

        # Configure session headers according to You.com API documentation
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ApeRAG-WebSearch/1.0",
        }
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 30,
        locale: str = "en-US",
        source: Optional[str] = None,
    ) -> List[WebSearchResultItem]:
        """
        Perform web search using the You.com Search API.

        According to You.com docs: POST https://ydc-index.io/v1/search
        with a JSON request body and an X-API-Key header.

        Args:
            query: Search query (can be empty for site-specific browsing)
            max_results: Maximum number of results to return (applied via the API count parameter)
            timeout: Request timeout in seconds
            locale: Browser locale (converted to BCP 47 language parameter)
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

        if not final_query.strip():
            raise ValueError("Search query cannot be empty")

        try:
            # Build request payload according to You.com Search API
            payload: Dict[str, Any] = {
                "query": final_query,
                "count": max_results,
            }

            if locale:
                # Convert locale to BCP 47 language format
                payload["language"] = locale.replace("_", "-")

            if target_domain:
                # You.com natively supports a strict domain allowlist
                payload["include_domains"] = [target_domain]

            logger.info(f"You.com search request: {self.base_url}")
            logger.debug(f"Request payload: {payload}")

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"You.com Search API returned status {response.status}: {response_text}")
                        return []

                    # Parse response as JSON
                    try:
                        response_data = await response.json()
                        return self._parse_youcom_json_response(response_data, target_domain, max_results)
                    except Exception as e:
                        logger.error(f"Failed to parse You.com JSON response: {e}")
                        return []

        except asyncio.TimeoutError:
            logger.error(f"You.com search timed out after {timeout} seconds")
            return []
        except Exception as e:
            logger.error(f"Error in You.com search: {e}")
            return []

    def _parse_youcom_json_response(
        self, response_data: Dict[str, Any], target_domain: Optional[str] = None, max_results: int = 5
    ) -> List[WebSearchResultItem]:
        """Parse You.com API JSON response into standardized result items."""
        results = []

        # Expected response format: {"results": {"web": [...], "news": [...]}}
        results_obj = response_data.get("results", {}) if isinstance(response_data, dict) else {}
        items = results_obj.get("web", []) or [] if isinstance(results_obj, dict) else []

        logger.info(f"Parsing {len(items)} items from You.com JSON response")

        for i, item in enumerate(items):
            if len(results) >= max_results:
                break

            try:
                if not isinstance(item, dict):
                    continue

                url = item.get("url", "") or ""
                if not url or not URLValidator.is_valid_url(url):
                    continue

                # Apply domain filtering if specified
                result_domain = URLValidator.extract_domain(url) or ""
                if target_domain and result_domain.lower() != target_domain.lower():
                    continue

                # Extract title and description with fallbacks
                title = (item.get("title", "") or "No Title").strip()

                snippet = (
                    item.get("description", "")
                    or " ".join(item.get("snippets", []) or [])
                    or "No description available"
                ).strip()

                results.append(
                    WebSearchResultItem(
                        url=url,
                        title=title,
                        snippet=snippet,
                        rank=len(results) + 1,  # Use filtered rank
                        domain=result_domain,
                        timestamp=datetime.now(),
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to parse You.com result item {i}: {e}")
                continue

        logger.info(
            f"You.com search JSON parsing completed: {len(results)} results"
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
