"""
Unit tests for the simplified web search endpoint.

The current design intentionally keeps web search thin:
- one regular search path
- JINA preferred when configured
- DuckDuckGo fallback
- provider failures soft-fail into non-500 responses with lightweight diagnostics
- unexpected internal errors still propagate
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.domains.web_access.api.routes import web_search_endpoint
from aperag.domains.web_access.schemas import WebSearchMeta, WebSearchRequest, WebSearchResponse, WebSearchResultItem


class TestWebSearchEndpoint:
    """Test the simplified web search endpoint behavior."""

    def setup_method(self):
        self.mock_user = MagicMock()
        self.mock_user.id = 1
        self.mock_user.username = "test_user"

    @pytest.mark.asyncio
    async def test_regular_search_only(self):
        mock_results = [
            WebSearchResultItem(
                rank=1,
                title="Test Result 1",
                url="https://example1.com",
                snippet="Test snippet 1",
                domain="example1.com",
            ),
            WebSearchResultItem(
                rank=2,
                title="Test Result 2",
                url="https://example2.com",
                snippet="Test snippet 2",
                domain="example2.com",
            ),
        ]
        mock_response = WebSearchResponse(
            query="test query",
            results=mock_results,
            total_results=2,
            search_time=0.1,
            meta=WebSearchMeta(
                search_status="ok",
                provider_used=["jina"],
                backend_used=["jina"],
                fallback_used=False,
                error_code=None,
            ),
        )

        with patch("aperag.domains.web_access.api.routes._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response

            request = WebSearchRequest(query="test query", max_results=5)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.query == "test query"
            assert len(response.results) == 2
            assert response.results[0].title == "Test Result 1"
            assert response.results[1].title == "Test Result 2"
            assert response.meta is not None
            assert response.meta.search_status == "ok"
            assert response.meta.provider_used == ["jina"]
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_site_specific_search(self):
        mock_results = [
            WebSearchResultItem(
                rank=1,
                title="Site Result",
                url="https://github.com/test",
                snippet="GitHub content",
                domain="github.com",
            )
        ]
        mock_response = WebSearchResponse(
            query="",
            results=mock_results,
            total_results=1,
            search_time=0.1,
            meta=WebSearchMeta(
                search_status="ok",
                provider_used=["jina"],
                backend_used=["jina"],
                fallback_used=False,
                error_code=None,
            ),
        )

        with patch("aperag.domains.web_access.api.routes._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response

            request = WebSearchRequest(source="github.com", max_results=3)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.query == "site:github.com"
            assert len(response.results) == 1
            assert response.results[0].url == "https://github.com/test"
            assert response.meta is not None
            assert response.meta.search_status == "ok"
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_no_params(self):
        request = WebSearchRequest()

        with pytest.raises(Exception) as exc_info:
            await web_search_endpoint(request, self.mock_user)

        assert "At least one search input is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_handling_empty_query(self):
        request = WebSearchRequest(query="")

        with pytest.raises(Exception) as exc_info:
            await web_search_endpoint(request, self.mock_user)

        assert "At least one search input is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_failure_handling_returns_unavailable_meta(self):
        """Test provider unavailability is soft-failed into an unavailable response."""
        with patch("aperag.domains.web_access.api.routes._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = WebSearchResponse(
                query="test query",
                results=[],
                total_results=0,
                search_time=0.05,
                meta=WebSearchMeta(
                    search_status="unavailable",
                    provider_used=["duckduckgo"],
                    backend_used=["duckduckgo:auto", "duckduckgo:html"],
                    fallback_used=True,
                    error_code="duckduckgo_unavailable",
                ),
            )

            request = WebSearchRequest(query="test query", max_results=5)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.query == "test query"
            assert response.results == []
            assert response.meta is not None
            assert response.meta.search_status == "unavailable"
            assert response.meta.error_code == "duckduckgo_unavailable"
            assert response.meta.fallback_used is True

    @pytest.mark.asyncio
    async def test_empty_results_remain_distinct_from_unavailable(self):
        """Test an empty search response is not mislabeled as provider unavailability."""
        with patch("aperag.domains.web_access.api.routes._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = WebSearchResponse(
                query="no matches",
                results=[],
                total_results=0,
                search_time=0.05,
                meta=WebSearchMeta(
                    search_status="empty",
                    provider_used=["duckduckgo"],
                    backend_used=["duckduckgo:auto"],
                    fallback_used=False,
                    error_code=None,
                ),
            )

            request = WebSearchRequest(query="no matches", max_results=5)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.results == []
            assert response.meta is not None
            assert response.meta.search_status == "empty"
            assert response.meta.error_code is None

    @pytest.mark.asyncio
    async def test_unexpected_internal_error_is_not_soft_failed(self):
        with patch("aperag.domains.web_access.api.routes._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = RuntimeError("unexpected bug")

            request = WebSearchRequest(query="test query", max_results=5)

            with pytest.raises(RuntimeError, match="unexpected bug"):
                await web_search_endpoint(request, self.mock_user)
