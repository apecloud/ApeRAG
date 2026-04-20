"""
Unit tests for the simplified web search endpoint.

The current design intentionally keeps web search thin:
- one regular search path
- JINA preferred when configured
- DuckDuckGo fallback
- provider failures soft-fail to empty results
- unexpected internal errors still propagate
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.schema.view_models import WebSearchRequest, WebSearchResponse, WebSearchResultItem
from aperag.views.web import web_search_endpoint


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
        mock_response = WebSearchResponse(query="test query", results=mock_results, total_results=2, search_time=0.1)

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response

            request = WebSearchRequest(query="test query", max_results=5)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.query == "test query"
            assert len(response.results) == 2
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
        mock_response = WebSearchResponse(query="", results=mock_results, total_results=1, search_time=0.1)

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response

            request = WebSearchRequest(source="github.com", max_results=3)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.query == "site:github.com"
            assert len(response.results) == 1
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
    async def test_provider_soft_failure_returns_empty_results(self):
        provider_failure_response = WebSearchResponse(query="test query", results=[], total_results=0, search_time=0.1)

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = provider_failure_response

            request = WebSearchRequest(query="test query", max_results=5)
            response = await web_search_endpoint(request, self.mock_user)

            assert response.query == "test query"
            assert response.results == []
            assert response.total_results == 0

    @pytest.mark.asyncio
    async def test_unexpected_internal_error_is_not_soft_failed(self):
        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = RuntimeError("unexpected bug")

            request = WebSearchRequest(query="test query", max_results=5)

            with pytest.raises(RuntimeError, match="unexpected bug"):
                await web_search_endpoint(request, self.mock_user)
