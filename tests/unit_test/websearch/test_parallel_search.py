"""
Unit tests for the new parallel search architecture.

Tests the web_search_endpoint function's ability to handle:
- Regular search only
- LLM.txt discovery only
- Site-specific search
- Combined parallel searches
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aperag.schema.view_models import WebSearchRequest, WebSearchResponse, WebSearchResultItem
from aperag.views.web import web_search_endpoint


class TestParallelSearchArchitecture:
    """Test the new parallel search architecture in web_search_endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock user for all tests
        self.mock_user = MagicMock()
        self.mock_user.id = 1
        self.mock_user.username = "test_user"

    @pytest.mark.asyncio
    async def test_regular_search_only(self):
        """Test regular search without LLM.txt discovery."""
        # Mock SearchService behavior
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

        # Mock the search response
        mock_response = WebSearchResponse(query="test query", results=mock_results, total_results=2, search_time=0.1)

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response

            # Test regular search only
            request = WebSearchRequest(query="test query", max_results=5)

            response = await web_search_endpoint(request, self.mock_user)

            # Verify response structure
            assert response.query == "test query"
            assert len(response.results) == 2
            assert response.results[0].title == "Test Result 1"
            assert response.results[1].title == "Test Result 2"

            # Verify that search was called once
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_txt_discovery_only(self):
        """Test LLM.txt discovery without regular search."""
        mock_results = [
            WebSearchResultItem(
                rank=1,
                title="LLM.txt File",
                url="https://example.com/llms.txt",
                snippet="AI-optimized content",
                domain="example.com",
            )
        ]

        mock_response = WebSearchResponse(
            query="LLM.txt:example.com", results=mock_results, total_results=1, search_time=0.05
        )

        with patch("aperag.views.web._search_llm_txt_discovery", new_callable=AsyncMock) as mock_llm_search:
            mock_llm_search.return_value = mock_response

            request = WebSearchRequest(search_llms_txt="example.com", max_results=5)

            response = await web_search_endpoint(request, self.mock_user)

            # Verify response
            assert response.query == "LLM.txt:example.com"
            assert len(response.results) == 1
            assert response.results[0].url == "https://example.com/llms.txt"

            # Verify LLM.txt search was called
            mock_llm_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_combined_parallel_search(self):
        """Test combined regular + LLM.txt search running in parallel."""
        # Mock different results for each search type
        regular_results = [
            WebSearchResultItem(
                rank=1,
                title="Regular Result",
                url="https://regular.com",
                snippet="Regular snippet",
                domain="regular.com",
            )
        ]

        llm_txt_results = [
            WebSearchResultItem(
                rank=1,
                title="LLM.txt Result",
                url="https://example.com/llms.txt",
                snippet="LLM snippet",
                domain="example.com",
            )
        ]

        # Mock responses
        regular_response = WebSearchResponse(
            query="test query", results=regular_results, total_results=1, search_time=0.05
        )

        llm_txt_response = WebSearchResponse(
            query="LLM.txt:example.com", results=llm_txt_results, total_results=1, search_time=0.03
        )

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_regular:
            with patch("aperag.views.web._search_llm_txt_discovery", new_callable=AsyncMock) as mock_llm:
                mock_regular.return_value = regular_response
                mock_llm.return_value = llm_txt_response

                request = WebSearchRequest(query="test query", search_llms_txt="example.com", max_results=5)

                response = await web_search_endpoint(request, self.mock_user)

                # Verify combined response
                assert response.query == "test query + LLM.txt:example.com"
                assert len(response.results) == 2  # Merged results

                # Verify both searches were called
                mock_regular.assert_called_once()
                mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_site_specific_search(self):
        """Test site-specific search with source parameter."""
        mock_results = [
            WebSearchResultItem(
                rank=1,
                title="Site Result",
                url="https://github.com/test",
                snippet="GitHub content",
                domain="github.com",
            )
        ]

        mock_response = WebSearchResponse(query="documentation", results=mock_results, total_results=1, search_time=0.1)

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_response

            request = WebSearchRequest(query="documentation", source="github.com", max_results=3)

            response = await web_search_endpoint(request, self.mock_user)

            # Verify site-specific search
            assert response.query == "documentation"
            assert len(response.results) == 1
            assert response.results[0].url == "https://github.com/test"

            # Verify search was called with correct parameters
            mock_search.assert_called_once()
            call_args = mock_search.call_args[0][0]  # First positional argument (request)
            assert call_args.source == "github.com"

    @pytest.mark.asyncio
    async def test_error_handling_no_params(self):
        """Test error handling when no search parameters are provided."""
        # Empty request should raise an error
        request = WebSearchRequest()

        with pytest.raises(Exception) as exc_info:
            await web_search_endpoint(request, self.mock_user)

        assert "At least one search type is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_handling_empty_query(self):
        """Test error handling with empty query."""
        request = WebSearchRequest(query="")

        with pytest.raises(Exception) as exc_info:
            await web_search_endpoint(request, self.mock_user)

        assert "At least one search type is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_failure_handling(self):
        """Test handling when searches fail."""
        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_search:
            # Make search raise an exception
            mock_search.side_effect = Exception("Network error")

            request = WebSearchRequest(query="test query", max_results=5)

            # Should raise HTTPException when all searches fail
            with pytest.raises(Exception) as exc_info:
                await web_search_endpoint(request, self.mock_user)

            # Verify the error message
            assert "All searches failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_search_failure(self):
        """Test handling when one search succeeds and another fails."""
        regular_results = [
            WebSearchResultItem(
                rank=1, title="Success Result", url="https://success.com", snippet="Success", domain="success.com"
            )
        ]

        regular_response = WebSearchResponse(
            query="test query", results=regular_results, total_results=1, search_time=0.05
        )

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_regular:
            with patch("aperag.views.web._search_llm_txt_discovery", new_callable=AsyncMock) as mock_llm:
                # First service succeeds
                mock_regular.return_value = regular_response

                # Second service fails
                mock_llm.side_effect = Exception("LLM.txt error")

                request = WebSearchRequest(query="test query", search_llms_txt="example.com", max_results=5)

                response = await web_search_endpoint(request, self.mock_user)

                # Should return successful results despite partial failure
                assert len(response.results) == 1
                assert response.results[0].title == "Success Result"

    @pytest.mark.asyncio
    async def test_result_deduplication(self):
        """Test that duplicate URLs are removed from combined results."""
        # Same URL in both result sets
        duplicate_url = "https://example.com/duplicate"

        regular_results = [
            WebSearchResultItem(
                rank=1, title="Regular Title", url=duplicate_url, snippet="Regular snippet", domain="example.com"
            ),
            WebSearchResultItem(
                rank=2,
                title="Unique Regular",
                url="https://unique1.com",
                snippet="Unique snippet",
                domain="unique1.com",
            ),
        ]

        llm_txt_results = [
            WebSearchResultItem(
                rank=1, title="LLM.txt Title", url=duplicate_url, snippet="LLM.txt snippet", domain="example.com"
            ),
            WebSearchResultItem(
                rank=2,
                title="Unique LLM.txt",
                url="https://unique2.com",
                snippet="Unique LLM snippet",
                domain="unique2.com",
            ),
        ]

        regular_response = WebSearchResponse(
            query="test query", results=regular_results, total_results=2, search_time=0.05
        )

        llm_txt_response = WebSearchResponse(
            query="LLM.txt:example.com", results=llm_txt_results, total_results=2, search_time=0.03
        )

        with patch("aperag.views.web._search_with_jina_fallback", new_callable=AsyncMock) as mock_regular:
            with patch("aperag.views.web._search_llm_txt_discovery", new_callable=AsyncMock) as mock_llm:
                mock_regular.return_value = regular_response
                mock_llm.return_value = llm_txt_response

                request = WebSearchRequest(query="test query", search_llms_txt="example.com", max_results=10)

                response = await web_search_endpoint(request, self.mock_user)

                # Should have 3 unique URLs (duplicate removed)
                assert len(response.results) == 3
                assert response.total_results == 3

                # Verify URLs are unique
                urls = [result.url for result in response.results]
                assert len(set(urls)) == 3  # All URLs should be unique
                assert duplicate_url in urls
                assert "https://unique1.com" in urls
                assert "https://unique2.com" in urls
