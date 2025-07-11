"""
Unit tests for the new parallel search architecture.

Tests the web_search_view function's ability to handle:
- Regular search only
- LLM.txt discovery only
- Site-specific search
- Combined parallel searches
- Error handling
"""

from unittest.mock import AsyncMock, patch

import pytest

from aperag.schema.view_models import WebSearchRequest, WebSearchResultItem
from aperag.views.web import web_search_view


class TestParallelSearchArchitecture:
    """Test the new parallel search architecture in web_search_view."""

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

        with patch("aperag.views.web.SearchService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Mock the search response
            mock_response = AsyncMock()
            mock_response.results = mock_results
            mock_service.search.return_value = mock_response

            # Test regular search only
            request = WebSearchRequest(query="test query", max_results=5)

            response = await web_search_view(request)

            # Verify response structure
            assert response.query == "test query"
            assert len(response.results) == 2
            assert response.search_engine == "parallel(1 sources)"
            assert response.results[0].title == "Test Result 1"
            assert response.results[1].title == "Test Result 2"

            # Verify that only one SearchService was created (regular search)
            assert mock_service_class.call_count == 1
            mock_service.search.assert_called_once()

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

        with patch("aperag.views.web.SearchService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            mock_response = AsyncMock()
            mock_response.results = mock_results
            mock_service.search.return_value = mock_response

            request = WebSearchRequest(search_llms_txt="example.com", max_results=5)

            response = await web_search_view(request)

            # Verify response
            assert response.query == "LLM.txt:example.com"
            assert len(response.results) == 1
            assert response.search_engine == "parallel(1 sources)"
            assert response.results[0].url == "https://example.com/llms.txt"

            # Verify SearchService was called for LLM.txt provider
            assert mock_service_class.call_count == 1
            call_args = mock_service_class.call_args
            assert call_args[1]["provider_name"] == "llm_txt"

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

        with patch("aperag.views.web.SearchService") as mock_service_class:
            # Create separate mock instances for each service
            mock_services = [AsyncMock(), AsyncMock()]
            mock_service_class.side_effect = mock_services

            # Configure responses
            mock_response_1 = AsyncMock()
            mock_response_1.results = regular_results
            mock_services[0].search.return_value = mock_response_1

            mock_response_2 = AsyncMock()
            mock_response_2.results = llm_txt_results
            mock_services[1].search.return_value = mock_response_2

            request = WebSearchRequest(query="test query", search_llms_txt="example.com", max_results=5)

            response = await web_search_view(request)

            # Verify combined response
            assert response.query == "test query + LLM.txt:example.com"
            assert len(response.results) == 2  # Merged results
            assert response.search_engine == "parallel(2 sources)"

            # Verify both SearchServices were created
            assert mock_service_class.call_count == 2

            # Check that both services were called
            mock_services[0].search.assert_called_once()
            mock_services[1].search.assert_called_once()

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

        with patch("aperag.views.web.SearchService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            mock_response = AsyncMock()
            mock_response.results = mock_results
            mock_service.search.return_value = mock_response

            request = WebSearchRequest(query="documentation", source="github.com", max_results=3)

            response = await web_search_view(request)

            # Verify site-specific search
            assert response.query == "documentation"
            assert len(response.results) == 1
            assert response.results[0].domain == "github.com"

            # Verify SearchService was called with source
            mock_service.search.assert_called_once()
            call_args = mock_service.search.call_args[0][0]  # WebSearchRequest
            assert call_args.source == "github.com"

    @pytest.mark.asyncio
    async def test_error_handling_no_params(self):
        """Test error handling when neither query nor search_llms_txt provided."""
        request = WebSearchRequest(max_results=5)

        with pytest.raises(Exception) as exc_info:
            await web_search_view(request)

        assert "At least one search type is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_handling_empty_query(self):
        """Test error handling with empty query string."""
        request = WebSearchRequest(
            query="   ",  # Only whitespace
            max_results=5,
        )

        with pytest.raises(Exception) as exc_info:
            await web_search_view(request)

        assert "At least one search type is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_failure_handling(self):
        """Test handling when searches fail."""
        with patch("aperag.views.web.SearchService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Make search raise an exception
            mock_service.search.side_effect = Exception("Network error")

            request = WebSearchRequest(query="test query", max_results=5)

            with pytest.raises(Exception) as exc_info:
                await web_search_view(request)

            assert "All searches failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_search_failure(self):
        """Test handling when one search succeeds and another fails."""
        regular_results = [
            WebSearchResultItem(
                rank=1, title="Success Result", url="https://success.com", snippet="Success", domain="success.com"
            )
        ]

        with patch("aperag.views.web.SearchService") as mock_service_class:
            # Create separate mock instances
            mock_services = [AsyncMock(), AsyncMock()]
            mock_service_class.side_effect = mock_services

            # First service succeeds
            mock_response = AsyncMock()
            mock_response.results = regular_results
            mock_services[0].search.return_value = mock_response

            # Second service fails
            mock_services[1].search.side_effect = Exception("LLM.txt error")

            request = WebSearchRequest(query="test query", search_llms_txt="example.com", max_results=5)

            response = await web_search_view(request)

            # Should return successful results despite partial failure
            assert len(response.results) == 1
            assert response.results[0].title == "Success Result"
            assert response.search_engine == "parallel(1 sources)"

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

        with patch("aperag.views.web.SearchService") as mock_service_class:
            mock_services = [AsyncMock(), AsyncMock()]
            mock_service_class.side_effect = mock_services

            # Configure responses
            mock_response_1 = AsyncMock()
            mock_response_1.results = regular_results
            mock_services[0].search.return_value = mock_response_1

            mock_response_2 = AsyncMock()
            mock_response_2.results = llm_txt_results
            mock_services[1].search.return_value = mock_response_2

            request = WebSearchRequest(query="test query", search_llms_txt="example.com", max_results=10)

            response = await web_search_view(request)

            # Should have 3 unique URLs (duplicate removed)
            assert len(response.results) == 3

            urls = [result.url for result in response.results]
            assert len(set(urls)) == 3  # All URLs should be unique
            assert duplicate_url in urls
            assert "https://unique1.com" in urls
            assert "https://unique2.com" in urls
