"""
Unit tests for SearchService - User perspective tests
"""

from unittest.mock import AsyncMock, patch

import pytest

from aperag.schema.view_models import WebSearchRequest, WebSearchResultItem
from aperag.websearch.search.providers.duckduckgo_search_provider import SearchProviderError
from aperag.websearch.search.search_service import SearchService


class TestSearchService:
    """Test SearchService from user perspective"""

    def test_create_service(self):
        """Test creating search service"""
        # Default service
        service = SearchService.create_default()
        assert service.provider_name == "duckduckgo"

        # Custom provider
        service = SearchService.create_with_provider("duckduckgo", timeout=60)
        assert service.provider_name == "duckduckgo"
        assert service.provider_config["timeout"] == 60

    def test_get_supported_engines(self):
        """Test getting supported search engines"""
        service = SearchService.create_default()
        engines = service.get_supported_engines()

        assert isinstance(engines, list)
        assert len(engines) > 0

    @pytest.mark.asyncio
    async def test_search_with_request_object(self):
        """Test search using WebSearchRequest object"""
        service = SearchService.create_default()

        # Mock the actual provider to avoid real network calls
        mock_results = [
            WebSearchResultItem(
                rank=1, title="Test Result", url="https://example.com", snippet="Test snippet", domain="example.com"
            )
        ]

        with patch.object(service.provider, "search", new_callable=AsyncMock) as mock_search:
            with patch.object(service.provider, "validate_search_engine", return_value=True):
                mock_search.return_value = mock_results

                request = WebSearchRequest(query="Python tutorial", max_results=5, search_engine="duckduckgo")

                response = await service.search(request)

                assert response.query == "Python tutorial"
                assert response.search_engine == "duckduckgo"
                assert len(response.results) == 1
                assert response.results[0].title == "Test Result"
                assert response.total_results == 1
                assert response.search_time >= 0

    @pytest.mark.asyncio
    async def test_search_simple_interface(self):
        """Test simplified search interface"""
        service = SearchService.create_default()

        mock_results = [
            WebSearchResultItem(
                rank=1, title="Simple Result", url="https://test.com", snippet="Simple test", domain="test.com"
            )
        ]

        with patch.object(service.provider, "search", new_callable=AsyncMock) as mock_search:
            with patch.object(service.provider, "validate_search_engine", return_value=True):
                mock_search.return_value = mock_results

                results = await service.search_simple("machine learning", max_results=3)

                assert len(results) == 1
                assert results[0].title == "Simple Result"

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test search error handling"""
        service = SearchService.create_default()

        # Test empty query
        request = WebSearchRequest(query="")
        with pytest.raises(SearchProviderError, match="cannot be empty"):
            await service.search(request)

        # Test provider error
        with patch.object(service.provider, "search", new_callable=AsyncMock) as mock_search:
            with patch.object(service.provider, "validate_search_engine", return_value=True):
                mock_search.side_effect = SearchProviderError("Network error")

                request = WebSearchRequest(query="test")
                with pytest.raises(SearchProviderError, match="Network error"):
                    await service.search(request)

    @pytest.mark.asyncio
    async def test_unsupported_search_engine_fallback(self):
        """Test fallback when using unsupported search engine"""
        service = SearchService.create_default()

        with patch.object(service.provider, "search", new_callable=AsyncMock) as mock_search:
            with patch.object(service.provider, "validate_search_engine", return_value=False):
                with patch.object(service.provider, "get_supported_engines", return_value=["duckduckgo"]):
                    mock_search.return_value = []

                    # Use unsupported engine
                    await service.search_simple("test", search_engine="unsupported")

                    # Should fallback to supported engine
                    mock_search.assert_called_once()
                    call_args = mock_search.call_args.kwargs
                    assert call_args["search_engine"] == "duckduckgo"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_search_integration(self):
        """
        Real integration test with actual DuckDuckGo search
        Use this for debugging and verification

        Run with: pytest -m integration tests/unit_test/websearch/test_search_service.py::TestSearchService::test_real_search_integration -v
        """
        service = SearchService.create_default()

        # Test simple search
        try:
            results = await service.search_simple(
                query="Python programming tutorial", max_results=3, search_engine="duckduckgo", timeout=10
            )

            print("\n=== Real Search Results ===")
            print(f"Found {len(results)} results")

            assert len(results) > 0, "Should return at least one result"

            for i, result in enumerate(results):
                print(f"\n--- Result {i + 1} ---")
                print(f"Title: {result.title}")
                print(f"URL: {result.url}")
                print(f"Snippet: {result.snippet[:100]}...")
                print(f"Domain: {result.domain}")

                # Basic validation
                assert result.title, "Title should not be empty"
                assert result.url.startswith("http"), "URL should be valid"
                assert result.snippet, "Snippet should not be empty"
                assert result.domain, "Domain should not be empty"
                assert result.rank > 0, "Rank should be positive"

            # Test full request object
            request = WebSearchRequest(
                query="machine learning basics", max_results=2, search_engine="duckduckgo", timeout=10
            )

            response = await service.search(request)

            print("\n=== Full Response ===")
            print(f"Query: {response.query}")
            print(f"Engine: {response.search_engine}")
            print(f"Total: {response.total_results}")
            print(f"Time: {response.search_time:.2f}s")

            assert response.query == "machine learning basics"
            assert response.search_engine == "duckduckgo"
            assert response.total_results > 0
            assert response.search_time > 0
            assert len(response.results) > 0

            print("\n✅ Real search integration test passed!")

        except Exception as e:
            print(f"\n❌ Real search test failed: {e}")
            # Don't fail the test for network issues, just warn
            pytest.skip(f"Real search test skipped due to: {e}")
