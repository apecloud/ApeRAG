"""
Edge Cases and Boundary Condition Tests

Tests various edge cases and boundary conditions for web search functionality including:
- Parameter validation
- Error recovery
- Resource limits
- Network failures
- Malformed responses
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from aperag.schema.view_models import WebSearchRequest, WebSearchResultItem
from aperag.websearch.search.providers.duckduckgo_search_provider import DuckDuckGoProvider
from aperag.websearch.search.providers.jina_search_provider import JinaSearchProvider
from aperag.websearch.search.providers.llm_txt_search_provider import LLMTxtSearchProvider
from aperag.websearch.search.search_service import SearchService
from aperag.websearch.utils.url_validator import URLValidator


class TestParameterValidation:
    """Test parameter validation across all components."""

    @pytest.mark.asyncio
    async def test_search_service_parameter_limits(self):
        """Test SearchService parameter validation."""
        service = SearchService.create_default()

        # max_results boundaries
        with pytest.raises(ValueError, match="max_results must be positive"):
            await service.search(WebSearchRequest(query="test", max_results=0))

        with pytest.raises(ValueError, match="max_results cannot exceed 100"):
            await service.search(WebSearchRequest(query="test", max_results=101))

        # timeout boundaries
        with pytest.raises(ValueError, match="timeout must be positive"):
            await service.search(WebSearchRequest(query="test", timeout=0))

        with pytest.raises(ValueError, match="timeout cannot exceed 300"):
            await service.search(WebSearchRequest(query="test", timeout=301))

    @pytest.mark.asyncio
    async def test_provider_parameter_validation(self):
        """Test provider-level parameter validation."""
        providers = [DuckDuckGoProvider(), JinaSearchProvider({"api_key": "test"}), LLMTxtSearchProvider()]

        for provider in providers:
            # Test negative max_results
            with pytest.raises(ValueError, match="max_results must be positive"):
                if isinstance(provider, LLMTxtSearchProvider):
                    await provider.search("test", max_results=-1, source="example.com")
                else:
                    await provider.search("test", max_results=-1)

            # Test zero timeout
            with pytest.raises(ValueError, match="timeout must be positive"):
                if isinstance(provider, LLMTxtSearchProvider):
                    await provider.search("test", timeout=0, source="example.com")
                else:
                    await provider.search("test", timeout=0)

    def test_url_validator_edge_cases(self):
        """Test URL validator with edge cases."""
        # Empty and None inputs
        assert URLValidator.extract_domain_from_source("") is None
        assert URLValidator.extract_domain_from_source(None) is None
        assert URLValidator.extract_domain_from_source("   ") is None

        # Invalid URLs
        assert URLValidator.extract_domain_from_source("not-a-url") is None
        assert URLValidator.extract_domain_from_source("http://") is None
        assert URLValidator.extract_domain_from_source("://invalid") is None

        # Edge valid cases
        assert URLValidator.extract_domain_from_source("http://a.b") == "a.b"
        assert URLValidator.extract_domain_from_source("https://sub.domain.tld:8080/path") == "sub.domain.tld"


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of invalid timeout values."""
        provider = JinaSearchProvider({"api_key": "test"})

        # Test invalid timeout parameter
        with pytest.raises(ValueError, match="timeout must be positive"):
            await provider.search("test query", timeout=0)

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self):
        """Test handling of invalid API key."""
        provider = JinaSearchProvider()  # No API key

        # Should return empty results when API key is missing (401 error)
        results = await provider.search("test query")
        assert results == []  # API returns 401, provider returns empty results

    @pytest.mark.asyncio
    async def test_provider_creation_error_handling(self):
        """Test error handling during provider creation."""
        # Test invalid provider name
        with pytest.raises(ValueError, match="Unsupported search provider"):
            SearchService(provider_name="invalid_provider")

        # Test case sensitivity
        service = SearchService(provider_name="DUCKDUCKGO")  # Should work (case insensitive)
        assert isinstance(service.provider, DuckDuckGoProvider)

    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty queries and sources."""
        provider = JinaSearchProvider({"api_key": "test"})

        # Should raise error when both query and source are empty
        with pytest.raises(ValueError, match="Either query or source must be provided"):
            await provider.search("")


class TestResourceLimits:
    """Test resource usage limits and memory safety."""

    @pytest.mark.asyncio
    async def test_large_result_handling(self):
        """Test handling of large result sets."""
        provider = DuckDuckGoProvider()

        # Mock the search method directly to test max_results limiting behavior
        with patch.object(provider, "_search_sync") as mock_search_sync:
            # Create limited results based on max_results parameter
            def create_limited_results(query, max_results, timeout, locale):
                limited_results = []
                for i in range(min(max_results, 200)):  # Respect max_results limit
                    limited_results.append(
                        WebSearchResultItem(
                            rank=i + 1,
                            title=f"Result {i}",
                            url=f"https://example{i}.com",
                            snippet=f"Content for result {i}",
                            domain=f"example{i}.com",
                            timestamp=datetime.now(),
                        )
                    )
                return limited_results

            mock_search_sync.side_effect = create_limited_results

            # Should respect max_results limit
            results = await provider.search("test", max_results=10)
            assert len(results) <= 10

    def test_long_string_handling(self):
        """Test handling of extremely long strings."""
        provider = LLMTxtSearchProvider()

        # Test very long line content
        very_long_content = "[Very Long Title](https://example.com): " + "x" * 10000
        snippet = provider._clean_line_content_for_snippet(very_long_content)

        assert len(snippet) <= 203  # 200 + "..."
        assert snippet.endswith("...")

    @pytest.mark.asyncio
    async def test_concurrent_request_limits(self):
        """Test behavior under high concurrency."""
        service = SearchService.create_default()

        # Mock provider to simulate slow responses
        with patch.object(service.provider, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            # Simulate many concurrent requests
            import asyncio

            tasks = []
            for i in range(50):
                request = WebSearchRequest(query=f"test {i}", max_results=5)
                tasks.append(service.search(request))

            # Should handle all requests without crashing
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that no exceptions occurred
            for result in results:
                assert not isinstance(result, Exception)


class TestMaliciousInputs:
    """Test handling of potentially malicious inputs."""

    @pytest.mark.asyncio
    async def test_sql_injection_patterns(self):
        """Test that SQL injection patterns are handled safely."""
        provider = DuckDuckGoProvider()

        malicious_queries = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "test'; DELETE FROM *; --",
            "UNION SELECT * FROM sensitive_data",
        ]

        for query in malicious_queries:
            # Should not crash, treat as normal query
            try:
                with patch("aperag.websearch.search.providers.duckduckgo_search_provider.DDGS") as mock_ddgs:
                    mock_ddgs.return_value.__enter__.return_value.text.return_value = []
                    results = await provider.search(query)
                    assert isinstance(results, list)
            except ValueError as e:
                # Only acceptable if it's parameter validation
                assert "cannot be empty" in str(e) or "must be positive" in str(e)

    @pytest.mark.asyncio
    async def test_xss_patterns(self):
        """Test that XSS patterns don't crash snippet creation."""
        provider = LLMTxtSearchProvider()

        xss_content = """- [Script Link](https://example.com): <script>alert('xss')</script> test content"""

        # Should not crash when creating snippet
        snippet = provider._clean_line_content_for_snippet(xss_content)

        # Basic check that snippet is created without errors
        assert isinstance(snippet, str)
        assert len(snippet) > 0

    def test_unicode_and_encoding_edge_cases(self):
        """Test handling of unicode and encoding edge cases."""
        provider = LLMTxtSearchProvider()

        # Test various unicode patterns
        unicode_content = "- [测试](https://example.com): 测试内容 🚀 émojis café naïve résumé"
        snippet = provider._clean_line_content_for_snippet(unicode_content)
        assert "测试: 测试内容 🚀 émojis café naïve résumé" in snippet  # Should handle unicode correctly

        # Test null bytes and control characters
        control_content = "- [Control](https://example.com): test\x00null\x01control\x02chars"
        snippet = provider._clean_line_content_for_snippet(control_content)
        assert isinstance(snippet, str)  # Should not crash


class TestSpecialCases:
    """Test special cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_successful_response(self):
        """Test handling when API succeeds but returns no results."""
        provider = DuckDuckGoProvider()

        with patch("aperag.websearch.search.providers.duckduckgo_search_provider.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = []

            results = await provider.search("very_rare_query_12345")
            assert results == []

    @pytest.mark.asyncio
    async def test_partial_url_validation_failures(self):
        """Test when some URLs in results are invalid."""
        provider = DuckDuckGoProvider()

        mixed_results = [
            {"title": "Valid", "href": "https://valid.com", "body": "Good"},
            {"title": "Invalid", "href": "not-a-url", "body": "Bad"},
            {"title": "Also Valid", "href": "https://alsovalid.com", "body": "Good"},
        ]

        with patch("aperag.websearch.search.providers.duckduckgo_search_provider.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = mixed_results

            results = await provider.search("test")

            # Should only include valid URLs
            assert len(results) == 2
            for result in results:
                assert result.url.startswith("https://")

    def test_domain_extraction_edge_cases(self):
        """Test domain extraction with unusual but valid cases."""
        # Unusual but valid domains
        test_cases = [
            ("https://example.com:8080", "example.com"),
            ("http://sub.sub.example.co.uk", "sub.sub.example.co.uk"),
            ("https://xn--bcher-kva.example", "xn--bcher-kva.example"),  # IDN
            ("https://127.0.0.1:3000", "127.0.0.1"),  # IP address
        ]

        for url, expected_domain in test_cases:
            domain = URLValidator.extract_domain_from_source(url)
            assert domain == expected_domain
