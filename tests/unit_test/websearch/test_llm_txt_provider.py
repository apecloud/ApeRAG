"""
Simplified LLM.txt Search Provider Tests

Tests the core functionality of LLMTxtSearchProvider including:
- Pattern-based discovery
- Direct URL support  
- Parameter validation
- Error handling
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from aperag.schema.view_models import WebReadResponse, WebReadResultItem
from aperag.websearch.search.providers.llm_txt_search_provider import LLMTxtSearchProvider


class TestLLMTxtSearchProvider:
    """Test core LLM.txt search provider functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for testing."""
        return LLMTxtSearchProvider()

    @pytest.fixture
    def mock_success_response(self):
        """Mock successful read response."""
        return WebReadResponse(
            results=[WebReadResultItem(
                url="https://example.com/llms.txt",
                status="success",
                title="Example LLM.txt",
                content="# Documentation\n\nThis is LLM-optimized content.",
                extracted_at=datetime.now(),
                word_count=50
            )],
            total_urls=1,
            successful=1,
            failed=0
        )

    @pytest.fixture  
    def mock_failed_response(self):
        """Mock failed read response."""
        return WebReadResponse(
            results=[WebReadResultItem(
                url="https://example.com/llms.txt",
                status="error",
                error="404 Not Found"
            )],
            total_urls=1,
            successful=0,
            failed=1
        )

    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.supported_engines == ["llm_txt"]
        assert len(provider.LLM_TXT_PATTERNS) == 8  # Simplified patterns
        assert "/llms.txt" in provider.LLM_TXT_PATTERNS
        assert "/.well-known/llms.txt" in provider.LLM_TXT_PATTERNS

    @pytest.mark.asyncio
    async def test_parameter_validation(self, provider):
        """Test parameter validation."""
        # Invalid max_results
        with pytest.raises(ValueError, match="max_results must be positive"):
            await provider.search(query="test", source="example.com", max_results=0)
            
        with pytest.raises(ValueError, match="max_results cannot exceed 100"):
            await provider.search(query="test", source="example.com", max_results=101)
            
        # Invalid timeout
        with pytest.raises(ValueError, match="timeout must be positive"):
            await provider.search(query="test", source="example.com", timeout=0)

    @pytest.mark.asyncio
    async def test_no_source_provided(self, provider):
        """Test behavior when no source is provided."""
        results = await provider.search(query="test")
        assert results == []

    @pytest.mark.asyncio
    async def test_direct_llms_txt_url(self, provider, mock_success_response):
        """Test direct LLM.txt URL detection and processing."""
        # Mock reader service
        mock_reader = AsyncMock()
        mock_reader.read.return_value = mock_success_response
        provider.reader_service = mock_reader
        
        # Test direct URL
        direct_url = "https://modelcontextprotocol.io/llms-full.txt"
        results = await provider.search(
            query="test",
            source=direct_url,
            max_results=5
        )
        
        assert len(results) == 1
        assert results[0].url == direct_url
        assert results[0].rank == 1
        assert results[0].domain == "modelcontextprotocol.io"

    @pytest.mark.asyncio
    async def test_domain_pattern_discovery(self, provider, mock_success_response):
        """Test pattern-based discovery from domain."""
        # Mock reader service
        mock_reader = AsyncMock()
        mock_reader.read.return_value = mock_success_response
        provider.reader_service = mock_reader
        
        # Test domain
        results = await provider.search(
            query="test", 
            source="example.com",
            max_results=5
        )
        
        assert len(results) == 1
        assert results[0].domain == "example.com"
        assert results[0].rank == 1

    @pytest.mark.asyncio
    async def test_pattern_failure_fallback(self, provider, mock_failed_response):
        """Test that provider tries multiple patterns when earlier ones fail."""
        # Mock reader service to fail
        mock_reader = AsyncMock()
        mock_reader.read.return_value = mock_failed_response
        provider.reader_service = mock_reader
        
        results = await provider.search(
            query="test",
            source="example.com", 
            max_results=5
        )
        
        # Should return empty results when all patterns fail
        assert results == []
        
        # Should have tried multiple patterns (reader called multiple times)
        assert mock_reader.read.call_count > 1

    @pytest.mark.asyncio
    async def test_invalid_source_url(self, provider):
        """Test handling of invalid source URLs."""
        results = await provider.search(
            query="test",
            source="not-a-valid-url",
            max_results=5
        )
        
        # Should return empty results for invalid domain
        assert results == []

    @pytest.mark.asyncio  
    async def test_snippet_creation(self, provider):
        """Test snippet creation from content."""
        # Test basic snippet creation
        content = "This is a test content with some information about the documentation."
        snippet = provider._create_snippet_from_content(content, max_length=50)
        
        assert len(snippet) <= 53  # 50 + "..."
        assert snippet.endswith("...")
        
        # Test short content (no truncation)
        short_content = "Short content"
        snippet = provider._create_snippet_from_content(short_content)
        assert snippet == "Short content"
        
        # Test markdown removal
        markdown_content = "# Title\n\nThis is **bold** and *italic* text with `code`."
        snippet = provider._create_snippet_from_content(markdown_content)
        assert "#" not in snippet
        assert "**" not in snippet
        assert "`" not in snippet

    @pytest.mark.asyncio
    async def test_url_detection_helper(self, provider):
        """Test LLM.txt URL detection helper."""
        # Valid LLM.txt URLs
        assert provider._is_llms_txt_url("https://example.com/llms.txt") == True
        assert provider._is_llms_txt_url("http://example.com/llms-full.txt") == True
        assert provider._is_llms_txt_url("https://sub.example.com/path/llms.txt") == True
        
        # Invalid URLs
        assert provider._is_llms_txt_url("https://example.com/docs.txt") == False
        assert provider._is_llms_txt_url("example.com/llms.txt") == False  # No protocol
        assert provider._is_llms_txt_url("") == False
        assert provider._is_llms_txt_url(None) == False

    @pytest.mark.asyncio
    async def test_cleanup(self, provider):
        """Test provider cleanup."""
        # Mock reader service with close method
        mock_reader = AsyncMock()
        mock_reader.close = AsyncMock()
        provider.reader_service = mock_reader
        
        await provider.close()
        
        mock_reader.close.assert_called_once()

    def test_supported_engines(self, provider):
        """Test supported engines method."""
        engines = provider.get_supported_engines()
        assert engines == ["llm_txt"]
        # Ensure it returns a copy
        engines.append("test")
        assert provider.get_supported_engines() == ["llm_txt"] 