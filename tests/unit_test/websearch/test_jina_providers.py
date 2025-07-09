"""
Tests for JINA Search and Reader providers.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from aperag.schema.view_models import WebReadResultItem, WebSearchResultItem
from aperag.websearch.reader.providers.jina_read_provider import JinaReaderProvider
from aperag.websearch.search.providers.jina_search_provider import JinaSearchProvider, SearchProviderError


class TestJinaSearchProvider:
    """Test JINA search provider."""

    def test_init_with_config(self):
        """Test provider initialization with config."""
        config = {"api_key": "test_key"}
        provider = JinaSearchProvider(config)

        assert provider.api_key == "test_key"
        assert provider.base_url == "https://s.jina.ai/"
        assert "Authorization" in provider.headers
        assert provider.headers["Authorization"] == "Bearer test_key"

    def test_init_without_api_key(self):
        """Test provider initialization without API key."""
        provider = JinaSearchProvider()

        # Should initialize without API key
        assert provider.api_key is None
        assert "Authorization" not in provider.headers

    def test_get_supported_engines(self):
        """Test getting supported search engines."""
        provider = JinaSearchProvider()
        engines = provider.get_supported_engines()

        assert isinstance(engines, list)
        assert "jina" in engines
        assert "google" in engines
        assert "bing" in engines

    def test_validate_search_engine(self):
        """Test search engine validation."""
        provider = JinaSearchProvider()

        assert provider.validate_search_engine("google") is True
        assert provider.validate_search_engine("jina") is True
        assert provider.validate_search_engine("bing") is True
        assert provider.validate_search_engine("yahoo") is False

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Test search with empty query."""
        provider = JinaSearchProvider({"api_key": "test_key"})

        with pytest.raises(SearchProviderError, match="Query cannot be empty"):
            await provider.search("")

    @pytest.mark.asyncio
    async def test_search_no_api_key(self):
        """Test search without API key."""
        provider = JinaSearchProvider()

        with pytest.raises(SearchProviderError, match="JINA API key is required"):
            await provider.search("test query")

    @pytest.mark.asyncio
    async def test_search_invalid_engine(self):
        """Test search with invalid search engine."""
        provider = JinaSearchProvider({"api_key": "test_key"})

        with pytest.raises(SearchProviderError, match="Unsupported search engine"):
            await provider.search("test query", search_engine="invalid_engine")

    @pytest.mark.asyncio
    @patch("aperag.websearch.search.providers.jina_search_provider.aiohttp.ClientSession")
    async def test_search_success(self, mock_session):
        """Test successful search."""
        # Mock response data
        mock_response_data = {
            "data": {
                "content": "Test search content with relevant information",
                "citations": [
                    {
                        "url": "https://example.com/result1",
                        "title": "Test Result 1",
                        "snippet": "This is the first test result",
                    },
                    {
                        "url": "https://example.com/result2",
                        "title": "Test Result 2",
                        "snippet": "This is the second test result",
                    },
                ],
            }
        }

        # Setup mock
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        mock_session_instance = AsyncMock()
        mock_session_instance.post = AsyncMock()
        mock_session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_instance.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaSearchProvider({"api_key": "test_key"})
        results = await provider.search("test query", max_results=5)

        assert isinstance(results, list)
        assert len(results) <= 5

        # If we have results, check their structure
        if results:
            result = results[0]
            assert isinstance(result, WebSearchResultItem)
            assert result.rank >= 1
            assert result.url
            assert result.title
            assert result.snippet

    @pytest.mark.asyncio
    @patch("aperag.websearch.search.providers.jina_search_provider.aiohttp.ClientSession")
    async def test_search_api_error(self, mock_session):
        """Test search with API error."""
        # Setup mock for error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")

        mock_session_instance = AsyncMock()
        mock_session_instance.post = AsyncMock()
        mock_session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_instance.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaSearchProvider({"api_key": "test_key"})

        with pytest.raises(SearchProviderError, match="JINA API returned status 400"):
            await provider.search("test query")


class TestJinaReaderProvider:
    """Test JINA reader provider."""

    def test_init_with_config(self):
        """Test provider initialization with config."""
        config = {"api_key": "test_key"}
        provider = JinaReaderProvider(config)

        assert provider.api_key == "test_key"
        assert provider.base_url == "https://r.jina.ai/"
        assert "Authorization" in provider.headers
        assert provider.headers["Authorization"] == "Bearer test_key"

    def test_init_without_api_key(self):
        """Test provider initialization without API key."""
        provider = JinaReaderProvider()

        # Should initialize without API key
        assert provider.api_key is None

    def test_validate_url(self):
        """Test URL validation."""
        provider = JinaReaderProvider()

        assert provider.validate_url("https://example.com") is True
        assert provider.validate_url("http://example.com") is True
        assert provider.validate_url("https://localhost") is False
        assert provider.validate_url("invalid_url") is False
        assert provider.validate_url("") is False

    @pytest.mark.asyncio
    async def test_read_empty_url(self):
        """Test reading with empty URL."""
        provider = JinaReaderProvider({"api_key": "test_key"})

        result = await provider.read("")
        assert result.status == "error"
        assert result.error_code == "INVALID_URL"

    @pytest.mark.asyncio
    async def test_read_invalid_url(self):
        """Test reading with invalid URL."""
        provider = JinaReaderProvider({"api_key": "test_key"})

        result = await provider.read("invalid_url")
        assert result.status == "error"
        assert result.error_code == "INVALID_URL_FORMAT"

    @pytest.mark.asyncio
    async def test_read_no_api_key(self):
        """Test reading without API key."""
        provider = JinaReaderProvider()

        result = await provider.read("https://example.com")
        assert result.status == "error"
        assert result.error_code == "MISSING_API_KEY"

    @pytest.mark.asyncio
    @patch("aperag.websearch.reader.providers.jina_read_provider.aiohttp.ClientSession")
    async def test_read_success(self, mock_session):
        """Test successful content reading."""
        # Mock response data
        mock_response_data = {
            "data": {
                "content": "# Test Article\n\nThis is test content from the webpage.",
                "title": "Test Article Title",
                "url": "https://example.com/article",
            }
        }

        # Setup mock
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        mock_session_instance = AsyncMock()
        mock_session_instance.post = AsyncMock()
        mock_session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_instance.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaReaderProvider({"api_key": "test_key"})
        result = await provider.read("https://example.com/article")

        assert result.status == "success"
        assert result.title == "Test Article Title"
        assert result.content == "# Test Article\n\nThis is test content from the webpage."
        assert result.url == "https://example.com/article"
        assert result.word_count > 0
        assert result.token_count > 0
        assert isinstance(result.extracted_at, datetime)

    @pytest.mark.asyncio
    @patch("aperag.websearch.reader.providers.jina_read_provider.aiohttp.ClientSession")
    async def test_read_api_error(self, mock_session):
        """Test reading with API error."""
        # Setup mock for error response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")

        mock_session_instance = AsyncMock()
        mock_session_instance.post = AsyncMock()
        mock_session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_instance.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaReaderProvider({"api_key": "test_key"})
        result = await provider.read("https://example.com/nonexistent")

        assert result.status == "error"
        assert result.error_code == "API_ERROR_404"
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_read_batch_empty_urls(self):
        """Test batch reading with empty URLs list."""
        provider = JinaReaderProvider({"api_key": "test_key"})

        results = await provider.read_batch([])
        assert results == []

    @pytest.mark.asyncio
    @patch("aperag.websearch.reader.providers.jina_read_provider.aiohttp.ClientSession")
    async def test_read_batch_success(self, mock_session):
        """Test successful batch reading."""
        # Mock response data
        mock_response_data = {
            "data": {"content": "Test content", "title": "Test Title", "url": "https://example.com/test"}
        }

        # Setup mock
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        mock_session_instance = AsyncMock()
        mock_session_instance.post = AsyncMock()
        mock_session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_instance.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaReaderProvider({"api_key": "test_key"})
        results = await provider.read_batch(["https://example.com/test1", "https://example.com/test2"])

        assert len(results) == 2
        for result in results:
            assert isinstance(result, WebReadResultItem)
            assert result.status == "success"
