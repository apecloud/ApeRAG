"""
Unit tests for YouComSearchProvider

Tests cover provider initialization, parameter validation, request building,
response parsing, domain filtering, and error handling. All network
interactions are mocked; no real requests are made.
"""

import asyncio
from unittest.mock import patch

import pytest

from aperag.websearch.search.providers.youcom_search_provider import YouComSearchProvider
from aperag.websearch.search.search_service import SearchService

PROVIDER_MODULE = "aperag.websearch.search.providers.youcom_search_provider"


def _fake_session_factory(status=200, payload=None, exc=None):
    """Build a fake aiohttp.ClientSession replacement capturing the request."""
    captured = {}

    class _FakeResponse:
        def __init__(self):
            self.status = status

        async def json(self):
            if isinstance(exc, Exception) and isinstance(exc, ValueError):
                raise exc
            return payload

        async def text(self):
            return ""

        async def __aenter__(self):
            if isinstance(exc, Exception) and not isinstance(exc, ValueError):
                raise exc
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    class _FakeSession:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def post(self, url, headers=None, json=None):
            captured.update({"url": url, "headers": headers, "json": json})
            return _FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    return _FakeSession, captured


SAMPLE_PAYLOAD = {
    "results": {
        "web": [
            {
                "url": "https://example.com/article",
                "title": "Example Article",
                "description": "A brief description",
                "snippets": ["first snippet"],
            },
            {
                "url": "https://another.com/page",
                "title": "Another Page",
                "snippets": ["snippet one", "snippet two"],
            },
            {
                "url": "not-a-valid-url",
                "title": "Broken Item",
                "description": "Should be skipped",
            },
        ]
    }
}


class TestYouComSearchProviderInit:
    """Test provider initialization and configuration."""

    def test_create_with_api_key(self):
        provider = YouComSearchProvider({"api_key": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.headers["X-API-Key"] == "test-key"

    def test_create_without_api_key(self):
        provider = YouComSearchProvider()
        assert provider.api_key is None
        assert "X-API-Key" not in provider.headers

    def test_supported_engines(self):
        provider = YouComSearchProvider({"api_key": "test-key"})
        engines = provider.get_supported_engines()
        assert "youcom" in engines
        assert "you_com" in engines


class TestYouComSearchProviderValidation:
    """Test parameter validation (shared provider contract)."""

    @pytest.mark.asyncio
    async def test_empty_query_and_source(self):
        provider = YouComSearchProvider({"api_key": "test-key"})
        with pytest.raises(ValueError, match="Either query or source must be provided"):
            await provider.search("")

    @pytest.mark.asyncio
    async def test_invalid_max_results(self):
        provider = YouComSearchProvider({"api_key": "test-key"})
        with pytest.raises(ValueError, match="max_results must be positive"):
            await provider.search("test", max_results=-1)
        with pytest.raises(ValueError, match="max_results cannot exceed 100"):
            await provider.search("test", max_results=101)

    @pytest.mark.asyncio
    async def test_invalid_timeout(self):
        provider = YouComSearchProvider({"api_key": "test-key"})
        with pytest.raises(ValueError, match="timeout must be positive"):
            await provider.search("test", timeout=0)

    @pytest.mark.asyncio
    async def test_invalid_source_without_query(self):
        provider = YouComSearchProvider({"api_key": "test-key"})
        with pytest.raises(ValueError, match="Invalid source domain and no query provided"):
            await provider.search("", source="not-a-url")


class TestYouComSearchProviderSearch:
    """Test search request building and response parsing."""

    @pytest.mark.asyncio
    async def test_search_parses_web_results(self):
        _FakeSession, captured = _fake_session_factory(payload=SAMPLE_PAYLOAD)
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query", max_results=5)

        assert len(results) == 2  # invalid URL item is skipped
        assert results[0].url == "https://example.com/article"
        assert results[0].title == "Example Article"
        assert results[0].snippet == "A brief description"
        assert results[0].domain == "example.com"
        assert results[0].rank == 1
        assert results[1].snippet == "snippet one snippet two"

        # Request contract
        assert captured["url"] == "https://ydc-index.io/v1/search"
        assert captured["headers"]["X-API-Key"] == "test-key"
        assert captured["json"]["query"] == "test query"
        assert captured["json"]["count"] == 5
        assert captured["json"]["language"] == "en-US"

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        _FakeSession, captured = _fake_session_factory(payload=SAMPLE_PAYLOAD)
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query", max_results=1)

        assert len(results) == 1
        assert captured["json"]["count"] == 1

    @pytest.mark.asyncio
    async def test_search_with_source_restricts_domain(self):
        _FakeSession, captured = _fake_session_factory(payload=SAMPLE_PAYLOAD)
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query", source="example.com")

        # Native include_domains allowlist is used and results are filtered
        assert captured["json"]["include_domains"] == ["example.com"]
        assert captured["json"]["query"] == "site:example.com test query"
        assert all(r.domain == "example.com" for r in results)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_source_only_browsing(self):
        _FakeSession, captured = _fake_session_factory(payload=SAMPLE_PAYLOAD)
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("", source="example.com")

        assert captured["json"]["query"] == "site:example.com"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_locale_underscore_normalized(self):
        _FakeSession, captured = _fake_session_factory(payload=SAMPLE_PAYLOAD)
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            await provider.search("测试查询", locale="zh_CN")

        assert captured["json"]["language"] == "zh-CN"


class TestYouComSearchProviderErrorHandling:
    """Test fail-safe error handling (returns empty results)."""

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        _FakeSession, _ = _fake_session_factory(status=401, payload={})
        provider = YouComSearchProvider()  # No API key: API returns 401

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        _FakeSession, _ = _fake_session_factory(exc=asyncio.TimeoutError())
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query", timeout=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_generic_error_returns_empty(self):
        _FakeSession, _ = _fake_session_factory(exc=RuntimeError("connection reset"))
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_malformed_response_returns_empty(self):
        _FakeSession, _ = _fake_session_factory(payload={"unexpected": "shape"})
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self):
        _FakeSession, _ = _fake_session_factory(payload=None, exc=ValueError("invalid json"))
        provider = YouComSearchProvider({"api_key": "test-key"})

        with patch(f"{PROVIDER_MODULE}.aiohttp.ClientSession", _FakeSession):
            results = await provider.search("test query")

        assert results == []


class TestYouComSearchProviderRegistry:
    """Test SearchService registration."""

    def test_registry_creates_youcom_provider(self):
        service = SearchService.create_with_provider("youcom", {"api_key": "test-key"})
        assert isinstance(service.provider, YouComSearchProvider)

    def test_registry_alias(self):
        service = SearchService.create_with_provider("you_com", {"api_key": "test-key"})
        assert isinstance(service.provider, YouComSearchProvider)

    def test_registry_case_insensitive(self):
        service = SearchService(provider_name="YOUCOM", provider_config={"api_key": "test-key"})
        assert isinstance(service.provider, YouComSearchProvider)
