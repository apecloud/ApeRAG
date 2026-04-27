from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aperag.cache.application import (
    ApplicationCache,
    ApplicationCachePolicy,
    NoopApplicationCacheBackend,
    SyncApplicationCache,
    SyncNoopApplicationCacheBackend,
)


class MemoryAsyncCache(ApplicationCache):
    def __init__(self):
        super().__init__(
            backend=NoopApplicationCacheBackend(),
            default_policy=ApplicationCachePolicy(namespace="test", ttl_seconds=60),
        )
        self.store = {}

    async def get_or_compute(self, *, namespace, key_data, compute, policy=None, should_cache=None):
        key = (namespace, repr(key_data))
        if key not in self.store:
            value = await compute()
            if should_cache is None or should_cache(value):
                self.store[key] = value
            return value
        return self.store[key]


class MemorySyncCache(SyncApplicationCache):
    def __init__(self):
        super().__init__(
            backend=SyncNoopApplicationCacheBackend(),
            default_policy=ApplicationCachePolicy(namespace="test", ttl_seconds=60),
        )
        self.store = {}

    def get_or_compute(self, *, namespace, key_data, compute, policy=None, should_cache=None):
        key = (namespace, repr(key_data))
        if key not in self.store:
            value = compute()
            if should_cache is None or should_cache(value):
                self.store[key] = value
            return value
        return self.store[key]

    def get_many_or_compute_missing(
        self, *, namespace, items, key_data_for_item, compute_missing, policy=None, should_cache=None
    ):
        values = []
        missing = []
        computed_values = {}
        for item in items:
            key = (namespace, repr(key_data_for_item(item)))
            if key in self.store:
                values.append(self.store[key])
            else:
                values.append(None)
                missing.append(item)
        if missing:
            computed = compute_missing(missing)
            for item, value in computed.items():
                computed_values[item] = value
                key = (namespace, repr(key_data_for_item(item)))
                if should_cache is None or should_cache(value):
                    self.store[key] = value
        return [
            self.store[(namespace, repr(key_data_for_item(item)))]
            if value is None and (namespace, repr(key_data_for_item(item))) in self.store
            else computed_values.get(item, value)
            for item, value in zip(items, values)
        ]


def test_embedding_adapter_caches_per_input_and_disables_litellm_cache():
    from aperag.llm.embed import embedding_service as module
    from aperag.llm.embed.embedding_service import EmbeddingService

    cache = MemorySyncCache()
    service = EmbeddingService("openai", "text-embedding-3-small", "https://example.invalid", "sk", 8)
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs["input"])
        return {"data": [{"embedding": [float(len(text))]} for text in kwargs["input"]]}

    with (
        patch.object(module, "get_sync_application_cache", return_value=cache),
        patch.object(module.litellm, "embedding", side_effect=fake_embedding) as mocked,
    ):
        first = service.embed_documents(["hot", "cold"])
        second = service.embed_documents(["hot", "warm"])

    assert first == [[3.0], [4.0]]
    assert second == [[3.0], [4.0]]
    assert calls == [["hot", "cold"], ["warm"]]
    assert mocked.call_args.kwargs["caching"] is False


@pytest.mark.asyncio
async def test_completion_adapter_caches_non_stream_and_disables_litellm_cache():
    from aperag.llm.completion import completion_service as module
    from aperag.llm.completion.completion_service import CompletionService

    cache = MemoryAsyncCache()
    service = CompletionService("openai", "gpt-test", "https://example.invalid", "sk")
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="cached"))])

    with (
        patch.object(module, "get_application_cache", return_value=cache),
        patch.object(module.litellm, "acompletion", new_callable=AsyncMock) as mocked,
    ):
        mocked.return_value = response
        assert await service.agenerate([], "hello") == "cached"
        assert await service.agenerate([], "hello") == "cached"

    assert mocked.call_count == 1
    assert mocked.call_args.kwargs["caching"] is False


@pytest.mark.asyncio
async def test_web_search_adapter_does_not_cache_unavailable_results():
    from aperag.domains.web_access.schemas import WebSearchRequest
    from aperag.domains.web_access.search import search_service as module
    from aperag.domains.web_access.search.search_service import SearchService

    cache = MemoryAsyncCache()
    service = SearchService.create_default()

    with (
        patch.object(module, "get_application_cache", return_value=cache),
        patch.object(service.provider, "search", new_callable=AsyncMock) as mocked,
    ):
        service.provider.last_search_meta = {"search_status": "unavailable", "error_code": "down"}
        mocked.return_value = []
        await service.search(WebSearchRequest(query="x"))
        await service.search(WebSearchRequest(query="x"))

    assert mocked.call_count == 2


@pytest.mark.asyncio
async def test_web_search_cache_key_includes_timeout():
    from aperag.domains.web_access.schemas import WebSearchRequest, WebSearchResultItem
    from aperag.domains.web_access.search import search_service as module
    from aperag.domains.web_access.search.search_service import SearchService

    cache = MemoryAsyncCache()
    service = SearchService.create_default()

    async def fake_search(**kwargs):
        return [
            WebSearchResultItem(
                rank=1,
                title=f"timeout-{kwargs['timeout']}",
                url="https://example.com",
                snippet="x",
                domain="example.com",
            )
        ]

    with (
        patch.object(module, "get_application_cache", return_value=cache),
        patch.object(service.provider, "search", new_callable=AsyncMock) as mocked,
    ):
        mocked.side_effect = fake_search
        fast = await service.search(WebSearchRequest(query="x", timeout=1))
        slow = await service.search(WebSearchRequest(query="x", timeout=30))

    assert fast.results[0].title == "timeout-1"
    assert slow.results[0].title == "timeout-30"
    assert mocked.call_count == 2


def test_image_parser_caches_ocr_result(tmp_path):
    from aperag.docparser import image_parser as module
    from aperag.docparser.image_parser import ImageParser

    cache = MemorySyncCache()
    image = tmp_path / "x.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
        b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    response = Mock()
    response.text = '{"results":[[{"text":"hello"}]]}'
    response.raise_for_status.return_value = None

    with (
        patch.object(module.settings, "paddleocr_host", "http://ocr"),
        patch.object(module, "get_sync_application_cache", return_value=cache),
        patch.object(module.requests, "post", return_value=response) as post,
    ):
        assert ImageParser().read_image_text(image) == "hello"
        assert ImageParser().read_image_text(image) == "hello"

    assert post.call_count == 1
