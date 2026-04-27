import asyncio
from unittest.mock import patch

import pytest

from aperag.cache.application import (
    ApplicationCache,
    ApplicationCacheMetrics,
    ApplicationCachePolicy,
    ApplicationRedisCacheBackend,
    SyncApplicationCache,
    SyncApplicationRedisCacheBackend,
)
from aperag.cache.key import build_cache_key


class FakeAsyncRedis:
    def __init__(self, *, fail_get=False, fail_set=False):
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.store = {}
        self.ttls = {}

    async def get(self, key):
        if self.fail_get:
            raise RuntimeError("down")
        return self.store.get(key)

    async def mget(self, keys):
        return [await self.get(key) for key in keys]

    async def set(self, key, value, ex=None):
        if self.fail_set:
            raise RuntimeError("down")
        self.store[key] = value
        self.ttls[key] = ex

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)


class FakeSyncRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def mget(self, keys):
        return [self.get(key) for key in keys]

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)


def test_build_cache_key_is_stable_and_scoped():
    a = build_cache_key("embedding", {"b": 2, "a": ["hello"]}, scope="tenant-a")
    b = build_cache_key("embedding", {"a": ["hello"], "b": 2}, scope="tenant-a")
    c = build_cache_key("embedding", {"a": ["hello"], "b": 2}, scope="tenant-b")

    assert a == b
    assert a != c
    assert a.startswith("aperag:cache:v1:embedding:")
    assert "hello" not in a


@pytest.mark.asyncio
async def test_application_cache_hits_redis_and_singleflights_misses():
    metrics = ApplicationCacheMetrics()
    cache = ApplicationCache(
        backend=ApplicationRedisCacheBackend(redis=FakeAsyncRedis()),
        default_policy=ApplicationCachePolicy(namespace="test", ttl_seconds=60),
    )
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"value": calls}

    results = await asyncio.gather(
        *[cache.get_or_compute(namespace="test", key_data={"k": "same"}, compute=compute) for _ in range(5)]
    )
    second = await cache.get_or_compute(namespace="test", key_data={"k": "same"}, compute=compute)

    assert results == [{"value": 1}] * 5
    assert second == {"value": 1}
    assert calls == 1
    assert metrics.snapshot() == {}


@pytest.mark.asyncio
async def test_application_cache_fail_open_and_should_cache():
    redis = FakeAsyncRedis(fail_get=True, fail_set=True)
    cache = ApplicationCache(
        backend=ApplicationRedisCacheBackend(redis=redis),
        default_policy=ApplicationCachePolicy(namespace="test", ttl_seconds=60),
    )

    result = await cache.get_or_compute(
        namespace="test",
        key_data={"k": "transient"},
        compute=lambda: {"status": "unavailable"},
        should_cache=lambda value: value["status"] == "ok",
    )

    assert result == {"status": "unavailable"}


def test_sync_application_cache_batch_partial_hits():
    redis = FakeSyncRedis()
    cache = SyncApplicationCache(
        backend=SyncApplicationRedisCacheBackend(redis=redis),
        default_policy=ApplicationCachePolicy(namespace="embedding", ttl_seconds=60),
    )
    calls = []

    def compute_missing(items):
        calls.append(items)
        return {item: [float(len(item))] for item in items}

    first = cache.get_many_or_compute_missing(
        namespace="embedding",
        items=["hot", "cold"],
        key_data_for_item=lambda item: {"text": item},
        compute_missing=compute_missing,
    )
    second = cache.get_many_or_compute_missing(
        namespace="embedding",
        items=["hot", "warm"],
        key_data_for_item=lambda item: {"text": item},
        compute_missing=compute_missing,
    )

    assert first == [[3.0], [4.0]]
    assert second == [[3.0], [4.0]]
    assert calls == [["hot", "cold"], ["warm"]]


def test_sync_application_cache_releases_per_key_locks_after_compute():
    cache = SyncApplicationCache(
        backend=SyncApplicationRedisCacheBackend(redis=FakeSyncRedis()),
        default_policy=ApplicationCachePolicy(namespace="parser", ttl_seconds=60),
    )

    for idx in range(20):
        cache.get_or_compute(namespace="parser", key_data={"idx": idx}, compute=lambda idx=idx: idx)

    assert cache._locks == {}


@pytest.mark.asyncio
async def test_application_runtime_uses_dedicated_cache_redis_url(monkeypatch):
    from aperag.cache import application_runtime

    class FakeRedisClient:
        async def ping(self):
            return True

    application_runtime.reset_application_cache_for_tests()
    monkeypatch.setattr(application_runtime.settings, "cache_enabled", True)
    monkeypatch.setattr(application_runtime.settings, "cache_redis_url", "redis://cache.example/2")

    with (
        patch.object(application_runtime.async_redis.Redis, "from_url", return_value=FakeRedisClient()) as from_url,
        patch.object(application_runtime.RedisConnectionManager, "get_async_client") as shared_client,
    ):
        await application_runtime.get_application_cache()

    from_url.assert_called_once()
    assert from_url.call_args.args[0] == "redis://cache.example/2"
    shared_client.assert_not_called()
    application_runtime.reset_application_cache_for_tests()
