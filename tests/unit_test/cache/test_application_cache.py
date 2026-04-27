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


# ---------------------------------------------------------------------
# Wave 6 #38 — Application cache cross-loop lazy-rebuild + WARN log
# (per spec §K.11.1 row 38 + huangheng PR #1734 sync point 7).
# ---------------------------------------------------------------------


def _run_in_new_loop(coro_factory):
    """Drive ``coro_factory()`` to completion on a fresh event loop and
    close it. Each call gives the cache module a distinct
    ``asyncio.AbstractEventLoop`` identity so the cross-loop branch is
    exercised."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_factory())
    finally:
        loop.close()


def test_application_cache_rebuilds_on_loop_switch_instead_of_silent_noop(monkeypatch, caplog):
    """The pre-Wave-6 implementation silently swapped the cache for a
    Noop backend whenever the running event loop changed. That meant
    a worker that switched loops (test process, asyncio reinit, etc.)
    paid full LiteLLM/embedding cost from then on with no log. Wave 6
    #38: detect the loop switch, emit a WARN, bump a metric counter,
    and rebuild against the real Redis backend on the new loop."""
    from aperag.cache import application_runtime
    from aperag.cache.application import application_cache_metrics

    application_runtime.reset_application_cache_for_tests()
    monkeypatch.setattr(application_runtime.settings, "cache_enabled", True)
    monkeypatch.setattr(application_runtime.settings, "cache_redis_url", "redis://cache.example/2")

    class FakeRedisClient:
        async def ping(self):
            return True

    fake_clients: list[FakeRedisClient] = []

    def _from_url(*args, **kwargs):
        client = FakeRedisClient()
        fake_clients.append(client)
        return client

    with patch.object(application_runtime.async_redis.Redis, "from_url", side_effect=_from_url):
        first = _run_in_new_loop(application_runtime.get_application_cache)
        # First call on a fresh loop should never count as a switch.
        assert application_cache_metrics.snapshot().get("application_runtime", {}).get("loop_switch_rebuild", 0) == 0

        with caplog.at_level("WARNING"):
            second = _run_in_new_loop(application_runtime.get_application_cache)

    # Two distinct loops → two distinct cache instances (rebuilt, not
    # the same singleton handed back).
    assert second is not first
    # And critically, NOT a Noop fallback — the rebuilt cache wraps the
    # real Redis backend on the new loop.
    assert isinstance(second._backend, application_runtime.ApplicationRedisCacheBackend), (
        "Wave 6 #38 invariant: cross-loop rebuild must produce a real Redis-backed cache, "
        "not a silent Noop fallback (per feedback_announce_equals_landed.md narrative-truth lock)"
    )
    # Metric counter records the loop switch event for operator
    # observability.
    counts = application_cache_metrics.snapshot()["application_runtime"]
    assert counts["loop_switch_rebuild"] == 1
    # WARN log emitted with the operator-actionable phrasing.
    assert any(
        "rebuilding for new event loop" in record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    )

    application_runtime.reset_application_cache_for_tests()


def test_application_cache_does_not_increment_loop_switch_metric_on_first_build(monkeypatch):
    """First-ever call (no prior cache) must NOT count as a loop switch
    — the counter should only record genuine rebuilds caused by an
    event-loop change."""
    from aperag.cache import application_runtime
    from aperag.cache.application import application_cache_metrics

    application_runtime.reset_application_cache_for_tests()
    monkeypatch.setattr(application_runtime.settings, "cache_enabled", True)
    monkeypatch.setattr(application_runtime.settings, "cache_redis_url", "redis://cache.example/2")

    class FakeRedisClient:
        async def ping(self):
            return True

    with patch.object(application_runtime.async_redis.Redis, "from_url", return_value=FakeRedisClient()):
        _run_in_new_loop(application_runtime.get_application_cache)

    snapshot = application_cache_metrics.snapshot()
    # The counter key may be entirely absent on first build — that's
    # the right behaviour. Don't even materialise the namespace.
    assert "loop_switch_rebuild" not in snapshot.get("application_runtime", {})

    application_runtime.reset_application_cache_for_tests()


def test_application_cache_repeat_call_in_same_loop_is_singleton_no_metric_bump(monkeypatch):
    """Repeated calls on the same loop must reuse the singleton cache
    and never bump the loop_switch counter."""
    from aperag.cache import application_runtime
    from aperag.cache.application import application_cache_metrics

    application_runtime.reset_application_cache_for_tests()
    monkeypatch.setattr(application_runtime.settings, "cache_enabled", True)
    monkeypatch.setattr(application_runtime.settings, "cache_redis_url", "redis://cache.example/2")

    class FakeRedisClient:
        async def ping(self):
            return True

    async def _double_get():
        first = await application_runtime.get_application_cache()
        second = await application_runtime.get_application_cache()
        return first, second

    with patch.object(application_runtime.async_redis.Redis, "from_url", return_value=FakeRedisClient()):
        first, second = _run_in_new_loop(_double_get)

    assert first is second  # Singleton on the same loop
    snapshot = application_cache_metrics.snapshot()
    assert "loop_switch_rebuild" not in snapshot.get("application_runtime", {})

    application_runtime.reset_application_cache_for_tests()


def test_application_cache_falls_back_to_noop_when_redis_unreachable_on_new_loop(monkeypatch):
    """If the real Redis cannot be reached on the new loop, the rebuild
    legitimately falls back to a Noop backend (existing behaviour for
    Redis-unavailable). Pre-Wave-6 the cross-loop branch silently used
    Noop *without* trying Redis; Wave 6 only falls back on a real
    Redis failure, and the existing Redis-unavailable WARN still fires."""
    from aperag.cache import application_runtime
    from aperag.cache.application import application_cache_metrics

    application_runtime.reset_application_cache_for_tests()
    monkeypatch.setattr(application_runtime.settings, "cache_enabled", True)
    monkeypatch.setattr(application_runtime.settings, "cache_redis_url", "redis://cache.example/2")

    class GoodRedis:
        async def ping(self):
            return True

    class BadRedis:
        async def ping(self):
            raise RuntimeError("redis down on new loop")

    sequence = [GoodRedis(), BadRedis()]

    def _from_url(*args, **kwargs):
        return sequence.pop(0)

    with patch.object(application_runtime.async_redis.Redis, "from_url", side_effect=_from_url):
        first = _run_in_new_loop(application_runtime.get_application_cache)
        second = _run_in_new_loop(application_runtime.get_application_cache)

    # First loop: real backend.
    assert isinstance(first._backend, application_runtime.ApplicationRedisCacheBackend)
    # Second loop: rebuild attempted real Redis but it failed → Noop.
    # Crucially this is a *real* failure path, not the silent
    # cross-loop downgrade we removed in Wave 6 #38.
    assert isinstance(second._backend, application_runtime.NoopApplicationCacheBackend)
    # The loop-switch metric still bumps (we did try to rebuild —
    # just landed on Noop because the new loop's Redis was down).
    counts = application_cache_metrics.snapshot()["application_runtime"]
    assert counts["loop_switch_rebuild"] == 1

    application_runtime.reset_application_cache_for_tests()
