import asyncio
import inspect
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeVar

from aperag.cache.key import build_cache_key

logger = logging.getLogger(__name__)

T = TypeVar("T")
ItemT = TypeVar("ItemT")

NAMESPACE_LLM_COMPLETION = "llm_completion"
NAMESPACE_EMBEDDING = "embedding"
NAMESPACE_EMBEDDING_DIMENSION = "embedding_dimension"
NAMESPACE_WEB_SEARCH = "web_search"
NAMESPACE_WEB_READ = "web_read"
NAMESPACE_PARSER_PREFLIGHT = "parser_preflight"
NAMESPACE_REMOTE_PARSER = "remote_parser"


@dataclass(frozen=True)
class ApplicationCachePolicy:
    namespace: str
    ttl_seconds: int
    enabled: bool = True
    scope: str | None = None
    cache_none: bool = False
    cache_empty: bool = True
    max_value_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("cache namespace must not be empty")
        if self.ttl_seconds <= 0:
            raise ValueError("cache ttl_seconds must be positive")


class ApplicationCacheMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: dict[str, defaultdict[str, int]] = {}

    def increment(self, namespace: str, event: str, amount: int = 1) -> None:
        with self._lock:
            if namespace not in self._counts:
                self._counts[namespace] = defaultdict(int)
            self._counts[namespace][event] += amount

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {namespace: dict(values) for namespace, values in self._counts.items()}

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()


application_cache_metrics = ApplicationCacheMetrics()


class ApplicationRedisCacheBackend:
    def __init__(self, *, redis):
        self._redis = redis

    async def get(self, key: str) -> str | bytes | None:
        try:
            return await self._redis.get(key)
        except Exception as e:
            logger.warning("Application cache Redis GET failed key=%s error=%s", key, e)
            return None

    async def mget(self, keys: list[str]) -> list[str | bytes | None]:
        if not keys:
            return []
        try:
            if hasattr(self._redis, "mget"):
                return list(await self._redis.mget(keys))
            return [await self._redis.get(key) for key in keys]
        except Exception as e:
            logger.warning("Application cache Redis MGET failed keys=%s error=%s", len(keys), e)
            return [None] * len(keys)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            await self._redis.set(key, value, ex=ttl_seconds)
        except Exception as e:
            logger.warning("Application cache Redis SET failed key=%s error=%s", key, e)

    async def delete(self, *keys: str) -> int:
        try:
            return int(await self._redis.delete(*keys)) if keys else 0
        except Exception as e:
            logger.warning("Application cache Redis DEL failed keys=%s error=%s", len(keys), e)
            return 0


class SyncApplicationRedisCacheBackend:
    def __init__(self, *, redis):
        self._redis = redis

    def get(self, key: str) -> str | bytes | None:
        try:
            return self._redis.get(key)
        except Exception as e:
            logger.warning("Application cache Redis GET failed key=%s error=%s", key, e)
            return None

    def mget(self, keys: list[str]) -> list[str | bytes | None]:
        if not keys:
            return []
        try:
            if hasattr(self._redis, "mget"):
                return list(self._redis.mget(keys))
            return [self._redis.get(key) for key in keys]
        except Exception as e:
            logger.warning("Application cache Redis MGET failed keys=%s error=%s", len(keys), e)
            return [None] * len(keys)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            self._redis.set(key, value, ex=ttl_seconds)
        except Exception as e:
            logger.warning("Application cache Redis SET failed key=%s error=%s", key, e)

    def delete(self, *keys: str) -> int:
        try:
            return int(self._redis.delete(*keys)) if keys else 0
        except Exception as e:
            logger.warning("Application cache Redis DEL failed keys=%s error=%s", len(keys), e)
            return 0


class NoopApplicationCacheBackend:
    async def get(self, key: str) -> str | bytes | None:
        return None

    async def mget(self, keys: list[str]) -> list[str | bytes | None]:
        return [None] * len(keys)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    async def delete(self, *keys: str) -> int:
        return 0


class SyncNoopApplicationCacheBackend:
    def get(self, key: str) -> str | bytes | None:
        return None

    def mget(self, keys: list[str]) -> list[str | bytes | None]:
        return [None] * len(keys)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    def delete(self, *keys: str) -> int:
        return 0


class ApplicationCache:
    def __init__(self, *, backend, default_policy: ApplicationCachePolicy):
        self._backend = backend
        self._default_policy = default_policy
        self._inflight: dict[str, asyncio.Task] = {}
        self._inflight_lock = asyncio.Lock()

    async def get_or_compute(
        self,
        *,
        namespace: str,
        key_data: Any,
        compute: Callable[[], T | Awaitable[T]],
        policy: ApplicationCachePolicy | None = None,
        should_cache: Callable[[T], bool] | None = None,
    ) -> T:
        active_policy = self._resolve_policy(namespace, policy)
        if not active_policy.enabled:
            application_cache_metrics.increment(namespace, "skip")
            return await _maybe_await(compute())
        key = build_cache_key(namespace, key_data, scope=active_policy.scope)
        raw = await self._backend.get(key)
        if raw is not None:
            try:
                application_cache_metrics.increment(namespace, "hit")
                return _loads(raw)
            except Exception:
                application_cache_metrics.increment(namespace, "decode_error")
                await self._backend.delete(key)
        application_cache_metrics.increment(namespace, "miss")
        return await self._singleflight_compute_store(key, active_policy, compute, should_cache)

    async def get_many_or_compute_missing(
        self,
        *,
        namespace: str,
        items: list[ItemT],
        key_data_for_item: Callable[[ItemT], Any],
        compute_missing: Callable[[list[ItemT]], dict[ItemT, T] | Awaitable[dict[ItemT, T]]],
        policy: ApplicationCachePolicy | None = None,
        should_cache: Callable[[T], bool] | None = None,
    ) -> list[T]:
        active_policy = self._resolve_policy(namespace, policy)
        if not items:
            return []
        if not active_policy.enabled:
            application_cache_metrics.increment(namespace, "skip", len(items))
            computed = await _maybe_await(compute_missing(items))
            return [computed[item] for item in items]

        keys = [build_cache_key(namespace, key_data_for_item(item), scope=active_policy.scope) for item in items]
        raw_values = await self._backend.mget(keys)
        values_by_key: dict[str, T] = {}
        missing_by_key: dict[str, ItemT] = {}
        for item, key, raw in zip(items, keys, raw_values):
            if raw is None:
                missing_by_key.setdefault(key, item)
                continue
            try:
                values_by_key[key] = _loads(raw)
                application_cache_metrics.increment(namespace, "hit")
            except Exception:
                application_cache_metrics.increment(namespace, "decode_error")
                missing_by_key.setdefault(key, item)

        if missing_by_key:
            missing_items = list(missing_by_key.values())
            application_cache_metrics.increment(namespace, "miss", len(missing_items))
            computed = await _maybe_await(compute_missing(missing_items))
            for key, item in missing_by_key.items():
                value = computed[item]
                values_by_key[key] = value
                if should_cache is None or should_cache(value):
                    await self._store(key, value, active_policy)
                else:
                    application_cache_metrics.increment(namespace, "skip")

        return [values_by_key[key] for key in keys]

    async def _singleflight_compute_store(self, key, policy, compute, should_cache):
        async with self._inflight_lock:
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._compute_and_store(key, policy, compute, should_cache))
                self._inflight[key] = task
        try:
            return await task
        finally:
            async with self._inflight_lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)

    async def _compute_and_store(self, key, policy, compute, should_cache):
        value = await _maybe_await(compute())
        if should_cache is None or should_cache(value):
            await self._store(key, value, policy)
        else:
            application_cache_metrics.increment(policy.namespace, "skip")
        return value

    async def _store(self, key: str, value: Any, policy: ApplicationCachePolicy) -> None:
        encoded = _dumps(value)
        if value is None and not policy.cache_none:
            application_cache_metrics.increment(policy.namespace, "skip")
            return
        if value in ("", [], {}) and not policy.cache_empty:
            application_cache_metrics.increment(policy.namespace, "skip")
            return
        if policy.max_value_bytes is not None and len(encoded.encode("utf-8")) > policy.max_value_bytes:
            application_cache_metrics.increment(policy.namespace, "skip_too_large")
            return
        await self._backend.set(key, encoded, policy.ttl_seconds)
        application_cache_metrics.increment(policy.namespace, "set")

    def _resolve_policy(self, namespace: str, policy: ApplicationCachePolicy | None) -> ApplicationCachePolicy:
        active = policy or self._default_policy
        if active.namespace == namespace:
            return active
        return ApplicationCachePolicy(
            namespace=namespace,
            ttl_seconds=active.ttl_seconds,
            enabled=active.enabled,
            scope=active.scope,
            cache_none=active.cache_none,
            cache_empty=active.cache_empty,
            max_value_bytes=active.max_value_bytes,
        )


class SyncApplicationCache:
    def __init__(self, *, backend, default_policy: ApplicationCachePolicy):
        self._backend = backend
        self._default_policy = default_policy
        self._locks: dict[str, Lock] = {}
        self._locks_lock = Lock()

    def get_or_compute(
        self,
        *,
        namespace: str,
        key_data: Any,
        compute: Callable[[], T],
        policy: ApplicationCachePolicy | None = None,
        should_cache: Callable[[T], bool] | None = None,
    ) -> T:
        active_policy = self._resolve_policy(namespace, policy)
        if not active_policy.enabled:
            application_cache_metrics.increment(namespace, "skip")
            return compute()
        key = build_cache_key(namespace, key_data, scope=active_policy.scope)
        cached = self._load(key, namespace)
        if cached is not _MISS:
            return cached
        lock = self._lock_for_key(key)
        with lock:
            cached = self._load(key, namespace)
            if cached is not _MISS:
                return cached
            try:
                application_cache_metrics.increment(namespace, "miss")
                value = compute()
                if should_cache is None or should_cache(value):
                    self._store(key, value, active_policy)
                else:
                    application_cache_metrics.increment(namespace, "skip")
                return value
            finally:
                with self._locks_lock:
                    if self._locks.get(key) is lock:
                        self._locks.pop(key, None)

    def get_many_or_compute_missing(
        self,
        *,
        namespace: str,
        items: list[ItemT],
        key_data_for_item: Callable[[ItemT], Any],
        compute_missing: Callable[[list[ItemT]], dict[ItemT, T]],
        policy: ApplicationCachePolicy | None = None,
        should_cache: Callable[[T], bool] | None = None,
    ) -> list[T]:
        active_policy = self._resolve_policy(namespace, policy)
        if not items:
            return []
        if not active_policy.enabled:
            application_cache_metrics.increment(namespace, "skip", len(items))
            computed = compute_missing(items)
            return [computed[item] for item in items]

        keys = [build_cache_key(namespace, key_data_for_item(item), scope=active_policy.scope) for item in items]
        raw_values = self._backend.mget(keys)
        values_by_key: dict[str, T] = {}
        missing_by_key: dict[str, ItemT] = {}
        for item, key, raw in zip(items, keys, raw_values):
            if raw is None:
                missing_by_key.setdefault(key, item)
                continue
            try:
                values_by_key[key] = _loads(raw)
                application_cache_metrics.increment(namespace, "hit")
            except Exception:
                application_cache_metrics.increment(namespace, "decode_error")
                missing_by_key.setdefault(key, item)
        if missing_by_key:
            missing_items = list(missing_by_key.values())
            application_cache_metrics.increment(namespace, "miss", len(missing_items))
            computed = compute_missing(missing_items)
            for key, item in missing_by_key.items():
                value = computed[item]
                values_by_key[key] = value
                if should_cache is None or should_cache(value):
                    self._store(key, value, active_policy)
                else:
                    application_cache_metrics.increment(namespace, "skip")
        return [values_by_key[key] for key in keys]

    def _load(self, key: str, namespace: str):
        raw = self._backend.get(key)
        if raw is None:
            return _MISS
        try:
            application_cache_metrics.increment(namespace, "hit")
            return _loads(raw)
        except Exception:
            application_cache_metrics.increment(namespace, "decode_error")
            self._backend.delete(key)
            return _MISS

    def _store(self, key: str, value: Any, policy: ApplicationCachePolicy) -> None:
        encoded = _dumps(value)
        if value is None and not policy.cache_none:
            application_cache_metrics.increment(policy.namespace, "skip")
            return
        if value in ("", [], {}) and not policy.cache_empty:
            application_cache_metrics.increment(policy.namespace, "skip")
            return
        if policy.max_value_bytes is not None and len(encoded.encode("utf-8")) > policy.max_value_bytes:
            application_cache_metrics.increment(policy.namespace, "skip_too_large")
            return
        self._backend.set(key, encoded, policy.ttl_seconds)
        application_cache_metrics.increment(policy.namespace, "set")

    def _lock_for_key(self, key: str) -> Lock:
        with self._locks_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = Lock()
                self._locks[key] = lock
            return lock

    def _resolve_policy(self, namespace: str, policy: ApplicationCachePolicy | None) -> ApplicationCachePolicy:
        active = policy or self._default_policy
        if active.namespace == namespace:
            return active
        return ApplicationCachePolicy(
            namespace=namespace,
            ttl_seconds=active.ttl_seconds,
            enabled=active.enabled,
            scope=active.scope,
            cache_none=active.cache_none,
            cache_empty=active.cache_empty,
            max_value_bytes=active.max_value_bytes,
        )


def _dumps(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | bytes) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class _Miss:
    pass


_MISS = _Miss()
