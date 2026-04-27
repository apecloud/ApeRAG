# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-tier cache primitives for read-primitive responses.

L1 is an in-process LRU bounded by
``Config.read_primitive_cache_l1_size``; L2 is a
Redis-backed adapter behind an injectable client (per architect Q3 lock
msg=a67974b3 — keyword-only DI). Both layers store and return JSON-
serialized strings; the higher-level :mod:`aperag.cache.read_primitive_cache`
takes care of Pydantic encode/decode.

Both layers expose the same minimal protocol — ``get`` / ``set`` /
``delete`` / ``delete_namespace`` — so callers don't pivot on layer
type. The composition order (L1 → L2 → compute) lives in
:class:`aperag.cache.read_primitive_cache.ReadPrimitiveCache`.

Key shape (used by the L2 Redis store):

* ``read_primitive:<namespace>:<key_part_1>:<key_part_2>:...``
* Reserved namespaces: ``outline`` / ``section`` / ``content`` /
  ``chunk`` / ``colmeta`` / ``docmeta``.

Explicit invalidation triggers may purge an entire namespace via
:meth:`L2Cache.delete_namespace`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

READ_PRIMITIVE_CACHE_PREFIX = "read_primitive"


# ---------------------------------------------------------------------------
# Cache layer protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheLayer(Protocol):
    """Minimal cache layer protocol shared by L1 and L2."""

    async def get(self, key: str) -> Optional[str]: ...

    async def set(self, key: str, value: str) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def delete_namespace(self, namespace: str) -> None: ...


@runtime_checkable
class AsyncRedisLike(Protocol):
    """The minimal async Redis surface :class:`L2Cache` exercises.

    Production wiring passes ``redis.asyncio.Redis`` from
    :mod:`aperag.config`'s ``memory_redis_url`` connection. Tests inject
    a fake; see ``tests/unit_test/cache/test_parse_version_cache.py``.
    """

    async def get(self, key: str) -> Optional[bytes]: ...

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None: ...

    async def delete(self, *keys: str) -> int: ...

    def scan_iter(self, match: Optional[str] = None) -> Any: ...


# ---------------------------------------------------------------------------
# L1: in-process LRU
# ---------------------------------------------------------------------------


class L1Cache:
    """In-process LRU cache, bounded by ``maxsize``.

    The values are JSON strings — the calling layer
    (:class:`aperag.cache.read_primitive_cache.ReadPrimitiveCache`) is
    responsible for serializing the Pydantic model in / out, so this
    class stays simple and trivially testable.

    The implementation uses an :class:`collections.OrderedDict` plus an
    :class:`asyncio.Lock` so the same instance can be safely shared
    across concurrent ``async`` tasks within one worker. The lock is
    held only for O(1) work — there is no blocking I/O under it.
    """

    def __init__(self, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("L1Cache maxsize must be positive")
        self._maxsize = maxsize
        self._store: "OrderedDict[str, str]" = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            value = self._store.get(key)
            if value is None:
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def delete_namespace(self, namespace: str) -> None:
        prefix = f"{READ_PRIMITIVE_CACHE_PREFIX}:{namespace}:"
        async with self._lock:
            doomed = [k for k in self._store if k.startswith(prefix)]
            for k in doomed:
                self._store.pop(k, None)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# L2: Redis-backed
# ---------------------------------------------------------------------------


class L2Cache:
    """Redis-backed cache with TTL.

    The Redis client is injected — production callers pass a real
    ``redis.asyncio.Redis`` instance, tests pass a fake. This decouples
    the cache from :mod:`aperag.concurrent_control.redis_lock`'s shared
    client (which has a different purpose — distributed locking).

    A failure to talk to Redis is logged and swallowed: the read
    primitive falls through to the compute step (cache miss is fail-
    open). The cache only accelerates; if
    it fails, the authoritative store still answers.
    """

    def __init__(self, *, redis: AsyncRedisLike, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("L2Cache ttl_seconds must be positive")
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    async def get(self, key: str) -> Optional[str]:
        try:
            raw = await self._redis.get(key)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("L2 cache GET failed for %s: %s", key, e)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return raw  # type: ignore[return-value]  # already str

    async def set(self, key: str, value: str) -> None:
        try:
            await self._redis.set(key, value, ex=self._ttl_seconds)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("L2 cache SET failed for %s: %s", key, e)

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("L2 cache DEL failed for %s: %s", key, e)

    async def delete_namespace(self, namespace: str) -> None:
        prefix = f"{READ_PRIMITIVE_CACHE_PREFIX}:{namespace}:"
        try:
            doomed: list[str] = []
            async for raw_key in self._redis.scan_iter(match=f"{prefix}*"):
                if isinstance(raw_key, bytes):
                    raw_key = raw_key.decode("utf-8")
                doomed.append(raw_key)
            if doomed:
                await self._redis.delete(*doomed)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("L2 cache namespace delete failed for %s: %s", prefix, e)


class NoopL2Cache:
    """L2 that always misses — used when Redis is not configured.

    Useful in tests and dev without a Redis dependency. Keeps the
    composition shape unchanged so the rest of the cache pipeline does
    not need to special-case "no L2".
    """

    async def get(self, key: str) -> Optional[str]:  # pragma: no cover - trivial
        return None

    async def set(self, key: str, value: str) -> None:  # pragma: no cover - trivial
        return None

    async def delete(self, key: str) -> None:  # pragma: no cover - trivial
        return None

    async def delete_namespace(self, namespace: str) -> None:  # pragma: no cover - trivial
        return None


__all__ = [
    "AsyncRedisLike",
    "CacheLayer",
    "L1Cache",
    "L2Cache",
    "NoopL2Cache",
    "READ_PRIMITIVE_CACHE_PREFIX",
]
