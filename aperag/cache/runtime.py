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

"""Production wiring helper for the D10.g read-primitive cache.

The four parse_version-keyed primitives in :mod:`aperag.mcp.tools` call
:func:`get_read_primitive_cache` once they have already passed tenancy
+ authorization gates. This helper builds (and memoizes) one
:class:`~aperag.cache.read_primitive_cache.ReadPrimitiveCache` per
process, wiring it to:

* the L1 size knob at :attr:`aperag.config.Config.d10_cache_l1_size`,
* the L2 TTL knob at :attr:`aperag.config.Config.d10_cache_l2_ttl_seconds`,
* the existing memory-redis client (via
  :class:`aperag.db.redis_manager.RedisConnectionManager` —
  ``settings.memory_redis_url``).

Tests construct their own cache instances and inject fakes via the
underlying :func:`~aperag.cache.read_primitive_cache.build_read_primitive_cache`;
they do not touch the singleton — see :func:`reset_read_primitive_cache`
for an explicit test-only reset hook.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aperag.cache.read_primitive_cache import (
    ReadPrimitiveCache,
    build_read_primitive_cache,
)
from aperag.config import settings

logger = logging.getLogger(__name__)


_cache: Optional[ReadPrimitiveCache] = None
_build_lock = asyncio.Lock()


async def get_read_primitive_cache() -> ReadPrimitiveCache:
    """Return the process-wide :class:`ReadPrimitiveCache`.

    Builds the singleton on first call (lazy). The L2 layer is wired
    to the existing memory-redis client when one is reachable; if not,
    a no-op L2 keeps the composition shape but degrades to L1-only
    (per :class:`~aperag.cache.parse_version_cache.NoopL2Cache`).
    """

    global _cache
    if _cache is not None:
        return _cache
    async with _build_lock:
        if _cache is not None:  # double-checked under lock
            return _cache
        _cache = await _build_singleton()
        return _cache


async def reset_read_primitive_cache() -> None:
    """Drop the cached singleton — test-only hook.

    Production code does not call this; the singleton lives for the
    process. Tests that exercise wiring (e.g., switching L2 to a fake
    Redis) should construct their own
    :class:`ReadPrimitiveCache` directly via
    :func:`~aperag.cache.read_primitive_cache.build_read_primitive_cache`
    and pass it in explicitly. This helper exists for the rare test
    that wants to verify the production singleton path itself.
    """

    global _cache
    async with _build_lock:
        _cache = None


async def _build_singleton() -> ReadPrimitiveCache:
    redis = await _resolve_memory_redis_client()
    return build_read_primitive_cache(
        redis=redis,
        l1_size=settings.d10_cache_l1_size,
        l2_ttl_seconds=settings.d10_cache_l2_ttl_seconds,
    )


async def _resolve_memory_redis_client():
    """Best-effort wiring of the existing memory-redis client.

    Failure to obtain a client (no Redis configured, connection
    refused, etc.) degrades to L1-only — the cache layer remains
    available and still amortizes parse cost within one worker. We
    log at WARNING so the operator can see the degradation; we do
    NOT raise, because §E.7 fail-open invariant explicitly says the
    cache must never block authoritative reads.
    """

    try:
        from aperag.db.redis_manager import RedisConnectionManager

        return await RedisConnectionManager.get_async_client()
    except Exception as e:
        logger.warning(
            "D10.g cache: memory-redis client unavailable, degrading L2 to noop: %s",
            e,
        )
        return None


__all__ = [
    "get_read_primitive_cache",
    "reset_read_primitive_cache",
]
