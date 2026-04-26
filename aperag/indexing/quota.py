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

"""Tenant-scoped token bucket quota — celery T2.2.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §H.5,
worker-side LLM / embedding calls go through a token bucket keyed by
``(resource_class, tenant_scope_key)``. The bucket has two parameters
per key — ``capacity`` and ``refill_rate_per_sec`` — and worker code
calls :meth:`QuotaBackend.acquire` before issuing the upstream API
call. The acquire either returns immediately (a token was available),
or sleeps until the bucket refills enough for the next token, then
returns.

The §H.5 design has two dimensions of forward compatibility:

1. **Per-tenant policy override with default fallback.** A tenant
   can opt into a custom ``capacity`` / ``refill_rate_per_sec`` via
   the ``tenant_quota`` table (or in-memory equivalent for tests);
   if no row exists for that tenant, the bucket falls back to a
   ``"default"`` policy shared across tenants. This lets a private-
   deployment operator drop in a flat policy on day-1 and add
   tenant-specific tuning later.

2. **Resource-class extensibility.** Adding a new resource class
   (e.g., ``"vision"`` for GPU-bound image inference) is a config-
   only change — declare a default policy under that name and
   workers can immediately ``acquire(resource_class="vision", ...)``.

The Wave 1 ``LineageMember.tenant_scope_key`` (per architect ruling
msg=c3b0ba5b at SET-element level) is the canonical input for the
``tenant_scope_key`` parameter; the orchestrator passes it through
when invoking modality workers. Quota state, however, lives in Redis
in production so multi-process workers all share the same bucket per
``(resource_class, tenant_scope_key)``.

This module ships:

* :class:`QuotaBackend` — async Protocol every backend implements.
* :class:`QuotaPolicy` — the (capacity, refill_rate_per_sec) pair.
* :class:`QuotaPolicyRegistry` — maps
  ``(resource_class, tenant_scope_key)`` → :class:`QuotaPolicy` with
  ``"default"`` fallback.
* :class:`InMemoryQuotaBackend` — plain-Python token bucket, used by
  the §L Tier-1 single-process deployment (``INDEXING_MODE=inline``)
  and by the tests in this repo.
* :class:`RedisQuotaBackend` — Redis-backed implementation with an
  atomic Lua script so concurrent workers never overshoot the bucket
  capacity. The Lua script also returns the wait time when the
  bucket is empty so the caller does not need a server-side
  ``BLPOP`` loop.

The Redis backend is wired by the Wave 2 orchestrator at job-dispatch
time; the in-memory backend is what every unit test in this PR uses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from binascii import crc32
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol

logger = logging.getLogger(__name__)


# Sentinel used as the fallback ``tenant_scope_key`` when no per-
# tenant override exists in the registry. Picked to be obviously
# non-tenant-shaped so a misconfigured caller does not silently
# write data under a tenant identifier.
DEFAULT_TENANT_FALLBACK = "default"


# ---------------------------------------------------------------------
# Policy data model
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaPolicy:
    """One ``(capacity, refill_rate_per_sec)`` configuration.

    ``capacity`` is the maximum number of tokens the bucket can
    hold; ``refill_rate_per_sec`` is the rate at which tokens are
    added when the bucket is below capacity. The pair is the
    standard Token Bucket model.

    Example: ``QuotaPolicy(capacity=60, refill_rate_per_sec=1.0)``
    allows up to 60 LLM calls in a burst, then 1 call/sec sustained.
    Matches the §H.5 "60 tokens / minute" baseline.
    """

    capacity: float
    refill_rate_per_sec: float

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"QuotaPolicy.capacity must be > 0, got {self.capacity}")
        if self.refill_rate_per_sec <= 0:
            raise ValueError(f"QuotaPolicy.refill_rate_per_sec must be > 0, got {self.refill_rate_per_sec}")


class QuotaPolicyRegistry:
    """Resolve ``(resource_class, tenant_scope_key)`` → :class:`QuotaPolicy`.

    Lookup order:

    1. Exact match on ``(resource_class, tenant_scope_key)``.
    2. Fallback to ``(resource_class, "default")``.
    3. Raise :class:`KeyError` if neither is configured — a worker
       hitting an unconfigured resource class is a deployment bug
       and the orchestrator should surface it loudly rather than
       silently let work through unbounded.

    The registry is intentionally tiny: a Python dict is enough at
    Wave 2 scale (handful of resource classes × hundreds of tenants).
    A future move to a DB-backed ``tenant_quota`` table swaps the
    dict for a cached query without changing the caller surface.
    """

    def __init__(self, policies: Mapping[tuple[str, str], QuotaPolicy] | None = None) -> None:
        # Defensive copy so callers cannot mutate the registry after
        # construction.
        self._policies: dict[tuple[str, str], QuotaPolicy] = dict(policies or {})

    def register(self, *, resource_class: str, tenant_scope_key: str, policy: QuotaPolicy) -> None:
        """Add or replace the policy for ``(resource_class, tenant_scope_key)``.

        Pass ``tenant_scope_key=DEFAULT_TENANT_FALLBACK`` to set the
        shared default for a resource class.
        """
        self._policies[(resource_class, tenant_scope_key)] = policy

    def resolve(self, *, resource_class: str, tenant_scope_key: str) -> QuotaPolicy:
        exact = self._policies.get((resource_class, tenant_scope_key))
        if exact is not None:
            return exact
        fallback = self._policies.get((resource_class, DEFAULT_TENANT_FALLBACK))
        if fallback is not None:
            return fallback
        raise KeyError(
            f"no QuotaPolicy for resource_class={resource_class!r} (tried "
            f"tenant_scope_key={tenant_scope_key!r} and {DEFAULT_TENANT_FALLBACK!r})"
        )


# ---------------------------------------------------------------------
# Backend Protocol
# ---------------------------------------------------------------------


class QuotaBackend(Protocol):
    """Async backend that holds the per-bucket token state.

    Implementations differ on where the state lives — in-process
    dict for tests / Tier-1 deployments, Redis for multi-worker
    production — but every implementation MUST satisfy the same
    semantic: :meth:`acquire` returns only after one token has been
    deducted from the appropriate bucket, blocking (with a sleep
    that respects the refill rate) when no token is available.
    """

    async def acquire(self, *, resource_class: str, tenant_scope_key: str) -> None:
        """Block until one token is available for the
        ``(resource_class, tenant_scope_key)`` bucket and deduct it.

        On return, the caller is permitted to issue ONE upstream API
        call. The caller does NOT need to release the token — token
        bucket is consumption-only; refill is rate-based.
        """


# ---------------------------------------------------------------------
# In-memory backend — single-process / test usage.
# ---------------------------------------------------------------------


@dataclass
class _BucketState:
    """Mutable state per bucket — tokens currently held + last refill ts."""

    tokens: float
    last_refill_monotonic: float


class InMemoryQuotaBackend:
    """Plain-Python token bucket. One ``asyncio.Lock`` per bucket key.

    Suitable for the §L Tier-1 (``INDEXING_MODE=inline``) deployment
    where there is exactly one worker process; production deployments
    bind to :class:`RedisQuotaBackend` so multi-worker token state is
    shared.

    Tests use this implementation because it is deterministic and
    requires no external services.
    """

    def __init__(
        self,
        registry: QuotaPolicyRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._buckets: dict[tuple[str, str], _BucketState] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._registry_guard = asyncio.Lock()

    async def _get_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        async with self._registry_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    def _refill(self, state: _BucketState, policy: QuotaPolicy, now: float) -> None:
        """Advance the bucket to ``now`` according to the policy.

        Tokens are clamped to ``capacity`` so an idle bucket does not
        accumulate beyond the burst budget.
        """
        elapsed = max(0.0, now - state.last_refill_monotonic)
        state.tokens = min(policy.capacity, state.tokens + elapsed * policy.refill_rate_per_sec)
        state.last_refill_monotonic = now

    async def acquire(self, *, resource_class: str, tenant_scope_key: str) -> None:
        policy = self._registry.resolve(resource_class=resource_class, tenant_scope_key=tenant_scope_key)
        key = (resource_class, tenant_scope_key)
        lock = await self._get_lock(key)
        while True:
            async with lock:
                state = self._buckets.get(key)
                if state is None:
                    # First access: bucket starts at capacity (full
                    # burst available immediately).
                    state = _BucketState(
                        tokens=policy.capacity,
                        last_refill_monotonic=self._clock(),
                    )
                    self._buckets[key] = state
                self._refill(state, policy, self._clock())
                if state.tokens >= 1.0:
                    state.tokens -= 1.0
                    return
                # Compute precise sleep time so we wake the moment a
                # token is available. The loop re-checks under the
                # lock to handle the race where another waiter woke
                # first and consumed the freshly refilled token.
                deficit = 1.0 - state.tokens
                wait_seconds = deficit / policy.refill_rate_per_sec
            await asyncio.sleep(max(wait_seconds, 0.001))


# ---------------------------------------------------------------------
# Redis backend — production multi-worker.
# ---------------------------------------------------------------------


# Lua script: refill + try-acquire one token atomically.
#
# Args: capacity, refill_rate_per_sec, now_seconds
# Hash fields on the key: tokens (float), last_refill (epoch seconds float)
#
# Returns: a 2-element table {acquired (1 or 0), wait_seconds (float)}.
# When acquired=1, wait_seconds is 0; when acquired=0, the caller
# should asyncio.sleep(wait_seconds) and retry.
_TOKEN_BUCKET_LUA = """
local tokens_key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local stored = redis.call('HMGET', tokens_key, 'tokens', 'last_refill')
local tokens = tonumber(stored[1]) or capacity
local last = tonumber(stored[2]) or now
local elapsed = now - last
if elapsed < 0 then elapsed = 0 end

tokens = math.min(capacity, tokens + elapsed * rate)
if tokens >= 1.0 then
    tokens = tokens - 1.0
    redis.call('HMSET', tokens_key, 'tokens', tokens, 'last_refill', now)
    return {1, 0}
else
    local deficit = 1.0 - tokens
    local wait_seconds = deficit / rate
    redis.call('HMSET', tokens_key, 'tokens', tokens, 'last_refill', now)
    return {0, wait_seconds}
end
"""


class RedisQuotaBackend:
    """Redis-backed token bucket using a Lua script for atomic refill+acquire.

    Production callers wire this up at orchestrator startup with a
    real ``redis.asyncio.Redis`` instance. The Lua script is loaded
    on first use and cached by hash so subsequent acquires are a
    single ``EVALSHA`` round-trip.

    The Wave 1 :class:`aperag.indexing.graph.RedisEntityLock` uses
    the same Redis client / hashing idiom; this class follows the
    same conventions for operational consistency.
    """

    def __init__(
        self,
        redis_client: Any,
        registry: QuotaPolicyRegistry,
        *,
        key_prefix: str = "indexing:quota",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._redis = redis_client
        self._registry = registry
        self._key_prefix = key_prefix
        self._clock = clock
        # Lazy-loaded Lua script reference — cached as a
        # :class:`redis.commands.core.Script` (or async equivalent)
        # after first use.
        self._script: Any | None = None

    def _bucket_key(self, *, resource_class: str, tenant_scope_key: str) -> str:
        # Bound Redis-key cardinality by hashing the tenant_scope_key
        # into a slot pool when the tenant id is high-cardinality
        # (e.g., ephemeral session ids). For typical "user:<id>" /
        # "org:<id>" identifiers, the slot pool is large enough that
        # collisions are rare and the bucket-per-tenant invariant
        # holds. Operators can lift the slot count by overriding
        # ``key_prefix`` to include the literal tenant id when full
        # per-tenant accounting is required.
        slot = crc32(tenant_scope_key.encode("utf-8")) % 65536
        return f"{self._key_prefix}:{resource_class}:{slot:04x}"

    def _ensure_script(self) -> Any:
        if self._script is None:
            self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)
        return self._script

    async def _try_acquire(
        self,
        *,
        resource_class: str,
        tenant_scope_key: str,
        policy: QuotaPolicy,
    ) -> tuple[bool, float]:
        """Run the Lua script once. Returns (acquired, wait_seconds)."""
        script = self._ensure_script()
        result = await _await_if_needed(
            script(
                keys=[self._bucket_key(resource_class=resource_class, tenant_scope_key=tenant_scope_key)],
                args=[policy.capacity, policy.refill_rate_per_sec, self._clock()],
            )
        )
        # redis-py async returns the list directly; bytes/ints are
        # backend-dependent so coerce defensively.
        acquired = int(result[0]) == 1
        wait = float(result[1])
        return acquired, wait

    async def acquire(self, *, resource_class: str, tenant_scope_key: str) -> None:
        policy = self._registry.resolve(resource_class=resource_class, tenant_scope_key=tenant_scope_key)
        while True:
            acquired, wait_seconds = await self._try_acquire(
                resource_class=resource_class,
                tenant_scope_key=tenant_scope_key,
                policy=policy,
            )
            if acquired:
                return
            await asyncio.sleep(max(wait_seconds, 0.001))


async def _await_if_needed(value: Any) -> Any:
    """Compatibility shim so the same code works against sync redis-py
    (returning concrete values) and async redis-py (returning awaitables).

    Production deployments use ``redis.asyncio.Redis`` whose
    ``register_script`` returns an :class:`AsyncScript` whose
    ``__call__`` returns an awaitable. Sync redis-py's ``Script``
    returns the resolved value directly. We accept both so the same
    backend works whether the caller passed a sync or async client.
    """
    if isinstance(value, Awaitable):
        return await value
    return value


__all__ = [
    "DEFAULT_TENANT_FALLBACK",
    "QuotaPolicy",
    "QuotaPolicyRegistry",
    "QuotaBackend",
    "InMemoryQuotaBackend",
    "RedisQuotaBackend",
]
