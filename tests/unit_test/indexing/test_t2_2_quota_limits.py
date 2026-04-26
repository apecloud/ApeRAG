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

"""T2.2 acceptance tests — quota + bulkhead (design pack §H.5 / §H.6).

Three coverage groups, mapping to the architect-locked acceptance
gates for the quota / bulkhead lane (msg=8420f12a + ruling msg=492315e8):

1. **§H.5 token bucket semantics** — :class:`InMemoryQuotaBackend`
   correctly refills, blocks when empty, isolates per
   ``(resource_class, tenant_scope_key)``, and falls back to the
   ``"default"`` policy when no per-tenant override exists.

2. **§H.6 bulkhead** — :func:`bulkhead_timeout` enforces wall-time
   ceilings; :func:`reject_if_oversize` rejects oversize uploads at
   the boundary.

3. **Construction-only Redis backend** — :class:`RedisQuotaBackend`
   constructs cleanly against a mock client and exposes the same
   :class:`QuotaBackend` Protocol surface as the in-memory variant.
   Full-Redis integration belongs in T2.3 load-test infra (real
   Redis fixture); this group pins the construction path so a future
   import / type regression fails CI loudly.

The tests use the in-memory backend with a deterministic clock fixture
so timing-sensitive assertions never flake on slow CI workers.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from aperag.indexing.limits import (
    EMBEDDING_CALL_TIMEOUT_SECONDS,
    LLM_CALL_TIMEOUT_SECONDS,
    UPLOAD_MAX_BYTES,
    bulkhead_timeout,
    reject_if_oversize,
)
from aperag.indexing.quota import (
    DEFAULT_TENANT_FALLBACK,
    InMemoryQuotaBackend,
    QuotaPolicy,
    QuotaPolicyRegistry,
    RedisQuotaBackend,
)

# ---------------------------------------------------------------------
# Group 1: QuotaPolicy validation
# ---------------------------------------------------------------------


def test_quota_policy_rejects_zero_or_negative_capacity():
    with pytest.raises(ValueError, match="capacity"):
        QuotaPolicy(capacity=0, refill_rate_per_sec=1.0)
    with pytest.raises(ValueError, match="capacity"):
        QuotaPolicy(capacity=-1, refill_rate_per_sec=1.0)


def test_quota_policy_rejects_zero_or_negative_refill_rate():
    with pytest.raises(ValueError, match="refill_rate"):
        QuotaPolicy(capacity=10, refill_rate_per_sec=0)
    with pytest.raises(ValueError, match="refill_rate"):
        QuotaPolicy(capacity=10, refill_rate_per_sec=-0.1)


def test_quota_policy_accepts_fractional_capacity_and_rate():
    """Fractional capacities are valid (e.g., 0.5 token / sec sustained)."""
    policy = QuotaPolicy(capacity=2.5, refill_rate_per_sec=0.5)
    assert policy.capacity == 2.5
    assert policy.refill_rate_per_sec == 0.5


# ---------------------------------------------------------------------
# Group 2: QuotaPolicyRegistry resolution
# ---------------------------------------------------------------------


def test_registry_exact_match_wins_over_default():
    registry = QuotaPolicyRegistry()
    default_policy = QuotaPolicy(capacity=5, refill_rate_per_sec=1.0)
    tenant_policy = QuotaPolicy(capacity=20, refill_rate_per_sec=2.0)

    registry.register(resource_class="llm", tenant_scope_key=DEFAULT_TENANT_FALLBACK, policy=default_policy)
    registry.register(resource_class="llm", tenant_scope_key="user:alice", policy=tenant_policy)

    assert registry.resolve(resource_class="llm", tenant_scope_key="user:alice") == tenant_policy
    # A different tenant with no override falls back.
    assert registry.resolve(resource_class="llm", tenant_scope_key="user:bob") == default_policy


def test_registry_raises_when_neither_tenant_nor_default_configured():
    registry = QuotaPolicyRegistry()
    with pytest.raises(KeyError, match="resource_class='vision'"):
        registry.resolve(resource_class="vision", tenant_scope_key="user:alice")


def test_registry_default_fallback_per_resource_class_only():
    """A default for ``llm`` must NOT serve as fallback for ``embedding``.

    Resource classes are independent — declaring an LLM default does
    not implicitly cap embeddings.
    """
    registry = QuotaPolicyRegistry()
    registry.register(
        resource_class="llm",
        tenant_scope_key=DEFAULT_TENANT_FALLBACK,
        policy=QuotaPolicy(capacity=10, refill_rate_per_sec=1),
    )
    with pytest.raises(KeyError, match="embedding"):
        registry.resolve(resource_class="embedding", tenant_scope_key="user:alice")


# ---------------------------------------------------------------------
# Group 3: InMemoryQuotaBackend token bucket semantics
# ---------------------------------------------------------------------


class _FakeClock:
    """Hand-cranked monotonic clock so timing assertions are deterministic."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_backend(
    clock: _FakeClock | None = None, *, capacity: float = 3, refill_rate: float = 1.0
) -> tuple[InMemoryQuotaBackend, _FakeClock]:
    """Build an in-memory backend with one ``llm`` policy of ``capacity`` /
    ``refill_rate``, plus a fake clock that the test drives manually.
    """
    fake_clock = clock or _FakeClock()
    registry = QuotaPolicyRegistry()
    registry.register(
        resource_class="llm",
        tenant_scope_key=DEFAULT_TENANT_FALLBACK,
        policy=QuotaPolicy(capacity=capacity, refill_rate_per_sec=refill_rate),
    )
    backend = InMemoryQuotaBackend(registry, clock=fake_clock)
    return backend, fake_clock


@pytest.mark.asyncio
async def test_initial_bucket_starts_at_capacity_and_burst_drains_immediately():
    backend, _ = _make_backend(capacity=3, refill_rate=1.0)
    # Three back-to-back acquires must all return immediately because
    # the bucket starts full.
    for _ in range(3):
        await asyncio.wait_for(
            backend.acquire(resource_class="llm", tenant_scope_key="user:test"),
            timeout=1.0,
        )


@pytest.mark.asyncio
async def test_drained_bucket_blocks_until_refill_under_fake_clock():
    """An empty bucket must block ``acquire`` until enough tokens
    refill; we drive the fake clock + monkey-patch ``asyncio.sleep``
    to assert the wait time matches the refill math without actually
    waiting.
    """
    fake_clock = _FakeClock()
    backend, _ = _make_backend(fake_clock, capacity=2, refill_rate=2.0)
    # Drain.
    for _ in range(2):
        await backend.acquire(resource_class="llm", tenant_scope_key="user:test")

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # Advance the fake clock so the next loop iteration's refill
        # math sees the elapsed time.
        fake_clock.advance(seconds)

    import aperag.indexing.quota as quota_mod

    real_sleep = quota_mod.asyncio.sleep
    quota_mod.asyncio.sleep = _fake_sleep  # type: ignore[assignment]
    try:
        await backend.acquire(resource_class="llm", tenant_scope_key="user:test")
    finally:
        quota_mod.asyncio.sleep = real_sleep  # type: ignore[assignment]

    # Bucket capacity 2 / refill 2 / sec → 0.5s between tokens once empty.
    # The first sleep should be approximately the deficit / rate ≈ 0.5s.
    assert sleep_calls
    assert 0.4 <= sleep_calls[0] <= 0.6, sleep_calls


@pytest.mark.asyncio
async def test_buckets_per_tenant_are_isolated():
    backend, _ = _make_backend(capacity=1, refill_rate=0.1)

    # Drain alice's bucket.
    await backend.acquire(resource_class="llm", tenant_scope_key="user:alice")
    # Bob's bucket is independent — the first acquire must still be
    # immediate because it has its own full bucket.
    await asyncio.wait_for(
        backend.acquire(resource_class="llm", tenant_scope_key="user:bob"),
        timeout=1.0,
    )


@pytest.mark.asyncio
async def test_buckets_per_resource_class_are_isolated():
    """Different resource classes have independent token state."""
    fake_clock = _FakeClock()
    registry = QuotaPolicyRegistry()
    registry.register(
        resource_class="llm",
        tenant_scope_key=DEFAULT_TENANT_FALLBACK,
        policy=QuotaPolicy(capacity=1, refill_rate_per_sec=0.1),
    )
    registry.register(
        resource_class="embedding",
        tenant_scope_key=DEFAULT_TENANT_FALLBACK,
        policy=QuotaPolicy(capacity=1, refill_rate_per_sec=0.1),
    )
    backend = InMemoryQuotaBackend(registry, clock=fake_clock)

    # Drain LLM bucket.
    await backend.acquire(resource_class="llm", tenant_scope_key="user:test")
    # Embedding bucket is independent.
    await asyncio.wait_for(
        backend.acquire(resource_class="embedding", tenant_scope_key="user:test"),
        timeout=1.0,
    )


@pytest.mark.asyncio
async def test_refill_caps_at_capacity_after_long_idle():
    """An idle bucket does not accumulate tokens beyond its capacity —
    after a long quiet period, only ``capacity`` tokens are available."""
    fake_clock = _FakeClock()
    backend, _ = _make_backend(fake_clock, capacity=3, refill_rate=1.0)

    # Drain.
    for _ in range(3):
        await backend.acquire(resource_class="llm", tenant_scope_key="user:test")

    # 1 hour of idle: would refill to 3600 tokens if unbounded; capped at 3.
    fake_clock.advance(3600.0)
    for _ in range(3):
        await asyncio.wait_for(
            backend.acquire(resource_class="llm", tenant_scope_key="user:test"),
            timeout=1.0,
        )

    # The 4th acquire must block (cap holds).
    sleep_called = asyncio.Event()

    async def _capture_sleep(_: float) -> None:
        sleep_called.set()
        await asyncio.sleep(0)  # yield once so the test can observe

    import aperag.indexing.quota as quota_mod

    real_sleep = quota_mod.asyncio.sleep
    quota_mod.asyncio.sleep = _capture_sleep  # type: ignore[assignment]
    task = asyncio.create_task(backend.acquire(resource_class="llm", tenant_scope_key="user:test"))
    try:
        await asyncio.wait_for(sleep_called.wait(), timeout=1.0)
    finally:
        quota_mod.asyncio.sleep = real_sleep  # type: ignore[assignment]
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass

    assert sleep_called.is_set()


@pytest.mark.asyncio
async def test_default_fallback_routes_unknown_tenant_through_shared_pool():
    """An unknown tenant resolves to the ``"default"`` policy — bucket
    state is per-tenant-key BUT every unknown tenant shares the same
    *capacity / refill rate*, not the same bucket state, since the
    bucket key is ``(resource_class, tenant_scope_key)``.
    """
    backend, _ = _make_backend(capacity=1, refill_rate=0.1)

    # Two unknown tenants — neither has an override, both fall through
    # to the default policy. They draw from independent buckets keyed
    # by their respective tenant_scope_key.
    await backend.acquire(resource_class="llm", tenant_scope_key="user:tenant_x")
    # The second tenant's first acquire must still be immediate.
    await asyncio.wait_for(
        backend.acquire(resource_class="llm", tenant_scope_key="user:tenant_y"),
        timeout=1.0,
    )


# ---------------------------------------------------------------------
# Group 4: Bulkhead (§H.6)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulkhead_timeout_completes_within_budget():
    async with bulkhead_timeout(1.0, label="test.fast"):
        await asyncio.sleep(0.001)
    # No exception raised — completion is the success signal.


@pytest.mark.asyncio
async def test_bulkhead_timeout_raises_when_exceeded():
    with pytest.raises(TimeoutError):
        async with bulkhead_timeout(0.05, label="test.slow"):
            await asyncio.sleep(1.0)


def test_reject_if_oversize_accepts_at_boundary():
    # Equal-to-cap is allowed; only strictly over the cap is rejected.
    reject_if_oversize(UPLOAD_MAX_BYTES, label="boundary")


def test_reject_if_oversize_rejects_strictly_over_cap():
    with pytest.raises(ValueError, match="exceeds"):
        reject_if_oversize(UPLOAD_MAX_BYTES + 1, label="oversize")


def test_limits_constants_match_design_pack():
    """Pin the §H.6 hard ceilings so a regression that quietly relaxes
    them fails CI loudly. If the design pack updates, this test must
    update in lockstep with the spec amendment."""
    assert LLM_CALL_TIMEOUT_SECONDS == 60.0
    assert EMBEDDING_CALL_TIMEOUT_SECONDS == 30.0
    assert UPLOAD_MAX_BYTES == 50 * 1024 * 1024


# ---------------------------------------------------------------------
# Group 5: RedisQuotaBackend construction + Lua integration smoke
# ---------------------------------------------------------------------


def test_redis_backend_constructs_and_exposes_acquire():
    """Pin the import / construction path. Full integration belongs
    in T2.3 load-test infra (real Redis fixture)."""
    redis_client = MagicMock()
    registry = QuotaPolicyRegistry()
    backend = RedisQuotaBackend(redis_client, registry)
    assert backend is not None
    # ``acquire`` is the Protocol-required method — checking it is
    # async + accepts the keyword arguments locks the API surface.
    assert asyncio.iscoroutinefunction(backend.acquire)


@pytest.mark.asyncio
async def test_redis_backend_runs_lua_script_and_acquires_when_token_available():
    """Drive the Redis backend with a fake Lua client to verify the
    acquire returns immediately when the script reports a token was
    granted."""
    fake_script_calls: list[dict[str, Any]] = []

    class _FakeScript:
        def __call__(self, keys: list[str], args: list[Any]) -> list[int]:
            fake_script_calls.append({"keys": keys, "args": args})
            # Simulate "token available, acquired".
            return [1, 0]

    fake_client = MagicMock()
    fake_client.register_script.return_value = _FakeScript()

    registry = QuotaPolicyRegistry()
    registry.register(
        resource_class="llm",
        tenant_scope_key=DEFAULT_TENANT_FALLBACK,
        policy=QuotaPolicy(capacity=10, refill_rate_per_sec=1.0),
    )
    backend = RedisQuotaBackend(fake_client, registry)

    await asyncio.wait_for(
        backend.acquire(resource_class="llm", tenant_scope_key="user:test"),
        timeout=1.0,
    )
    assert len(fake_script_calls) == 1
    assert fake_script_calls[0]["args"][0] == 10  # capacity
    assert fake_script_calls[0]["args"][1] == 1.0  # refill_rate


@pytest.mark.asyncio
async def test_redis_backend_loops_when_lua_reports_wait():
    """When the Lua script returns ``[0, wait_seconds]`` the backend
    must sleep and retry. Verify two-call loop."""
    call_count = {"n": 0}

    class _FakeScript:
        def __call__(self, keys: list[str], args: list[Any]) -> list[float]:
            call_count["n"] += 1
            # First call: bucket empty, wait 0.001s. Second call:
            # token granted.
            if call_count["n"] == 1:
                return [0, 0.001]
            return [1, 0]

    fake_client = MagicMock()
    fake_client.register_script.return_value = _FakeScript()

    registry = QuotaPolicyRegistry()
    registry.register(
        resource_class="llm",
        tenant_scope_key=DEFAULT_TENANT_FALLBACK,
        policy=QuotaPolicy(capacity=10, refill_rate_per_sec=1.0),
    )
    backend = RedisQuotaBackend(fake_client, registry)

    await asyncio.wait_for(
        backend.acquire(resource_class="llm", tenant_scope_key="user:test"),
        timeout=2.0,
    )
    assert call_count["n"] == 2
