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

"""L1 / L2 cache layer contract tests for read-primitive ."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from aperag.cache.parse_version_cache import L1Cache, L2Cache, NoopL2Cache

# ---------------------------------------------------------------------------
# L1 LRU
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l1_get_miss_returns_none():
    cache = L1Cache(maxsize=4)
    assert await cache.get("missing") is None


@pytest.mark.asyncio
async def test_l1_set_then_get_hits():
    cache = L1Cache(maxsize=4)
    await cache.set("k", "v")
    assert await cache.get("k") == "v"


@pytest.mark.asyncio
async def test_l1_evicts_least_recently_used():
    cache = L1Cache(maxsize=2)
    await cache.set("a", "A")
    await cache.set("b", "B")
    # Access "a" so "b" becomes least recently used.
    await cache.get("a")
    await cache.set("c", "C")
    assert await cache.get("b") is None
    assert await cache.get("a") == "A"
    assert await cache.get("c") == "C"


@pytest.mark.asyncio
async def test_l1_overwrite_updates_value_and_moves_to_end():
    cache = L1Cache(maxsize=2)
    await cache.set("a", "A1")
    await cache.set("b", "B")
    await cache.set("a", "A2")  # update + move to end
    await cache.set("c", "C")  # evicts b, keeps a + c
    assert await cache.get("a") == "A2"
    assert await cache.get("b") is None
    assert await cache.get("c") == "C"


@pytest.mark.asyncio
async def test_l1_delete_removes_key():
    cache = L1Cache(maxsize=2)
    await cache.set("a", "A")
    await cache.delete("a")
    assert await cache.get("a") is None


@pytest.mark.asyncio
async def test_l1_delete_namespace_only_purges_matching_prefix():
    cache = L1Cache(maxsize=8)
    await cache.set("read_primitive:outline:doc-1:v1", "x")
    await cache.set("read_primitive:outline:doc-2:v1", "y")
    await cache.set("read_primitive:section:doc-1:v1::a", "z")
    await cache.set("read_primitive:chunk:c-1", "k")

    await cache.delete_namespace("outline")

    assert await cache.get("read_primitive:outline:doc-1:v1") is None
    assert await cache.get("read_primitive:outline:doc-2:v1") is None
    # Other namespaces unaffected.
    assert await cache.get("read_primitive:section:doc-1:v1::a") == "z"
    assert await cache.get("read_primitive:chunk:c-1") == "k"


@pytest.mark.asyncio
async def test_l1_rejects_zero_or_negative_maxsize():
    with pytest.raises(ValueError):
        L1Cache(maxsize=0)
    with pytest.raises(ValueError):
        L1Cache(maxsize=-1)


# ---------------------------------------------------------------------------
# L2 Redis-backed (with fake client)
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal in-memory async Redis stand-in for tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, Optional[int]]] = []

    async def get(self, key: str) -> Optional[bytes]:
        v = self.store.get(key)
        return v.encode("utf-8") if v is not None else None

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self.store[key] = value
        self.set_calls.append((key, value, ex))

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
        return n

    async def scan_iter(self, match: Optional[str] = None) -> Any:
        # Match on simple "prefix*" form — sufficient for read-primitive key shape.
        if match is None:
            keys = list(self.store.keys())
        else:
            assert match.endswith("*"), "fake supports prefix glob only"
            prefix = match[:-1]
            keys = [k for k in self.store.keys() if k.startswith(prefix)]
        for k in keys:
            yield k


@pytest.mark.asyncio
async def test_l2_set_uses_configured_ttl():
    fake = _FakeRedis()
    l2 = L2Cache(redis=fake, ttl_seconds=42)
    await l2.set("k", "v")
    assert fake.set_calls == [("k", "v", 42)]


@pytest.mark.asyncio
async def test_l2_get_returns_decoded_value():
    fake = _FakeRedis()
    l2 = L2Cache(redis=fake, ttl_seconds=60)
    await l2.set("k", "hello")
    assert await l2.get("k") == "hello"


@pytest.mark.asyncio
async def test_l2_get_missing_returns_none():
    fake = _FakeRedis()
    l2 = L2Cache(redis=fake, ttl_seconds=60)
    assert await l2.get("nope") is None


@pytest.mark.asyncio
async def test_l2_delete_namespace_purges_matching_prefix_only():
    fake = _FakeRedis()
    l2 = L2Cache(redis=fake, ttl_seconds=60)
    await l2.set("read_primitive:outline:doc-1:v1", "x")
    await l2.set("read_primitive:outline:doc-2:v1", "y")
    await l2.set("read_primitive:section:doc-1:v1::a", "z")
    await l2.set("read_primitive:chunk:c-1", "k")

    await l2.delete_namespace("outline")

    assert await l2.get("read_primitive:outline:doc-1:v1") is None
    assert await l2.get("read_primitive:outline:doc-2:v1") is None
    assert await l2.get("read_primitive:section:doc-1:v1::a") == "z"
    assert await l2.get("read_primitive:chunk:c-1") == "k"


@pytest.mark.asyncio
async def test_l2_rejects_zero_or_negative_ttl():
    fake = _FakeRedis()
    with pytest.raises(ValueError):
        L2Cache(redis=fake, ttl_seconds=0)
    with pytest.raises(ValueError):
        L2Cache(redis=fake, ttl_seconds=-30)


@pytest.mark.asyncio
async def test_l2_get_failure_falls_through_as_miss():
    class _BrokenRedis(_FakeRedis):
        async def get(self, key: str) -> Optional[bytes]:  # type: ignore[override]
            raise RuntimeError("redis is down")

    l2 = L2Cache(redis=_BrokenRedis(), ttl_seconds=60)
    # Fail-open: a backend error returns None (cache miss), it does
    # not propagate. The read primitive falls through to compute.
    assert await l2.get("k") is None


# ---------------------------------------------------------------------------
# Noop L2 (no-Redis dev / test fallback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noop_l2_always_misses():
    noop = NoopL2Cache()
    await noop.set("k", "v")
    assert await noop.get("k") is None
    await noop.delete("k")
    await noop.delete_namespace("outline")
