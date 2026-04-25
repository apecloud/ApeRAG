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

"""Contract tests for D9 §A7 raw-args backend-private cache (Phase 8 #75)."""

from __future__ import annotations

import pytest

from aperag.domains.agent_runtime.tools.args_cache import (
    InMemoryRawArgsCache,
    args_hash,
    args_preview,
)


@pytest.mark.asyncio
async def test_put_then_get_returns_raw_args():
    cache = InMemoryRawArgsCache()
    raw = {"path": "/tmp/secret", "content": "very-secret"}
    await cache.put("consent-1", raw)
    got = await cache.get("consent-1")
    assert got == raw


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_consent_id():
    cache = InMemoryRawArgsCache()
    assert await cache.get("never-stored") is None


@pytest.mark.asyncio
async def test_delete_removes_entry():
    cache = InMemoryRawArgsCache()
    await cache.put("c1", {"x": 1})
    assert await cache.delete("c1") is True
    assert await cache.get("c1") is None


@pytest.mark.asyncio
async def test_delete_idempotent_on_unknown_consent_id():
    cache = InMemoryRawArgsCache()
    assert await cache.delete("never") is False


@pytest.mark.asyncio
async def test_entry_expires_after_ttl():
    clock = [0.0]
    cache = InMemoryRawArgsCache(clock=lambda: clock[0])
    await cache.put("c1", {"x": 1}, ttl_seconds=10)
    clock[0] = 5.0
    assert await cache.get("c1") == {"x": 1}
    clock[0] = 11.0  # past TTL
    assert await cache.get("c1") is None


@pytest.mark.asyncio
async def test_cleanup_expired_evicts_only_past_ttl_entries():
    clock = [0.0]
    cache = InMemoryRawArgsCache(clock=lambda: clock[0])
    await cache.put("alive", {"x": 1}, ttl_seconds=100)
    await cache.put("dead", {"x": 2}, ttl_seconds=10)
    clock[0] = 50.0
    evicted = await cache.cleanup_expired()
    assert evicted == 1
    assert await cache.get("alive") == {"x": 1}
    assert await cache.get("dead") is None


@pytest.mark.asyncio
async def test_put_rejects_empty_consent_id():
    cache = InMemoryRawArgsCache()
    with pytest.raises(ValueError):
        await cache.put("", {"x": 1})


@pytest.mark.asyncio
async def test_put_rejects_non_positive_ttl():
    cache = InMemoryRawArgsCache()
    with pytest.raises(ValueError):
        await cache.put("c1", {"x": 1}, ttl_seconds=0)
    with pytest.raises(ValueError):
        await cache.put("c1", {"x": 1}, ttl_seconds=-1)


def test_args_preview_truncates_long_payload():
    big = {"k": "x" * 1000}
    preview = args_preview(big, limit=200)
    assert len(preview) <= 200 + len("...<truncated>")
    assert preview.endswith("...<truncated>")


def test_args_preview_does_not_truncate_short_payload():
    small = {"k": "v"}
    preview = args_preview(small)
    assert "<truncated>" not in preview


def test_args_hash_is_stable_and_key_order_independent():
    a = args_hash({"a": 1, "b": 2})
    b = args_hash({"b": 2, "a": 1})
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_args_hash_differs_on_value_change():
    a = args_hash({"x": 1})
    b = args_hash({"x": 2})
    assert a != b
