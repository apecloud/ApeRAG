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

"""Integration tests for ``RedisWorkQueue`` — Wave 4 T4.

Pin the Wave 4 #6 production-readiness invariant: when
``INDEXING_QUEUE_BACKEND=redis`` (multi-process scale-out per design
pack §E.2), enqueue/dequeue is BLPOP-atomic across multiple workers
sharing the same key — no payload is delivered to two workers, no
payload is silently dropped.

Tests skip when the configured Redis URL is unreachable so the
non-Redis CI lane (lint-and-unit) stays green; the e2e-http-compose CI
lane brings a real Redis up, so these tests run there.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from aperag.indexing import Modality, RedisWorkQueue

# Use a separate Redis logical DB (15) for tests so a leftover key from
# a flaky run never shadows production / other test suites.
_REDIS_URL = os.environ.get(
    "TEST_INDEXING_QUEUE_REDIS_URL",
    "redis://default:password@127.0.0.1:6379/15",
)


def _redis_reachable(url: str) -> bool:
    """Synchronous reachability probe so the skip decision is taken
    before pytest tries to schedule the async test on the event loop.
    """

    try:
        import redis  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — `redis` is a hard dep
        return False

    try:
        client = redis.from_url(url, socket_connect_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


_REDIS_OK = _redis_reachable(_REDIS_URL)
pytestmark = pytest.mark.skipif(
    not _REDIS_OK,
    reason=f"Redis at {_REDIS_URL} unreachable; skipping RedisWorkQueue integration suite",
)


@pytest.fixture
async def queue():
    """Fresh ``RedisWorkQueue`` per test, with a per-test key prefix
    that keys won't collide between concurrent runs.
    """

    q = RedisWorkQueue(redis_url=_REDIS_URL)
    # Per-run key suffix so parallel CI / re-runs don't see each
    # other's leftover payloads.
    suffix = uuid.uuid4().hex[:8]
    original_template = RedisWorkQueue.KEY_TEMPLATE
    RedisWorkQueue.KEY_TEMPLATE = f"q:test:{suffix}:{{modality}}"
    try:
        yield q
    finally:
        # Drain any leftover keys this test wrote so the test DB doesn't
        # accumulate cruft, then restore the production-shape template.
        client = await q._get_client()
        for modality in Modality:
            await client.delete(RedisWorkQueue._key(modality))
        await q.close()
        RedisWorkQueue.KEY_TEMPLATE = original_template


@pytest.mark.asyncio
async def test_push_then_pop_roundtrips_payload(queue):
    """Single producer + single consumer roundtrip — payload bytes
    survive JSON encode/decode through Redis identity.
    """

    payload = {
        "index_id": 42,
        "document_id": "doc-1",
        "parse_version": "abc123",
        "modality": Modality.VECTOR.value,
        "source_path": "collections/c/documents/doc-1/derived/parse_abc123/chunks.jsonl",
        "collection_id": "c",
    }
    await queue.push(modality=Modality.VECTOR, payload=payload)
    received = await queue.pop(modality=Modality.VECTOR, timeout_seconds=2)
    assert received == payload


@pytest.mark.asyncio
async def test_pop_returns_none_on_timeout(queue):
    """Empty queue + timeout → ``None`` (not a hang, not a raise).
    Exercise the worker BLPOP-then-check-shutdown loop pattern.
    """

    received = await queue.pop(modality=Modality.VECTOR, timeout_seconds=1)
    assert received is None


@pytest.mark.asyncio
async def test_each_payload_delivered_to_exactly_one_consumer(queue):
    """Multi-consumer demux — N consumers BLPOP'ing the same key
    receive N disjoint payloads when N payloads are pushed.

    This is the §E.2 multi-process scale-out invariant Wave 4 T4
    enables: production deploys multiple FastAPI/worker pods, each
    BLPOP'ing the same queue, and Redis must atomically demux so no
    payload runs twice.
    """

    n = 5
    pushed = [
        {
            "index_id": i,
            "document_id": f"doc-{i}",
            "parse_version": f"pv-{i:04d}",
            "modality": Modality.FULLTEXT.value,
            "source_path": f"path-{i}",
            "collection_id": "c",
        }
        for i in range(n)
    ]
    for payload in pushed:
        await queue.push(modality=Modality.FULLTEXT, payload=payload)

    # Spawn N consumers concurrently — each BLPOPs once. The set union
    # of received payloads should equal the set of pushed payloads
    # (no duplicates, no losses).
    async def consume() -> dict | None:
        return await queue.pop(modality=Modality.FULLTEXT, timeout_seconds=2)

    received = await asyncio.gather(*(consume() for _ in range(n)))
    assert all(item is not None for item in received), "all consumers must receive a payload"
    received_index_ids = sorted(item["index_id"] for item in received)
    assert received_index_ids == list(range(n)), "every push must reach exactly one consumer (no duplicate, no loss)"


@pytest.mark.asyncio
async def test_per_modality_queue_isolation(queue):
    """Pushes to ``modality=VECTOR`` must not be visible to a
    consumer popping ``modality=FULLTEXT`` — each modality has its
    own BLPOP key.
    """

    await queue.push(
        modality=Modality.VECTOR,
        payload={
            "index_id": 1,
            "document_id": "d",
            "parse_version": "v",
            "modality": Modality.VECTOR.value,
            "source_path": "p",
            "collection_id": "c",
        },
    )
    # Pop on a different modality — should time out.
    other = await queue.pop(modality=Modality.FULLTEXT, timeout_seconds=1)
    assert other is None
    # Pop on the right modality should still see our payload.
    own = await queue.pop(modality=Modality.VECTOR, timeout_seconds=1)
    assert own is not None
    assert own["modality"] == Modality.VECTOR.value


@pytest.mark.asyncio
async def test_qsize_reflects_pending_payloads(queue):
    """``qsize`` exposes the current backlog length so the §J.1
    ``queue_depth`` SLI emitter and tests can assert payloads
    landed without consuming them.
    """

    assert await queue.qsize(Modality.SUMMARY) == 0
    for i in range(3):
        await queue.push(
            modality=Modality.SUMMARY,
            payload={
                "index_id": i,
                "document_id": "d",
                "parse_version": "v",
                "modality": Modality.SUMMARY.value,
                "source_path": "p",
                "collection_id": "c",
            },
        )
    assert await queue.qsize(Modality.SUMMARY) == 3
    await queue.pop(modality=Modality.SUMMARY, timeout_seconds=1)
    assert await queue.qsize(Modality.SUMMARY) == 2


@pytest.mark.asyncio
async def test_close_releases_underlying_client(queue):
    """``close`` returns the client to a re-connectable state — calling
    a method after ``close`` lazy-reopens the connection. Exercise the
    FastAPI lifespan shutdown path that calls ``await queue.close()``
    on SIGTERM.
    """

    await queue.push(
        modality=Modality.GRAPH,
        payload={
            "index_id": 1,
            "document_id": "d",
            "parse_version": "v",
            "modality": Modality.GRAPH.value,
            "source_path": "p",
            "collection_id": "c",
        },
    )
    await queue.close()
    # After close, _client is None and a subsequent operation must
    # transparently reconnect.
    assert queue._client is None
    received = await queue.pop(modality=Modality.GRAPH, timeout_seconds=2)
    assert received is not None
