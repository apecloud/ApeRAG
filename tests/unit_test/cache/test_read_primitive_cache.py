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

"""ReadPrimitiveCache facade contract tests for D10.g (task #99)."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from aperag.cache.parse_version_cache import L1Cache, NoopL2Cache
from aperag.cache.read_primitive_cache import (
    NAMESPACE_CHUNK,
    NAMESPACE_CONTENT,
    NAMESPACE_OUTLINE,
    NAMESPACE_SECTION,
    ReadPrimitiveCache,
)


# Lightweight stand-in for the real Pydantic envelopes
# (DocumentOutline / DocumentSection / DocumentContent / DocumentChunk).
# The cache must remain agnostic to the actual response model, so the
# tests pin behavior with a minimal stand-in.
class _Payload(BaseModel):
    document_id: str
    parse_version: str
    body: str


def _make_cache() -> ReadPrimitiveCache:
    return ReadPrimitiveCache(l1=L1Cache(maxsize=16), l2=NoopL2Cache())


@pytest.mark.asyncio
async def test_first_call_invokes_compute_subsequent_hits_l1():
    cache = _make_cache()
    calls: list[int] = []

    async def compute() -> _Payload:
        calls.append(1)
        return _Payload(document_id="doc", parse_version="abc1234567890def", body="x")

    p1 = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="abc1234567890def",
        compute=compute,
        model_cls=_Payload,
    )
    p2 = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="abc1234567890def",
        compute=compute,
        model_cls=_Payload,
    )
    assert p1 == p2
    assert len(calls) == 1, "second call must hit L1, not invoke compute"


@pytest.mark.asyncio
async def test_distinct_parse_versions_compute_independently():
    cache = _make_cache()
    calls: list[str] = []

    async def make_compute(version: str):
        async def _compute() -> _Payload:
            calls.append(version)
            return _Payload(document_id="doc", parse_version=version, body=version)

        return _compute

    p_v1 = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="0000000000000001",
        compute=await make_compute("0000000000000001"),
        model_cls=_Payload,
    )
    p_v2 = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="0000000000000002",
        compute=await make_compute("0000000000000002"),
        model_cls=_Payload,
    )

    assert p_v1.body == "0000000000000001"
    assert p_v2.body == "0000000000000002"
    assert calls == ["0000000000000001", "0000000000000002"]


@pytest.mark.asyncio
async def test_outline_cache_keys_include_max_depth():
    cache = _make_cache()
    calls: list[int] = []

    async def compute(depth: int):
        async def _compute() -> _Payload:
            calls.append(depth)
            return _Payload(document_id="doc", parse_version="0000000000000001", body=f"depth-{depth}")

        return _compute

    shallow = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="0000000000000001",
        max_depth=1,
        compute=await compute(1),
        model_cls=_Payload,
    )
    deep = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="0000000000000001",
        max_depth=6,
        compute=await compute(6),
        model_cls=_Payload,
    )

    assert shallow.body == "depth-1"
    assert deep.body == "depth-6"
    assert calls == [1, 6]


@pytest.mark.asyncio
async def test_section_cache_keys_include_section_path_and_anchor():
    cache = _make_cache()
    counter = {"n": 0}

    async def compute(label: str):
        async def _c() -> _Payload:
            counter["n"] += 1
            return _Payload(document_id="doc", parse_version="0000000000000001", body=label)

        return _c

    a = await cache.get_or_compute_section(
        document_id="doc",
        parse_version="0000000000000001",
        section_path="1/2",
        heading_anchor=None,
        compute=await compute("by-path"),
        model_cls=_Payload,
    )
    b = await cache.get_or_compute_section(
        document_id="doc",
        parse_version="0000000000000001",
        section_path=None,
        heading_anchor="#chapter-2",
        compute=await compute("by-anchor"),
        model_cls=_Payload,
    )
    # Distinct identifying keys MUST NOT collide.
    assert a.body == "by-path"
    assert b.body == "by-anchor"
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_chunk_cache_key_includes_collection_document_and_chunk_id():
    """Chunk ids are not guaranteed globally unique across tenants or
    documents, so the cache key must include the full read scope.
    """

    cache = _make_cache()

    async def compute() -> _Payload:
        return _Payload(document_id="doc", parse_version="0000000000000001", body="ok")

    await cache.get_or_compute_chunk(
        collection_id="col-1",
        document_id="doc-1",
        chunk_id="c-1",
        compute=compute,
        model_cls=_Payload,
    )
    raw_key = "d10:chunk:col-1:doc-1:c-1"
    assert await cache.l1.get(raw_key) is not None


@pytest.mark.asyncio
async def test_concurrent_misses_collapse_onto_single_compute():
    cache = _make_cache()
    started = asyncio.Event()
    proceed = asyncio.Event()
    invocations = 0

    async def compute() -> _Payload:
        nonlocal invocations
        invocations += 1
        started.set()
        await proceed.wait()
        return _Payload(document_id="doc", parse_version="aaaaaaaaaaaaaaaa", body="x")

    async def call() -> _Payload:
        return await cache.get_or_compute_content(
            document_id="doc",
            parse_version="aaaaaaaaaaaaaaaa",
            compute=compute,
            model_cls=_Payload,
        )

    t1 = asyncio.create_task(call())
    t2 = asyncio.create_task(call())
    await started.wait()
    proceed.set()
    p1, p2 = await asyncio.gather(t1, t2)
    assert p1 == p2
    assert invocations == 1


@pytest.mark.asyncio
async def test_compute_must_return_declared_model_class():
    cache = _make_cache()

    class _Other(BaseModel):
        x: int

    async def bad_compute() -> _Other:
        return _Other(x=1)

    with pytest.raises(TypeError):
        await cache.get_or_compute_outline(
            document_id="doc",
            parse_version="0000000000000001",
            compute=bad_compute,  # type: ignore[arg-type]
            model_cls=_Payload,
        )


@pytest.mark.asyncio
async def test_undecodable_l1_entry_falls_back_to_compute():
    cache = _make_cache()
    bad_key = f"d10:{NAMESPACE_OUTLINE}:doc:0000000000000001:6"
    # Manually inject corrupted JSON to simulate a partial deploy with a
    # newer model_cls than the stored payload was serialized for.
    await cache.l1.set(bad_key, "{not valid json")

    async def compute() -> _Payload:
        return _Payload(
            document_id="doc",
            parse_version="0000000000000001",
            body="recomputed",
        )

    result = await cache.get_or_compute_outline(
        document_id="doc",
        parse_version="0000000000000001",
        compute=compute,
        model_cls=_Payload,
    )
    assert result.body == "recomputed"
    # And the corrupt entry was evicted (replaced with the new payload).
    assert (await cache.l1.get(bad_key)) is not None
    assert "recomputed" in (await cache.l1.get(bad_key))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_namespace_constants_match_design_pack_keys():
    # Sanity: the public constants must match the spec literals so the
    # invalidation helpers and the cache surface stay in sync.
    assert NAMESPACE_OUTLINE == "outline"
    assert NAMESPACE_SECTION == "section"
    assert NAMESPACE_CONTENT == "content"
    assert NAMESPACE_CHUNK == "chunk"
