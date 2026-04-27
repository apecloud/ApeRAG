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

"""Explicit invalidation helpers for D10.g (task #99)."""

from __future__ import annotations

import pytest

from aperag.cache.invalidation import invalidate_collection, invalidate_document
from aperag.cache.parse_version_cache import L1Cache, NoopL2Cache
from aperag.cache.read_primitive_cache import ReadPrimitiveCache


def _make_cache() -> ReadPrimitiveCache:
    return ReadPrimitiveCache(l1=L1Cache(maxsize=16), l2=NoopL2Cache())


@pytest.mark.asyncio
async def test_invalidate_document_purges_parse_version_namespaces():
    cache = _make_cache()
    # Pre-populate L1 with sample entries across all namespaces.
    await cache.l1.set("d10:outline:doc-1:0000000000000001:6", "x")
    await cache.l1.set("d10:section:doc-1:0000000000000001::a", "y")
    await cache.l1.set("d10:content:doc-1:0000000000000001", "z")
    await cache.l1.set("d10:chunk:col-1:doc-1:c-1", "k")

    await invalidate_document(cache, document_id="doc-1")

    assert await cache.l1.get("d10:outline:doc-1:0000000000000001:6") is None
    assert await cache.l1.get("d10:section:doc-1:0000000000000001::a") is None
    assert await cache.l1.get("d10:content:doc-1:0000000000000001") is None
    assert await cache.l1.get("d10:chunk:col-1:doc-1:c-1") is None


@pytest.mark.asyncio
async def test_invalidate_collection_purges_parse_version_namespaces():
    cache = _make_cache()
    await cache.l1.set("d10:outline:doc-1:0000000000000001:6", "x")
    await cache.l1.set("d10:section:doc-2:0000000000000002::a", "y")
    await cache.l1.set("d10:content:doc-3:0000000000000003", "z")
    await cache.l1.set("d10:chunk:col-1:doc-1:c-1", "k")

    await invalidate_collection(cache, collection_id="col-1")

    assert await cache.l1.get("d10:outline:doc-1:0000000000000001:6") is None
    assert await cache.l1.get("d10:section:doc-2:0000000000000002::a") is None
    assert await cache.l1.get("d10:content:doc-3:0000000000000003") is None
    assert await cache.l1.get("d10:chunk:col-1:doc-1:c-1") is None
