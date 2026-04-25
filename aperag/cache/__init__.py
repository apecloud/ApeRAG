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

"""D10.g read-primitive persistence cache (task #99).

Per ``docs/modularization/d10-design-pack.md`` §E (Lock #7), the read
primitives in :mod:`aperag.mcp.tools` MAY be wrapped by a two-tier cache
to amortize parse cost over repeated reads of the same parsed view of a
document. This package owns that cache, and only that cache.

Architecture (§E.1):

* **L1** — in-process LRU per worker (``functools.lru_cache``-backed
  ``LRUCacheStore``); bounded by ``Config.d10_cache_l1_size`` (default
  256).
* **L2** — Redis tier-2, keyed by ``parse_version`` so re-parsing
  produces a fresh key without explicit invalidation; TTL is
  ``Config.d10_cache_l2_ttl_seconds`` (default 3600 = 1h).

§E.7 hard lock — **the cache only accelerates; it never changes
semantics**. Tenancy and authorization gates run on every invocation
of a read primitive, before a cache lookup or compute. The cache is a
*post-gate* memo of the **content** that the primitive would have
returned; it is *not* a permission shortcut.

Public surface (re-exported here for callers in ``aperag.mcp.tools``):

* :class:`ReadPrimitiveCache` — the main facade (L1 ∘ L2).
* :class:`L1Cache` — in-process LRU layer (testable in isolation).
* :class:`L2Cache` — Redis-backed layer (testable with a fake client).
* :func:`build_read_primitive_cache` — production wiring helper.
* :func:`invalidate_document` / :func:`invalidate_collection` —
  explicit invalidation triggers reserved for the D11+ write tools.

This package does NOT cache:

* search primitives (``search_*``) — see §E spec, search has its own
  cache path and is out of D10.g scope (#96 D10.d territory);
* paginated list envelopes (``list_collections`` / ``list_documents``)
  — cursor-bound shape; cache the per-item primitives instead;
* metadata primitives (``get_collection_metadata`` /
  ``get_document_metadata``) — short-TTL caching for these is a future
  D10.g extension, not first-cut scope.
"""

from __future__ import annotations

from aperag.cache.invalidation import invalidate_collection, invalidate_document
from aperag.cache.parse_version_cache import (
    L1Cache,
    L2Cache,
    NoopL2Cache,
)
from aperag.cache.read_primitive_cache import (
    ReadPrimitiveCache,
    build_read_primitive_cache,
)
from aperag.cache.runtime import (
    get_read_primitive_cache,
    reset_read_primitive_cache,
)

__all__ = [
    "L1Cache",
    "L2Cache",
    "NoopL2Cache",
    "ReadPrimitiveCache",
    "build_read_primitive_cache",
    "get_read_primitive_cache",
    "invalidate_collection",
    "invalidate_document",
    "reset_read_primitive_cache",
]
