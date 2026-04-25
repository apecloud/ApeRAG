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

"""Explicit cache invalidation triggers for D10.g (task #99).

§E.5 distinguishes implicit and explicit invalidation:

* **Implicit** — ``parse_version`` change produces a fresh key, so the
  old key is simply unreferenced and decays out of L1 / L2 by their
  natural eviction. Most invalidation flows naturally; D10.c does not
  need to call any helper here for a re-parse.
* **Explicit** — ``delete_document`` / ``rebuild_indexes`` /
  admin-flush actions that need to purge cached entries even though
  ``parse_version`` did not move (e.g., the document is being removed
  outright). Those callers go through this module.

D10.g first-cut **does not auto-wire** these helpers into any code
path; the actual wiring is the D11+ write-tools lane's responsibility
(per §E.5 + §G D10.g write-set boundary). The helpers are provided
here so that lane has a stable API to call instead of hand-rolling
``redis.scan_iter`` matches.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aperag.cache.read_primitive_cache import ReadPrimitiveCache

logger = logging.getLogger(__name__)


async def invalidate_document(
    cache: "ReadPrimitiveCache",
    *,
    document_id: str,
) -> None:
    """Purge cached entries that pin to a specific document.

    Removes ``d10:outline:{document_id}:*``,
    ``d10:section:{document_id}:*`` and ``d10:content:{document_id}:*``
    from both L1 and L2.

    NOTE: ``d10:chunk:*`` is keyed by ``chunk_id`` (not ``document_id``),
    so a per-document invalidation does **not** purge chunk entries.
    Callers that delete a document and need chunk-level invalidation
    must walk the document's chunk list and invalidate each chunk_id
    separately. This is intentional — keeps the chunk namespace
    indexing-immutable per §E.6 and avoids requiring the cache to know
    chunk-to-document mapping.
    """

    namespaces = ("outline", "section", "content")
    for ns in namespaces:
        # The current cache stores a flat key per ``(namespace,
        # document_id, parse_version, ...)`` so we cannot pinpoint a
        # single document_id without scanning. ``delete_namespace``
        # over-purges (drops all entries in the namespace), which is
        # the conservatively-correct path for D11+ write tools that
        # rarely happens. The cost is tolerable.
        # TODO(D11+ write tools): implement narrower "scan by prefix
        # ``d10:<ns>:<document_id>:``" once Redis SCAN match support is
        # exposed through ``AsyncRedisLike``.
        await cache.l1.delete_namespace(ns)
        await cache.l2.delete_namespace(ns)
    logger.info("D10.g cache: invalidated all entries for document_id=%s", document_id)


async def invalidate_collection(
    cache: "ReadPrimitiveCache",
    *,
    collection_id: str,
) -> None:
    """Purge cached entries that pin to a specific collection.

    Same conservative-over-purge semantics as
    :func:`invalidate_document` — the cache layer doesn't know which
    documents belong to which collection without an external mapping,
    so on D11+ write-tool calls we simply purge the parse-version-bound
    namespaces. ``d10:chunk:*`` again is not affected (chunk_id is
    indexing-immutable).
    """

    namespaces = ("outline", "section", "content")
    for ns in namespaces:
        await cache.l1.delete_namespace(ns)
        await cache.l2.delete_namespace(ns)
    logger.info("D10.g cache: invalidated all entries for collection_id=%s", collection_id)


__all__ = [
    "invalidate_collection",
    "invalidate_document",
]
