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

"""Read-primitive cache facade for D10.g (task #99).

The four parse_version-keyed primitives in :mod:`aperag.mcp.tools` —
``read_document`` / ``read_document_outline`` / ``read_document_section``
/ ``read_document_chunk`` — are the §E.3 budget hot path (PDF cold parse
1-3s, OCR cold parse 5-30s). :class:`ReadPrimitiveCache` is the seam
those primitives can call once tenancy + authorization have already run
to amortize the parse cost on subsequent reads of the same parsed view.

§E.7 hard lock — the cache **never** runs tenancy / authorization; it
trusts that the caller has already done so. The single exception is
``read_document_chunk`` whose key is ``(chunk_id,)`` only (per §E.6
chunk_id is indexing-immutable and does not need parse_version
weighting).

Cache key shape (one namespace per primitive):

* ``d10:outline:<document_id>:<parse_version>``
* ``d10:section:<document_id>:<parse_version>:<section_path or "">:<heading_anchor or "">``
* ``d10:content:<document_id>:<parse_version>``
* ``d10:chunk:<chunk_id>``

Values are JSON-serialized via Pydantic ``model_dump_json()`` and
validated back via ``Model.model_validate_json()``. A serialization or
deserialization failure is treated as a cache miss (logged) — the
authoritative store is the source of truth.

Wiring (D10.c primitive bodies, post Phase 2B follow-up): callers pass
the cache instance + a no-arg ``compute`` callable that performs the
authoritative fetch:

.. code-block:: python

    section = await cache.get_or_compute_section(
        document_id=document.id,
        parse_version=parse_version,
        section_path=section_path,
        heading_anchor=heading_anchor,
        compute=lambda: _fetch_section_authoritative(...),
    )

The cache instance itself is constructed at app bootstrap via
:func:`build_read_primitive_cache`; tests build their own instances
with fake L2 stores.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from aperag.cache.parse_version_cache import (
    AsyncRedisLike,
    CacheLayer,
    L1Cache,
    L2Cache,
    NoopL2Cache,
)

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


# Reserved cache namespaces. Keep in sync with the design pack §E
# canonical key prefixes; the invalidation helpers grep on these.
NAMESPACE_OUTLINE = "outline"
NAMESPACE_SECTION = "section"
NAMESPACE_CONTENT = "content"
NAMESPACE_CHUNK = "chunk"


def _build_key(namespace: str, *parts: str) -> str:
    """Build a canonical cache key.

    Empty parts are kept (rendered as empty segments) so that distinct
    inputs ``(section_path=None, heading_anchor="x")`` and
    ``(section_path="x", heading_anchor=None)`` never collide. A
    surrounding caller normalizes ``None`` to ``""`` before passing in.
    """

    return ":".join(("d10", namespace, *parts))


class ReadPrimitiveCache:
    """Two-tier (L1 ∘ L2) read-primitive cache.

    The class is intentionally thin: it composes two
    :class:`~aperag.cache.parse_version_cache.CacheLayer` instances and
    handles the JSON encode / decode for Pydantic models. It exposes
    one ``get_or_compute_*`` per primitive so callers do not build
    cache keys by hand and so the test surface stays small.

    Per §E.1 sequencing (read primitive → L1 → L2 → compute → write
    L2 + L1), ``get_or_compute_*`` writes back to **both** layers on a
    miss so the next request from any worker hits L1 directly while
    other workers can hit the shared L2.
    """

    def __init__(
        self,
        *,
        l1: CacheLayer,
        l2: CacheLayer,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        # Per-key compute lock so two concurrent miss-paths for the
        # same key don't both pay the cold-parse cost. Bounded growth
        # matches the L1 cache size in practice (entries are removed
        # after the compute resolves).
        self._inflight: dict[str, asyncio.Task] = {}
        self._inflight_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Per-primitive helpers (typed)
    # ------------------------------------------------------------------

    async def get_or_compute_outline(
        self,
        *,
        document_id: str,
        parse_version: str,
        compute: Callable[[], Awaitable[ModelT]],
        model_cls: type[ModelT],
    ) -> ModelT:
        key = _build_key(NAMESPACE_OUTLINE, document_id, parse_version)
        return await self._get_or_compute(key=key, compute=compute, model_cls=model_cls)

    async def get_or_compute_section(
        self,
        *,
        document_id: str,
        parse_version: str,
        section_path: Optional[str],
        heading_anchor: Optional[str],
        compute: Callable[[], Awaitable[ModelT]],
        model_cls: type[ModelT],
    ) -> ModelT:
        key = _build_key(
            NAMESPACE_SECTION,
            document_id,
            parse_version,
            section_path or "",
            heading_anchor or "",
        )
        return await self._get_or_compute(key=key, compute=compute, model_cls=model_cls)

    async def get_or_compute_content(
        self,
        *,
        document_id: str,
        parse_version: str,
        compute: Callable[[], Awaitable[ModelT]],
        model_cls: type[ModelT],
    ) -> ModelT:
        key = _build_key(NAMESPACE_CONTENT, document_id, parse_version)
        return await self._get_or_compute(key=key, compute=compute, model_cls=model_cls)

    async def get_or_compute_chunk(
        self,
        *,
        chunk_id: str,
        compute: Callable[[], Awaitable[ModelT]],
        model_cls: type[ModelT],
    ) -> ModelT:
        # §E.6: chunk_id is indexing-layer-immutable, so there is no
        # parse_version dimension on this namespace.
        key = _build_key(NAMESPACE_CHUNK, chunk_id)
        return await self._get_or_compute(key=key, compute=compute, model_cls=model_cls)

    # ------------------------------------------------------------------
    # Core compose: L1 → L2 → compute
    # ------------------------------------------------------------------

    async def _get_or_compute(
        self,
        *,
        key: str,
        compute: Callable[[], Awaitable[ModelT]],
        model_cls: type[ModelT],
    ) -> ModelT:
        # 1. L1 lookup (per worker, fastest path).
        l1_value = await self._l1.get(key)
        if l1_value is not None:
            decoded = self._decode(l1_value, model_cls=model_cls)
            if decoded is not None:
                return decoded
            # Treat undecodable L1 entry as a miss; evict and proceed.
            await self._l1.delete(key)

        # 2. L2 lookup (shared across workers, second-fastest path).
        l2_value = await self._l2.get(key)
        if l2_value is not None:
            decoded = self._decode(l2_value, model_cls=model_cls)
            if decoded is not None:
                # Re-warm L1 so subsequent reads on this worker skip L2.
                await self._l1.set(key, l2_value)
                return decoded
            await self._l2.delete(key)

        # 3. Compute under per-key lock so concurrent miss requests
        # collapse onto a single authoritative fetch.
        async with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is None:
                existing = asyncio.create_task(self._compute_and_store(key=key, compute=compute))
                self._inflight[key] = existing
        try:
            value = await existing
        finally:
            async with self._inflight_lock:
                # Only the original creator removes the entry — concurrent
                # awaiters may have already seen another task overwrite it.
                if self._inflight.get(key) is existing:
                    self._inflight.pop(key, None)

        if not isinstance(value, model_cls):
            # Defensive: compute path must return the declared type.
            raise TypeError(f"D10.g cache compute returned {type(value).__name__}, expected {model_cls.__name__}")
        return value

    async def _compute_and_store(
        self,
        *,
        key: str,
        compute: Callable[[], Awaitable[ModelT]],
    ) -> ModelT:
        value = await compute()
        try:
            payload = value.model_dump_json()
        except Exception as e:  # pragma: no cover - Pydantic encode is robust
            logger.warning("D10.g cache encode failed for %s: %s", key, e)
            return value
        # Write-back: L2 first (shared) so other workers see it sooner;
        # then L1 so this worker's next read is L1-cheap.
        await self._l2.set(key, payload)
        await self._l1.set(key, payload)
        return value

    @staticmethod
    def _decode(payload: str, *, model_cls: type[ModelT]) -> Optional[ModelT]:
        try:
            return model_cls.model_validate_json(payload)
        except ValidationError as e:
            logger.warning(
                "D10.g cache decode failed for model %s: %s — treating as miss",
                model_cls.__name__,
                e,
            )
            return None
        except Exception as e:  # pragma: no cover - JSON / unexpected
            logger.warning(
                "D10.g cache decode raised %s for model %s — treating as miss",
                type(e).__name__,
                model_cls.__name__,
            )
            return None

    # ------------------------------------------------------------------
    # Layers (exposed for tests / invalidation helpers)
    # ------------------------------------------------------------------

    @property
    def l1(self) -> CacheLayer:
        return self._l1

    @property
    def l2(self) -> CacheLayer:
        return self._l2


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


def build_read_primitive_cache(
    *,
    redis: Optional[AsyncRedisLike] = None,
    l1_size: int,
    l2_ttl_seconds: int,
) -> ReadPrimitiveCache:
    """Build a :class:`ReadPrimitiveCache` for production use.

    ``redis`` is keyword-only and may be ``None`` in environments where
    Redis is not configured (dev / unit tests); a :class:`NoopL2Cache`
    is substituted so the API surface stays the same. ``l1_size`` and
    ``l2_ttl_seconds`` come from :class:`aperag.config.Config` knobs.
    """

    l1 = L1Cache(maxsize=l1_size)
    l2: CacheLayer
    if redis is None:
        l2 = NoopL2Cache()
    else:
        l2 = L2Cache(redis=redis, ttl_seconds=l2_ttl_seconds)
    return ReadPrimitiveCache(l1=l1, l2=l2)


__all__ = [
    "NAMESPACE_CHUNK",
    "NAMESPACE_CONTENT",
    "NAMESPACE_OUTLINE",
    "NAMESPACE_SECTION",
    "ReadPrimitiveCache",
    "build_read_primitive_cache",
]
