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

"""Abstract contract every vector-DB backend implements.

This is the whole contract — six methods, all of them backend-neutral in
their signature. A concrete implementation (``QdrantVectorStoreConnector``,
``PgvectorVectorStoreConnector``, …) is correct iff:

1. It accepts/returns the DTOs in ``aperag.vectorstore.dto`` only.
2. It never lets a backend-specific type (``qdrant_client.models.*``,
   ``psycopg.*``, ``pymilvus.*``) leak into the return values or the
   raised exception messages that reach the caller.
3. It is **multitenant-safe**: in multitenant mode every read/write/delete
   enforces the tenant guard based on the connector's ``TenantRef``.
   Passing another tenant's ids to ``delete()`` or ``retrieve()`` must be
   a no-op or filtered-out, never a data leak.
4. ``ensure_collection()`` is idempotent and safe to call on every
   connector instantiation.

Everything else (index construction knobs, connection pooling, payload
serialization quirks) is the backend's implementation detail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from aperag.vectorstore.dto import (
    QueryRequest,
    SearchHit,
    TenantRef,
    VectorPoint,
    VectorShape,
)
from aperag.vectorstore.filters import VectorFilter


class VectorStoreConnector(ABC):
    """Abstract contract for per-tenant vector storage.

    The constructor receives a ``ctx`` dict so we don't have to rewrite
    every ``build_vector_db_context`` caller every time the parameter
    list changes. Subclasses pick out the keys they need and ignore the
    rest; unknown keys must never be a hard error.
    """

    def __init__(self, ctx: Dict[str, Any], **_kwargs: Any) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------ meta
    @property
    @abstractmethod
    def tenant(self) -> TenantRef:
        """Tenant this connector is bound to."""

    @property
    @abstractmethod
    def shape(self) -> VectorShape:
        """Vector shape this connector routes to."""

    # ------------------------------------------------------------ collection
    @abstractmethod
    def ensure_collection(self) -> None:
        """Idempotently make sure the physical storage (Qdrant collection,
        pgvector table, …) exists for this connector's shape.

        Must be safe to call from many connectors concurrently: typical
        implementations use ``CREATE IF NOT EXISTS`` / ``collection_exists
        ? no-op : create`` with module-level deduping caches.
        """

    @abstractmethod
    def drop_tenant(self, *, purge_all_shards: bool = False) -> None:
        """Remove all of this tenant's vectors.

        * Multitenant layouts: delete all points with ``collection_id ==
          self.tenant.id``; the shared physical collection stays alive.
        * Legacy per-tenant layouts: drop the physical collection entirely.
        * When ``purge_all_shards=True`` the connector should scan every
          shape-variant shard and delete any points tagged with this
          tenant, even if they live in a shard this connector doesn't
          route to. This is the safety net for "the embedding provider
          was removed, ``vector_size`` cannot be resolved any more" and
          must never touch shards that don't belong to the multitenant
          layout.
        """

    # ------------------------------------------------------------ write path
    @abstractmethod
    def upsert(self, points: Sequence[VectorPoint]) -> List[str]:
        """Insert (or overwrite on id conflict) the given points.

        Must inject the tenant guard into each point's storage so later
        searches / deletes can filter by it. Returns the ids written (in
        input order). Raises on write failure — callers treat the batch
        as atomic per-point.
        """

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Delete points by id.

        Defense-in-depth: the implementation must also match the tenant
        guard so that a caller passing another tenant's ids becomes a
        silent no-op rather than a data leak. No-op when ``ids`` is empty.
        """

    @abstractmethod
    def delete_by_filter(self, flt: VectorFilter) -> None:
        """Delete every point matching the filter (tenant guard AND-ed)."""

    # ------------------------------------------------------------- read path
    @abstractmethod
    def search(self, request: QueryRequest) -> List[SearchHit]:
        """Top-k nearest neighbors, optionally constrained by ``request.flt``.

        Score semantics:
        * Cosine: similarity in [-1, 1] — higher is better.
        * Dot: raw dot product — higher is better.
        * Euclid: negative L2 distance — higher is "closer to 0", better.

        ``request.score_threshold`` (when set) filters on the same scale,
        so callers don't have to branch on distance type.
        """

    @abstractmethod
    def retrieve(
        self,
        ids: Sequence[str],
        *,
        with_vectors: bool = False,
    ) -> List[VectorPoint]:
        """Fetch points by id. Enforces tenant guard: foreign-tenant ids
        are silently skipped."""
