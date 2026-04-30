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
5. **Score normalization (task #61 P0-B)**: ``SearchHit.score`` is a
   uniform similarity in ``[0, 1]`` where higher is better, regardless
   of the underlying distance metric (cosine / euclid / dot). Callers and
   FE/MCP/API integrations do not branch on metric to interpret the
   score. ``QueryRequest.score_threshold`` is on the same scale.
6. **Filter fail-loud (task #61 P0-A)**: an unsupported ``VectorFilter``
   shape is a programmer error and must raise
   :class:`UnsupportedFilterError` synchronously; backends never silently
   drop the filter and return unfiltered results.

Everything else (index construction knobs, connection pooling, payload
serialization quirks) is the backend's implementation detail.
"""

from __future__ import annotations

import math
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

# ---------------------------------------------------------------------------
# Cross-adapter exception types
# ---------------------------------------------------------------------------


class UnsupportedFilterError(TypeError):
    """Raised when a ``VectorFilter`` node is not supported by the backend.

    Per task #61 P0-A: every backend translator MUST raise this (rather
    than silently logging + returning an unfiltered query) when it
    encounters a filter shape it cannot translate. Subclassing
    :class:`TypeError` keeps backwards compatibility with callers that
    already ``except TypeError`` from the pgvector translator.
    """


# ---------------------------------------------------------------------------
# Cross-adapter score normalization (task #61 P0-B)
# ---------------------------------------------------------------------------
#
# These helpers operate on the **canonical "higher-is-better raw"
# convention**, which is what pgvector's ``_score_expr`` already produces
# directly and what Qdrant's native score is for cosine + dot. Qdrant's
# euclid distance is the asymmetric case — the SDK returns the *positive*
# L2 distance (smaller = better), so the Qdrant adapter negates at the
# boundary before feeding the helper (see ``qdrant_connector.search``).
# Adapters are responsible for converting their backend's raw score to
# this canonical convention; ``normalize_score`` does the math only.
#
# Canonical raw conventions assumed below:
#
# * cosine   — similarity, ≈[0, 1] for unit-norm embeddings, [-1, 1] in
#              general. Both adapters already produce similarity
#              directly: pgvector ``1 - <=>``, Qdrant native ``p.score``.
# * euclid   — *negative* L2 distance, range ``(-inf, 0]``, so that
#              higher = closer. pgvector emits this via ``-(<->)``;
#              Qdrant returns positive L2 natively and the adapter
#              negates at the boundary.
# * dot      — raw inner product, range ``(-inf, +inf)``. Both adapters
#              emit this directly: pgvector via ``-(<#>)`` (since the
#              ``<#>`` operator returns the negated inner product),
#              Qdrant via native ``p.score``.
#
# To honour the §5 contract above we squash all three onto ``[0, 1]`` with
# higher = better. The transforms below preserve the ranking (monotone in
# the underlying score) so callers / tests can rely on top-k order being
# unchanged versus the raw form.

_SCORE_FLOOR = 0.0
_SCORE_CEIL = 1.0


def normalize_score(distance: str, raw: float) -> float:
    """Map a backend's raw search score to ``[0, 1]``, higher = better.

    The transformation is monotone non-decreasing per metric, so
    *ordering* is preserved — callers that only care about top-k get the
    same set in the same order regardless of normalization.

    Cosine
        Pass-through, clamped to ``[0, 1]`` to absorb float drift /
        non-unit-norm inputs that occasionally produce tiny negatives or
        slight overshoots above 1.0.

    Euclid
        Input is "negative L2 distance" (≤ 0). Map to ``(0, 1]`` via
        ``1 / (1 + L2_distance)`` which approaches 1 as the points
        coincide and 0 as they grow apart.

    Dot
        Input is the raw inner product on ``(-inf, +inf)``. Map via the
        logistic sigmoid ``1 / (1 + exp(-x))`` which is bijective onto
        ``(0, 1)`` and monotone, so ranking is preserved while keeping
        the score on the same scale as cosine / euclid.
    """
    if distance == "cosine":
        return max(_SCORE_FLOOR, min(_SCORE_CEIL, float(raw)))
    if distance == "euclid":
        # raw should be ≤ 0 (negated distance); guard against drift.
        l2 = max(0.0, -float(raw))
        return 1.0 / (1.0 + l2)
    if distance == "dot":
        # Sigmoid; numerically stable for large |x|.
        x = float(raw)
        if x >= 0:
            ex = math.exp(-x)
            return 1.0 / (1.0 + ex)
        ex = math.exp(x)
        return ex / (1.0 + ex)
    raise ValueError(f"Unknown distance metric: {distance!r}")


def denormalize_threshold_to_native(distance: str, normalized: float) -> float:
    """Invert :func:`normalize_score` so a backend-native ``score_threshold``
    pushdown produces the same set of survivors as a Python post-filter
    on the normalized score.

    Used when the backend's native query supports a fast pre-filter (e.g.
    Qdrant's ``query_points(score_threshold=...)`` or pgvector's
    ``WHERE score >= ...``). The point is to keep the network / scan
    boundary exactly equivalent to ``[h for h in raw_hits if normalize(
    h.score) >= normalized_threshold]`` while letting the database do
    the work.

    Returns the raw-score threshold to pass to the backend; clamping /
    pinning at infinity is done so the caller can pass it verbatim.
    """
    n = max(_SCORE_FLOOR, min(_SCORE_CEIL, float(normalized)))
    if distance == "cosine":
        # Cosine raw == cosine normalized for our purposes (clamp).
        return n
    if distance == "euclid":
        # normalize = 1/(1+l2), raw = -l2 → raw = -(1/n - 1) = (n - 1)/n.
        if n <= 0.0:
            return -math.inf  # threshold is "anything ≥ 0", L2 ≤ +inf
        if n >= 1.0:
            return 0.0  # threshold is exactly "L2 == 0"
        return (n - 1.0) / n
    if distance == "dot":
        # normalize = sigmoid(raw) → raw = log(n / (1-n)).
        if n <= 0.0:
            return -math.inf
        if n >= 1.0:
            return math.inf
        return math.log(n / (1.0 - n))
    raise ValueError(f"Unknown distance metric: {distance!r}")


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

        Score semantics (task #61 P0-B contract):

        * ``SearchHit.score`` is a normalized similarity in ``[0, 1]``,
          higher = better, regardless of the underlying metric (cosine,
          euclid, dot). Adapters apply :func:`normalize_score` before
          returning so callers can compare scores across backends and
          metrics on a single scale.
        * ``request.score_threshold`` is on the same ``[0, 1]`` scale.
          When set, the connector returns only hits with
          ``hit.score >= score_threshold`` after normalization.

        Raises :class:`UnsupportedFilterError` (a ``TypeError`` subclass)
        synchronously if ``request.flt`` contains a node the backend
        cannot translate; backends never silently drop the filter.
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
