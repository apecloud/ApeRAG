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

"""Qdrant backend for the vector-store abstraction.

The connector supports two physical layouts, switchable per-context:

* **multitenant** (default, new): one Qdrant collection per ``(vector_size,
  distance)`` pair, shared by all ApeRAG collections. Every point carries a
  ``collection_id`` payload field used for tenant isolation. A keyword
  index with ``is_tenant=True`` is registered so the Qdrant optimizer
  groups points by tenant on disk, which keeps query cost comparable to
  per-tenant collections while cutting memory overhead by one to two
  orders of magnitude.

* **legacy** (``multitenant=false``): one Qdrant collection per ApeRAG
  collection (the historical behavior). Preserved for rollback during the
  migration window and for small fleets that don't yet want to re-ingest.

Both layouts implement the same ``VectorStoreConnector`` interface; no
caller needs to know which layout is active.

This module is **not** allowed to leak Qdrant SDK types out of its public
methods. All inputs/outputs are backend-neutral DTOs from
``aperag.vectorstore.dto``. Internal helpers (``_translate_filter``,
``_merge_tenant_filter``, …) work in ``qdrant_client.models.*`` and are
strictly implementation detail.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

import qdrant_client
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from aperag.vectorstore.base import (
    UnsupportedFilterError,
    VectorBackendCapabilities,
    VectorStoreConnector,
    denormalize_threshold_to_native,
    normalize_score,
)
from aperag.vectorstore.dto import (
    QueryRequest,
    SearchHit,
    TenantRef,
    VectorPoint,
    VectorShape,
)
from aperag.vectorstore.filters import And, Eq, In, IsEmpty, Not, Or, VectorFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# constants & helpers
# ---------------------------------------------------------------------------

# Payload field used as the tenant discriminator. When is_tenant=True is set
# on the keyword index over this field, Qdrant physically groups points with
# the same tenant into the same segments, which restores per-tenant locality
# while keeping only one collection.
TENANT_PAYLOAD_KEY = "collection_id"

# Prefix used for every multi-tenant shard. Kept in one place so the
# "purge_all_shards" safety net and tests can't drift from what
# ``global_collection_name`` writes.
_MULTITENANT_PREFIX = "aperag_vectors_"

# Module-level cache of physical collections already ensured within this
# process. Avoids an RPC on every connector instantiation. Thread-safe
# because it's only appended to, never mutated concurrently in conflicting
# ways.
_ENSURED_COLLECTIONS: set = set()
_ENSURE_LOCK = threading.Lock()


def global_collection_name(vector_size: int, distance: str) -> str:
    """Return the physical Qdrant collection name for a given vector shape.

    We partition by ``(vector_size, distance)`` because Qdrant requires
    vectors in a single collection to share both; different embedding
    models produce different vector shapes and must therefore live in
    distinct collections.
    """
    return f"{_MULTITENANT_PREFIX}{int(vector_size)}_{distance.lower()}"


def _coerce_distance(distance: Any) -> rest.Distance:
    """Normalize a distance value into ``rest.Distance``."""
    if isinstance(distance, rest.Distance):
        return distance
    name = str(distance).strip().upper()
    if name == "EUCLIDIAN":  # common typo we've seen in configs
        name = "EUCLID"
    try:
        return rest.Distance[name]
    except KeyError as e:
        raise ValueError(f"unsupported qdrant distance: {distance!r}") from e


def _quantization_config(cfg: Dict[str, Any]) -> Optional[rest.QuantizationConfig]:
    """Build a QuantizationConfig from ctx fields (or return None)."""
    if not cfg.get("quantization_enabled", False):
        return None
    qtype = str(cfg.get("quantization_type", "int8")).lower()
    if qtype == "int8":
        return rest.ScalarQuantization(
            scalar=rest.ScalarQuantizationConfig(
                type=rest.ScalarType.INT8,
                quantile=float(cfg.get("quantization_quantile", 0.99)),
                always_ram=bool(cfg.get("quantization_always_ram", True)),
            )
        )
    if qtype in ("binary", "bin"):
        return rest.BinaryQuantization(
            binary=rest.BinaryQuantizationConfig(always_ram=bool(cfg.get("quantization_always_ram", True)))
        )
    raise ValueError(f"unsupported qdrant quantization type: {qtype!r}")


def _hnsw_config(cfg: Dict[str, Any]) -> rest.HnswConfigDiff:
    return rest.HnswConfigDiff(
        m=int(cfg.get("hnsw_m", 16)),
        ef_construct=int(cfg.get("hnsw_ef_construct", 100)),
        on_disk=bool(cfg.get("hnsw_on_disk", True)),
    )


def _optimizers_config(cfg: Dict[str, Any]) -> rest.OptimizersConfigDiff:
    return rest.OptimizersConfigDiff(
        default_segment_number=int(cfg.get("default_segment_number", 2)),
        memmap_threshold=int(cfg.get("mmap_threshold_kb", 20480)),
    )


def _ensure_tenant_payload_index(client: Any, collection_name: str) -> None:
    """Register the keyword index on the tenant payload field.

    Tries ``is_tenant=True`` first (Qdrant >= 1.11, enables segment-level
    defragmentation so queries touch only per-tenant blocks). Falls back
    to a plain keyword index on older servers (Qdrant 1.10) — multitenancy
    still works at the filter level, just without the storage-layout
    optimization. Idempotent.
    """
    # Attempt the optimized shape first.
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=TENANT_PAYLOAD_KEY,
            field_schema=rest.KeywordIndexParams(
                type=rest.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
        return
    except UnexpectedResponse as e:
        msg = str(e).lower()
        if "already exists" in msg:
            return
        logger.warning(
            "qdrant: is_tenant keyword index rejected on %s (%s). "
            "Falling back to plain keyword index; multitenancy filter still works "
            "but per-tenant segment defragmentation (Qdrant >= 1.11) is unavailable.",
            collection_name,
            e,
        )

    # Plain fallback.
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=TENANT_PAYLOAD_KEY,
            field_schema=rest.KeywordIndexParams(type=rest.KeywordIndexType.KEYWORD),
        )
    except UnexpectedResponse as e:
        if "already exists" in str(e).lower():
            return
        logger.warning(
            "qdrant: plain keyword index on %s/%s also failed: %s",
            collection_name,
            TENANT_PAYLOAD_KEY,
            e,
        )
        # Last resort: legacy "field_schema as string" API for very old servers.
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=TENANT_PAYLOAD_KEY,
                field_schema="keyword",
            )
        except UnexpectedResponse as e2:
            if "already exists" not in str(e2).lower():
                raise


def _merge_tenant_filter(user_filter: Any, tenant_id: Optional[str]) -> Any:
    """Combine an externally provided filter with the tenant guard.

    The tenant guard is always required in multitenant mode. If the caller
    already passed a ``rest.Filter``, we wrap it under ``must`` so both
    constraints are AND-ed. If the caller passed nothing, we just return
    the tenant filter.
    """
    if not tenant_id:
        return user_filter
    tenant_clause = rest.FieldCondition(key=TENANT_PAYLOAD_KEY, match=rest.MatchValue(value=tenant_id))
    if user_filter is None:
        return rest.Filter(must=[tenant_clause])
    if isinstance(user_filter, rest.Filter):
        must = list(user_filter.must or [])
        must.append(tenant_clause)
        return rest.Filter(
            must=must,
            should=user_filter.should,
            must_not=user_filter.must_not,
            min_should=user_filter.min_should,
        )
    # Unknown filter shape: log and pass through only the tenant guard. A
    # stricter alternative would be to raise, but this path only runs when
    # a pre-abstraction caller sneaks in; we prefer the guard to still
    # protect them rather than hard-fail.
    logger.warning(
        "qdrant: dropping user_filter of unsupported type %s when applying tenant guard",
        type(user_filter).__name__,
    )
    return rest.Filter(must=[tenant_clause])


# ---------------------------------------------------------------------------
# VectorFilter DSL -> qdrant_client.models.Filter
# ---------------------------------------------------------------------------


def _translate_filter(flt: Optional[VectorFilter]) -> Optional[rest.Filter]:
    """Convert a backend-neutral DSL tree into a Qdrant ``Filter``.

    Returns ``None`` for a ``None`` input so callers don't need to check.
    Always returns a top-level ``Filter`` (never a raw Condition) so that
    ``_merge_tenant_filter`` has a consistent shape to work with.
    """
    if flt is None:
        return None
    if isinstance(flt, Eq):
        return rest.Filter(must=[rest.FieldCondition(key=flt.key, match=rest.MatchValue(value=flt.value))])
    if isinstance(flt, In):
        if not flt.values:
            raise ValueError(f"In filter on key {flt.key!r} has empty values list")
        return rest.Filter(must=[rest.FieldCondition(key=flt.key, match=rest.MatchAny(any=list(flt.values)))])
    if isinstance(flt, IsEmpty):
        return rest.Filter(must=[rest.IsEmptyCondition(is_empty=rest.PayloadField(key=flt.key))])
    if isinstance(flt, And):
        subs = [_translate_filter(p) for p in flt.parts]
        subs = [s for s in subs if s is not None]
        return rest.Filter(must=subs)
    if isinstance(flt, Or):
        subs = [_translate_filter(p) for p in flt.parts]
        subs = [s for s in subs if s is not None]
        # task #61 P1-V3 defense-in-depth: ``Or.__post_init__`` already
        # rejects empty ``parts`` at DSL construction, so this list is
        # normally non-empty. The translator-level guard catches future
        # refactors that bypass the DSL constructor (e.g.
        # ``dataclasses.replace(or_node, parts=())``) before they reach
        # Qdrant. Without this, ``rest.Filter(should=[])`` is a vacuous
        # disjunction that Qdrant treats as "match everything" — a
        # silent data-correctness footgun. Cross-adapter parity with
        # the pgvector ``_SqlFilter._walk`` Or-empty guard.
        if not subs:
            raise UnsupportedFilterError(
                "qdrant: Or filter has zero translatable parts after pruning; "
                "an empty Or is a vacuous disjunction that would match every "
                "point in the collection (task #61 P1-V3)."
            )
        return rest.Filter(should=subs)
    if isinstance(flt, Not):
        sub = _translate_filter(flt.inner)
        return rest.Filter(must_not=[sub] if sub is not None else [])

    raise TypeError(
        f"Unsupported VectorFilter node: {type(flt).__name__}. "
        "Add a branch in aperag.vectorstore.qdrant_connector._translate_filter"
    )


def _normalize_filter_input(flt: Any) -> Optional[rest.Filter]:
    """Accept a DSL node, an already-translated Qdrant Filter, or None.

    The ``rest.Filter`` pass-through exists solely for the migration
    script (``scripts/migrate_qdrant_multitenancy.py``) which hand-rolls
    native filters. Once the script stops doing that this branch can go.
    Normal production callers go through the DSL path.

    Per task #61 P0-A contract: an unsupported shape raises
    :class:`UnsupportedFilterError` synchronously rather than being
    silently dropped. Returning ``None`` here would degrade the search
    into an unfiltered full-collection scan — a correctness footgun, not
    a graceful degradation.
    """
    if flt is None:
        return None
    if isinstance(flt, rest.Filter):
        return flt
    if isinstance(flt, (Eq, In, IsEmpty, And, Or, Not)):
        return _translate_filter(flt)
    raise UnsupportedFilterError(
        f"qdrant: unsupported filter input of type {type(flt).__name__}; "
        "expected one of VectorFilter DSL nodes (Eq/In/IsEmpty/And/Or/Not), "
        "an already-translated qdrant_client.models.Filter, or None."
    )


# ---------------------------------------------------------------------------
# QdrantClient process-level pool
# ---------------------------------------------------------------------------
#
# Historical behavior: every ``QdrantVectorStoreConnector(ctx)`` built a
# fresh ``QdrantClient``, which opens its own HTTP/gRPC connection pool
# (and, in HTTPS mode, a fresh TLS handshake). The cache below keys by the
# subset of ctx that actually identifies the destination endpoint, so a
# read-heavy request path only pays the TCP/TLS setup cost once per
# endpoint per process.
#
# ``:memory:`` clients are deliberately NOT cached — they back per-test
# isolated stores. Sharing them across tests would be a subtle bug magnet.
_CLIENT_CACHE: Dict[tuple, qdrant_client.QdrantClient] = {}
_CLIENT_LOCK = threading.Lock()


def _client_cache_key(
    url: str,
    port: int,
    grpc_port: int,
    prefer_grpc: bool,
    https: bool,
    api_key: Optional[str],
) -> tuple:
    return (url, int(port), int(grpc_port), bool(prefer_grpc), bool(https), api_key or "")


def _get_or_create_client(
    url: str,
    port: int = 6333,
    grpc_port: int = 6334,
    prefer_grpc: bool = False,
    https: bool = False,
    api_key: Optional[str] = None,
    timeout: int = 300,
    **extra: Any,
) -> qdrant_client.QdrantClient:
    """Return a process-shared QdrantClient for the given endpoint."""
    if url == ":memory:":
        return qdrant_client.QdrantClient(":memory:")

    key = _client_cache_key(url, port, grpc_port, prefer_grpc, https, api_key)
    cached = _CLIENT_CACHE.get(key)
    if cached is not None:
        return cached

    with _CLIENT_LOCK:
        cached = _CLIENT_CACHE.get(key)
        if cached is not None:
            return cached
        client = qdrant_client.QdrantClient(
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            timeout=timeout,
            **extra,
        )
        _CLIENT_CACHE[key] = client
        return client


def _reset_client_cache() -> None:
    """Clear the process-level client cache. Tests only."""
    with _CLIENT_LOCK:
        _CLIENT_CACHE.clear()


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


def _coerce_point_id(raw: Any) -> str:
    """Return a stable string id for a Qdrant Record / ScoredPoint.

    Qdrant returns ids as int or UUID depending on how they were written;
    we always expose string ids at the abstraction boundary.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return str(raw)


def _extract_vector(p: Any) -> Optional[List[float]]:
    """Pull the ``vector`` off a Qdrant Record / ScoredPoint, normalising to
    ``list[float]`` (single-vector collections only — multi-vector is a
    feature we never enable)."""
    v = getattr(p, "vector", None)
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, dict) and v:
        first = next(iter(v.values()))
        return first if isinstance(first, list) else None
    return None


class QdrantVectorStoreConnector(VectorStoreConnector):
    """Qdrant implementation of ``VectorStoreConnector``."""

    #: Static capability declaration (task #61 P1-V2 / P1-V4).
    #: Qdrant's ``client.upsert(points, wait=True)`` is best-effort
    #: per-point — a mid-batch failure can leave a partial write — so
    #: ``supports_atomic_batch_upsert=False``. The legacy
    #: (``multitenant=False``) layout is supported, where each ApeRAG
    #: collection gets its own physical Qdrant collection; new
    #: deployments default to multitenant.
    BACKEND_CAPABILITIES = VectorBackendCapabilities(
        supports_atomic_batch_upsert=False,
        supports_legacy_mode=True,
    )

    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)

        # --------- layout flags
        self.multitenant: bool = bool(ctx.get("multitenant", True))
        self.cfg = ctx  # alias used by config builders

        # --------- tenant
        tenant_raw = ctx.get("collection")
        if self.multitenant and not tenant_raw:
            # Silent fallback would cross-tenant-write into a pseudo-tenant.
            # Refuse loudly.
            raise ValueError(
                "QdrantVectorStoreConnector(multitenant=True) requires ctx['collection'] "
                "(the ApeRAG collection id used as tenant key); got empty/missing."
            )
        tenant_id = str(tenant_raw) if tenant_raw else "collection"
        self._tenant = TenantRef(id=tenant_id)

        # --------- shape
        self._shape = VectorShape(
            size=int(ctx.get("vector_size", 1536)),
            distance=str(ctx.get("distance", "Cosine")),
        )

        # --------- endpoint
        self.url = ctx.get("url", "http://localhost")
        self.port = ctx.get("port", 6333)
        self.grpc_port = ctx.get("grpc_port", 6334)
        self.prefer_grpc = ctx.get("prefer_grpc", False)
        self.https = ctx.get("https", False)
        self.timeout = ctx.get("timeout", 300)

        # --------- physical name
        if self.multitenant:
            self.collection_name = global_collection_name(self._shape.size, self._shape.canonical)
        else:
            # Legacy layout: physical collection name == tenant id.
            self.collection_name = self._tenant.id

        # --------- client (pool-reused)
        self.client = _get_or_create_client(
            url=self.url,
            port=self.port,
            grpc_port=self.grpc_port,
            prefer_grpc=self.prefer_grpc,
            https=self.https,
            api_key=ctx.get("api_key"),
            timeout=self.timeout,
            **kwargs,
        )

        # --------- ensure collection eagerly.
        #
        # We *could* defer this until the first write / read, but the
        # abstraction contract says ``ensure_collection`` must be safe in
        # __init__ and every production caller immediately follows
        # connector construction with a write or search. The per-process
        # ``_ENSURED_COLLECTIONS`` cache makes repeat calls effectively
        # free.
        self.ensure_collection()

    # ---- convenience aliases for internal code that pre-dates tenant/shape
    @property
    def tenant(self) -> TenantRef:
        return self._tenant

    @property
    def shape(self) -> VectorShape:
        return self._shape

    @property
    def tenant_id(self) -> str:
        """Backward-compat alias. New code should use ``self.tenant.id``."""
        return self._tenant.id

    @property
    def vector_size(self) -> int:
        """Backward-compat alias for the migration script / tests."""
        return self._shape.size

    @property
    def distance(self) -> str:
        """Backward-compat alias returning the original (non-normalized)
        distance string the caller supplied."""
        return self.cfg.get("distance", "Cosine")

    # ================================================================ ensure
    def ensure_collection(self) -> None:
        """Idempotently create the physical Qdrant collection + tenant index.

        Thread-safe and process-wide-cached: calling this from many
        connectors or many workers in the same process results in exactly
        one RPC per (url, port, collection_name) after the first caller.
        """
        cache_key = f"{self.url}:{self.port}:{self.collection_name}"
        if cache_key in _ENSURED_COLLECTIONS:
            return

        with _ENSURE_LOCK:
            if cache_key in _ENSURED_COLLECTIONS:
                return
            try:
                exists = self.client.collection_exists(self.collection_name)
            except Exception:
                logger.exception("qdrant: collection_exists check failed for %s", self.collection_name)
                raise

            if not exists:
                logger.info(
                    "qdrant: creating collection %s (size=%d, distance=%s, multitenant=%s)",
                    self.collection_name,
                    self._shape.size,
                    self._shape.canonical,
                    self.multitenant,
                )
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=rest.VectorParams(
                            size=self._shape.size,
                            distance=_coerce_distance(self._shape.canonical),
                            on_disk=bool(self.cfg.get("vectors_on_disk", True)),
                        ),
                        hnsw_config=_hnsw_config(self.cfg),
                        optimizers_config=_optimizers_config(self.cfg),
                        quantization_config=_quantization_config(self.cfg),
                        on_disk_payload=bool(self.cfg.get("on_disk_payload", True)),
                    )
                except UnexpectedResponse as e:
                    # Another process raced us; accept the existing collection.
                    if "already exists" not in str(e).lower():
                        raise
                    logger.info("qdrant: collection %s already exists (race ok)", self.collection_name)

            if self.multitenant:
                try:
                    _ensure_tenant_payload_index(self.client, self.collection_name)
                except Exception:
                    logger.exception(
                        "qdrant: failed to create tenant index on %s (will retry next time)",
                        self.collection_name,
                    )
                    raise

            _ENSURED_COLLECTIONS.add(cache_key)

    # ================================================================ upsert
    def upsert(self, points: Sequence[VectorPoint]) -> List[str]:
        """Insert / overwrite ``points`` into the Qdrant collection.

        Implementation notes:
        * Writes are native (``client.upsert`` with ``rest.PointStruct``);
          we intentionally do **not** go through
          ``llama_index.vector_stores.QdrantVectorStore.add(nodes)`` so
          this backend is symmetric with pgvector / Milvus, none of which
          have a LlamaIndex adapter wired in.
        * In multitenant mode the tenant guard (``collection_id``) is
          injected into every point's payload before sending. Callers that
          already set it are respected; the tenant connector would not
          write a conflicting value to a different tenant's id anyway.
        """
        if not points:
            return []

        structs: List[rest.PointStruct] = []
        ids: List[str] = []
        for p in points:
            payload = dict(p.payload)
            if self.multitenant:
                # Defense-in-depth: overwrite any stale ``collection_id``
                # that doesn't match this connector's tenant. Cheaper than
                # verifying and refusing to write; cross-tenant writes via
                # mis-labeled points are too risky to allow.
                payload[TENANT_PAYLOAD_KEY] = self._tenant.id
            # Qdrant accepts ``str`` (UUID or int-as-string) or ``int`` for
            # point id but NOT a ``uuid.UUID`` instance. Our VectorPoint.id
            # is already ``str``; pass through unchanged.
            structs.append(rest.PointStruct(id=p.id, vector=list(p.vector), payload=payload))
            ids.append(p.id)

        self.client.upsert(
            collection_name=self.collection_name,
            points=structs,
            wait=True,
        )
        return ids

    # ================================================================ search
    def search(self, request: QueryRequest) -> List[SearchHit]:
        """Top-k nearest neighbors, tenant-guarded in multitenant mode."""
        filter_obj = _normalize_filter_input(request.flt)
        if self.multitenant:
            filter_obj = _merge_tenant_filter(filter_obj, self._tenant.id)

        # hints accepted from callers — we read only the ones Qdrant
        # understands, ignore everything else. This keeps portability:
        # a caller that passes pgvector-only hints won't crash when
        # swapped onto Qdrant.
        hints = request.hints or {}
        search_params = None
        hnsw_ef = hints.get("hnsw_ef")
        exact = hints.get("exact")
        if hnsw_ef is not None or exact is not None:
            search_params = rest.SearchParams(
                hnsw_ef=int(hnsw_ef) if hnsw_ef is not None else None,
                exact=bool(exact) if exact is not None else None,
            )
        consistency = hints.get("consistency", "majority")

        # ``request.score_threshold`` is on the normalized [0, 1] scale per
        # base.py P0-B contract; invert it to Qdrant's raw-score units so
        # the server-side cutoff matches what a Python post-filter on the
        # normalized score would produce.
        #
        # Convention asymmetry caught by Weston (msg=86e05a8e): the shared
        # ``normalize_score`` / ``denormalize_threshold_to_native`` helpers
        # assume **higher-is-better raw**, which matches Qdrant's native
        # convention for cosine + dot but not euclid — Qdrant returns
        # positive L2 distance (lower = better) and ``score_threshold``
        # is interpreted as an *upper* bound for euclid. So we negate at
        # the Qdrant boundary for the euclid case only: the raw score
        # before normalize, and the native threshold before the RPC.
        is_euclid = self._shape.canonical == "euclid"
        native_threshold: Optional[float] = None
        if request.score_threshold is not None:
            inv = denormalize_threshold_to_native(self._shape.canonical, float(request.score_threshold))
            # Qdrant rejects ``inf``; ``-inf`` means "all rows pass" so
            # we just omit the param.
            if inv == float("inf"):
                # Threshold of 1.0 on dot/euclid → unreachable in raw
                # scale. Return empty without an RPC.
                return []
            if inv != float("-inf"):
                # For euclid, the helper returns "negative L2" but Qdrant
                # wants "positive L2 upper bound". Flip sign at the
                # boundary so server-side pruning agrees with the
                # normalized [0, 1] contract.
                native_threshold = -inv if is_euclid else inv

        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=list(request.embedding),
            with_vectors=bool(request.with_vectors),
            with_payload=True,
            limit=request.top_k,
            consistency=consistency,
            search_params=search_params,
            score_threshold=native_threshold,
            query_filter=filter_obj,
        )

        hits: List[SearchHit] = []
        for p in resp.points:
            # See note above: Qdrant returns positive L2 for euclid.
            raw = float(p.score)
            if is_euclid:
                raw = -raw
            normalized = normalize_score(self._shape.canonical, raw)
            # Belt-and-braces: drop rows where inverse-threshold roundoff
            # let a hit slip through, so the [0, 1] contract holds exactly.
            if request.score_threshold is not None and normalized < float(request.score_threshold):
                continue
            hits.append(
                SearchHit(
                    id=_coerce_point_id(p.id),
                    score=normalized,
                    payload=dict(p.payload or {}),
                    vector=_extract_vector(p) if request.with_vectors else None,
                )
            )
        return hits

    # ============================================================== retrieve
    def retrieve(
        self,
        ids: Sequence[str],
        *,
        with_vectors: bool = False,
    ) -> List[VectorPoint]:
        if not ids:
            return []
        # Qdrant accepts both string UUIDs and int ids; pass through as-is.
        raw = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=with_vectors,
        )
        out = [
            VectorPoint(
                id=_coerce_point_id(p.id),
                vector=_extract_vector(p) if with_vectors else [],
                payload=dict(p.payload or {}),
            )
            for p in raw
        ]
        # task #61 P1-V4 defense-in-depth — payload-level tenant
        # filter applied with **mode-specific semantics** (per Weston
        # msg=910cad66 BLOCKER catch on initial commit: a uniform
        # "no payload key → pass through" branch leaked stray ``{}``
        # payload rows in the shared multitenant collection to every
        # tenant on ``retrieve(ids=...)``):
        #
        # * Multitenant mode (shared physical collection): payload
        #   key is the **primary** isolation layer. STRICT — a row
        #   must carry ``TENANT_PAYLOAD_KEY`` matching this tenant.
        #   No "no payload key → pass through" branch here, because
        #   in the shared collection a missing key would expose the
        #   row to every tenant, not just this one. ``upsert()``
        #   stamps the key on every point we write, so the only way
        #   a missing-key row reaches the collection is tooling
        #   drift / migration drift, exactly the case this gate is
        #   meant to catch.
        # * Legacy mode (per-tenant physical collection): the
        #   collection name is the primary isolation layer
        #   (``collection_name == tenant_id``); the payload-level
        #   filter is belt-and-braces. PERMISSIVE — a row that
        #   doesn't carry the payload key still passes through
        #   (typical legacy data shape pre-multitenant), but a stray
        #   foreign-tenant payload gets dropped (catches tooling
        #   drift / migration mistakes).
        #
        # Lesson #14 multi-iteration cleanup family — legacy mode is
        # preserved for migration rollback only; a future PR can drop
        # the mode entirely once telemetry confirms zero production
        # usage.
        if self.multitenant:
            return [p for p in out if p.payload.get(TENANT_PAYLOAD_KEY) == self._tenant.id]
        return [
            p
            for p in out
            if TENANT_PAYLOAD_KEY not in p.payload or p.payload.get(TENANT_PAYLOAD_KEY) == self._tenant.id
        ]

    # ================================================================ delete
    def delete(self, ids: Sequence[str]) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        if self.multitenant:
            # Defense-in-depth: bind the tenant_id so a rogue id list
            # cannot cross-tenant-delete. UUID collisions are already
            # astronomically unlikely, but the guard is nearly free.
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key=TENANT_PAYLOAD_KEY,
                                match=rest.MatchValue(value=self._tenant.id),
                            ),
                            rest.HasIdCondition(has_id=ids_list),
                        ]
                    )
                ),
            )
        else:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(points=ids_list),
            )

    def delete_by_filter(self, flt: VectorFilter) -> None:
        """Delete every point matching the filter, tenant-guarded."""
        qfilter = _translate_filter(flt)
        if qfilter is None:
            # Safety: an empty / unrecognized filter must never translate
            # into "delete every point" for the current tenant. We check
            # **before** the tenant guard merge, otherwise ``None`` would
            # silently turn into "drop_tenant semantics" — same outcome,
            # but without the caller asking for it.
            raise ValueError("delete_by_filter requires a non-empty filter")
        if self.multitenant:
            qfilter = _merge_tenant_filter(qfilter, self._tenant.id)
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.FilterSelector(filter=qfilter),
            )
        except UnexpectedResponse as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                return
            raise

    # ============================================================ drop_tenant
    def drop_tenant(self, *, purge_all_shards: bool = False) -> None:
        """Remove this tenant's vectors.

        See ``VectorStoreConnector.drop_tenant`` for the contract.
        """
        if self.multitenant:
            if purge_all_shards:
                self._purge_tenant_from_all_global_collections()
                return
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest.FilterSelector(
                        filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key=TENANT_PAYLOAD_KEY,
                                    match=rest.MatchValue(value=self._tenant.id),
                                )
                            ]
                        )
                    ),
                )
            except UnexpectedResponse as e:
                msg = str(e).lower()
                if "not found" in msg or "doesn't exist" in msg:
                    return
                raise
            return

        # Legacy layout: drop the whole physical collection.
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except UnexpectedResponse as e:
            msg = str(e).lower()
            if "not found" in msg or "doesn't exist" in msg:
                return
            raise

    def _purge_tenant_from_all_global_collections(self) -> None:
        """Best-effort purge of this tenant's points across every global shard.

        Called from the drop path when ``vector_size`` cannot be resolved,
        so we cannot route to a single ``aperag_vectors_{size}_{distance}``
        collection. We iterate all collections whose name starts with the
        multi-tenant naming prefix and issue a filtered delete on each.

        This is explicitly best-effort:
        * failures on individual collections are logged, not re-raised —
          we'd rather succeed on 7 of 8 shards than zero;
        * we never touch non-``aperag_vectors_*`` collections (including
          legacy per-tenant ``col<hex>`` names), so this is safe to run
          even in mixed deployments during the migration window.
        """
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception:
            logger.exception("qdrant: could not list collections for orphan purge")
            return
        for name in existing:
            if not name.startswith(_MULTITENANT_PREFIX):
                continue
            try:
                self.client.delete(
                    collection_name=name,
                    points_selector=rest.FilterSelector(
                        filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key=TENANT_PAYLOAD_KEY,
                                    match=rest.MatchValue(value=self._tenant.id),
                                )
                            ]
                        )
                    ),
                )
                logger.info("qdrant: purged tenant %s from %s", self._tenant.id, name)
            except UnexpectedResponse as e:
                if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                    continue
                logger.warning("qdrant: failed to purge %s from %s: %s", self._tenant.id, name, e)
            except Exception:
                logger.exception("qdrant: unexpected error purging %s from %s", self._tenant.id, name)


# Re-export for callers that imported these from this module historically.
__all__ = [
    "QdrantVectorStoreConnector",
    "TENANT_PAYLOAD_KEY",
    "global_collection_name",
    "_coerce_distance",
    "_hnsw_config",
    "_optimizers_config",
    "_quantization_config",
    "_ensure_tenant_payload_index",
    "_merge_tenant_filter",
    "_translate_filter",
    "_normalize_filter_input",
    "_get_or_create_client",
    "_reset_client_cache",
    "_ENSURED_COLLECTIONS",
]

# Preserve `json` import for backward compatibility with any code that does
# `from aperag.vectorstore.qdrant_connector import json` (grepped; none
# found, but the cost of keeping the name is zero).
_ = json
