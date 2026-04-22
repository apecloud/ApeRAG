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

"""Qdrant vector store connector.

The connector supports two physical layouts, switchable per-context:

* **multitenant** (default, new): one Qdrant collection per ``(vector_size, distance)``
  pair, shared by all ApeRAG collections. Every point carries a ``collection_id``
  payload field used for tenant isolation. A keyword index with ``is_tenant=True``
  is registered so the Qdrant optimizer groups points by tenant on disk, which
  keeps query cost comparable to per-tenant collections.

* **legacy** (``multitenant=false``): one Qdrant collection per ApeRAG collection
  (the historical behavior). Preserved for rollback during the migration window.

Both layouts use the same connector API. Callers pass ``collection`` in the ctx
(the ApeRAG collection id); the connector internally maps it to either the
physical collection name (legacy) or the tenant filter (multitenant).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScoredPoint

from aperag.query.query import DocumentWithScore, QueryResult, QueryWithEmbedding
from aperag.vectorstore.base import VectorPoint, VectorStoreConnector
from aperag.vectorstore.filters import And, Eq, In, IsEmpty, Not, Or, VectorFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# constants & helpers
# ---------------------------------------------------------------------------

# Payload field used as the tenant discriminator. When is_tenant=True is set on
# the keyword index over this field, Qdrant physically groups points with the
# same tenant into the same segments, which restores per-tenant locality while
# keeping only one collection.
TENANT_PAYLOAD_KEY = "collection_id"

# Module-level cache of physical collections already ensured within this
# process. Avoids an RPC on every connector instantiation. Thread-safe because
# it's only appended to, never mutated concurrently in conflicting ways.
_ENSURED_COLLECTIONS: set = set()
_ENSURE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# QdrantClient process-level pool
# ---------------------------------------------------------------------------
#
# Historical behavior: every ``QdrantVectorStoreConnector(ctx)`` built a fresh
# ``QdrantClient``, which in turn opens its own HTTP/gRPC connection pool
# (and, in HTTPS mode, a fresh TLS handshake). In a read-heavy deployment
# every search request went through ``search_pipeline_service._vector_search``
# → ``ContextManager(...)`` → new connector → new client. Under load that
# adds up to a lot of avoidable setup cost and file descriptors.
#
# The cache below keys by the subset of ctx that actually identifies the
# destination endpoint. It's intentionally tiny: behavior-affecting knobs like
# ``timeout`` are intentionally NOT part of the key, because changing them
# between connectors should just update the existing client's setting, not
# spawn a new connection. If callers need truly different transport configs
# they should hit different endpoints (different ``url``).
#
# ``:memory:`` clients are deliberately NOT cached — they are used by tests
# as isolated, per-test stores. Caching them would silently share state
# across tests and is exactly the "surprise in a production config that works
# differently in test mode" footgun we're trying to avoid.
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
    """Return a process-shared QdrantClient for the given endpoint.

    Safe under threads: creation is guarded by a lock and wrapped with a
    double-check, so concurrent first callers don't stampede into building
    N clients for the same key. For ``:memory:`` URLs we bypass the cache
    entirely — each caller gets its own isolated store, which is what
    tests (and only tests) rely on.
    """
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
    """Clear the process-level client cache. Intended for tests only."""
    with _CLIENT_LOCK:
        _CLIENT_CACHE.clear()


def global_collection_name(vector_size: int, distance: str) -> str:
    """Return the physical Qdrant collection name for a given vector shape.

    We partition by ``(vector_size, distance)`` because Qdrant requires vectors
    in a single collection to share both; different embedding models produce
    different vector shapes and must therefore live in distinct collections.
    """
    return f"aperag_vectors_{int(vector_size)}_{distance.lower()}"


def _coerce_distance(distance: Any) -> rest.Distance:
    """Normalize a distance value into ``rest.Distance``.

    Qdrant's enum keys are uppercase (``COSINE``, ``EUCLID``, ``DOT``,
    ``MANHATTAN``) but their string values are capitalized ("Cosine", …).
    Callers in this repo use the capitalized string form (from VECTOR_DB_CONTEXT
    JSON), so we accept both: enum lookup by member name, case-insensitive.
    """
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
    """Build a QuantizationConfig from ctx fields (or return None).

    INT8 scalar quantization is the only mode we currently enable by default.
    ``quantile=0.99`` clips outliers so a few extreme vector values don't blow
    up the int8 range. ``always_ram=True`` keeps the quantized vectors in RAM
    while the full-precision vectors are served from mmap — this is the
    recommended high-throughput/low-RAM tradeoff.
    """
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
    defragmentation so queries touch only per-tenant blocks). Falls back to a
    plain keyword index on older servers (Qdrant 1.10) — multitenancy still
    works at the filter level, just without the storage-layout optimization.

    Idempotent: "already exists" responses are swallowed; the first successful
    shape (plain or is_tenant) wins and stays.
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
        # Older Qdrant: unknown field, unrecognized variant, or 400. Fall
        # through and try without is_tenant.
        logger.warning(
            "qdrant: is_tenant keyword index rejected on %s (%s). "
            "Falling back to plain keyword index; multitenancy filter still works "
            "but per-tenant segment defragmentation (Qdrant >= 1.11) is unavailable.",
            collection_name,
            e,
        )

    # Plain fallback — just a keyword index, no is_tenant.
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=TENANT_PAYLOAD_KEY,
            field_schema=rest.KeywordIndexParams(type=rest.KeywordIndexType.KEYWORD),
        )
    except UnexpectedResponse as e:
        msg = str(e).lower()
        if "already exists" not in msg:
            logger.warning(
                "qdrant: plain keyword index on %s/%s also failed: %s",
                collection_name,
                TENANT_PAYLOAD_KEY,
                e,
            )
            # Fall through to legacy "field_schema as string" API as a last
            # resort on very old clients / servers.
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
    already passed a Filter, we wrap it under ``must`` so both constraints are
    AND-ed. If the caller passed nothing, we just return the tenant filter.
    """
    if not tenant_id:
        return user_filter
    tenant_clause = rest.FieldCondition(key=TENANT_PAYLOAD_KEY, match=rest.MatchValue(value=tenant_id))
    if user_filter is None:
        return rest.Filter(must=[tenant_clause])
    if isinstance(user_filter, rest.Filter):
        # If the existing filter already has a must, just append; else wrap.
        must = list(user_filter.must or [])
        must.append(tenant_clause)
        return rest.Filter(
            must=must,
            should=user_filter.should,
            must_not=user_filter.must_not,
            min_should=user_filter.min_should,
        )
    # Unknown filter shape: log and pass through only the tenant guard to be
    # safe — merging arbitrary objects into a Filter.must risks pydantic
    # validation failures at the Qdrant API boundary.
    logger.warning("dropping user_filter of unsupported type %s when applying tenant guard", type(user_filter).__name__)
    return rest.Filter(must=[tenant_clause])


# ---------------------------------------------------------------------------
# VectorFilter DSL -> qdrant_client.models.Filter
# ---------------------------------------------------------------------------
#
# Translation is intentionally tree-walking and stateless. Any future node
# added to ``aperag.vectorstore.filters.VectorFilter`` must have a branch
# here or ``_translate_filter`` raises — this is the single choke point for
# the Qdrant backend and we want the error loud rather than silent.


def _translate_filter(flt: Optional[VectorFilter]) -> Optional[rest.Filter]:
    """Convert a backend-neutral DSL tree into a Qdrant ``Filter``.

    Returns ``None`` for a ``None`` input so callers don't need to check.
    Always returns a top-level ``Filter`` (never a raw ``Condition``) so
    that ``_merge_tenant_filter`` has a consistent shape to work with.
    """
    if flt is None:
        return None
    # Leaf nodes: wrap in Filter(must=[cond]) so the caller always gets a
    # Filter regardless of whether the leaf is top-level or nested.
    if isinstance(flt, Eq):
        return rest.Filter(must=[rest.FieldCondition(key=flt.key, match=rest.MatchValue(value=flt.value))])
    if isinstance(flt, In):
        # Empty In is a logic bug: matches nothing, which is almost never
        # what the caller intended. Surface it loudly.
        if not flt.values:
            raise ValueError(f"In filter on key {flt.key!r} has empty values list")
        return rest.Filter(must=[rest.FieldCondition(key=flt.key, match=rest.MatchAny(any=list(flt.values)))])
    if isinstance(flt, IsEmpty):
        return rest.Filter(must=[rest.IsEmptyCondition(is_empty=rest.PayloadField(key=flt.key))])

    # Boolean combinators: translate children, then attach under the right
    # slot. Children are already Filter objects (never Conditions), so
    # nesting is valid Qdrant syntax: Filter(must=[Filter(...), Filter(...)])
    # is equivalent to AND-ing the two inner Filters.
    if isinstance(flt, And):
        subs = [_translate_filter(p) for p in flt.parts]
        subs = [s for s in subs if s is not None]
        return rest.Filter(must=subs)
    if isinstance(flt, Or):
        subs = [_translate_filter(p) for p in flt.parts]
        subs = [s for s in subs if s is not None]
        return rest.Filter(should=subs)
    if isinstance(flt, Not):
        sub = _translate_filter(flt.inner)
        return rest.Filter(must_not=[sub] if sub is not None else [])

    raise TypeError(
        f"Unsupported VectorFilter node: {type(flt).__name__}. "
        "Add a branch in aperag.vectorstore.qdrant_connector._translate_filter"
    )


def _normalize_filter_input(flt: Any) -> Optional[rest.Filter]:
    """Accept either a DSL node, an already-translated Qdrant Filter, or None.

    * ``None`` -> ``None``
    * DSL node (``Eq`` / ``In`` / ...) -> translated Filter
    * ``rest.Filter`` -> passed through (used by the migration script and
      any caller that still hand-rolls a raw Qdrant Filter; once those go
      away this branch can be removed).
    * anything else -> ``None`` with a warning, same as the pre-DSL behavior
      so we don't introduce a new crash mode.
    """
    if flt is None:
        return None
    if isinstance(flt, rest.Filter):
        return flt
    if isinstance(flt, (Eq, In, IsEmpty, And, Or, Not)):
        return _translate_filter(flt)
    logger.warning(
        "ignoring filter of unsupported type %s (expected VectorFilter DSL or qdrant Filter)",
        type(flt).__name__,
    )
    return None


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


class QdrantVectorStoreConnector(VectorStoreConnector):
    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)
        self.ctx = ctx

        # Storage layout flags (defaults match the optimized / safe production layout).
        self.multitenant: bool = bool(ctx.get("multitenant", True))
        self.cfg = ctx  # retained for _quantization_config / _hnsw_config

        # Tenant = ApeRAG collection id. In multitenant mode we refuse to
        # construct the connector without one: a silent fallback to a
        # placeholder would write points under a shared pseudo-tenant and
        # cross-tenant reads would match them — i.e. a silent data leak.
        tenant_raw = ctx.get("collection")
        if self.multitenant and not tenant_raw:
            raise ValueError(
                "QdrantVectorStoreConnector(multitenant=True) requires ctx['collection'] "
                "(the ApeRAG collection id used as tenant key); got empty/missing."
            )
        self.tenant_id: str = str(tenant_raw) if tenant_raw else "collection"

        self.url = ctx.get("url", "http://localhost")
        self.port = ctx.get("port", 6333)
        self.grpc_port = ctx.get("grpc_port", 6334)
        self.prefer_grpc = ctx.get("prefer_grpc", False)
        self.https = ctx.get("https", False)
        self.timeout = ctx.get("timeout", 300)
        self.vector_size = int(ctx.get("vector_size", 1536))
        self.distance = ctx.get("distance", "Cosine")

        # Physical Qdrant collection name.
        if self.multitenant:
            self.collection_name = global_collection_name(self.vector_size, str(self.distance))
        else:
            self.collection_name = self.tenant_id

        # Client — reuse a process-level pool keyed by endpoint. Creating a
        # new QdrantClient per connector was measurable overhead under load
        # (one TCP/TLS setup per query for high-QPS workloads); the cache is
        # keyed on the subset of ctx that identifies the endpoint. Tests that
        # want isolation pass ``url=":memory:"`` which bypasses the cache.
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

        # In multitenant mode pre-create the global collection (idempotent)
        # so llama_index's auto-creation path sees "exists" and doesn't try to
        # recreate it with a stripped-down config.
        if self.multitenant:
            self._ensure_collection()

        # llama_index facade used for inserts / delete-by-id.
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=self.vector_size,
                distance=_coerce_distance(self.distance),
            ),
        )

    # ------------------------------------------------------------------ ensure
    def _ensure_collection(self) -> None:
        """Idempotently make sure the physical collection and tenant index exist.

        Cached at module level so subsequent connector instantiations in the
        same process skip the RPC.
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
                # If we cannot even check, don't cache — try again next time.
                logger.exception("qdrant: collection_exists check failed for %s", self.collection_name)
                raise

            if not exists:
                logger.info(
                    "qdrant: creating global collection %s (size=%d, distance=%s, multitenant=%s)",
                    self.collection_name,
                    self.vector_size,
                    self.distance,
                    self.multitenant,
                )
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=rest.VectorParams(
                            size=self.vector_size,
                            distance=_coerce_distance(self.distance),
                            on_disk=bool(self.cfg.get("vectors_on_disk", True)),
                        ),
                        hnsw_config=_hnsw_config(self.cfg),
                        optimizers_config=_optimizers_config(self.cfg),
                        quantization_config=_quantization_config(self.cfg),
                        on_disk_payload=bool(self.cfg.get("on_disk_payload", True)),
                    )
                except UnexpectedResponse as e:
                    # Another process raced us to create it; treat as success.
                    if "already exists" not in str(e).lower():
                        raise
                    logger.info("qdrant: collection %s already exists (race)", self.collection_name)

            # Create tenant payload index. Uses ``is_tenant=True`` when the
            # server supports it (Qdrant >= 1.11), otherwise falls back to a
            # plain keyword index. See _ensure_tenant_payload_index for why.
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

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: QueryWithEmbedding,
        *,
        filter: Optional[Any] = None,
        score_threshold: float = 0.1,
        **kwargs: Any,
    ) -> QueryResult:
        """Top-k vector search, optionally filtered.

        ``filter`` accepts either a ``VectorFilter`` DSL tree (preferred)
        or a raw ``qdrant_client.models.Filter`` (legacy; tolerated so
        the migration script / tests can still hand-roll filters). The
        DSL path is documented in ``aperag.vectorstore.filters``.
        """
        consistency = kwargs.get("consistency", "majority")
        search_params = kwargs.get("search_params")
        filter_conditions = _normalize_filter_input(filter)

        if self.multitenant:
            filter_conditions = _merge_tenant_filter(filter_conditions, self.tenant_id)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query.embedding,
            with_vectors=True,
            limit=query.top_k,
            consistency=consistency,
            search_params=search_params,
            score_threshold=score_threshold,
            query_filter=filter_conditions,
        )

        results = [self._convert_scored_point_to_document_with_score(point) for point in hits.points]
        results = [result for result in results if result is not None]

        return QueryResult(
            query=query.query,
            results=results,
        )

    def _convert_scored_point_to_document_with_score(self, scored_point: ScoredPoint) -> DocumentWithScore | None:
        try:
            payload = scored_point.payload or {}
            # Points written through llama_index carry a serialized node under
            # ``_node_content`` and usually also a top-level ``text`` field.
            # Points written directly (migration script / ad-hoc tooling /
            # tests) may carry only a top-level ``text``. Handle both without
            # raising KeyError on older or externally-written rows.
            node_content_raw = payload.get("_node_content")
            node_content = None
            if isinstance(node_content_raw, str):
                try:
                    node_content = json.loads(node_content_raw)
                except json.JSONDecodeError:
                    logger.warning("qdrant: _node_content is not valid JSON for point %s", scored_point.id)

            text = payload.get("text")
            if text is None and node_content is not None:
                text = node_content.get("text")

            metadata = payload.get("metadata")
            if metadata is None and node_content is not None:
                metadata = node_content.get("metadata")
            if metadata is None:
                # Fall back to the raw payload for externally-written points.
                # We shallow-copy so callers can mutate without corrupting
                # the connector's in-flight state.
                metadata = {k: v for k, v in payload.items() if k not in ("_node_content", "text")}

            relationships = node_content.get("relationships") if node_content is not None else None
            if relationships is not None and isinstance(metadata, dict) and metadata.get("source") is None:
                try:
                    source = relationships.get("1", {}).get("metadata", {}).get("source")
                    if source:
                        metadata["source"] = os.path.basename(source)
                except AttributeError:
                    pass

            return DocumentWithScore(
                id=scored_point.id,
                text=text,
                metadata=metadata,
                embedding=scored_point.vector,
                score=scored_point.score,
            )
        except Exception:
            logger.exception("Failed to convert scored point to document")
            return None

    # ------------------------------------------------------------------ delete
    def delete(self, **delete_kwargs: Any):
        ids = delete_kwargs.get("ids")
        if not ids:
            return

        if self.multitenant:
            # Defense-in-depth: also bind the tenant_id so a rogue id list
            # cannot cross-tenant-delete. IDs are UUIDs so collisions are
            # already astronomically unlikely, but the guard is cheap.
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(
                        must=[
                            rest.FieldCondition(key=TENANT_PAYLOAD_KEY, match=rest.MatchValue(value=self.tenant_id)),
                            rest.HasIdCondition(has_id=list(ids)),
                        ]
                    )
                ),
            )
        else:
            self.store.delete_nodes(list(ids))

    # -------------------------------------------------------- create/delete col
    def create_collection(self, **kwargs: Any):
        """Create / ensure the physical Qdrant collection.

        * In multitenant mode this is a no-op beyond re-validating the global
          collection. Each ApeRAG collection shares the global one — the
          per-tenant identity lives purely in the payload.
        * In legacy mode (``multitenant=False``) this provisions a dedicated
          collection named after ``tenant_id``, with the same optimizations
          (INT8 quantization, on-disk HNSW, smaller segments).
        """
        vector_size = int(kwargs.get("vector_size") or self.vector_size)
        self.vector_size = vector_size

        if self.multitenant:
            # Physical collection may have been sized at connector init; if the
            # caller passed a different vector_size, route to the correct
            # global collection and ensure it *before* recreating the
            # llama_index store. This ordering matters: QdrantVectorStore's
            # __init__ caches `_collection_initialized` from a
            # `collection_exists` probe, so if we build it first and the
            # collection doesn't exist yet, the first add() call will try to
            # create it again with llama_index's minimal config (no
            # quantization / HNSW on_disk / segment count), which would then
            # lose to "already exists" but generate a spurious RPC.
            self.collection_name = global_collection_name(vector_size, str(self.distance))
            self._ensure_collection()
            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=_coerce_distance(self.distance),
                ),
            )
            return

        # Legacy path: one Qdrant collection per tenant, but still with the
        # optimized defaults so new legacy collections don't regress.
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=_coerce_distance(self.distance),
                on_disk=bool(self.cfg.get("vectors_on_disk", True)),
            ),
            hnsw_config=_hnsw_config(self.cfg),
            optimizers_config=_optimizers_config(self.cfg),
            quantization_config=_quantization_config(self.cfg),
            on_disk_payload=bool(self.cfg.get("on_disk_payload", True)),
        )

    def delete_collection(self, **kwargs: Any):
        """Remove *this tenant's* data.

        * Multitenant: delete all points whose payload matches ``collection_id``,
          leaving the global collection (and other tenants) untouched.
        * Legacy: drop the whole physical collection.

        Pass ``purge_all_shards=True`` to scan every ``aperag_vectors_*``
        collection and delete all points tagged with this tenant. Useful when
        the caller cannot resolve the correct ``vector_size`` any more (e.g.
        the collection's embedding provider has been removed from config): a
        normal ``delete_collection`` would route to the connector's default
        global collection and silently leave orphans behind.
        """
        if self.multitenant:
            if kwargs.get("purge_all_shards"):
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
                                    match=rest.MatchValue(value=self.tenant_id),
                                )
                            ]
                        )
                    ),
                )
            except UnexpectedResponse as e:
                # If the global collection itself is gone (e.g. fresh cluster,
                # tenant never wrote anything) treat as already-deleted.
                if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                    return
                raise
            return

        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except UnexpectedResponse as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                return
            raise

    def _purge_tenant_from_all_global_collections(self) -> None:
        """Best-effort purge of this tenant's points across every global shard.

        Called from the delete path when ``vector_size`` cannot be resolved,
        so we cannot route to a single ``aperag_vectors_{size}_{distance}``
        collection. We iterate all collections whose name starts with the
        multi-tenant naming prefix and issue a filtered delete on each.

        This is explicitly best-effort:
        * failures on individual collections are logged, not re-raised — we'd
          rather succeed on 7 of 8 shards than zero;
        * we never touch non-``aperag_vectors_*`` collections (including
          legacy per-tenant ``col<hex>`` names), so this is safe to run even
          in mixed deployments during the migration window.
        """
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception:
            logger.exception("qdrant: could not list collections for orphan purge")
            return
        prefix = "aperag_vectors_"
        for name in existing:
            if not name.startswith(prefix):
                continue
            try:
                self.client.delete(
                    collection_name=name,
                    points_selector=rest.FilterSelector(
                        filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key=TENANT_PAYLOAD_KEY,
                                    match=rest.MatchValue(value=self.tenant_id),
                                )
                            ]
                        )
                    ),
                )
                logger.info("qdrant: purged tenant %s from %s", self.tenant_id, name)
            except UnexpectedResponse as e:
                if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                    continue
                logger.warning("qdrant: failed to purge %s from %s: %s", self.tenant_id, name, e)
            except Exception:
                logger.exception("qdrant: unexpected error purging %s from %s", self.tenant_id, name)

    # ---------------------------------------------------------------- retrieval
    def retrieve(
        self,
        ids: Sequence[str],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[VectorPoint]:
        """Retrieve points by id, enforcing the tenant guard in multitenant mode.

        Exposed because several service-layer call sites (document preview /
        chunk listing) used to call ``self.client.retrieve`` directly against a
        per-tenant collection; after multitenancy they must both target the
        global collection *and* filter by tenant.

        Returns backend-neutral ``VectorPoint`` objects so callers don't pick
        up a dependency on ``qdrant_client.http.models.Record``.
        """
        raw = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        # Normalize Qdrant Records -> VectorPoint. We normalize id to str
        # because that's VectorPoint's contract (see base.py); Pydantic
        # Chunk.id downstream is Optional[str] so round-trip is stable.
        def _vec(p: Any) -> Optional[List[float]]:
            v = getattr(p, "vector", None)
            if v is None:
                return None
            if isinstance(v, list):
                return v
            # qdrant can return dict[name, list] for multi-vector collections;
            # we only ever store single-vector, so pick the first value.
            if isinstance(v, dict) and v:
                first = next(iter(v.values()))
                return first if isinstance(first, list) else None
            return None

        out = [
            VectorPoint(
                id=str(p.id),
                payload=dict(p.payload or {}),
                vector=_vec(p),
            )
            for p in raw
        ]

        if not self.multitenant:
            return out
        # Defense-in-depth filter: drop any points that don't match the tenant.
        return [p for p in out if (p.payload or {}).get(TENANT_PAYLOAD_KEY) == self.tenant_id]
