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
from typing import Any, Dict, List, Optional

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScoredPoint

from aperag.query.query import DocumentWithScore, QueryResult, QueryWithEmbedding
from aperag.vectorstore.base import VectorStoreConnector

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

        # Client
        if self.url == ":memory:":
            self.client = qdrant_client.QdrantClient(":memory:")
        else:
            self.client = qdrant_client.QdrantClient(
                url=self.url,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                https=self.https,
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
    def search(self, query: QueryWithEmbedding, **kwargs):
        consistency = kwargs.get("consistency", "majority")
        search_params = kwargs.get("search_params")
        score_threshold = kwargs.get("score_threshold", 0.1)
        filter_conditions = kwargs.get("filter")

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
        """
        if self.multitenant:
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

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, ids: List[str], with_payload: bool = True, with_vectors: bool = False):
        """Retrieve points by id, enforcing the tenant guard in multitenant mode.

        Exposed because several service-layer call sites (document preview /
        chunk listing) used to call ``self.client.retrieve`` directly against a
        per-tenant collection; after multitenancy they must both target the
        global collection *and* filter by tenant.
        """
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        if not self.multitenant:
            return points
        # Defense-in-depth filter: drop any points that don't match the tenant.
        return [p for p in points if (p.payload or {}).get(TENANT_PAYLOAD_KEY) == self.tenant_id]
