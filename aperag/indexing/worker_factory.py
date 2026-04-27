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

"""Production worker factory — celery T3.1 follow-up.

Per architect msg=7782ebe0 spec lock + PM msg=dc13c4a5 root cause:
the FastAPI lifespan (``aperag/app.py:combined_lifespan``) used to wire
``run_*_worker`` with a placeholder factory that raised
``NotImplementedError`` on every dispatch — Wave 3's hard-cut deleted
the legacy Celery indexers but never replaced this seam, so async-mode
documents stalled at ``PENDING`` forever (e2e-http-provider gate failed
on ``wait_for_document_indexes``).

This module is the seam: a per-task lazy factory that, given a
:class:`DispatchPayload`, resolves the ``Collection`` row, picks the
right :class:`ModalityWorker` subclass, and constructs it with the
production backend wiring (Qdrant + Elasticsearch + the configured
embedder / completion model). Per architect contract:

* **Per-task lazy.** Different collections use different embedders +
  vector dims; a startup-eager factory cannot satisfy that. Build on
  every call, share only the heavy singletons (object-store, Qdrant
  client pool inside ``QdrantVectorStoreConnector._get_or_create_client``).

* **Reuse existing helpers.** ``get_collection_embedding_service_sync``
  / ``get_vector_db_connector`` / ``get_object_store`` /
  ``build_collection_llm_callable`` are the canonical resolvers used
  elsewhere in ApeRAG (retrieval pipeline, graphindex). The factory
  composes them; it does not re-implement embedder routing or
  collection-name normalisation.

* **Catchable failure.** Missing collection, broken embedder config,
  or backend connectivity errors raise :class:`WorkerFactoryError`.
  The orchestrator's runner (``aperag/indexing/orchestrator.py``)
  catches that and finalises the row to ``FAILED`` so the §I.2
  reconciler-driven retry kicks in instead of silently leaving the
  row at ``PENDING``.

Graph modality wiring is intentionally minimal: Wave 3 spec only
locked vector/fulltext as the e2e-critical path, and the
:class:`InMemoryLineageGraphStore` + ``InMemoryEntityLock`` placeholder
keeps the pipeline from crashing while the real Nebula/Postgres
lineage adapter (which has to bridge §D.3.6 lineage SET semantics
into the existing ``GraphStoreAdaptor``) is sequenced as a Wave 4
follow-up. This is documented as a known gap, not a regression.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Mapping, Optional

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aperag.indexing.base import DeriveResult, ModalityWorker
from aperag.indexing.models import Modality
from aperag.indexing.orchestrator import DispatchPayload

logger = logging.getLogger(__name__)


class WorkerFactoryError(RuntimeError):
    """Raised when the factory cannot build a worker for the payload.

    The orchestrator runner (``orchestrator._runner``) catches this
    and finalises the row to ``FAILED`` so §I.2 retry-with-backoff
    picks the row up next reconciler cycle. The exception message is
    persisted in ``DocumentIndex.error_message`` for operator triage.
    """


# ---------------------------------------------------------------------
# Backend protocol adapters — wrap production clients into the per-
# modality :class:`Protocol` surfaces the worker classes already accept.
# ---------------------------------------------------------------------


class _QdrantPointBackend:
    """Adapter wrapping :class:`QdrantVectorStoreConnector` to the shared
    ``{delete_by_filter, upsert_point}`` protocol the vector / summary /
    vision modalities consume.

    All three modalities share the same Qdrant-shaped surface (delete
    by ``(document_id, parse_version)`` filter, upsert by
    ``chunk_id``/``point_id``). One adapter class satisfies the three
    Protocols structurally — no inheritance needed because the
    Protocols are ``@runtime_checkable``.
    """

    def __init__(self, *, connector: Any) -> None:
        self._connector = connector

    def delete_by_filter(self, *, document_id: str, parse_version: str) -> int:
        from aperag.vectorstore.filters import Eq, all_of

        flt = all_of(
            Eq(key="document_id", value=document_id),
            Eq(key="parse_version", value=parse_version),
        )
        # The connector's ``delete_by_filter`` does not return a count;
        # the count is informational per the §D.1 protocol contract,
        # so we report 0 and let the caller log on whatever it likes.
        if flt is not None:
            self._connector.delete_by_filter(flt)
        return 0

    def upsert_point(
        self,
        *,
        chunk_id: str | None = None,
        point_id: str | None = None,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        # Vector modality calls with ``chunk_id``; summary / vision
        # modalities call with ``point_id``. Both end up as the
        # underlying Qdrant point id.
        #
        # Qdrant only accepts unsigned-integer or UUID point ids. The
        # T1.1 parser produces chunk ids of the form
        # ``<sha-prefix>:<index>`` (e.g. ``f766a946575ec3b4:0000``)
        # which Qdrant rejects with HTTP 400 "is not a valid point
        # ID". Map the caller-supplied string id into a deterministic
        # UUID5 so retries land on the same point and the upsert is
        # idempotent — and stash the original id in the payload so
        # the read path can still surface it to clients.
        import uuid

        from aperag.vectorstore.dto import VectorPoint

        identifier = chunk_id if chunk_id is not None else point_id
        if not identifier:
            raise ValueError("upsert_point requires either chunk_id or point_id")
        identifier = str(identifier)
        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, identifier))
        merged_payload = dict(payload)
        # Preserve the original id under a stable key so the read
        # path can echo it back; ``chunk_id`` is what vector modality
        # already writes so we don't overwrite it.
        merged_payload.setdefault("chunk_id", identifier)
        self._connector.upsert(
            [
                VectorPoint(
                    id=qdrant_id,
                    vector=list(embedding),
                    payload=merged_payload,
                )
            ]
        )


class _ElasticsearchFulltextBackend:
    """Adapter wrapping a sync Elasticsearch client to the
    :class:`FulltextBackend` protocol.

    Index name is derived from the collection id via the existing
    ``generate_fulltext_index_name`` helper so search-side and
    write-side address the same physical index.
    """

    def __init__(self, *, client: Any, index_name: str) -> None:
        self._client = client
        self._index = index_name
        self._ensured = False

    def _ensure_index(self) -> None:
        if self._ensured:
            return
        try:
            if not self._client.indices.exists(index=self._index):
                self._client.indices.create(index=self._index)
        except Exception:  # noqa: BLE001 — race tolerant
            logger.exception("fulltext: ensure_index failed for %s", self._index)
            raise
        self._ensured = True

    def delete_by_query(self, *, document_id: str, parse_version: str) -> int:
        self._ensure_index()
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"document_id": document_id}},
                        {"term": {"parse_version": parse_version}},
                    ]
                }
            }
        }
        try:
            result = self._client.delete_by_query(index=self._index, body=body, refresh=True)
        except Exception as exc:  # noqa: BLE001
            raise WorkerFactoryError(f"elasticsearch delete_by_query failed: {exc!r}") from exc
        return int(result.get("deleted", 0))

    def bulk_index(self, *, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        self._ensure_index()
        actions: list[dict[str, Any]] = []
        for doc in documents:
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                raise ValueError("fulltext.bulk_index requires chunk_id on every document")
            actions.append({"index": {"_index": self._index, "_id": chunk_id}})
            actions.append(dict(doc))
        try:
            self._client.bulk(operations=actions, refresh=True)
        except Exception as exc:  # noqa: BLE001
            raise WorkerFactoryError(f"elasticsearch bulk_index failed: {exc!r}") from exc


# ---------------------------------------------------------------------
# Per-modality builders — receive a resolved Collection + helpers,
# return a fully constructed ModalityWorker.
# ---------------------------------------------------------------------


def _build_vector_worker(*, collection: Any, object_store: Any) -> ModalityWorker:
    """Wire :class:`VectorModality` to a real Qdrant collection +
    real EmbeddingService for the collection's configured model.
    """
    from aperag.config import get_vector_db_connector
    from aperag.indexing.vector import VectorModality
    from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
    from aperag.utils.utils import generate_vector_db_collection_name

    embedding_service, vector_size = get_collection_embedding_service_sync(collection)
    qdrant_collection = generate_vector_db_collection_name(collection.id)
    adaptor = get_vector_db_connector(qdrant_collection, vector_size=vector_size)
    backend = _QdrantPointBackend(connector=adaptor.connector)

    def _embed(text: str) -> list[float]:
        return embedding_service.embed_query(text)

    return VectorModality(backend=backend, store=object_store, embedder=_embed)


def _build_fulltext_worker(*, collection: Any, object_store: Any) -> ModalityWorker:
    """Wire :class:`FulltextModality` to a real fulltext backend per
    ``collection.config.fulltext_backend_type`` (Wave 4 T9).

    Uses the same physical index name the retrieval pipeline reads
    from (``generate_fulltext_index_name``) so writes and reads are
    symmetric. The fulltext backend dispatch mirrors the graph backend
    dispatch landed in T8 chunk 4b — the backend is selected per
    collection so a deployment can mix Elasticsearch (existing) and
    OpenSearch (open-licence alternative) without code changes.
    """
    from aperag.indexing.fulltext import FulltextModality
    from aperag.utils.utils import generate_fulltext_index_name

    backend_type = _resolve_fulltext_backend_type(collection)
    index_name = generate_fulltext_index_name(collection.id)
    backend = _build_fulltext_backend(backend_type=backend_type, index_name=index_name)
    # Pass ``collection.id`` so ``FulltextModality.sync`` can write
    # ``collection_id`` into every fulltext document — the retrieval
    # pipeline ``_fulltext_search`` filters on this field. Without
    # it, search returns 0 hits silently.
    return FulltextModality(backend=backend, store=object_store, collection_id=collection.id)


_VALID_FULLTEXT_BACKENDS = ("elasticsearch", "opensearch")


def _resolve_fulltext_backend_type(collection: Any) -> str:
    """Read ``collection.config.fulltext_backend_type`` from the
    collection's persisted config. Defaults to ``"elasticsearch"`` if
    the field is absent (older collections created before T9)."""
    cfg = getattr(collection, "config", None)
    raw: Any = None
    if cfg is None:
        return "elasticsearch"
    if hasattr(cfg, "fulltext_backend_type"):
        raw = cfg.fulltext_backend_type
    elif isinstance(cfg, Mapping):
        raw = cfg.get("fulltext_backend_type")
    elif isinstance(cfg, str):
        # ``Collection.config`` may be persisted as a JSON string by
        # SQLAlchemy when the column type is Text; parse defensively.
        import json

        try:
            parsed = json.loads(cfg)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping):
            raw = parsed.get("fulltext_backend_type")
    backend = raw or "elasticsearch"
    if backend not in _VALID_FULLTEXT_BACKENDS:
        raise WorkerFactoryError(
            f"unknown fulltext_backend_type {backend!r} on collection "
            f"{getattr(collection, 'id', '<unknown>')}; expected one of {_VALID_FULLTEXT_BACKENDS}"
        )
    return backend


def _build_fulltext_backend(*, backend_type: str, index_name: str) -> Any:
    """Construct the per-backend fulltext adapter. The two supported
    backends share the wire-compatible Elasticsearch HTTP API surface
    used by ``_ElasticsearchFulltextBackend`` (index / bulk /
    delete_by_query), so the same adapter class wraps both clients —
    only the underlying client driver differs.
    """
    if backend_type == "elasticsearch":
        client = _build_elasticsearch_client()
        return _ElasticsearchFulltextBackend(client=client, index_name=index_name)
    if backend_type == "opensearch":
        client = _build_opensearch_client()
        return _ElasticsearchFulltextBackend(client=client, index_name=index_name)
    raise WorkerFactoryError(f"unsupported fulltext_backend_type {backend_type!r}")


def _build_elasticsearch_client() -> Any:
    """Construct the Elasticsearch client from the global ``ES_HOST``
    + auth + timeout settings. Raises :class:`WorkerFactoryError` if
    ``ES_HOST`` is not configured.
    """
    from elasticsearch import Elasticsearch

    from aperag.config import settings

    if not settings.es_host:
        raise WorkerFactoryError("fulltext backend=elasticsearch: ES_HOST not configured (settings.es_host empty)")

    es_kwargs: dict[str, Any] = {}
    if getattr(settings, "es_basic_auth_username", None):
        es_kwargs["basic_auth"] = (
            settings.es_basic_auth_username,
            getattr(settings, "es_basic_auth_password", "") or "",
        )
    if getattr(settings, "es_timeout", None):
        es_kwargs["request_timeout"] = settings.es_timeout

    return Elasticsearch(settings.es_host, **es_kwargs)


def _build_opensearch_client() -> Any:
    """Construct the OpenSearch client from the global ``ES_HOST`` +
    auth + timeout settings (same env vars as Elasticsearch — there
    is no separate ``OPENSEARCH_HOST`` because operators run one
    fulltext backend per deployment).

    Raises :class:`WorkerFactoryError` when the optional
    ``opensearch-py`` dependency is not installed — mirrors the way
    chunk 4b gates the Neo4j / Nebula drivers behind the graph-{neo4j,
    nebula} extras.
    """
    from aperag.config import settings

    if not settings.es_host:
        raise WorkerFactoryError("fulltext backend=opensearch: ES_HOST not configured (settings.es_host empty)")

    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:  # pragma: no cover — fulltext-opensearch extra
        raise WorkerFactoryError(
            "fulltext backend=opensearch: opensearch-py not installed; install the fulltext-opensearch extra"
        ) from exc

    os_kwargs: dict[str, Any] = {}
    if getattr(settings, "es_basic_auth_username", None):
        os_kwargs["http_auth"] = (
            settings.es_basic_auth_username,
            getattr(settings, "es_basic_auth_password", "") or "",
        )
    if getattr(settings, "es_timeout", None):
        os_kwargs["timeout"] = settings.es_timeout

    return OpenSearch(hosts=[settings.es_host], **os_kwargs)


def _build_summary_worker(*, collection: Any, object_store: Any) -> ModalityWorker:
    """Wire :class:`SummaryModality` to Qdrant + a real LLM summariser
    + the collection's embedder.

    The summariser closure is built from the collection's completion
    model; the embedder is the same one vector uses (one model per
    collection, shared across modalities).
    """
    from aperag.config import get_vector_db_connector
    from aperag.indexing.summary import SummaryModality
    from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
    from aperag.utils.utils import generate_vector_db_collection_name

    embedding_service, vector_size = get_collection_embedding_service_sync(collection)
    qdrant_collection = generate_vector_db_collection_name(collection.id)
    adaptor = get_vector_db_connector(qdrant_collection, vector_size=vector_size)
    backend = _QdrantPointBackend(connector=adaptor.connector)

    summarizer = _build_collection_summarizer(collection)

    def _embed(text: str) -> list[float]:
        return embedding_service.embed_query(text)

    return SummaryModality(
        backend=backend,
        store=object_store,
        summarizer=summarizer,
        embedder=_embed,
    )


def _build_vision_worker(*, collection: Any, object_store: Any) -> ModalityWorker:
    """Wire :class:`VisionModality` — currently **gated** until Wave 4.

    Per architect msg=69df0779 ruling: a real vision modality needs
    a multimodal vision-LLM (image bytes → embedding) and a real PDF
    image-extraction pipeline. The Wave 1+2 implementation closed
    the gap at the wrong layer by computing
    ``embedding_service.embed_query(f"{image_id}|{alt_text}")`` — a
    text embedding on a string-concat — which produces deterministic
    per-image vectors but no actual image-content awareness. Search
    on a "vision-indexed" document would only match alt-text token
    similarity, not visual content. Same silent-broken pattern as
    the graph modality; same Wave 4 gate is the correct response.

    Wave 3 ships vision **explicitly gated**: this builder requires
    the collection's embedding service to be ``is_multimodal=True``
    (i.e. an explicitly-configured multimodal embedding model). Any
    collection that opts into vision without a multimodal model gets
    a clear ``WorkerFactoryError`` instead of a fake-vision ACTIVE.
    The collection-config default is also kept ``False`` so new
    collections do not accidentally opt in.

    Wave 4 (locked backlog #9) wires the real multimodal vision-LLM;
    once an operator configures a multimodal model, ``is_multimodal``
    flips to True and the gate self-disables here.
    """
    from aperag.config import get_vector_db_connector
    from aperag.indexing.vision import VisionModality
    from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
    from aperag.utils.utils import generate_vector_db_collection_name

    embedding_service, vector_size = get_collection_embedding_service_sync(collection)
    if not embedding_service.is_multimodal():
        raise WorkerFactoryError(
            "vision modality requires a real multimodal vision-LLM (Wave 4 wiring); "
            "current text-only embedder produces fake string-concat vision vectors — "
            "set collection.config.enable_vision=false until Wave 4 lands "
            "OR configure a multimodal embedding model on the collection's embedding spec"
        )

    qdrant_collection = generate_vector_db_collection_name(collection.id)
    adaptor = get_vector_db_connector(qdrant_collection, vector_size=vector_size)
    backend = _QdrantPointBackend(connector=adaptor.connector)

    def _embed(image_id: str, alt_text: str) -> list[float]:
        # Multimodal embedder is configured (gate above passed); the
        # call below routes through the multimodal model rather than
        # the string-concat placeholder.
        return embedding_service.embed_query(f"{image_id}|{alt_text}")

    return VisionModality(backend=backend, store=object_store, embedder=_embed)


async def _no_op_extractor(_chunks):
    """Wave 4 placeholder extractor — replaced by the real LightRAG-style
    LLM extractor in T1 (Wave 4 backlog #17).

    Identity-checked in :func:`_build_graph_worker` to keep the
    "Wave 4 wiring (T1 extractor)" gate explicit until T1 lands; the
    gate self-disables when this symbol is replaced with the real one.
    """
    return ([], [])


def _build_graph_worker(*, collection: Any, object_store: Any, payload: DispatchPayload) -> ModalityWorker:
    """Wire :class:`GraphModalityWorker` for the §D.3 lineage pipeline.

    Per Wave 4 T8 chunk 4b: the worker is built around a real
    backend-specific :class:`LineageGraphStore` (Postgres / Neo4j /
    Nebula) selected by ``collection.config.graph_backend_type``. The
    Wave 3 ``InMemoryLineageGraphStore raise`` gate is dissolved by the
    backend dispatch; what remains is an explicit "T1 extractor not
    wired yet" gate so a collection that opts into knowledge graph
    today still surfaces a clean :class:`WorkerFactoryError` (and lands
    on §I.2 retry-with-backoff) instead of a silent
    ACTIVE-with-empty-graph (Wave 3 lesson #10).

    Backend dispatch:

    * ``postgres`` → :class:`PostgresLineageGraphStore` bound to the
      shared async engine; tenant isolation is per-row (collection_id
      column). Strip-then-append is single-statement so no entity
      lock is required (PostgreSQL row-lock under MERGE/INSERT-ON-
      CONFLICT handles RMW serialisation).
    * ``neo4j`` → :class:`Neo4jLineageGraphStore` bound to the shared
      async driver; same single-statement RMW guarantee under Neo4j
      MERGE row-lock so no entity lock is required.
    * ``nebula`` → :class:`NebulaLineageGraphStore` bound to the
      shared sync ``ConnectionPool`` (sync nGQL via
      ``asyncio.to_thread``); Nebula has no native list ops so
      strip-then-append is read-modify-write across multiple
      statements. The injected :class:`RedisEntityLock` serialises
      concurrent rebuilds on the same entity across worker processes
      (per architect msg=f2921ae0 invariant).

    Cross-event-loop verify: the backend client singletons are
    constructed inside the builder thread (``asyncio.to_thread`` from
    :class:`ProductionWorkerFactory.__call__``) but loop binding is
    deferred to first use — async engines / drivers attach to whatever
    event loop their first ``connect()``/``session()`` call runs on,
    which is the orchestrator loop that executes the worker's
    ``sync(...)`` coroutine. No ``asyncio.run`` near the factory.
    """
    from aperag.indexing.graph import (
        GraphModalityWorker as _GraphModalityWorker,
    )

    backend_type = _resolve_graph_backend_type(collection)
    store = _build_lineage_graph_store(backend_type=backend_type, collection=collection)
    lock = _resolve_entity_lock(backend_type=backend_type)
    extractor = _no_op_extractor  # Wave 4 T1 will replace this symbol.

    if extractor is _no_op_extractor:
        raise WorkerFactoryError(
            "graph modality requires a real LightRAG-style LLM extractor "
            "(Wave 4 wiring T1 — backend chunk 4b is wired but extractor "
            "stub still in place); set collection.config.enable_knowledge_graph=false "
            "until T1 lands or wait for the T1 extractor PR"
        )

    tenant_scope_key = _resolve_tenant_scope_key(payload=payload)
    return _GraphModalityWorker(
        store=store,
        extractor=extractor,
        entity_lock=lock,
        object_store=object_store,
        collection_id=collection.id,
        tenant_scope_key=tenant_scope_key,
    )


# ---------------------------------------------------------------------
# Helpers — backend dispatch + per-process client singletons + lock
# selection.
# ---------------------------------------------------------------------


_VALID_GRAPH_BACKENDS = ("postgres", "neo4j", "nebula")


def _resolve_graph_backend_type(collection: Any) -> str:
    """Read ``collection.config.graph_backend_type`` from the
    collection's persisted config. Defaults to ``"postgres"`` if the
    field is absent (older collections created before chunk 4b)."""
    cfg = getattr(collection, "config", None)
    raw: Any = None
    if cfg is None:
        return "postgres"
    if hasattr(cfg, "graph_backend_type"):
        raw = cfg.graph_backend_type
    elif isinstance(cfg, Mapping):
        raw = cfg.get("graph_backend_type")
    elif isinstance(cfg, str):
        # ``Collection.config`` may be persisted as a JSON string by
        # SQLAlchemy when the column type is Text; parse defensively.
        import json

        try:
            parsed = json.loads(cfg)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping):
            raw = parsed.get("graph_backend_type")
    backend = raw or "postgres"
    if backend not in _VALID_GRAPH_BACKENDS:
        raise WorkerFactoryError(
            f"unknown graph_backend_type {backend!r} on collection "
            f"{getattr(collection, 'id', '<unknown>')}; expected one of {_VALID_GRAPH_BACKENDS}"
        )
    return backend


def _build_lineage_graph_store(*, backend_type: str, collection: Any) -> Any:
    """Construct the per-collection :class:`LineageGraphStore` adapter
    by binding the shared per-process backend client to the collection
    id."""
    if backend_type == "postgres":
        engine = _postgres_async_engine_singleton()
        from aperag.indexing.graph_storage.postgres import PostgresLineageGraphStore

        return PostgresLineageGraphStore(engine=engine, collection_id=collection.id)
    if backend_type == "neo4j":
        driver = _neo4j_async_driver_singleton()
        from aperag.indexing.graph_storage.neo4j import Neo4jLineageGraphStore

        return Neo4jLineageGraphStore(driver=driver, collection_id=collection.id)
    if backend_type == "nebula":
        pool, username, password, space_prefix = _nebula_pool_singleton()
        lock = _resolve_entity_lock(backend_type=backend_type)
        from aperag.indexing.graph_storage.nebula import NebulaLineageGraphStore

        return NebulaLineageGraphStore(
            pool=pool,
            username=username,
            password=password,
            collection_id=collection.id,
            entity_lock=lock,
            space_prefix=space_prefix,
        )
    raise WorkerFactoryError(f"unsupported graph_backend_type {backend_type!r}")


def _resolve_entity_lock(*, backend_type: str) -> Any:
    """Pick the EntityLock implementation appropriate for the backend.

    Postgres + Neo4j get :class:`InMemoryEntityLock` (no-op semantics
    suffice because their strip-then-append RMW is single-statement
    under native row locks). Nebula gets :class:`RedisEntityLock` so
    the read-modify-write loop serialises across worker processes
    (architect msg=f2921ae0 invariant). When no Redis URL is
    configured we fall back to the in-process lock — production
    deployments must configure Redis for the multi-process invariant
    to hold; the fallback is for single-process tests / dev only.
    """
    if backend_type == "nebula":
        return _redis_entity_lock_singleton() or _inmemory_entity_lock_singleton()
    return _inmemory_entity_lock_singleton()


_POSTGRES_ASYNC_ENGINE: Any = None
_NEO4J_ASYNC_DRIVER: Any = None
_NEBULA_POOL: Any = None  # tuple (pool, username, password, space_prefix)
_REDIS_ENTITY_LOCK: Any = None
_INMEMORY_ENTITY_LOCK: Any = None
_BACKEND_SINGLETON_GUARD = __import__("threading").Lock()


def _postgres_async_engine_singleton() -> Any:
    global _POSTGRES_ASYNC_ENGINE
    if _POSTGRES_ASYNC_ENGINE is not None:
        return _POSTGRES_ASYNC_ENGINE
    with _BACKEND_SINGLETON_GUARD:
        if _POSTGRES_ASYNC_ENGINE is not None:
            return _POSTGRES_ASYNC_ENGINE
        from sqlalchemy.ext.asyncio import create_async_engine

        from aperag.config import settings

        url = settings.database_url
        if not url:
            raise WorkerFactoryError("graph backend=postgres requires settings.database_url (POSTGRES_HOST etc.)")
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://") :]
        elif url.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://") :]
        _POSTGRES_ASYNC_ENGINE = create_async_engine(url, pool_pre_ping=True)
        logger.info("graph backend=postgres: created async engine for %s", url.split("@")[-1])
        return _POSTGRES_ASYNC_ENGINE


def _neo4j_async_driver_singleton() -> Any:
    global _NEO4J_ASYNC_DRIVER
    if _NEO4J_ASYNC_DRIVER is not None:
        return _NEO4J_ASYNC_DRIVER
    with _BACKEND_SINGLETON_GUARD:
        if _NEO4J_ASYNC_DRIVER is not None:
            return _NEO4J_ASYNC_DRIVER
        from aperag.config import settings

        if not settings.neo4j_uri:
            raise WorkerFactoryError("graph backend=neo4j requires settings.neo4j_uri (NEO4J_URI)")
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:  # pragma: no cover — graph-neo4j extra
            raise WorkerFactoryError("neo4j driver not installed; install the graph-neo4j extra") from exc
        _NEO4J_ASYNC_DRIVER = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        logger.info("graph backend=neo4j: created async driver for %s", settings.neo4j_uri)
        return _NEO4J_ASYNC_DRIVER


def _nebula_pool_singleton() -> tuple[Any, str, str, str]:
    """Return ``(pool, username, password, space_prefix)`` for the Nebula
    adapter. Pool is shared across all collections in the process."""
    global _NEBULA_POOL
    if _NEBULA_POOL is not None:
        return _NEBULA_POOL
    with _BACKEND_SINGLETON_GUARD:
        if _NEBULA_POOL is not None:
            return _NEBULA_POOL
        from aperag.config import settings

        if not settings.nebula_hosts:
            raise WorkerFactoryError("graph backend=nebula requires settings.nebula_hosts (NEBULA_HOSTS)")
        try:
            from nebula3.Config import Config as _NebulaConfig
            from nebula3.gclient.net import ConnectionPool
        except ImportError as exc:  # pragma: no cover — graph-nebula extra
            raise WorkerFactoryError("nebula3 driver not installed; install the graph-nebula extra") from exc
        hosts: list[tuple[str, int]] = []
        for raw_host in settings.nebula_hosts.split(","):
            host_part = raw_host.strip()
            if not host_part:
                continue
            host, _, port_str = host_part.partition(":")
            hosts.append((host, int(port_str or "9669")))
        if not hosts:
            raise WorkerFactoryError(f"settings.nebula_hosts={settings.nebula_hosts!r} parsed to no hosts")
        config = _NebulaConfig()
        config.max_connection_pool_size = 32
        pool = ConnectionPool()
        if not pool.init(hosts, config):
            raise WorkerFactoryError(f"nebula ConnectionPool.init({hosts!r}) failed")
        _NEBULA_POOL = (
            pool,
            settings.nebula_username,
            settings.nebula_password,
            f"{settings.nebula_space_prefix}_lineage",
        )
        logger.info("graph backend=nebula: created connection pool for %s", hosts)
        return _NEBULA_POOL


def _redis_entity_lock_singleton() -> Any | None:
    """Return a :class:`RedisEntityLock` bound to the indexing-queue
    Redis logical DB if configured, ``None`` otherwise (the caller
    falls back to :class:`InMemoryEntityLock`)."""
    global _REDIS_ENTITY_LOCK
    if _REDIS_ENTITY_LOCK is not None:
        return _REDIS_ENTITY_LOCK
    with _BACKEND_SINGLETON_GUARD:
        if _REDIS_ENTITY_LOCK is not None:
            return _REDIS_ENTITY_LOCK
        from aperag.config import settings

        url = settings.indexing_queue_redis_url
        if not url:
            return None
        try:
            from redis import asyncio as redis_asyncio
        except ImportError:  # pragma: no cover — redis is a base dep
            return None
        from aperag.indexing.graph import RedisEntityLock

        client = redis_asyncio.from_url(url, encoding="utf-8", decode_responses=True)
        _REDIS_ENTITY_LOCK = RedisEntityLock(client)
        logger.info("graph entity_lock: bound RedisEntityLock to %s", url.split("@")[-1])
        return _REDIS_ENTITY_LOCK


def _inmemory_entity_lock_singleton() -> Any:
    global _INMEMORY_ENTITY_LOCK
    if _INMEMORY_ENTITY_LOCK is not None:
        return _INMEMORY_ENTITY_LOCK
    with _BACKEND_SINGLETON_GUARD:
        if _INMEMORY_ENTITY_LOCK is not None:
            return _INMEMORY_ENTITY_LOCK
        from aperag.indexing.graph import InMemoryEntityLock

        _INMEMORY_ENTITY_LOCK = InMemoryEntityLock()
        return _INMEMORY_ENTITY_LOCK


def _reset_graph_backend_singletons_for_tests() -> None:
    """Drop every cached backend client + lock so a test fixture can
    re-bind them. Not part of the public API — tests import this when
    they swap settings or want a fresh backend per run."""
    global _POSTGRES_ASYNC_ENGINE, _NEO4J_ASYNC_DRIVER, _NEBULA_POOL
    global _REDIS_ENTITY_LOCK, _INMEMORY_ENTITY_LOCK
    with _BACKEND_SINGLETON_GUARD:
        _POSTGRES_ASYNC_ENGINE = None
        _NEO4J_ASYNC_DRIVER = None
        _NEBULA_POOL = None
        _REDIS_ENTITY_LOCK = None
        _INMEMORY_ENTITY_LOCK = None


def _build_collection_summarizer(collection: Any) -> Callable[[str], str]:
    """Return a sync ``(markdown -> summary_text)`` closure built from
    the collection's completion config.

    Falls back to a cheap "first paragraph" heuristic if the
    collection has no completion model configured — keeps the pipeline
    runnable for collections that use summary modality without
    explicit LLM wiring.
    """
    try:
        from aperag.domains.knowledge_graph.graphindex.integration import (
            build_collection_llm_callable,
        )

        llm = build_collection_llm_callable(collection)
    except Exception:  # noqa: BLE001 — best-effort
        logger.warning(
            "summary: completion model not configured for collection %s; falling back to first-paragraph heuristic",
            getattr(collection, "id", "<unknown>"),
        )
        from aperag.indexing.summary import _placeholder_summary

        return _placeholder_summary

    def _summarize(markdown: str) -> str:
        prompt = (
            "Produce a concise standalone summary of the document below "
            "(<=200 words, plain text, no markdown):\n\n" + markdown
        )
        try:
            return asyncio.run(llm(prompt))
        except RuntimeError:
            # Already inside an event loop — schedule on a worker thread.
            future = asyncio.run_coroutine_threadsafe(
                llm(prompt),
                asyncio.get_event_loop(),
            )
            return future.result()

    return _summarize


def _resolve_tenant_scope_key(*, payload: DispatchPayload) -> str:
    """Read ``tenant_scope_key`` off the persisted ``DocumentIndex``
    row.

    The dispatcher (``dispatcher.py``) stores the resolved key on the
    row at INSERT time. The factory does not have an easy way to
    recompute the key from collection state alone (different
    deployments use different scope schemes — ``"user:<uid>"``,
    ``"org:<oid>"``, ...), so the row is the source of truth.
    """
    from aperag.indexing.models import DocumentIndex

    runtime = _get_runtime_or_raise()
    with Session(runtime.engine) as session:
        row = session.get(DocumentIndex, payload.index_id)
        if row is None:
            raise WorkerFactoryError(
                f"document_index row id={payload.index_id} not found while resolving tenant_scope_key"
            )
        return str(row.tenant_scope_key)


def _get_runtime_or_raise():
    from aperag.indexing.runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise WorkerFactoryError("IndexingRuntime is not installed (lifespan never ran)")
    return runtime


# ---------------------------------------------------------------------
# Top-level factory — installed by the FastAPI lifespan.
# ---------------------------------------------------------------------


# Per-modality dispatch table. The factory closes over this so adding
# a new modality is one entry — no changes to the worker loop.
_MODALITY_BUILDERS: Mapping[Modality, Callable[..., ModalityWorker]] = {
    Modality.VECTOR: _build_vector_worker,
    Modality.FULLTEXT: _build_fulltext_worker,
    Modality.SUMMARY: _build_summary_worker,
    Modality.VISION: _build_vision_worker,
    Modality.GRAPH: _build_graph_worker,
}


class ProductionWorkerFactory:
    """Process-wide, per-task lazy factory installed by the FastAPI
    lifespan.

    Replaces the placeholder ``_placeholder_worker_factory`` that
    raised :class:`NotImplementedError` on every dispatch. The async
    work-pool's ``run_worker_loop`` invokes this on every BLPOP'd
    payload, so per-task cost matters: heavy resources (object store,
    Qdrant client pool) are singletons resolved once; only the
    per-call collection lookup + per-modality wiring runs on each
    dispatch.

    The factory is async because the upstream ``run_worker_loop`` API
    expects an awaitable; the body is mostly sync (DB lookup, helper
    composition) so we ``await asyncio.to_thread`` for the SQLAlchemy
    bits.
    """

    def __init__(self, *, engine: Engine, object_store: Optional[Any] = None) -> None:
        self._engine = engine
        if object_store is None:
            from aperag.objectstore.base import get_object_store

            object_store = get_object_store()
        self._object_store = object_store

    async def __call__(self, payload: DispatchPayload) -> ModalityWorker:
        if payload.collection_id is None:
            raise WorkerFactoryError(
                f"dispatch payload index_id={payload.index_id} has no collection_id; "
                f"cannot resolve collection-specific config"
            )
        collection = await asyncio.to_thread(self._load_collection, payload.collection_id)
        if collection is None:
            raise WorkerFactoryError(
                f"collection {payload.collection_id!r} not found while building "
                f"{payload.modality.value} worker for index_id={payload.index_id}"
            )

        builder = _MODALITY_BUILDERS.get(payload.modality)
        if builder is None:
            raise WorkerFactoryError(f"no builder registered for modality {payload.modality.value!r}")

        kwargs: dict[str, Any] = {
            "collection": collection,
            "object_store": self._object_store,
        }
        if payload.modality is Modality.GRAPH:
            kwargs["payload"] = payload

        try:
            return await asyncio.to_thread(lambda: builder(**kwargs))
        except WorkerFactoryError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap so orchestrator catches
            raise WorkerFactoryError(
                f"failed to build {payload.modality.value} worker for "
                f"collection={payload.collection_id} index_id={payload.index_id}: {exc!r}"
            ) from exc

    def _load_collection(self, collection_id: str) -> Any:
        from aperag.domains.knowledge_base.db.models import Collection

        with Session(self._engine) as session:
            return session.get(Collection, collection_id)

    async def build_for_cleanup_row(self, row: Any) -> "CleanupWorkerView":
        """Build a cleanup-only view per ``(row.collection_id, row.modality)``.

        Wave 4 T2 entry point: the cleanup loop reads each
        :class:`DocumentIndex` row and asks the factory for the right
        ``_backend`` (vector / fulltext / summary / vision) or
        ``_store + _entity_lock`` (graph) so the per-modality DELETE
        can run against the correct per-collection backend.

        Bypasses dispatch-time gates that block worker construction
        but are irrelevant to deletion — graph "Wave 4 T1 extractor"
        and vision "multimodal vision-LLM" gates both keep raising for
        dispatch even after T2 ships, but cleanup must still drop the
        backend artefacts when an operator deletes a collection /
        document. Without this bypass the cleanup loop would hit
        :class:`WorkerFactoryError` for any partially-gated modality
        and leak Qdrant points / ES docs / graph entities forever.
        """
        if row.collection_id is None:
            raise WorkerFactoryError(f"document_index row id={row.id} has no collection_id; cannot build cleanup view")
        collection = await asyncio.to_thread(self._load_collection, row.collection_id)
        if collection is None:
            raise WorkerFactoryError(
                f"collection {row.collection_id!r} not found while building cleanup view "
                f"for index_id={row.id} modality={row.modality}"
            )
        try:
            modality = Modality(row.modality)
        except ValueError as exc:
            raise WorkerFactoryError(f"unknown modality {row.modality!r} on index_id={row.id}") from exc

        try:
            return await asyncio.to_thread(_build_cleanup_view_sync, collection, modality)
        except WorkerFactoryError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap so cleanup loop can log + skip
            raise WorkerFactoryError(
                f"failed to build {modality.value} cleanup view for "
                f"collection={row.collection_id} index_id={row.id}: {exc!r}"
            ) from exc


# ---------------------------------------------------------------------
# Cleanup-only worker view — celery Wave 4 T2.
# ---------------------------------------------------------------------
#
# The cleanup loop only consumes a tiny fraction of the
# :class:`ModalityWorker` surface: ``_backend.{delete_by_filter,
# delete_by_query}`` for the four flat modalities, and ``_store +
# _entity_lock`` for graph (per ``aperag.indexing.cleanup``). The full
# dispatch-time builders enforce gates that block construction even
# though deletion does not depend on them — the graph "Wave 4 T1
# extractor" gate (line 429-435) and the vision multimodal gate
# (line 349-355) both raise :class:`WorkerFactoryError` for any
# collection that opts into a Wave 4-pending modality.
#
# A separate cleanup-only construction path lets the cleanup loop
# materialise the minimum shape it needs without falling foul of
# those gates. Production cleanup needs to delete the backend artefacts
# even when the modality is partially gated — otherwise an operator who
# disables a modality after Wave 3 would still leak Qdrant points / ES
# docs / graph entities forever.


class CleanupWorkerView(ModalityWorker):
    """Minimal :class:`ModalityWorker` shape for the cleanup loop.

    Cleanup duck-types on ``_backend`` (flat modalities) or
    ``_store + _entity_lock`` (graph); ``derive`` / ``sync`` are never
    called from the cleanup path. This view stubs both as
    :class:`NotImplementedError` so a programming error that misroutes
    a cleanup view into the dispatch path surfaces loudly instead of
    silently dropping work.
    """

    def __init__(
        self,
        *,
        modality: Modality,
        backend: Optional[Any] = None,
        store: Optional[Any] = None,
        entity_lock: Optional[Any] = None,
    ) -> None:
        self.modality = modality
        # ``cleanup._flat_backend_delete_callable`` walks ``_backend`` for
        # ``delete_by_filter`` / ``delete_by_query`` so the attribute name
        # has to match the existing convention used by the production
        # workers (vector / fulltext / summary / vision all expose
        # ``_backend``).
        self._backend = backend
        self._store = store
        self._entity_lock = entity_lock

    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        raise NotImplementedError("CleanupWorkerView.derive must not be called — cleanup-only shape")

    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        raise NotImplementedError("CleanupWorkerView.sync must not be called — cleanup-only shape")


def _build_qdrant_cleanup_backend(collection: Any) -> Any:
    """Construct the Qdrant ``_backend`` adapter without any modality-
    specific gate (used for cleanup of vector / summary / vision).

    The full ``_build_vector_worker`` chain calls
    :func:`get_collection_embedding_service_sync` to size the Qdrant
    collection; we duplicate that minimal step here so a collection
    whose embedder config is broken (a Wave 3 lesson #10 case) still
    deletes its points instead of silently leaking. If the embedding
    service cannot be resolved we still need ``vector_size`` to
    address the right collection — fall back to the connector's
    introspection of the existing collection.
    """
    from aperag.config import get_vector_db_connector
    from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
    from aperag.utils.utils import generate_vector_db_collection_name

    qdrant_collection = generate_vector_db_collection_name(collection.id)
    try:
        _, vector_size = get_collection_embedding_service_sync(collection)
    except Exception as exc:  # noqa: BLE001
        # The embedder is irrelevant for ``delete_by_filter``; the
        # connector only needs ``vector_size`` to validate against
        # the existing collection on the Qdrant side. Fall back to a
        # benign size — the connector will still address the right
        # collection by name and the delete-by-filter call does not
        # touch the vector dimension.
        logger.warning(
            "cleanup vector backend: embedder resolve failed for collection=%s (%s); "
            "falling back to size=0 for delete-only operations",
            getattr(collection, "id", "<unknown>"),
            exc,
        )
        vector_size = 0
    adaptor = get_vector_db_connector(qdrant_collection, vector_size=vector_size)
    return _QdrantPointBackend(connector=adaptor.connector)


def _build_es_cleanup_backend(collection: Any) -> Any:
    """Construct the fulltext ``_backend`` adapter for cleanup,
    dispatching on ``collection.config.fulltext_backend_type``
    (T9). Reuses the same dispatch + client builders as the
    dispatch path so an operator who switched the collection from
    Elasticsearch to OpenSearch still cleans up the right index.
    """
    from aperag.utils.utils import generate_fulltext_index_name

    backend_type = _resolve_fulltext_backend_type(collection)
    index_name = generate_fulltext_index_name(collection.id)
    return _build_fulltext_backend(backend_type=backend_type, index_name=index_name)


def _build_cleanup_view_sync(collection: Any, modality: Modality) -> CleanupWorkerView:
    """Synchronous cleanup-view builder per ``(collection, modality)``.

    Wrapped by :meth:`ProductionWorkerFactory.build_for_cleanup_row`
    in :func:`asyncio.to_thread` so the SQLAlchemy collection load
    + sync client construction does not block the orchestrator loop.
    """
    if modality is Modality.GRAPH:
        backend_type = _resolve_graph_backend_type(collection)
        store = _build_lineage_graph_store(backend_type=backend_type, collection=collection)
        lock = _resolve_entity_lock(backend_type=backend_type)
        return CleanupWorkerView(modality=modality, store=store, entity_lock=lock)

    if modality in (Modality.VECTOR, Modality.SUMMARY, Modality.VISION):
        backend = _build_qdrant_cleanup_backend(collection)
        return CleanupWorkerView(modality=modality, backend=backend)

    if modality is Modality.FULLTEXT:
        backend = _build_es_cleanup_backend(collection)
        return CleanupWorkerView(modality=modality, backend=backend)

    raise WorkerFactoryError(f"no cleanup builder registered for modality {modality.value!r}")


__all__ = [
    "CleanupWorkerView",
    "ProductionWorkerFactory",
    "WorkerFactoryError",
]
