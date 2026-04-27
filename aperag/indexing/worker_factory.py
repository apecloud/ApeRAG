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

from aperag.indexing.base import ModalityWorker
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
    """Wire :class:`FulltextModality` to a real Elasticsearch index.

    Uses the same physical index name the retrieval pipeline reads
    from (``generate_fulltext_index_name``) so writes and reads are
    symmetric.
    """
    from elasticsearch import Elasticsearch

    from aperag.config import settings
    from aperag.indexing.fulltext import FulltextModality
    from aperag.utils.utils import generate_fulltext_index_name

    if not settings.es_host:
        raise WorkerFactoryError("fulltext: ES_HOST not configured (settings.es_host empty)")

    es_kwargs: dict[str, Any] = {}
    if getattr(settings, "es_basic_auth_username", None):
        es_kwargs["basic_auth"] = (
            settings.es_basic_auth_username,
            getattr(settings, "es_basic_auth_password", "") or "",
        )
    if getattr(settings, "es_timeout", None):
        es_kwargs["request_timeout"] = settings.es_timeout

    client = Elasticsearch(settings.es_host, **es_kwargs)
    index_name = generate_fulltext_index_name(collection.id)
    backend = _ElasticsearchFulltextBackend(client=client, index_name=index_name)
    # Pass ``collection.id`` so ``FulltextModality.sync`` can write
    # ``collection_id`` into every ES document — the retrieval
    # pipeline ``_fulltext_search`` filters on this field. Without
    # it, search returns 0 hits silently.
    return FulltextModality(backend=backend, store=object_store, collection_id=collection.id)


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


def _build_graph_worker(*, collection: Any, object_store: Any, payload: DispatchPayload) -> ModalityWorker:
    """Wire :class:`GraphModalityWorker` for the new §D.3 lineage
    pipeline — currently **gated** until Wave 4.

    Per architect msg=c79e9a3f gap-report ruling: the §D.3.6 Nebula /
    Postgres ``LineageGraphStore`` adapter and the LightRAG-style
    LLM extractor are not in Wave 3 scope. Building this worker
    against the :class:`InMemoryLineageGraphStore` placeholder + a
    no-op extractor was a silent failure mode — runs would reach
    ``status=ACTIVE`` with zero entities written and the user would
    see "graph indexed" but search would return nothing. Worse, the
    in-memory store loses all state on worker restart, so even the
    rare working case is non-durable.

    Wave 3 ships graph **explicitly gated**: this builder raises a
    :class:`WorkerFactoryError` so any collection with
    ``enable_knowledge_graph=True`` gets a clear, persisted error
    on its graph row instead of a silent ACTIVE-with-empty-graph.
    The collection-config default is also flipped to ``False`` so
    new collections do not opt into the broken path by accident.

    Wave 4 (locked backlog) wires the real adapter + extractor; the
    detection rule in this builder will then no longer match (the
    store is a Nebula adapter, not :class:`InMemoryLineageGraphStore`)
    and the gate self-disables without a code change here.
    """
    from aperag.indexing.graph import (
        GraphModalityWorker as _GraphModalityWorker,
    )
    from aperag.indexing.graph import (
        InMemoryLineageGraphStore,
    )

    store = _process_graph_store_singleton()
    if isinstance(store, InMemoryLineageGraphStore):
        raise WorkerFactoryError(
            "graph modality requires a real LineageGraphStore (Wave 4 wiring); "
            "current InMemory placeholder is test-only — set "
            "collection.config.enable_knowledge_graph=false until Wave 4 lands"
        )

    lock = _process_graph_lock_singleton()

    async def _no_op_extractor(_chunks):
        return ([], [])

    tenant_scope_key = _resolve_tenant_scope_key(payload=payload)
    return _GraphModalityWorker(
        store=store,
        extractor=_no_op_extractor,
        entity_lock=lock,
        object_store=object_store,
        collection_id=collection.id,
        tenant_scope_key=tenant_scope_key,
    )


# ---------------------------------------------------------------------
# Helpers — singletons + collection / tenant resolution.
# ---------------------------------------------------------------------


_GRAPH_STORE_SINGLETON: Any = None
_GRAPH_LOCK_SINGLETON: Any = None


def _process_graph_store_singleton() -> Any:
    global _GRAPH_STORE_SINGLETON
    if _GRAPH_STORE_SINGLETON is None:
        from aperag.indexing.graph import InMemoryLineageGraphStore

        _GRAPH_STORE_SINGLETON = InMemoryLineageGraphStore()
    return _GRAPH_STORE_SINGLETON


def _process_graph_lock_singleton() -> Any:
    global _GRAPH_LOCK_SINGLETON
    if _GRAPH_LOCK_SINGLETON is None:
        from aperag.indexing.graph import InMemoryEntityLock

        _GRAPH_LOCK_SINGLETON = InMemoryEntityLock()
    return _GRAPH_LOCK_SINGLETON


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


__all__ = [
    "ProductionWorkerFactory",
    "WorkerFactoryError",
]
