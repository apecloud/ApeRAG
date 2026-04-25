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

"""Wiring layer: turn an ApeRAG ``CollectionRow`` into a ready-to-use
``GraphIndexService``.

This is the **only** file inside ``aperag/graphindex/`` that imports
from outside the module. Pure module code (``service.py``, ``dto.py``,
``engine``, ``storage``, ``config``) does not know what an ApeRAG
``CollectionRow`` is, what a ``CompletionService`` is, or which Postgres
URL to use. Keeping that knowledge in one file means the rest of the
module is reusable in tests, scripts, and (eventually) a separate
service deployment without dragging the whole ApeRAG stack along.

Two factory functions:

* ``make_service_for_collection(collection)`` — async, used by the
  FastAPI request path.
* ``run_index_document_sync(collection, doc_id, content, file_path)``
  / ``run_delete_document_sync(...)`` — sync wrappers for Celery tasks
  that don't have an event loop on their thread.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Optional

from aperag.config import async_engine, build_graph_db_context, settings
from aperag.db.ops import db_ops
from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import (
    DeleteDocumentResult,
    IndexDocumentResult,
)
from aperag.domains.knowledge_graph.graphindex.service import GraphIndexService
from aperag.domains.knowledge_graph.graphindex.storage.connector import GraphStoreAdaptor
from aperag.domains.knowledge_graph.ports import CollectionRow
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-level singletons
# ---------------------------------------------------------------------------
#
# The graph store and the GraphIndexConfig are stable for the life of
# a worker process — there's no per-collection variation on either.
# Building one of each at module import time and reusing them avoids the
# per-request setup tax that v1 paid via ``initialize_storages`` /
# ``finalize_storages``.

_DEFAULT_CONFIG = GraphIndexConfig(
    chunk_token_size=getattr(settings, "chunk_size", None) or 1200,
    chunk_overlap_token_size=getattr(settings, "chunk_overlap_size", None) or 100,
)


def _build_store():
    """Build the graph store according to ``settings.graph_db_type``.

    PostgreSQL reuses the application's ``async_engine``; Neo4j and
    Nebula read their connection details from env (via
    ``build_graph_db_context``). The returned object implements the
    ``GraphStore`` Protocol regardless of backend.
    """
    ctx = build_graph_db_context()
    if settings.graph_db_type == "postgresql":
        ctx["engine"] = async_engine
    return GraphStoreAdaptor(settings.graph_db_type, ctx=ctx).store


_DEFAULT_STORE = _build_store()


# ---------------------------------------------------------------------------
# Per-collection LLM wiring
# ---------------------------------------------------------------------------


def build_collection_llm_callable(collection: CollectionRow):
    """Construct the ``LLMCall`` for a specific collection.

    Reads the collection's completion config (provider, model, base_url,
    api key from the user's provider record) and returns an async
    function ``(prompt) -> str``. The function is closure-bound to the
    config so multiple concurrent calls can share it safely without
    serialising on a global LLM client.

    The ``CompletionService`` is instantiated once per callable and
    reused across every chunk extraction for the same document. Before
    this, each per-chunk call rebuilt the client (litellm import, HTTP
    session, etc.) — a meaningful overhead on high-chunk documents.
    """
    config = parseCollectionConfig(collection.config)
    if not config.completion or not config.completion.model_id:
        raise RuntimeError(f"graphindex: completion model not configured (collection {collection.id})")
    row = db_ops.query_model_runtime(config.completion.model_id, collection.user)
    if not row:
        raise RuntimeError(f"graphindex: model not found {config.completion.model_id!r} (collection {collection.id})")
    model, account = row

    # Local import: CompletionService pulls in litellm which is heavy at
    # import time and we don't want to pay it just for ``import aperag.domains.knowledge_graph.graphindex``.
    from aperag.llm.completion.completion_service import CompletionService
    from aperag.llm.runtime.resolver import resolve_model_invocation_from_records

    invocation = resolve_model_invocation_from_records(model=model, account=account)
    provider = invocation.runner_config.get("provider")
    if not provider:
        provider = "openai" if invocation.runner_type == "openai_compatible" else invocation.provider_type

    svc = CompletionService(
        provider=provider,
        model=invocation.provider_model_id,
        base_url=invocation.base_url,
        api_key=invocation.api_key,
        temperature=0.0,  # deterministic output for extraction
        max_tokens=None,
        caching=False,
    )

    async def _llm(prompt: str) -> str:
        # No history, no images, no memory. Single-turn JSON request.
        return await svc.agenerate(history=[], prompt=prompt, images=[], memory=False)

    return _llm


def build_collection_embed_callables(collection: CollectionRow):
    """Build the ``embed_query`` and ``embed_texts`` callables for
    vector-based entity/relation recall.

    Returns ``(embed_query, embed_texts)`` or ``(None, None)`` if the
    collection's embedding model cannot be resolved (e.g. provider not
    configured).
    """
    try:
        from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync

        embedding_model, _vector_size = get_collection_embedding_service_sync(collection)
    except Exception:
        logger.warning("graphindex: could not build embedding for collection %s; vector recall disabled", collection.id)
        return None, None

    async def _embed_query(text: str) -> list[float]:
        import asyncio

        return await asyncio.to_thread(embedding_model.embed_query, text)

    async def _embed_texts(texts: list[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(embedding_model.embed_documents, texts)

    return _embed_query, _embed_texts


def build_collection_vector_connector_or_none(collection: CollectionRow):
    """Return a ``VectorStoreConnector`` for writing/reading entity and
    relation embeddings, or ``None`` if configuration is incomplete.

    Reuses the same vectorstore backend and collection that chunk
    embeddings use — entity/relation vectors are distinguished by the
    ``index_type`` payload field, not by a separate physical store.
    """
    try:
        from aperag.config import get_vector_db_connector
        from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
        from aperag.utils.utils import generate_vector_db_collection_name

        _embedding_model, vector_size = get_collection_embedding_service_sync(collection)
        collection_name = generate_vector_db_collection_name(collection.id)
        adaptor = get_vector_db_connector(collection_name, vector_size=vector_size)
        return adaptor.connector
    except Exception:
        logger.warning("graphindex: could not build vector connector for collection %s", collection.id)
        return None


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def make_service_for_collection(
    collection: CollectionRow,
    *,
    config: Optional[GraphIndexConfig] = None,
) -> GraphIndexService:
    """Build a ``GraphIndexService`` ready to operate on this collection.

    Cheap to call: the heavy bits (engine, store) are process singletons.
    Only the LLM closure is built fresh per call, and it doesn't open
    any connections until the first ``llm(prompt)`` call.
    """
    embed_query, embed_texts = build_collection_embed_callables(collection)
    vector_connector = build_collection_vector_connector_or_none(collection)
    return GraphIndexService(
        store=_DEFAULT_STORE,
        llm=build_collection_llm_callable(collection),
        embed_query=embed_query,
        embed_texts=embed_texts,
        vector_connector=vector_connector,
        config=config or _DEFAULT_CONFIG,
    )


# ---------------------------------------------------------------------------
# Sync wrappers for Celery
# ---------------------------------------------------------------------------


def run_index_document_sync(
    collection: CollectionRow,
    doc_id: str,
    content: str,
    file_path: str,
) -> IndexDocumentResult:
    """Sync entry point for Celery tasks.

    Builds a fresh event loop because Celery workers don't run one by
    default. Mirrors what v1 ``process_document_for_celery`` did, but
    against the new service.
    """

    async def _run() -> IndexDocumentResult:
        svc = make_service_for_collection(collection)
        return await svc.index_document(
            collection_id=str(collection.id),
            doc_id=doc_id,
            content=content,
            file_path=file_path,
        )

    return _run_in_new_loop(_run())


def run_delete_document_sync(collection: CollectionRow, doc_id: str) -> DeleteDocumentResult:
    """Sync delete entry point for Celery tasks."""

    async def _run() -> DeleteDocumentResult:
        svc = make_service_for_collection(collection)
        return await svc.delete_document(collection_id=str(collection.id), doc_id=doc_id)

    return _run_in_new_loop(_run())


def run_drop_collection_sync(collection_id: str) -> None:
    """Sync collection-wipe entry point for the collection-delete task.

    Doesn't need a CollectionRow object — operates by id only, since by
    delete time the collection row may already be soft-deleted.
    """

    async def _run() -> None:
        svc = GraphIndexService(
            store=_DEFAULT_STORE,
            # No LLM needed; this is a pure data-wipe.
            llm=_no_op_llm,
            config=_DEFAULT_CONFIG,
        )
        await svc.drop_collection(collection_id=collection_id)

    return _run_in_new_loop(_run())


async def _no_op_llm(_prompt: str) -> str:
    raise RuntimeError("graphindex: LLM was called from a no-LLM service; this is a wiring bug")


def _run_in_new_loop(coro: Awaitable[Any]) -> Any:
    """Run ``coro`` in a new event loop, with task cancellation cleanup
    on exit. Replicates the v1 ``_run_in_new_loop`` semantics so the
    Celery contract is unchanged."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=1.0))
        except Exception:
            pass
        finally:
            loop.close()
            asyncio.set_event_loop(None)


__all__ = [
    "build_collection_llm_callable",
    "build_collection_embed_callables",
    "build_collection_vector_connector_or_none",
    "make_service_for_collection",
    "run_index_document_sync",
    "run_delete_document_sync",
    "run_drop_collection_sync",
]
