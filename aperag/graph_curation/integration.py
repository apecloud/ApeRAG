# Copyright 2026 ApeCloud, Inc.
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

from __future__ import annotations

import asyncio
from typing import Any, Awaitable

from aperag.db.ops import db_ops
from aperag.graph_curation.service import graph_curation_service
from aperag.indexing.llm import build_collection_llm_callable


def run_graph_curation_run_sync(run_id: str, collection_id: str) -> None:
    """Wave 7 W7-10: Celery sync entry point for the user-triggered
    curation run.

    Resolves the four Wave 7 dependencies (``LineageGraphStore``,
    ``VectorStoreConnector``, sync embedder, async LLM callable) via
    the same factories the indexer / curation merger use, so the
    user-driven sweep + the sync-driven detector + the user-driven
    merge all converge on a single set of resources.
    """
    collection = db_ops.query_collection_by_id(collection_id, ignore_deleted=False)
    if collection is None:
        raise ValueError(f"Collection {collection_id!r} not found")

    async def _run() -> None:
        # Lazy imports keep the integration module free of indexing /
        # vectorstore deps at import time (mirrors the same pattern in
        # ``aperag/domains/knowledge_graph/service.py``).
        from aperag.indexing.worker_factory import (
            _build_collection_graph_vector_writer,
            _build_lineage_graph_store_inner,
            _resolve_graph_backend_type,
        )

        backend_type = _resolve_graph_backend_type(collection)
        # Use the *inner* (non-decorated) store: the curation sweep
        # walks canonical names directly, so the alias-redirect
        # decorator (which the indexer hot path needs) would only add
        # latency without changing semantics.
        store = _build_lineage_graph_store_inner(backend_type=backend_type, collection=collection)
        vector_connector, embedder = _build_collection_graph_vector_writer(collection)
        if vector_connector is None or embedder is None:
            raise RuntimeError(
                f"graph_curation: could not resolve vector connector / embedder for collection "
                f"{getattr(collection, 'id', '<unknown>')} — curation sweep needs the same "
                f"per-collection vector + embedder bound to the indexer Phase 3 write path"
            )

        class _SyncEmbedderShim:
            """Adapt the sync ``(text -> list[float])`` callable into
            the ``embed_query`` shape the curation service expects
            (mirrors the shim in ``worker_factory`` for the merge
            candidate detector)."""

            def __init__(self, fn: Any) -> None:
                self._fn = fn

            def embed_query(self, text: str) -> list[float]:
                return self._fn(text)

        await graph_curation_service.generate_run(
            run_id=run_id,
            collection=collection,
            store=store,
            vector_connector=vector_connector,
            embedder=_SyncEmbedderShim(embedder),
            llm=build_collection_llm_callable(collection),
        )

    _run_in_new_loop(_run())


def run_expire_graph_curation_collection_sync(collection_id: str, reason: str) -> None:
    async def _run() -> None:
        await graph_curation_service.expire_pending_for_collection(collection_id, reason=reason)

    _run_in_new_loop(_run())


def run_purge_graph_curation_collection_sync(collection_id: str) -> None:
    async def _run() -> None:
        await graph_curation_service.purge_collection(collection_id)

    _run_in_new_loop(_run())


def _run_in_new_loop(coro: Awaitable[Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
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
    "run_graph_curation_run_sync",
    "run_expire_graph_curation_collection_sync",
    "run_purge_graph_curation_collection_sync",
]
