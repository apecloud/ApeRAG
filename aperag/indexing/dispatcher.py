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

"""Indexing dispatcher — celery T3.1.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §K Wave 3
+ architect msg=268f9022 wire-in spec, the dispatcher is the
upload-side helper that bridges the FastAPI document-ingest path with
the indexing orchestrator (Wave 2 ``aperag/indexing/orchestrator.py``).

Responsibilities:

1. ``INSERT 5 rows`` into ``document_index_v2`` (status=PENDING) for
   each modality the collection enables. Rows carry the
   ``collection_id`` + ``source_path`` dispatch columns added in T2.1
   (alembic c2e8d5a1f3b9), promoted to NOT NULL in T3.1
   (d0f4c1b9a8e2).

2. Dispatch per :class:`IndexingMode`:

   * ``ASYNC`` — push a :class:`DispatchPayload` to the per-modality
     queue (Redis ``RPUSH q:<modality>`` in production); the worker
     pool's BLPOP loop picks up. Default for tier-2/3 deployments
     (per design pack §L).

   * ``INLINE`` — invoke :func:`process_one_task` synchronously per
     modality in the calling coroutine. No Redis, no worker process.
     Default for tier-1 single-machine private deployments (T3.3
     follow-up doc lane).

The dispatcher is intentionally infrastructure-light: no database
session pool of its own, no concurrency primitives. The caller (the
HTTP handler or a background task) injects the SQLAlchemy ``Engine``,
the :class:`WorkQueue`, and the per-modality worker registry.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from sqlalchemy import Engine, insert
from sqlalchemy.orm import Session

from aperag.indexing.base import ModalityWorker
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
from aperag.indexing.orchestrator import (
    DispatchPayload,
    WorkQueue,
    process_one_task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Indexing mode — async (default, queue + worker pool) vs inline
# (private deploy, synchronous per-modality call).
# ---------------------------------------------------------------------


class IndexingMode(str, enum.Enum):
    """Selects how :func:`dispatch_indexing` finalizes a request.

    Both modes INSERT the same per-modality ``document_index`` rows;
    the difference is whether work is then pushed to a queue (async)
    or driven inline by the calling coroutine (inline).
    """

    ASYNC = "async"
    INLINE = "inline"


# Default modality set the dispatcher fans out to. Callers can narrow
# this (e.g. summary-only collection) by passing an explicit
# ``modalities`` list to :func:`dispatch_indexing`.
#
# 任务 #5: 老 ``Modality.GRAPH`` 拆分为 ``GRAPH_FACTS`` (事实层) +
# ``GRAPH_VECTORS`` (向量层). 上传时 dispatcher 只插事实层一行;
# 向量层由 :func:`reconcile_graph_vectors_enqueue` 在事实层 ACTIVE 之后
# 自动 INSERT + 入队 (设计文档 §4.4 conservative serial scheduling).
DEFAULT_MODALITIES: tuple[Modality, ...] = (
    Modality.VECTOR,
    Modality.FULLTEXT,
    Modality.GRAPH_FACTS,
    Modality.SUMMARY,
    Modality.VISION,
)


# ---------------------------------------------------------------------
# Request envelope.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchRequest:
    """Per-document indexing dispatch request.

    Constructed by the upload-side handler after the source artifact
    is durable in the object store and ``parse_version`` has been
    computed (per D10.g ``compute_parse_version``).

    ``modalities`` is the subset that will be indexed for this
    document; defaults to all 5. Allowing a subset keeps the
    dispatcher useful for "vector-only" or "summary-only" collections
    without requiring a separate code path.
    """

    collection_id: str
    document_id: str
    parse_version: str
    source_path: str
    tenant_scope_key: str
    modalities: tuple[Modality, ...] = field(default=DEFAULT_MODALITIES)


# ---------------------------------------------------------------------
# Dispatch entry point.
# ---------------------------------------------------------------------


async def dispatch_indexing(
    *,
    engine: Engine,
    queue: WorkQueue | None,
    workers: Mapping[Modality, ModalityWorker] | None,
    request: DispatchRequest,
    mode: IndexingMode = IndexingMode.ASYNC,
) -> list[int]:
    """Insert per-modality ``DocumentIndex`` rows + finalize per ``mode``.

    Returns the list of newly inserted row ids in the same order as
    ``request.modalities``. Useful for callers that want to track the
    rows for status polling.

    Raises ``ValueError`` if the chosen mode's required dependency
    (queue for ASYNC, workers for INLINE) is missing — fail fast at
    the HTTP boundary rather than mid-INSERT.
    """
    if mode is IndexingMode.ASYNC and queue is None:
        raise ValueError("dispatch_indexing(mode=ASYNC) requires a non-None queue")
    if mode is IndexingMode.INLINE and not workers:
        raise ValueError("dispatch_indexing(mode=INLINE) requires a non-empty workers registry")

    row_ids = await _insert_rows(engine, request)

    if mode is IndexingMode.ASYNC:
        assert queue is not None  # narrow type for mypy
        for row_id, modality in zip(row_ids, request.modalities):
            payload = DispatchPayload(
                index_id=row_id,
                document_id=request.document_id,
                parse_version=request.parse_version,
                modality=modality,
                source_path=request.source_path,
                collection_id=request.collection_id,
            )
            await queue.push(modality=modality, payload=payload.to_dict())
        logger.info(
            "dispatch_indexing async: collection=%s document=%s parse_version=%s rows=%d",
            request.collection_id,
            request.document_id,
            request.parse_version,
            len(row_ids),
        )
    else:
        assert workers  # narrow type for mypy
        for row_id, modality in zip(row_ids, request.modalities):
            payload = DispatchPayload(
                index_id=row_id,
                document_id=request.document_id,
                parse_version=request.parse_version,
                modality=modality,
                source_path=request.source_path,
                collection_id=request.collection_id,
            )
            worker = workers.get(modality)
            if worker is None:
                logger.warning(
                    "dispatch_indexing inline: no worker registered for modality=%s row id=%d — skipping",
                    modality.value,
                    row_id,
                )
                continue
            # heartbeat_interval_seconds=0 disables the periodic
            # bump task — inline mode runs in the request-handler
            # coroutine which already owns the task lifetime.
            await process_one_task(
                engine=engine,
                payload=payload,
                worker=worker,
                heartbeat_interval_seconds=0,
            )
        logger.info(
            "dispatch_indexing inline: collection=%s document=%s parse_version=%s rows=%d",
            request.collection_id,
            request.document_id,
            request.parse_version,
            len(row_ids),
        )

    return row_ids


import asyncio  # noqa: E402 — defer to avoid circular at module-load time


async def _insert_rows(engine: Engine, request: DispatchRequest) -> list[int]:
    """Bulk INSERT one PENDING row per requested modality. Returns row ids.

    Single transaction so a partial failure (e.g. DB connection lost
    mid-INSERT) does not leave the document in a half-dispatched state.
    """
    return await asyncio.to_thread(_insert_rows_sync, engine, request)


def _insert_rows_sync(engine: Engine, request: DispatchRequest) -> list[int]:
    row_ids: list[int] = []
    with Session(engine) as session, session.begin():
        for modality in request.modalities:
            result = session.execute(
                insert(DocumentIndex)
                .values(
                    document_id=request.document_id,
                    parse_version=request.parse_version,
                    modality=modality.value,
                    status=IndexStatus.PENDING.value,
                    tenant_scope_key=request.tenant_scope_key,
                    collection_id=request.collection_id,
                    source_path=request.source_path,
                    is_serving=False,
                )
                .returning(DocumentIndex.id)
            )
            row_ids.append(int(result.scalar_one()))
    return row_ids


# ---------------------------------------------------------------------
# Subset-of-modalities convenience for the upload handler.
# ---------------------------------------------------------------------


def modalities_for_collection(
    *,
    enable_vector: bool = True,
    enable_fulltext: bool = True,
    enable_graph: bool = True,
    enable_summary: bool = True,
    enable_vision: bool = True,
) -> tuple[Modality, ...]:
    """Return the modality tuple to pass into :class:`DispatchRequest`.

    Convenience for HTTP handlers that map a Collection's per-modality
    enable flags to the dispatcher's ``modalities`` argument. Always
    yields modalities in the canonical order so dispatch row order is
    deterministic across requests (helpful for snapshot tests).
    """
    requested: list[Modality] = []
    if enable_vector:
        requested.append(Modality.VECTOR)
    if enable_fulltext:
        requested.append(Modality.FULLTEXT)
    if enable_graph:
        # 任务 #5: 上传时只入队事实层; 向量层由 reconciler 在事实层
        # ACTIVE 之后自动 enqueue (设计文档 §4.4).
        requested.append(Modality.GRAPH_FACTS)
    if enable_summary:
        requested.append(Modality.SUMMARY)
    if enable_vision:
        requested.append(Modality.VISION)
    return tuple(requested)


def all_modalities() -> tuple[Modality, ...]:
    """Helper alias — returns the 5 canonical modalities."""
    return DEFAULT_MODALITIES


# Type stub: re-export Iterable for explicit type annotations (some
# callers want to pass ``Iterable[Modality]`` without importing from
# typing themselves).
_: Iterable[Modality] = ()


__all__ = [
    "DEFAULT_MODALITIES",
    "DispatchRequest",
    "IndexingMode",
    "all_modalities",
    "dispatch_indexing",
    "modalities_for_collection",
]
