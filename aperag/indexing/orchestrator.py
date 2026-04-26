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

"""Worker pool orchestrator — celery T2.1.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §E.2 +
§I.2 + §G.3, the orchestrator is the per-modality worker process that:

1. ``BLPOP`` from ``q:<modality>`` to receive a dispatch payload
   (``{index_id, document_id, parse_version, source_path, ...}``).
2. Atomically claim the row via
   ``UPDATE document_index_v2 SET status='RUNNING', last_heartbeat=now()
    WHERE id=$id AND status IN ('PENDING','FAILED')``.
   Zero rows updated => task already claimed / cancelled => skip.
3. Spawn a heartbeat task that bumps ``last_heartbeat`` every
   :data:`HEARTBEAT_INTERVAL_SECONDS` so the reconciler does not
   reclaim the row out from under us.
4. Run ``modality.derive(...)`` then ``modality.sync(...)``.
5. On success: ``UPDATE status='ACTIVE', derived_artifact_path=...``.
   The §F.3 cutover transaction (flip ``is_serving``) is the
   reconciler's job (per-modality cutover trigger), not the orchestrator.
6. On failure: ``UPDATE status='FAILED', error_message=..., retry_count++,
   retry_after=now()+backoff(retry_count)``. The §I.2 backoff schedule
   (30s → 60s → 120s → 240s → 480s) caps at 5 retries; past that the
   row stays FAILED with ``retry_after=NULL`` and waits for operator.

The 5 per-modality worker entrypoints (``run_vector_worker`` etc.)
wire the same orchestrator core to a different modality + queue name
+ asyncio concurrency cap (per §E.2 table). Production runs each
entrypoint as a single asyncio process.

Tests inject :class:`InMemoryWorkQueue` (in place of Redis BLPOP) +
SQLAlchemy ``sqlite:///:memory:`` engine + the Wave 1 InMemory backends
to assert the full claim → derive → sync → finalize cycle without
external infrastructure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from sqlalchemy import Engine, and_, select, update
from sqlalchemy.orm import Session

from aperag.indexing.base import ModalityWorker
from aperag.indexing.models import DocumentIndex, IndexStatus, Modality

logger = logging.getLogger(__name__)


# Heartbeat / reclaim tunables — match design pack §E.4 stale window
# (60s reclaim threshold). The orchestrator bumps every
# HEARTBEAT_INTERVAL_SECONDS so a single missed bump is recoverable;
# only a real worker death (no bump for the full reclaim window)
# triggers reconciler reclaim.
HEARTBEAT_INTERVAL_SECONDS = 20

# §I.2 retry backoff schedule — exponential, capped at 5 retries.
# Sequence: 30s, 60s, 120s, 240s, 480s.
MAX_RETRY_COUNT = 5
INITIAL_RETRY_DELAY_SECONDS = 30


def _retry_delay_for(retry_count: int) -> int:
    """Return the §I.2 backoff seconds for the *next* retry attempt.

    ``retry_count`` is the count *after* the failure that just
    happened (1 for first failure, 2 for second, ...). Clamped to
    :data:`MAX_RETRY_COUNT` so the schedule never overflows.
    """
    capped = max(1, min(retry_count, MAX_RETRY_COUNT))
    return INITIAL_RETRY_DELAY_SECONDS * (2 ** (capped - 1))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------
# Queue protocol — Redis BLPOP in production, InMemoryWorkQueue in tests.
# ---------------------------------------------------------------------


@runtime_checkable
class WorkQueue(Protocol):
    """Minimal Redis-shaped queue surface the orchestrator depends on.

    Production wires this to a Redis-backed queue (``RPUSH`` enqueue +
    ``BLPOP`` dequeue keyed by ``q:<modality>``). Tests inject
    :class:`InMemoryWorkQueue` for synchronous deterministic dispatch.
    """

    async def pop(self, *, modality: Modality, timeout_seconds: float) -> dict[str, Any] | None:
        """Block up to ``timeout_seconds`` for the next dispatch payload.

        Returns the deserialized payload dict, or ``None`` on timeout.
        """

    async def push(self, *, modality: Modality, payload: Mapping[str, Any]) -> None:
        """Enqueue a payload for the given modality (reconciler dispatch path)."""


class InMemoryWorkQueue:
    """Process-local asyncio queue mirror of the :class:`WorkQueue` protocol.

    One ``asyncio.Queue`` per modality. Suitable for unit / contract
    tests that want to drive the orchestrator without standing up Redis.
    """

    def __init__(self) -> None:
        self._queues: dict[Modality, asyncio.Queue[dict[str, Any]]] = {}

    def _q(self, modality: Modality) -> asyncio.Queue[dict[str, Any]]:
        if modality not in self._queues:
            self._queues[modality] = asyncio.Queue()
        return self._queues[modality]

    async def pop(self, *, modality: Modality, timeout_seconds: float) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._q(modality).get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    async def push(self, *, modality: Modality, payload: Mapping[str, Any]) -> None:
        await self._q(modality).put(dict(payload))

    def qsize(self, modality: Modality) -> int:
        if modality not in self._queues:
            return 0
        return self._queues[modality].qsize()


# ---------------------------------------------------------------------
# Dispatch payload (round-trips through Redis as JSON).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchPayload:
    """Decoded queue payload — the unit of work the orchestrator runs."""

    index_id: int
    document_id: str
    parse_version: str
    modality: Modality
    source_path: str
    collection_id: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DispatchPayload":
        return cls(
            index_id=int(raw["index_id"]),
            document_id=str(raw["document_id"]),
            parse_version=str(raw["parse_version"]),
            modality=Modality(raw["modality"]),
            source_path=str(raw["source_path"]),
            collection_id=raw.get("collection_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_id": self.index_id,
            "document_id": self.document_id,
            "parse_version": self.parse_version,
            "modality": self.modality.value,
            "source_path": self.source_path,
            "collection_id": self.collection_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------
# Orchestrator core — single-task and run-loop entrypoints.
# ---------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Per-modality worker tuning (matches §E.2 architecture diagram).

    Production sets these from environment / config; tests construct
    directly. ``poll_timeout_seconds`` controls the BLPOP block — set
    short (~1s) so a shutdown signal is responsive without busy-looping.
    """

    modality: Modality
    concurrency: int = 4
    poll_timeout_seconds: float = 1.0
    heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS


async def _heartbeat_loop(
    engine: Engine,
    index_id: int,
    interval_seconds: int,
) -> None:
    """Bump ``last_heartbeat=now()`` every ``interval_seconds`` while RUNNING.

    Cancelled by the orchestrator when the task finishes (success or
    failure). Tolerates DB hiccups by logging + continuing — a
    single missed bump is well within the reconciler stale window.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            return
        try:
            await asyncio.to_thread(_bump_heartbeat, engine, index_id)
        except Exception as exc:  # noqa: BLE001 — heartbeat MUST NOT crash the worker
            logger.warning(
                "orchestrator heartbeat bump failed for index_id=%d: %s",
                index_id,
                exc,
            )


def _bump_heartbeat(engine: Engine, index_id: int) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            update(DocumentIndex)
            .where(
                and_(
                    DocumentIndex.id == index_id,
                    DocumentIndex.status == IndexStatus.RUNNING.value,
                )
            )
            .values(last_heartbeat=_utcnow())
        )


def _claim_row(engine: Engine, index_id: int) -> bool:
    """Atomically transition (PENDING|FAILED) → RUNNING. Returns True on win.

    A losing claim — zero rows updated — means another worker beat us
    or the row was cancelled / already advanced. The caller drops the
    payload silently in that case.
    """
    with Session(engine) as session, session.begin():
        result = session.execute(
            update(DocumentIndex)
            .where(
                and_(
                    DocumentIndex.id == index_id,
                    DocumentIndex.status.in_([IndexStatus.PENDING.value, IndexStatus.FAILED.value]),
                )
            )
            .values(
                status=IndexStatus.RUNNING.value,
                last_heartbeat=_utcnow(),
                error_message=None,
                retry_after=None,
            )
        )
    return (result.rowcount or 0) > 0


def _finalize_active(engine: Engine, index_id: int, derived_artifact_path: str) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            update(DocumentIndex)
            .where(DocumentIndex.id == index_id)
            .values(
                status=IndexStatus.ACTIVE.value,
                derived_artifact_path=derived_artifact_path,
                last_heartbeat=_utcnow(),
                error_message=None,
                retry_after=None,
            )
        )


def _finalize_failed(
    engine: Engine,
    index_id: int,
    error_message: str,
) -> None:
    """Increment ``retry_count`` + set FAILED + schedule next ``retry_after``.

    Past :data:`MAX_RETRY_COUNT` the row stays FAILED with
    ``retry_after=NULL`` so the reconciler stops re-queueing it; the
    operator must intervene (manual reset or row delete).
    """
    with Session(engine) as session, session.begin():
        row = session.execute(select(DocumentIndex.retry_count).where(DocumentIndex.id == index_id)).first()
        prior = int(row[0]) if row else 0
        new_count = prior + 1
        retry_after: datetime | None
        if new_count <= MAX_RETRY_COUNT:
            retry_after = _utcnow() + timedelta(seconds=_retry_delay_for(new_count))
        else:
            retry_after = None
        session.execute(
            update(DocumentIndex)
            .where(DocumentIndex.id == index_id)
            .values(
                status=IndexStatus.FAILED.value,
                error_message=error_message[:4096],
                retry_count=new_count,
                retry_after=retry_after,
                last_heartbeat=_utcnow(),
            )
        )


def _release_to_pending(engine: Engine, index_id: int) -> None:
    """Reset RUNNING → PENDING without incrementing retry_count.

    Used when ``derive`` returns an empty path (§C.7 "upstream not
    ready" reschedule semantic) — not a real failure, so retry_count
    must not advance.
    """
    with Session(engine) as session, session.begin():
        session.execute(
            update(DocumentIndex)
            .where(
                and_(
                    DocumentIndex.id == index_id,
                    DocumentIndex.status == IndexStatus.RUNNING.value,
                )
            )
            .values(
                status=IndexStatus.PENDING.value,
                last_heartbeat=None,
            )
        )


async def process_one_task(
    *,
    engine: Engine,
    payload: DispatchPayload,
    worker: ModalityWorker,
    heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
) -> str:
    """Run the full claim → derive → sync → finalize cycle for one payload.

    Returns one of ``"claimed"``, ``"skipped"``, ``"completed"``,
    ``"rescheduled"``, ``"failed"`` so the run-loop / tests can assert
    on the per-task outcome without scraping logs.

    The function is the seam tests use to drive the orchestrator
    directly (no queue, no heartbeat thread races) — the run-loop is
    a thin wrapper that pops from the queue and calls this.
    """
    if not _claim_row(engine, payload.index_id):
        logger.info(
            "orchestrator skip — claim lost for index_id=%d (already claimed / cancelled)",
            payload.index_id,
        )
        return "skipped"

    heartbeat_task: asyncio.Task[None] | None = None
    if heartbeat_interval_seconds > 0:
        heartbeat_task = asyncio.create_task(_heartbeat_loop(engine, payload.index_id, heartbeat_interval_seconds))

    try:
        derive_result = await worker.derive(
            document_id=payload.document_id,
            parse_version=payload.parse_version,
            source_path=payload.source_path,
        )
        if not derive_result.derived_artifact_path:
            # §C.7 reschedule semantic: derive returned empty because
            # an upstream input is not yet ready. Don't increment
            # retry_count; just put the row back on PENDING so the
            # reconciler picks it up next cycle.
            await asyncio.to_thread(_release_to_pending, engine, payload.index_id)
            logger.info(
                "orchestrator reschedule — derive returned empty for index_id=%d (upstream not ready)",
                payload.index_id,
            )
            return "rescheduled"

        await worker.sync(
            document_id=payload.document_id,
            parse_version=payload.parse_version,
            derived_artifact_path=derive_result.derived_artifact_path,
        )
        await asyncio.to_thread(
            _finalize_active,
            engine,
            payload.index_id,
            derive_result.derived_artifact_path,
        )
        return "completed"

    except Exception as exc:  # noqa: BLE001 — capture and surface via DB
        logger.exception(
            "orchestrator failure for index_id=%d modality=%s: %s",
            payload.index_id,
            payload.modality.value,
            exc,
        )
        await asyncio.to_thread(_finalize_failed, engine, payload.index_id, repr(exc))
        return "failed"

    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task


# ---------------------------------------------------------------------
# Run loop (per modality worker process).
# ---------------------------------------------------------------------


# A factory so each task gets a fresh (or pooled) ModalityWorker; the
# orchestrator does not assume a single process-wide singleton because
# graph workers are scoped to (collection_id, tenant_scope_key) at
# construction time.
ModalityWorkerFactory = Callable[[DispatchPayload], Awaitable[ModalityWorker]]


async def run_worker_loop(
    *,
    config: OrchestratorConfig,
    engine: Engine,
    queue: WorkQueue,
    worker_factory: ModalityWorkerFactory,
    shutdown: asyncio.Event,
) -> None:
    """Per-modality worker process main loop.

    Pops payloads from ``q:<modality>`` and dispatches to
    :func:`process_one_task` under an :class:`asyncio.Semaphore` whose
    permit count == ``config.concurrency``. Exits cleanly when
    ``shutdown`` is set, draining in-flight tasks first.

    Heartbeat threading is per-task (started inside
    :func:`process_one_task`) so a hung modality call doesn't block
    other in-flight tasks from heartbeating.
    """
    semaphore = asyncio.Semaphore(config.concurrency)
    in_flight: set[asyncio.Task[str]] = set()

    async def _runner(payload: DispatchPayload) -> str:
        async with semaphore:
            worker = await worker_factory(payload)
            return await process_one_task(
                engine=engine,
                payload=payload,
                worker=worker,
                heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            )

    while not shutdown.is_set():
        raw = await queue.pop(
            modality=config.modality,
            timeout_seconds=config.poll_timeout_seconds,
        )
        if raw is None:
            # Drain any completed in-flight tasks so the set doesn't
            # grow unboundedly under sustained empty-queue polling.
            in_flight = {t for t in in_flight if not t.done()}
            continue

        try:
            payload = DispatchPayload.from_dict(raw)
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "orchestrator dropping malformed payload for modality=%s: %r (%s)",
                config.modality.value,
                raw,
                exc,
            )
            continue

        task = asyncio.create_task(_runner(payload))
        in_flight.add(task)

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


# Convenience: 5 per-modality entrypoints — thin wrappers that bind
# the modality + concurrency cap from the §E.2 table. Production wires
# these as the entrypoint of each worker process.
def _entrypoint(
    modality: Modality,
    concurrency: int,
) -> Callable[..., Awaitable[None]]:
    async def _run(
        *,
        engine: Engine,
        queue: WorkQueue,
        worker_factory: ModalityWorkerFactory,
        shutdown: asyncio.Event,
    ) -> None:
        await run_worker_loop(
            config=OrchestratorConfig(modality=modality, concurrency=concurrency),
            engine=engine,
            queue=queue,
            worker_factory=worker_factory,
            shutdown=shutdown,
        )

    _run.__name__ = f"run_{modality.value}_worker"
    _run.__qualname__ = _run.__name__
    return _run


# Per-modality concurrency caps from design pack §E.2. Vector +
# fulltext are the throughput-heavy lanes; graph / summary / vision
# are LLM-bound so capped lower to keep LLM rate-limit pressure sane.
run_vector_worker = _entrypoint(Modality.VECTOR, concurrency=16)
run_fulltext_worker = _entrypoint(Modality.FULLTEXT, concurrency=32)
run_graph_worker = _entrypoint(Modality.GRAPH, concurrency=4)
run_summary_worker = _entrypoint(Modality.SUMMARY, concurrency=4)
run_vision_worker = _entrypoint(Modality.VISION, concurrency=4)


# ---------------------------------------------------------------------
# Test fixture: in-memory queue inspector helper for assertions.
# ---------------------------------------------------------------------


def drain_queue_sync(queue: InMemoryWorkQueue, modality: Modality) -> list[dict[str, Any]]:
    """Drain everything currently buffered for ``modality`` (test helper).

    Returns the payloads in queue order without blocking. Useful for
    tests that want to assert on the dispatch shape after a reconciler
    cycle without consuming via the orchestrator.
    """
    out: deque[dict[str, Any]] = deque()
    inner = queue._q(modality)  # noqa: SLF001 — test helper
    while not inner.empty():
        out.append(inner.get_nowait())
    return list(out)


__all__ = [
    "DispatchPayload",
    "InMemoryWorkQueue",
    "MAX_RETRY_COUNT",
    "INITIAL_RETRY_DELAY_SECONDS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "ModalityWorkerFactory",
    "OrchestratorConfig",
    "WorkQueue",
    "drain_queue_sync",
    "process_one_task",
    "run_fulltext_worker",
    "run_graph_worker",
    "run_summary_worker",
    "run_vector_worker",
    "run_vision_worker",
    "run_worker_loop",
]
