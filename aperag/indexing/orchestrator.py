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
5. On success: §F.3 atomic cutover — three statements in a single
   worker-side transaction (``status=ACTIVE`` → demote prior
   ``is_serving`` → promote new). Per architect ruling msg=492315e8
   Ruling 1, the cutover MUST run in the worker session, never split
   across a reconciler cycle (which would create an
   ACTIVE-but-not-is_serving inconsistency window §F.4 disallows).
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

    The protocol covers two queue families:

    * **Per-modality queues** (``push`` / ``pop``) — keyed by
      :class:`Modality`. The 5 modality worker pools (vector / fulltext
      / graph / summary / vision) consume these. Backed by Redis lists
      named ``q:indexing:<modality>``.

    * **Parse queue** (``push_parse`` / ``pop_parse``) — un-keyed
      single queue feeding the parse worker pool (Wave 4 T3 chunk 2,
      design pack §E.2). Parse jobs are dispatched here by the upload
      handler so the HTTP request returns 202 immediately instead of
      blocking on a 30s+ DocParser run; the parse worker pops, parses,
      and then fans out to the per-modality queues. Backed by a Redis
      list named ``q:parse``.
    """

    async def pop(self, *, modality: Modality, timeout_seconds: float) -> dict[str, Any] | None:
        """Block up to ``timeout_seconds`` for the next dispatch payload.

        Returns the deserialized payload dict, or ``None`` on timeout.
        """

    async def push(self, *, modality: Modality, payload: Mapping[str, Any]) -> None:
        """Enqueue a payload for the given modality (reconciler dispatch path)."""

    async def push_parse(self, *, payload: Mapping[str, Any]) -> None:
        """Enqueue a parse payload onto ``q:parse`` (Wave 4 T3 chunk 2).

        Called from the upload handler (``_create_or_update_document_indexes``)
        to hand the document off to the parse worker pool without
        blocking the HTTP request on parse latency.
        """

    async def pop_parse(self, *, timeout_seconds: float) -> dict[str, Any] | None:
        """Block up to ``timeout_seconds`` for the next parse payload.

        Consumed by the parse worker run loop. Returns the deserialised
        payload dict (matching :class:`ParseDispatchPayload.to_dict`) or
        ``None`` on timeout.
        """


class InMemoryWorkQueue:
    """Process-local asyncio queue mirror of the :class:`WorkQueue` protocol.

    One ``asyncio.Queue`` per modality plus a single un-keyed parse
    queue. Suitable for unit / contract tests that want to drive the
    orchestrator without standing up Redis.
    """

    def __init__(self) -> None:
        self._queues: dict[Modality, asyncio.Queue[dict[str, Any]]] = {}
        self._parse_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

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

    async def push_parse(self, *, payload: Mapping[str, Any]) -> None:
        await self._parse_queue.put(dict(payload))

    async def pop_parse(self, *, timeout_seconds: float) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._parse_queue.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    def qsize(self, modality: Modality) -> int:
        if modality not in self._queues:
            return 0
        return self._queues[modality].qsize()

    def parse_qsize(self) -> int:
        return self._parse_queue.qsize()


class RedisWorkQueue:
    """Redis-backed :class:`WorkQueue` — celery T2.1 + Wave 4 T4.

    Production multi-process worker pool transport per design pack §E.2:
    each modality gets a Redis list keyed ``q:indexing:<modality>``;
    enqueue is ``RPUSH`` of a JSON-serialized payload, dequeue is
    ``BLPOP`` with a timeout. Multiple worker processes BLPOP'ing the
    same key are atomically demuxed by Redis — at most one worker
    receives any given payload.

    Replaces the Wave 1+2 default :class:`InMemoryWorkQueue` which is
    process-local (multi-pod / multi-worker deployments lose tasks
    that get pushed to one process and BLPOP'd by another). Wave 4
    follow-up #6 per architect msg=fab88774 (Wave 1+2 gap report).

    Lazy-connects on first ``push`` / ``pop``; the underlying
    ``redis.asyncio.Redis`` client owns its own pool. Production
    deployments share one client per process: API lifespan for enqueue,
    indexing-worker CLI for dequeue.
    """

    #: Redis list key template — keyed by modality so each modality
    #: has its own BLPOP queue.
    KEY_TEMPLATE = "q:indexing:{modality}"

    #: Redis list key for the parse worker pool (Wave 4 T3 chunk 2).
    #: Matches the design pack §E.2 ASCII diagram (``q:parse``).
    PARSE_KEY = "q:parse"

    def __init__(self, redis_url: str) -> None:
        if not redis_url:
            raise ValueError("RedisWorkQueue requires a non-empty redis_url")
        self._url = redis_url
        self._client: Any | None = None  # redis.asyncio.Redis, lazy

    async def _get_client(self) -> Any:
        if self._client is None:
            from redis import asyncio as redis_asyncio

            self._client = redis_asyncio.from_url(self._url, encoding="utf-8", decode_responses=True)
        return self._client

    @classmethod
    def _key(cls, modality: Modality) -> str:
        return cls.KEY_TEMPLATE.format(modality=modality.value)

    async def push(self, *, modality: Modality, payload: Mapping[str, Any]) -> None:
        client = await self._get_client()
        await client.rpush(self._key(modality), json.dumps(dict(payload)))

    async def pop(self, *, modality: Modality, timeout_seconds: float) -> dict[str, Any] | None:
        client = await self._get_client()
        # ``BLPOP`` blocks server-side for ``timeout`` seconds (0 = forever);
        # we always pass a positive bound so the worker loop can periodically
        # check ``shutdown`` between BLPOP calls. Floor sub-second timeouts
        # to 1 — Redis BLPOP's resolution is integer seconds.
        timeout = max(1, int(timeout_seconds))
        result = await client.blpop(self._key(modality), timeout=timeout)
        if result is None:
            return None
        # ``result`` is ``(key, value)``; we only care about the value.
        _key, raw = result
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.error(
                "RedisWorkQueue.pop got non-JSON payload on key=%s: %s; dropping",
                _key,
                exc,
            )
            return None

    async def push_parse(self, *, payload: Mapping[str, Any]) -> None:
        client = await self._get_client()
        await client.rpush(self.PARSE_KEY, json.dumps(dict(payload)))

    async def pop_parse(self, *, timeout_seconds: float) -> dict[str, Any] | None:
        client = await self._get_client()
        timeout = max(1, int(timeout_seconds))
        result = await client.blpop(self.PARSE_KEY, timeout=timeout)
        if result is None:
            return None
        _key, raw = result
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.error(
                "RedisWorkQueue.pop_parse got non-JSON payload on key=%s: %s; dropping",
                _key,
                exc,
            )
            return None

    async def parse_qsize(self) -> int:
        """Inspector helper for the parse queue — current backlog length.
        Mirrors :meth:`qsize` for the per-modality queues.
        """
        client = await self._get_client()
        return int(await client.llen(self.PARSE_KEY))

    async def qsize(self, modality: Modality) -> int:
        """Inspector helper — returns the current backlog length for
        the modality. Useful for §J.1 ``queue_depth`` SLI emission and
        for tests that want to assert payloads landed.
        """
        client = await self._get_client()
        return int(await client.llen(self._key(modality)))

    async def close(self) -> None:
        """Release the underlying Redis client. Called on FastAPI
        lifespan shutdown so the connection pool drains cleanly.
        """
        if self._client is not None:
            client = self._client
            self._client = None
            try:
                await client.aclose()
            except AttributeError:  # pragma: no cover — older redis-py
                await client.close()


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


def _finalize_active_with_cutover(
    engine: Engine,
    index_id: int,
    derived_artifact_path: str,
    document_id: str,
    modality: Modality,
) -> None:
    """§F.3 atomic cutover — three statements in a single transaction.

    Per architect ruling msg=492315e8 Ruling 1, the cutover MUST run
    inside the worker's own DB session immediately after ``sync()``
    succeeds. Splitting across reconciler cycles introduces an
    ACTIVE-but-not-is_serving inconsistency window that §F.4 does not
    sanction.

    Statements (run in this exact order under one ``BEGIN ... COMMIT``):

    1. ``UPDATE document_index_v2 SET status='ACTIVE', derived_artifact_path=$path
        WHERE id=$row_id`` — marks the new sync's output as canonical.
    2. ``UPDATE document_index_v2 SET is_serving=FALSE
        WHERE document_id=$doc AND modality=$mod AND is_serving=TRUE`` —
       demotes the prior serving row (if any).
    3. ``UPDATE document_index_v2 SET is_serving=TRUE
        WHERE id=$row_id`` — promotes the new row.

    The §F.1 partial unique index ``uniq_document_index_v2_serving``
    guarantees that even under concurrent worker / reconciler
    pressure no two rows can sit at ``is_serving=TRUE`` for the same
    ``(document_id, modality)`` — the second TX would conflict and
    abort. Statement 2 (demote-FALSE) precedes statement 3
    (promote-TRUE) so the partial-unique index never sees two TRUE
    rows mid-transaction.
    """
    with Session(engine) as session, session.begin():
        # Statement 1: status=ACTIVE.
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
        # Statement 2: demote prior serving row for the same (doc, modality).
        # ``id != index_id`` so a no-op cycle (e.g. reprocessing an
        # already-promoted row) doesn't demote-then-promote itself.
        session.execute(
            update(DocumentIndex)
            .where(
                and_(
                    DocumentIndex.document_id == document_id,
                    DocumentIndex.modality == modality.value,
                    DocumentIndex.is_serving.is_(True),
                    DocumentIndex.id != index_id,
                )
            )
            .values(is_serving=False)
        )
        # Statement 3: promote this row.
        session.execute(update(DocumentIndex).where(DocumentIndex.id == index_id).values(is_serving=True))


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
            _finalize_active_with_cutover,
            engine,
            payload.index_id,
            derive_result.derived_artifact_path,
            payload.document_id,
            payload.modality,
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
            try:
                worker = await worker_factory(payload)
            except Exception as exc:  # noqa: BLE001 — surface via DB so §I.2 retry kicks in
                # Without this catch, a factory failure (e.g.
                # broken collection config, transient backend
                # connectivity error) would propagate out of the
                # asyncio.Task spawned by the run-loop and be
                # silently swallowed — the row would stay PENDING
                # forever and the reconciler would dispatch the
                # same broken payload again indefinitely. Instead,
                # claim the row and finalise it FAILED so the §I.2
                # backoff schedule can apply and the operator gets
                # a real error_message to triage.
                logger.exception(
                    "orchestrator worker_factory failed for index_id=%d modality=%s: %s",
                    payload.index_id,
                    payload.modality.value,
                    exc,
                )
                claimed = await asyncio.to_thread(_claim_row, engine, payload.index_id)
                if claimed:
                    await asyncio.to_thread(
                        _finalize_failed,
                        engine,
                        payload.index_id,
                        f"worker_factory failed: {exc!r}",
                    )
                return "factory_failed"
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
# 任务 #5: 老 GRAPH 单段 worker 拆分成事实层 + 向量层.
# 事实层 (GRAPH_FACTS) 不依赖 LLM 描述压缩, 但仍要跑 entity extraction
# (LLM 调用), 所以并发与老 GRAPH 一致 (4). 向量层 (GRAPH_VECTORS) 跑嵌入 +
# 候选合并检测, LLM 触达更轻 (没有 extractor / compactor), 但仍是 LLM-bound,
# 维持 4 的稳健默认; 后续可以根据生产观测单独调.
run_graph_facts_worker = _entrypoint(Modality.GRAPH_FACTS, concurrency=4)
run_graph_vectors_worker = _entrypoint(Modality.GRAPH_VECTORS, concurrency=4)
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
