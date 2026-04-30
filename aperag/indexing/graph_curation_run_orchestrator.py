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

"""Graph curation run worker — task #31 Phase A1 (per spec
``task-31-graph-node-merge-spec-v1.md`` § 3.1.1).

This module owns the **independent worker lane** that consumes the
``q:graph_curation_run`` Redis queue and runs full-sweep graph node
merge candidate detection. Manual / cron triggers go through here:
``GraphCurationService.start_run`` (API process) just enqueues the
``run_id`` and returns; the actual merge candidate detection +
suggestion write happens in the worker process.

Why a dedicated lane (per ziang msg=92321bcc + Bryce msg=4c23f87e
BLOCKER 1):

* The existing ``WorkQueue.push/pop`` family is keyed by
  :class:`Modality` (Redis key ``q:indexing:<modality>``) and the
  ``_entrypoint(Modality, concurrency)`` machinery in
  :mod:`aperag.indexing.orchestrator` is bound to
  :class:`ModalityWorkerFactory` + :class:`DocumentIndex` payloads.
  A graph curation run is a **per-collection / per-run** job — not a
  per-document modality state. Reusing :class:`Modality` would
  pollute ``DocumentIndex`` / reconciler / index_state semantics.
* So we build a parallel queue family (``q:graph_curation_run``) and a
  parallel run loop here, on top of the same shared
  :class:`RedisWorkQueue` connection / quota / metrics infrastructure
  but with state isolation.

The ``auto_post_ingest`` strategy (existing
:meth:`MergeCandidateDetector.detect_for_sync` end-of-sync inline path)
**does not** route through this worker — see spec § 3.1.1.b. That
strategy stays as a write-only quick path inside
:class:`GraphModalityWorker.sync` and is description-free fixed by
task #77 A3.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from sqlalchemy import Engine

from aperag.indexing.orchestrator import WorkQueue

logger = logging.getLogger(__name__)


# Default worker pool sizing. Graph curation runs are heavy
# (LLM-bound entity-pair adjudication + multi-collection scans), so
# concurrency is intentionally lower than the per-modality lanes — most
# deployments will run 1-2 sweeps per process at a time. Production
# overrides via ``GraphCurationRunOrchestratorConfig`` ctor.
DEFAULT_GRAPH_CURATION_RUN_CONCURRENCY = 1
DEFAULT_POLL_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class GraphCurationRunDispatchPayload:
    """Decoded queue payload for a graph curation run dispatch.

    Carries only the minimum identifiers — the worker re-resolves the
    full ``GraphCurationRun`` row from PostgreSQL on pop. This keeps
    Redis payload small and avoids stale config drift between enqueue
    and consumption (the worker reads the latest row state, not a
    snapshot at enqueue time).
    """

    run_id: str
    collection_id: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphCurationRunDispatchPayload":
        return cls(
            run_id=str(raw["run_id"]),
            collection_id=str(raw["collection_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "collection_id": self.collection_id,
        }


@dataclass
class GraphCurationRunOrchestratorConfig:
    """Per-process tuning for the graph curation run worker.

    ``poll_timeout_seconds`` controls the BLPOP block — kept short
    (~1s) so a shutdown signal is responsive. ``concurrency`` caps
    how many runs the worker will process in parallel inside this
    process; runs that exceed the cap wait their turn at the
    semaphore.
    """

    concurrency: int = DEFAULT_GRAPH_CURATION_RUN_CONCURRENCY
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS


async def _process_one_run(
    *,
    engine: Engine,
    payload: GraphCurationRunDispatchPayload,
) -> None:
    """Run the integration entrypoint for a single curation run.

    Runs the existing ``generate_graph_curation_run_task`` (which is
    sync — the same function the pre-A1 fire-and-forget
    ``asyncio.to_thread`` call in
    ``GraphCurationService.start_run`` invoked) on a worker thread so
    the asyncio event loop stays free. The worker re-loads the
    ``GraphCurationRun`` row from PG on entry, so any state changes
    that happened between enqueue and pop are picked up.
    """

    # Lazy import — avoids pulling the LLM completion stack into the
    # ``aperag.indexing`` package's __init__ at module load (same
    # circular-import discipline as the parse worker; see
    # ``aperag.indexing.__init__`` Wave 3 T3.1 chunk 2 note).
    from aperag.domains.knowledge_graph.tasks import generate_graph_curation_run_task

    try:
        await asyncio.to_thread(
            generate_graph_curation_run_task,
            payload.run_id,
            payload.collection_id,
        )
    except Exception:
        # Task-level failures are recorded by ``generate_graph_curation_run_task``
        # itself (it marks the run FAILED in PG before raising). Logging
        # at this level so the operator sees the worker did pop and
        # dispatch even when the run itself errored.
        logger.exception(
            "graph-curation-run worker: run %s (collection %s) raised; task-level failure already persisted in PG",
            payload.run_id,
            payload.collection_id,
        )


async def run_graph_curation_run_worker_loop(
    *,
    config: GraphCurationRunOrchestratorConfig,
    engine: Engine,
    queue: WorkQueue,
    shutdown: asyncio.Event,
) -> None:
    """Pop graph curation run payloads + dispatch under a concurrency cap.

    Mirrors the ``run_parse_worker_loop`` shape — same shutdown
    discipline, same in-flight set housekeeping, same malformed-payload
    drop semantics. Drains in-flight runs before returning when
    ``shutdown`` is set.
    """

    semaphore = asyncio.Semaphore(config.concurrency)
    in_flight: set[asyncio.Task[None]] = set()

    async def _runner(payload: GraphCurationRunDispatchPayload) -> None:
        async with semaphore:
            await _process_one_run(engine=engine, payload=payload)

    while not shutdown.is_set():
        raw = await queue.pop_graph_curation_run(timeout_seconds=config.poll_timeout_seconds)
        if raw is None:
            in_flight = {t for t in in_flight if not t.done()}
            continue

        try:
            payload = GraphCurationRunDispatchPayload.from_dict(raw)
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "graph-curation-run worker dropping malformed payload: %r (%s)",
                raw,
                exc,
            )
            continue

        task = asyncio.create_task(_runner(payload))
        in_flight.add(task)

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


async def run_graph_curation_run_worker(
    *,
    engine: Engine,
    queue: WorkQueue,
    shutdown: asyncio.Event,
    config: GraphCurationRunOrchestratorConfig | None = None,
) -> None:
    """Thin entrypoint mirroring ``run_*_worker`` for the modality
    lanes + ``run_parse_worker``.

    Wired into the standalone indexing-worker process by
    :mod:`aperag.cli.indexing_worker` as a single asyncio task. The
    default :class:`GraphCurationRunOrchestratorConfig` matches spec
    § 3.1.1 (low concurrency since runs are LLM-heavy).
    """

    await run_graph_curation_run_worker_loop(
        config=config or GraphCurationRunOrchestratorConfig(),
        engine=engine,
        queue=queue,
        shutdown=shutdown,
    )


def drain_graph_curation_run_queue_sync(
    queue: Any,
) -> list[MutableMapping[str, Any]]:
    """Drain the in-memory graph curation run queue synchronously.

    Test helper — production code never calls this. Mirrors
    :func:`drain_queue_sync` for the modality queues; iterates the
    underlying :class:`asyncio.Queue` ``_queue`` deque without
    awaiting so unit tests can assert exactly which payloads landed
    after a synchronous orchestrator step. Only valid against
    :class:`InMemoryWorkQueue`.
    """

    drained: list[MutableMapping[str, Any]] = []
    inner = getattr(queue, "_graph_curation_run_queue", None)
    if inner is None:
        return drained
    while not inner.empty():
        drained.append(inner.get_nowait())
    return drained


__all__ = [
    "DEFAULT_GRAPH_CURATION_RUN_CONCURRENCY",
    "DEFAULT_POLL_TIMEOUT_SECONDS",
    "GraphCurationRunDispatchPayload",
    "GraphCurationRunOrchestratorConfig",
    "drain_graph_curation_run_queue_sync",
    "run_graph_curation_run_worker",
    "run_graph_curation_run_worker_loop",
]
