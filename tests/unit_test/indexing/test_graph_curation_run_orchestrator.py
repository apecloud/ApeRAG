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

"""Unit tests for the graph curation run worker (task #31 Phase A1).

Pinned by spec ``task-31-graph-node-merge-spec-v1.md`` §§ 3.1.1 +
3.1.1.b + 5.2.a. The contract:

* ``WorkQueue.push_graph_curation_run`` / ``pop_graph_curation_run``
  is an **independent queue family** — payloads pushed onto it MUST
  NOT leak into the per-:class:`Modality` lanes, and vice versa.
* ``run_graph_curation_run_worker`` blocks on
  ``pop_graph_curation_run``, decodes the payload via
  :class:`GraphCurationRunDispatchPayload`, and dispatches to the
  existing ``generate_graph_curation_run_task`` integration path
  inside a ``asyncio.to_thread`` so the asyncio loop stays free.
* Malformed payloads are dropped with a logged error rather than
  taking the loop down.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from aperag.indexing import (
    GraphCurationRunDispatchPayload,
    GraphCurationRunOrchestratorConfig,
    InMemoryWorkQueue,
    drain_graph_curation_run_queue_sync,
    run_graph_curation_run_worker_loop,
)
from aperag.indexing.models import Modality
from aperag.indexing.orchestrator import RedisWorkQueue

# ---------------------------------------------------------------------
# Payload roundtrip
# ---------------------------------------------------------------------


def test_payload_roundtrip_minimal_keys():
    payload = GraphCurationRunDispatchPayload(run_id="run-123", collection_id="col-456")
    raw = payload.to_dict()
    assert raw == {"run_id": "run-123", "collection_id": "col-456"}
    back = GraphCurationRunDispatchPayload.from_dict(raw)
    assert back == payload


def test_payload_from_dict_coerces_to_str():
    """Even if the queue carries ints (e.g. test fixtures), the
    payload must normalise to strings — downstream PG lookup expects
    string ids."""
    payload = GraphCurationRunDispatchPayload.from_dict({"run_id": 42, "collection_id": "c1"})
    assert payload.run_id == "42"
    assert payload.collection_id == "c1"


def test_payload_from_dict_rejects_missing_required():
    with pytest.raises(KeyError):
        GraphCurationRunDispatchPayload.from_dict({"collection_id": "c1"})
    with pytest.raises(KeyError):
        GraphCurationRunDispatchPayload.from_dict({"run_id": "r1"})


# ---------------------------------------------------------------------
# WorkQueue independence — pinned by spec § 3.1.1 + § 5.2.a
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_graph_curation_run_queue_is_independent_of_modality_queues():
    """Pushing onto ``q:graph_curation_run`` MUST NOT show up on any
    per-:class:`Modality` queue. This is the core "independent queue
    family" invariant ziang msg=92321bcc + Bryce msg=4c23f87e
    BLOCKER 1 caught."""
    queue = InMemoryWorkQueue()
    await queue.push_graph_curation_run(payload={"run_id": "r1", "collection_id": "c1"})

    # Every modality queue must be empty.
    for modality in Modality:
        assert queue.qsize(modality) == 0, (
            f"q:graph_curation_run payload leaked into Modality.{modality.value} — queue families must be isolated"
        )
    assert queue.parse_qsize() == 0
    # The dedicated counter sees it.
    assert queue.graph_curation_run_qsize() == 1


@pytest.mark.asyncio
async def test_inmemory_modality_push_does_not_leak_to_graph_curation_run_queue():
    """Symmetric: pushing onto a :class:`Modality` queue must not
    show up on ``q:graph_curation_run``."""
    queue = InMemoryWorkQueue()
    await queue.push(modality=Modality.VECTOR, payload={"index_id": 1})
    assert queue.graph_curation_run_qsize() == 0
    assert queue.qsize(Modality.VECTOR) == 1


@pytest.mark.asyncio
async def test_inmemory_pop_graph_curation_run_returns_pushed_payload():
    queue = InMemoryWorkQueue()
    await queue.push_graph_curation_run(payload={"run_id": "r1", "collection_id": "c1"})
    raw = await queue.pop_graph_curation_run(timeout_seconds=0.5)
    assert raw == {"run_id": "r1", "collection_id": "c1"}


@pytest.mark.asyncio
async def test_inmemory_pop_graph_curation_run_times_out_when_empty():
    queue = InMemoryWorkQueue()
    raw = await queue.pop_graph_curation_run(timeout_seconds=0.05)
    assert raw is None


def test_redis_graph_curation_run_key_constant_is_distinct():
    """Pinned by spec § 3.1.1: Redis key ``q:graph_curation_run`` MUST
    be distinct from the modality template ``q:indexing:<modality>``
    so a Redis ``KEYS`` audit can't confuse the two families."""
    assert RedisWorkQueue.GRAPH_CURATION_RUN_KEY == "q:graph_curation_run"
    # No modality value should ever produce the graph curation key.
    for modality in Modality:
        assert RedisWorkQueue._key(modality) != RedisWorkQueue.GRAPH_CURATION_RUN_KEY, (
            f"Modality.{modality.value} key collides with graph_curation_run key"
        )


def test_drain_graph_curation_run_queue_sync_helper():
    queue = InMemoryWorkQueue()
    queue._graph_curation_run_queue.put_nowait({"run_id": "r1", "collection_id": "c1"})
    queue._graph_curation_run_queue.put_nowait({"run_id": "r2", "collection_id": "c2"})
    drained = drain_graph_curation_run_queue_sync(queue)
    assert drained == [
        {"run_id": "r1", "collection_id": "c1"},
        {"run_id": "r2", "collection_id": "c2"},
    ]
    assert queue.graph_curation_run_qsize() == 0


# ---------------------------------------------------------------------
# Worker loop — pop + dispatch + shutdown semantics
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_loop_dispatches_to_generate_task():
    """The loop must pop a payload and call
    ``generate_graph_curation_run_task(run_id, collection_id)`` via
    ``asyncio.to_thread`` (so the event loop stays free)."""
    queue = InMemoryWorkQueue()
    await queue.push_graph_curation_run(payload={"run_id": "r1", "collection_id": "c1"})

    captured: list[tuple[str, str]] = []

    def _fake_task(run_id: str, collection_id: str) -> None:
        captured.append((run_id, collection_id))

    shutdown = asyncio.Event()
    config = GraphCurationRunOrchestratorConfig(concurrency=1, poll_timeout_seconds=0.05)

    with patch(
        "aperag.domains.knowledge_graph.tasks.generate_graph_curation_run_task",
        new=_fake_task,
    ):
        loop_task = asyncio.create_task(
            run_graph_curation_run_worker_loop(
                config=config,
                engine=None,  # not used in this test path
                queue=queue,
                shutdown=shutdown,
            )
        )
        # Give the loop time to pop + dispatch + return through to_thread.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if captured:
                break
        shutdown.set()
        await asyncio.wait_for(loop_task, timeout=2.0)

    assert captured == [("r1", "c1")]


@pytest.mark.asyncio
async def test_worker_loop_drops_malformed_payload_without_crashing():
    """Malformed payloads (missing ``run_id`` / ``collection_id``)
    must be logged + dropped, not crash the loop. Pinned because a
    crash would silently halt ALL future runs in this process."""
    queue = InMemoryWorkQueue()
    # Malformed: missing run_id.
    await queue.push_graph_curation_run(payload={"collection_id": "c1"})
    # Then a well-formed one to confirm the loop continued.
    await queue.push_graph_curation_run(payload={"run_id": "r2", "collection_id": "c2"})

    captured: list[tuple[str, str]] = []

    def _fake_task(run_id: str, collection_id: str) -> None:
        captured.append((run_id, collection_id))

    shutdown = asyncio.Event()
    config = GraphCurationRunOrchestratorConfig(concurrency=1, poll_timeout_seconds=0.05)

    with patch(
        "aperag.domains.knowledge_graph.tasks.generate_graph_curation_run_task",
        new=_fake_task,
    ):
        loop_task = asyncio.create_task(
            run_graph_curation_run_worker_loop(
                config=config,
                engine=None,
                queue=queue,
                shutdown=shutdown,
            )
        )
        for _ in range(40):
            await asyncio.sleep(0.05)
            if captured:
                break
        shutdown.set()
        await asyncio.wait_for(loop_task, timeout=2.0)

    assert captured == [("r2", "c2")]


@pytest.mark.asyncio
async def test_worker_loop_swallows_task_exception():
    """A raise inside ``generate_graph_curation_run_task`` must not
    crash the loop — the integration path is responsible for marking
    the run FAILED in PG before raising; the worker logs and moves
    on. Pinned because a propagating exception would halt the whole
    indexing-worker process."""
    queue = InMemoryWorkQueue()
    await queue.push_graph_curation_run(payload={"run_id": "r1", "collection_id": "c1"})
    await queue.push_graph_curation_run(payload={"run_id": "r2", "collection_id": "c2"})

    seen: list[str] = []

    def _fake_task(run_id: str, collection_id: str) -> None:
        seen.append(run_id)
        if run_id == "r1":
            raise RuntimeError("simulated task failure")

    shutdown = asyncio.Event()
    config = GraphCurationRunOrchestratorConfig(concurrency=1, poll_timeout_seconds=0.05)

    with patch(
        "aperag.domains.knowledge_graph.tasks.generate_graph_curation_run_task",
        new=_fake_task,
    ):
        loop_task = asyncio.create_task(
            run_graph_curation_run_worker_loop(
                config=config,
                engine=None,
                queue=queue,
                shutdown=shutdown,
            )
        )
        for _ in range(60):
            await asyncio.sleep(0.05)
            if "r2" in seen:
                break
        shutdown.set()
        await asyncio.wait_for(loop_task, timeout=2.0)

    assert seen == ["r1", "r2"], (
        "Worker loop did not continue past a task failure — first raise "
        "halted the loop, leaving subsequent runs stranded"
    )


@pytest.mark.asyncio
async def test_worker_loop_drains_in_flight_on_shutdown():
    """Shutdown must wait for in-flight runs to finish (or the
    timeout in :mod:`aperag.cli.indexing_worker`'s outer
    ``asyncio.wait_for`` will cancel them). The loop itself just
    awaits its in_flight set — pin that here so the shutdown
    discipline is explicit and tested.

    Implementation note: the worker hands the sync task to
    ``asyncio.to_thread``, so we signal back to the event loop with
    :meth:`asyncio.AbstractEventLoop.call_soon_threadsafe`.
    """
    queue = InMemoryWorkQueue()
    await queue.push_graph_curation_run(payload={"run_id": "r1", "collection_id": "c1"})

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()

    def _slow_task(run_id: str, collection_id: str) -> None:
        import time

        loop.call_soon_threadsafe(started.set)
        time.sleep(0.1)
        loop.call_soon_threadsafe(finished.set)

    shutdown = asyncio.Event()
    config = GraphCurationRunOrchestratorConfig(concurrency=1, poll_timeout_seconds=0.05)

    with patch(
        "aperag.domains.knowledge_graph.tasks.generate_graph_curation_run_task",
        new=_slow_task,
    ):
        loop_task = asyncio.create_task(
            run_graph_curation_run_worker_loop(
                config=config,
                engine=None,
                queue=queue,
                shutdown=shutdown,
            )
        )
        # Wait for the task to actually start.
        await asyncio.wait_for(started.wait(), timeout=2.0)
        # Now request shutdown — the loop should drain in-flight before returning.
        shutdown.set()
        await asyncio.wait_for(loop_task, timeout=3.0)

    assert finished.is_set(), "Shutdown did not drain in-flight task before returning"
