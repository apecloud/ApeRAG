"""Focused tests for the evaluation-v3 worker state machine (#evaluation #20 / PR-1b).

Scope (per PR-1b hard points):

* ``execute_evaluation_run`` reads ``evaluation_run_items`` snapshot columns
  only — never ``evaluation_dataset_items``.
* One attempt is persisted per item, even on failure.
* Item status transitions: PENDING → RUNNING → COMPLETED/FAILED/CANCELLED.
* Run status transitions: QUEUED/RUNNING → RUNNING → COMPLETED (any
  success), FAILED (all failed), or CANCELLED (external cancellation or
  all items cancelled).
* ``run.summary`` counters are updated incrementally; final summary
  equals the number of completed/failed/cancelled items.
* The worker module does not leak legacy Benchmark concepts
  (``benchmark_``, ``dataset_version_id``).
* ``EvaluationRunService.launch_run`` calls the Celery task ``.delay``
  rather than the old no-op.

Runtime integration (``agent_runtime_manager`` / chat persistence / turn
creation) is deliberately NOT re-tested here; that surface is owned by
the #13 agent-runtime suite and must stay mockable through the
``dispatch_fn`` seam per @符炫炜's CR preview.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

from aperag.db.models import (
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunItemAttemptStatus,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)
from aperag.evaluation_v2 import worker
from aperag.evaluation_v2.worker import (
    TurnDispatchOutcome,
    execute_evaluation_run,
)

# ---------------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------------


class _FakeOps:
    """Minimal AsyncDatabaseOps stand-in used by execute_evaluation_run.

    Explicitly does NOT implement ``list_all_evaluation_dataset_items`` —
    any accidental call would raise ``AttributeError`` and fail the test,
    enforcing the "snapshot-only read" hard point.
    """

    def __init__(self, run: EvaluationRun, items: list[EvaluationRunItem]):
        self._run = run
        self._items = {item.id: item for item in items}
        self.update_run_calls: list[tuple[EvaluationRunStatus, dict | None, str | None]] = []
        self.attempts: list[dict] = []
        self.finalized: list[tuple[str, EvaluationRunItemStatus, str | None, str | None]] = []
        self.mark_running_calls: list[str] = []
        self.dataset_items_accessed = False  # flipped only by the guard below

    # --- the small subset of AsyncDatabaseOps the worker uses --------------

    async def get_run_for_worker(self, run_id: str):
        return self._run if self._run.id == run_id else None

    async def list_all_run_items(self, run_id: str):
        assert run_id == self._run.id
        return list(self._items.values())

    async def update_run_status(self, run_id, status, *, error_message=None, summary=None):
        assert run_id == self._run.id
        self.update_run_calls.append((status, summary, error_message))
        self._run.status = status
        if summary is not None:
            self._run.summary = summary
        return self._run

    async def mark_run_item_running(self, item_id: str):
        self.mark_running_calls.append(item_id)
        item = self._items[item_id]
        item.status = EvaluationRunItemStatus.RUNNING
        item.attempt_count = (item.attempt_count or 0) + 1
        return item

    async def create_run_item_attempt(
        self,
        *,
        run_id,
        run_item_id,
        attempt_no,
        status,
        agent_chat_id=None,
        agent_turn_id=None,
        answer_text=None,
        error_message=None,
        latency_ms=None,
    ):
        record = {
            "id": f"attempt_{len(self.attempts) + 1:04d}",
            "run_id": run_id,
            "run_item_id": run_item_id,
            "attempt_no": attempt_no,
            "status": status,
            "agent_chat_id": agent_chat_id,
            "agent_turn_id": agent_turn_id,
            "answer_text": answer_text,
            "error_message": error_message,
            "latency_ms": latency_ms,
        }
        self.attempts.append(record)
        return types.SimpleNamespace(**record)

    async def finalize_run_item(
        self,
        *,
        item_id,
        status,
        latest_attempt_id=None,
        error_message=None,
    ):
        self.finalized.append((item_id, status, latest_attempt_id, error_message))
        item = self._items[item_id]
        item.status = status
        item.latest_attempt_id = latest_attempt_id
        item.error_message = error_message
        return item

    # --- absence guard ------------------------------------------------------

    def __getattr__(self, name):  # pragma: no cover - guard path
        if name == "list_all_evaluation_dataset_items":
            self.dataset_items_accessed = True
            raise AssertionError(
                "worker must not read evaluation_dataset_items; "
                "attempted attribute access: list_all_evaluation_dataset_items"
            )
        raise AttributeError(name)


def _make_run_and_items(*, item_count: int = 3) -> tuple[EvaluationRun, list[EvaluationRunItem]]:
    run = EvaluationRun(
        id="run_0001",
        user_id="user_abc",
        bot_id="bot_xyz",
        dataset_id="eds_0001",
        dataset_name="Smoke QA",
        status=EvaluationRunStatus.QUEUED,
        summary={"total": item_count, "pending": item_count, "running": 0, "completed": 0, "failed": 0, "cancelled": 0},
    )
    items = [
        EvaluationRunItem(
            id=f"item_{idx:04d}",
            run_id=run.id,
            source_dataset_item_id=f"edi_{idx:04d}",
            case_key=f"case_{idx:04d}",
            sort_key=idx,
            input_message=f"What is {idx}?",
            expected_answer=f"{idx}",
            status=EvaluationRunItemStatus.PENDING,
            attempt_count=0,
        )
        for idx in range(item_count)
    ]
    return run, items


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_evaluation_run_success_transitions_all_items_to_completed():
    run, items = _make_run_and_items(item_count=3)
    ops = _FakeOps(run, items)
    captured_inputs: list[str] = []

    async def fake_dispatch(*, user_id, bot_id, input_message, **kwargs):
        captured_inputs.append(input_message)
        assert user_id == "user_abc"
        assert bot_id == "bot_xyz"
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.COMPLETED,
            agent_chat_id=f"chat_{input_message}",
            agent_turn_id=f"turn_{input_message}",
            answer_text=f"answer for {input_message}",
            latency_ms=42,
        )

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.COMPLETED
    assert run.status is EvaluationRunStatus.COMPLETED
    assert [item.status for item in items] == [EvaluationRunItemStatus.COMPLETED] * 3
    assert [a["status"] for a in ops.attempts] == [EvaluationRunItemAttemptStatus.COMPLETED] * 3
    assert captured_inputs == [item.input_message for item in items]
    assert ops.mark_running_calls == [item.id for item in items]
    assert all(a["answer_text"] and a["agent_turn_id"] for a in ops.attempts)

    final_summary = run.summary
    assert final_summary["completed"] == 3
    assert final_summary["failed"] == 0
    assert final_summary["cancelled"] == 0
    assert final_summary["running"] == 0
    assert final_summary["pending"] == 0
    assert final_summary["total"] == 3


def test_execute_evaluation_run_tracks_incremental_progress_in_summary():
    run, items = _make_run_and_items(item_count=2)
    ops = _FakeOps(run, items)

    async def fake_dispatch(**_):
        return TurnDispatchOutcome(status=EvaluationRunItemAttemptStatus.COMPLETED, answer_text="ok")

    asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    # First call: move run into RUNNING with fresh summary.
    first_call = ops.update_run_calls[0]
    assert first_call[0] is EvaluationRunStatus.RUNNING
    assert first_call[1]["pending"] == 2 and first_call[1]["running"] == 0

    # Intermediate RUNNING updates must reflect at least one completed item
    # before the final COMPLETED transition.
    running_snapshots = [call[1] for call in ops.update_run_calls if call[0] is EvaluationRunStatus.RUNNING]
    assert any(s.get("completed", 0) >= 1 for s in running_snapshots[1:]), (
        "summary should reflect completed items before the final run transition"
    )

    # Final update: COMPLETED.
    assert ops.update_run_calls[-1][0] is EvaluationRunStatus.COMPLETED


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_execute_evaluation_run_all_failed_sets_run_failed_and_records_attempts():
    run, items = _make_run_and_items(item_count=2)
    ops = _FakeOps(run, items)

    async def fake_dispatch(**_):
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.FAILED,
            agent_chat_id="chat_x",
            agent_turn_id="turn_x",
            error_message="boom",
            latency_ms=10,
        )

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.FAILED
    assert run.status is EvaluationRunStatus.FAILED
    assert [item.status for item in items] == [EvaluationRunItemStatus.FAILED, EvaluationRunItemStatus.FAILED]
    assert len(ops.attempts) == 2
    for attempt in ops.attempts:
        assert attempt["status"] is EvaluationRunItemAttemptStatus.FAILED
        assert attempt["error_message"] == "boom"
    assert run.summary["failed"] == 2 and run.summary["completed"] == 0


def test_execute_evaluation_run_mixed_outcome_completes_run_and_records_both_attempts():
    run, items = _make_run_and_items(item_count=2)
    ops = _FakeOps(run, items)

    outcomes = iter(
        [
            TurnDispatchOutcome(status=EvaluationRunItemAttemptStatus.COMPLETED, answer_text="ok"),
            TurnDispatchOutcome(
                status=EvaluationRunItemAttemptStatus.FAILED,
                error_message="second item blew up",
            ),
        ]
    )

    async def fake_dispatch(**_):
        return next(outcomes)

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    # One success is enough to mark the run COMPLETED; tracking failed stays in summary.
    assert final is EvaluationRunStatus.COMPLETED
    assert run.summary["completed"] == 1 and run.summary["failed"] == 1
    assert [a["status"] for a in ops.attempts] == [
        EvaluationRunItemAttemptStatus.COMPLETED,
        EvaluationRunItemAttemptStatus.FAILED,
    ]
    assert [item.status for item in items] == [
        EvaluationRunItemStatus.COMPLETED,
        EvaluationRunItemStatus.FAILED,
    ]


def test_execute_evaluation_run_stops_mid_flight_when_run_cancelled():
    run, items = _make_run_and_items(item_count=2)
    ops = _FakeOps(run, items)
    captured_inputs: list[str] = []

    async def fake_dispatch(*, input_message, **_):
        captured_inputs.append(input_message)
        ops._run.status = EvaluationRunStatus.CANCELLED
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.COMPLETED,
            answer_text="first item completed before cancellation",
        )

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.CANCELLED
    assert run.status is EvaluationRunStatus.CANCELLED
    assert captured_inputs == [items[0].input_message]
    assert len(ops.attempts) == 1
    assert items[0].status is EvaluationRunItemStatus.COMPLETED
    assert items[1].status is EvaluationRunItemStatus.PENDING
    assert run.summary["completed"] == 1
    assert run.summary["pending"] == 1
    assert run.summary["running"] == 0


def test_execute_evaluation_run_returns_cancelled_when_all_items_cancelled():
    run, items = _make_run_and_items(item_count=2)
    ops = _FakeOps(run, items)

    async def fake_dispatch(**_):
        return TurnDispatchOutcome(status=EvaluationRunItemAttemptStatus.CANCELLED)

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.CANCELLED
    assert run.status is EvaluationRunStatus.CANCELLED
    assert [item.status for item in items] == [
        EvaluationRunItemStatus.CANCELLED,
        EvaluationRunItemStatus.CANCELLED,
    ]
    assert [attempt["status"] for attempt in ops.attempts] == [
        EvaluationRunItemAttemptStatus.CANCELLED,
        EvaluationRunItemAttemptStatus.CANCELLED,
    ]
    assert run.summary["cancelled"] == 2
    assert run.summary["completed"] == 0
    assert run.summary["failed"] == 0


def test_execute_evaluation_run_does_not_overwrite_externally_failed_run_status():
    run, items = _make_run_and_items(item_count=2)
    ops = _FakeOps(run, items)
    captured_inputs: list[str] = []

    async def fake_dispatch(*, input_message, **_):
        captured_inputs.append(input_message)
        ops._run.status = EvaluationRunStatus.FAILED
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.COMPLETED,
            answer_text="first item completed before external force-fail",
        )

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.FAILED
    assert run.status is EvaluationRunStatus.FAILED
    assert captured_inputs == [items[0].input_message]
    assert len(ops.attempts) == 1
    assert items[0].status is EvaluationRunItemStatus.COMPLETED
    assert items[1].status is EvaluationRunItemStatus.PENDING
    assert run.summary["completed"] == 1
    assert run.summary["pending"] == 1
    assert run.summary["running"] == 0


def test_execute_evaluation_run_catches_dispatch_exception_and_records_failed_attempt():
    run, items = _make_run_and_items(item_count=1)
    ops = _FakeOps(run, items)

    async def crashing_dispatch(**_):
        raise RuntimeError("runtime unreachable")

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=crashing_dispatch))

    assert final is EvaluationRunStatus.FAILED
    assert items[0].status is EvaluationRunItemStatus.FAILED
    assert len(ops.attempts) == 1
    assert ops.attempts[0]["status"] is EvaluationRunItemAttemptStatus.FAILED
    assert "runtime unreachable" in ops.attempts[0]["error_message"]


# ---------------------------------------------------------------------------
# Idempotency + short-circuit
# ---------------------------------------------------------------------------


def test_execute_evaluation_run_short_circuits_when_run_missing():
    run, _ = _make_run_and_items()
    ops = _FakeOps(run, [])
    run.id = "run_present"

    dispatched: list[str] = []

    async def fake_dispatch(**kwargs):  # pragma: no cover - should not run
        dispatched.append(kwargs["input_message"])
        return TurnDispatchOutcome(status=EvaluationRunItemAttemptStatus.COMPLETED)

    final = asyncio.run(execute_evaluation_run("missing_run", db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.FAILED
    assert ops.update_run_calls == [] and dispatched == []


def test_execute_evaluation_run_short_circuits_when_run_already_terminal():
    run, _ = _make_run_and_items()
    run.status = EvaluationRunStatus.COMPLETED
    ops = _FakeOps(run, [])

    async def fake_dispatch(**_):  # pragma: no cover - should not run
        raise AssertionError("dispatch_fn must not be invoked for a terminal run")

    final = asyncio.run(execute_evaluation_run(run.id, db_ops=ops, dispatch_fn=fake_dispatch))

    assert final is EvaluationRunStatus.COMPLETED
    assert ops.update_run_calls == []


# ---------------------------------------------------------------------------
# Static guard: worker must not leak legacy Benchmark concepts
# ---------------------------------------------------------------------------


def test_worker_module_source_has_no_benchmark_or_dataset_version_references():
    source = Path(worker.__file__).read_text(encoding="utf-8").lower()
    forbidden = ("benchmark_", "dataset_version_id", "/api/v2/bots", "httpx")
    for token in forbidden:
        assert token not in source, f"worker.py must not reference '{token}'"


# ---------------------------------------------------------------------------
# Service seam: launch_run calls the Celery task's .delay, not no-op.
# ---------------------------------------------------------------------------


def test_evaluation_run_service_launch_run_dispatches_celery_task(monkeypatch):
    from aperag.evaluation_v2 import services as services_module
    from aperag.evaluation_v2 import tasks as tasks_module

    calls: list[str] = []

    class _FakeTask:
        name = "aperag.evaluation_v2.tasks.run_evaluation_run"

        def delay(self, run_id: str):
            calls.append(run_id)

    fake_task = _FakeTask()
    monkeypatch.setattr(tasks_module, "run_evaluation_run", fake_task)

    svc = services_module.EvaluationRunService(db_ops=object())
    asyncio.run(svc.launch_run("run_abc123"))

    assert calls == ["run_abc123"]
