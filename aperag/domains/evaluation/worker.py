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

"""Runtime execution for evaluation-v3 runs (#evaluation #20 / PR-1b).

``execute_evaluation_run`` is the state-machine orchestration the Celery
task wraps. It loops ``evaluation_run_items`` read from value-copy snapshot
columns — never joining back to mutable ``evaluation_dataset_items`` — and
dispatches each item through ``dispatch_evaluation_turn``, the single
mockable seam to the agent runtime.

Everything that touches ``agent_runtime.runtime.agent_runtime_manager``
lives in ``dispatch_evaluation_turn``; tests mock at that seam so they can
exercise the full state-machine (QUEUED → RUNNING → COMPLETED/FAILED +
``attempts`` + ``run.summary``) without re-asserting chat persistence or
turn creation (those are the ``#13 Chat/agent-runtime`` test domain).

PR-1b scope: do NOT implement judge scoring, ``best_score`` metric, or
complex retry. Attempt status is purely ``COMPLETED``/``FAILED`` based on
the turn's terminal status.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.evaluation.db.models import (
    EvaluationRunItem,
    EvaluationRunItemAttemptStatus,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)
from aperag.domains.evaluation.schemas import EvaluationRunSummary

logger = logging.getLogger(__name__)


@dataclass
class TurnDispatchOutcome:
    """Outcome of a single evaluation-turn dispatch.

    The worker records this verbatim into ``evaluation_run_item_attempts``
    and derives the item status from ``status``. ``agent_chat_id`` and
    ``agent_turn_id`` are free-form strings (not FKs into the chat tables)
    so that chat/turn row deletion never breaks historical run snapshots —
    the identifiers stay auditable even after the underlying records are
    garbage-collected.
    """

    status: EvaluationRunItemAttemptStatus
    agent_chat_id: Optional[str] = None
    agent_turn_id: Optional[str] = None
    answer_text: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None


DispatchFn = Callable[..., Awaitable[TurnDispatchOutcome]]


# ---------------------------------------------------------------------------
# Mockable seam
# ---------------------------------------------------------------------------


async def dispatch_evaluation_turn(
    *,
    user_id: str,
    bot_id: str,
    input_message: str,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.25,
) -> TurnDispatchOutcome:
    """Dispatch a single evaluation turn through ``agent_runtime_manager``.

    This is the only place in the evaluation worker that imports
    ``agent_runtime`` / ``chat_service``. Tests replace it via the
    ``dispatch_fn`` parameter of ``execute_evaluation_run`` so they never
    have to stand up the runtime, a Redis store, or a Celery broker.

    The implementation MUST stay runtime-API-only: no HTTP side-channel
    back to the bot router, no synchronous shortcut around
    ``agent_runtime_manager`` — per the ``#13 agent-runtime`` contract.
    """

    # Local imports: keep the module import-safe for tests that do not
    # stub the runtime (e.g. the router-level contract tests). All
    # dependencies now live in per-domain canonical homes after Phase 5
    # steps 5-S4* (conversation) and 5-S5b (agent_runtime), so the
    # imports are direct cross-domain references — G1 allows
    # domain→domain. ``AgentTurnStatus`` lives next to the runtime in
    # the agent_runtime domain.
    from aperag.domains.agent_runtime.db.models import AgentTurnStatus
    from aperag.domains.agent_runtime.runtime import agent_runtime_manager
    from aperag.domains.agent_runtime.schemas import CreateTurnRequest
    from aperag.domains.conversation.service.chat_service import chat_service_global

    start = time.monotonic()
    chat_view = await chat_service_global.create_chat(user_id, bot_id)
    chat_id = chat_view.id

    turn_request = CreateTurnRequest(query=input_message)
    _chat, _bot, turn, _created = await agent_runtime_manager.turn_service.create_or_get_turn(
        user_id, chat_id, turn_request
    )

    lease_owner = await agent_runtime_manager.claim_turn(turn.id)
    if not lease_owner:
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.FAILED,
            agent_chat_id=chat_id,
            agent_turn_id=turn.id,
            error_message="Could not claim turn for evaluation dispatch",
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    agent_runtime_manager.launch_turn(
        turn=turn,
        chat=_chat,
        bot=_bot,
        user=user_id,
        request=turn_request,
        lease_owner=lease_owner,
    )

    terminal_statuses = {
        AgentTurnStatus.COMPLETED.value,
        AgentTurnStatus.FAILED.value,
        AgentTurnStatus.CANCELLED.value,
    }

    deadline = start + timeout_seconds
    final_status: Optional[str] = None
    while True:
        current = await agent_runtime_manager.turn_service.db_ops.query_agent_turn(user_id, chat_id, turn.id)
        status_value = (
            current.status.value if current and hasattr(current.status, "value") else (current and current.status)
        )
        if status_value in terminal_statuses:
            final_status = status_value
            break
        if time.monotonic() >= deadline:
            try:
                await agent_runtime_manager.cancel_turn(turn.id)
            except Exception:  # noqa: BLE001 - best-effort cancel, still mark failed
                logger.exception("cancel_turn failed for evaluation turn %s", turn.id)
            return TurnDispatchOutcome(
                status=EvaluationRunItemAttemptStatus.FAILED,
                agent_chat_id=chat_id,
                agent_turn_id=turn.id,
                error_message=f"Evaluation turn timed out after {timeout_seconds:.0f}s",
                latency_ms=int((time.monotonic() - start) * 1000),
            )
        await asyncio.sleep(poll_interval_seconds)

    latency_ms = int((time.monotonic() - start) * 1000)

    if final_status == AgentTurnStatus.COMPLETED.value:
        # Phase 8 D8.6 (#80) chunk-2: read canonical UIMessage parts
        # the runtime persists at end-of-turn and join all TextPart
        # contributions into the evaluation answer text.
        persisted = await agent_runtime_manager.uimessage_store.read(turn.id)
        parts = list(persisted.parts) if persisted and persisted.parts else []
        answer_text = _extract_answer_text(parts)
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.COMPLETED,
            agent_chat_id=chat_id,
            agent_turn_id=turn.id,
            answer_text=answer_text,
            latency_ms=latency_ms,
        )

    if final_status == AgentTurnStatus.CANCELLED.value:
        return TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.CANCELLED,
            agent_chat_id=chat_id,
            agent_turn_id=turn.id,
            latency_ms=latency_ms,
        )

    # FAILED
    error_message = None
    try:
        turn_row = await agent_runtime_manager.turn_service.db_ops.query_agent_turn(user_id, chat_id, turn.id)
        if turn_row is not None:
            error_message = getattr(turn_row, "error_message", None)
    except Exception:  # noqa: BLE001
        logger.debug("could not read agent_turn error_message for %s", turn.id, exc_info=True)
    return TurnDispatchOutcome(
        status=EvaluationRunItemAttemptStatus.FAILED,
        agent_chat_id=chat_id,
        agent_turn_id=turn.id,
        error_message=error_message or "Agent turn failed",
        latency_ms=latency_ms,
    )


def _extract_answer_text(parts) -> str:
    """Join the assistant's ``TextPart`` contents into a single string."""
    from aperag.domains.agent_runtime.uimessage import TextPart

    chunks = [part.text for part in parts if isinstance(part, TextPart) and part.text]
    return "".join(chunks)


# ---------------------------------------------------------------------------
# State-machine orchestration
# ---------------------------------------------------------------------------


_ITEM_STATUS_BY_ATTEMPT: dict[EvaluationRunItemAttemptStatus, EvaluationRunItemStatus] = {
    EvaluationRunItemAttemptStatus.COMPLETED: EvaluationRunItemStatus.COMPLETED,
    EvaluationRunItemAttemptStatus.FAILED: EvaluationRunItemStatus.FAILED,
    EvaluationRunItemAttemptStatus.CANCELLED: EvaluationRunItemStatus.CANCELLED,
}


async def execute_evaluation_run(
    run_id: str,
    *,
    db_ops: Optional[AsyncDatabaseOps] = None,
    dispatch_fn: Optional[DispatchFn] = None,
) -> EvaluationRunStatus:
    """Execute an evaluation run end-to-end using only snapshot columns.

    The Celery task wraps this with ``asyncio.run``. Tests call it directly
    with in-memory fakes for ``db_ops`` and ``dispatch_fn``. Returns the
    final run status so the caller can propagate it for observability.

    Contract (per PR-1b hard points):
      * Read run items from ``evaluation_run_items`` snapshot columns ONLY.
        Never call ``list_all_evaluation_dataset_items`` here.
      * Create one ``evaluation_run_item_attempt`` per dispatch, even when
        the outcome is FAILED/CANCELLED.
      * Run transitions QUEUED → RUNNING → COMPLETED (all items succeeded
        or at least one completed), FAILED (all items failed), or CANCELLED
        (external run cancellation or all items cancelled).
      * ``run.summary`` is updated after every item so polling clients see
        incremental progress.
    """

    ops = db_ops or async_db_ops
    dispatch = dispatch_fn or dispatch_evaluation_turn

    run = await ops.get_run_for_worker(run_id)
    if run is None:
        logger.warning("evaluation run %s vanished before worker could pick it up", run_id)
        return EvaluationRunStatus.FAILED

    if EvaluationRunStatus.is_terminal(run.status):
        logger.info(
            "evaluation run %s already in terminal status %s; worker skipping",
            run_id,
            run.status,
        )
        return run.status

    items = await ops.list_all_run_items(run_id)
    summary = EvaluationRunSummary(total=len(items), pending=len(items))
    await ops.update_run_status(
        run_id,
        EvaluationRunStatus.RUNNING,
        summary=summary.model_dump(),
    )

    user_id = run.user_id
    bot_id = run.bot_id

    for item in items:
        current_run = await ops.get_run_for_worker(run_id)
        if current_run is None:
            logger.warning("evaluation run %s vanished while worker was processing items", run_id)
            return EvaluationRunStatus.FAILED
        if EvaluationRunStatus.is_terminal(current_run.status):
            logger.info(
                "evaluation run %s moved to terminal status %s; worker stops dispatching remaining items",
                run_id,
                current_run.status,
            )
            run.status = current_run.status
            break

        await _process_run_item(
            run=run,
            item=item,
            user_id=user_id,
            bot_id=bot_id,
            summary=summary,
            ops=ops,
            dispatch=dispatch,
        )

    latest_run = await ops.get_run_for_worker(run_id)
    final_status = (
        latest_run.status
        if latest_run is not None and EvaluationRunStatus.is_terminal(latest_run.status)
        else _final_run_status(summary)
    )
    await ops.update_run_status(run_id, final_status, summary=summary.model_dump())
    return final_status


async def _process_run_item(
    *,
    run,
    item: EvaluationRunItem,
    user_id: str,
    bot_id: str,
    summary: EvaluationRunSummary,
    ops: AsyncDatabaseOps,
    dispatch: DispatchFn,
) -> None:
    """Drive one item through the state machine and persist its attempt."""

    updated = await ops.mark_run_item_running(item.id)
    attempt_no = updated.attempt_count if updated else 1
    summary.pending = max(0, summary.pending - 1)
    summary.running += 1
    await ops.update_run_status(run.id, EvaluationRunStatus.RUNNING, summary=summary.model_dump())

    try:
        # Value-copy snapshot read — no dataset_items access.
        outcome = await dispatch(
            user_id=user_id,
            bot_id=bot_id,
            input_message=item.input_message,
        )
    except Exception as exc:  # noqa: BLE001 - translate runtime errors to FAILED attempt
        logger.exception("dispatch_evaluation_turn crashed for run_item %s", item.id)
        outcome = TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.FAILED,
            error_message=f"dispatch crashed: {exc}",
        )

    attempt = await ops.create_run_item_attempt(
        run_id=run.id,
        run_item_id=item.id,
        attempt_no=attempt_no,
        status=outcome.status,
        agent_chat_id=outcome.agent_chat_id,
        agent_turn_id=outcome.agent_turn_id,
        answer_text=outcome.answer_text,
        error_message=outcome.error_message,
        latency_ms=outcome.latency_ms,
    )

    final_item_status = _ITEM_STATUS_BY_ATTEMPT.get(outcome.status, EvaluationRunItemStatus.FAILED)
    await ops.finalize_run_item(
        item_id=item.id,
        status=final_item_status,
        latest_attempt_id=attempt.id,
        error_message=outcome.error_message,
    )

    summary.running = max(0, summary.running - 1)
    if final_item_status == EvaluationRunItemStatus.COMPLETED:
        summary.completed += 1
    elif final_item_status == EvaluationRunItemStatus.CANCELLED:
        summary.cancelled += 1
    else:
        summary.failed += 1


def _final_run_status(summary: EvaluationRunSummary) -> EvaluationRunStatus:
    if summary.completed == 0 and summary.failed == 0 and summary.cancelled > 0:
        return EvaluationRunStatus.CANCELLED
    if summary.completed == 0 and summary.failed > 0 and summary.cancelled == 0:
        return EvaluationRunStatus.FAILED
    return EvaluationRunStatus.COMPLETED
