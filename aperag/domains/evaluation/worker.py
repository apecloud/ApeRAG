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
    completion=None,
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

    ``completion`` (a ``ModelSpec``) is the answer-side model the agent
    runtime v3 needs after PR #1697 / Wave 10. The caller resolves it
    from the run's ``collection_id`` → ``collection.config.completion``
    so each evaluation case is answered by the same LLM the collection
    is configured with — without it the runtime raises ``Model
    specification is required for agent runtime v3`` and every case in
    the run fails (regression caught via @earayu2 msg=c44beebc).
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

    turn_request = CreateTurnRequest(query=input_message, completion=completion)
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

    # Resolve the answer-side completion model spec once per run.
    # Priority: ``run.answer_model`` (FE override) → ``collection.config.completion``
    # → ``None`` (let the runtime raise its own missing-model error).
    completion = await _resolve_run_completion(
        user_id=user_id,
        collection_id=getattr(run, "collection_id", None),
        run_model_id=getattr(run, "answer_model", None),
    )

    # Build the judge once per run. ``judge_config.mode`` selects the
    # branch; ``LLM_AS_JUDGE`` resolves a per-run LLM callable from
    # ``run.judge_model`` → ``collection.config.completion`` and falls
    # through to ``ExactMatchJudge`` if the collection has no LLM
    # configured (so the run still finishes; operators see the score
    # collapse to 0/1 and can fix the config).
    from aperag.domains.evaluation.judges import build_judge
    from aperag.domains.evaluation.schemas import JudgeConfig

    judge_config = run.judge_config
    if isinstance(judge_config, dict):
        judge_config = JudgeConfig(**judge_config)
    judge_llm = await _resolve_judge_llm(
        user_id=user_id,
        collection_id=getattr(run, "collection_id", None),
        run_model_id=getattr(run, "judge_model", None),
    )
    judge = build_judge(judge_config, llm=judge_llm)

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
            completion=completion,
            judge=judge,
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


async def _resolve_run_completion(
    *,
    user_id: str,
    collection_id: Optional[str],
    run_model_id: Optional[str] = None,
):
    """Return the answer-side ``ModelSpec`` for the run.

    Resolution priority (architect spec ``msg=2424afe2``):

    1. ``run_model_id`` — the FE-provided ``run.answer_model`` override
    2. ``collection.config.completion`` — the collection's default LLM
    3. ``None`` — let the runtime raise its native missing-model error

    Mirrors the ``_invoke_summary_agent`` pattern from PR #1825.
    """
    from aperag.db.ops import async_db_ops as _db_ops
    from aperag.schema.common import ModelSpec
    from aperag.schema.utils import parseCollectionConfig

    if run_model_id:
        # FE override path — answer model is whatever the operator
        # picked when launching the run. Temperature is not part of the
        # ``run.answer_model`` storage; default to the same value the
        # collection-default branch uses (None → runtime default).
        return ModelSpec(model_id=run_model_id)
    if not collection_id:
        return None

    collection = await _db_ops.query_collection(user_id, collection_id)
    if collection is None:
        logger.info(
            "evaluation worker: collection %s for run vanished or unauthorized; "
            "agent runtime will surface its own missing-model error",
            collection_id,
        )
        return None
    try:
        parsed = parseCollectionConfig(collection.config)
    except ValueError:
        logger.exception(
            "evaluation worker: failed to parse collection %s config; falling through",
            collection_id,
        )
        return None
    completion = parsed.completion if parsed else None
    if completion is None or not completion.model_id:
        logger.info(
            "evaluation worker: collection %s has no completion model — agent runtime "
            "will surface its own missing-model error",
            collection_id,
        )
        return None
    return ModelSpec(
        model_id=completion.model_id,
        temperature=completion.temperature,
    )


async def _resolve_judge_llm(
    *,
    user_id: str,
    collection_id: Optional[str],
    run_model_id: Optional[str] = None,
):
    """Build the per-run async ``(prompt) -> str`` callable for the
    LLM-as-judge MVP, or ``None`` when no judge model can be resolved.

    Resolution mirrors ``_resolve_run_completion``: ``run.judge_model``
    override → ``collection.config.completion`` → ``None``. ``None``
    causes ``build_judge`` to fall back to ``ExactMatchJudge`` so the
    run still finishes deterministically; operators see the
    ``judge_score`` collapse to ``0`` / ``1`` and can fix config.
    """
    from aperag.db.ops import async_db_ops as _db_ops

    if run_model_id:
        # Looking up an arbitrary model_id without a collection context
        # requires a model-resolver helper that we do not have today.
        # When the FE provides an override but no collection, fall back
        # to ``None`` and let ``ExactMatchJudge`` run — the override
        # branch only fires once Phase 6 ratifies a global resolver.
        if not collection_id:
            logger.info(
                "evaluation worker: judge_model override set but run has no "
                "collection_id — cannot resolve model runtime, falling back to "
                "exact-match judge"
            )
            return None
        # Build a callable from the override model_id. Reuse the same
        # ``build_collection_llm_callable`` machinery by stuffing the
        # override into a copy of the collection's config.
        collection = await _db_ops.query_collection(user_id, collection_id)
        if collection is None:
            return None
        return _build_collection_llm_with_override(collection, run_model_id)

    if not collection_id:
        return None
    collection = await _db_ops.query_collection(user_id, collection_id)
    if collection is None:
        return None
    try:
        from aperag.indexing.llm import build_collection_llm_callable

        return build_collection_llm_callable(collection)
    except RuntimeError:
        logger.info(
            "evaluation worker: collection %s has no completion model for the "
            "judge LLM either; LlmAsJudge will fall back to ExactMatch",
            collection_id,
        )
        return None
    except Exception:  # noqa: BLE001 — judge LLM build must never poison the run
        logger.exception("evaluation worker: build_collection_llm_callable raised for judge LLM")
        return None


def _build_collection_llm_with_override(collection, override_model_id: str):
    """Mint an async LLM callable bound to ``override_model_id`` while
    reusing the collection's provider/account/api-key resolution
    machinery. Returns ``None`` if the override model cannot be
    resolved.
    """
    import json

    try:
        config_dict = json.loads(collection.config or "{}")
    except json.JSONDecodeError:
        return None
    completion = config_dict.setdefault("completion", {})
    completion["model_id"] = override_model_id
    completion.pop("temperature", None)  # use override defaults

    # Attach a shallow clone with the patched config so the helper sees
    # ``override_model_id`` without mutating the persisted row.
    from types import SimpleNamespace

    patched = SimpleNamespace(
        id=getattr(collection, "id", None),
        user=getattr(collection, "user", None),
        config=json.dumps(config_dict),
    )
    try:
        from aperag.indexing.llm import build_collection_llm_callable

        return build_collection_llm_callable(patched)
    except Exception:  # noqa: BLE001 — same fall-through as above
        logger.exception(
            "evaluation worker: failed to bind judge_model override %s on collection %s",
            override_model_id,
            getattr(collection, "id", None),
        )
        return None


async def _process_run_item(
    *,
    run,
    item: EvaluationRunItem,
    user_id: str,
    bot_id: str,
    completion,
    judge,
    summary: EvaluationRunSummary,
    ops: AsyncDatabaseOps,
    dispatch: DispatchFn,
) -> None:
    """Drive one item through the state machine and persist its attempt.

    Architect spec ``msg=2424afe2``: after a successful turn dispatch
    we run the configured ``judge`` to compute ``judge_score`` /
    ``judge_reason`` / ``judge_breakdown`` and persist them on the
    attempt + finalize_run_item rows.
    """

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
            completion=completion,
        )
    except Exception as exc:  # noqa: BLE001 - translate runtime errors to FAILED attempt
        logger.exception("dispatch_evaluation_turn crashed for run_item %s", item.id)
        outcome = TurnDispatchOutcome(
            status=EvaluationRunItemAttemptStatus.FAILED,
            error_message=f"dispatch crashed: {exc}",
        )

    judge_score: Optional[float] = None
    judge_result: Optional[dict] = None
    judge_breakdown: Optional[dict] = None

    if outcome.status == EvaluationRunItemAttemptStatus.COMPLETED and judge is not None:
        from aperag.domains.evaluation.judges import JudgeInput

        try:
            verdict = await judge.judge(
                JudgeInput(
                    case_key=item.case_key,
                    question=item.input_message,
                    expected_answer=item.expected_answer,
                    reference_context=item.reference_context,
                    actual_answer=outcome.answer_text or "",
                )
            )
        except Exception:  # noqa: BLE001 — judge failure must not corrupt the attempt
            logger.exception(
                "judge crashed for run_item %s; persisting attempt without score",
                item.id,
            )
        else:
            judge_score = float(verdict.score)
            judge_result = {
                "score": judge_score,
                "reason": verdict.reasoning,
                "passed": verdict.passed,
                "raw": verdict.raw,
            }
            judge_breakdown = verdict.breakdown

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
        score=judge_score,
        judge_result=judge_result,
    )

    final_item_status = _ITEM_STATUS_BY_ATTEMPT.get(outcome.status, EvaluationRunItemStatus.FAILED)
    await ops.finalize_run_item(
        item_id=item.id,
        status=final_item_status,
        latest_attempt_id=attempt.id,
        error_message=outcome.error_message,
        best_score=judge_score,
        judge_breakdown=judge_breakdown,
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
