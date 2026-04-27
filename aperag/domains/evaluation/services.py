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

"""Orchestration services for the simplified evaluation API.

User-visible surface is ``Dataset + Evaluation/Run`` only. The dataset service
owns CRUD for ``evaluation_datasets`` / ``evaluation_dataset_items``; the run
service owns the run lifecycle and implements snapshot-on-run. Run detail /
list / attempt read paths must only read ``evaluation_run_items`` snapshot
columns and must never join back to mutable ``evaluation_dataset_items`` rows.
"""

from typing import Optional

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.evaluation.constants import DEFAULT_AGENT_BOT_TITLE
from aperag.domains.evaluation.db.models import (
    EvaluationDataset,
    EvaluationDatasetItem,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)
from aperag.domains.evaluation.schemas import (
    EvaluationDatasetCreate,
    EvaluationDatasetEnvelope,
    EvaluationDatasetItemCreate,
    EvaluationDatasetItemEnvelope,
    EvaluationDatasetItemUpdate,
    EvaluationDatasetUpdate,
    EvaluationRunCreate,
    EvaluationRunDetailResponse,
    EvaluationRunEnvelope,
    EvaluationRunItemAttemptEnvelope,
    EvaluationRunItemEnvelope,
    EvaluationRunProgress,
    EvaluationRunSummary,
    JudgeConfig,
)
from aperag.exceptions import ResourceNotFoundException, ValidationException


def _to_dataset_envelope(dataset: EvaluationDataset) -> EvaluationDatasetEnvelope:
    return EvaluationDatasetEnvelope.model_validate(dataset)


def _to_dataset_item_envelope(item: EvaluationDatasetItem) -> EvaluationDatasetItemEnvelope:
    return EvaluationDatasetItemEnvelope.model_validate(item)


def _to_run_envelope(run: EvaluationRun) -> EvaluationRunEnvelope:
    envelope = EvaluationRunEnvelope.model_validate(run)
    if run.summary:
        try:
            envelope.summary = EvaluationRunSummary.model_validate(run.summary)
        except Exception:
            envelope.summary = None
    if run.judge_config:
        try:
            envelope.judge_config = JudgeConfig.model_validate(run.judge_config)
        except Exception:
            envelope.judge_config = None
    return envelope


def _to_run_progress(summary: Optional[EvaluationRunSummary]) -> EvaluationRunProgress:
    if not summary or summary.total <= 0:
        return EvaluationRunProgress(percent=0)

    resolved = summary.completed + summary.failed + summary.cancelled
    return EvaluationRunProgress(percent=round((resolved / summary.total) * 100))


def _requeue_run_summary(
    summary_payload: Optional[dict],
    item_status: EvaluationRunItemStatus,
) -> Optional[dict]:
    if not summary_payload:
        return None

    summary = EvaluationRunSummary.model_validate(summary_payload)
    summary.pending += 1

    if item_status == EvaluationRunItemStatus.FAILED and summary.failed > 0:
        summary.failed -= 1
    if item_status == EvaluationRunItemStatus.CANCELLED and summary.cancelled > 0:
        summary.cancelled -= 1

    return summary.model_dump()


def _auto_case_key(index: int) -> str:
    return f"case_{index + 1:04d}"


def _items_from_payload(
    dataset_id: str,
    payloads: list[EvaluationDatasetItemCreate],
    start_index: int = 0,
) -> list[EvaluationDatasetItem]:
    seen_keys: set[str] = set()
    items: list[EvaluationDatasetItem] = []
    for offset, payload in enumerate(payloads):
        idx = start_index + offset
        key = (payload.case_key or _auto_case_key(idx)).strip()
        if not key:
            key = _auto_case_key(idx)
        if key in seen_keys:
            raise ValidationException(f"Duplicate case_key within dataset payload: {key}")
        seen_keys.add(key)
        items.append(
            EvaluationDatasetItem(
                dataset_id=dataset_id,
                case_key=key,
                input_message=payload.input_message,
                expected_answer=payload.expected_answer,
                reference_context=payload.reference_context,
                tags=payload.tags,
                case_metadata=payload.case_metadata,
                sort_key=payload.sort_key if payload.sort_key is not None else idx,
            )
        )
    return items


class EvaluationDatasetService:
    """CRUD + item management for ``evaluation_datasets`` / ``evaluation_dataset_items``."""

    def __init__(self, db_ops: Optional[AsyncDatabaseOps] = None):
        self.db_ops = db_ops or async_db_ops

    async def create_dataset(self, user_id: str, request: EvaluationDatasetCreate) -> EvaluationDatasetEnvelope:
        dataset = EvaluationDataset(
            user_id=user_id,
            collection_id=request.collection_id,
            name=request.name,
            description=request.description,
            source_type=request.source_type,
            schema_hint=request.schema_hint,
            item_count=0,
        )
        payload_items = list(request.items or [])
        items = _items_from_payload(dataset_id="", payloads=payload_items)
        created = await self.db_ops.create_evaluation_dataset(dataset, items)
        return _to_dataset_envelope(created)

    async def list_datasets(
        self,
        user_id: str,
        collection_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[EvaluationDatasetEnvelope], int]:
        rows, total = await self.db_ops.list_evaluation_datasets(
            user_id=user_id,
            collection_id=collection_id,
            page=page,
            page_size=page_size,
        )
        return [_to_dataset_envelope(r) for r in rows], total

    async def get_dataset(self, user_id: str, dataset_id: str) -> EvaluationDatasetEnvelope:
        dataset = await self._require_dataset(user_id, dataset_id)
        return _to_dataset_envelope(dataset)

    async def update_dataset(
        self,
        user_id: str,
        dataset_id: str,
        request: EvaluationDatasetUpdate,
    ) -> EvaluationDatasetEnvelope:
        updated = await self.db_ops.update_evaluation_dataset(
            user_id,
            dataset_id,
            name=request.name,
            description=request.description,
        )
        if not updated:
            raise ResourceNotFoundException("EvaluationDataset", dataset_id)
        return _to_dataset_envelope(updated)

    async def delete_dataset(self, user_id: str, dataset_id: str) -> None:
        ok = await self.db_ops.soft_delete_evaluation_dataset(user_id, dataset_id)
        if not ok:
            raise ResourceNotFoundException("EvaluationDataset", dataset_id)

    # Items ------------------------------------------------------------

    async def list_items(
        self,
        user_id: str,
        dataset_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[EvaluationDatasetItemEnvelope], int]:
        await self._require_dataset(user_id, dataset_id)
        rows, total = await self.db_ops.list_evaluation_dataset_items(
            dataset_id=dataset_id, page=page, page_size=page_size
        )
        return [_to_dataset_item_envelope(r) for r in rows], total

    async def append_items(
        self,
        user_id: str,
        dataset_id: str,
        payloads: list[EvaluationDatasetItemCreate],
    ) -> list[EvaluationDatasetItemEnvelope]:
        dataset = await self._require_dataset(user_id, dataset_id)
        start_index = dataset.item_count or 0
        items = _items_from_payload(dataset_id=dataset.id, payloads=payloads, start_index=start_index)
        created = await self.db_ops.append_evaluation_dataset_items(dataset.id, items)
        return [_to_dataset_item_envelope(r) for r in created]

    async def update_item(
        self,
        user_id: str,
        dataset_id: str,
        item_id: str,
        request: EvaluationDatasetItemUpdate,
    ) -> EvaluationDatasetItemEnvelope:
        await self._require_dataset(user_id, dataset_id)
        updated = await self.db_ops.update_evaluation_dataset_item(
            dataset_id=dataset_id,
            item_id=item_id,
            **request.model_dump(exclude_unset=True),
        )
        if not updated:
            raise ResourceNotFoundException("EvaluationDatasetItem", item_id)
        return _to_dataset_item_envelope(updated)

    async def delete_item(self, user_id: str, dataset_id: str, item_id: str) -> None:
        await self._require_dataset(user_id, dataset_id)
        ok = await self.db_ops.soft_delete_evaluation_dataset_item(dataset_id, item_id)
        if not ok:
            raise ResourceNotFoundException("EvaluationDatasetItem", item_id)

    async def list_all_items(self, dataset_id: str) -> list[EvaluationDatasetItem]:
        return await self.db_ops.list_all_evaluation_dataset_items(dataset_id)

    async def _require_dataset(self, user_id: str, dataset_id: str) -> EvaluationDataset:
        dataset = await self.db_ops.get_evaluation_dataset(user_id, dataset_id)
        if not dataset:
            raise ResourceNotFoundException("EvaluationDataset", dataset_id)
        return dataset


class EvaluationRunService:
    """Evaluation run lifecycle: create / list / get / cancel / retry.

    ``create_run`` owns the snapshot-on-run transaction: value-copy dataset
    items into ``evaluation_run_items``. Run detail / list / attempt read paths
    only read these snapshot rows and never join the mutable
    ``evaluation_dataset_items`` rows.
    """

    def __init__(
        self,
        db_ops: Optional[AsyncDatabaseOps] = None,
        dataset_service: Optional[EvaluationDatasetService] = None,
    ):
        self.db_ops = db_ops or async_db_ops
        self.dataset_service = dataset_service or EvaluationDatasetService(self.db_ops)

    async def create_run(self, user_id: str, request: EvaluationRunCreate) -> EvaluationRunEnvelope:
        dataset = await self.dataset_service._require_dataset(user_id, request.dataset_id)
        items = await self.dataset_service.list_all_items(dataset.id)
        if not items:
            raise ValidationException("Evaluation dataset has no items")

        bot_id = await self._resolve_bot_for_run(user_id, request.bot_id)

        judge_payload = request.judge.model_dump() if request.judge else None
        run = EvaluationRun(
            user_id=user_id,
            bot_id=bot_id,
            dataset_id=dataset.id,
            collection_id=dataset.collection_id,
            dataset_name=dataset.name,
            name=request.name,
            bot_config_snapshot=request.bot_config_snapshot,
            model_config_snapshot=request.model_config_snapshot,
            judge_config=judge_payload,
            status=EvaluationRunStatus.QUEUED,
            summary=EvaluationRunSummary(total=len(items), pending=len(items)).model_dump(),
        )
        run_items = [
            EvaluationRunItem(
                source_dataset_item_id=item.id,
                case_key=item.case_key,
                sort_key=item.sort_key,
                # Value-copy snapshot — must not join back to mutable dataset items.
                input_message=item.input_message,
                expected_answer=item.expected_answer,
                reference_context=item.reference_context,
                tags=item.tags,
                case_metadata=item.case_metadata,
            )
            for item in items
        ]

        created = await self.db_ops.create_evaluation_run(run, run_items)
        await self.launch_run(created.id)
        return _to_run_envelope(created)

    async def _resolve_bot_for_run(self, user_id: str, explicit_bot_id: Optional[str]) -> str:
        """Pick the bot for an evaluation run.

        Priority (no HTTP side-channels):
          (a) explicit ``bot_id`` — must be active and user-owned;
          (b) otherwise the user's active bot whose title matches
              ``DEFAULT_AGENT_BOT_TITLE``;
          (c) otherwise the user's oldest active bot (``gmt_created ASC``);
          (d) no active bot → ``ValidationException``.
        """
        if explicit_bot_id:
            bot = await self.db_ops.query_bot(user_id, explicit_bot_id)
            if not bot:
                raise ValidationException("bot_id is not owned by the current user or is inactive")
            return bot.id

        resolved = await self.db_ops.resolve_default_evaluation_bot(user_id, default_title=DEFAULT_AGENT_BOT_TITLE)
        if not resolved:
            raise ValidationException("No bot available for evaluation run; specify bot_id explicitly")
        return resolved.id

    async def launch_run(self, run_id: str) -> None:
        """Hand the run off to the async worker pipeline (Celery producer seam).

        The task implementation lives in :mod:`aperag.evaluation_v2.tasks`
        and drains ``evaluation_run_items`` through
        :func:`aperag.evaluation_v2.worker.execute_evaluation_run`. The
        import is lazy so unit tests for create/list/cancel paths do not
        have to stand up Celery just to exercise the service.
        """

        # Wave 3 T3.1 chunk 2 + post-pass-6 fix: Pattern C
        # fire-and-forget — formerly ``run_evaluation_run.delay(run_id)``
        # Celery enqueue. ``run_evaluation_run`` is a coroutine, so we
        # schedule it directly on the caller's event loop. Wrapping in
        # ``asyncio.to_thread`` (the prior recipe) spawned a new event
        # loop on a worker thread and blew up asyncpg pool affinity
        # ("Future attached to a different loop"); see the docstring on
        # ``run_evaluation_run`` for the failure mode.
        import asyncio

        from aperag.domains.evaluation.tasks import run_evaluation_run

        asyncio.create_task(run_evaluation_run(run_id))

    async def list_runs(
        self,
        user_id: str,
        bot_id: Optional[str],
        dataset_id: Optional[str],
        collection_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[EvaluationRunEnvelope], int]:
        rows, total = await self.db_ops.list_evaluation_runs(
            user_id=user_id,
            bot_id=bot_id,
            dataset_id=dataset_id,
            collection_id=collection_id,
            page=page,
            page_size=page_size,
        )
        return [_to_run_envelope(r) for r in rows], total

    async def get_run(self, user_id: str, run_id: str) -> EvaluationRunEnvelope:
        run = await self._require_run(user_id, run_id)
        return _to_run_envelope(run)

    async def get_run_detail(self, user_id: str, run_id: str) -> EvaluationRunDetailResponse:
        run = await self._require_run(user_id, run_id)
        envelope = _to_run_envelope(run)
        return EvaluationRunDetailResponse(
            run=envelope,
            summary=envelope.summary,
            progress=_to_run_progress(envelope.summary),
        )

    async def cancel_run(self, user_id: str, run_id: str) -> EvaluationRunEnvelope:
        run = await self._require_run(user_id, run_id)
        if run.status in (
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.FAILED,
            EvaluationRunStatus.CANCELLED,
        ):
            return _to_run_envelope(run)

        updated = await self.db_ops.update_run_status(run_id, EvaluationRunStatus.CANCELLED)
        return _to_run_envelope(updated)

    async def retry_run_item(self, user_id: str, run_id: str, item_id: str) -> EvaluationRunItemEnvelope:
        run = await self._require_run(user_id, run_id)
        item = await self.db_ops.get_run_item(item_id)
        if not item or item.run_id != run_id:
            raise ResourceNotFoundException("EvaluationRunItem", item_id)
        if item.status not in (
            EvaluationRunItemStatus.FAILED,
            EvaluationRunItemStatus.CANCELLED,
        ):
            raise ValidationException("Only failed or cancelled run items can be retried")

        next_run_status = run.status
        if run.status in (
            EvaluationRunStatus.CANCELLED,
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.FAILED,
        ):
            next_run_status = EvaluationRunStatus.QUEUED

        await self.db_ops.update_run_status(
            run_id,
            next_run_status,
            error_message=None,
            summary=_requeue_run_summary(run.summary, item.status),
        )

        item = await self.db_ops.requeue_run_item(run_id, item_id)
        await self.launch_run(run_id)
        return EvaluationRunItemEnvelope.model_validate(item)

    async def list_run_items(
        self, user_id: str, run_id: str, page: int, page_size: int
    ) -> tuple[list[EvaluationRunItemEnvelope], int]:
        await self._require_run(user_id, run_id)
        rows, total = await self.db_ops.list_run_items(run_id, page, page_size)
        return [EvaluationRunItemEnvelope.model_validate(r) for r in rows], total

    async def list_run_item_attempts(
        self, user_id: str, run_id: str, item_id: str
    ) -> list[EvaluationRunItemAttemptEnvelope]:
        item = await self.db_ops.get_run_item(item_id)
        if not item or item.run_id != run_id:
            raise ResourceNotFoundException("EvaluationRunItem", item_id)
        run = await self.db_ops.get_evaluation_run(user_id, item.run_id)
        if not run:
            raise ResourceNotFoundException("EvaluationRun", item.run_id)
        attempts = await self.db_ops.list_attempts_for_item(item_id)
        return [EvaluationRunItemAttemptEnvelope.model_validate(a) for a in attempts]

    async def _require_run(self, user_id: str, run_id: str) -> EvaluationRun:
        run = await self.db_ops.get_evaluation_run(user_id, run_id)
        if not run:
            raise ResourceNotFoundException("EvaluationRun", run_id)
        return run


evaluation_dataset_service = EvaluationDatasetService()
evaluation_run_service = EvaluationRunService(dataset_service=evaluation_dataset_service)
