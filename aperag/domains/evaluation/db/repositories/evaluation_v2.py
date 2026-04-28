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

"""Data-access for the simplified evaluation API.

Tables: ``evaluation_datasets`` / ``evaluation_dataset_items`` /
``evaluation_runs`` / ``evaluation_run_items`` / ``evaluation_run_item_attempts``.
The legacy ``benchmark_*`` tables were removed in the Phase 2 destructive
migration; there is no Benchmark / Dataset Version concept here any more.
"""

from typing import Optional

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.domains.conversation.db.models import Bot, BotStatus
from aperag.domains.evaluation.db.models import (
    EvaluationDataset,
    EvaluationDatasetItem,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunItemAttempt,
    EvaluationRunItemAttemptStatus,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)
from aperag.utils.utils import utc_now


class AsyncEvaluationV2RepositoryMixin(AsyncRepositoryProtocol):
    """Read/write helpers for evaluation resources."""

    # -- EvaluationRun ------------------------------------------------------

    async def create_evaluation_run(
        self,
        run: EvaluationRun,
        items: list[EvaluationRunItem],
    ) -> EvaluationRun:
        async def _operation(session: AsyncSession):
            session.add(run)
            await session.flush()
            await session.refresh(run)
            for item in items:
                item.run_id = run.id
                session.add(item)
            if items:
                await session.flush()
            return run

        return await self.execute_with_transaction(_operation)

    async def get_evaluation_run(self, user_id: str, run_id: str) -> Optional[EvaluationRun]:
        async def _query(session: AsyncSession):
            stmt = select(EvaluationRun).where(
                EvaluationRun.id == run_id,
                EvaluationRun.user_id == user_id,
            )
            return (await session.execute(stmt)).scalars().first()

        return await self._execute_query(_query)

    async def list_evaluation_runs(
        self,
        user_id: str,
        bot_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[EvaluationRun], int]:
        async def _query(session: AsyncSession):
            conditions = [EvaluationRun.user_id == user_id]
            if bot_id:
                conditions.append(EvaluationRun.bot_id == bot_id)
            if dataset_id:
                conditions.append(EvaluationRun.dataset_id == dataset_id)
            if collection_id:
                conditions.append(EvaluationRun.collection_id == collection_id)
            base = select(EvaluationRun).where(and_(*conditions))
            total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
            rows = (
                (
                    await session.execute(
                        base.order_by(EvaluationRun.gmt_created.desc()).offset((page - 1) * page_size).limit(page_size)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows), total

        return await self._execute_query(_query)

    async def update_run_status(
        self,
        run_id: str,
        status: EvaluationRunStatus,
        *,
        error_message: Optional[str] = None,
        summary: Optional[dict] = None,
    ) -> Optional[EvaluationRun]:
        """Update a run's status with a terminal → running guard.

        If the persisted run is already in a terminal status
        (COMPLETED / FAILED / CANCELLED) and the requested transition
        is to ``RUNNING``, this is a no-op and returns the existing
        row unchanged — preventing a late worker
        ``update_run_status(RUNNING)`` from overwriting a cancel that
        raced in between ``get_run_for_worker`` and this call. That
        race produced the symptom tracked in task #23 where
        ``gmt_finished`` ended up earlier than ``gmt_started`` and the
        run flipped back to ``running`` right after cancel returned
        ``cancelled``. Explicit terminal → QUEUED transitions (the
        ``service.retry_run_item`` re-queue flow) are still allowed,
        so retries continue to work.
        """

        async def _operation(session: AsyncSession):
            stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
            run = (await session.execute(stmt)).scalars().first()
            if not run:
                return None
            if EvaluationRunStatus.is_terminal(run.status) and status == EvaluationRunStatus.RUNNING:
                return run
            now = utc_now()
            run.status = status
            if error_message is not None:
                run.error_message = error_message
            if summary is not None:
                run.summary = summary
            if status == EvaluationRunStatus.RUNNING and run.gmt_started is None:
                run.gmt_started = now
            if EvaluationRunStatus.is_terminal(status):
                run.gmt_finished = now
            else:
                run.gmt_finished = None
            run.gmt_updated = now
            await session.flush()
            await session.refresh(run)
            return run

        return await self.execute_with_transaction(_operation)

    # -- EvaluationRunItem / Attempt ---------------------------------------

    async def list_run_items(self, run_id: str, page: int, page_size: int) -> tuple[list[EvaluationRunItem], int]:
        async def _query(session: AsyncSession):
            base = select(EvaluationRunItem).where(EvaluationRunItem.run_id == run_id)
            total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
            rows = (
                (
                    await session.execute(
                        base.order_by(EvaluationRunItem.gmt_created.asc())
                        .offset((page - 1) * page_size)
                        .limit(page_size)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows), total

        return await self._execute_query(_query)

    async def list_all_run_items(self, run_id: str) -> list[EvaluationRunItem]:
        async def _query(session: AsyncSession):
            stmt = (
                select(EvaluationRunItem)
                .where(EvaluationRunItem.run_id == run_id)
                .order_by(EvaluationRunItem.gmt_created.asc())
            )
            return list((await session.execute(stmt)).scalars().all())

        return await self._execute_query(_query)

    async def get_run_item(self, item_id: str) -> Optional[EvaluationRunItem]:
        async def _query(session: AsyncSession):
            stmt = select(EvaluationRunItem).where(EvaluationRunItem.id == item_id)
            return (await session.execute(stmt)).scalars().first()

        return await self._execute_query(_query)

    async def list_attempts_for_item(self, run_item_id: str) -> list[EvaluationRunItemAttempt]:
        async def _query(session: AsyncSession):
            stmt = (
                select(EvaluationRunItemAttempt)
                .where(EvaluationRunItemAttempt.run_item_id == run_item_id)
                .order_by(EvaluationRunItemAttempt.attempt_no.asc())
            )
            return list((await session.execute(stmt)).scalars().all())

        return await self._execute_query(_query)

    async def requeue_run_item(self, run_id: str, item_id: str) -> Optional[EvaluationRunItem]:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationRunItem).where(
                EvaluationRunItem.id == item_id,
                EvaluationRunItem.run_id == run_id,
            )
            instance = (await session.execute(stmt)).scalars().first()
            if not instance:
                return None
            instance.status = EvaluationRunItemStatus.PENDING
            instance.error_message = None
            instance.gmt_updated = utc_now()
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    # -- Worker-side helpers (PR-1b) ---------------------------------------
    #
    # These helpers intentionally do not filter by ``user_id``: the Celery
    # worker receives only a ``run_id`` and must read/update run state on
    # behalf of the run's owner. Authorization stays on the HTTP edge.

    async def get_run_for_worker(self, run_id: str) -> Optional[EvaluationRun]:
        """Load a run without scoping to a specific user — worker-side use only."""

        async def _query(session: AsyncSession):
            stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
            return (await session.execute(stmt)).scalars().first()

        return await self._execute_query(_query)

    async def mark_run_item_running(self, item_id: str) -> Optional[EvaluationRunItem]:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationRunItem).where(EvaluationRunItem.id == item_id)
            item = (await session.execute(stmt)).scalars().first()
            if not item:
                return None
            now = utc_now()
            item.status = EvaluationRunItemStatus.RUNNING
            item.attempt_count = (item.attempt_count or 0) + 1
            item.gmt_updated = now
            await session.flush()
            await session.refresh(item)
            return item

        return await self.execute_with_transaction(_operation)

    async def create_run_item_attempt(
        self,
        *,
        run_id: str,
        run_item_id: str,
        attempt_no: int,
        status: EvaluationRunItemAttemptStatus,
        agent_chat_id: Optional[str] = None,
        agent_turn_id: Optional[str] = None,
        answer_text: Optional[str] = None,
        error_message: Optional[str] = None,
        latency_ms: Optional[int] = None,
        score: Optional[float] = None,
        judge_result: Optional[dict] = None,
    ) -> EvaluationRunItemAttempt:
        """Persist the outcome of a single worker turn dispatch.

        ``score`` and ``judge_result`` are populated when LLM-as-judge
        ran (architect spec ``msg=2424afe2``); they stay ``NULL`` for
        Noop / ExactMatch judges so historical rows are unchanged.
        """

        async def _operation(session: AsyncSession):
            now = utc_now()
            finished = (
                now
                if status
                in (
                    EvaluationRunItemAttemptStatus.COMPLETED,
                    EvaluationRunItemAttemptStatus.FAILED,
                    EvaluationRunItemAttemptStatus.CANCELLED,
                )
                else None
            )
            attempt = EvaluationRunItemAttempt(
                run_id=run_id,
                run_item_id=run_item_id,
                attempt_no=attempt_no,
                status=status,
                agent_chat_id=agent_chat_id,
                agent_turn_id=agent_turn_id,
                answer_text=answer_text,
                error_message=error_message,
                latency_ms=latency_ms,
                score=score,
                judge_result=judge_result,
                gmt_started=now,
                gmt_finished=finished,
            )
            session.add(attempt)
            await session.flush()
            await session.refresh(attempt)
            return attempt

        return await self.execute_with_transaction(_operation)

    async def finalize_run_item(
        self,
        *,
        item_id: str,
        status: EvaluationRunItemStatus,
        latest_attempt_id: Optional[str] = None,
        error_message: Optional[str] = None,
        best_score: Optional[float] = None,
        judge_breakdown: Optional[dict] = None,
    ) -> Optional[EvaluationRunItem]:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationRunItem).where(EvaluationRunItem.id == item_id)
            item = (await session.execute(stmt)).scalars().first()
            if not item:
                return None
            item.status = status
            if latest_attempt_id is not None:
                item.latest_attempt_id = latest_attempt_id
            item.error_message = error_message
            if best_score is not None:
                item.best_score = best_score
            if judge_breakdown is not None:
                item.judge_breakdown = judge_breakdown
            item.gmt_updated = utc_now()
            await session.flush()
            await session.refresh(item)
            return item

        return await self.execute_with_transaction(_operation)

    # -- EvaluationDataset (evaluation-v3, Phase 1 additive) -----------------
    #
    # Helpers below operate on the new ``evaluation_datasets`` /
    # ``evaluation_dataset_items`` tables. Phase 1 only adds these helpers; no
    # production code path (benchmark service / views / runtime) calls into
    # them yet. The contract switch happens in Phase 2.

    async def create_evaluation_dataset(
        self,
        dataset: EvaluationDataset,
        items: Optional[list[EvaluationDatasetItem]] = None,
    ) -> EvaluationDataset:
        async def _operation(session: AsyncSession):
            session.add(dataset)
            await session.flush()
            if items:
                for item in items:
                    item.dataset_id = dataset.id
                    session.add(item)
                dataset.item_count = len(items)
                await session.flush()
            await session.refresh(dataset)
            return dataset

        return await self.execute_with_transaction(_operation)

    async def get_evaluation_dataset(self, user_id: str, dataset_id: str) -> Optional[EvaluationDataset]:
        async def _query(session: AsyncSession):
            stmt = select(EvaluationDataset).where(
                EvaluationDataset.id == dataset_id,
                EvaluationDataset.user_id == user_id,
                EvaluationDataset.gmt_deleted.is_(None),
            )
            return (await session.execute(stmt)).scalars().first()

        return await self._execute_query(_query)

    async def list_evaluation_datasets(
        self,
        user_id: str,
        collection_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[EvaluationDataset], int]:
        async def _query(session: AsyncSession):
            conditions = [
                EvaluationDataset.user_id == user_id,
                EvaluationDataset.gmt_deleted.is_(None),
            ]
            if collection_id:
                conditions.append(EvaluationDataset.collection_id == collection_id)
            base = select(EvaluationDataset).where(and_(*conditions))
            total_stmt = select(func.count()).select_from(base.subquery())
            total = (await session.execute(total_stmt)).scalar() or 0
            stmt = (
                base.order_by(EvaluationDataset.gmt_created.desc())
                .offset(max(page - 1, 0) * page_size)
                .limit(page_size)
            )
            rows = list((await session.execute(stmt)).scalars().all())
            return rows, int(total)

        return await self._execute_query(_query)

    async def update_evaluation_dataset(
        self,
        user_id: str,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[EvaluationDataset]:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationDataset).where(
                EvaluationDataset.id == dataset_id,
                EvaluationDataset.user_id == user_id,
                EvaluationDataset.gmt_deleted.is_(None),
            )
            instance = (await session.execute(stmt)).scalars().first()
            if not instance:
                return None
            if name is not None:
                instance.name = name
            if description is not None:
                instance.description = description
            instance.gmt_updated = utc_now()
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def soft_delete_evaluation_dataset(self, user_id: str, dataset_id: str) -> bool:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationDataset).where(
                EvaluationDataset.id == dataset_id,
                EvaluationDataset.user_id == user_id,
                EvaluationDataset.gmt_deleted.is_(None),
            )
            instance = (await session.execute(stmt)).scalars().first()
            if not instance:
                return False
            now = utc_now()
            instance.gmt_deleted = now
            instance.gmt_updated = now
            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)

    async def list_evaluation_dataset_items(
        self, dataset_id: str, page: int, page_size: int
    ) -> tuple[list[EvaluationDatasetItem], int]:
        async def _query(session: AsyncSession):
            conditions = [
                EvaluationDatasetItem.dataset_id == dataset_id,
                EvaluationDatasetItem.gmt_deleted.is_(None),
            ]
            base = select(EvaluationDatasetItem).where(and_(*conditions))
            total_stmt = select(func.count()).select_from(base.subquery())
            total = (await session.execute(total_stmt)).scalar() or 0
            stmt = (
                base.order_by(
                    EvaluationDatasetItem.sort_key.asc(),
                    EvaluationDatasetItem.gmt_created.asc(),
                )
                .offset(max(page - 1, 0) * page_size)
                .limit(page_size)
            )
            rows = list((await session.execute(stmt)).scalars().all())
            return rows, int(total)

        return await self._execute_query(_query)

    async def list_all_evaluation_dataset_items(self, dataset_id: str) -> list[EvaluationDatasetItem]:
        async def _query(session: AsyncSession):
            stmt = (
                select(EvaluationDatasetItem)
                .where(
                    EvaluationDatasetItem.dataset_id == dataset_id,
                    EvaluationDatasetItem.gmt_deleted.is_(None),
                )
                .order_by(
                    EvaluationDatasetItem.sort_key.asc(),
                    EvaluationDatasetItem.gmt_created.asc(),
                )
            )
            return list((await session.execute(stmt)).scalars().all())

        return await self._execute_query(_query)

    async def append_evaluation_dataset_items(
        self, dataset_id: str, items: list[EvaluationDatasetItem]
    ) -> list[EvaluationDatasetItem]:
        async def _operation(session: AsyncSession):
            for item in items:
                item.dataset_id = dataset_id
                session.add(item)
            await session.flush()
            dataset_stmt = select(EvaluationDataset).where(EvaluationDataset.id == dataset_id)
            dataset = (await session.execute(dataset_stmt)).scalars().first()
            if dataset is not None:
                dataset.item_count = (dataset.item_count or 0) + len(items)
                dataset.gmt_updated = utc_now()
            for item in items:
                await session.refresh(item)
            return list(items)

        return await self.execute_with_transaction(_operation)

    async def update_evaluation_dataset_item(
        self,
        dataset_id: str,
        item_id: str,
        **fields,
    ) -> Optional[EvaluationDatasetItem]:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationDatasetItem).where(
                EvaluationDatasetItem.id == item_id,
                EvaluationDatasetItem.dataset_id == dataset_id,
                EvaluationDatasetItem.gmt_deleted.is_(None),
            )
            instance = (await session.execute(stmt)).scalars().first()
            if not instance:
                return None
            allowed = {
                "case_key",
                "input_message",
                "expected_answer",
                "reference_context",
                "tags",
                "case_metadata",
                "sort_key",
            }
            for key, value in fields.items():
                if key in allowed and value is not None:
                    setattr(instance, key, value)
            instance.gmt_updated = utc_now()
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def soft_delete_evaluation_dataset_item(self, dataset_id: str, item_id: str) -> bool:
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationDatasetItem).where(
                EvaluationDatasetItem.id == item_id,
                EvaluationDatasetItem.dataset_id == dataset_id,
                EvaluationDatasetItem.gmt_deleted.is_(None),
            )
            instance = (await session.execute(stmt)).scalars().first()
            if not instance:
                return False
            now = utc_now()
            instance.gmt_deleted = now
            instance.gmt_updated = now
            dataset_stmt = select(EvaluationDataset).where(EvaluationDataset.id == dataset_id)
            dataset = (await session.execute(dataset_stmt)).scalars().first()
            if dataset is not None and (dataset.item_count or 0) > 0:
                dataset.item_count = (dataset.item_count or 0) - 1
                dataset.gmt_updated = now
            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)

    # -- Default-bot resolver (evaluation-v3, Phase 1 additive) --------------

    async def resolve_default_evaluation_bot(self, user_id: str, default_title: str) -> Optional[Bot]:
        """Pick a bot for an evaluation run without an explicit ``bot_id``.

        The resolver is DB-only (no HTTP bot route side-channel) and prefers
        the user's active bot whose ``title`` equals ``default_title`` (the
        caller passes ``DEFAULT_AGENT_BOT_TITLE``). Ties break on
        ``gmt_created ASC`` so the result is deterministic. If no such bot
        exists the resolver falls back to the oldest active bot for the user.
        Only active, non-soft-deleted bots are considered.
        """

        async def _query(session: AsyncSession):
            active_filter = and_(
                Bot.user == user_id,
                Bot.status == BotStatus.ACTIVE,
                Bot.gmt_deleted.is_(None),
            )
            preferred_stmt = (
                select(Bot)
                .where(and_(active_filter, Bot.title == default_title))
                .order_by(Bot.gmt_created.asc())
                .limit(1)
            )
            preferred = (await session.execute(preferred_stmt)).scalars().first()
            if preferred is not None:
                return preferred

            fallback_stmt = select(Bot).where(active_filter).order_by(Bot.gmt_created.asc()).limit(1)
            return (await session.execute(fallback_stmt)).scalars().first()

        return await self._execute_query(_query)
