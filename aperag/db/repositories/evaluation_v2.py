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

"""Data-access for the evaluation-v2 product line.

Tables: benchmark_datasets / benchmark_dataset_versions / benchmark_cases /
evaluation_runs / evaluation_run_items / evaluation_run_item_attempts.
"""

from typing import Optional

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDatasetVersion,
    BenchmarkDatasetVersionStatus,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunItemAttempt,
    EvaluationRunStatus,
)
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncEvaluationV2RepositoryMixin(AsyncRepositoryProtocol):
    """Read/write helpers for evaluation-v2 resources."""

    # -- BenchmarkDataset ---------------------------------------------------

    async def create_benchmark_dataset(self, dataset: BenchmarkDataset) -> BenchmarkDataset:
        async def _operation(session: AsyncSession):
            session.add(dataset)
            await session.flush()
            await session.refresh(dataset)
            return dataset

        return await self.execute_with_transaction(_operation)

    async def get_benchmark_dataset(self, user_id: str, dataset_id: str) -> Optional[BenchmarkDataset]:
        async def _query(session: AsyncSession):
            stmt = select(BenchmarkDataset).where(
                BenchmarkDataset.id == dataset_id,
                BenchmarkDataset.user_id == user_id,
                BenchmarkDataset.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def list_benchmark_datasets(
        self,
        user_id: str,
        collection_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[BenchmarkDataset], int]:
        async def _query(session: AsyncSession):
            conditions = [
                BenchmarkDataset.user_id == user_id,
                BenchmarkDataset.gmt_deleted.is_(None),
            ]
            if collection_id:
                conditions.append(BenchmarkDataset.collection_id == collection_id)
            base = select(BenchmarkDataset).where(and_(*conditions))

            total = (
                await session.execute(
                    select(func.count()).select_from(base.subquery())
                )
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(BenchmarkDataset.gmt_created.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).scalars().all()
            return list(rows), total

        return await self._execute_query(_query)

    async def update_benchmark_dataset(
        self,
        user_id: str,
        dataset_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[BenchmarkDataset]:
        async def _operation(session: AsyncSession):
            stmt = select(BenchmarkDataset).where(
                BenchmarkDataset.id == dataset_id,
                BenchmarkDataset.user_id == user_id,
                BenchmarkDataset.gmt_deleted.is_(None),
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

    async def soft_delete_benchmark_dataset(self, user_id: str, dataset_id: str) -> bool:
        async def _operation(session: AsyncSession):
            stmt = select(BenchmarkDataset).where(
                BenchmarkDataset.id == dataset_id,
                BenchmarkDataset.user_id == user_id,
                BenchmarkDataset.gmt_deleted.is_(None),
            )
            instance = (await session.execute(stmt)).scalars().first()
            if not instance:
                return False
            instance.gmt_deleted = utc_now()
            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)

    # -- BenchmarkDatasetVersion -------------------------------------------

    async def create_benchmark_dataset_version(
        self,
        dataset_id: str,
        version: BenchmarkDatasetVersion,
        cases: list[BenchmarkCase],
    ) -> BenchmarkDatasetVersion:
        async def _operation(session: AsyncSession):
            version.dataset_id = dataset_id
            session.add(version)
            await session.flush()
            await session.refresh(version)

            for case in cases:
                case.dataset_version_id = version.id
                session.add(case)
            if cases:
                await session.flush()
                version.case_count = len(cases)
                await session.flush()
                await session.refresh(version)
            return version

        return await self.execute_with_transaction(_operation)

    async def next_dataset_version_number(self, dataset_id: str) -> int:
        async def _query(session: AsyncSession):
            stmt = select(func.coalesce(func.max(BenchmarkDatasetVersion.version), 0)).where(
                BenchmarkDatasetVersion.dataset_id == dataset_id
            )
            return (await session.execute(stmt)).scalar_one() + 1

        return await self._execute_query(_query)

    async def get_dataset_version(self, version_id: str) -> Optional[BenchmarkDatasetVersion]:
        async def _query(session: AsyncSession):
            stmt = select(BenchmarkDatasetVersion).where(BenchmarkDatasetVersion.id == version_id)
            return (await session.execute(stmt)).scalars().first()

        return await self._execute_query(_query)

    async def list_dataset_versions(self, dataset_id: str) -> list[BenchmarkDatasetVersion]:
        async def _query(session: AsyncSession):
            stmt = (
                select(BenchmarkDatasetVersion)
                .where(BenchmarkDatasetVersion.dataset_id == dataset_id)
                .order_by(BenchmarkDatasetVersion.version.desc())
            )
            return list((await session.execute(stmt)).scalars().all())

        return await self._execute_query(_query)

    async def latest_published_version(self, dataset_id: str) -> Optional[BenchmarkDatasetVersion]:
        async def _query(session: AsyncSession):
            stmt = (
                select(BenchmarkDatasetVersion)
                .where(
                    BenchmarkDatasetVersion.dataset_id == dataset_id,
                    BenchmarkDatasetVersion.status == BenchmarkDatasetVersionStatus.PUBLISHED,
                )
                .order_by(BenchmarkDatasetVersion.version.desc())
                .limit(1)
            )
            return (await session.execute(stmt)).scalars().first()

        return await self._execute_query(_query)

    # -- BenchmarkCase ------------------------------------------------------

    async def list_cases_for_version(
        self, version_id: str, page: int, page_size: int
    ) -> tuple[list[BenchmarkCase], int]:
        async def _query(session: AsyncSession):
            base = select(BenchmarkCase).where(BenchmarkCase.dataset_version_id == version_id)
            total = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(BenchmarkCase.sort_key.asc(), BenchmarkCase.gmt_created.asc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).scalars().all()
            return list(rows), total

        return await self._execute_query(_query)

    async def list_all_cases_for_version(self, version_id: str) -> list[BenchmarkCase]:
        async def _query(session: AsyncSession):
            stmt = (
                select(BenchmarkCase)
                .where(BenchmarkCase.dataset_version_id == version_id)
                .order_by(BenchmarkCase.sort_key.asc(), BenchmarkCase.gmt_created.asc())
            )
            return list((await session.execute(stmt)).scalars().all())

        return await self._execute_query(_query)

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
        bot_id: Optional[str],
        page: int,
        page_size: int,
    ) -> tuple[list[EvaluationRun], int]:
        async def _query(session: AsyncSession):
            conditions = [EvaluationRun.user_id == user_id]
            if bot_id:
                conditions.append(EvaluationRun.bot_id == bot_id)
            base = select(EvaluationRun).where(and_(*conditions))
            total = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(EvaluationRun.gmt_created.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).scalars().all()
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
        async def _operation(session: AsyncSession):
            stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
            run = (await session.execute(stmt)).scalars().first()
            if not run:
                return None
            now = utc_now()
            run.status = status
            if error_message is not None:
                run.error_message = error_message
            if summary is not None:
                run.summary = summary
            if status == EvaluationRunStatus.RUNNING and run.gmt_started is None:
                run.gmt_started = now
            if status in (
                EvaluationRunStatus.COMPLETED,
                EvaluationRunStatus.FAILED,
                EvaluationRunStatus.CANCELLED,
            ):
                run.gmt_finished = now
            run.gmt_updated = now
            await session.flush()
            await session.refresh(run)
            return run

        return await self.execute_with_transaction(_operation)

    # -- EvaluationRunItem / Attempt ---------------------------------------

    async def list_run_items(
        self, run_id: str, page: int, page_size: int
    ) -> tuple[list[EvaluationRunItem], int]:
        async def _query(session: AsyncSession):
            base = select(EvaluationRunItem).where(EvaluationRunItem.run_id == run_id)
            total = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(EvaluationRunItem.gmt_created.asc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).scalars().all()
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
