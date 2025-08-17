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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Evaluation, EvaluationItem
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncEvaluationRepositoryMixin(AsyncRepositoryProtocol):
    async def create_evaluation(self, evaluation: Evaluation) -> Evaluation:
        """Creates a new evaluation."""

        async def _operation(session: AsyncSession):
            session.add(evaluation)
            await session.flush()
            await session.refresh(evaluation)
            return evaluation

        return await self.execute_with_transaction(_operation)

    async def get_evaluation_items_by_eval_id(self, eval_id: str) -> list[EvaluationItem]:
        """Gets all evaluation items for a given evaluation."""

        async def _query(session: AsyncSession):
            stmt = select(EvaluationItem).where(
                EvaluationItem.evaluation_id == eval_id
            ).order_by(EvaluationItem.gmt_created.asc())
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_evaluation_by_id(self, eval_id: str, user_id: str) -> Evaluation | None:
        """Gets an evaluation by its ID."""

        async def _query(session: AsyncSession):
            stmt = select(Evaluation).where(
                Evaluation.id == eval_id,
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def list_evaluations_by_user(self, user_id: str, page: int, page_size: int) -> tuple[list[Evaluation], int]:
        """Lists all evaluations for a user."""

        async def _query(session: AsyncSession):
            stmt = (
                select(Evaluation)
                .where(Evaluation.user_id == user_id, Evaluation.gmt_deleted.is_(None))
                .offset((page - 1) * page_size)
                .limit(page_size)
                .order_by(Evaluation.gmt_created.desc())
            )
            result = await session.execute(stmt)
            items = result.scalars().all()

            count_stmt = select(func.count(Evaluation.id)).where(
                Evaluation.user_id == user_id, Evaluation.gmt_deleted.is_(None)
            )
            total = await session.scalar(count_stmt)

            return items, total

        return await self._execute_query(_query)

    async def delete_evaluation_by_id(self, eval_id: str, user_id: str) -> bool:
        """Deletes an evaluation by its ID."""

        async def _operation(session: AsyncSession):
            stmt = select(Evaluation).where(
                Evaluation.id == eval_id,
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            db_evaluation = result.scalars().first()

            if not db_evaluation:
                return False

            db_evaluation.gmt_deleted = utc_now()
            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)
