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

from typing import List

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Evaluation, EvaluationItem, EvaluationItemStatus, EvaluationStatus, Question
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncEvaluationRepositoryMixin(AsyncRepositoryProtocol):
    async def create_evaluation(self, evaluation: Evaluation, questions: List[Question]) -> Evaluation:
        """Creates a new evaluation."""

        async def _operation(session: AsyncSession):
            # Insert evaluation into db and retrieve the id
            session.add(evaluation)
            await session.flush()
            await session.refresh(evaluation)

            for question in questions:
                eval_item = EvaluationItem(
                    evaluation_id=evaluation.id,
                    question_id=question.id,
                    question_text=question.question_text,
                    ground_truth=question.ground_truth,
                    status=EvaluationItemStatus.PENDING,
                )
                session.add(eval_item)

            await session.flush()
            return evaluation

        return await self.execute_with_transaction(_operation)

    async def retry_evaluation(self, eval_id: str, user_id: str, scope: str) -> Evaluation | None:
        """Retries items in an evaluation based on the scope."""

        async def _operation(session: AsyncSession):
            # 1. Get the evaluation and lock it for update
            stmt = (
                select(Evaluation)
                .where(Evaluation.id == eval_id, Evaluation.user_id == user_id, Evaluation.gmt_deleted.is_(None))
                .with_for_update()
            )
            evaluation = (await session.execute(stmt)).scalar_one_or_none()

            if not evaluation:
                return None

            # 2. Reset status of items based on scope
            if scope == "all":
                items_stmt = (
                    update(EvaluationItem)
                    .where(EvaluationItem.evaluation_id == eval_id)
                    .values(
                        status=EvaluationItemStatus.PENDING,
                        rag_answer=None,
                        rag_answer_details=None,
                        llm_judge_score=0,
                        llm_judge_reasoning="",
                    )
                )
            else:  # scope == "failed"
                items_stmt = (
                    update(EvaluationItem)
                    .where(
                        EvaluationItem.evaluation_id == eval_id, EvaluationItem.status == EvaluationItemStatus.FAILED
                    )
                    .values(
                        status=EvaluationItemStatus.PENDING,
                        rag_answer=None,
                        rag_answer_details=None,
                        llm_judge_score=0,
                        llm_judge_reasoning="",
                    )
                )
            result = await session.execute(items_stmt)

            # 3. Reset evaluation status to RUNNING to allow re-processing.
            evaluation.status = EvaluationStatus.RUNNING
            if scope == "all":
                evaluation.completed_questions = 0
                evaluation.average_score = 0
            else:
                evaluation.completed_questions = evaluation.completed_questions - result.rowcount
            await session.flush()
            await session.refresh(evaluation)

            return evaluation

        return await self.execute_with_transaction(_operation)

    async def update_evaluation_status(
        self, eval_id: str, user_id: str, new_status: EvaluationStatus, current_statuses: List[EvaluationStatus] = None
    ) -> Evaluation | None:
        """Updates the status of an evaluation, optionally checking its current status."""

        async def _operation(session: AsyncSession):
            # 1. Fetch the evaluation to ensure it exists and belongs to the user.
            stmt = select(Evaluation).where(
                Evaluation.id == eval_id,
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            )
            if current_statuses:
                stmt = stmt.where(Evaluation.status.in_(current_statuses))

            evaluation = (await session.execute(stmt)).scalar_one_or_none()

            if not evaluation:
                return None

            # 2. Update the status.
            evaluation.status = new_status
            await session.flush()
            await session.refresh(evaluation)
            return evaluation

        return await self.execute_with_transaction(_operation)

    async def get_evaluation_items_by_eval_id(self, eval_id: str) -> list[EvaluationItem]:
        """Gets all evaluation items for a given evaluation."""

        async def _query(session: AsyncSession):
            stmt = (
                select(EvaluationItem)
                .where(EvaluationItem.evaluation_id == eval_id)
                .order_by(EvaluationItem.gmt_created.asc())
            )
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

    async def list_evaluations_by_user(
        self, user_id: str, collection_id: str | None, page: int, page_size: int
    ) -> tuple[list[Evaluation], int]:
        """Lists all evaluations for a user."""

        async def _query(session: AsyncSession):
            where_conditions = [
                Evaluation.user_id == user_id,
                Evaluation.gmt_deleted.is_(None),
            ]
            if collection_id:
                where_conditions.append(Evaluation.collection_id == collection_id)

            stmt = (
                select(Evaluation)
                .where(*where_conditions)
                .offset((page - 1) * page_size)
                .limit(page_size)
                .order_by(Evaluation.gmt_created.desc())
            )
            result = await session.execute(stmt)
            items = result.scalars().all()

            count_stmt = select(func.count(Evaluation.id)).where(*where_conditions)
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
