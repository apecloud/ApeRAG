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

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Question, QuestionSet
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.schema import view_models
from aperag.utils.utils import utc_now


class AsyncQuestionSetRepositoryMixin(AsyncRepositoryProtocol):
    async def create_question_set(self, question_set: QuestionSet) -> QuestionSet:
        """Creates a new question set."""

        async def _operation(session: AsyncSession):
            session.add(question_set)
            await session.flush()
            await session.refresh(question_set)
            return question_set

        return await self.execute_with_transaction(_operation)

    async def get_question_set_by_id(self, qs_id: str, user_id: str) -> QuestionSet | None:
        """Gets a question set by its ID."""

        async def _query(session: AsyncSession):
            stmt = select(QuestionSet).where(
                QuestionSet.id == qs_id,
                QuestionSet.user_id == user_id,
                QuestionSet.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def list_question_sets_by_user(
        self, user_id: str, page: int, page_size: int
    ) -> tuple[list[QuestionSet], int]:
        """Lists all question sets for a user."""

        async def _query(session: AsyncSession):
            stmt = (
                select(QuestionSet)
                .where(QuestionSet.user_id == user_id, QuestionSet.gmt_deleted.is_(None))
                .offset((page - 1) * page_size)
                .limit(page_size)
                .order_by(QuestionSet.gmt_created.desc())
            )
            result = await session.execute(stmt)
            items = result.scalars().all()

            count_stmt = select(func.count(QuestionSet.id)).where(
                QuestionSet.user_id == user_id, QuestionSet.gmt_deleted.is_(None)
            )
            total = await session.scalar(count_stmt)

            return items, total

        return await self._execute_query(_query)

    async def update_question_set(
        self, qs_id: str, request: view_models.QuestionSetUpdate, user_id: str
    ) -> QuestionSet | None:
        """Updates a question set."""

        async def _operation(session: AsyncSession):
            stmt = select(QuestionSet).where(
                QuestionSet.id == qs_id,
                QuestionSet.user_id == user_id,
                QuestionSet.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            db_question_set = result.scalars().first()

            if not db_question_set:
                return None

            update_data = request.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_question_set, key, value)

            await session.flush()
            await session.refresh(db_question_set)
            return db_question_set

        return await self.execute_with_transaction(_operation)

    async def delete_question_set_by_id(self, qs_id: str, user_id: str) -> bool:
        """Hard deletes a question set and its associated questions by its ID."""

        async def _operation(session: AsyncSession):
            # First, find the question set to ensure it exists and belongs to the user
            stmt = select(QuestionSet).where(
                QuestionSet.id == qs_id,
                QuestionSet.user_id == user_id,
                QuestionSet.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            db_question_set = result.scalars().first()

            if not db_question_set:
                return False

            # Delete all questions associated with the question set
            await session.execute(delete(Question).where(Question.question_set_id == qs_id))

            # Delete the question set itself
            await session.delete(db_question_set)

            return True

        return await self.execute_with_transaction(_operation)

    async def create_question(self, question: Question) -> Question:
        """Creates a new question."""

        async def _operation(session: AsyncSession):
            session.add(question)
            await session.flush()
            await session.refresh(question)
            return question

        return await self.execute_with_transaction(_operation)

    async def update_question(self, q_id: str, request: view_models.QuestionUpdate) -> Question | None:
        """Updates a question."""

        async def _operation(session: AsyncSession):
            stmt = select(Question).where(Question.id == q_id, Question.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            db_question = result.scalars().first()

            if not db_question:
                return None

            update_data = request.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_question, key, value)

            await session.flush()
            await session.refresh(db_question)
            return db_question

        return await self.execute_with_transaction(_operation)

    async def delete_question_by_id(self, q_id: str) -> bool:
        """Hard deletes a question by its ID."""

        async def _operation(session: AsyncSession):
            stmt = select(Question).where(Question.id == q_id, Question.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            db_question = result.scalars().first()

            if not db_question:
                return False

            await session.delete(db_question)
            return True

        return await self.execute_with_transaction(_operation)
