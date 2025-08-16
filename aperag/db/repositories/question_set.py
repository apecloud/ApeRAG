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

from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Question, QuestionSet
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncQuestionSetRepositoryMixin(AsyncRepositoryProtocol):
    async def create_question_set(self, question_set: QuestionSet, questions: list[Question] = None) -> QuestionSet:
        """Creates a new question set and optionally a list of questions in a single transaction."""

        async def _operation(session: AsyncSession):
            session.add(question_set)
            await session.flush()  # Flush to get the ID for the question_set

            if questions:
                for q in questions:
                    q.question_set_id = question_set.id
                session.add_all(questions)
                await session.flush()

            await session.refresh(question_set)
            # To load the newly created questions, we can query for them.
            # However, the service layer currently doesn't expect them to be returned.
            # If that changes, we would need to adjust the relationship loading here.
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
        self, qs_id: str, user_id: str, name: Optional[str], description: Optional[str]
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

            if name:
                db_question_set.name = name
            if description:
                db_question_set.description = description

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

    async def create_questions_in_bulk(self, questions: list[Question]) -> list[Question]:
        """Creates multiple questions in a single transaction."""

        async def _operation(session: AsyncSession):
            session.add_all(questions)
            await session.flush()
            return questions

        return await self.execute_with_transaction(_operation)

    async def update_question(
        self, q_id: str, question_text: Optional[str], ground_truth: Optional[str], question_type: Optional[str]
    ) -> Question | None:
        """Updates a question."""

        async def _operation(session: AsyncSession):
            stmt = select(Question).where(Question.id == q_id, Question.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            db_question = result.scalars().first()

            if not db_question:
                return None

            if question_text:
                db_question.question_text = question_text
            if ground_truth:
                db_question.ground_truth = ground_truth
            if question_type:
                db_question.question_type = question_type

            await session.flush()
            await session.refresh(db_question)
            return db_question

        return await self.execute_with_transaction(_operation)

    async def list_questions_by_set_id(self, qs_id: str, page: int, page_size: int) -> tuple[list[Question], int]:
        """Lists all questions for a question set."""

        async def _query(session: AsyncSession):
            stmt = (
                select(Question)
                .where(Question.question_set_id == qs_id, Question.gmt_deleted.is_(None))
                .offset((page - 1) * page_size)
                .limit(page_size)
                .order_by(Question.gmt_created.asc())
            )
            result = await session.execute(stmt)
            items = result.scalars().all()

            count_stmt = select(func.count(Question.id)).where(
                Question.question_set_id == qs_id, Question.gmt_deleted.is_(None)
            )
            total = await session.scalar(count_stmt)

            return items, total

        return await self._execute_query(_query)

    async def list_all_questions_by_set_id(self, qs_id: str) -> list[Question]:
        """Lists all questions for a question set without pagination."""

        async def _query(session: AsyncSession):
            stmt = (
                select(Question)
                .where(Question.question_set_id == qs_id, Question.gmt_deleted.is_(None))
                .order_by(Question.gmt_created.asc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

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
