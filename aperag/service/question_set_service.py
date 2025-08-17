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

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Question, QuestionSet
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.schema import view_models


class QuestionSetService:
    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    async def create_question_set(self, request: view_models.QuestionSetCreate, user_id: str) -> QuestionSet:
        """Creates a new question set."""
        db_question_set = QuestionSet(
            user_id=user_id,
            name=request.name,
            description=request.description,
            collection_id=request.collection_id,
        )

        questions_to_create = []
        if request.questions:
            questions_to_create = [
                Question(
                    question_type=q.question_type,
                    question_text=q.question_text,
                    ground_truth=q.ground_truth,
                )
                for q in request.questions
            ]

        return await self.db_ops.create_question_set(db_question_set, questions_to_create)

    async def get_question_set(self, qs_id: str, user_id: str) -> QuestionSet | None:
        """Gets a question set by its ID."""
        return await self.db_ops.get_question_set_by_id(qs_id, user_id)

    async def list_question_sets(self, user_id: str, page: int, page_size: int) -> tuple[list[QuestionSet], int]:
        """Lists all question sets for a user."""
        return await self.db_ops.list_question_sets_by_user(user_id, page, page_size)

    async def update_question_set(
        self, qs_id: str, request: view_models.QuestionSetUpdate, user_id: str
    ) -> QuestionSet | None:
        """Updates a question set."""
        return await self.db_ops.update_question_set(qs_id, user_id, request.name, request.description)

    async def delete_question_set(self, qs_id: str, user_id: str) -> bool:
        """Deletes a question set."""
        return await self.db_ops.delete_question_set_by_id(qs_id, user_id)

    async def add_question(self, qs_id: str, request: view_models.Question) -> Question | None:
        """Adds a question to a question set."""
        db_question = Question(
            question_set_id=qs_id,
            question_type=request.question_type,
            question_text=request.question_text,
            ground_truth=request.ground_truth,
        )
        return await self.db_ops.create_question(db_question)

    async def update_question(self, q_id: str, request: view_models.QuestionUpdate) -> Question | None:
        """Updates a question."""
        return await self.db_ops.update_question(
            q_id, request.question_text, request.ground_truth, request.question_type
        )

    async def delete_question(self, q_id: str) -> bool:
        """Deletes a question."""
        return await self.db_ops.delete_question_by_id(q_id)

    async def list_questions_by_set_id(self, qs_id: str, page: int, page_size: int) -> tuple[list[Question], int]:
        """Lists all questions for a question set."""
        return await self.db_ops.list_questions_by_set_id(qs_id, page, page_size)

    async def list_all_questions(self, qs_id: str) -> list[Question]:
        """Lists all questions for a question set."""
        return await self.db_ops.list_all_questions_by_set_id(qs_id)


question_set_service = QuestionSetService()
