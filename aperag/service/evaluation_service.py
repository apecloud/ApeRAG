# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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

from aperag.db.models import Evaluation
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.schema import view_models


class EvaluationService:
    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    async def create_evaluation(self, request: view_models.EvaluationCreate, user_id: str) -> Evaluation:
        """Creates a new evaluation task."""
        # TODO: Add logic to trigger async task runner
        # TODO: check quota limit
        db_evaluation = Evaluation(
            user_id=user_id,
            name=request.name,
            collection_id=request.collection_id,
            question_set_id=request.question_set_id,
            agent_llm_config=request.agent_llm_config,
            judge_llm_config=request.judge_llm_config,
        )
        return await self.db_ops.create_evaluation(db_evaluation)

    async def get_evaluation(self, eval_id: str, user_id: str) -> Evaluation | None:
        """Gets an evaluation by its ID."""
        return await self.db_ops.get_evaluation_by_id(eval_id, user_id)

    async def list_evaluations(self, user_id: str, page: int, page_size: int) -> tuple[list[Evaluation], int]:
        """Lists all evaluations for a user."""
        return await self.db_ops.list_evaluations_by_user(user_id, page, page_size)

    async def delete_evaluation(self, eval_id: str, user_id: str) -> bool:
        """Deletes an evaluation."""
        # TODO: Add logic to stop the running task if it's in progress
        return await self.db_ops.delete_evaluation_by_id(eval_id, user_id)


evaluation_service = EvaluationService()
