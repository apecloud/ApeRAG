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

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException, ValidationException
from aperag.schema import view_models


def _coerce_turn_feedback(feedback) -> view_models.TurnFeedback:
    return view_models.TurnFeedback(
        turn_id=feedback.turn_id,
        type=feedback.type,
        tag=feedback.tag,
        message=feedback.message,
        created=feedback.gmt_created,
        updated=feedback.gmt_updated,
    )


class TurnFeedbackService:
    def __init__(self, session: AsyncSession = None):
        self.db_ops = async_db_ops if session is None else AsyncDatabaseOps(session)

    async def list_turn_feedbacks(self, user: str, chat_id: str) -> view_models.TurnFeedbackList:
        chat = await self.db_ops.query_chat_by_id(user, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        feedbacks = await self.db_ops.query_turn_feedbacks(user, chat_id)
        return view_models.TurnFeedbackList(items=[_coerce_turn_feedback(feedback) for feedback in feedbacks])

    async def upsert_turn_feedback(
        self, user: str, chat_id: str, turn_id: str, feedback_in: view_models.Feedback
    ) -> view_models.TurnFeedback:
        if not feedback_in.type:
            raise ValidationException("Feedback type is required")

        turn = await self.db_ops.query_agent_turn(user, chat_id, turn_id)
        if turn is None:
            raise ResourceNotFoundException("Turn", turn_id)

        feedback = await self.db_ops.set_turn_feedback_state(
            user=user,
            chat_id=chat_id,
            turn_id=turn.id,
            feedback_type=feedback_in.type,
            feedback_tag=feedback_in.tag,
            feedback_message=feedback_in.message,
        )
        return _coerce_turn_feedback(feedback)

    async def delete_turn_feedback(self, user: str, chat_id: str, turn_id: str) -> bool:
        turn = await self.db_ops.query_agent_turn(user, chat_id, turn_id)
        if turn is None:
            raise ResourceNotFoundException("Turn", turn_id)

        return await self.db_ops.remove_turn_feedback(user, chat_id, turn.id)


turn_feedback_service_global = TurnFeedbackService()
