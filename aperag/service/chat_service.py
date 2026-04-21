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

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException, ValidationException
from aperag.schema import view_models
from aperag.schema.view_models import Chat, ChatDetails
from aperag.utils.history import (
    RedisChatMessageHistory,
    get_async_redis_client,
)

logger = logging.getLogger(__name__)


def _artifact_type_value(artifact) -> Optional[str]:
    if not artifact:
        return None
    artifact_type = getattr(artifact, "artifact_type", None)
    return artifact_type.value if hasattr(artifact_type, "value") else artifact_type


def _extract_artifact_text(artifact) -> str:
    if not artifact or not isinstance(getattr(artifact, "payload", None), dict):
        return ""
    return artifact.payload.get("text") or artifact.payload.get("content") or artifact.payload.get("message") or ""


def _coerce_feedback(feedback) -> Optional[view_models.Feedback]:
    if not feedback:
        return None
    return view_models.Feedback(type=feedback.type, tag=feedback.tag, message=feedback.message)


def _coerce_timestamp(value) -> Optional[float]:
    return value.timestamp() if value else None


def _map_reference_item(item: dict) -> view_models.Reference:
    return view_models.Reference(
        score=item.get("score"),
        text=item.get("snippet") or "",
        metadata={
            **(item.get("metadata") or {}),
            "title": item.get("title"),
            "source_type": item.get("source_type"),
            "source_id": item.get("source_id"),
            "uri": item.get("uri"),
        },
    )


def _extract_references(artifact) -> list[view_models.Reference]:
    if not artifact or not isinstance(getattr(artifact, "payload", None), dict):
        return []
    items = artifact.payload.get("items")
    if not isinstance(items, list):
        return []
    return [_map_reference_item(item) for item in items if isinstance(item, dict)]


class ChatService:
    """Chat service that handles business logic for chats"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def build_chat_response(self, chat: db_models.Chat) -> view_models.Chat:
        """Build Chat response object for API return."""
        return Chat(
            id=chat.id,
            title=chat.title,
            bot_id=chat.bot_id,
            peer_type=chat.peer_type,
            peer_id=chat.peer_id,
            created=chat.gmt_created.isoformat(),
            updated=chat.gmt_updated.isoformat(),
        )

    async def _build_v3_chat_history(self, user: str, chat_id: str) -> list[list[view_models.ChatMessage]]:
        turns = await self.db_ops.query_agent_turns(user, chat_id)
        feedback_map = {
            feedback.message_id: _coerce_feedback(feedback)
            for feedback in await self.db_ops.query_chat_feedbacks(user, chat_id)
        }

        history: list[list[view_models.ChatMessage]] = []
        for turn in turns:
            history.append(
                [
                    view_models.ChatMessage(
                        id=turn.id,
                        type="message",
                        role="human",
                        data=turn.input_text,
                        timestamp=_coerce_timestamp(turn.gmt_created),
                    )
                ]
            )

            artifacts = await self.db_ops.query_agent_artifacts_by_turn(turn.id)
            answer_artifact = (
                next((artifact for artifact in artifacts if artifact.id == turn.answer_artifact_id), None)
                if turn.answer_artifact_id
                else None
            )
            if not answer_artifact:
                answer_artifact = next(
                    (
                        artifact
                        for artifact in artifacts
                        if _artifact_type_value(artifact) == db_models.AgentArtifactType.ANSWER.value
                    ),
                    None,
                )

            reference_artifact = (
                next((artifact for artifact in artifacts if artifact.id == turn.reference_bundle_artifact_id), None)
                if turn.reference_bundle_artifact_id
                else None
            )
            if not reference_artifact:
                reference_artifact = next(
                    (
                        artifact
                        for artifact in artifacts
                        if _artifact_type_value(artifact) == db_models.AgentArtifactType.REFERENCE_BUNDLE.value
                    ),
                    None,
                )

            answer_text = _extract_artifact_text(answer_artifact)
            if not answer_text and turn.status in {
                db_models.AgentTurnStatus.FAILED,
                db_models.AgentTurnStatus.CANCELLED,
            }:
                answer_text = turn.error_message or ""

            ai_parts: list[view_models.ChatMessage] = []
            if answer_text:
                ai_parts.append(
                    view_models.ChatMessage(
                        id=turn.id,
                        type="message",
                        role="ai",
                        data=answer_text,
                        timestamp=_coerce_timestamp(turn.gmt_finished) or _coerce_timestamp(turn.gmt_created),
                    )
                )
            else:
                ai_parts.append(
                    view_models.ChatMessage(
                        id=turn.id,
                        type="start",
                        role="ai",
                        data="",
                        timestamp=_coerce_timestamp(turn.gmt_created),
                    )
                )

            references = _extract_references(reference_artifact)
            feedback = feedback_map.get(turn.id)
            if references or feedback:
                ai_parts.append(
                    view_models.ChatMessage(
                        id=turn.id,
                        type="references",
                        role="ai",
                        data="",
                        references=references,
                        feedback=feedback,
                        timestamp=_coerce_timestamp(turn.gmt_finished) or _coerce_timestamp(turn.gmt_created),
                    )
                )

            history.append(ai_parts)

        return history

    async def create_chat(self, user: str, bot_id: str) -> view_models.Chat:
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)
        if bot.type != db_models.BotType.AGENT:
            raise ValidationException("Only agent bots are supported")

        # Direct call to repository method, which handles its own transaction
        chat = await self.db_ops.create_chat(user=user, bot_id=bot_id)

        return self.build_chat_response(chat)

    async def list_chats(
        self,
        user: str,
        bot_id: str,
        page: int = 1,
        page_size: int = 50,
    ):
        """List chats with pagination, sorting and search capabilities."""

        # Define sort field mapping
        sort_mapping = {
            "created": db_models.Chat.gmt_created,
        }

        # Define search fields mapping
        search_fields = {"title": db_models.Chat.title}

        async def _execute_paginated_query(session):
            from sqlalchemy import and_, desc, select

            # Build base query
            query = select(db_models.Chat).where(
                and_(
                    db_models.Chat.user == user,
                    db_models.Chat.bot_id == bot_id,
                    db_models.Chat.status != db_models.ChatStatus.DELETED,
                )
            )

            # Build query parameters
            from aperag.utils.pagination import ListParams, PaginationHelper, PaginationParams, SortParams

            params = ListParams(
                pagination=PaginationParams(page=page, page_size=page_size),
                sort=SortParams(sort_by="created", sort_order="desc"),
            )

            # Use pagination helper
            items, total = await PaginationHelper.paginate_query(
                query=query,
                session=session,
                params=params,
                sort_mapping=sort_mapping,
                search_fields=search_fields,
                default_sort=desc(db_models.Chat.gmt_created),
            )

            # Build chat responses
            chat_responses = []
            for chat in items:
                chat_responses.append(self.build_chat_response(chat))

            return PaginationHelper.build_response(items=chat_responses, total=total, page=page, page_size=page_size)

        return await self.db_ops._execute_query(_execute_paginated_query)

    async def get_chat(self, user: str, bot_id: str, chat_id: str) -> view_models.ChatDetails:
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        messages = await self._build_v3_chat_history(user, chat_id)

        # Build response object
        chat_obj = self.build_chat_response(chat)
        return ChatDetails(**chat_obj.model_dump(), history=messages)

    async def update_chat(
        self, user: str, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate
    ) -> view_models.Chat:
        # First check if chat exists
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        # Direct call to repository method, which handles its own transaction
        updated_chat = await self.db_ops.update_chat_by_id(user, bot_id, chat_id, chat_in.title)

        if not updated_chat:
            raise ChatNotFoundException(chat_id)

        return self.build_chat_response(updated_chat)

    async def delete_chat(self, user: str, bot_id: str, chat_id: str) -> Optional[view_models.Chat]:
        """Delete chat by ID (idempotent operation)

        Returns the deleted chat or None if already deleted/not found
        """
        # Check if chat exists - if not, silently succeed (idempotent)
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            return None

        # Direct call to repository method, which handles its own transaction
        deleted_chat = await self.db_ops.delete_chat_by_id(user, bot_id, chat_id)

        if deleted_chat:
            # Clear chat history from Redis
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
            await history.clear()

            return self.build_chat_response(deleted_chat)

        return None

    async def feedback_message(
        self,
        user: str,
        chat_id: str,
        message_id: str,
        feedback_type: str = None,
        feedback_tag: str = None,
        feedback_message: str = None,
    ) -> dict:
        """Handle message feedback for chat messages"""
        turn = await self.db_ops.query_agent_turn(user, chat_id, message_id)
        if not turn:
            raise ResourceNotFoundException("AI Message", message_id)

        artifacts = await self.db_ops.query_agent_artifacts_by_turn(turn.id)
        answer_artifact = (
            next((artifact for artifact in artifacts if artifact.id == turn.answer_artifact_id), None)
            if turn.answer_artifact_id
            else None
        )
        if not answer_artifact:
            answer_artifact = next(
                (
                    artifact
                    for artifact in artifacts
                    if _artifact_type_value(artifact) == db_models.AgentArtifactType.ANSWER.value
                ),
                None,
            )

        answer_text = _extract_artifact_text(answer_artifact)
        if not answer_text and turn.status in {
            db_models.AgentTurnStatus.FAILED,
            db_models.AgentTurnStatus.CANCELLED,
        }:
            answer_text = turn.error_message or ""

        if not answer_text:
            raise ResourceNotFoundException("AI Message", message_id)

        # Handle feedback state change based on UX design principles
        if feedback_type is None:
            # User wants to remove feedback (cancel like/dislike)
            success_removed = await self.db_ops.remove_message_feedback(user, chat_id, message_id)
            result = {"action": "deleted", "success": success_removed}
        else:
            # User wants to set feedback state (like/dislike)
            feedback = await self.db_ops.set_message_feedback_state(
                user=user,
                chat_id=chat_id,
                message_id=message_id,
                feedback_type=feedback_type,
                feedback_tag=feedback_tag,
                feedback_message=feedback_message,
                question=turn.input_text,
                original_answer=answer_text,
            )
            result = {"action": "upserted", "feedback": feedback}
        return result


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
chat_service_global = ChatService()
