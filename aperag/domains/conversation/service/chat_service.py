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

"""Chat service moved to the ``conversation`` domain in Phase 5 step 5-S4b.

The legacy ``aperag/service/chat_service.py`` now re-exports
``ChatService`` and ``chat_service_global`` from here. All DB / schema
imports are rewritten to their Phase 5 domain paths so the module no
longer touches the three G1-banned legacy aggregate modules
(``aperag.db.models`` / ``aperag.schema.view_models`` /
``aperag.service.*``):

* ``Chat`` / ``ChatStatus`` → ``aperag.domains.conversation.db.models``
* ``BotType`` → ``aperag.domains.conversation.db.models``
* ``AgentArtifactType`` / ``AgentTurnStatus`` →
  ``aperag.domains.agent_runtime.db.models``
* Pydantic ``Chat`` / ``ChatDetails`` / ``ChatMessage`` / ``Reference``
  → ``aperag.domains.conversation.schemas``

Chat service intentionally has no dependency on ``bot_service`` /
``chat_collection_service`` / other conversation siblings — it goes
straight to the ``db_ops`` layer — so moving it into the domain does
not force any other service to move first.
"""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.agent_runtime.db.models import AgentArtifactType, AgentTurnStatus
from aperag.domains.conversation.db.models import BotType, ChatStatus
from aperag.domains.conversation.db.models import Chat as ChatRow
from aperag.domains.conversation.schemas import Chat, ChatDetails, ChatMessage, ChatUpdate, Reference
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException, ValidationException
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


def _coerce_timestamp(value) -> Optional[float]:
    return value.timestamp() if value else None


def _map_reference_item(item: dict) -> Reference:
    return Reference(
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


def _extract_references(artifact) -> list[Reference]:
    if not artifact or not isinstance(getattr(artifact, "payload", None), dict):
        return []
    items = artifact.payload.get("items")
    if not isinstance(items, list):
        return []
    return [_map_reference_item(item) for item in items if isinstance(item, dict)]


class ChatService:
    """Chat service that handles business logic for chats"""

    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    def build_chat_response(self, chat: ChatRow) -> Chat:
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

    async def _build_v3_chat_history(self, user: str, chat_id: str) -> list[list[ChatMessage]]:
        turns = await self.db_ops.query_agent_turns(user, chat_id)

        history: list[list[ChatMessage]] = []
        for turn in turns:
            history.append(
                [
                    ChatMessage(
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
                        if _artifact_type_value(artifact) == AgentArtifactType.ANSWER.value
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
                        if _artifact_type_value(artifact) == AgentArtifactType.REFERENCE_BUNDLE.value
                    ),
                    None,
                )

            answer_text = _extract_artifact_text(answer_artifact)
            if not answer_text and turn.status in {
                AgentTurnStatus.FAILED,
                AgentTurnStatus.CANCELLED,
            }:
                answer_text = turn.error_message or ""

            ai_parts: list[ChatMessage] = []
            if answer_text:
                ai_parts.append(
                    ChatMessage(
                        id=turn.id,
                        type="message",
                        role="ai",
                        data=answer_text,
                        timestamp=_coerce_timestamp(turn.gmt_finished) or _coerce_timestamp(turn.gmt_created),
                    )
                )
            else:
                ai_parts.append(
                    ChatMessage(
                        id=turn.id,
                        type="start",
                        role="ai",
                        data="",
                        timestamp=_coerce_timestamp(turn.gmt_created),
                    )
                )

            references = _extract_references(reference_artifact)
            if references:
                ai_parts.append(
                    ChatMessage(
                        id=turn.id,
                        type="references",
                        role="ai",
                        data="",
                        references=references,
                        timestamp=_coerce_timestamp(turn.gmt_finished) or _coerce_timestamp(turn.gmt_created),
                    )
                )

            history.append(ai_parts)

        return history

    async def create_chat(self, user: str, bot_id: str) -> Chat:
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)
        if bot.type != BotType.AGENT:
            raise ValidationException("Only agent bots are supported")

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

        sort_mapping = {
            "created": ChatRow.gmt_created,
        }

        search_fields = {"title": ChatRow.title}

        async def _execute_paginated_query(session):
            from sqlalchemy import and_, desc, select

            query = select(ChatRow).where(
                and_(
                    ChatRow.user == user,
                    ChatRow.bot_id == bot_id,
                    ChatRow.status != ChatStatus.DELETED,
                )
            )

            from aperag.utils.pagination import ListParams, PaginationHelper, PaginationParams, SortParams

            params = ListParams(
                pagination=PaginationParams(page=page, page_size=page_size),
                sort=SortParams(sort_by="created", sort_order="desc"),
            )

            items, total = await PaginationHelper.paginate_query(
                query=query,
                session=session,
                params=params,
                sort_mapping=sort_mapping,
                search_fields=search_fields,
                default_sort=desc(ChatRow.gmt_created),
            )

            chat_responses = []
            for chat in items:
                chat_responses.append(self.build_chat_response(chat))

            return PaginationHelper.build_response(items=chat_responses, total=total, page=page, page_size=page_size)

        return await self.db_ops._execute_query(_execute_paginated_query)

    async def get_chat(self, user: str, bot_id: str, chat_id: str) -> ChatDetails:
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        messages = await self._build_v3_chat_history(user, chat_id)

        chat_obj = self.build_chat_response(chat)
        return ChatDetails(**chat_obj.model_dump(), history=messages)

    async def update_chat(self, user: str, bot_id: str, chat_id: str, chat_in: ChatUpdate) -> Chat:
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            raise ChatNotFoundException(chat_id)

        updated_chat = await self.db_ops.update_chat_by_id(user, bot_id, chat_id, chat_in.title)

        if not updated_chat:
            raise ChatNotFoundException(chat_id)

        return self.build_chat_response(updated_chat)

    async def delete_chat(self, user: str, bot_id: str, chat_id: str) -> Optional[Chat]:
        """Delete chat by ID (idempotent operation)

        Returns the deleted chat or None if already deleted/not found
        """
        chat = await self.db_ops.query_chat(user, bot_id, chat_id)
        if chat is None:
            return None

        deleted_chat = await self.db_ops.delete_chat_by_id(user, bot_id, chat_id)

        if deleted_chat:
            history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
            await history.clear()

            return self.build_chat_response(deleted_chat)

        return None


# Global service instance — wire via legacy shim at ``aperag.service.chat_service``.
chat_service_global = ChatService()
