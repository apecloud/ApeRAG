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
from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.snapshot_assembler import (
    assemble_parts_from_artifacts,
    extract_error_text,
)
from aperag.domains.agent_runtime.uimessage import AgentTurnSnapshot
from aperag.domains.conversation.db.models import BotType, ChatStatus
from aperag.domains.conversation.db.models import Chat as ChatRow
from aperag.domains.conversation.schemas import Chat, ChatDetails, ChatUpdate
from aperag.exceptions import ChatNotFoundException, ResourceNotFoundException, ValidationException

logger = logging.getLogger(__name__)


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

    async def _build_v3_chat_history(self, user: str, chat_id: str) -> list[AgentTurnSnapshot]:
        """Build the chat history as canonical ``AgentTurnSnapshot`` envelopes.

        Phase 8 D8.5-BE (#92) flips this from the legacy
        ``list[list[ChatMessage]]`` shape to one ``AgentTurnSnapshot``
        per assistant turn. Each snapshot carries the same
        ``UIMessagePart[]`` shape the FE consumes from the live SSE
        stream, so historical and live turns render through a single
        canonical path (D8 §2 wire/at-rest byte-equal).

        The user query is exposed at ``input_text`` on the snapshot
        envelope rather than as a separate ``role=human`` ChatMessage
        entry; the FE renders it from there. The assistant turn's
        ``answer`` / ``reference_bundle`` / ``error_summary``
        artifacts are projected into ``parts`` via
        :func:`assemble_parts_from_artifacts`, mirroring the snapshot
        endpoint (#90 / D8.4d). FAILED / CANCELLED turns surface their
        message via ``error_text`` (preferring an ``error_summary``
        artifact, falling back to ``turn.error_message``), again
        mirroring the snapshot endpoint contract.

        Once the wire emitter starts populating ``agent_message.parts``
        directly (D8.6 / #80), this method can short-circuit to
        :meth:`UIMessageStore.read` per-turn; until then the artifact
        projection is the single source. ``runtime_kind`` is hardcoded
        to ``agent_runtime`` here — non-agent runtimes will write
        ``direct_chat`` / ``rag_chat`` rows directly via the future
        non-agent write path and this method only needs to surface
        them when they exist (a no-op until then per architect lock
        msg=01918929).
        """

        turns = await self.db_ops.query_agent_turns(user, chat_id)

        history: list[AgentTurnSnapshot] = []
        for turn in turns:
            artifacts = await self.db_ops.query_agent_artifacts_by_turn(turn.id)
            parts = list(assemble_parts_from_artifacts(artifacts))

            error_text: Optional[str] = None
            if turn.status in {AgentTurnStatus.FAILED, AgentTurnStatus.CANCELLED}:
                error_text = extract_error_text(artifacts) or turn.error_message

            status_value = turn.status.value if hasattr(turn.status, "value") else str(turn.status)
            history.append(
                AgentTurnSnapshot(
                    turn_id=turn.id,
                    chat_id=turn.chat_id,
                    runtime_kind="agent_runtime",
                    status=status_value,
                    parts=parts,
                    error_text=error_text,
                    timeline_cursor=turn.timeline_cursor or 0,
                    started_at=turn.gmt_started,
                    finished_at=turn.gmt_finished,
                    created_at=turn.gmt_created,
                    updated_at=turn.gmt_updated,
                    input_text=turn.input_text,
                )
            )

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
            return self.build_chat_response(deleted_chat)

        return None


# Global service instance — wire via legacy shim at ``aperag.service.chat_service``.
chat_service_global = ChatService()
