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

from sqlalchemy import desc, select

from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.domains.conversation.db.models import (
    Chat,
    ChatStatus,
    TurnFeedback,
)
from aperag.utils.utils import utc_now


class AsyncChatRepositoryMixin(AsyncRepositoryProtocol):
    async def query_chat_by_id(self, user: str, chat_id: str):
        async def _query(session):
            stmt = select(Chat).where(Chat.id == chat_id, Chat.user == user, Chat.status != ChatStatus.DELETED)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_chat(self, user: str, bot_id: str, chat_id: str):
        async def _query(session):
            stmt = select(Chat).where(
                Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != ChatStatus.DELETED
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_chat_by_peer(self, user: str, peer_type, peer_id: str):
        async def _query(session):
            stmt = select(Chat).where(
                Chat.user == user,
                Chat.peer_type == peer_type,
                Chat.peer_id == peer_id,
                Chat.status != ChatStatus.DELETED,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_chats(self, user: str, bot_id: str):
        async def _query(session):
            stmt = (
                select(Chat)
                .where(Chat.user == user, Chat.bot_id == bot_id, Chat.status != ChatStatus.DELETED)
                .order_by(desc(Chat.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_turn_feedbacks(self, user: str, chat_id: str):
        """Query all feedback records for a chat."""

        async def _query(session):
            stmt = (
                select(TurnFeedback)
                .where(
                    TurnFeedback.chat_id == chat_id,
                    TurnFeedback.user == user,
                )
                .order_by(desc(TurnFeedback.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_turn_feedback(self, user: str, chat_id: str, turn_id: str):
        """Query the feedback record for a single turn."""

        async def _query(session):
            stmt = select(TurnFeedback).where(
                TurnFeedback.chat_id == chat_id,
                TurnFeedback.turn_id == turn_id,
                TurnFeedback.user == user,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    # Chat Operations
    async def create_chat(
        self, user: str, bot_id: str, title: str = "New Chat", peer_type=None, peer_id: str = None
    ) -> Chat:
        """Create a new chat in database with optional peer information"""
        from aperag.domains.conversation.db.models import ChatPeerType

        async def _operation(session):
            instance = Chat(
                user=user,
                bot_id=bot_id,
                title=title,
                status=ChatStatus.ACTIVE,
                peer_type=peer_type or ChatPeerType.SYSTEM,
                peer_id=peer_id,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_chat_by_id(self, user: str, bot_id: str, chat_id: str, title: str) -> Optional[Chat]:
        """Update chat by ID"""

        async def _operation(session):
            stmt = select(Chat).where(
                Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != ChatStatus.DELETED
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.title = title
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_chat_by_id(self, user: str, bot_id: str, chat_id: str) -> Optional[Chat]:
        """Soft delete chat by ID"""

        async def _operation(session):
            stmt = select(Chat).where(
                Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != ChatStatus.DELETED
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.status = ChatStatus.DELETED
                instance.gmt_deleted = utc_now()
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def remove_turn_feedback(self, user: str, chat_id: str, turn_id: str) -> bool:
        """Hard-delete feedback for a turn."""

        async def _operation(session):
            from sqlalchemy import delete

            stmt = delete(TurnFeedback).where(
                TurnFeedback.user == user,
                TurnFeedback.chat_id == chat_id,
                TurnFeedback.turn_id == turn_id,
            )
            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount > 0

        return await self.execute_with_transaction(_operation)

    async def set_turn_feedback_state(
        self,
        user: str,
        chat_id: str,
        turn_id: str,
        feedback_type: str,
        feedback_tag: str = None,
        feedback_message: str = None,
    ) -> TurnFeedback:
        """Upsert feedback state for a turn."""

        async def _operation(session):
            from sqlalchemy.dialects.postgresql import insert

            current_time = utc_now()
            feedback_data = {
                "user": user,
                "chat_id": chat_id,
                "turn_id": turn_id,
                "type": feedback_type,
                "tag": feedback_tag,
                "message": feedback_message,
                "gmt_created": current_time,
                "gmt_updated": current_time,
            }

            stmt = insert(TurnFeedback).values(**feedback_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["chat_id", "turn_id"],
                set_={
                    "user": stmt.excluded.user,
                    "type": stmt.excluded.type,
                    "tag": stmt.excluded.tag,
                    "message": stmt.excluded.message,
                    "gmt_updated": stmt.excluded.gmt_updated,
                },
            )
            stmt = stmt.returning(TurnFeedback)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self.execute_with_transaction(_operation)
