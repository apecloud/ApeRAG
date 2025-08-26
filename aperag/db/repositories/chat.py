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

from aperag.db.models import Chat, ChatStatus, MessageFeedback
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncChatRepositoryMixin(AsyncRepositoryProtocol):
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

    async def query_chat_feedbacks(self, user: str, chat_id: str):
        """Query all active feedback for a chat (no soft delete check needed)"""

        async def _query(session):
            stmt = (
                select(MessageFeedback)
                .where(
                    MessageFeedback.chat_id == chat_id,
                    MessageFeedback.user == user,
                )
                .order_by(desc(MessageFeedback.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_message_feedback(self, user: str, chat_id: str, message_id: str):
        """Query specific message feedback (no soft delete check needed)"""

        async def _query(session):
            stmt = select(MessageFeedback).where(
                MessageFeedback.chat_id == chat_id,
                MessageFeedback.message_id == message_id,
                MessageFeedback.user == user,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    # Chat Operations
    async def create_chat(
        self, user: str, bot_id: str, title: str = "New Chat", peer_type=None, peer_id: str = None
    ) -> Chat:
        """Create a new chat in database with optional peer information"""
        from aperag.db.models import ChatPeerType

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

    # Message Feedback Operations
    async def create_message_feedback(
        self,
        user: str,
        chat_id: str,
        message_id: str,
        feedback_type: str,
        feedback_tag: str = None,
        feedback_message: str = None,
        question: str = None,
        original_answer: str = None,
        collection_id: str = None,
    ) -> MessageFeedback:
        """Create message feedback"""

        async def _operation(session):
            from aperag.db.models import MessageFeedbackStatus

            instance = MessageFeedback(
                user=user,
                chat_id=chat_id,
                message_id=message_id,
                type=feedback_type,
                tag=feedback_tag,
                message=feedback_message,
                question=question,
                original_answer=original_answer,
                collection_id=collection_id,
                status=MessageFeedbackStatus.PENDING,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_message_feedback(
        self,
        user: str,
        chat_id: str,
        message_id: str,
        feedback_type: str = None,
        feedback_tag: str = None,
        feedback_message: str = None,
        question: str = None,
        original_answer: str = None,
    ) -> Optional[MessageFeedback]:
        """Update existing message feedback"""

        async def _operation(session):
            stmt = select(MessageFeedback).where(
                MessageFeedback.user == user,
                MessageFeedback.chat_id == chat_id,
                MessageFeedback.message_id == message_id,
                MessageFeedback.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            feedback = result.scalars().first()

            if feedback:
                if feedback_type is not None:
                    feedback.type = feedback_type
                if feedback_tag is not None:
                    feedback.tag = feedback_tag
                if feedback_message is not None:
                    feedback.message = feedback_message
                if question is not None:
                    feedback.question = question
                if original_answer is not None:
                    feedback.original_answer = original_answer

                feedback.gmt_updated = utc_now()
                session.add(feedback)
                await session.flush()
                await session.refresh(feedback)

            return feedback

        return await self.execute_with_transaction(_operation)

    async def remove_message_feedback(self, user: str, chat_id: str, message_id: str) -> bool:
        """Remove message feedback completely (hard delete)

        UX Design Philosophy:
        - "Cancel like" means "no feedback state" - clean slate
        - Users don't need feedback history
        - Next feedback should be fresh, not restoration
        """

        async def _operation(session):
            from sqlalchemy import delete

            # Hard delete - clean removal for better UX
            stmt = delete(MessageFeedback).where(
                MessageFeedback.user == user,
                MessageFeedback.chat_id == chat_id,
                MessageFeedback.message_id == message_id,
            )
            result = await session.execute(stmt)
            await session.flush()

            # Return True if any row was deleted
            return result.rowcount > 0

        return await self.execute_with_transaction(_operation)

    async def set_message_feedback_state(
        self,
        user: str,
        chat_id: str,
        message_id: str,
        feedback_type: str = None,
        feedback_tag: str = None,
        feedback_message: str = None,
        question: str = None,
        original_answer: str = None,
        collection_id: str = None,
    ) -> MessageFeedback:
        """Set message feedback state using PostgreSQL UPSERT for atomic operation

        UX Design Philosophy:
        - Feedback is a STATE, not a history record
        - Users want simple toggle behavior (like/unlike)
        - System should handle all edge cases gracefully
        - No technical errors should reach users
        """

        async def _operation(session):
            from sqlalchemy.dialects.postgresql import insert

            from aperag.db.models import MessageFeedbackStatus

            current_time = utc_now()

            # Prepare feedback state data
            feedback_data = {
                "user": user,
                "chat_id": chat_id,
                "message_id": message_id,
                "type": feedback_type,
                "tag": feedback_tag,
                "message": feedback_message,
                "question": question,
                "original_answer": original_answer,
                "status": MessageFeedbackStatus.PENDING,
                "gmt_created": current_time,
                "gmt_updated": current_time,
                "gmt_deleted": None,  # Always active state
            }

            # PostgreSQL UPSERT - atomic operation
            stmt = insert(MessageFeedback).values(**feedback_data)

            # On primary key conflict, update to new state
            stmt = stmt.on_conflict_do_update(
                index_elements=["chat_id", "message_id"],
                set_={
                    "user": stmt.excluded.user,  # Ensure user consistency
                    "type": stmt.excluded.type,
                    "tag": stmt.excluded.tag,
                    "message": stmt.excluded.message,
                    "question": stmt.excluded.question,
                    "original_answer": stmt.excluded.original_answer,
                    "status": stmt.excluded.status,
                    "gmt_updated": stmt.excluded.gmt_updated,
                    "gmt_deleted": None,  # Reset to active state
                },
            )

            # Return the final state
            stmt = stmt.returning(MessageFeedback)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self.execute_with_transaction(_operation)
