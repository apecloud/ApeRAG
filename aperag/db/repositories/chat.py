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

from datetime import datetime
from typing import Optional

from sqlalchemy import desc, select

from aperag.db.models import Chat, ChatStatus, MessageFeedback
from aperag.db.repositories.base import AsyncRepositoryProtocol


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
        async def _query(session):
            stmt = (
                select(MessageFeedback)
                .where(
                    MessageFeedback.chat_id == chat_id,
                    MessageFeedback.gmt_deleted.is_(None),
                    MessageFeedback.user == user,
                )
                .order_by(desc(MessageFeedback.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_message_feedback(self, user: str, chat_id: str, message_id: str):
        async def _query(session):
            stmt = select(MessageFeedback).where(
                MessageFeedback.chat_id == chat_id,
                MessageFeedback.message_id == message_id,
                MessageFeedback.gmt_deleted.is_(None),
                MessageFeedback.user == user,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    # Chat Operations
    async def create_chat(self, user: str, bot_id: str, title: str = "New Chat") -> Chat:
        """Create a new chat in database"""

        async def _operation(session):
            instance = Chat(
                user=user,
                bot_id=bot_id,
                title=title,
                status=ChatStatus.ACTIVE,
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
                instance.gmt_deleted = datetime.utcnow()
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

                feedback.gmt_updated = datetime.utcnow()
                session.add(feedback)
                await session.flush()
                await session.refresh(feedback)

            return feedback

        return await self.execute_with_transaction(_operation)

    async def delete_message_feedback(self, user: str, chat_id: str, message_id: str) -> bool:
        """Delete message feedback (soft delete)"""

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
                feedback.gmt_deleted = datetime.utcnow()
                session.add(feedback)
                await session.flush()
                return True
            return False

        return await self.execute_with_transaction(_operation)

    async def upsert_message_feedback(
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
        """Create or update message feedback (upsert operation)"""

        async def _operation(session):
            # Try to find existing feedback
            stmt = select(MessageFeedback).where(
                MessageFeedback.user == user,
                MessageFeedback.chat_id == chat_id,
                MessageFeedback.message_id == message_id,
                MessageFeedback.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            feedback = result.scalars().first()

            if feedback:
                # Update existing
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
                feedback.gmt_updated = datetime.utcnow()
            else:
                # Create new
                from aperag.db.models import MessageFeedbackStatus

                feedback = MessageFeedback(
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

            session.add(feedback)
            await session.flush()
            await session.refresh(feedback)
            return feedback

        return await self.execute_with_transaction(_operation)
