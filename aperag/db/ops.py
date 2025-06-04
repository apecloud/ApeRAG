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
from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from aperag.config import get_session
from aperag.db.models import (
    ApiKey,
    ApiKeyStatus,
    Bot,
    BotStatus,
    Chat,
    Collection,
    CollectionStatus,
    ConfigModel,
    Document,
    Invitation,
    MessageFeedback,
    ModelServiceProvider,
    ModelServiceProviderStatus,
    Role,
    SearchTestHistory,
    User,
    UserQuota,
)

logger = logging.getLogger(__name__)


class DatabaseOps:
    """Database operations manager that handles session management"""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def get_session(self) -> AsyncSession:
        """Get database session, create new one if not provided"""
        if self._session:
            return self._session

        # For global instance usage, use the dependency injection pattern
        # This will be managed by the caller (usually in service layer _execute_with_session)
        async for session in get_session():
            return session

    async def _execute_query(self, query_func):
        """Execute a read-only query with proper session management"""
        if self._session:
            # Use provided session
            return await query_func(self._session)
        else:
            # Create new session and manage its lifecycle
            async for session in get_session():
                try:
                    return await query_func(session)
                finally:
                    # Session is automatically closed by the context manager
                    pass

    # Collection Operations
    async def create_collection(
        self, user: str, title: str, description: str, collection_type, config: str = None
    ) -> Collection:
        """Create a new collection in database"""
        session = await self.get_session()
        instance = Collection(
            user=user,
            type=collection_type,
            status=CollectionStatus.INACTIVE,
            title=title,
            description=description,
            config=config,
        )
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def update_collection_by_id(
        self, user: str, collection_id: str, title: str, description: str, config: str
    ) -> Optional[Collection]:
        """Update collection by ID"""
        session = await self.get_session()
        stmt = select(Collection).where(
            Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
        )
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            instance.title = title
            instance.description = description
            instance.config = config
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    async def delete_collection_by_id(self, user: str, collection_id: str) -> Optional[Collection]:
        """Soft delete collection by ID"""
        session = await self.get_session()
        stmt = select(Collection).where(
            Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
        )
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            # Check if collection has related bots
            collection_bots = await instance.bots(session, only_ids=True)
            if len(collection_bots) > 0:
                raise ValueError(f"Collection has related to bots {','.join(collection_bots)}, can not be deleted")

            instance.status = CollectionStatus.DELETED
            instance.gmt_deleted = datetime.utcnow()
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    # Search Test Operations
    async def create_search_test(
        self,
        user: str,
        collection_id: str,
        query: str,
        vector_search: dict = None,
        fulltext_search: dict = None,
        graph_search: dict = None,
        items: List[dict] = None,
    ) -> SearchTestHistory:
        """Create a search test record"""
        session = await self.get_session()
        record = SearchTestHistory(
            user=user,
            query=query,
            collection_id=collection_id,
            vector_search=vector_search,
            fulltext_search=fulltext_search,
            graph_search=graph_search,
            items=items or [],
        )
        session.add(record)
        await session.flush()
        await session.refresh(record)
        return record

    async def query_search_tests(self, user: str, collection_id: str) -> List[SearchTestHistory]:
        """Query search tests for a collection"""
        session = await self.get_session()
        stmt = (
            select(SearchTestHistory)
            .where(SearchTestHistory.user == user, SearchTestHistory.collection_id == collection_id)
            .order_by(desc(SearchTestHistory.gmt_created))
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    async def delete_search_test(self, user: str, collection_id: str, search_test_id: str) -> bool:
        """Delete a search test record"""
        session = await self.get_session()
        stmt = select(SearchTestHistory).where(
            SearchTestHistory.id == search_test_id,
            SearchTestHistory.user == user,
            SearchTestHistory.collection_id == collection_id,
        )
        result = await session.execute(stmt)
        search_test = result.scalars().first()

        if search_test:
            await session.delete(search_test)
            return True
        return False

    async def query_collection(self, user: str, collection_id: str):
        async def _query(session):
            stmt = select(Collection).where(
                Collection.id == collection_id, Collection.user == user, Collection.status != CollectionStatus.DELETED
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_collections(self, users: List[str]):
        async def _query(session):
            stmt = (
                select(Collection)
                .where(Collection.user.in_(users), Collection.status != CollectionStatus.DELETED)
                .order_by(desc(Collection.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_collections_count(self, user: str):
        async def _query(session):
            stmt = (
                select(func.count())
                .select_from(Collection)
                .where(Collection.user == user, Collection.status != CollectionStatus.DELETED)
            )
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    async def query_collection_without_user(self, collection_id: str):
        async def _query(session):
            stmt = select(Collection).where(
                Collection.id == collection_id, Collection.status != CollectionStatus.DELETED
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_document(self, user: str, collection_id: str, document_id: str):
        async def _query(session):
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != CollectionStatus.DELETED,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_documents(self, users: List[str], collection_id: str):
        async def _query(session):
            stmt = (
                select(Document)
                .where(
                    Document.user.in_(users),
                    Document.collection_id == collection_id,
                    Document.status != CollectionStatus.DELETED,
                )
                .order_by(desc(Document.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_documents_count(self, user: str, collection_id: str):
        async def _query(session):
            stmt = (
                select(func.count())
                .select_from(Document)
                .where(
                    Document.user == user,
                    Document.collection_id == collection_id,
                    Document.status != CollectionStatus.DELETED,
                )
            )
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    async def query_chat(self, user: str, bot_id: str, chat_id: str):
        async def _query(session):
            stmt = select(Chat).where(
                Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != CollectionStatus.DELETED
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
                Chat.status != CollectionStatus.DELETED,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_chats(self, user: str, bot_id: str):
        async def _query(session):
            stmt = (
                select(Chat)
                .where(Chat.user == user, Chat.bot_id == bot_id, Chat.status != CollectionStatus.DELETED)
                .order_by(desc(Chat.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_bot(self, user: str, bot_id: str):
        async def _query(session):
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_bots(self, users: List[str]):
        async def _query(session):
            stmt = (
                select(Bot).where(Bot.user.in_(users), Bot.status != BotStatus.DELETED).order_by(desc(Bot.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_bots_count(self, user: str):
        async def _query(session):
            stmt = select(func.count()).select_from(Bot).where(Bot.user == user, Bot.status != BotStatus.DELETED)
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    async def query_config(self, key):
        async def _query(session):
            stmt = select(ConfigModel).where(ConfigModel.key == key)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_quota(self, user: str, key: str):
        async def _query(session):
            stmt = select(UserQuota).where(UserQuota.user == user, UserQuota.key == key)
            result = await session.execute(stmt)
            uq = result.scalars().first()
            return uq.value if uq else None

        return await self._execute_query(_query)

    async def query_msp_list(self, user: str):
        async def _query(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.user == user, ModelServiceProvider.status != ModelServiceProviderStatus.DELETED
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_msp_dict(self, user: str):
        async def _query(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.user == user, ModelServiceProvider.status != ModelServiceProviderStatus.DELETED
            )
            result = await session.execute(stmt)
            return {msp.name: msp for msp in result.scalars().all()}

        return await self._execute_query(_query)

    async def query_msp(self, user: str, provider: str, filterDeletion: bool = True):
        async def _query(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.user == user, ModelServiceProvider.name == provider
            )
            if filterDeletion:
                stmt = stmt.where(ModelServiceProvider.status != ModelServiceProviderStatus.DELETED)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_by_username(self, username: str):
        async def _query(session):
            stmt = select(User).where(User.username == username)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_by_email(self, email: str):
        async def _query(session):
            stmt = select(User).where(User.email == email)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_exists(self, username: str = None, email: str = None):
        async def _query(session):
            stmt = select(User)
            if username:
                stmt = stmt.where(User.username == username)
            if email:
                stmt = stmt.where(User.email == email)
            result = await session.execute(stmt)
            return result.scalars().first() is not None

        return await self._execute_query(_query)

    async def create_user(self, username: str, email: str, password: str, role: Role):
        session = await self.get_session()
        user = User(username=username, email=email, password=password, role=role, is_staff=(role == Role.ADMIN))
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    async def set_user_password(self, user: User, password: str):
        session = await self.get_session()
        user.password = password
        session.add(user)
        await session.commit()

    async def delete_user(self, user: User):
        session = await self.get_session()
        await session.delete(user)
        await session.commit()

    async def query_invitation_by_token(self, token: str):
        async def _query(session):
            stmt = select(Invitation).where(Invitation.token == token)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_invitation(self, email: str, token: str, created_by: str, role: Role):
        session = await self.get_session()
        invitation = Invitation(email=email, token=token, created_by=created_by, role=role)
        session.add(invitation)
        await session.commit()
        await session.refresh(invitation)
        return invitation

    async def mark_invitation_used(self, invitation: Invitation):
        session = await self.get_session()
        await invitation.use(session)

    async def query_invitations(self):
        """Query all valid invitations (not used), ordered by created_at descending."""

        async def _query(session):
            stmt = select(Invitation).where(not Invitation.is_used).order_by(desc(Invitation.created_at))
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def list_user_api_keys(self, user: str):
        """List all active API keys for a user"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def create_api_key(self, user: str, description: Optional[str] = None) -> ApiKey:
        """Create a new API key for a user"""
        session = await self.get_session()
        api_key = ApiKey(user=user, description=description, status=ApiKeyStatus.ACTIVE)
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)
        return api_key

    async def delete_api_key(self, user: str, key_id: str) -> bool:
        """Delete an API key (soft delete)"""
        session = await self.get_session()
        stmt = select(ApiKey).where(
            ApiKey.id == key_id, ApiKey.user == user, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        api_key = result.scalars().first()
        if api_key:
            api_key.status = ApiKeyStatus.DELETED
            from datetime import datetime as dt

            api_key.gmt_deleted = dt.utcnow()
            session.add(api_key)
            await session.commit()
            return True
        return False

    async def get_api_key_by_id(self, user: str, id: str) -> Optional[ApiKey]:
        """Get API key by id string"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user, ApiKey.id == id, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_api_key_by_key(self, key: str) -> Optional[ApiKey]:
        """Get API key by key string"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.key == key, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

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

    async def query_first_user_exists(self):
        async def _query(session):
            stmt = select(User).where(User.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().first() is not None

        return await self._execute_query(_query)

    async def query_admin_count(self):
        async def _query(session):
            stmt = select(func.count()).select_from(User).where(User.role == Role.ADMIN, User.gmt_deleted.is_(None))
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    # Bot Operations
    async def create_bot(self, user: str, title: str, description: str, bot_type, config: str = "{}") -> Bot:
        """Create a new bot in database"""
        session = await self.get_session()
        instance = Bot(
            user=user,
            title=title,
            type=bot_type,
            status=BotStatus.ACTIVE,
            description=description,
            config=config,
        )
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def update_bot_by_id(
        self, user: str, bot_id: str, title: str, description: str, bot_type, config: str
    ) -> Optional[Bot]:
        """Update bot by ID"""
        session = await self.get_session()
        stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            instance.title = title
            instance.description = description
            instance.type = bot_type
            instance.config = config
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    async def delete_bot_by_id(self, user: str, bot_id: str) -> Optional[Bot]:
        """Soft delete bot by ID"""
        session = await self.get_session()
        stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            instance.status = BotStatus.DELETED
            instance.gmt_deleted = datetime.utcnow()
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    async def create_bot_collection_relation(self, bot_id: str, collection_id: str):
        """Create bot-collection relation"""
        from aperag.db.models import BotCollectionRelation

        session = await self.get_session()
        relation = BotCollectionRelation(bot_id=bot_id, collection_id=collection_id)
        session.add(relation)
        await session.flush()
        return relation

    async def delete_bot_collection_relations(self, bot_id: str):
        """Soft delete all bot-collection relations for a bot"""
        from aperag.db.models import BotCollectionRelation

        session = await self.get_session()
        stmt = select(BotCollectionRelation).where(
            BotCollectionRelation.bot_id == bot_id, BotCollectionRelation.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        relations = result.scalars().all()
        for rel in relations:
            rel.gmt_deleted = datetime.utcnow()
            session.add(rel)
        await session.flush()
        return len(relations)

    # Document Operations
    async def create_document(
        self, user: str, collection_id: str, name: str, size: int, object_path: str = None, metadata: str = None
    ) -> Document:
        """Create a new document in database"""
        session = await self.get_session()
        instance = Document(
            user=user,
            name=name,
            status=CollectionStatus.PENDING,  # Note: using CollectionStatus for DocumentStatus
            size=size,
            collection_id=collection_id,
            object_path=object_path,
            metadata=metadata,
        )
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def update_document_by_id(
        self, user: str, collection_id: str, document_id: str, metadata: str = None
    ) -> Optional[Document]:
        """Update document by ID"""
        session = await self.get_session()
        stmt = select(Document).where(
            Document.id == document_id,
            Document.collection_id == collection_id,
            Document.user == user,
            Document.status != CollectionStatus.DELETED,
        )
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance and metadata is not None:
            instance.metadata = metadata
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    async def delete_document_by_id(self, user: str, collection_id: str, document_id: str) -> Optional[Document]:
        """Soft delete document by ID"""
        session = await self.get_session()
        stmt = select(Document).where(
            Document.id == document_id,
            Document.collection_id == collection_id,
            Document.user == user,
            Document.status != CollectionStatus.DELETED,
        )
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            from aperag.db.models import DocumentStatus

            instance.status = DocumentStatus.DELETING
            instance.gmt_deleted = datetime.utcnow()
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    async def delete_documents_by_ids(self, user: str, collection_id: str, document_ids: List[str]) -> tuple:
        """Bulk soft delete documents by IDs"""
        session = await self.get_session()
        stmt = select(Document).where(
            Document.id.in_(document_ids),
            Document.collection_id == collection_id,
            Document.user == user,
            Document.status != CollectionStatus.DELETED,
        )
        result = await session.execute(stmt)
        documents = result.scalars().all()

        success_ids = []
        for doc in documents:
            try:
                from aperag.db.models import DocumentStatus

                doc.status = DocumentStatus.DELETING
                doc.gmt_deleted = datetime.utcnow()
                session.add(doc)
                success_ids.append(doc.id)
            except Exception:
                continue

        await session.flush()
        failed_ids = [doc_id for doc_id in document_ids if doc_id not in success_ids]
        return success_ids, failed_ids

    # Chat Operations
    async def create_chat(self, user: str, bot_id: str, title: str = "New Chat") -> Chat:
        """Create a new chat in database"""
        session = await self.get_session()
        instance = Chat(
            user=user,
            bot_id=bot_id,
            title=title,
            status=CollectionStatus.ACTIVE,  # Note: using CollectionStatus for ChatStatus
        )
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def update_chat_by_id(self, user: str, bot_id: str, chat_id: str, title: str) -> Optional[Chat]:
        """Update chat by ID"""
        session = await self.get_session()
        stmt = select(Chat).where(
            Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != CollectionStatus.DELETED
        )
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            instance.title = title
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

    async def delete_chat_by_id(self, user: str, bot_id: str, chat_id: str) -> Optional[Chat]:
        """Soft delete chat by ID"""
        session = await self.get_session()
        stmt = select(Chat).where(
            Chat.id == chat_id, Chat.bot_id == bot_id, Chat.user == user, Chat.status != CollectionStatus.DELETED
        )
        result = await session.execute(stmt)
        instance = result.scalars().first()

        if instance:
            instance.status = CollectionStatus.DELETED
            instance.gmt_deleted = datetime.utcnow()
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

        return instance

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
        session = await self.get_session()
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
        session = await self.get_session()
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

    async def delete_message_feedback(self, user: str, chat_id: str, message_id: str) -> bool:
        """Delete message feedback (soft delete)"""
        session = await self.get_session()
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
        session = await self.get_session()

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

    async def update_api_key_by_id(self, user: str, key_id: str, description: str) -> Optional[ApiKey]:
        """Update API key description"""
        session = await self.get_session()
        stmt = select(ApiKey).where(
            ApiKey.user == user, ApiKey.id == key_id, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        api_key = result.scalars().first()

        if api_key:
            api_key.description = description
            session.add(api_key)
            await session.flush()
            await session.refresh(api_key)

        return api_key


# Create a global instance for backwards compatibility and easy access
# This can be used in places where session dependency injection is not available
db_ops = DatabaseOps()


# Keep the original function names for backwards compatibility during transition
async def query_collection(session, user: str, collection_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_collection(user, collection_id)


async def query_collections(session, users: List[str]):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_collections(users)


async def query_collections_count(session, user: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_collections_count(user)


async def query_collection_without_user(session, collection_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_collection_without_user(collection_id)


async def query_document(session, user: str, collection_id: str, document_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_document(user, collection_id, document_id)


async def query_documents(session, users: List[str], collection_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_documents(users, collection_id)


async def query_documents_count(session, user: str, collection_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_documents_count(user, collection_id)


async def query_chat(session, user: str, bot_id: str, chat_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_chat(user, bot_id, chat_id)


async def query_chat_by_peer(session, user: str, peer_type, peer_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_chat_by_peer(user, peer_type, peer_id)


async def query_chats(session, user: str, bot_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_chats(user, bot_id)


async def query_bot(session, user: str, bot_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_bot(user, bot_id)


async def query_bots(session, users: List[str]):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_bots(users)


async def query_bots_count(session, user: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_bots_count(user)


async def query_config(session, key):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_config(key)


async def query_user_quota(session, user: str, key: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_user_quota(user, key)


async def query_msp_list(session, user: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_msp_list(user)


async def query_msp_dict(session, user: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_msp_dict(user)


async def query_msp(session, user: str, provider: str, filterDeletion: bool = True):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_msp(user, provider, filterDeletion)


async def query_user_by_username(session, username: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_user_by_username(username)


async def query_user_by_email(session, email: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_user_by_email(email)


async def query_user_exists(session, username: str = None, email: str = None):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_user_exists(username, email)


async def create_user(session, username: str, email: str, password: str, role: Role):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).create_user(username, email, password, role)


async def set_user_password(session, user: User, password: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).set_user_password(user, password)


async def delete_user(session, user: User):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).delete_user(user)


async def query_invitation_by_token(session, token: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_invitation_by_token(token)


async def create_invitation(session, email: str, token: str, created_by: str, role: Role):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).create_invitation(email, token, created_by, role)


async def mark_invitation_used(session, invitation: Invitation):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).mark_invitation_used(invitation)


async def query_invitations(session):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_invitations()


async def list_user_api_keys(session, user: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).list_user_api_keys(user)


async def create_api_key(session, user: str, description: Optional[str] = None) -> ApiKey:
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).create_api_key(user, description)


async def delete_api_key(session, user: str, key_id: str) -> bool:
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).delete_api_key(user, key_id)


async def get_api_key_by_id(session, user: str, id: str) -> Optional[ApiKey]:
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).get_api_key_by_id(user, id)


async def get_api_key_by_key(session, key: str) -> Optional[ApiKey]:
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).get_api_key_by_key(key)


async def query_chat_feedbacks(session, user: str, chat_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_chat_feedbacks(user, chat_id)


async def query_message_feedback(session, user: str, chat_id: str, message_id: str):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_message_feedback(user, chat_id, message_id)


async def query_first_user_exists(session):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_first_user_exists()


async def query_admin_count(session):
    """Deprecated: Use DatabaseOps instance instead"""
    return await DatabaseOps(session).query_admin_count()
