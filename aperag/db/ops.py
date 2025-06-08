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
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import select

from aperag.config import async_engine, get_async_session, get_sync_session, sync_engine
from aperag.db.models import (
    ApiKey,
    ApiKeyStatus,
    Bot,
    BotStatus,
    Chat,
    ChatStatus,
    Collection,
    CollectionStatus,
    ConfigModel,
    Document,
    DocumentStatus,
    Invitation,
    LLMProvider,
    LLMProviderModel,
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
    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def _get_session(self):
        if self._session:
            return self._session
        else:
            sync_session = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)
            with sync_session() as session:
                return session

    def _execute_query(self, query_func):
        if self._session:
            return query_func(self._session)
        else:
            sync_session = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)
            with sync_session() as session:
                return query_func(session)

    def _execute_transaction(self, operation):
        if self._session:
            # Use provided session, caller manages transaction
            return operation(self._session)
        else:
            # Create new session and manage transaction lifecycle
            for session in get_sync_session():
                try:
                    result = operation(session)
                    session.commit()
                    return result
                except Exception:
                    session.rollback()
                    raise

    def query_document_by_id(self, document_id: str) -> Document:
        def _query(session):
            return session.get(Document, document_id)

        return self._execute_query(_query)

    def query_collection_by_id(self, collection_id: str) -> Collection:
        def _query(session):
            return session.get(Collection, collection_id)

        return self._execute_query(_query)

    def update_document(self, document: Document):
        session = self._get_session()
        session.add(document)
        session.commit()
        session.refresh(document)
        return document

    def update_collection(self, collection: Collection):
        session = self._get_session()
        session.add(collection)
        session.commit()
        session.refresh(collection)
        return collection

    def query_msp_dict(self, user: str):
        def _query(session):
            stmt = select(ModelServiceProvider).where(ModelServiceProvider.user == user)
            result = session.execute(stmt)
            return {msp.name: msp for msp in result.scalars().all()}

        return self._execute_query(_query)

    def query_llm_provider_by_name(self, name: str):
        def _query(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name)
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)


class AsyncDatabaseOps:
    """Database operations manager that handles session management"""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get database session, create new one if not provided

        This method is primarily used for write operations (create, update, delete)
        where you need full control over transaction boundaries and explicit session management.

        Usage pattern for write operations:
        1. Call this method to get a session
        2. Perform database operations (add, delete, modify)
        3. Manually call session.flush(), session.commit(), session.refresh() as needed
        4. Handle transaction rollback if errors occur

        The caller is responsible for session lifecycle management when using this method.
        """
        if self._session:
            return self._session

        # This should not be used directly for global instance
        # Instead, use execute_with_transaction for proper session management
        raise RuntimeError(
            "Cannot create session without explicit session management. Use execute_with_transaction instead."
        )

    async def _execute_query(self, query_func):
        """Execute a read-only query with proper session management

        This method is designed for read-only database operations (SELECT queries)
        and provides automatic session lifecycle management. It follows the pattern
        of accepting a query function that encapsulates the database operation.

        Key benefits:
        1. Automatic session creation and cleanup for read operations
        2. Consistent session management across all query methods
        3. Support for both injected sessions and auto-created sessions
        4. Simplified code for read-only operations

        Usage pattern for read operations:
        1. Define an inner async function that takes a session parameter
        2. Write your SELECT query logic inside the inner function
        3. Pass the inner function to this method
        4. Session lifecycle is handled automatically

        Example:
            async def query_user(self, user_id: str):
                async def _query(session):
                    stmt = select(User).where(User.id == user_id)
                    result = await session.execute(stmt)
                    return result.scalars().first()
                return await self._execute_query(_query)
        """
        if self._session:
            return await query_func(self._session)
        else:
            async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session() as session:
                return await query_func(session)

    async def execute_with_transaction(self, operation):
        """Execute multiple database operations in a single transaction

        This method is used when you need to perform multiple database operations
        that must all succeed or all fail together. Individual DatabaseOps methods
        will automatically detect that they're running within a managed transaction
        and will only flush (not commit) their changes.

        Design philosophy:
        - Single operations: Use DatabaseOps methods directly, they handle their own transactions
        - Multiple operations: Use this method to wrap them in a single transaction

        Usage pattern:
        1. Define an operation function that takes a session parameter
        2. Create DatabaseOps instance with the session
        3. Perform multiple database operations within the function
        4. All operations will be executed in a single transaction
        5. Automatic commit on success, rollback on error
        """
        if self._session:
            # Use provided session, caller manages transaction
            return await operation(self._session)
        else:
            # Create new session and manage transaction lifecycle
            async for session in get_async_session():
                try:
                    result = await operation(session)
                    await session.commit()
                    return result
                except Exception:
                    await session.rollback()
                    raise

    # Collection Operations
    async def create_collection(
        self, user: str, title: str, description: str, collection_type, config: str = None
    ) -> Collection:
        """Create a new collection in database"""

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    async def update_collection_by_id(
        self, user: str, collection_id: str, title: str, description: str, config: str
    ) -> Optional[Collection]:
        """Update collection by ID"""

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    async def delete_collection_by_id(self, user: str, collection_id: str) -> Optional[Collection]:
        """Soft delete collection by ID"""

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

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

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    async def query_search_tests(self, user: str, collection_id: str) -> List[SearchTestHistory]:
        """Query search tests for a collection"""

        async def _query(session):
            stmt = (
                select(SearchTestHistory)
                .where(SearchTestHistory.user == user, SearchTestHistory.collection_id == collection_id)
                .order_by(desc(SearchTestHistory.gmt_created))
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def delete_search_test(self, user: str, collection_id: str, search_test_id: str) -> bool:
        """Delete a search test record"""

        async def _operation(session):
            stmt = select(SearchTestHistory).where(
                SearchTestHistory.id == search_test_id,
                SearchTestHistory.user == user,
                SearchTestHistory.collection_id == collection_id,
            )
            result = await session.execute(stmt)
            search_test = result.scalars().first()

            if search_test:
                await session.delete(search_test)
                await session.flush()
                return True
            return False

        return await self.execute_with_transaction(_operation)

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
                Document.status != DocumentStatus.DELETED,
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
                    Document.status != DocumentStatus.DELETED,
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
                    Document.status != DocumentStatus.DELETED,
                )
            )
            return await session.scalar(stmt)

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

    async def update_msp(self, msp: ModelServiceProvider):
        async def _operation(session):
            session.add(msp)
            await session.flush()
            await session.refresh(msp)
            return msp

        return await self.execute_with_transaction(_operation)

    async def delete_msp(self, msp: ModelServiceProvider):
        async def _operation(session):
            msp.status = ModelServiceProviderStatus.DELETED
            msp.gmt_deleted = datetime.utcnow()
            session.add(msp)
            await session.flush()
            return msp

        return await self.execute_with_transaction(_operation)

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
        async def _operation(session):
            user = User(username=username, email=email, password=password, role=role)
            session.add(user)
            await session.flush()
            await session.refresh(user)
            return user

        return await self.execute_with_transaction(_operation)

    async def delete_user(self, user: User):
        async def _operation(session):
            await session.delete(user)
            await session.flush()

        return await self.execute_with_transaction(_operation)

    async def query_invitation_by_token(self, token: str):
        async def _query(session):
            stmt = select(Invitation).where(Invitation.token == token)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_invitation(self, email: str, token: str, created_by: str, role: Role):
        async def _operation(session):
            invitation = Invitation(email=email, token=token, created_by=created_by, role=role)
            session.add(invitation)
            await session.flush()
            await session.refresh(invitation)
            return invitation

        return await self.execute_with_transaction(_operation)

    async def mark_invitation_used(self, invitation: Invitation):
        async def _operation(session):
            await invitation.use(session)

        return await self.execute_with_transaction(_operation)

    async def query_invitations(self):
        """Query all valid invitations (not used), ordered by created_at descending."""

        async def _query(session):
            stmt = select(Invitation).where(not Invitation.is_used).order_by(desc(Invitation.created_at))
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_api_keys(self, user: str):
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

        async def _operation(session):
            api_key = ApiKey(user=user, description=description, status=ApiKeyStatus.ACTIVE)
            session.add(api_key)
            await session.flush()
            await session.refresh(api_key)
            return api_key

        return await self.execute_with_transaction(_operation)

    async def delete_api_key(self, user: str, key_id: str) -> bool:
        """Delete an API key (soft delete)"""

        async def _operation(session):
            stmt = select(ApiKey).where(
                ApiKey.id == key_id,
                ApiKey.user == user,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            api_key = result.scalars().first()
            if api_key:
                api_key.status = ApiKeyStatus.DELETED
                from datetime import datetime as dt

                api_key.gmt_deleted = dt.utcnow()
                session.add(api_key)
                await session.flush()
                return True
            return False

        return await self.execute_with_transaction(_operation)

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

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    async def update_bot_by_id(
        self, user: str, bot_id: str, title: str, description: str, bot_type, config: str
    ) -> Optional[Bot]:
        """Update bot by ID"""

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    async def delete_bot_by_id(self, user: str, bot_id: str) -> Optional[Bot]:
        """Soft delete bot by ID"""

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    async def create_bot_collection_relation(self, bot_id: str, collection_id: str):
        """Create bot-collection relation"""
        from aperag.db.models import BotCollectionRelation

        async def _operation(session):
            relation = BotCollectionRelation(bot_id=bot_id, collection_id=collection_id)
            session.add(relation)
            await session.flush()
            return relation

        return await self.execute_with_transaction(_operation)

    async def delete_bot_collection_relations(self, bot_id: str):
        """Soft delete all bot-collection relations for a bot"""
        from aperag.db.models import BotCollectionRelation

        async def _operation(session):
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

        return await self.execute_with_transaction(_operation)

    # Document Operations
    async def create_document(
        self, user: str, collection_id: str, name: str, size: int, object_path: str = None, metadata: str = None
    ) -> Document:
        """Create a new document in database"""

        async def _operation(session):
            instance = Document(
                user=user,
                name=name,
                status=DocumentStatus.PENDING,
                size=size,
                collection_id=collection_id,
                object_path=object_path,
                doc_metadata=metadata,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_document_by_id(
        self, user: str, collection_id: str, document_id: str, metadata: str = None
    ) -> Optional[Document]:
        """Update document by ID"""

        async def _operation(session):
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance and metadata is not None:
                instance.doc_metadata = metadata
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_document_by_id(self, user: str, collection_id: str, document_id: str) -> Optional[Document]:
        """Soft delete document by ID"""
        from aperag.db.models import DocumentStatus

        async def _operation(session):
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.status = DocumentStatus.DELETING
                instance.gmt_deleted = datetime.utcnow()
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_documents_by_ids(self, user: str, collection_id: str, document_ids: List[str]) -> tuple:
        """Bulk soft delete documents by IDs"""
        from aperag.db.models import DocumentStatus

        async def _operation(session):
            stmt = select(Document).where(
                Document.id.in_(document_ids),
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            documents = result.scalars().all()

            success_ids = []
            for doc in documents:
                try:
                    doc.status = DocumentStatus.DELETING
                    doc.gmt_deleted = datetime.utcnow()
                    session.add(doc)
                    success_ids.append(doc.id)
                except Exception:
                    continue

            await session.flush()
            failed_ids = [doc_id for doc_id in document_ids if doc_id not in success_ids]
            return success_ids, failed_ids

        return await self.execute_with_transaction(_operation)

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

    async def update_api_key_by_id(self, user: str, key_id: str, description: str) -> Optional[ApiKey]:
        """Update API key description"""

        async def _operation(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user,
                ApiKey.id == key_id,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            api_key = result.scalars().first()

            if api_key:
                api_key.description = description
                session.add(api_key)
                await session.flush()
                await session.refresh(api_key)

            return api_key

        return await self.execute_with_transaction(_operation)

    # LLM Provider Operations
    async def query_llm_providers(self):
        """Get all active LLM providers"""

        async def _query(session):
            stmt = select(LLMProvider).where(LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_llm_provider_by_name(self, name: str):
        """Get LLM provider by name"""

        async def _query(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_llm_provider(
        self,
        name: str,
        label: str,
        completion_dialect: str = "openai",
        embedding_dialect: str = "openai",
        rerank_dialect: str = "jina_ai",
        allow_custom_base_url: bool = False,
        base_url: str = "",
        extra: str = None,
    ) -> LLMProvider:
        """Create a new LLM provider"""

        async def _operation(session):
            provider = LLMProvider(
                name=name,
                label=label,
                completion_dialect=completion_dialect,
                embedding_dialect=embedding_dialect,
                rerank_dialect=rerank_dialect,
                allow_custom_base_url=allow_custom_base_url,
                base_url=base_url,
                extra=extra,
            )
            session.add(provider)
            await session.flush()
            await session.refresh(provider)
            return provider

        return await self.execute_with_transaction(_operation)

    async def update_llm_provider(
        self,
        name: str,
        label: str = None,
        completion_dialect: str = None,
        embedding_dialect: str = None,
        rerank_dialect: str = None,
        allow_custom_base_url: bool = None,
        base_url: str = None,
        extra: str = None,
    ) -> Optional[LLMProvider]:
        """Update an existing LLM provider"""

        async def _operation(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            provider = result.scalars().first()

            if provider:
                if label is not None:
                    provider.label = label
                if completion_dialect is not None:
                    provider.completion_dialect = completion_dialect
                if embedding_dialect is not None:
                    provider.embedding_dialect = embedding_dialect
                if rerank_dialect is not None:
                    provider.rerank_dialect = rerank_dialect
                if allow_custom_base_url is not None:
                    provider.allow_custom_base_url = allow_custom_base_url
                if base_url is not None:
                    provider.base_url = base_url
                if extra is not None:
                    provider.extra = extra

                provider.gmt_updated = datetime.utcnow()
                session.add(provider)
                await session.flush()
                await session.refresh(provider)

            return provider

        return await self.execute_with_transaction(_operation)

    async def delete_llm_provider(self, name: str) -> Optional[LLMProvider]:
        """Soft delete LLM provider and its models"""

        async def _operation(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            provider = result.scalars().first()

            if provider:
                # Soft delete the provider
                provider.gmt_deleted = datetime.utcnow()
                provider.gmt_updated = datetime.utcnow()
                session.add(provider)

                # Also soft delete all models for this provider
                models_stmt = select(LLMProviderModel).where(
                    LLMProviderModel.provider_name == name, LLMProviderModel.gmt_deleted.is_(None)
                )
                models_result = await session.execute(models_stmt)
                models = models_result.scalars().all()
                for model in models:
                    model.gmt_deleted = datetime.utcnow()
                    model.gmt_updated = datetime.utcnow()
                    session.add(model)

                await session.flush()
                await session.refresh(provider)

            return provider

        return await self.execute_with_transaction(_operation)

    async def restore_llm_provider(self, name: str) -> Optional[LLMProvider]:
        """Restore a soft-deleted LLM provider"""

        async def _operation(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_not(None))
            result = await session.execute(stmt)
            provider = result.scalars().first()

            if provider:
                provider.gmt_deleted = None
                provider.gmt_updated = datetime.utcnow()
                session.add(provider)

                # Also restore all models for this provider
                models_stmt = select(LLMProviderModel).where(
                    LLMProviderModel.provider_name == name, LLMProviderModel.gmt_deleted.is_not(None)
                )
                models_result = await session.execute(models_stmt)
                models = models_result.scalars().all()
                for model in models:
                    model.gmt_deleted = None
                    model.gmt_updated = datetime.utcnow()
                    session.add(model)

                await session.flush()
                await session.refresh(provider)

            return provider

        return await self.execute_with_transaction(_operation)

    # LLM Provider Model Operations
    async def query_llm_provider_models(self, provider_name: str = None):
        """Get all active LLM provider models, optionally filtered by provider"""

        async def _query(session):
            stmt = select(LLMProviderModel).where(LLMProviderModel.gmt_deleted.is_(None))
            if provider_name:
                stmt = stmt.where(LLMProviderModel.provider_name == provider_name)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_llm_provider_model(self, provider_name: str, api: str, model: str):
        """Get a specific LLM provider model"""

        async def _query(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_llm_provider_model(
        self,
        provider_name: str,
        api: str,
        model: str,
        custom_llm_provider: str,
        max_tokens: int = None,
        tags: list = None,
    ) -> LLMProviderModel:
        """Create a new LLM provider model"""

        async def _operation(session):
            from aperag.db.models import APIType

            # Convert enum to string if needed
            api_value = api.value if isinstance(api, APIType) else api

            model_obj = LLMProviderModel(
                provider_name=provider_name,
                api=api_value,
                model=model,
                custom_llm_provider=custom_llm_provider,
                max_tokens=max_tokens,
                tags=tags or [],
            )
            session.add(model_obj)
            await session.flush()
            await session.refresh(model_obj)
            return model_obj

        return await self.execute_with_transaction(_operation)

    async def update_llm_provider_model(
        self,
        provider_name: str,
        api: str,
        model: str,
        custom_llm_provider: str = None,
        max_tokens: int = None,
        tags: list = None,
    ) -> Optional[LLMProviderModel]:
        """Update an existing LLM provider model"""

        async def _operation(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            model_obj = result.scalars().first()

            if model_obj:
                if custom_llm_provider is not None:
                    model_obj.custom_llm_provider = custom_llm_provider
                if max_tokens is not None:
                    model_obj.max_tokens = max_tokens
                if tags is not None:
                    model_obj.tags = tags

                model_obj.gmt_updated = datetime.utcnow()
                session.add(model_obj)
                await session.flush()
                await session.refresh(model_obj)

            return model_obj

        return await self.execute_with_transaction(_operation)

    async def delete_llm_provider_model(self, provider_name: str, api: str, model: str) -> Optional[LLMProviderModel]:
        """Soft delete LLM provider model"""

        async def _operation(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            model_obj = result.scalars().first()

            if model_obj:
                model_obj.gmt_deleted = datetime.utcnow()
                model_obj.gmt_updated = datetime.utcnow()
                session.add(model_obj)
                await session.flush()
                await session.refresh(model_obj)

            return model_obj

        return await self.execute_with_transaction(_operation)

    async def restore_llm_provider_model(self, provider_name: str, api: str, model: str) -> Optional[LLMProviderModel]:
        """Restore a soft-deleted LLM provider model"""

        async def _operation(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_not(None),
            )
            result = await session.execute(stmt)
            model_obj = result.scalars().first()

            if model_obj:
                model_obj.gmt_deleted = None
                model_obj.gmt_updated = datetime.utcnow()
                session.add(model_obj)
                await session.flush()
                await session.refresh(model_obj)

            return model_obj

        return await self.execute_with_transaction(_operation)


# Create a global instance for backwards compatibility and easy access
# This can be used in places where session dependency injection is not available
async_db_ops = AsyncDatabaseOps()
db_ops = DatabaseOps()
