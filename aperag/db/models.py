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

import random
import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import JSON as SAJSON
from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import declarative_base

from aperag.config import SessionDep

Base = declarative_base()


# Helper function for random id generation
def random_id():
    """Generate a random ID string"""
    return "".join(random.sample(uuid.uuid4().hex, 16))


# Enums for choices
class CollectionStatus(PyEnum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class CollectionSyncStatus(PyEnum):
    RUNNING = "RUNNING"
    CANCELED = "CANCELED"
    COMPLETED = "COMPLETED"


class CollectionType(PyEnum):
    DOCUMENT = "document"


class DocumentStatus(PyEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class IndexStatus(PyEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class BotStatus(PyEnum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class BotType(PyEnum):
    KNOWLEDGE = "knowledge"
    COMMON = "common"


class Role(PyEnum):
    ADMIN = "admin"
    RW = "rw"
    RO = "ro"


class ChatStatus(PyEnum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class ChatPeerType(PyEnum):
    SYSTEM = "system"
    FEISHU = "feishu"
    WEIXIN = "weixin"
    WEIXIN_OFFICIAL = "weixin_official"
    WEB = "web"
    DINGTALK = "dingtalk"


class MessageFeedbackStatus(PyEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class MessageFeedbackType(PyEnum):
    GOOD = "good"
    BAD = "bad"


class MessageFeedbackTag(PyEnum):
    HARMFUL = "Harmful"
    UNSAFE = "Unsafe"
    FAKE = "Fake"
    UNHELPFUL = "Unhelpful"
    OTHER = "Other"


class ModelServiceProviderStatus(PyEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETED = "DELETED"


# Models
class Collection(Base):
    __tablename__ = "collection"
    __table_args__ = (UniqueConstraint("id", name="uq_collection_id"),)

    id = Column(String(24), primary_key=True, default=lambda: "col" + random_id())
    title = Column(String(256), nullable=False)
    description = Column(String, nullable=True)
    user = Column(String(256), nullable=False)
    status = Column(Enum(CollectionStatus), nullable=False)
    type = Column(Enum(CollectionType), nullable=False)
    config = Column(String, nullable=False)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)

    async def bots(self, session, only_ids: bool = False):
        """Get all active bots related to this collection"""
        from aperag.db.models import Bot, BotCollectionRelation

        stmt = select(BotCollectionRelation).where(
            BotCollectionRelation.collection_id == self.id, BotCollectionRelation.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        rels = result.scalars().all()
        if only_ids:
            return [rel.bot_id for rel in rels]
        else:
            bots = []
            for rel in rels:
                bot = await session.get(Bot, rel.bot_id)
                if bot:
                    bots.append(bot)
            return bots


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("collection_id", "name", name="uq_document_collection_name"),)

    id = Column(String(24), primary_key=True, default=lambda: "doc" + random_id())
    name = Column(String(1024), nullable=False)
    user = Column(String(256), nullable=False)
    config = Column(String, nullable=True)
    collection_id = Column(String(24), ForeignKey("collection.id"), nullable=True)
    status = Column(Enum(DocumentStatus), nullable=False)
    vector_index_status = Column(Enum(IndexStatus), default=IndexStatus.PENDING, nullable=False)
    fulltext_index_status = Column(Enum(IndexStatus), default=IndexStatus.PENDING, nullable=False)
    graph_index_status = Column(Enum(IndexStatus), default=IndexStatus.PENDING, nullable=False)
    size = Column(Integer, nullable=False)
    object_path = Column(String, nullable=True)
    relate_ids = Column(String, nullable=False)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)

    def get_overall_status(self) -> "DocumentStatus":
        """Calculate overall status based on individual index statuses"""
        index_statuses = [self.vector_index_status, self.fulltext_index_status, self.graph_index_status]
        if any(status == IndexStatus.FAILED for status in index_statuses):
            return DocumentStatus.FAILED
        elif any(status == IndexStatus.RUNNING for status in index_statuses):
            return DocumentStatus.RUNNING
        elif all(status in [IndexStatus.COMPLETE, IndexStatus.SKIPPED] for status in index_statuses):
            return DocumentStatus.COMPLETE
        else:
            return DocumentStatus.PENDING

    def update_overall_status(self):
        """Update overall status field"""
        self.status = self.get_overall_status()

    def object_store_base_path(self) -> str:
        """Generate the base path for object store"""
        user = self.user.replace("|", "-")
        return f"user-{user}/{self.collection_id}/{self.id}"

    async def get_collection(self, session):
        """Get the associated collection object"""
        from aperag.db.models import Collection

        return await session.get(Collection, self.collection_id)

    async def set_collection(self, collection):
        """Set the collection_id by Collection object or id"""
        if hasattr(collection, "id"):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection


class Bot(Base):
    __tablename__ = "bot"
    id = Column(String(24), primary_key=True, default=lambda: "bot" + random_id())
    user = Column(String(256), nullable=False)
    title = Column(String(256), nullable=False)
    type = Column(Enum(BotType), default=BotType.KNOWLEDGE, nullable=False)
    description = Column(String, nullable=True)
    status = Column(Enum(BotStatus), nullable=False)
    config = Column(String, nullable=False)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)

    async def collections(self, session, only_ids: bool = False):
        """Get all active collections related to this bot"""
        from aperag.db.models import BotCollectionRelation, Collection

        stmt = select(BotCollectionRelation).where(
            BotCollectionRelation.bot_id == self.id, BotCollectionRelation.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        rels = result.scalars().all()
        if only_ids:
            return [rel.collection_id for rel in rels]
        else:
            collections = []
            for rel in rels:
                collection = await session.get(Collection, rel.collection_id)
                if collection:
                    collections.append(collection)
            return collections


class BotCollectionRelation(Base):
    __tablename__ = "bot_collection_relation"
    __table_args__ = (UniqueConstraint("bot_id", "collection_id", name="unique_active_bot_collection"),)

    bot_id = Column(String(24), ForeignKey("bot.id"), primary_key=True)
    collection_id = Column(String(24), ForeignKey("collection.id"), primary_key=True)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)


class ConfigModel(Base):
    __tablename__ = "config"
    key = Column(String(256), primary_key=True)
    value = Column(String, nullable=False)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)


class UserQuota(Base):
    __tablename__ = "user_quota"
    user = Column(String(256), primary_key=True)
    key = Column(String(256), primary_key=True)
    value = Column(Integer, default=0)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)


class Chat(Base):
    __tablename__ = "chat"
    __table_args__ = (UniqueConstraint("bot_id", "peer_type", "peer_id", name="uq_chat_bot_peer"),)

    id = Column(String(24), primary_key=True, default=lambda: "chat" + random_id())
    user = Column(String(256), nullable=False)
    peer_type = Column(Enum(ChatPeerType), default=ChatPeerType.SYSTEM, nullable=False)
    peer_id = Column(String(256), nullable=True)
    status = Column(Enum(ChatStatus), nullable=False)
    bot_id = Column(String(24), ForeignKey("bot.id"), nullable=False)
    title = Column(String, nullable=False)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)

    async def get_bot(self, session: SessionDep):
        """Get the associated bot object"""
        from aperag.db.models import Bot

        return await session.get(Bot, self.bot_id)

    async def set_bot(self, bot):
        """Set the bot_id by Bot object or id"""
        if hasattr(bot, "id"):
            self.bot_id = bot.id
        elif isinstance(bot, str):
            self.bot_id = bot


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    __table_args__ = (UniqueConstraint("chat_id", "message_id", name="uq_messagefeedback_chat_message"),)

    user = Column(String(256), primary_key=True)
    collection_id = Column(String(24), ForeignKey("collection.id"), nullable=True)
    chat_id = Column(String(24), ForeignKey("chat.id"), primary_key=True)
    message_id = Column(String(256), primary_key=True)
    type = Column(Enum(MessageFeedbackType), nullable=True)
    tag = Column(Enum(MessageFeedbackTag), nullable=True)
    message = Column(String, nullable=True)
    question = Column(String, nullable=True)
    status = Column(Enum(MessageFeedbackStatus), nullable=True)
    original_answer = Column(String, nullable=True)
    revised_answer = Column(String, nullable=True)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)

    async def get_collection(self, session):
        """Get the associated collection object"""
        from aperag.db.models import Collection

        return await session.get(Collection, self.collection_id)

    async def set_collection(self, collection):
        """Set the collection_id by Collection object or id"""
        if hasattr(collection, "id"):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection

    async def get_chat(self, session):
        """Get the associated chat object"""
        from aperag.db.models import Chat

        return await session.get(Chat, self.chat_id)

    async def set_chat(self, chat):
        """Set the chat_id by Chat object or id"""
        if hasattr(chat, "id"):
            self.chat_id = chat.id
        elif isinstance(chat, str):
            self.chat_id = chat


class ApiKey(Base):
    __tablename__ = "api_key"
    id = Column(String(24), primary_key=True, default=lambda: "".join(random.sample(uuid.uuid4().hex, 12)))
    key = Column(String(40), default=lambda: f"sk-{uuid.uuid4().hex}")
    user = Column(String(256), nullable=False)
    description = Column(String(256), nullable=True)
    status = Column(Enum(BotStatus), default=BotStatus.ACTIVE, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)

    @staticmethod
    def generate_key() -> str:
        """Generate a random API key with sk- prefix"""
        return f"sk-{uuid.uuid4().hex}"

    async def update_last_used(self, session):
        """Update the last used timestamp"""
        from datetime import datetime as dt

        self.last_used_at = dt.utcnow()
        session.add(self)
        await session.commit()


class CollectionSyncHistory(Base):
    __tablename__ = "collection_sync_history"
    id = Column(String(24), primary_key=True, default=lambda: "colhist" + random_id())
    user = Column(String(256), nullable=False)
    collection_id = Column(String(24), ForeignKey("collection.id"), nullable=False)
    total_documents = Column(Integer, default=0)
    new_documents = Column(Integer, default=0)
    deleted_documents = Column(Integer, default=0)
    modified_documents = Column(Integer, default=0)
    processing_documents = Column(Integer, default=0)
    pending_documents = Column(Integer, default=0)
    failed_documents = Column(Integer, default=0)
    successful_documents = Column(Integer, default=0)
    total_documents_to_sync = Column(Integer, default=0)
    execution_time = Column(Integer, nullable=True)  # store as seconds
    start_time = Column(DateTime, default=datetime.utcnow)
    task_context = Column(SAJSON, default=dict)
    status = Column(Enum(CollectionSyncStatus), default=CollectionSyncStatus.RUNNING, nullable=False)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)


class ModelServiceProvider(Base):
    __tablename__ = "model_service_provider"
    name = Column(String(24), primary_key=True, default=lambda: "int" + random_id())
    user = Column(String(256), nullable=False)
    status = Column(Enum(ModelServiceProviderStatus), nullable=False)
    dialect = Column(String(32), default="openai", nullable=False)
    base_url = Column(String(256), nullable=True)
    api_key = Column(String(256), nullable=False)
    extra = Column(String, nullable=True)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(150), unique=True, nullable=False)
    email = Column(String(254), unique=True, nullable=True)
    role = Column(Enum(Role), default=Role.RO, nullable=False)
    password = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_staff = Column(Boolean, default=False)
    date_joined = Column(DateTime, default=datetime.utcnow)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_updated = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)


class Invitation(Base):
    __tablename__ = "invitation"
    id = Column(String(24), primary_key=True, default=lambda: "invite" + random_id())
    email = Column(String(254), nullable=False)
    token = Column(String(64), unique=True, nullable=False)
    created_by = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_used = Column(Boolean, default=False)
    used_at = Column(DateTime, nullable=True)
    role = Column(Enum(Role), default=Role.RO, nullable=False)

    def is_valid(self) -> bool:
        """Check if the invitation is valid (not used and not expired)"""
        from datetime import datetime as dt

        return not self.is_used and self.expires_at > dt.utcnow()

    async def use(self, session):
        """Mark invitation as used and set used_at"""
        from datetime import datetime as dt

        self.is_used = True
        self.used_at = dt.utcnow()
        session.add(self)
        await session.commit()


class SearchTestHistory(Base):
    __tablename__ = "searchtesthistory"
    id = Column(String(24), primary_key=True, default=lambda: "sth" + random_id())
    user = Column(String(256), nullable=False)
    collection_id = Column(String(24), ForeignKey("collection.id"), nullable=True)
    query = Column(String, nullable=False)
    vector_search = Column(SAJSON, default=dict)
    fulltext_search = Column(SAJSON, default=dict)
    graph_search = Column(SAJSON, default=dict)
    items = Column(SAJSON, default=list)
    gmt_created = Column(DateTime, default=datetime.utcnow)
    gmt_deleted = Column(DateTime, nullable=True)
