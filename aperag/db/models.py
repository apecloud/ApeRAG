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
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel, UniqueConstraint, select

from aperag.config import SessionDep


# Helper function for random id generation
def random_id():
    """Generate a random ID string"""
    return "".join(random.sample(uuid.uuid4().hex, 16))


# Enums for choices
class CollectionStatus(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class CollectionType(str, Enum):
    DOCUMENT = "document"


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class IndexStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class BotStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class BotType(str, Enum):
    KNOWLEDGE = "knowledge"
    COMMON = "common"


class Role(str, Enum):
    ADMIN = "admin"
    RW = "rw"
    RO = "ro"


class ChatStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class ChatPeerType(str, Enum):
    SYSTEM = "system"
    FEISHU = "feishu"
    WEIXIN = "weixin"
    WEIXIN_OFFICIAL = "weixin_official"
    WEB = "web"
    DINGTALK = "dingtalk"


class MessageFeedbackStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class MessageFeedbackType(str, Enum):
    GOOD = "good"
    BAD = "bad"


class MessageFeedbackTag(str, Enum):
    HARMFUL = "Harmful"
    UNSAFE = "Unsafe"
    FAKE = "Fake"
    UNHELPFUL = "Unhelpful"
    OTHER = "Other"


class ModelServiceProviderStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETED = "DELETED"


class ApiKeyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


# Models
class Collection(SQLModel, table=True):
    __tablename__ = "collection"
    __table_args__ = (UniqueConstraint("id", name="uq_collection_id"),)

    id: str = Field(default_factory=lambda: "col" + random_id(), primary_key=True, max_length=24)
    title: str = Field(max_length=256)
    description: Optional[str] = Field(default=None)
    user: str = Field(max_length=256)
    status: CollectionStatus
    type: CollectionType
    config: str
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    async def bots(self, session: SessionDep, only_ids: bool = False):
        """Get all active bots related to this collection"""
        from aperag.db.models import Bot, BotCollectionRelation

        stmt = select(BotCollectionRelation).where(
            BotCollectionRelation.collection_id == self.id, BotCollectionRelation.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        rels = result.all()
        if only_ids:
            return [rel.bot_id for rel in rels]
        else:
            bots = []
            for rel in rels:
                bot = await session.get(Bot, rel.bot_id)
                if bot:
                    bots.append(bot)
            return bots


class Document(SQLModel, table=True):
    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("collection_id", "name", name="uq_document_collection_name"),)

    id: str = Field(default_factory=lambda: "doc" + random_id(), primary_key=True, max_length=24)
    name: str = Field(max_length=1024)
    user: str = Field(max_length=256)
    config: Optional[str] = None
    collection_id: Optional[str] = Field(default=None, foreign_key="collection.id", max_length=24)
    status: DocumentStatus
    vector_index_status: IndexStatus = IndexStatus.PENDING
    fulltext_index_status: IndexStatus = IndexStatus.PENDING
    graph_index_status: IndexStatus = IndexStatus.PENDING
    size: int
    object_path: Optional[str] = None
    relate_ids: str
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

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

    async def get_collection(self, session: SessionDep):
        """Get the associated collection object"""
        from aperag.db.models import Collection

        return await session.get(Collection, self.collection_id)

    async def set_collection(self, collection):
        """Set the collection_id by Collection object or id"""
        if hasattr(collection, "id"):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection


class Bot(SQLModel, table=True):
    __tablename__ = "bot"
    id: str = Field(default_factory=lambda: "bot" + random_id(), primary_key=True, max_length=24)
    user: str = Field(max_length=256)
    title: Optional[str] = Field(default=None, max_length=256)
    type: BotType = BotType.KNOWLEDGE
    description: Optional[str] = None
    status: BotStatus
    config: str
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    async def collections(self, session: SessionDep, only_ids: bool = False):
        """Get all active collections related to this bot"""
        from aperag.db.models import BotCollectionRelation, Collection

        stmt = select(BotCollectionRelation).where(
            BotCollectionRelation.bot_id == self.id, BotCollectionRelation.gmt_deleted.is_(None)
        )
        result = await session.execute(stmt)
        rels = result.all()
        if only_ids:
            return [rel.collection_id for rel in rels]
        else:
            collections = []
            for rel in rels:
                collection = await session.get(Collection, rel.collection_id)
                if collection:
                    collections.append(collection)
            return collections


class BotCollectionRelation(SQLModel, table=True):
    __tablename__ = "bot_collection_relation"
    __table_args__ = (UniqueConstraint("bot_id", "collection_id", name="unique_active_bot_collection"),)

    bot_id: str = Field(foreign_key="bot.id", max_length=24, primary_key=True)
    collection_id: str = Field(foreign_key="collection.id", max_length=24, primary_key=True)
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None


class ConfigModel(SQLModel, table=True):
    __tablename__ = "config"
    key: str = Field(primary_key=True, max_length=256)
    value: str
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None


class UserQuota(SQLModel, table=True):
    __tablename__ = "user_quota"
    user: str = Field(max_length=256, primary_key=True)
    key: str = Field(max_length=256, primary_key=True)
    value: int = 0
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None


class Chat(SQLModel, table=True):
    __tablename__ = "chat"
    __table_args__ = (UniqueConstraint("bot_id", "peer_type", "peer_id", name="uq_chat_bot_peer"),)

    id: str = Field(default_factory=lambda: "chat" + random_id(), primary_key=True, max_length=24)
    user: str = Field(max_length=256)
    peer_type: ChatPeerType = ChatPeerType.SYSTEM
    peer_id: Optional[str] = Field(default=None, max_length=256)
    status: ChatStatus
    bot_id: str = Field(foreign_key="bot.id", max_length=24)
    title: Optional[str] = Field(default=None, max_length=256)
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

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


class MessageFeedback(SQLModel, table=True):
    __tablename__ = "message_feedback"
    __table_args__ = (UniqueConstraint("chat_id", "message_id", name="uq_messagefeedback_chat_message"),)

    user: str = Field(max_length=256)
    collection_id: Optional[str] = Field(default=None, foreign_key="collection.id", max_length=24)
    chat_id: str = Field(foreign_key="chat.id", max_length=24, primary_key=True)
    message_id: str = Field(max_length=256, primary_key=True)
    type: Optional[MessageFeedbackType] = None
    tag: Optional[MessageFeedbackTag] = None
    message: Optional[str] = None
    question: Optional[str] = None
    status: Optional[MessageFeedbackStatus] = None
    original_answer: Optional[str] = None
    revised_answer: Optional[str] = None
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    async def get_collection(self, session: SessionDep):
        """Get the associated collection object"""
        from aperag.db.models import Collection

        return await session.get(Collection, self.collection_id)

    async def set_collection(self, collection):
        """Set the collection_id by Collection object or id"""
        if hasattr(collection, "id"):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection

    async def get_chat(self, session: SessionDep):
        """Get the associated chat object"""
        from aperag.db.models import Chat

        return await session.get(Chat, self.chat_id)

    async def set_chat(self, chat):
        """Set the chat_id by Chat object or id"""
        if hasattr(chat, "id"):
            self.chat_id = chat.id
        elif isinstance(chat, str):
            self.chat_id = chat


class ApiKey(SQLModel, table=True):
    __tablename__ = "api_key"
    id: str = Field(
        default_factory=lambda: "".join(random.sample(uuid.uuid4().hex, 12)), primary_key=True, max_length=24
    )
    key: str = Field(default_factory=lambda: f"sk-{uuid.uuid4().hex}", max_length=40)
    user: str = Field(max_length=256)
    description: Optional[str] = Field(default=None, max_length=256)
    status: ApiKeyStatus
    last_used_at: Optional[datetime] = None
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    @staticmethod
    def generate_key() -> str:
        """Generate a random API key with sk- prefix"""
        return f"sk-{uuid.uuid4().hex}"

    async def update_last_used(self, session: SessionDep):
        """Update the last used timestamp"""
        from datetime import datetime as dt

        self.last_used_at = dt.utcnow()
        session.add(self)
        await session.commit()


class ModelServiceProvider(SQLModel, table=True):
    __tablename__ = "model_service_provider"
    name: str = Field(default_factory=lambda: "int" + random_id(), primary_key=True, max_length=24)
    user: str = Field(max_length=256)
    status: ModelServiceProviderStatus
    dialect: str = Field(default="openai", max_length=32)
    base_url: Optional[str] = Field(default=None, max_length=256)
    api_key: str = Field(max_length=256)
    extra: Optional[str] = None
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None


class User(SQLModel, table=True):
    __tablename__ = "user"
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(max_length=150, unique=True)
    email: Optional[str] = Field(default=None, unique=True, max_length=254)
    role: Role = Role.RO
    hashed_password: str = Field(max_length=128)  # fastapi-users expects hashed_password
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = True  # fastapi-users requires is_verified
    is_staff: bool = False
    date_joined: datetime = Field(default_factory=datetime.utcnow)
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    # For backward compatibility with existing code
    @property
    def password(self):
        return self.hashed_password

    @password.setter
    def password(self, value):
        self.hashed_password = value


class Invitation(SQLModel, table=True):
    __tablename__ = "invitation"
    id: str = Field(default_factory=lambda: "invite" + random_id(), primary_key=True, max_length=24)
    email: str = Field(max_length=254)
    token: str = Field(max_length=64, unique=True)
    created_by: str = Field(max_length=256)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    is_used: bool = False
    used_at: Optional[datetime] = None
    role: Role = Role.RO

    def is_valid(self) -> bool:
        """Check if the invitation is valid (not used and not expired)"""
        from datetime import datetime as dt

        return not self.is_used and self.expires_at > dt.utcnow()

    async def use(self, session: SessionDep):
        """Mark invitation as used and set used_at"""
        from datetime import datetime as dt

        self.is_used = True
        self.used_at = dt.utcnow()
        session.add(self)
        await session.commit()


class SearchTestHistory(SQLModel, table=True):
    __tablename__ = "searchtesthistory"
    id: str = Field(default_factory=lambda: "sth" + random_id(), primary_key=True, max_length=24)
    user: str = Field(max_length=256)
    collection_id: Optional[str] = Field(default=None, foreign_key="collection.id", max_length=24)
    query: str
    vector_search: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    fulltext_search: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    graph_search: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    items: Optional[list] = Field(default_factory=list, sa_column=Column(JSON))
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None
