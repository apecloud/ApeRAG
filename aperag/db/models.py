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

import enum
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Type

from sqlalchemy import JSON, Column, String, TypeDecorator
from sqlmodel import Field, SQLModel, UniqueConstraint, select

from aperag.config import AsyncSessionDep


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


class DocumentIndexStatus(str, Enum):
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


class EnumAsSQLALCHEMYString(TypeDecorator):
    """
    Stores Python Enums as their string values in a VARCHAR column.
    Retrieves them back as Enum members.
    """

    impl = String  # The underlying SQLAlchemy type is String (VARCHAR)
    cache_ok = True  # Important for performance with TypeDecorator

    def __init__(self, enum_class: Type[enum.Enum], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enum_class = enum_class

    def process_bind_param(self, value: Optional[enum.Enum], dialect) -> Optional[str]:
        # Called when sending data to the database
        if value is None:
            return None
        if not isinstance(value, self._enum_class):
            raise ValueError(f"Value {value!r} is not an instance of enum {self._enum_class.__name__}")
        return value.value  # Store the enum's .value (e.g., "completion")

    def process_result_value(self, value: Optional[str], dialect) -> Optional[enum.Enum]:
        # Called when retrieving data from the database
        if value is None:
            return None
        try:
            return self._enum_class(value)  # Convert string value back to Enum member
        except ValueError:
            # Handle cases where the value from DB doesn't match any enum member
            # This could happen if enum definitions change or data is manually inserted
            # You might want to log this, return None, or raise a different error
            raise ValueError(f"Invalid value '{value}' for enum {self._enum_class.__name__}")

    # If you want to ensure the VARCHAR has a specific length,
    # you can override load_dialect_impl.
    # This is often preferred over passing length directly to String in __init__
    # if you want the TypeDecorator to be more self-contained.
    def load_dialect_impl(self, dialect):
        # Get the implementation for the specific dialect
        # Here, we ensure the String type has the length provided during __init__
        # The 'length' attribute is passed from Column(EnumAsSQLALCHEMYString(APIType, length=50))
        if hasattr(self, "length"):  # 'length' will be passed from Column's constructor args
            return dialect.type_descriptor(String(self.length))
        return dialect.type_descriptor(String)  # Or a default if no length given


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

    async def bots(self, session: AsyncSessionDep, only_ids: bool = False):
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


class Document(SQLModel, table=True):
    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("collection_id", "name", "gmt_deleted", name="uq_document_collection_name_deleted"),
    )

    id: str = Field(default_factory=lambda: "doc" + random_id(), primary_key=True, max_length=24)
    name: str = Field(max_length=1024)
    user: str = Field(max_length=256)
    collection_id: Optional[str] = Field(default=None, max_length=24)
    status: DocumentStatus
    vector_index_status: DocumentIndexStatus = DocumentIndexStatus.PENDING
    fulltext_index_status: DocumentIndexStatus = DocumentIndexStatus.PENDING
    graph_index_status: DocumentIndexStatus = DocumentIndexStatus.PENDING
    size: int
    object_path: Optional[str] = None
    doc_metadata: Optional[str] = None  # Store document metadata as JSON string
    relate_ids: Optional[str] = None
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    def get_overall_status(self) -> "DocumentStatus":
        """Calculate overall status based on individual index statuses"""
        index_statuses = [self.vector_index_status, self.fulltext_index_status, self.graph_index_status]
        if any(status == DocumentIndexStatus.FAILED for status in index_statuses):
            return DocumentStatus.FAILED
        elif any(status == DocumentIndexStatus.RUNNING for status in index_statuses):
            return DocumentStatus.RUNNING
        elif all(status in [DocumentIndexStatus.COMPLETE, DocumentIndexStatus.SKIPPED] for status in index_statuses):
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

    async def get_collection(self, session: AsyncSessionDep):
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

    async def collections(self, session: AsyncSessionDep, only_ids: bool = False):
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


class BotCollectionRelation(SQLModel, table=True):
    __tablename__ = "bot_collection_relation"
    __table_args__ = (UniqueConstraint("bot_id", "collection_id", "gmt_deleted", name="uq_bot_collection_deleted"),)

    id: str = Field(default_factory=lambda: "bcr" + random_id(), primary_key=True, max_length=24)
    bot_id: str = Field(max_length=24)
    collection_id: str = Field(max_length=24)
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
    __table_args__ = (
        UniqueConstraint("bot_id", "peer_type", "peer_id", "gmt_deleted", name="uq_chat_bot_peer_deleted"),
    )

    id: str = Field(default_factory=lambda: "chat" + random_id(), primary_key=True, max_length=24)
    user: str = Field(max_length=256)
    peer_type: ChatPeerType = ChatPeerType.SYSTEM
    peer_id: Optional[str] = Field(default=None, max_length=256)
    status: ChatStatus
    bot_id: str = Field(max_length=24)
    title: Optional[str] = Field(default=None, max_length=256)
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    async def get_bot(self, session: AsyncSessionDep):
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
    __table_args__ = (
        UniqueConstraint("chat_id", "message_id", "gmt_deleted", name="uq_feedback_chat_message_deleted"),
    )

    user: str = Field(max_length=256)
    collection_id: Optional[str] = Field(default=None, max_length=24)
    chat_id: str = Field(max_length=24, primary_key=True)
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

    async def get_collection(self, session: AsyncSessionDep):
        """Get the associated collection object"""
        from aperag.db.models import Collection

        return await session.get(Collection, self.collection_id)

    async def set_collection(self, collection):
        """Set the collection_id by Collection object or id"""
        if hasattr(collection, "id"):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection

    async def get_chat(self, session: AsyncSessionDep):
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
        """Generate a unique API key"""
        return f"sk-{uuid.uuid4().hex}"

    async def update_last_used(self, session: AsyncSessionDep):
        """Update the last_used_at timestamp"""
        self.last_used_at = datetime.utcnow()
        session.add(self)
        await session.commit()


class ModelServiceProvider(SQLModel, table=True):
    __tablename__ = "model_service_provider"
    __table_args__ = (
        UniqueConstraint("name", "user", "gmt_deleted", name="uq_model_service_provider_name_user_deleted"),
    )

    id: str = Field(default_factory=lambda: "msp" + random_id(), primary_key=True, max_length=24)
    name: str = Field(max_length=256)
    user: str = Field(max_length=256)
    status: ModelServiceProviderStatus
    api_key: str = Field(max_length=256)
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None


class LLMProvider(SQLModel, table=True):
    """LLM Provider configuration model

    This model stores the provider-level configuration that was previously
    stored in model_configs.json file. Each provider has basic information
    and dialect configurations for different API types.
    """

    __tablename__ = "llm_provider"

    name: str = Field(primary_key=True, max_length=128)  # Unique provider name identifier
    label: str = Field(max_length=256)  # Human-readable provider display name
    completion_dialect: str = Field(max_length=64)  # API dialect for completion/chat APIs
    embedding_dialect: str = Field(max_length=64)  # API dialect for embedding APIs
    rerank_dialect: str = Field(max_length=64)  # API dialect for rerank APIs
    allow_custom_base_url: bool = Field(default=False)  # Whether custom base URLs are allowed
    base_url: str = Field(max_length=512)  # Default API base URL for this provider
    extra: Optional[str] = Field(default=None)  # Additional configuration data in JSON format
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    def __str__(self):
        return f"LLMProvider(name={self.name}, label={self.label})"


class APIType(str, Enum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class LLMProviderModel(SQLModel, table=True):
    """LLM Provider Model configuration

    This model stores individual model configurations for each provider.
    Each model belongs to a provider and has a specific API type (completion, embedding, rerank).
    """

    __tablename__ = "llm_provider_models"
    __table_args__ = ()

    provider_name: str = Field(primary_key=True, max_length=128)  # Reference to LLMProvider.name
    api: APIType = Field(sa_column=Column(EnumAsSQLALCHEMYString(APIType, length=50), nullable=False, primary_key=True))
    model: str = Field(primary_key=True, max_length=256)  # Model name/identifier
    custom_llm_provider: str = Field(max_length=128)  # Custom LLM provider implementation
    max_tokens: Optional[int] = Field(default=None)  # Maximum tokens for this model
    tags: Optional[list] = Field(default_factory=list, sa_column=Column(JSON))  # Tags for model categorization
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_updated: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None

    def __str__(self):
        return f"LLMProviderModel(provider={self.provider_name}, api={self.api}, model={self.model})"

    async def get_provider(self, session: AsyncSessionDep):
        """Get the associated provider object"""
        return await session.get(LLMProvider, self.provider_name)

    async def set_provider(self, provider):
        """Set the provider_name by LLMProvider object or name"""
        if hasattr(provider, "name"):
            self.provider_name = provider.name
        elif isinstance(provider, str):
            self.provider_name = provider

    def has_tag(self, tag: str) -> bool:
        """Check if model has a specific tag"""
        return tag in (self.tags or [])

    def add_tag(self, tag: str) -> bool:
        """Add a tag to model. Returns True if tag was added, False if already exists"""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            return True
        return False

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from model. Returns True if tag was removed, False if not found"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

    def get_tags(self) -> list:
        """Get all tags for this model"""
        return self.tags or []


class User(SQLModel, table=True):
    __tablename__ = "user"
    id: str = Field(default_factory=lambda: "user" + random_id(), primary_key=True, max_length=24)
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

    @property
    def password(self):
        raise AttributeError("password is not a readable attribute")

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
        """Check if invitation is still valid"""
        now = datetime.utcnow()
        return not self.is_used and now < self.expires_at

    async def use(self, session: AsyncSessionDep):
        """Mark invitation as used"""
        self.is_used = True
        self.used_at = datetime.utcnow()
        session.add(self)
        await session.commit()

        # Auto-expire after use (optional)
        # self.expires_at = datetime.utcnow()


class SearchTestHistory(SQLModel, table=True):
    __tablename__ = "searchtesthistory"
    id: str = Field(default_factory=lambda: "sth" + random_id(), primary_key=True, max_length=24)
    user: str = Field(max_length=256)
    collection_id: Optional[str] = Field(default=None, max_length=24)
    query: str
    vector_search: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    fulltext_search: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    graph_search: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    items: Optional[list] = Field(default_factory=list, sa_column=Column(JSON))
    gmt_created: datetime = Field(default_factory=datetime.utcnow)
    gmt_deleted: Optional[datetime] = None
