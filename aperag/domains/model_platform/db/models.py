"""Model-platform-domain SQLAlchemy models."""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    return "".join(random.sample(uuid.uuid4().hex, 16))


def _enum_column(enum_class):
    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class ModelCapability(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ModelAccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class ModelStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class ModelUseScenario(str, Enum):
    AGENT_CHAT = "agent_chat"
    COLLECTION_COMPLETION = "collection_completion"
    COLLECTION_EMBEDDING = "collection_embedding"
    RETRIEVAL_RERANK = "retrieval_rerank"
    BACKGROUND_TASK = "background_task"


class ModelUseStrategy(str, Enum):
    SINGLE = "single"
    FALLBACK = "fallback"


class ModelProvider(Base):
    """Built-in provider template, not a user's credential."""

    __tablename__ = "model_provider"

    id = Column(String(24), primary_key=True, default=lambda: "mp" + _random_id())
    provider_type = Column(String(128), nullable=False, unique=True, index=True)
    display_name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    supported_capabilities = Column(JSON, default=lambda: [], nullable=False)
    account_schema = Column(JSON, default=lambda: {}, nullable=False)
    default_base_url = Column(String(512), nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)
    extra = Column(JSON, default=lambda: {}, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)


class ModelAccount(Base):
    """A user's real endpoint and credential for a provider."""

    __tablename__ = "model_account"

    id = Column(String(24), primary_key=True, default=lambda: "ma" + _random_id())
    user_id = Column(String(256), nullable=False, index=True)
    provider_type = Column(String(128), nullable=False, index=True)
    name = Column(String(128), nullable=False, index=True)
    display_name = Column(String(256), nullable=False)
    base_url = Column(String(512), nullable=False)
    encrypted_api_key = Column(Text, nullable=False)
    auth_config = Column(JSON, default=lambda: {}, nullable=False)
    status = Column(_enum_column(ModelAccountStatus), default=ModelAccountStatus.ACTIVE.value, nullable=False)
    last_validated_at = Column(DateTime(timezone=True), nullable=True)
    validation_error = Column(Text, nullable=True)
    extra = Column(JSON, default=lambda: {}, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)


class Model(Base):
    """Concrete callable model available through a model account."""

    __tablename__ = "model"

    id = Column(String(24), primary_key=True, default=lambda: "mdl" + _random_id())
    account_id = Column(String(24), nullable=False, index=True)
    provider_model_id = Column(String(256), nullable=False, index=True)
    display_name = Column(String(256), nullable=False)
    capability = Column(_enum_column(ModelCapability), nullable=False, index=True)
    runner_type = Column(String(128), nullable=False)
    runner_config = Column(JSON, default=lambda: {}, nullable=False)
    context_window = Column(Integer, nullable=True)
    max_input_tokens = Column(Integer, nullable=True)
    max_output_tokens = Column(Integer, nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    supports_vision = Column(Boolean, default=False, nullable=False)
    supports_tool_calling = Column(Boolean, default=False, nullable=False)
    # Wave 5 P2 chunk 3: typed capability flag for embedding models
    # that accept image bytes; surfaced through the v3 model API +
    # consumed by ``base_embedding.get_embedding_service`` to set
    # ``EmbeddingService.multimodal=True``.
    supports_multimodal_embedding = Column(Boolean, default=False, nullable=False)
    status = Column(_enum_column(ModelStatus), default=ModelStatus.ACTIVE.value, nullable=False, index=True)
    extra = Column(JSON, default=lambda: {}, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)


class ModelUse(Base):
    """Default model selection for a product scenario."""

    __tablename__ = "model_use"

    id = Column(String(24), primary_key=True, default=lambda: "mu" + _random_id())
    user_id = Column(String(256), nullable=False, index=True)
    scenario = Column(_enum_column(ModelUseScenario), nullable=False, index=True)
    capability = Column(_enum_column(ModelCapability), nullable=False)
    strategy = Column(_enum_column(ModelUseStrategy), default=ModelUseStrategy.SINGLE.value, nullable=False)
    primary_model_id = Column(String(24), nullable=True, index=True)
    fallback_model_ids = Column(JSON, default=lambda: [], nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    extra = Column(JSON, default=lambda: {}, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)


class PromptTemplate(Base):
    __tablename__ = "prompt_template"

    id = Column(String(24), primary_key=True, default=lambda: "pt" + _random_id())
    prompt_type = Column(String(50), nullable=False, index=True)
    scope = Column(String(20), nullable=False, index=True)
    user_id = Column(String(256), nullable=True, index=True)
    content = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)


__all__ = [
    "Model",
    "ModelAccount",
    "ModelAccountStatus",
    "ModelCapability",
    "ModelProvider",
    "ModelStatus",
    "ModelUse",
    "ModelUseScenario",
    "ModelUseStrategy",
    "PromptTemplate",
]
