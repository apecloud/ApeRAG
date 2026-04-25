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

"""Model-platform-domain SQLAlchemy models.

Owns the provider-metadata tables the model_platform domain is
responsible for:

* ``LLMProvider`` — per-provider (OpenAI / Anthropic / etc.)
  configuration (dialects, base URL, per-user overrides).
* ``LLMProviderModel`` — per-model metadata rows scoped to a
  provider (name, API type, context window, tags).
* ``ModelServiceProvider`` — soft-deletable per-name provider API
  key store (legacy table fed by ``llm_provider_service``).
* ``PromptTemplate`` — system / per-user prompt template store
  consumed by ``prompt_template_service`` (standalone-infra DI seam).

Plus their classification enums (``APIType`` /
``ModelServiceProviderStatus``). ``LLMProvider`` / ``LLMProviderModel``
/ ``APIType`` moved here from ``aperag.db.models`` in Phase 4
Step 4-S2d; ``ModelServiceProvider`` / ``ModelServiceProviderStatus``
/ ``PromptTemplate`` joined in Phase 8 Task #39 (legacy ORM carve).

``aperag.llm.*`` (embedding / rerank / completion runtime wrappers
over HTTP APIs) is **not** part of this domain — it stays as shared
infrastructure per Phase 4 canonical msg=d47fa490 Section 7.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, Integer, String, Text, text

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id``. Phase 6 cleanup
    consolidates."""

    return "".join(random.sample(uuid.uuid4().hex, 16))


def _enum_column(enum_class):
    """Mirror of ``aperag.db.models.EnumColumn``. Phase 6 cleanup
    consolidates."""

    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class APIType(str, Enum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ModelServiceProviderStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETED = "DELETED"


class LLMProvider(Base):
    """LLM Provider configuration model

    This model stores the provider-level configuration that was previously
    stored in model_configs.json file. Each provider has basic information
    and dialect configurations for different API types.
    """

    __tablename__ = "llm_provider"

    name = Column(String(128), primary_key=True)  # Unique provider name identifier
    user_id = Column(String(256), nullable=False, index=True)  # Owner of the provider config, "public" for global
    label = Column(String(256), nullable=False)  # Human-readable provider display name
    completion_dialect = Column(String(64), nullable=False)  # API dialect for completion/chat APIs
    embedding_dialect = Column(String(64), nullable=False)  # API dialect for embedding APIs
    rerank_dialect = Column(String(64), nullable=False)  # API dialect for rerank APIs
    allow_custom_base_url = Column(Boolean, default=False, nullable=False)  # Whether custom base URLs are allowed
    base_url = Column(String(512), nullable=False)  # Default API base URL for this provider
    extra = Column(Text, nullable=True)  # Additional configuration data in JSON format
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def __str__(self):
        return f"LLMProvider(name={self.name}, label={self.label}, user_id={self.user_id})"


class LLMProviderModel(Base):
    """LLM Provider Model configuration

    This model stores individual model configurations for each provider.
    Each model belongs to a provider and has a specific API type (completion, embedding, rerank).
    """

    __tablename__ = "llm_provider_models"

    provider_name = Column(String(128), primary_key=True)  # Reference to LLMProvider.name
    api = Column(_enum_column(APIType), nullable=False, primary_key=True)
    model = Column(String(256), primary_key=True)  # Model name/identifier
    custom_llm_provider = Column(String(128), nullable=False)  # Custom LLM provider implementation
    context_window = Column(Integer, nullable=True)  # Context window size (total tokens)
    max_input_tokens = Column(Integer, nullable=True)  # Maximum input tokens
    max_output_tokens = Column(Integer, nullable=True)  # Maximum output tokens
    tags = Column(JSON, default=lambda: [], nullable=True)  # Tags for model categorization
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def __str__(self):
        return f"LLMProviderModel(provider={self.provider_name}, api={self.api}, model={self.model})"

    async def get_provider(self, session):
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


class ModelServiceProvider(Base):
    __tablename__ = "model_service_provider"
    __table_args__ = (
        # Partial unique index: enforce uniqueness only over active
        # (``gmt_deleted IS NULL``) rows. Phase 8 Task #39 Part D
        # converts the legacy ``UniqueConstraint("name", "gmt_deleted")``
        # which Postgres did not actually enforce because NULL != NULL.
        Index(
            "uq_model_service_provider_name_active",
            "name",
            unique=True,
            postgresql_where=text("gmt_deleted IS NULL"),
        ),
    )

    id = Column(String(24), primary_key=True, default=lambda: "msp" + _random_id())
    name = Column(String(256), nullable=False, index=True)  # Reference to LLMProvider.name
    status = Column(
        _enum_column(ModelServiceProviderStatus), nullable=False, index=True
    )  # Add index for status queries
    api_key = Column(String(256), nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)  # Add index for soft delete queries


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
    "APIType",
    "LLMProvider",
    "LLMProviderModel",
    "ModelServiceProvider",
    "ModelServiceProviderStatus",
    "PromptTemplate",
]
