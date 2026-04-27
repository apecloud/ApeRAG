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

"""Governance-domain SQLAlchemy models.

Owns the two entities the governance domain is responsible for:

* ``ApiKey`` — per-user API access tokens.
* ``AuditLog`` — immutable audit trail of mutating HTTP operations.

Plus their classification enums (``ApiKeyStatus`` /
``AuditResource``). Moved here from ``aperag.db.models`` in Phase 4
Step 4-S2b; the legacy aggregate module retains a re-export shim so
pre-migration callers (audit decorator / api_key_service / fastapi
handlers) continue to resolve the same class objects until Phase 6
cleanup.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Index, Integer, String, Text

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id``. Phase 6 cleanup
    consolidates the helper twins onto ``aperag.db.base``."""

    return "".join(random.sample(uuid.uuid4().hex, 16))


def _enum_column(enum_class):
    """Mirror of ``aperag.db.models.EnumColumn``. Phase 6 cleanup
    consolidates."""

    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class ApiKeyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class AuditResource(str, Enum):
    """Audit resource types"""

    COLLECTION = "collection"
    DOCUMENT = "document"
    BOT = "bot"
    CHAT = "chat"
    MESSAGE = "message"
    API_KEY = "api_key"
    LLM_PROVIDER = "llm_provider"
    LLM_PROVIDER_MODEL = "llm_provider_model"
    MODEL_SERVICE_PROVIDER = "model_service_provider"
    USER = "user"
    CONFIG = "config"
    INVITATION = "invitation"
    AUTH = "auth"
    CHAT_COMPLETION = "chat_completion"
    SEARCH = "search"
    LLM = "llm"
    FLOW = "flow"
    SYSTEM = "system"
    INDEX = "index"


class ApiKey(Base):
    __tablename__ = "api_key"

    id = Column(String(24), primary_key=True, default=lambda: "key" + _random_id())
    key = Column(String(64), default=lambda: f"sk-{uuid.uuid4().hex}", nullable=False)
    user = Column(String(256), nullable=False, index=True)  # Add index for user queries
    description = Column(String(256), nullable=True)
    status = Column(_enum_column(ApiKeyStatus), nullable=False, index=True)  # Add index for status queries
    is_system = Column(Boolean, default=False, nullable=False, index=True)  # Mark system-generated API keys
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)  # Add index for soft delete queries

    @staticmethod
    def generate_key() -> str:
        """Generate a unique API key"""
        return f"sk-{uuid.uuid4().hex}"

    async def update_last_used(self, session):
        """Update the last_used_at timestamp"""
        self.last_used_at = utc_now()
        session.add(self)
        await session.commit()


class AuditLog(Base):
    """Audit log model to track all system operations"""

    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True, comment="User ID")
    username = Column(String(255), nullable=True, comment="Username")
    resource_type = Column(_enum_column(AuditResource), nullable=True, comment="Resource type")
    resource_id = Column(String(255), nullable=True, comment="Resource ID (extracted at query time)")
    api_name = Column(String(255), nullable=False, comment="API operation name")
    http_method = Column(String(10), nullable=False, comment="HTTP method (POST, PUT, DELETE)")
    path = Column(String(512), nullable=False, comment="API path")
    status_code = Column(Integer, nullable=True, comment="HTTP status code")
    request_data = Column(Text, nullable=True, comment="Request data (JSON)")
    response_data = Column(Text, nullable=True, comment="Response data (JSON)")
    error_message = Column(Text, nullable=True, comment="Error message if failed")
    ip_address = Column(String(45), nullable=True, comment="Client IP address")
    user_agent = Column(String(500), nullable=True, comment="User agent string")
    request_id = Column(String(255), nullable=False, comment="Request ID for tracking")
    start_time = Column(BigInteger, nullable=False, comment="Request start time (milliseconds since epoch)")
    end_time = Column(BigInteger, nullable=True, comment="Request end time (milliseconds since epoch)")
    gmt_created = Column(DateTime(timezone=True), nullable=False, default=utc_now, comment="Created time")

    # Index for better query performance
    __table_args__ = (
        Index("idx_audit_user_id", "user_id"),
        Index("idx_audit_resource_type", "resource_type"),
        Index("idx_audit_api_name", "api_name"),
        Index("idx_audit_http_method", "http_method"),
        Index("idx_audit_status_code", "status_code"),
        Index("idx_audit_gmt_created", "gmt_created"),
        Index("idx_audit_resource_id", "resource_id"),
        Index("idx_audit_request_id", "request_id"),
        Index("idx_audit_start_time", "start_time"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, user={self.username}, api={self.api_name}, method={self.http_method}, status={self.status_code})>"


__all__ = [
    "ApiKey",
    "ApiKeyStatus",
    "AuditLog",
    "AuditResource",
]
