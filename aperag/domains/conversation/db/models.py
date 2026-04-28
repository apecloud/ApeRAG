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

"""Conversation-domain SQLAlchemy models.

Owns ``Bot`` + ``Chat`` + ``TurnFeedback`` — the three entities the
conversation domain is responsible for — plus their lifecycle /
classification enums (``BotStatus`` / ``BotType`` / ``ChatStatus`` /
``ChatPeerType`` / ``TurnFeedbackType`` / ``TurnFeedbackTag``). Moved
here from ``aperag.db.models`` in Phase 5 step 5-S2 as the first piece
of the Phase 5 DB split; the legacy aggregate module retains a
re-export shim with the full post-Phase-5 symbol list until Phase 6
cleanup.

``Chat`` carries two helper methods (``get_bot`` / ``set_bot``) that
reach the sibling ``Bot`` row; ``TurnFeedback`` has the same pair for
``Chat``. Both are intra-domain lookups, so no cross-domain dep.

Agent turn / event / artifact rows stay in the ``agent_runtime``
domain (Phase 5 canonical Section 3) even though they key on
``chat_id`` — agent_runtime is a separate bounded context with its
own lifecycle and storage_ref semantics.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    String,
    Text,
    text,
)

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id`` so this module does
    not have to import from the G1 strict-ban aggregate. Phase 6 cleanup
    collapses the helper twins (one in each per-domain DB module) onto
    ``aperag.db.base``.
    """

    return "".join(random.sample(uuid.uuid4().hex, 16))


def _enum_column(enum_class):
    """Mirror of ``aperag.db.models.EnumColumn``. Formula:
    ``max(max_value_len + 20, 50)``. Keep in sync until Phase 6
    consolidates these helper twins.
    """

    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class BotStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class BotType(str, Enum):
    KNOWLEDGE = "knowledge"
    COMMON = "common"
    AGENT = "agent"
    # Wave 10 §K.13: hidden per-user bot used by
    # ``collection_regen_service`` to drive Stage 1 agent-runtime
    # free-explore. ``is_system=True`` rows are filtered out of UI
    # listings. Tool subset is hardcoded (13 read-only tools) by the
    # agent runtime layer based on ``bot.type == SUMMARY``.
    SUMMARY = "summary"


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


class TurnFeedbackType(str, Enum):
    GOOD = "good"
    BAD = "bad"


class TurnFeedbackTag(str, Enum):
    HARMFUL = "Harmful"
    UNSAFE = "Unsafe"
    FAKE = "Fake"
    UNHELPFUL = "Unhelpful"
    OTHER = "Other"


class Bot(Base):
    __tablename__ = "bot"
    __table_args__ = (
        # Wave 10 §K.13: at most one ``is_system=True`` bot of a given
        # type per user. Defends against race conditions during
        # register-time creation (main path) and lazy-create fallback
        # (defense-in-depth path). The partial index is over active
        # rows only (``gmt_deleted IS NULL``) so a soft-deleted system
        # bot does not block re-creation.
        Index(
            "uq_bot_user_type_system_active",
            "user",
            "type",
            "is_system",
            unique=True,
            postgresql_where=text("gmt_deleted IS NULL AND is_system = TRUE"),
        ),
    )

    id = Column(String(24), primary_key=True, default=lambda: "bot" + _random_id())
    user = Column(String(256), nullable=False, index=True)
    title = Column(String(256), nullable=True)
    type = Column(_enum_column(BotType), nullable=False, default=BotType.KNOWLEDGE)
    description = Column(Text, nullable=True)
    status = Column(_enum_column(BotStatus), nullable=False, index=True)
    config = Column(Text, nullable=False)
    # Wave 10 §K.13: ``True`` for system-managed bots that are hidden
    # from user-facing UI listings. Mirrors existing
    # ``ApiKey.is_system`` precedent (``governance/db/models.py:98``).
    is_system = Column(Boolean, default=False, nullable=False, index=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)


class Chat(Base):
    __tablename__ = "chat"
    __table_args__ = (
        # Partial unique index: enforce uniqueness only over active
        # (``gmt_deleted IS NULL``) rows. Phase 8 Task #39 Part D
        # converts the legacy ``UniqueConstraint(..., "gmt_deleted")``
        # which Postgres did not actually enforce because NULL != NULL.
        Index(
            "uq_chat_bot_peer_active",
            "bot_id",
            "peer_type",
            "peer_id",
            unique=True,
            postgresql_where=text("gmt_deleted IS NULL"),
        ),
    )

    id = Column(String(24), primary_key=True, default=lambda: "chat" + _random_id())
    user = Column(String(256), nullable=False, index=True)
    peer_type = Column(_enum_column(ChatPeerType), nullable=False, default=ChatPeerType.SYSTEM)
    peer_id = Column(String(256), nullable=True)
    status = Column(_enum_column(ChatStatus), nullable=False, index=True)
    bot_id = Column(String(24), nullable=False, index=True)
    title = Column(String(256), nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)

    async def get_bot(self, session):
        """Get the associated bot object"""
        return await session.get(Bot, self.bot_id)

    async def set_bot(self, bot):
        """Set the bot_id by Bot object or id"""
        if hasattr(bot, "id"):
            self.bot_id = bot.id
        elif isinstance(bot, str):
            self.bot_id = bot


class TurnFeedback(Base):
    __tablename__ = "turn_feedback"

    user = Column(String(256), nullable=False, index=True)
    chat_id = Column(String(24), primary_key=True)
    turn_id = Column(String(256), primary_key=True)
    type = Column(_enum_column(TurnFeedbackType), nullable=False)
    tag = Column(_enum_column(TurnFeedbackTag), nullable=True)
    message = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    async def get_chat(self, session):
        """Get the associated chat object"""
        return await session.get(Chat, self.chat_id)

    async def set_chat(self, chat):
        """Set the chat_id by Chat object or id"""
        if hasattr(chat, "id"):
            self.chat_id = chat.id
        elif isinstance(chat, str):
            self.chat_id = chat
