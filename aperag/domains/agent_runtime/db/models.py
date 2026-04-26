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

"""Agent-runtime-domain SQLAlchemy models.

Owns ``AgentTurn`` + ``AgentTimelineEvent`` + ``AgentMessage`` plus
``AgentTurnStatus`` / ``AgentEventActor`` enums. Moved here from
``aperag.db.models`` in Phase 5 step 5-S2b; the legacy aggregate
module retains a re-export shim until Phase 6 cleanup.

Phase 8 D8.6 (#80) chunk-2 hard-cut removed the legacy
``AgentArtifact`` table and the ``AgentArtifactType`` enum — at-rest
answer/citation persistence is now canonical via ``AgentMessage``
(D8.2 #74) populated by the runtime emit path.

Tables key on ``turn_id`` (events + messages) or ``chat_id`` (turn);
these are opaque string FKs, so no Python-level cross-domain import
is required. ``chat_id`` is logically owned by the ``conversation``
domain but the relationship is traversed at query time via the turn
service, not via ORM relationship() bindings.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id`` so this module does
    not have to import from the G1 strict-ban aggregate. Phase 6 cleanup
    collapses the helper twins onto ``aperag.db.base``.
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


class AgentTurnStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentEventActor(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


class AgentTurn(Base):
    __tablename__ = "agent_turn"
    __table_args__ = (
        UniqueConstraint("chat_id", "client_idempotency_key", name="uq_agent_turn_chat_idempotency"),
        Index("idx_agent_turn_chat_created", "chat_id", "gmt_created"),
        Index("idx_agent_turn_user_status", "user", "status"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "turn" + _random_id())
    chat_id = Column(String(24), nullable=False, index=True)
    user = Column(String(256), nullable=False, index=True)
    bot_id = Column(String(24), nullable=False, index=True)
    request_id = Column(String(64), nullable=False, unique=True, index=True)
    client_idempotency_key = Column(String(128), nullable=False)
    status = Column(_enum_column(AgentTurnStatus), nullable=False, default=AgentTurnStatus.QUEUED, index=True)
    input_text = Column(Text, nullable=False)
    model_profile = Column(JSON, default=lambda: {}, nullable=False)
    error_code = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    timeline_cursor = Column(Integer, default=0, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_started = Column(DateTime(timezone=True), nullable=True)
    gmt_finished = Column(DateTime(timezone=True), nullable=True)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentTimelineEvent(Base):
    __tablename__ = "agent_timeline_event"
    __table_args__ = (
        UniqueConstraint("turn_id", "sequence", name="uq_agent_timeline_event_turn_sequence"),
        Index("idx_agent_timeline_event_turn_timestamp", "turn_id", "timestamp"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "evt" + _random_id())
    turn_id = Column(String(24), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    type = Column(String(128), nullable=False, index=True)
    label = Column(String(128), nullable=True)
    status = Column(String(64), nullable=True)
    actor = Column(_enum_column(AgentEventActor), nullable=False, default=AgentEventActor.SYSTEM)
    data = Column(JSON, default=lambda: {}, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentMessage(Base):
    """At-rest UIMessage envelope (Phase 8 D8.2 first-cut, D8.5-BE refined).

    One row per assistant turn (1:1 with ``AgentTurn`` for now via
    ``turn_id``). ``parts`` holds the JSON-serialised
    ``UIMessagePart`` array per the canonical schema in
    ``aperag.domains.agent_runtime.uimessage``; ``role`` follows
    ChatML (``user`` / ``assistant`` / ``system``); ``schema_version``
    is the runtime contract version tag so future renderer updates
    can branch without a separate negotiation.

    ``runtime_kind`` (D8.5-BE / #92) tags the runtime that produced the
    message — ``agent_runtime`` for the agent reasoning loop (D8.x
    Phase A), and a forward-compat enum for direct LLM (``direct_chat``)
    or RAG-only (``rag_chat``) paths. ``role`` retains its speaker
    semantics independent of runtime origin per Weston msg=94dac98a /
    architect canonical lock msg=e01e9b4b.

    Phase 8 D8.6 (#80) chunk-2 dropped the legacy ``AgentArtifact``
    table and column FKs from ``AgentTurn``; ``AgentMessage`` is now
    the single authoritative at-rest store for assistant turn
    contents. ``AgentTimelineEvent`` remains pending chunk-3 cleanup
    (replay/reload semantic change is reviewed separately).
    """

    __tablename__ = "agent_message"
    __table_args__ = (
        UniqueConstraint("turn_id", name="uq_agent_message_turn"),
        Index("idx_agent_message_chat_created", "chat_id", "gmt_created"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "msg" + _random_id())
    turn_id = Column(String(24), nullable=False, index=True)
    chat_id = Column(String(24), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    runtime_kind = Column(String(24), nullable=False, default="agent_runtime", server_default="agent_runtime")
    schema_version = Column(String(64), nullable=False)
    parts = Column(JSON, default=lambda: [], nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
