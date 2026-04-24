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
from enum import Enum

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)

from aperag.db.base import Base
from aperag.domains.identity.db.models import (  # noqa: F401  Phase 4 Step 4-S2a per-domain re-export
    OAuthAccount,
    Role,
    User,
)
from aperag.utils.utils import utc_now

# ``Base`` is re-exported from ``aperag.db.base`` so existing call sites
# (``from aperag.db.models import Base`` — notably ``aperag/graphindex/models.py``
# and the Alembic ``env.py``) continue to resolve the same declarative base
# during Phase 3's per-domain DB split. See ``aperag/db/base.py`` for the why.


# Helper function for random id generation
def random_id():
    """Generate a random ID string"""
    return "".join(random.sample(uuid.uuid4().hex, 16))


# Helper function for creating enum columns that store values as varchar instead of database enum
def EnumColumn(enum_class, **kwargs):
    """Create a String column for enum values to avoid database enum constraints"""
    # Remove enum-specific kwargs that don't apply to String columns
    kwargs.pop("name", None)

    # Determine the maximum length needed for enum values
    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    # Add some buffer for future enum values
    max_length = max(max_length + 20, 50)

    # Set default length if not specified
    kwargs.setdefault("length", max_length)

    return String(**kwargs)


# Enums for choices
# ``CollectionMarketplaceStatusEnum`` moved to
# ``aperag.domains.marketplace.db.models`` in Phase 4 Step 4-S2c;
# re-exported at the bottom of this module.


class BotStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class BotType(str, Enum):
    KNOWLEDGE = "knowledge"
    COMMON = "common"
    AGENT = "agent"


# ``Role`` moved to ``aperag.domains.identity.db.models`` in Phase 4
# Step 4-S2a; re-exported via the shim block at the bottom of this
# module so pre-migration callers continue to resolve it here.
# Phase 4 G15 canonical forbids cross-domain imports of the enum —
# consumers compare ``user.role == "admin"`` by literal instead.


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


class AgentArtifactType(str, Enum):
    ANSWER = "answer"
    REFERENCE_BUNDLE = "reference_bundle"
    TOOL_RESULT_SUMMARY = "tool_result_summary"
    SEARCH_RESULT_SUMMARY = "search_result_summary"
    ERROR_SUMMARY = "error_summary"


class TurnFeedbackType(str, Enum):
    GOOD = "good"
    BAD = "bad"


class TurnFeedbackTag(str, Enum):
    HARMFUL = "Harmful"
    UNSAFE = "Unsafe"
    FAKE = "Fake"
    UNHELPFUL = "Unhelpful"
    OTHER = "Other"


class ModelServiceProviderStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETED = "DELETED"


# ``ApiKeyStatus`` moved to
# ``aperag.domains.governance.db.models`` in Phase 4 Step 4-S2b;
# re-exported at the bottom of this module.


# ``APIType`` moved to
# ``aperag.domains.model_platform.db.models`` in Phase 4 Step 4-S2d;
# re-exported at the bottom of this module.


class QuestionType(str, Enum):
    """Question type enumeration"""

    FACTUAL = "FACTUAL"
    INFERENTIAL = "INFERENTIAL"
    USER_DEFINED = "USER_DEFINED"


class EvaluationStatus(str, Enum):
    """Evaluation task lifecycle status"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvaluationItemStatus(str, Enum):
    """Evaluation item lifecycle status"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvaluationDatasetSourceType(str, Enum):
    """Origin of an ``EvaluationDataset``.

    The simplified evaluation model exposes only ``Dataset`` + ``Run`` to
    users. ``source_type`` lets the backend distinguish how the items were
    created (manual entry, file import, LLM-generated).
    """

    MANUAL = "manual"
    IMPORT = "import"
    GENERATED = "generated"


class EvaluationRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationRunItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationRunItemAttemptStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationJudgeMode(str, Enum):
    NONE = "none"
    EXACT_MATCH = "exact_match"
    LLM_AS_JUDGE = "llm_as_judge"


# Models
# ``CollectionMarketplace`` + ``UserCollectionSubscription`` moved
# to ``aperag.domains.marketplace.db.models`` in Phase 4 Step 4-S2c;
# re-exported at the bottom of this module.


class Bot(Base):
    __tablename__ = "bot"

    id = Column(String(24), primary_key=True, default=lambda: "bot" + random_id())
    user = Column(String(256), nullable=False, index=True)  # Add index for user queries
    title = Column(String(256), nullable=True)
    type = Column(EnumColumn(BotType), nullable=False, default=BotType.KNOWLEDGE)
    description = Column(Text, nullable=True)
    status = Column(EnumColumn(BotStatus), nullable=False, index=True)  # Add index for status queries
    config = Column(Text, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)  # Add index for soft delete queries


class ConfigModel(Base):
    __tablename__ = "config"

    key = Column(String(256), primary_key=True)
    value = Column(Text, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)


class UserQuota(Base):
    __tablename__ = "user_quota"

    user = Column(String(256), primary_key=True)
    key = Column(String(256), primary_key=True)
    quota_limit = Column(Integer, default=0, nullable=False)  # Renamed from 'value' for clarity
    current_usage = Column(Integer, default=0, nullable=False)  # New field to track current usage
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def is_quota_exceeded(self, additional_usage: int = 1) -> bool:
        """Check if adding additional usage would exceed the quota limit"""
        return (self.current_usage + additional_usage) > self.quota_limit

    def can_consume(self, amount: int = 1) -> bool:
        """Check if the specified amount can be consumed without exceeding quota"""
        return not self.is_quota_exceeded(amount)


class Chat(Base):
    __tablename__ = "chat"
    __table_args__ = (
        UniqueConstraint("bot_id", "peer_type", "peer_id", "gmt_deleted", name="uq_chat_bot_peer_deleted"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "chat" + random_id())
    user = Column(String(256), nullable=False, index=True)  # Add index for user queries
    peer_type = Column(EnumColumn(ChatPeerType), nullable=False, default=ChatPeerType.SYSTEM)
    peer_id = Column(String(256), nullable=True)
    status = Column(EnumColumn(ChatStatus), nullable=False, index=True)  # Add index for status queries
    bot_id = Column(String(24), nullable=False, index=True)  # Add index for bot queries
    title = Column(String(256), nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)  # Add index for soft delete queries

    async def get_bot(self, session):
        """Get the associated bot object"""
        return await session.get(Bot, self.bot_id)

    async def set_bot(self, bot):
        """Set the bot_id by Bot object or id"""
        if hasattr(bot, "id"):
            self.bot_id = bot.id
        elif isinstance(bot, str):
            self.bot_id = bot


class AgentTurn(Base):
    __tablename__ = "agent_turn"
    __table_args__ = (
        UniqueConstraint("chat_id", "client_idempotency_key", name="uq_agent_turn_chat_idempotency"),
        Index("idx_agent_turn_chat_created", "chat_id", "gmt_created"),
        Index("idx_agent_turn_user_status", "user", "status"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "turn" + random_id())
    chat_id = Column(String(24), nullable=False, index=True)
    user = Column(String(256), nullable=False, index=True)
    bot_id = Column(String(24), nullable=False, index=True)
    request_id = Column(String(64), nullable=False, unique=True, index=True)
    client_idempotency_key = Column(String(128), nullable=False)
    status = Column(EnumColumn(AgentTurnStatus), nullable=False, default=AgentTurnStatus.QUEUED, index=True)
    input_text = Column(Text, nullable=False)
    model_profile = Column(JSON, default=lambda: {}, nullable=False)
    error_code = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    answer_artifact_id = Column(String(24), nullable=True, index=True)
    reference_bundle_artifact_id = Column(String(24), nullable=True, index=True)
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

    id = Column(String(24), primary_key=True, default=lambda: "evt" + random_id())
    turn_id = Column(String(24), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    type = Column(String(128), nullable=False, index=True)
    label = Column(String(128), nullable=True)
    status = Column(String(64), nullable=True)
    actor = Column(EnumColumn(AgentEventActor), nullable=False, default=AgentEventActor.SYSTEM)
    data = Column(JSON, default=lambda: {}, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentArtifact(Base):
    __tablename__ = "agent_artifact"
    __table_args__ = (Index("idx_agent_artifact_turn_type", "turn_id", "artifact_type"),)

    id = Column(String(24), primary_key=True, default=lambda: "art" + random_id())
    turn_id = Column(String(24), nullable=False, index=True)
    artifact_type = Column(EnumColumn(AgentArtifactType), nullable=False, index=True)
    summary = Column(Text, nullable=True)
    payload = Column(JSON, default=lambda: {}, nullable=False)
    storage_ref = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class TurnFeedback(Base):
    __tablename__ = "turn_feedback"

    user = Column(String(256), nullable=False, index=True)
    chat_id = Column(String(24), primary_key=True)
    turn_id = Column(String(256), primary_key=True)
    type = Column(EnumColumn(TurnFeedbackType), nullable=False)
    tag = Column(EnumColumn(TurnFeedbackTag), nullable=True)
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


# ``ApiKey`` moved to ``aperag.domains.governance.db.models`` in
# Phase 4 Step 4-S2b; re-exported at the bottom of this module.


class ModelServiceProvider(Base):
    __tablename__ = "model_service_provider"
    __table_args__ = (UniqueConstraint("name", "gmt_deleted", name="uq_model_service_provider_name_deleted"),)

    id = Column(String(24), primary_key=True, default=lambda: "msp" + random_id())
    name = Column(String(256), nullable=False, index=True)  # Reference to LLMProvider.name
    status = Column(EnumColumn(ModelServiceProviderStatus), nullable=False, index=True)  # Add index for status queries
    api_key = Column(String(256), nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)  # Add index for soft delete queries


# ``LLMProvider`` + ``LLMProviderModel`` moved to
# ``aperag.domains.model_platform.db.models`` in Phase 4 Step 4-S2d;
# re-exported at the bottom of this module. ``APIType`` moved with
# them (used only by ``LLMProviderModel.api``).


# ``User`` + ``OAuthAccount`` moved to
# ``aperag.domains.identity.db.models`` in Phase 4 Step 4-S2a; the
# re-export shim at the bottom of this module keeps pre-migration
# ``from aperag.db.models import User`` callers working.


class Invitation(Base):
    __tablename__ = "invitation"

    id = Column(String(24), primary_key=True, default=lambda: "invite" + random_id())
    email = Column(String(254), nullable=False)
    token = Column(String(64), unique=True, nullable=False)
    created_by = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)
    role = Column(EnumColumn(Role), nullable=False, default=Role.RO)

    def is_valid(self) -> bool:
        """Check if invitation is still valid"""
        now = utc_now()
        return not self.is_used and now < self.expires_at

    async def use(self, session):
        """Mark invitation as used"""
        self.is_used = True
        self.used_at = utc_now()
        session.add(self)
        await session.commit()

        # Auto-expire after use (optional)
        # self.expires_at = utc_now()


# ``AuditResource`` + ``AuditLog`` moved to
# ``aperag.domains.governance.db.models`` in Phase 4 Step 4-S2b;
# re-exported at the bottom of this module.


class QuestionSet(Base):
    __tablename__ = "question_sets"
    __table_args__ = (
        Index("idx_question_sets_user_id", "user_id"),
        Index("idx_question_sets_collection_id", "collection_id"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "qs_" + random_id()[:16])
    user_id = Column(String(24), nullable=False)
    collection_id = Column(String(24), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<QuestionSet(id={self.id}, name={self.name}, user_id={self.user_id})>"


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (Index("idx_questions_question_set_id", "question_set_id"),)

    id = Column(String(24), primary_key=True, default=lambda: "q_" + random_id()[:16])
    question_set_id = Column(String(24), nullable=False)
    question_type = Column(EnumColumn(QuestionType), nullable=True)
    question_text = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Question(id={self.id}, qs_id={self.question_set_id})>"


class Evaluation(Base):
    __tablename__ = "evaluations"
    __table_args__ = (
        Index("idx_evaluations_user_id", "user_id"),
        Index("idx_evaluations_status", "status"),
        Index("idx_evaluations_collection_id", "collection_id"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "eval_" + random_id()[:16])
    user_id = Column(String(24), nullable=False)
    name = Column(String(255), nullable=False)
    collection_id = Column(String(24), nullable=False)
    question_set_id = Column(String(24), nullable=False)
    agent_llm_config = Column(JSON, nullable=False)
    judge_llm_config = Column(JSON, nullable=False)
    status = Column(EnumColumn(EvaluationStatus), nullable=False, default=EvaluationStatus.PENDING)
    error_message = Column(Text, nullable=True)
    total_questions = Column(Integer, nullable=False, default=0)
    completed_questions = Column(Integer, nullable=False, default=0)
    average_score = Column(Numeric(3, 2), nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Evaluation(id={self.id}, name={self.name}, status={self.status})>"


class EvaluationItem(Base):
    __tablename__ = "evaluation_items"
    __table_args__ = (Index("idx_evaluation_items_evaluation_id", "evaluation_id"),)

    id = Column(String(24), primary_key=True, default=lambda: "item_" + random_id()[:16])
    evaluation_id = Column(String(24), nullable=False)
    question_id = Column(String(24), nullable=True)
    status = Column(EnumColumn(EvaluationItemStatus), nullable=False, default=EvaluationItemStatus.PENDING, index=True)
    question_text = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    rag_answer = Column(Text, nullable=True)
    rag_answer_details = Column(JSON, nullable=True)
    llm_judge_score = Column(Integer, nullable=True)
    llm_judge_reasoning = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    def __repr__(self):
        return f"<EvaluationItem(id={self.id}, eval_id={self.evaluation_id}, q_id={self.question_id})>"


class Setting(Base):
    __tablename__ = "setting"

    key = Column(String(256), primary_key=True)
    value = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)


class ExportTaskStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class ExportTask(Base):
    __tablename__ = "export_task"
    __table_args__ = (
        Index("idx_export_task_user_status", "user", "status"),
        Index("idx_export_task_expires", "status", "gmt_expires"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "export" + random_id()[:16])
    user = Column(String(256), nullable=False, index=True)
    collection_id = Column(String(24), nullable=False, index=True)

    status = Column(EnumColumn(ExportTaskStatus), nullable=False, default=ExportTaskStatus.PENDING)
    progress = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    object_store_path = Column(Text, nullable=True)
    file_size = Column(BigInteger, nullable=True)

    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_completed = Column(DateTime(timezone=True), nullable=True)
    gmt_expires = Column(DateTime(timezone=True), nullable=True)


class PromptTemplate(Base):
    __tablename__ = "prompt_template"

    id = Column(String(24), primary_key=True, default=lambda: "pt" + random_id())
    prompt_type = Column(String(50), nullable=False, index=True)
    scope = Column(String(20), nullable=False, index=True)
    user_id = Column(String(256), nullable=True, index=True)
    content = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)


# ===== Evaluation (simplified: Dataset + Run model, no Benchmark/Version layer) =====


class EvaluationDataset(Base):
    """Dataset of evaluation QA items owned by a user.

    ``user_id`` anchors access control; ``collection_id`` is scope metadata
    only and does NOT inherit collection sharing ACLs.
    """

    __tablename__ = "evaluation_datasets"
    __table_args__ = (
        Index("idx_evaluation_datasets_user", "user_id"),
        Index("idx_evaluation_datasets_collection", "collection_id"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "eds_" + random_id()[:16])
    user_id = Column(String(256), nullable=False)
    collection_id = Column(String(24), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(
        EnumColumn(EvaluationDatasetSourceType),
        nullable=False,
        default=EvaluationDatasetSourceType.MANUAL,
    )
    schema_hint = Column(JSON, nullable=True)
    item_count = Column(Integer, nullable=False, default=0)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)


class EvaluationDatasetItem(Base):
    """A single QA item inside an ``EvaluationDataset``.

    At run-create time these fields are value-copied into
    ``evaluation_run_items`` (snapshot-on-run) so later edits or soft-deletes
    of the dataset item do not mutate historical run semantics.
    """

    __tablename__ = "evaluation_dataset_items"
    __table_args__ = (
        UniqueConstraint("dataset_id", "case_key", name="uq_evaluation_dataset_item_case_key"),
        Index("idx_evaluation_dataset_items_dataset", "dataset_id"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "edi_" + random_id()[:16])
    dataset_id = Column(String(32), nullable=False)
    case_key = Column(String(128), nullable=False)
    input_message = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    reference_context = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    case_metadata = Column(JSON, nullable=True)
    sort_key = Column(Integer, nullable=False, default=0)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("idx_evaluation_runs_user", "user_id"),
        Index("idx_evaluation_runs_bot", "bot_id"),
        Index("idx_evaluation_runs_status", "status"),
        Index("idx_evaluation_runs_dataset", "dataset_id"),
        Index("idx_evaluation_runs_collection", "collection_id"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "er_" + random_id()[:16])
    user_id = Column(String(256), nullable=False)
    bot_id = Column(String(24), nullable=False)  # resolved at create-time, immutable
    dataset_id = Column(String(32), nullable=False)
    # Snapshot of dataset.collection_id so run list can filter by collection scope
    # even after the dataset is soft-deleted or its collection_id is updated.
    collection_id = Column(String(24), nullable=True)
    # Snapshot of dataset.name for run list/detail UIs — avoids joining back to
    # evaluation_datasets when the dataset has been soft-deleted.
    dataset_name = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    bot_config_snapshot = Column(JSON, nullable=True)
    model_config_snapshot = Column(JSON, nullable=True)
    judge_config = Column(JSON, nullable=True)
    status = Column(
        EnumColumn(EvaluationRunStatus),
        nullable=False,
        default=EvaluationRunStatus.QUEUED,
    )
    summary = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    gmt_started = Column(DateTime(timezone=True), nullable=True)
    gmt_finished = Column(DateTime(timezone=True), nullable=True)


class EvaluationRunItem(Base):
    __tablename__ = "evaluation_run_items"
    __table_args__ = (
        Index("idx_evaluation_run_items_run", "run_id"),
        Index("idx_evaluation_run_items_status", "status"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "eri_" + random_id()[:16])
    run_id = Column(String(32), nullable=False)
    # Audit-only back-reference to the source dataset item. Not an FK; soft- or
    # hard-delete of the dataset item must NOT cascade into historical run items.
    source_dataset_item_id = Column(String(32), nullable=True)
    case_key = Column(String(128), nullable=False)
    sort_key = Column(Integer, nullable=False, default=0)
    # Snapshot fields are value-copied from the dataset item at run-create time.
    # Run detail / list / attempt read paths must only read these columns and
    # must never join back to mutable evaluation_dataset_items rows.
    input_message = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    reference_context = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    case_metadata = Column(JSON, nullable=True)
    status = Column(
        EnumColumn(EvaluationRunItemStatus),
        nullable=False,
        default=EvaluationRunItemStatus.PENDING,
    )
    best_score = Column(Numeric(6, 3), nullable=True)
    latest_attempt_id = Column(String(32), nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class EvaluationRunItemAttempt(Base):
    __tablename__ = "evaluation_run_item_attempts"
    __table_args__ = (
        UniqueConstraint("run_item_id", "attempt_no", name="uq_evaluation_run_item_attempt_no"),
        Index("idx_evaluation_run_item_attempts_item", "run_item_id"),
        Index("idx_evaluation_run_item_attempts_run", "run_id"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "era_" + random_id()[:16])
    run_item_id = Column(String(32), nullable=False)
    run_id = Column(String(32), nullable=False)
    attempt_no = Column(Integer, nullable=False)
    status = Column(
        EnumColumn(EvaluationRunItemAttemptStatus),
        nullable=False,
        default=EvaluationRunItemAttemptStatus.QUEUED,
    )
    agent_chat_id = Column(String(24), nullable=True)
    agent_turn_id = Column(String(24), nullable=True)
    answer_text = Column(Text, nullable=True)
    judge_result = Column(JSON, nullable=True)
    score = Column(Numeric(6, 3), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    token_usage = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_reason = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_started = Column(DateTime(timezone=True), nullable=True)
    gmt_finished = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Phase 3 per-domain DB re-exports (decision D — msg=02acb01a + msg=226b2584)
#
# Physical owners of these classes have moved into the new domain DB modules
# under ``aperag/domains/<domain>/db/models.py``. ``aperag.db.models`` keeps
# re-exporting them so the 76 pre-Phase-3 callers — plus the Alembic
# ``env.py`` metadata registration — continue to work without a rename
# sweep. The full 7 DB + 8 enum = 15 symbol list locks in at the end of
# Step 4 (see Phase 3 design-lock G11 exact class-name audit). Step 2
# delivers the first 3 symbols from the ``indexing`` domain.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 4 per-domain DB re-exports (canonical @符炫炜 msg=d47fa490
# Section 1 4-S2 + msg=896584ee). G11 extends from Phase 3 15 symbols
# toward the Phase 4 ~28-symbol final. Step 4-S2a adds the three
# identity-domain symbols; subsequent sub-commits add governance
# (ApiKey / AuditLog / AuditResource), marketplace
# (CollectionMarketplace / UserCollectionSubscription), and
# model_platform (LLMProvider / LLMProviderModel).
# ---------------------------------------------------------------------------
from aperag.domains.governance.db.models import (  # noqa: E402, F401  re-export for back-compat (Phase 4 Step 4-S2b)
    ApiKey,
    ApiKeyStatus,
    AuditLog,
    AuditResource,
)
from aperag.domains.indexing.db.models import (  # noqa: E402, F401  re-export for back-compat
    DocumentIndex,
    DocumentIndexStatus,
    DocumentIndexType,
)
from aperag.domains.knowledge_base.db.models import (  # noqa: E402, F401  re-export for back-compat
    Collection,
    CollectionStatus,
    CollectionSummary,
    CollectionSummaryStatus,
    CollectionType,
    Document,
    DocumentStatus,
)
from aperag.domains.knowledge_graph.db.models import (  # noqa: E402, F401  re-export for back-compat
    GraphCurationRun,
    GraphCurationRunStatus,
    GraphCurationSuggestion,
    GraphCurationSuggestionStatus,
)
from aperag.domains.marketplace.db.models import (  # noqa: E402, F401  re-export for back-compat (Phase 4 Step 4-S2c)
    CollectionMarketplace,
    CollectionMarketplaceStatusEnum,
    UserCollectionSubscription,
)
from aperag.domains.model_platform.db.models import (  # noqa: E402, F401  re-export for back-compat (Phase 4 Step 4-S2d)
    APIType,
    LLMProvider,
    LLMProviderModel,
)
from aperag.domains.retrieval.db.models import SearchHistory  # noqa: E402, F401  re-export for back-compat
