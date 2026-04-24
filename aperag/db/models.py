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


# ``Role`` moved to ``aperag.domains.identity.db.models`` in Phase 4
# Step 4-S2a; re-exported via the shim block at the bottom of this
# module so pre-migration callers continue to resolve it here.
# Phase 4 G15 canonical forbids cross-domain imports of the enum —
# consumers compare ``user.role == "admin"`` by literal instead.
#
# ``BotStatus`` / ``BotType`` moved to
# ``aperag.domains.conversation.db.models`` in Phase 5 Step 5-S2a;
# re-exported via the shim block at the bottom of this module.


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


# Models
# ``CollectionMarketplace`` + ``UserCollectionSubscription`` moved
# to ``aperag.domains.marketplace.db.models`` in Phase 4 Step 4-S2c;
# re-exported at the bottom of this module.


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


# ``ApiKey`` moved to ``aperag.domains.governance.db.models`` in
# Phase 4 Step 4-S2b; re-exported at the bottom of this module.
#
# ``Chat`` + ``AgentTurn`` + ``AgentTimelineEvent`` + ``AgentArtifact``
# + ``TurnFeedback`` moved to their respective conversation /
# agent_runtime domain ``db/models.py`` in Phase 5 Step 5-S2a / 5-S2b;
# re-exported at the bottom of this module.


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


# ---------------------------------------------------------------------------
# Per-domain DB re-exports (Phase 3 decision D / msg=02acb01a +
# msg=226b2584; extended by Phase 5 step 5-S2 into the conversation /
# agent_runtime / evaluation domains).
#
# Physical owners of these classes have moved into
# ``aperag/domains/<domain>/db/models.py``. The aggregate module keeps
# re-exporting them so the pre-refactor caller base — plus the Alembic
# ``env.py`` metadata registration — works without a rename sweep. The
# full symbol list locks in at G11 (Phase 3 end-of-step-4: 15 symbols;
# Phase 5 extends it with the Phase 5 DB split: conversation adds 9 in
# 5-S2a). Phase 6 cleanup deletes this block once every remaining
# import site has migrated to the canonical per-domain path.
# ---------------------------------------------------------------------------

from aperag.domains.agent_runtime.db.models import (  # noqa: E402, F401  re-export for back-compat (Phase 5 Step 5-S2b)
    AgentArtifact,
    AgentArtifactType,
    AgentEventActor,
    AgentTimelineEvent,
    AgentTurn,
    AgentTurnStatus,
)
from aperag.domains.conversation.db.models import (  # noqa: E402, F401  re-export for back-compat (Phase 5 Step 5-S2a)
    Bot,
    BotStatus,
    BotType,
    Chat,
    ChatPeerType,
    ChatStatus,
    TurnFeedback,
    TurnFeedbackTag,
    TurnFeedbackType,
)
from aperag.domains.evaluation.db.models import (  # noqa: E402, F401  re-export for back-compat (Phase 5 Step 5-S2c)
    EvaluationDataset,
    EvaluationDatasetItem,
    EvaluationDatasetSourceType,
    EvaluationJudgeMode,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunItemAttempt,
    EvaluationRunItemAttemptStatus,
    EvaluationRunItemStatus,
    EvaluationRunStatus,
)
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
