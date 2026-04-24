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

"""Evaluation-domain SQLAlchemy models.

Owns the ``evaluation_v2`` dataset / run pipeline — five tables with
their four status enums plus ``EvaluationDatasetSourceType`` /
``EvaluationJudgeMode``. Moved here from ``aperag.db.models`` in
Phase 5 step 5-S2c; the legacy aggregate retains a re-export shim.

Semantics worth knowing while editing:

* ``EvaluationRun`` / ``EvaluationRunItem`` / ``EvaluationRunItemAttempt``
  form the snapshot-on-run audit trail — dataset edits must never
  mutate historical run data, so the item rows carry value-copied
  ``input_message`` / ``expected_answer`` columns rather than joining
  back to the mutable dataset rows.
* ``EvaluationRunStatus.is_terminal()`` is the canonical terminal-
  status predicate — task #23 used a module-level ``_TERMINAL_RUN_STATUSES``
  frozenset to close the cancel→running TOCTOU race; Phase 6 promoted
  it onto the enum so the check travels with the type.

Legacy v1 tables (``Evaluation`` / ``EvaluationItem`` / ``QuestionSet`` /
``Question``) are **not** moved here — they belong to an older
pre-v2 evaluation flow that Phase 6 cleanup will scrub along with
the rest of the re-export block.
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
    Numeric,
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

    @classmethod
    def is_terminal(cls, status) -> bool:
        # ``EvaluationRun.status`` is persisted as a plain ``String`` column
        # (see ``_enum_column``) so at read time ``status`` may be either an
        # enum member or its raw string value — both compare equal to the
        # enum members because ``EvaluationRunStatus`` inherits from ``str``.
        return status in (cls.COMPLETED, cls.FAILED, cls.CANCELLED)


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

    id = Column(String(32), primary_key=True, default=lambda: "eds_" + _random_id()[:16])
    user_id = Column(String(256), nullable=False)
    collection_id = Column(String(24), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(
        _enum_column(EvaluationDatasetSourceType),
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

    id = Column(String(32), primary_key=True, default=lambda: "edi_" + _random_id()[:16])
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

    id = Column(String(32), primary_key=True, default=lambda: "er_" + _random_id()[:16])
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
        _enum_column(EvaluationRunStatus),
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

    id = Column(String(32), primary_key=True, default=lambda: "eri_" + _random_id()[:16])
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
        _enum_column(EvaluationRunItemStatus),
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

    id = Column(String(32), primary_key=True, default=lambda: "era_" + _random_id()[:16])
    run_item_id = Column(String(32), nullable=False)
    run_id = Column(String(32), nullable=False)
    attempt_no = Column(Integer, nullable=False)
    status = Column(
        _enum_column(EvaluationRunItemAttemptStatus),
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
