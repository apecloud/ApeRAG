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

"""Knowledge-graph-domain SQLAlchemy models.

Owns ``GraphCurationRun`` / ``GraphCurationSuggestion`` (the curation
audit rows produced by the graph-entity reviewer) plus
their lifecycle enums. The table bodies were moved here from
``aperag.db.models`` in Phase 3 step 3 as part of the knowledge_graph
domain split; the legacy aggregate module re-exports the three class
names + two enums so the 76 pre-Phase-3 callers keep working until
Phase 6 cleanup.
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
    Numeric,
    String,
    Text,
)

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id`` to keep this domain's
    DB module free of the G1 strict-ban target list. Phase 6 cleanup
    should collapse this + the indexing domain's ``_enum_column`` twin
    into a shared helper on ``aperag.db.base``.
    """

    return "".join(random.sample(uuid.uuid4().hex, 16))


def _enum_column(enum_class):
    """Mirror of ``aperag.db.models.EnumColumn`` kept in-domain. Formula:
    ``max(max_value_len + 20, 50)``. Keep in sync with the authoritative
    implementation until Phase 6 consolidates them.
    """

    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class GraphCurationRunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GraphCurationSuggestionStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


class GraphCurationRun(Base):
    __tablename__ = "graph_curation_runs"
    __table_args__ = (
        Index("idx_graph_curation_runs_collection_status", "collection_id", "status"),
        Index("idx_graph_curation_runs_user_created", "user_id", "gmt_created"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "gcr_" + _random_id()[:16])
    user_id = Column(String(256), nullable=False)
    collection_id = Column(String(24), nullable=False)
    status = Column(
        _enum_column(GraphCurationRunStatus),
        nullable=False,
        default=GraphCurationRunStatus.PENDING,
    )
    config_json = Column(JSON, nullable=True)
    stats = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    gmt_started = Column(DateTime(timezone=True), nullable=True)
    gmt_finished = Column(DateTime(timezone=True), nullable=True)


class GraphCurationSuggestion(Base):
    __tablename__ = "graph_curation_suggestions"
    __table_args__ = (
        Index("idx_graph_curation_suggestions_run", "run_id"),
        Index("idx_graph_curation_suggestions_collection_status", "collection_id", "status"),
        Index("idx_graph_curation_suggestions_collection_created", "collection_id", "gmt_created"),
    )

    id = Column(String(32), primary_key=True, default=lambda: "gcs_" + _random_id()[:16])
    run_id = Column(String(32), nullable=False)
    user_id = Column(String(256), nullable=False)
    collection_id = Column(String(24), nullable=False)
    status = Column(
        _enum_column(GraphCurationSuggestionStatus),
        nullable=False,
        default=GraphCurationSuggestionStatus.PENDING,
    )
    entity_ids = Column(JSON, nullable=False)
    entity_snapshots = Column(JSON, nullable=False)
    target_entity_id = Column(String(255), nullable=False)
    confidence_score = Column(Numeric(6, 3), nullable=False)
    reason = Column(Text, nullable=False)
    evidence = Column(JSON, nullable=True)
    resolution_note = Column(Text, nullable=True)
    operated_by = Column(String(256), nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    gmt_operated = Column(DateTime(timezone=True), nullable=True)


class LineageEntityAlias(Base):
    """Persistent record of a user-driven entity merge — Wave 7 §K.12.7
    + §K.12.10b. ``(collection_id, alias_name)`` is unique; the row
    points the alias at the (already-flattened) ``canonical_name``.

    Survives the canonical entity's GC: a future indexer write to the
    alias name still resolves to the (now-empty) canonical, preserving
    user intent (spec §K.12.7 decision X)."""

    __tablename__ = "aperag_lineage_entity_alias"
    __table_args__ = (
        Index(
            "ix_aperag_lineage_entity_alias_canonical",
            "collection_id",
            "canonical_name",
        ),
    )

    collection_id = Column(String(64), primary_key=True, nullable=False)
    alias_name = Column(String(512), primary_key=True, nullable=False)
    canonical_name = Column(String(512), nullable=False)
    merged_by = Column(String(256), nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


__all__ = [
    "GraphCurationRun",
    "GraphCurationRunStatus",
    "GraphCurationSuggestion",
    "GraphCurationSuggestionStatus",
    "LineageEntityAlias",
]
