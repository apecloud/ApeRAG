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

"""``DocumentIndex`` ORM model — celery T1.1 Foundation.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §F.1, this
is the per-(document, parse_version, modality) state row that replaces
the legacy ``aperag/tasks/`` + ``aperag/domains/indexing/`` Celery
orchestration state. The DB-layer partial unique index
``uniq_document_index_serving`` (alembic ``f9c4d2a8e1b5``) enforces
"at most one serving row per (document_id, modality)" — see Bryce v2
review msg=7ccb176f #2.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from aperag.db.base import Base


class Modality(str, enum.Enum):
    """5 supported modalities — design pack §C/§D Idempotency contract.

    Stored as ``VARCHAR`` in the DB (string-mode enum) so a future
    sixth modality can be added without an alembic enum migration.
    """

    VECTOR = "vector"
    FULLTEXT = "fulltext"
    GRAPH = "graph"
    SUMMARY = "summary"
    VISION = "vision"


class IndexStatus(str, enum.Enum):
    """4 lifecycle states per design pack §F.2.

    Orthogonal to ``is_serving`` — a row may be ``ACTIVE`` but not
    serving (cutover swap pending), or ``RUNNING`` while a previous
    row is still serving.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class DocumentIndex(Base):
    """Per-(document, parse_version, modality) state row.

    The schema is intentionally narrow: a state machine triple plus an
    ``is_serving`` flag plus retry / heartbeat metadata. All
    cross-modality coordination is removed (§F.6) — each row's
    cutover is local to its own ``(document_id, modality)`` partial
    unique index slot.
    """

    __tablename__ = "document_index_v2"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=False)
    parse_version: Mapped[str] = mapped_column(String(16), nullable=False)
    modality: Mapped[str] = mapped_column(String(32), nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    retry_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    derived_artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # T2.1 dispatch columns (alembic c2e8d5a1f3b9). collection_id scopes
    # cleanup-worker GC + tenant queries without needing to parse the
    # canonical layout out of source_path; source_path is the modality's
    # ``derive`` input artifact path (chunks.jsonl for vector/fulltext/
    # graph; markdown.md for summary; modality-specific for vision).
    # Both nullable for back-compat with Wave 1 fixtures; the
    # orchestrator skips rows missing source_path (leaves PENDING for
    # the next reconciler cycle).
    collection_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # §H.2 multi-tenant isolation: tenant_scope_key is the rate-limit /
    # quota / bulkhead partition key (e.g. ``"user:<uid>"`` or
    # ``"org:<org_id>"``). Carried on every document_index row so the
    # T2.2 quota lane can throttle per-tenant without a join, and the
    # T2.1 worker pool can isolate noisy neighbours by partition.
    tenant_scope_key: Mapped[str] = mapped_column(String(64), nullable=False)

    is_serving: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "parse_version",
            "modality",
            name="uq_document_index_v2_triple",
        ),
        Index(
            "idx_document_index_v2_status_modality",
            "status",
            "modality",
        ),
        Index(
            "idx_document_index_v2_document_modality",
            "document_id",
            "modality",
        ),
        # §H.2 tenant scope index — used by T2.2 quota / bulkhead
        # partitioning to look up "all in-flight rows for tenant X".
        Index(
            "idx_document_index_v2_tenant_scope",
            "tenant_scope_key",
        ),
        # T2.1 cleanup-worker scoping index (per-collection GC scan).
        Index(
            "idx_document_index_v2_collection",
            "collection_id",
        ),
        # §F.1 partial unique invariant — DB-enforced "at most one
        # serving row per (document_id, modality)".
        Index(
            "uniq_document_index_v2_serving",
            "document_id",
            "modality",
            unique=True,
            postgresql_where=text("is_serving = TRUE"),
            sqlite_where=text("is_serving = TRUE"),
        ),
    )


__all__ = ["DocumentIndex", "Modality", "IndexStatus"]
