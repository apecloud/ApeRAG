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

"""Indexing-domain SQLAlchemy models.

This module owns the ``DocumentIndex`` row and the two enums that
describe its lifecycle (``DocumentIndexType`` / ``DocumentIndexStatus``).
The models were moved here from ``aperag.db.models`` in Phase 3 step 2
as part of the knowledge_base / indexing domain split.

``aperag.db.models`` re-exports all three symbols for back-compat until
Phase 6 cleanup, so external callers that have not yet migrated can
continue to ``from aperag.db.models import DocumentIndex`` without a
deprecation warning.
"""

from __future__ import annotations

from enum import Enum

from sqlalchemy import (
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


def _enum_column(enum_class):
    """Mirror of ``aperag.db.models.EnumColumn`` used for enum-backed VARCHAR.

    Reproduced locally so the indexing domain's db module does not have
    to import from ``aperag.db.models`` (G1 strict-ban target). The
    formula — ``max(max_value_len + 20, 50)`` — matches the authoritative
    implementation in ``aperag.db.models`` one-for-one; keep them in sync
    or consolidate into ``aperag.db.base`` during Phase 6 cleanup.
    """

    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    max_length = max(max_length + 20, 50)
    return String(length=max_length)


class DocumentIndexType(str, Enum):
    """Document index type enumeration."""

    VECTOR = "VECTOR"
    FULLTEXT = "FULLTEXT"
    GRAPH = "GRAPH"
    SUMMARY = "SUMMARY"
    VISION = "VISION"


class DocumentIndexStatus(str, Enum):
    """Document index lifecycle status."""

    PENDING = "PENDING"  # Awaiting processing (create/update)
    CREATING = "CREATING"  # Task claimed, creation/update in progress
    ACTIVE = "ACTIVE"  # Index is up-to-date and ready for use
    DELETING = "DELETING"  # Deletion has been requested
    DELETION_IN_PROGRESS = "DELETION_IN_PROGRESS"  # Task claimed, deletion in progress
    FAILED = "FAILED"  # The last operation failed


class DocumentIndex(Base):
    """Document index - single status model."""

    __tablename__ = "document_index"
    __table_args__ = (
        UniqueConstraint("document_id", "index_type", name="uq_document_index"),
        Index("idx_document_index_status_lease", "status", "lease_expires_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(24), nullable=False, index=True)
    index_type = Column(_enum_column(DocumentIndexType), nullable=False, index=True)

    status = Column(_enum_column(DocumentIndexStatus), nullable=False, default=DocumentIndexStatus.PENDING, index=True)
    version = Column(Integer, nullable=False, default=1)  # Incremented on each spec change
    observed_version = Column(Integer, nullable=False, default=0)  # Last processed spec version

    # Index data and task tracking
    index_data = Column(Text, nullable=True)  # JSON string for index-specific data
    error_message = Column(Text, nullable=True)
    processing_token = Column(String(64), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_last_reconciled = Column(DateTime(timezone=True), nullable=True)  # Last reconciliation attempt

    def __repr__(self):
        return f"<DocumentIndex(id={self.id}, document_id={self.document_id}, type={self.index_type}, status={self.status}, version={self.version})>"

    def update_version(self):
        """Update the version to trigger reconciliation."""
        self.version += 1
        self.gmt_updated = utc_now()


__all__ = ["DocumentIndex", "DocumentIndexStatus", "DocumentIndexType"]
