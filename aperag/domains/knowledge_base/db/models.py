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

"""Knowledge-base-domain SQLAlchemy models.

Owns ``Collection`` + ``CollectionSummary`` + ``Document`` — the three
entities the knowledge_base domain is responsible for — plus their
lifecycle / classification enums (``CollectionStatus`` /
``CollectionSummaryStatus`` / ``CollectionType`` / ``DocumentStatus``).
Moved here from ``aperag.db.models`` in Phase 3 step 4 as the final
piece of the Phase 3 DB split; the legacy aggregate module retains a
re-export shim with the full 15-symbol Phase 3 list until Phase 6
cleanup.

``Document`` carries two helper methods (``get_document_indexes`` /
``get_overall_index_status``) that query the ``DocumentIndex`` table.
Because indexing is now its own domain those imports cross a domain
boundary, but ``aperag.domains.indexing.db.models`` is outside the
three LEGACY_AGGREGATE_MODULES (``aperag.db.models`` /
``aperag.service`` / ``aperag.schema.view_models``), so the direct
import here is allowed by G1. Phase 5 or 6 can split the helpers into
an indexing-side service if we want a stricter consumer-owned
Protocol shape — non-blocking.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
    text,
)

from aperag.db.base import Base
from aperag.domains.indexing.db.models import DocumentIndex, DocumentIndexStatus
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


class CollectionStatus(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class CollectionSummaryStatus(str, Enum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class CollectionType(str, Enum):
    DOCUMENT = "document"
    CHAT = "CHAT"


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    DELETED = "DELETED"


class Collection(Base):
    __tablename__ = "collection"

    id = Column(String(24), primary_key=True, default=lambda: "col" + _random_id())
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    user = Column(String(256), nullable=False, index=True)
    status = Column(_enum_column(CollectionStatus), nullable=False, index=True)
    type = Column(_enum_column(CollectionType), nullable=False)
    config = Column(Text, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)


class CollectionSummary(Base):
    __tablename__ = "collection_summary"
    __table_args__ = (
        UniqueConstraint("collection_id", name="uq_collection_summary"),
        Index("idx_collection_summary_status_lease", "status", "lease_expires_at"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "cs" + _random_id())
    collection_id = Column(String(24), nullable=False, index=True)

    status = Column(
        _enum_column(CollectionSummaryStatus), nullable=False, default=CollectionSummaryStatus.PENDING, index=True
    )
    version = Column(Integer, nullable=False, default=1)
    observed_version = Column(Integer, nullable=False, default=0)

    summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_token = Column(String(64), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)

    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_last_reconciled = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<CollectionSummary(id={self.id}, collection_id={self.collection_id}, status={self.status}, version={self.version})>"

    def update_version(self):
        self.version += 1
        self.gmt_updated = utc_now()


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        Index(
            "uq_document_collection_name_active",
            "collection_id",
            "name",
            unique=True,
            postgresql_where=text("gmt_deleted IS NULL"),
        ),
    )

    id = Column(String(24), primary_key=True, default=lambda: "doc" + _random_id())
    name = Column(String(1024), nullable=False)
    user = Column(String(256), nullable=False, index=True)
    collection_id = Column(String(24), nullable=True, index=True)
    status = Column(_enum_column(DocumentStatus), nullable=False, index=True)
    size = Column(BigInteger, nullable=False)
    content_hash = Column(String(64), nullable=True, index=True)
    object_path = Column(Text, nullable=True)
    doc_metadata = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)

    def get_document_indexes(self, session):
        stmt = select(DocumentIndex).where(DocumentIndex.document_id == self.id)
        result = session.execute(stmt)
        return result.scalars().all()

    def get_overall_index_status(self, session) -> "DocumentStatus":
        document_indexes = self.get_document_indexes(session)

        if not document_indexes:
            return DocumentStatus.PENDING

        statuses = [idx.status for idx in document_indexes]

        if any(status == DocumentIndexStatus.FAILED for status in statuses):
            return DocumentStatus.FAILED
        elif any(
            status in [DocumentIndexStatus.CREATING, DocumentIndexStatus.DELETION_IN_PROGRESS] for status in statuses
        ):
            return DocumentStatus.RUNNING
        elif all(status == DocumentIndexStatus.ACTIVE for status in statuses):
            return DocumentStatus.COMPLETE
        else:
            return DocumentStatus.PENDING

    def object_store_base_path(self) -> str:
        user = self.user.replace("|", "-")
        return f"user-{user}/{self.collection_id}/{self.id}"

    async def get_collection(self, session):
        return await session.get(Collection, self.collection_id)

    async def set_collection(self, collection):
        if hasattr(collection, "id"):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection


__all__ = [
    "Collection",
    "CollectionStatus",
    "CollectionSummary",
    "CollectionSummaryStatus",
    "CollectionType",
    "Document",
    "DocumentStatus",
]
