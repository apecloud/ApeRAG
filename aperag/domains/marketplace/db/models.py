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

"""Marketplace-domain SQLAlchemy models.

Owns the two entities the marketplace domain is responsible for:

* ``CollectionMarketplace`` — per-collection publish status in the
  public marketplace catalog.
* ``UserCollectionSubscription`` — per-user active subscription to a
  published collection.

Plus the publish-status enum ``CollectionMarketplaceStatusEnum``.
Moved here from ``aperag.db.models`` in Phase 4 Step 4-S2c.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum

from sqlalchemy import Column, DateTime, Index, String, UniqueConstraint

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id``. Phase 6 cleanup
    consolidates."""

    return "".join(random.sample(uuid.uuid4().hex, 16))


class CollectionMarketplaceStatusEnum(str, Enum):
    """Collection marketplace sharing status enumeration"""

    DRAFT = "DRAFT"  # Not published, only owner can see
    PUBLISHED = "PUBLISHED"  # Published to marketplace, publicly visible


class CollectionMarketplace(Base):
    """Collection sharing status table"""

    __tablename__ = "collection_marketplace"
    __table_args__ = (
        UniqueConstraint("collection_id", name="uq_collection_marketplace_collection"),
        Index("idx_collection_marketplace_status", "status"),
        Index("idx_collection_marketplace_gmt_deleted", "gmt_deleted"),
        Index("idx_collection_marketplace_collection_id", "collection_id"),
        Index("idx_collection_marketplace_list", "status", "gmt_created"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "market_" + _random_id()[:16])
    collection_id = Column(String(24), nullable=False)

    # Sharing status: use VARCHAR storage, not database enum type, validated at application layer
    status = Column(String(20), nullable=False, default=CollectionMarketplaceStatusEnum.DRAFT.value)

    # Timestamp fields
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)  # Updated in code layer
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<CollectionMarketplace(id={self.id}, collection_id={self.collection_id}, status={self.status})>"


class UserCollectionSubscription(Base):
    """User subscription to published collections table"""

    __tablename__ = "user_collection_subscription"
    __table_args__ = (
        # Allow multiple history records, but active subscription (gmt_deleted=NULL) must be unique
        UniqueConstraint(
            "user_id", "collection_marketplace_id", "gmt_deleted", name="idx_user_marketplace_history_unique"
        ),
        Index("idx_user_subscription_marketplace", "collection_marketplace_id"),
        Index("idx_user_subscription_user", "user_id"),
        Index("idx_user_subscription_gmt_deleted", "gmt_deleted"),
    )

    id = Column(String(24), primary_key=True, default=lambda: "sub_" + _random_id()[:16])
    user_id = Column(String(24), nullable=False)  # Related to users table, maintained at application layer
    collection_marketplace_id = Column(
        String(24), nullable=False
    )  # Related to collection_marketplace table, maintained at application layer

    # Timestamp fields
    gmt_subscribed = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)  # Soft delete: NULL means active subscription

    def __repr__(self):
        return f"<UserCollectionSubscription(id={self.id}, user_id={self.user_id}, marketplace_id={self.collection_marketplace_id})>"


__all__ = [
    "CollectionMarketplace",
    "CollectionMarketplaceStatusEnum",
    "UserCollectionSubscription",
]
