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

import logging
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import (
    AlreadySubscribedError,
    CollectionNotFoundException,
    CollectionNotPublishedError,
    PermissionDeniedError,
    SelfSubscriptionError,
)
from aperag.schema import view_models
from aperag.schema.utils import convertToSharedCollectionConfig, parseCollectionConfig

logger = logging.getLogger(__name__)


class MarketplaceService:
    """
    Marketplace business logic service
    Responsibilities: Handle all marketplace and sharing related business logic
    """

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def publish_collection(self, user_id: str, collection_id: str) -> None:
        """Publish Collection to marketplace"""
        # Verify user ownership
        await self._verify_collection_ownership(user_id, collection_id)

        # Create or update collection_marketplace record
        await self.db_ops.create_or_update_collection_marketplace(
            collection_id=collection_id, status=db_models.CollectionMarketplaceStatusEnum.PUBLISHED.value
        )

    async def unpublish_collection(self, user_id: str, collection_id: str) -> None:
        """Remove Collection from marketplace"""
        # Verify user ownership
        await self._verify_collection_ownership(user_id, collection_id)

        # Update collection_marketplace record status to 'DRAFT' and invalidate related subscriptions
        # Note: This uses transaction to ensure data consistency
        marketplace = await self.db_ops.unpublish_collection(collection_id)
        if marketplace is None:
            raise CollectionNotPublishedError(collection_id)

    async def get_sharing_status(self, user_id: str, collection_id: str) -> Tuple[bool, Optional[datetime]]:
        """Get Collection sharing status"""
        # Verify user ownership first
        await self._verify_collection_ownership(user_id, collection_id)

        marketplace = await self.db_ops.get_collection_marketplace_by_collection_id(collection_id)
        if marketplace is None:
            return False, None

        is_published = marketplace.status == db_models.CollectionMarketplaceStatusEnum.PUBLISHED.value
        published_at = marketplace.gmt_created if is_published else None

        return is_published, published_at

    async def get_raw_sharing_status(self, collection_id: str) -> Optional[db_models.CollectionMarketplace]:
        """Get raw sharing status (for permission checks)"""
        return await self.db_ops.get_collection_marketplace_by_collection_id(collection_id)

    async def list_published_collections(
        self, user_id: str, page: int = 1, page_size: int = 12
    ) -> view_models.SharedCollectionList:
        """List all published Collections in marketplace"""
        collections_data, total = await self.db_ops.list_published_collections_with_subscription_status(
            user_id=user_id, page=page, page_size=page_size
        )

        # Convert to SharedCollection objects
        collections = []
        for data in collections_data:
            # Parse collection config and convert to SharedCollectionConfig
            collection_config = parseCollectionConfig(data["config"])
            shared_config = convertToSharedCollectionConfig(collection_config)

            shared_collection = view_models.SharedCollection(
                id=data["id"],
                title=data["title"],
                description=data["description"],
                owner_user_id=data["owner_user_id"],
                owner_username=data["owner_username"],
                subscription_id=data["subscription_id"],
                gmt_subscribed=data["gmt_subscribed"],
                config=shared_config,
            )
            collections.append(shared_collection)

        return view_models.SharedCollectionList(items=collections, total=total, page=page, page_size=page_size)

    async def subscribe_collection(self, user_id: str, collection_id: str) -> view_models.SharedCollection:
        """Subscribe to Collection"""
        # 1. Find Collection's corresponding published marketplace record (status = 'PUBLISHED', gmt_deleted IS NULL)
        marketplace = await self.db_ops.get_collection_marketplace_by_collection_id(collection_id)
        if marketplace is None or marketplace.status != db_models.CollectionMarketplaceStatusEnum.PUBLISHED.value:
            raise CollectionNotPublishedError(collection_id)

        # 2. Verify user is not the Collection owner (user_id != collection.user)
        collection = await self.db_ops.query_collection_by_id(collection_id)
        if collection is None:
            raise CollectionNotFoundException(collection_id)

        if collection.user == user_id:
            raise SelfSubscriptionError(collection_id)

        # 3. Check if already subscribed to this marketplace instance, prevent duplicate subscription
        existing_subscription = await self.db_ops.get_user_subscription_by_marketplace_id(
            user_id=user_id, collection_marketplace_id=marketplace.id
        )
        if existing_subscription is not None:
            raise AlreadySubscribedError(collection_id)

        # 4. Create user_collection_subscription record (associated with collection_marketplace_id)
        subscription = await self.db_ops.create_subscription(user_id=user_id, collection_marketplace_id=marketplace.id)

        # Get owner information
        owner = await self.db_ops.query_user_by_username(collection.user)
        owner_username = owner.username if owner else collection.user

        # Parse collection config and convert to SharedCollectionConfig
        collection_config = parseCollectionConfig(collection.config)
        shared_config = convertToSharedCollectionConfig(collection_config)

        return view_models.SharedCollection(
            id=collection.id,
            title=collection.title,
            description=collection.description,
            owner_user_id=collection.user,
            owner_username=owner_username,
            subscription_id=subscription.id,
            gmt_subscribed=subscription.gmt_subscribed,
            config=shared_config,
        )

    async def unsubscribe_collection(self, user_id: str, collection_id: str) -> None:
        """Unsubscribe from Collection"""
        # Check if user has subscribed to this Collection
        subscription = await self.db_ops.get_user_subscription_by_collection_id(user_id, collection_id)

        # If no active subscription found, return silently (idempotent operation)
        # This handles cases where user clicks unsubscribe multiple times
        if subscription is None:
            logger.info(
                f"User {user_id} attempted to unsubscribe from collection {collection_id}, but no active subscription found. Operation treated as successful (idempotent)."
            )
            return

        # Soft delete subscription record (set gmt_deleted = current_timestamp)
        await self.db_ops.unsubscribe_collection(user_id, collection_id)

    async def get_user_subscription(
        self, user_id: str, collection_id: str
    ) -> Optional[db_models.UserCollectionSubscription]:
        """Get user's active subscription status for specified Collection"""
        # Find published marketplace record through collection_id, then find corresponding subscription record
        # Used by permission check functions
        # Returns None if not subscribed or already unsubscribed
        return await self.db_ops.get_user_subscription_by_collection_id(user_id, collection_id)

    async def list_user_subscribed_collections(
        self, user_id: str, page: int = 1, page_size: int = 12
    ) -> view_models.SharedCollectionList:
        """Get all active subscribed Collections for user"""
        # Query WHERE gmt_deleted IS NULL
        # Join query to get Collection details and original owner information
        # Support pagination
        collections_data, total = await self.db_ops.list_user_subscribed_collections(
            user_id=user_id, page=page, page_size=page_size
        )

        # Convert to SharedCollection objects
        collections = []
        for data in collections_data:
            # Parse collection config and convert to SharedCollectionConfig
            collection_config = parseCollectionConfig(data["config"])
            shared_config = convertToSharedCollectionConfig(collection_config)

            shared_collection = view_models.SharedCollection(
                id=data["id"],
                title=data["title"],
                description=data["description"],
                owner_user_id=data["owner_user_id"],
                owner_username=data["owner_username"],
                subscription_id=data["subscription_id"],
                gmt_subscribed=data["gmt_subscribed"],
                config=shared_config,
            )
            collections.append(shared_collection)

        return view_models.SharedCollectionList(items=collections, total=total, page=page, page_size=page_size)

    async def cleanup_collection_marketplace_data(self, collection_id: str) -> None:
        """Cleanup marketplace data when collection is deleted"""
        # This method will:
        # 1. Soft delete collection_marketplace record (set gmt_deleted)
        # 2. Batch soft delete user_collection_subscription records (set gmt_deleted)
        # 3. Use transaction to ensure data consistency
        await self.db_ops.soft_delete_collection_marketplace(collection_id)

    async def _verify_collection_ownership(self, user_id: str, collection_id: str) -> db_models.Collection:
        """Verify user owns the collection"""
        collection = await self.db_ops.query_collection_by_id(collection_id)
        if collection is None:
            raise CollectionNotFoundException(collection_id)

        if collection.user != user_id:
            raise PermissionDeniedError(f"You don't have permission to manage collection {collection_id}")

        return collection


# Global marketplace service instance
marketplace_service = MarketplaceService()
