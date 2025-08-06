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

from typing import Optional, Tuple

from sqlalchemy import select

from aperag.db.models import (
    Collection,
    CollectionMarketplace,
    CollectionMarketplaceStatusEnum,
    CollectionStatus,
    User,
    UserCollectionSubscription,
)
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncMarketplaceCollectionRepositoryMixin(AsyncRepositoryProtocol):
    """Repository for marketplace collection access operations (read-only for subscribers)"""

    async def check_subscription_access(self, user_id: str, collection_id: str) -> Tuple[bool, Optional[dict]]:
        """
        Check if user has valid subscription access to a collection
        Returns: (has_access, subscription_info)
        """

        async def _query(session):
            # Check if collection exists and is published
            marketplace_stmt = (
                select(CollectionMarketplace, Collection, User)
                .join(Collection, CollectionMarketplace.collection_id == Collection.id)
                .join(User, Collection.user == User.id)
                .where(
                    CollectionMarketplace.collection_id == collection_id,
                    CollectionMarketplace.status == CollectionMarketplaceStatusEnum.PUBLISHED.value,
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
            )
            marketplace_result = await session.execute(marketplace_stmt)
            marketplace_data = marketplace_result.first()

            if not marketplace_data:
                return False, None

            marketplace, collection, owner = marketplace_data

            # Check if user has active subscription
            subscription_stmt = select(UserCollectionSubscription).where(
                UserCollectionSubscription.user_id == user_id,
                UserCollectionSubscription.collection_marketplace_id == marketplace.id,
                UserCollectionSubscription.gmt_deleted.is_(None),
            )
            subscription_result = await session.execute(subscription_stmt)
            subscription = subscription_result.scalars().first()

            if not subscription:
                return False, None

            # Return access granted with subscription info
            subscription_info = {
                "subscription_id": subscription.id,
                "marketplace_id": marketplace.id,
                "collection_id": collection.id,
                "collection_title": collection.title,
                "collection_description": collection.description,
                "owner_user_id": owner.id,
                "owner_username": owner.username,
                "gmt_subscribed": subscription.gmt_subscribed,
            }

            return True, subscription_info

        return await self._execute_query(_query)

    async def get_marketplace_collection_info(self, user_id: str, collection_id: str) -> Optional[dict]:
        """
        Get marketplace collection info for subscribed user
        This method assumes access has already been verified
        """

        async def _query(session):
            stmt = (
                select(
                    Collection.id.label("collection_id"),
                    Collection.title,
                    Collection.description,
                    Collection.user.label("owner_user_id"),
                    User.username.label("owner_username"),
                    UserCollectionSubscription.id.label("subscription_id"),
                    UserCollectionSubscription.gmt_subscribed,
                )
                .select_from(UserCollectionSubscription)
                .join(
                    CollectionMarketplace,
                    UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                )
                .join(Collection, CollectionMarketplace.collection_id == Collection.id)
                .join(User, Collection.user == User.id)
                .where(
                    UserCollectionSubscription.user_id == user_id,
                    CollectionMarketplace.collection_id == collection_id,
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.status == CollectionMarketplaceStatusEnum.PUBLISHED.value,
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
            )

            result = await session.execute(stmt)
            row = result.first()

            if not row:
                return None

            return {
                "id": row.collection_id,
                "title": row.title,
                "description": row.description,
                "owner_user_id": row.owner_user_id,
                "owner_username": row.owner_username,
                "subscription_id": row.subscription_id,
                "gmt_subscribed": row.gmt_subscribed,
            }

        return await self._execute_query(_query)

    async def verify_collection_subscription(
        self, user_id: str, collection_id: str
    ) -> Optional[UserCollectionSubscription]:
        """
        Verify that user has valid subscription to the collection
        Used for permission checks in marketplace collection endpoints
        """

        async def _query(session):
            stmt = (
                select(UserCollectionSubscription)
                .join(
                    CollectionMarketplace,
                    UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                )
                .join(Collection, CollectionMarketplace.collection_id == Collection.id)
                .where(
                    UserCollectionSubscription.user_id == user_id,
                    CollectionMarketplace.collection_id == collection_id,
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.status == CollectionMarketplaceStatusEnum.PUBLISHED.value,
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
            )

            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_collection_by_subscription_access(self, user_id: str, collection_id: str) -> Optional[Collection]:
        """
        Get collection object for user with valid subscription
        Used when marketplace collection endpoints need the actual Collection object
        """

        async def _query(session):
            stmt = (
                select(Collection)
                .join(CollectionMarketplace, CollectionMarketplace.collection_id == Collection.id)
                .join(
                    UserCollectionSubscription,
                    UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                )
                .where(
                    UserCollectionSubscription.user_id == user_id,
                    Collection.id == collection_id,
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.status == CollectionMarketplaceStatusEnum.PUBLISHED.value,
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
            )

            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)
