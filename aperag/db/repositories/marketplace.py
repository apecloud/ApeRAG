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

from typing import List, Optional, Tuple

from sqlalchemy import and_, desc, func, select

from aperag.db.models import (
    Collection,
    CollectionMarketplace,
    CollectionMarketplaceStatusEnum,
    CollectionStatus,
    User,
    UserCollectionSubscription,
)
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncMarketplaceRepositoryMixin(AsyncRepositoryProtocol):
    """Repository for marketplace-related operations"""

    # Marketplace sharing operations
    async def create_or_update_collection_marketplace(
        self, collection_id: str, status: str = CollectionMarketplaceStatusEnum.PUBLISHED.value
    ) -> CollectionMarketplace:
        """Create or update collection marketplace record"""

        async def _operation(session):
            # Check if marketplace record already exists
            stmt = select(CollectionMarketplace).where(
                CollectionMarketplace.collection_id == collection_id, CollectionMarketplace.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            marketplace = result.scalars().first()

            current_time = utc_now()

            if marketplace:
                # Update existing record
                marketplace.status = status
                marketplace.gmt_updated = current_time
                session.add(marketplace)
            else:
                # Create new record
                marketplace = CollectionMarketplace(
                    collection_id=collection_id, status=status, gmt_created=current_time, gmt_updated=current_time
                )
                session.add(marketplace)

            await session.flush()
            await session.refresh(marketplace)
            return marketplace

        return await self.execute_with_transaction(_operation)

    async def get_collection_marketplace_by_collection_id(
        self, collection_id: str, ignore_deleted: bool = True
    ) -> Optional[CollectionMarketplace]:
        """Get marketplace record by collection ID"""

        async def _query(session):
            stmt = select(CollectionMarketplace).where(CollectionMarketplace.collection_id == collection_id)
            if ignore_deleted:
                stmt = stmt.where(CollectionMarketplace.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_collection_marketplace_by_id(
        self, marketplace_id: str, ignore_deleted: bool = True
    ) -> Optional[CollectionMarketplace]:
        """Get marketplace record by marketplace ID"""

        async def _query(session):
            stmt = select(CollectionMarketplace).where(CollectionMarketplace.id == marketplace_id)
            if ignore_deleted:
                stmt = stmt.where(CollectionMarketplace.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def unpublish_collection(self, collection_id: str) -> Optional[CollectionMarketplace]:
        """Unpublish collection by changing status to DRAFT and invalidate related subscriptions"""

        async def _operation(session):
            # Update marketplace status to DRAFT
            stmt = select(CollectionMarketplace).where(
                CollectionMarketplace.collection_id == collection_id, CollectionMarketplace.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            marketplace = result.scalars().first()

            if not marketplace:
                return None

            current_time = utc_now()
            marketplace.status = CollectionMarketplaceStatusEnum.DRAFT.value
            marketplace.gmt_updated = current_time
            session.add(marketplace)

            # Soft delete all related active subscriptions
            subscriptions_stmt = select(UserCollectionSubscription).where(
                UserCollectionSubscription.collection_marketplace_id == marketplace.id,
                UserCollectionSubscription.gmt_deleted.is_(None),
            )
            subscriptions_result = await session.execute(subscriptions_stmt)
            subscriptions = subscriptions_result.scalars().all()

            for subscription in subscriptions:
                subscription.gmt_deleted = current_time
                session.add(subscription)

            await session.flush()
            await session.refresh(marketplace)
            return marketplace

        return await self.execute_with_transaction(_operation)

    async def soft_delete_collection_marketplace(self, collection_id: str) -> bool:
        """Soft delete marketplace record and all related subscriptions when collection is deleted"""

        async def _operation(session):
            # Soft delete marketplace record
            stmt = select(CollectionMarketplace).where(
                CollectionMarketplace.collection_id == collection_id, CollectionMarketplace.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            marketplace = result.scalars().first()

            if not marketplace:
                return False

            current_time = utc_now()
            marketplace.gmt_deleted = current_time
            session.add(marketplace)

            # Soft delete all related subscriptions
            subscriptions_stmt = select(UserCollectionSubscription).where(
                UserCollectionSubscription.collection_marketplace_id == marketplace.id,
                UserCollectionSubscription.gmt_deleted.is_(None),
            )
            subscriptions_result = await session.execute(subscriptions_stmt)
            subscriptions = subscriptions_result.scalars().all()

            for subscription in subscriptions:
                subscription.gmt_deleted = current_time
                session.add(subscription)

            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)

    # Marketplace listing operations
    async def list_published_collections_with_subscription_status(
        self, user_id: str, page: int = 1, page_size: int = 12
    ) -> Tuple[List[dict], int]:
        """List all published collections with current user's subscription status"""

        async def _query(session):
            # Base query for published collections
            base_stmt = (
                select(
                    CollectionMarketplace.id.label("marketplace_id"),
                    Collection.id.label("collection_id"),
                    Collection.title,
                    Collection.description,
                    Collection.user.label("owner_user_id"),
                    User.username.label("owner_username"),
                    CollectionMarketplace.gmt_created.label("published_at"),
                    UserCollectionSubscription.id.label("subscription_id"),
                    UserCollectionSubscription.gmt_subscribed,
                )
                .select_from(CollectionMarketplace)
                .join(Collection, CollectionMarketplace.collection_id == Collection.id)
                .join(User, Collection.user == User.id)
                .outerjoin(
                    UserCollectionSubscription,
                    and_(
                        UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                        UserCollectionSubscription.user_id == user_id,
                        UserCollectionSubscription.gmt_deleted.is_(None),
                    ),
                )
                .where(
                    CollectionMarketplace.status == CollectionMarketplaceStatusEnum.PUBLISHED.value,
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
                .order_by(desc(CollectionMarketplace.gmt_created))
            )

            # Count total
            count_stmt = (
                select(func.count(CollectionMarketplace.id))
                .select_from(CollectionMarketplace)
                .join(Collection, CollectionMarketplace.collection_id == Collection.id)
                .where(
                    CollectionMarketplace.status == CollectionMarketplaceStatusEnum.PUBLISHED.value,
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
            )

            count_result = await session.execute(count_stmt)
            total = count_result.scalar()

            # Paginated query
            offset = (page - 1) * page_size
            stmt = base_stmt.limit(page_size).offset(offset)
            result = await session.execute(stmt)

            collections = []
            for row in result:
                collections.append(
                    {
                        "marketplace_id": row.marketplace_id,
                        "id": row.collection_id,
                        "title": row.title,
                        "description": row.description,
                        "owner_user_id": row.owner_user_id,
                        "owner_username": row.owner_username,
                        "published_at": row.published_at,
                        "subscription_id": row.subscription_id,
                        "gmt_subscribed": row.gmt_subscribed,
                    }
                )

            return collections, total

        return await self._execute_query(_query)

    # Subscription operations
    async def create_subscription(self, user_id: str, collection_marketplace_id: str) -> UserCollectionSubscription:
        """Create a new subscription"""

        async def _operation(session):
            subscription = UserCollectionSubscription(
                user_id=user_id, collection_marketplace_id=collection_marketplace_id, gmt_subscribed=utc_now()
            )
            session.add(subscription)
            await session.flush()
            await session.refresh(subscription)
            return subscription

        return await self.execute_with_transaction(_operation)

    async def get_user_subscription_by_collection_id(
        self, user_id: str, collection_id: str
    ) -> Optional[UserCollectionSubscription]:
        """Get user's active subscription by collection ID"""

        async def _query(session):
            stmt = (
                select(UserCollectionSubscription)
                .join(
                    CollectionMarketplace,
                    UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                )
                .where(
                    UserCollectionSubscription.user_id == user_id,
                    CollectionMarketplace.collection_id == collection_id,
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.gmt_deleted.is_(None),
                )
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_user_subscription_by_marketplace_id(
        self, user_id: str, collection_marketplace_id: str
    ) -> Optional[UserCollectionSubscription]:
        """Get user's active subscription by marketplace ID"""

        async def _query(session):
            stmt = select(UserCollectionSubscription).where(
                UserCollectionSubscription.user_id == user_id,
                UserCollectionSubscription.collection_marketplace_id == collection_marketplace_id,
                UserCollectionSubscription.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def unsubscribe_collection(self, user_id: str, collection_id: str) -> Optional[UserCollectionSubscription]:
        """Unsubscribe from collection by soft deleting subscription record"""

        async def _operation(session):
            # Find the subscription through marketplace
            stmt = (
                select(UserCollectionSubscription)
                .join(
                    CollectionMarketplace,
                    UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                )
                .where(
                    UserCollectionSubscription.user_id == user_id,
                    CollectionMarketplace.collection_id == collection_id,
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.gmt_deleted.is_(None),
                )
            )
            result = await session.execute(stmt)
            subscription = result.scalars().first()

            if subscription:
                subscription.gmt_deleted = utc_now()
                session.add(subscription)
                await session.flush()
                await session.refresh(subscription)

            return subscription

        return await self.execute_with_transaction(_operation)

    async def list_user_subscribed_collections(
        self, user_id: str, page: int = 1, page_size: int = 12
    ) -> Tuple[List[dict], int]:
        """List all collections subscribed by user"""

        async def _query(session):
            # Base query for user subscriptions
            base_stmt = (
                select(
                    UserCollectionSubscription.id.label("subscription_id"),
                    Collection.id.label("collection_id"),
                    Collection.title,
                    Collection.description,
                    Collection.type,
                    Collection.status,
                    Collection.gmt_created,
                    Collection.gmt_updated,
                    CollectionMarketplace.status.label("marketplace_status"),
                    CollectionMarketplace.gmt_created.label("published_at"),
                    Collection.user.label("owner_user_id"),
                    User.username.label("owner_username"),
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
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
                .order_by(desc(UserCollectionSubscription.gmt_subscribed))
            )

            # Count total
            count_stmt = (
                select(func.count(UserCollectionSubscription.id))
                .select_from(UserCollectionSubscription)
                .join(
                    CollectionMarketplace,
                    UserCollectionSubscription.collection_marketplace_id == CollectionMarketplace.id,
                )
                .join(Collection, CollectionMarketplace.collection_id == Collection.id)
                .where(
                    UserCollectionSubscription.user_id == user_id,
                    UserCollectionSubscription.gmt_deleted.is_(None),
                    CollectionMarketplace.gmt_deleted.is_(None),
                    Collection.status != CollectionStatus.DELETED,
                    Collection.gmt_deleted.is_(None),
                )
            )

            count_result = await session.execute(count_stmt)
            total = count_result.scalar()

            # Paginated query
            offset = (page - 1) * page_size
            stmt = base_stmt.limit(page_size).offset(offset)
            result = await session.execute(stmt)

            collections = []
            for row in result:
                collections.append(
                    {
                        "subscription_id": row.subscription_id,
                        "id": row.collection_id,
                        "title": row.title,
                        "description": row.description,
                        "type": row.type,
                        "status": row.status,
                        "gmt_created": row.gmt_created,
                        "gmt_updated": row.gmt_updated,
                        "marketplace_status": row.marketplace_status,
                        "published_at": row.published_at,
                        "owner_user_id": row.owner_user_id,
                        "owner_username": row.owner_username,
                        "gmt_subscribed": row.gmt_subscribed,
                    }
                )

            return collections, total

        return await self._execute_query(_query)
