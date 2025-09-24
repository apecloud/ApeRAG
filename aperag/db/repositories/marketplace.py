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
    Bot,
    BotMarketplace,
    BotMarketplaceStatusEnum,
    BotStatus,
    Collection,
    CollectionMarketplace,
    CollectionMarketplaceStatusEnum,
    CollectionStatus,
    Department,
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
                    Collection.config,
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
                        "config": row.config,
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
                    Collection.config,
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
                        "config": row.config,
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

    # Bot marketplace sharing operations
    async def publish_bot_to_departments(
        self, bot_id: str, group_ids: List[str], status: str = BotMarketplaceStatusEnum.PUBLISHED.value
    ) -> List[BotMarketplace]:
        """Publish a bot to specific departments or globally"""

        async def _operation(session):
            # First, soft delete existing marketplace entries for this bot
            existing_stmt = select(BotMarketplace).where(
                BotMarketplace.bot_id == bot_id, BotMarketplace.gmt_deleted.is_(None)
            )
            result = await session.execute(existing_stmt)
            existing_entries = result.scalars().all()

            current_time = utc_now()
            
            # Soft delete existing entries
            for entry in existing_entries:
                entry.gmt_deleted = current_time
                entry.gmt_updated = current_time
                session.add(entry)

            # Create new marketplace entries for each group
            new_entries = []
            for group_id in group_ids:
                marketplace_entry = BotMarketplace(
                    bot_id=bot_id,
                    group_id=group_id,
                    status=status,
                    gmt_created=current_time,
                    gmt_updated=current_time,
                )
                session.add(marketplace_entry)
                new_entries.append(marketplace_entry)

            await session.flush()
            for entry in new_entries:
                await session.refresh(entry)
            return new_entries

        return await self.execute_with_transaction(_operation)

    async def get_bot_marketplace_entries_by_bot_id(self, bot_id: str) -> List[BotMarketplace]:
        """Get all bot marketplace records by bot_id"""

        async def _query(session):
            stmt = select(BotMarketplace).where(
                BotMarketplace.bot_id == bot_id, BotMarketplace.gmt_deleted.is_(None)
            ).order_by(BotMarketplace.gmt_created.desc())
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def unpublish_bot(self, bot_id: str) -> int:
        """
        Unpublish bot from marketplace by soft deleting all marketplace entries
        Returns the number of entries that were unpublished
        """

        async def _operation(session):
            # Find all published marketplace records for this bot
            stmt = select(BotMarketplace).where(
                BotMarketplace.bot_id == bot_id,
                BotMarketplace.status == BotMarketplaceStatusEnum.PUBLISHED.value,
                BotMarketplace.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            marketplace_entries = result.scalars().all()

            if not marketplace_entries:
                return 0

            current_time = utc_now()
            count = 0

            # Soft delete all marketplace entries
            for marketplace in marketplace_entries:
                marketplace.gmt_deleted = current_time
                marketplace.gmt_updated = current_time
                session.add(marketplace)
                count += 1

            await session.flush()
            return count

        return await self.execute_with_transaction(_operation)

    async def soft_delete_bot_marketplace(self, bot_id: str) -> bool:
        """
        Soft delete all bot marketplace records
        This is called when a bot is deleted
        Returns True if marketplace records were found and deleted, False otherwise
        """

        async def _operation(session):
            # Find all marketplace records for this bot
            stmt = select(BotMarketplace).where(
                BotMarketplace.bot_id == bot_id, BotMarketplace.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            marketplace_entries = result.scalars().all()

            if not marketplace_entries:
                return False

            current_time = utc_now()

            # Soft delete all marketplace records
            for marketplace in marketplace_entries:
                marketplace.gmt_deleted = current_time
                marketplace.gmt_updated = current_time
                session.add(marketplace)

            await session.flush()
            return True

        return await self.execute_with_transaction(_operation)

    async def get_accessible_bots_for_user(
        self, user_department_id: Optional[str], page: int = 1, page_size: int = 12, bot_type: str = None
    ) -> Tuple[List[dict], int]:
        """
        Get all bots accessible to a user based on their department
        Returns (bots_data, total_count)
        """

        async def _query(session):
            # Get user's department hierarchy (all parent departments)
            accessible_group_ids = ["*"]  # Global bots are always accessible

            if user_department_id:
                # Get user's department
                dept_result = await session.execute(
                    select(Department).where(Department.id == user_department_id)
                )
                user_dept = dept_result.scalars().first()

                if user_dept and user_dept.group_path:
                    # Extract all parent department IDs from group_path
                    # e.g., "/dp_1/dp_2" -> ["dp_1", "dp_2"]
                    path_parts = user_dept.group_path.strip("/").split("/")
                    accessible_group_ids.extend([part for part in path_parts if part])

                # Also include the user's direct department
                accessible_group_ids.append(user_department_id)

            # Base query for accessible published bots
            base_stmt = (
                select(
                    Bot.id.label("bot_id"),
                    Bot.title,
                    Bot.description,
                    Bot.type,
                    Bot.config,
                    Bot.gmt_created,
                    Bot.gmt_updated,
                    Bot.user.label("owner_user_id"),
                    User.username.label("owner_username"),
                    BotMarketplace.id.label("marketplace_id"),
                    BotMarketplace.group_id,
                    BotMarketplace.status.label("marketplace_status"),
                    BotMarketplace.gmt_created.label("published_at"),
                )
                .select_from(Bot)
                .join(BotMarketplace, Bot.id == BotMarketplace.bot_id)
                .outerjoin(User, Bot.user == User.id)
                .where(
                    and_(
                        Bot.status == BotStatus.ACTIVE.value,
                        Bot.gmt_deleted.is_(None),
                        BotMarketplace.status == BotMarketplaceStatusEnum.PUBLISHED.value,
                        BotMarketplace.group_id.in_(accessible_group_ids),
                        BotMarketplace.gmt_deleted.is_(None),
                    )
                )
            )

            # Add bot type filter if specified
            if bot_type:
                base_stmt = base_stmt.where(Bot.type == bot_type)

            # Order by bot ID first (required for DISTINCT ON), then by published time (newest first)
            base_stmt = base_stmt.order_by(Bot.id, desc(BotMarketplace.gmt_created))

            # Count total unique bots (since one bot might be published to multiple departments)
            count_stmt = (
                select(func.count(func.distinct(Bot.id)))
                .select_from(Bot)
                .join(BotMarketplace, Bot.id == BotMarketplace.bot_id)
            ).where(
                and_(
                    Bot.status == BotStatus.ACTIVE.value,
                    Bot.gmt_deleted.is_(None),
                    BotMarketplace.status == BotMarketplaceStatusEnum.PUBLISHED.value,
                    BotMarketplace.group_id.in_(accessible_group_ids),
                    BotMarketplace.gmt_deleted.is_(None),
                )
            )
            
            if bot_type:
                count_stmt = count_stmt.where(Bot.type == bot_type)
                
            count_result = await session.execute(count_stmt)
            total = count_result.scalar()

            # Apply pagination and get distinct bots
            offset = (page - 1) * page_size
            stmt = base_stmt.distinct(Bot.id).limit(page_size).offset(offset)
            result = await session.execute(stmt)

            bots = []
            for row in result:
                bots.append(
                    {
                        "id": row.bot_id,
                        "title": row.title,
                        "description": row.description,
                        "type": row.type,
                        "config": row.config,
                        "gmt_created": row.gmt_created,
                        "gmt_updated": row.gmt_updated,
                        "marketplace_status": row.marketplace_status,
                        "published_at": row.published_at,
                        "owner_user_id": row.owner_user_id,
                        "owner_username": row.owner_username,
                        "group_id": row.group_id,
                    }
                )

            return bots, total

        return await self._execute_query(_query)

