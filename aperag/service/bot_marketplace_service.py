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
    AlreadySubscribedToBotError,
    BotNotFoundException,
    BotNotPublishedError,
    PermissionDeniedError,
    SelfBotSubscriptionError,
)
from aperag.schema import view_models
from aperag.schema.utils import parseBotConfig, convertToSharedBotConfig

logger = logging.getLogger(__name__)


class BotMarketplaceService:
    """
    Bot marketplace business logic service
    Responsibilities: Handle all bot marketplace and sharing related business logic
    """

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def publish_bot(self, user_id: str, bot_id: str) -> None:
        """Publish bot to marketplace"""
        # Verify user ownership
        await self._verify_bot_ownership(user_id, bot_id)

        # Create or update bot_marketplace record
        await self.db_ops.create_or_update_bot_marketplace(
            bot_id=bot_id, status=db_models.BotMarketplaceStatusEnum.PUBLISHED.value
        )

    async def unpublish_bot(self, user_id: str, bot_id: str) -> None:
        """Remove bot from marketplace"""
        # Verify user ownership
        await self._verify_bot_ownership(user_id, bot_id)

        # Update bot_marketplace record status to 'DRAFT' and invalidate related subscriptions
        # Note: This uses transaction to ensure data consistency
        marketplace = await self.db_ops.unpublish_bot(bot_id)
        if marketplace is None:
            raise BotNotPublishedError(bot_id)

    async def get_sharing_status(self, user_id: str, bot_id: str) -> Tuple[bool, Optional[datetime]]:
        """Get bot sharing status"""
        # Verify user ownership first
        await self._verify_bot_ownership(user_id, bot_id)

        marketplace = await self.db_ops.get_bot_marketplace_by_bot_id(bot_id)
        if marketplace is None:
            return False, None

        is_published = marketplace.status == db_models.BotMarketplaceStatusEnum.PUBLISHED.value
        published_at = marketplace.gmt_created if is_published else None

        return is_published, published_at

    async def get_raw_sharing_status(self, bot_id: str) -> Optional[db_models.BotMarketplace]:
        """Get raw sharing status (for permission checks)"""
        return await self.db_ops.get_bot_marketplace_by_bot_id(bot_id)

    async def validate_marketplace_bot(self, bot_id: str):
        """
        Validate if bot is published in marketplace

        Args:
            bot_id: Bot ID to validate

        Returns:
            bool: True if bot is published in marketplace

        Raises:
            HTTPException: If bot is not published in marketplace
        """
        from fastapi import HTTPException

        marketplace_record = await self.get_raw_sharing_status(bot_id)
        if (
            not marketplace_record
            or marketplace_record.status != db_models.BotMarketplaceStatusEnum.PUBLISHED.value
        ):
            raise HTTPException(status_code=401, detail="Authentication required")

    async def list_published_bots(
        self, user_id: str, page: int = 1, page_size: int = 12, bot_type: str = None
    ) -> view_models.SharedBotList:
        """List all published bots in marketplace"""
        bots_data, total = await self.db_ops.list_published_bots_with_subscription_status(
            user_id=user_id, page=page, page_size=page_size, bot_type=bot_type
        )

        # Convert to SharedBot objects
        bots = []
        for data in bots_data:
            # Parse bot config and convert to SharedBotConfig
            bot_config = parseBotConfig(data["config"])
            shared_config = convertToSharedBotConfig(bot_config)

            shared_bot = view_models.SharedBot(
                id=data["id"],
                title=data["title"],
                description=data["description"],
                type=data["type"],
                owner_user_id=data["owner_user_id"],
                owner_username=data["owner_username"],
                subscription_id=data["subscription_id"],
                gmt_subscribed=data["gmt_subscribed"],
                config=shared_config,
                is_subscribed=data["subscription_id"] is not None,
                is_owner=data["owner_user_id"] == user_id,
            )
            bots.append(shared_bot)

        return view_models.SharedBotList(items=bots, total=total, page=page, page_size=page_size)

    async def subscribe_bot(self, user_id: str, bot_id: str) -> view_models.SharedBot:
        """Subscribe to bot"""
        # 1. Find bot's corresponding published marketplace record (status = 'PUBLISHED', gmt_deleted IS NULL)
        marketplace = await self.db_ops.get_bot_marketplace_by_bot_id(bot_id)
        if marketplace is None or marketplace.status != db_models.BotMarketplaceStatusEnum.PUBLISHED.value:
            raise BotNotPublishedError(bot_id)

        # 2. Verify user is not the bot owner (user_id != bot.user)
        bot = await self.db_ops.query_bot_by_id(bot_id)
        if bot is None:
            raise BotNotFoundException(bot_id)

        if bot.user == user_id:
            raise SelfBotSubscriptionError(bot_id)

        # 3. Check if already subscribed to this marketplace instance, prevent duplicate subscription
        existing_subscription = await self.db_ops.get_user_bot_subscription_by_marketplace_id(
            user_id=user_id, bot_marketplace_id=marketplace.id
        )
        if existing_subscription is not None:
            raise AlreadySubscribedToBotError(bot_id)

        # 4. Create user_bot_subscription record (associated with bot_marketplace_id)
        subscription = await self.db_ops.create_bot_subscription(user_id=user_id, bot_marketplace_id=marketplace.id)

        # Get owner information
        owner = await self.db_ops.query_user_by_username(bot.user)
        owner_username = owner.username if owner else bot.user

        # Parse bot config and convert to SharedBotConfig
        bot_config = parseBotConfig(bot.config)
        shared_config = convertToSharedBotConfig(bot_config)

        return view_models.SharedBot(
            id=bot.id,
            title=bot.title,
            description=bot.description,
            type=bot.type,
            owner_user_id=bot.user,
            owner_username=owner_username,
            subscription_id=subscription.id,
            gmt_subscribed=subscription.gmt_subscribed,
            config=shared_config,
            is_subscribed=True,
            is_owner=False,
        )

    async def unsubscribe_bot(self, user_id: str, bot_id: str) -> None:
        """Unsubscribe from bot"""
        # Check if user has subscribed to this bot
        subscription = await self.db_ops.get_user_bot_subscription_by_bot_id(user_id, bot_id)

        # If no active subscription found, return silently (idempotent operation)
        # This handles cases where user clicks unsubscribe multiple times
        if subscription is None:
            logger.info(
                f"User {user_id} attempted to unsubscribe from bot {bot_id}, but no active subscription found. Operation treated as successful (idempotent)."
            )
            return

        # Soft delete subscription record (set gmt_deleted = current_timestamp)
        await self.db_ops.unsubscribe_bot(user_id, bot_id)

    async def get_user_subscription(
        self, user_id: str, bot_id: str
    ) -> Optional[db_models.UserBotSubscription]:
        """Get user's active subscription status for specified bot"""
        # Find published marketplace record through bot_id, then find corresponding subscription record
        # Used by permission check functions
        # Returns None if not subscribed or already unsubscribed
        return await self.db_ops.get_user_bot_subscription_by_bot_id(user_id, bot_id)

    async def list_user_subscribed_bots(
        self, user_id: str, page: int = 1, page_size: int = 12
    ) -> view_models.SharedBotList:
        """Get all active subscribed bots for user"""
        # Query WHERE gmt_deleted IS NULL
        # Join query to get bot details and original owner information
        # Support pagination
        bots_data, total = await self.db_ops.list_user_subscribed_bots(
            user_id=user_id, page=page, page_size=page_size
        )

        # Convert to SharedBot objects
        bots = []
        for data in bots_data:
            # Parse bot config and convert to SharedBotConfig
            bot_config = parseBotConfig(data["config"])
            shared_config = convertToSharedBotConfig(bot_config)

            shared_bot = view_models.SharedBot(
                id=data["id"],
                title=data["title"],
                description=data["description"],
                type=data["type"],
                owner_user_id=data["owner_user_id"],
                owner_username=data["owner_username"],
                subscription_id=data["subscription_id"],
                gmt_subscribed=data["gmt_subscribed"],
                config=shared_config,
                is_subscribed=True,
                is_owner=False,
            )
            bots.append(shared_bot)

        return view_models.SharedBotList(items=bots, total=total, page=page, page_size=page_size)

    async def cleanup_bot_marketplace_data(self, bot_id: str) -> None:
        """Cleanup marketplace data when bot is deleted"""
        # This method will:
        # 1. Soft delete bot_marketplace record (set gmt_deleted)
        # 2. Batch soft delete user_bot_subscription records (set gmt_deleted)
        # 3. Use transaction to ensure data consistency
        await self.db_ops.soft_delete_bot_marketplace(bot_id)

    async def _verify_bot_ownership(self, user_id: str, bot_id: str) -> db_models.Bot:
        """Verify user owns the bot"""
        bot = await self.db_ops.query_bot_by_id(bot_id)
        if bot is None:
            raise BotNotFoundException(bot_id)

        if bot.user != user_id:
            raise PermissionDeniedError(f"You don't have permission to manage bot {bot_id}")

        return bot


# Global bot marketplace service instance
bot_marketplace_service = BotMarketplaceService()
