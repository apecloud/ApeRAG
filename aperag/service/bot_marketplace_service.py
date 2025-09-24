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
from typing import List, Optional, Tuple

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

    async def publish_bot(self, user_id: str, bot_id: str, group_ids: List[str]) -> None:
        """Publish bot to marketplace with department scope"""
        # Verify user ownership
        await self._verify_bot_ownership(user_id, bot_id)

        # Publish bot to specified departments
        await self.db_ops.publish_bot_to_departments(
            bot_id=bot_id, group_ids=group_ids
        )

    async def unpublish_bot(self, user_id: str, bot_id: str) -> None:
        """Remove bot from marketplace"""
        # Verify user ownership
        await self._verify_bot_ownership(user_id, bot_id)

        # Unpublish bot from all departments
        count = await self.db_ops.unpublish_bot(bot_id)
        if count == 0:
            raise BotNotPublishedError(bot_id)

    async def get_sharing_status(self, user_id: str, bot_id: str) -> view_models.BotSharingStatusResponse:
        """Get bot sharing status"""
        # Verify user ownership first
        await self._verify_bot_ownership(user_id, bot_id)

        marketplace_entries = await self.db_ops.get_bot_marketplace_entries_by_bot_id(bot_id)
        if not marketplace_entries:
            return view_models.BotSharingStatusResponse(is_published=False)

        # Check if any entry is published
        published_entries = [
            entry for entry in marketplace_entries 
            if entry.status == db_models.BotMarketplaceStatusEnum.PUBLISHED.value
        ]
        
        if not published_entries:
            return view_models.BotSharingStatusResponse(is_published=False)

        # Return the earliest published date
        group_ids = [entry.group_id for entry in published_entries]
        return view_models.BotSharingStatusResponse(is_published=True, group_ids=group_ids)

    async def get_raw_sharing_status(self, bot_id: str) -> Optional[db_models.BotMarketplace]:
        """Get raw sharing status (for permission checks) - returns first published entry"""
        marketplace_entries = await self.db_ops.get_bot_marketplace_entries_by_bot_id(bot_id)
        published_entries = [
            entry for entry in marketplace_entries 
            if entry.status == db_models.BotMarketplaceStatusEnum.PUBLISHED.value
        ]
        return published_entries[0] if published_entries else None

    async def list_published_bots(
        self, user_id: str, page: int = 1, page_size: int = 12, bot_type: str = None
    ) -> view_models.SharedBotList:
        """List all published bots accessible to user based on department"""
        # Get user's department ID
        user = await self.db_ops.query_user_by_id(user_id)
        user_department_id = user.department_id if user else None
        
        bots_data, total = await self.db_ops.get_accessible_bots_for_user(
            user_department_id=user_department_id, page=page, page_size=page_size, bot_type=bot_type
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
                subscription_id=None,  # No subscription needed anymore
                gmt_subscribed=None,
                config=shared_config,
                is_subscribed=True,  # All accessible bots are "subscribed" (accessible)
                is_owner=data["owner_user_id"] == user_id,
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
