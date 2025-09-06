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

import json
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import (
    CollectionInactiveException,
    ResourceNotFoundException,
    invalid_param,
)
from aperag.schema import view_models
from aperag.schema.view_models import Bot, BotList
from aperag.service.quota_service import quota_service


class BotService:
    """Bot service that handles business logic for bots"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def build_bot_response(self, bot: db_models.Bot) -> view_models.Bot:
        """Build Bot response object for API return."""
        # Parse config from JSON string to BotConfig object
        bot_config = None
        if bot.config:
            config_dict = json.loads(bot.config)
            bot_config = view_models.BotConfig(**config_dict)
        
        return Bot(
            id=bot.id,
            title=bot.title,
            description=bot.description,
            type=bot.type,
            config=bot_config,
            created=bot.gmt_created.isoformat(),
            updated=bot.gmt_updated.isoformat(),
        )
    
    async def validate_collections(self, user: str, bot_config: view_models.BotConfig):
        if bot_config and bot_config.agent and bot_config.agent.collections:
            collection_ids = [collection.id for collection in bot_config.agent.collections]
            collections = await self.db_ops.query_collections_by_ids(user, collection_ids)
            if not collections or len(collections) != len(collection_ids):
                raise ResourceNotFoundException("Collection", collection_ids)

    async def create_bot(
        self, user: str, bot_in: view_models.BotCreate, skip_quota_check: bool = False
    ) -> view_models.Bot:
        # Create bot atomically in a single transaction
        async def _create_bot_atomically(session):
            from aperag.db.models import Bot, BotStatus

            # Check and consume quota within the transaction (unless skipped for system bots)
            if not skip_quota_check:
                await quota_service.check_and_consume_quota(user, "max_bot_count", 1, session)
            
            await self.validate_collections(user, bot_in.config)

            # Create bot in database directly using session
            # Serialize bot config to JSON string
            config_str = "{}"
            if bot_in.config:
                config_str = json.dumps(bot_in.config.model_dump(exclude_none=True))

            bot = Bot(
                user=user,
                title=bot_in.title,
                type=bot_in.type,
                status=BotStatus.ACTIVE,
                description=bot_in.description,
                config=config_str,
            )
            session.add(bot)
            await session.flush()
            await session.refresh(bot)

            return bot

        bot = await self.db_ops.execute_with_transaction(_create_bot_atomically)

        return await self.build_bot_response(bot)

    async def list_bots(self, user: str) -> view_models.BotList:
        bots = await self.db_ops.query_bots([user])
        return BotList(items=[await self.build_bot_response(bot) for bot in bots])

    async def get_bot(self, user: str, bot_id: str) -> view_models.Bot:
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        return await self.build_bot_response(bot)

    async def update_bot(self, user: str, bot_id: str, bot_in: view_models.BotUpdate) -> view_models.Bot:
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        # Serialize new config to JSON string
        new_config_str = None
        if bot_in.config:
            new_config_str = json.dumps(bot_in.config.model_dump(exclude_none=True))
            
        # Get collection IDs from bot config for validation
        await self.validate_collections(user, bot_in.config)

        # Update bot atomically in a single transaction
        async def _update_bot_atomically(session):
            from sqlalchemy import select
            from aperag.db.models import Bot, BotStatus

            # Update bot
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            bot_to_update = result.scalars().first()

            if not bot_to_update:
                raise ResourceNotFoundException("Bot", bot_id)

            # Use the new config directly
            if bot_in.title is not None:
                bot_to_update.title = bot_in.title
            if bot_in.description is not None:
                bot_to_update.description = bot_in.description
            if new_config_str is not None:
                bot_to_update.config = new_config_str
            session.add(bot_to_update)
            await session.flush()
            await session.refresh(bot_to_update)

            return bot_to_update

        updated_bot = await self.db_ops.execute_with_transaction(_update_bot_atomically)

        return await self.build_bot_response(updated_bot)

    async def delete_bot(self, user: str, bot_id: str) -> Optional[view_models.Bot]:
        """Delete bot by ID (idempotent operation)

        Returns the deleted bot or None if already deleted/not found
        """
        # Check if bot exists - if not, silently succeed (idempotent)
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            return None

        # Delete bot atomically in a single transaction
        async def _delete_bot_atomically(session):
            from sqlalchemy import select
            from aperag.db.models import Bot, BotStatus, utc_now

            # Get and delete bot
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            bot_to_delete = result.scalars().first()

            if not bot_to_delete:
                return None

            # Soft delete bot
            bot_to_delete.status = BotStatus.DELETED
            bot_to_delete.gmt_deleted = utc_now()
            session.add(bot_to_delete)
            await session.flush()
            await session.refresh(bot_to_delete)

            # Release quota within the transaction (only for non-system bots)
            if bot_to_delete.title != "Default Agent Bot":
                await quota_service.release_quota(user, "max_bot_count", 1, session)

            return bot_to_delete

        deleted_bot = await self.db_ops.execute_with_transaction(_delete_bot_atomically)

        if deleted_bot:
            return await self.build_bot_response(deleted_bot)

        return None


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
bot_service = BotService()
