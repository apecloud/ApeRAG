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

from aperag.config import settings
from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import (
    CollectionInactiveException,
    QuotaExceededException,
    ResourceNotFoundException,
    invalid_param,
)
from aperag.schema import view_models
from aperag.schema.view_models import Bot, BotList
from aperag.utils.constant import QuotaType
from aperag.views.utils import validate_bot_config


class BotService:
    """Bot service that handles business logic for bots"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def build_bot_response(self, bot: db_models.Bot, collection_ids: List[str]) -> view_models.Bot:
        """Build Bot response object for API return."""
        return Bot(
            id=bot.id,
            title=bot.title,
            description=bot.description,
            type=bot.type,
            config=bot.config,
            collection_ids=collection_ids,
            created=bot.gmt_created.isoformat(),
            updated=bot.gmt_updated.isoformat(),
        )

    async def create_bot(self, user: str, bot_in: view_models.BotCreate) -> view_models.Bot:
        # Check quota limit
        if settings.max_bot_count:
            bot_limit = await self.db_ops.query_user_quota(user, QuotaType.MAX_BOT_COUNT)
            if bot_limit is None:
                bot_limit = settings.max_bot_count
            if await self.db_ops.query_bots_count(user) >= bot_limit:
                raise QuotaExceededException("bot", bot_limit)

        # Validate collections before starting transaction
        collection_ids = []
        if bot_in.collection_ids is not None:
            for cid in bot_in.collection_ids:
                collection = await self.db_ops.query_collection(user, cid)
                if not collection:
                    raise ResourceNotFoundException("Collection", cid)
                if collection.status == db_models.CollectionStatus.INACTIVE:
                    raise CollectionInactiveException(cid)
                collection_ids.append(cid)

        # Create bot and collection relations atomically in a single transaction
        async def _create_bot_atomically(session):
            from aperag.db.models import Bot, BotCollectionRelation, BotStatus

            # Create bot in database directly using session
            bot = Bot(
                user=user,
                title=bot_in.title,
                type=bot_in.type,
                status=BotStatus.ACTIVE,
                description=bot_in.description,
                config="{}",
            )
            session.add(bot)
            await session.flush()
            await session.refresh(bot)

            # Create collection relations
            for cid in collection_ids:
                relation = BotCollectionRelation(bot_id=bot.id, collection_id=cid)
                session.add(relation)
                await session.flush()

            return bot

        bot = await self.db_ops.execute_with_transaction(_create_bot_atomically)

        return self.build_bot_response(bot, collection_ids=collection_ids)

    async def list_bots(self, user: str) -> view_models.BotList:
        bots = await self.db_ops.query_bots([user])
        response = []

        # Use _execute_query pattern to get collection IDs for all bots safely
        async def _get_bot_collections_data(session):
            bot_responses = []
            for bot in bots:
                # Handle legacy model names
                bot_config = json.loads(bot.config)
                model = bot_config.get("model", None)
                if model in ["chatgpt-3.5", "gpt-3.5-turbo-instruct"]:
                    bot_config["model"] = "gpt-3.5-turbo"
                elif model == "chatgpt-4":
                    bot_config["model"] = "gpt-4"
                elif model in ["gpt-4-vision-preview", "gpt-4-32k", "gpt-4-32k-0613"]:
                    bot_config["model"] = "gpt-4-1106-preview"
                bot.config = json.dumps(bot_config)

                # Get collection IDs for this bot using the session
                collection_ids = await bot.collections(session, only_ids=True)
                bot_responses.append(self.build_bot_response(bot, collection_ids=collection_ids))
            return bot_responses

        response = await self.db_ops._execute_query(_get_bot_collections_data)
        return BotList(items=response)

    async def get_bot(self, user: str, bot_id: str) -> view_models.Bot:
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        # Use _execute_query pattern to get collection IDs safely
        async def _get_bot_collections(session):
            collection_ids = await bot.collections(session, only_ids=True)
            return collection_ids

        collection_ids = await self.db_ops._execute_query(_get_bot_collections)
        return self.build_bot_response(bot, collection_ids=collection_ids)

    async def update_bot(self, user: str, bot_id: str, bot_in: view_models.BotUpdate) -> view_models.Bot:
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            raise ResourceNotFoundException("Bot", bot_id)

        # Validate configuration
        new_config = json.loads(bot_in.config)
        model_service_provider = new_config.get("model_service_provider")
        model_name = new_config.get("model_name")
        memory = new_config.get("memory", False)
        llm_config = new_config.get("llm")

        # Get API key for the model service provider
        api_key = await async_db_ops.query_provider_api_key(model_service_provider, user)
        if not api_key:
            raise invalid_param(
                "model_service_provider", f"API KEY not found for LLM Provider: {model_service_provider}"
            )

        # Get base_url from LLMProvider
        try:
            llm_provider = await async_db_ops.query_llm_provider_by_name(model_service_provider)
            base_url = llm_provider.base_url
        except Exception:
            raise ResourceNotFoundException("LLMProvider", model_service_provider)

        valid, msg = validate_bot_config(
            model_service_provider, model_name, base_url, api_key, llm_config, bot_in.type, memory
        )
        if not valid:
            raise invalid_param("config", msg)

        # Validate collections before starting transaction
        validated_collection_ids = []
        if bot_in.collection_ids is not None:
            for cid in bot_in.collection_ids:
                collection = await self.db_ops.query_collection(user, cid)
                if not collection:
                    raise ResourceNotFoundException("Collection", cid)
                if collection.status == db_models.CollectionStatus.INACTIVE:
                    raise CollectionInactiveException(cid)
                validated_collection_ids.append(cid)

        # Update bot and collection relations atomically in a single transaction
        async def _update_bot_atomically(session):
            from sqlalchemy import select

            from aperag.db.models import Bot, BotCollectionRelation, BotStatus, utc_now

            # Update bot
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            bot_to_update = result.scalars().first()

            if not bot_to_update:
                raise ResourceNotFoundException("Bot", bot_id)

            old_config = json.loads(bot_to_update.config)
            old_config.update(new_config)
            config_str = json.dumps(old_config)

            bot_to_update.title = bot_in.title
            bot_to_update.description = bot_in.description
            bot_to_update.type = bot_in.type
            bot_to_update.config = config_str
            session.add(bot_to_update)
            await session.flush()
            await session.refresh(bot_to_update)

            # Handle collection relations update
            collection_ids = []
            if bot_in.collection_ids is not None:
                # Delete old relations
                stmt = select(BotCollectionRelation).where(
                    BotCollectionRelation.bot_id == bot_id, BotCollectionRelation.gmt_deleted.is_(None)
                )
                result = await session.execute(stmt)
                relations = result.scalars().all()
                for rel in relations:
                    rel.gmt_deleted = utc_now()
                    session.add(rel)

                # Add new relations
                for cid in validated_collection_ids:
                    relation = BotCollectionRelation(bot_id=bot_id, collection_id=cid)
                    session.add(relation)
                    collection_ids.append(cid)
                    await session.flush()
            else:
                # Get existing collection IDs for response
                collection_ids = await bot_to_update.collections(session, only_ids=True)

            return bot_to_update, collection_ids

        updated_bot, collection_ids = await self.db_ops.execute_with_transaction(_update_bot_atomically)

        return self.build_bot_response(updated_bot, collection_ids=collection_ids)

    async def delete_bot(self, user: str, bot_id: str) -> Optional[view_models.Bot]:
        """Delete bot by ID (idempotent operation)

        Returns the deleted bot or None if already deleted/not found
        """
        # Check if bot exists - if not, silently succeed (idempotent)
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            return None

        # Delete bot and collection relations atomically in a single transaction
        async def _delete_bot_atomically(session):
            from sqlalchemy import select

            from aperag.db.models import Bot, BotCollectionRelation, BotStatus, utc_now

            # Get and delete bot
            stmt = select(Bot).where(Bot.id == bot_id, Bot.user == user, Bot.status != BotStatus.DELETED)
            result = await session.execute(stmt)
            bot_to_delete = result.scalars().first()

            if not bot_to_delete:
                return None, []

            # Get collection IDs before deleting relations
            collection_ids = await bot_to_delete.collections(session, only_ids=True)

            # Soft delete bot
            bot_to_delete.status = BotStatus.DELETED
            bot_to_delete.gmt_deleted = utc_now()
            session.add(bot_to_delete)
            await session.flush()
            await session.refresh(bot_to_delete)

            # Delete all relations
            stmt = select(BotCollectionRelation).where(
                BotCollectionRelation.bot_id == bot_id, BotCollectionRelation.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            relations = result.scalars().all()
            for rel in relations:
                rel.gmt_deleted = utc_now()
                session.add(rel)
            await session.flush()

            return bot_to_delete, collection_ids

        deleted_bot, collection_ids = await self.db_ops.execute_with_transaction(_delete_bot_atomically)

        if deleted_bot:
            return self.build_bot_response(deleted_bot, collection_ids=collection_ids)

        return None


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
bot_service = BotService()
